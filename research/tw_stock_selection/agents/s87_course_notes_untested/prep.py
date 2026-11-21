#!/usr/bin/env python3
"""S87 第 0 步（寫 PREREG 之前）：驗收＋機械檢查，不看任何候選格的報酬。
A0 現行標準路徑（100 萬、位移 0）＝正式數字（讀 official_nav/out/res_official.json 當下的 cagr_since2019）；
A1 自建 0050 含息日報酬 vs 引擎 P.E50；A2 恆等：空決定表的載具＝P.E50（逐位）、全放 0050 的決定表跑引擎淨值與現行相差 < 1e-9；
M  各格機械統計（只有檔數、換手次數、持有哪幾檔的月數；不含報酬）。輸出 data_check.json。python3 -I -B prep.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s87lib as L
from s87lib import s10lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = L.base(P)
L.s17lib.e50_fix_suspension(P)
L.slim(P); gc.collect()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
# 正式結果檔在本題執行途中（2026-11-21 10:02）由工程切換成修正版（多開 REDUCE_FIX＋0050 含息改自建，48.86%）。
# 本題依指示只開三個正式開關、不開 REDUCE_FIX：驗收基準＝切換前的正式檔（本題開工時讀到的 48.76%，現封存於 archive_2026-11-21_pre_v2）；切換後的值一併記錄。
ARCH = sorted((H.parent / 'official_nav' / 'archive_2026-11-21_pre_v2').rglob('res_official.json'))[0]
off = json.loads(ARCH.read_text())
o = off['strats']['E0|W10_70']['std']
now = json.loads((H.parent / 'official_nav' / 'out' / 'res_official.json').read_text())
on = now['strats']['E0|W10_70']['std']
out = dict(flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK, REDUCE_FIX=bool(getattr(cslib, 'REDUCE_FIX', False))),
           official_file_pre_switch=dict(path=str(ARCH.relative_to(H.parent.parent.parent.parent)), sha256=L.sha(ARCH), flags=off['flags'],
                              cagr_since2019=o['cagr_since2019'], cagr_full=o['cagr_full'], maxdd=o['maxdd'], n_trades_closed=o['n_trades_closed']),
           official_file_now=dict(path='research/tw_stock_selection/agents/official_nav/out/res_official.json', sha256=L.sha(H.parent / 'official_nav' / 'out' / 'res_official.json'),
                              flags=now['flags'], cagr_since2019=on['cagr_since2019'], cagr_full=on['cagr_full'], maxdd=on['maxdd'], n_trades_closed=on['n_trades_closed']))
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
nav, lg, st = L.sim(P, S0, U)
a0 = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()))
a0['gap_since2019_pp'] = (a0['since2019'] - o['cagr_since2019']) * 100
a0['gap_vs_official_now_pp'] = (a0['since2019'] - on['cagr_since2019']) * 100
a0['ok'] = bool(abs(a0['gap_since2019_pp']) <= 0.02 and a0['n_closed'] == o['n_trades_closed'] and abs(a0['maxdd'] - o['maxdd']) < 1e-9)
out['A0'] = a0
print('A0', a0, flush=True)

E = L.etf_panel(P)
r_eng = P.E50[P.t0 + 1:] / P.E50[P.t0:-1] - 1
r_my = E['TR'][P.t0 + 1:, 0] / E['TR'][P.t0:-1, 0] - 1
out['A1'] = dict(max_abs_daily_diff=float(np.nanmax(np.abs(r_eng - r_my))), n_days_diff_gt_1bp=int((np.abs(r_eng - r_my) > 1e-4).sum()),
                 cagr_engine=float((P.E50[-1] / P.E50[P.t0]) ** (365.25 / (DE - D0).days) - 1), cagr_mine=float((E['TR'][-1, 0] / E['TR'][P.t0, 0]) ** (365.25 / (DE - D0).days) - 1))
print('A1', out['A1'], flush=True)
V0 = L.vehicle(P, E, {})
me = L.month_ends(P)
allz = {int(d) + 1: {'0050': 1.0} for d in me[12:] if d + 1 < P.T}
V1 = L.vehicle(P, E, allz)
n1 = L.sim(P, S0, U, V=V1)[0]
out['A2'] = dict(empty_bitwise=bool(np.array_equal(V0['Dc'], P.E50) and np.array_equal(V0['Do'], P.E50O)),
                 all0050_vehicle_max_rel=float(np.nanmax(np.abs(V1['Dc'][P.t0:] / P.E50[P.t0:] - 1))), all0050_cost=V1['cost_sum'],
                 all0050_nav_max_rel=float(np.nanmax(np.abs(n1.to_numpy() / nav.to_numpy() - 1))))
out['A2']['ok'] = bool(out['A2']['empty_bitwise'] and out['A2']['all0050_nav_max_rel'] < 1e-9 and out['A2']['all0050_cost'] == 0.0)
print('A2', out['A2'], flush=True)

# ---- 機械統計（不含報酬）
TB = int(np.flatnonzero(P.IDX == lastb('2022-01-01'))[0])
M = {}
for cell in L.CELLS:
    dec, rec = L.decisions(P, E, cell)
    tr = [r for r in rec if r['dec'] <= '2021-12-30']
    held = {}
    for r in tr:
        for c in r['hold']:
            held[c] = held.get(c, 0) + 1
    V = L.vehicle(P, E, {t: w for t, w in dec.items() if t <= TB})
    M[cell['id']] = dict(n_months_train=len(tr), n_months_with_tailwind=int(sum(1 for r in tr if r['hold'])), held_months_by_code=held,
                         n_switch_months_train=int(V['n_trades']), turnover_train=float(V['turnover']), cost_sum_train=float(V['cost_sum']))
out['mech_train'] = M
# 合格池（流動性＋上市滿回看月數）在訓練期各月的檔數
elig = {}
for pool in L.POOLS:
    cnt = []
    ci = {c: i for i, c in enumerate(E['codes'])}
    for q in range(12, len(me)):
        d = int(me[q])
        if str(P.IDX[d].date()) > '2021-12-30':
            break
        n = 0
        for c in L.POOLS[pool]:
            m = ci[c]
            a = E['AMT'][max(d - L.LIQ_WIN + 1, 0):d + 1, m]
            if E['HAS'][d, m] and np.isfinite(E['TR'][int(me[q - 12]), m]) and np.isfinite(a).sum() >= L.LIQ_MINOBS and np.nanmean(a) >= L.LIQ_MIN:
                n += 1
        cnt.append(n)
    elig[pool] = dict(median=float(np.median(cnt)), min=int(min(cnt)), max=int(max(cnt)), n_months=len(cnt))
out['eligible_count_train_L12'] = elig
out['etf_info'] = E['info']
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
for k, v in M.items():
    print(k, v)
print(elig, out['meta'])

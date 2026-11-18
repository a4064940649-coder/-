#!/usr/bin/env python3
"""S68 驗收（寫 PREREG 之前；只算現行只上市 CUR 的報酬，上櫃／合併股池只算「檔數」，不算報酬）。python3 -I -B prep.py → prep_check.json
- A0：在 S68 的資料設定（上櫃除權息 'main'＋營收補洞）下，現行 100 萬位移 0 逐位重現 2019 起 48.76%／全期 34.40%／回撤 −32.12%／已平倉 381 筆（與 deadline_engine_fix ALL 逐位相同）
      ＝證明上櫃資料層處理沒有動到上市；
- A2：現行 20 條路徑與 deadline_engine_fix ALL 逐條相同；
- A3：引擎副本 simulate_s68 與原引擎 simulate_s10 在現行 20 條路徑逐位相同；
- A4：三種上櫃除權息模式（none／main／zero）下，上市欄位（任一日曾為上市的代號……改為：當日 MK＝1 的格子）的 R、REF 逐位相同；
- 檔數：股池大小、SURGE 有值檔數、訊號月份數、名單檔數（OTC／ALL；不含任何報酬）。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s68lib as L
from s68lib import s10lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
out = dict(flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK), src_sha256=L.SRC_SHA)
# A4：上市格子不受上櫃除權息模式影響
ref = {}
for mode in ('none', 'zero', 'main'):
    P = L.load(mode)
    tw = P.MK == 1
    ref[mode] = (P.R[tw].copy(), P.REF[tw].copy(), dict(P.otc_info), P.exright_info)
    if mode != 'main':
        del P; gc.collect()
out['A4'] = dict(twse_R_same=bool(all(np.array_equal(ref['none'][0], ref[m][0]) for m in ('zero', 'main'))),
                 twse_REF_same=bool(all(np.array_equal(ref['none'][1], ref[m][1], equal_nan=True) for m in ('zero', 'main'))),
                 otc_info={m: ref[m][2] for m in ref}, exright_info_twse=ref['main'][3])
del ref; gc.collect()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
UU, RT = L.base(P)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
dl = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
same = lambda a, b: bool(np.array_equal(a[0].to_numpy(), b[0].to_numpy(), equal_nan=True) and a[1][cols].equals(b[1][cols]))
out['P0'] = dict(T=P.T, N=P.N, first=str(D0.date()), last=str(DE.date()), n_patch_rows=P.n_patch_rows, e50_fix=P.e50_fix)
paths = {}; a3 = True
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, UU['CUR'], RT, shift=s)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        r0 = L.sim(P, sg, UU['CUR'], cap=cap, orig=True); r1 = L.sim(P, sg, UU['CUR'], cap=cap)
        a3 &= same(r0, r1)
        nav, lg, st = r0
        paths[key] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE))
        if s == 0 and cap == 1e6:
            main = dict(paths[key], n_closed=int((lg['why'] != 'open_end').sum()))
out['A0'] = dict(main, tol_ok=bool(abs(main['since2019'] - L.ACC['since2019']) < 2.5e-4 and abs(main['full'] - L.ACC['full']) < 2.5e-4
                                   and abs(main['maxdd'] - L.ACC['maxdd']) < 2.5e-4 and main['n_closed'] == L.ACC['n_closed']),
                 bitwise_vs_deadline_ALL=bool(main['since2019'] == dl['standard']['ALL']['since2019'] and main['full'] == dl['standard']['ALL']['full']
                                              and main['maxdd'] == dl['standard']['ALL']['maxdd_full']))
d2 = [abs(paths[k]['since2019'] - v['since2019']) for k, v in dl['paths']['ALL'].items()]
out['A2'] = dict(n=len(d2), max_abs_diff=float(max(d2)), ok=bool(len(d2) == 20 and max(d2) < 1e-9))
out['A3_all_ok'] = bool(a3)
out['e50_since2019'] = stlib.cagr(pd.Series(P.E50, index=P.IDX), D19, DE)
# 檔數（不含報酬）
cnt = {}
for c in L.CELLS:
    sg = s10lib.make_signal_s10(P, UU[c], RT, shift=0)
    ks = [x['k'] for x in sg['info']]
    S = RT['S']
    nfin = [int((UU[c][x['k']] & np.isfinite(S.loc[pd.Period(x['month'], 'M')].to_numpy())).sum()) for x in sg['info']]
    nt = np.array([x['n_tgt'] for x in sg['info']])
    cnt[c] = dict(n_signal_months=len(ks), first_signal=sg['info'][0]['sig_date'], last_signal=sg['info'][-1]['sig_date'],
                  avg_universe=float(np.mean([UU[c][k].sum() for k in ks])), avg_surge_finite=float(np.mean(nfin)), min_surge_finite=int(min(nfin)),
                  avg_n_ma=float(np.mean([x['n_ma'] for x in sg['info']])), avg_n_tgt=float(nt.mean()), months_n_tgt_le2=int((nt <= 2).sum()), months_n_tgt_0=int((nt == 0).sum()))
out['counts_no_returns'] = cnt
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
(H / 'prep_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
assert out['A0']['tol_ok'] and out['A0']['bitwise_vs_deadline_ALL'] and out['A2']['ok'] and a3 and out['A4']['twse_R_same'] and out['A4']['twse_REF_same'], '驗收失敗'

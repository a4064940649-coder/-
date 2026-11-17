#!/usr/bin/env python3
"""S43 第 0 步（不算任何新規則的報酬）：資料盤點 → data_check.json。
(1) 現行 CUR 經本題包裝重現 48.99／34.54／−32.11／381；s39 CUR 包裝 SURGE 與 s17lib 逐位相同；
(2) 28 個月中，現行持股（CUR 主跑，位移 0、100 萬）在官方訊號列前「新營收已首次可見」的比例、早幾個交易日；
(3) 這些持股的首次值 SURGE 分布、首次值 vs 現行值差異；上月 L_MA 第 20 名門檻分布；
(4) 各規則×θ 會被標記的持股數、其中官方日本來就被 E0 賣掉的比例（只數筆數，不看報酬）。python3 -I -B prep.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s43lib
from s43lib import s10lib, s4lib, stlib, s17lib, COSTS

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
chk = {}
P = s43lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT, SFP, okcur = s43lib.surges(P)
chk['s39_cur_wrapper_bitwise'] = okcur
assert okcur
s17lib.e50_fix_suspension(P)
_, seen = s43lib.s36lib.load_seen(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
S2, ekw = s10lib.variant_S(S0, 'E0')
nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, sat_w=0.7, sizing='W10', lock=False, **ekw)
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
a0 = dict(since2019=stlib.cagr(nav, D19, P.IDX[-1]), full=stlib.cagr(nav, P.IDX[P.t0], P.IDX[-1]), maxdd=stlib.maxdd(nav, P.IDX[P.t0], P.IDX[-1]),
          n_trades=int((lg['why'] != 'open_end').sum()))
a0['ok'] = bool(abs(a0['since2019'] - 0.4899) < 0.00025 and abs(a0['full'] - 0.3454) < 0.00025 and abs(a0['maxdd'] + 0.3211) < 0.00025 and a0['n_trades'] == 381)
chk['A0_current_reproduced'] = a0
assert a0['ok'], a0
print('A0', a0, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

months = s43lib.TRAIN_MONTHS + s43lib.VALID_MONTHS
Scur = RT['S']
p20 = s43lib.prev20_thr(P, U, RT, months, 0)
rows = []
for m in months:
    k0 = RT['av'][m]
    if k0 >= P.T:
        continue
    held = lg[(lg['ent_t'] <= k0) & (lg['exit_t'] > k0)]      # 官方訊號列 k0 收盤時持有
    sc = Scur.loc[m].to_numpy(); sf = SFP.loc[m].to_numpy()
    o = s43lib.lma_order(P, U, sc, k0); top20 = set(int(j) for j in o[:20])
    for y in held.itertuples():
        j = int(y.j); c = y.code
        r = seen.get((m, c))
        early = r is not None and r < k0
        held_at_r = bool(early and y.ent_t <= r)
        rows.append(dict(month=str(m), period='train' if m in s43lib.TRAIN_MONTHS else 'valid', code=c, k=int(k0), r=(int(r) if r is not None else None),
                         seen=r is not None, early=bool(early), held_at_r=held_at_r, lead=(int(k0 - r) if early else None),
                         surge_fp=float(sf[j]), surge_cur=float(sc[j]), ma20_at_r=(bool(P.F_MA20[r, j]) if early else None),
                         prev20=float(p20[m]), e0_exit_at_k=bool(j not in top20 or not P.F_MA60[k0, j]), exit_t=int(y.exit_t), why=y.why))
H0 = pd.DataFrame(rows)
E = H0[H0['held_at_r']]
chk['holdings'] = dict(
    n_position_months=int(len(H0)), seen_share=float(H0['seen'].mean()), early_share=float(H0['early'].mean()), held_at_r_share=float(H0['held_at_r'].mean()),
    lead_days_dist={int(k): int(v) for k, v in E['lead'].value_counts().sort_index().items()},
    surge_fp_quantiles={str(q): float(E['surge_fp'].quantile(q)) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)},
    surge_fp_nan=int(E['surge_fp'].isna().sum()),
    fp_vs_cur_differs=int((~np.isclose(E['surge_fp'], E['surge_cur'], equal_nan=True)).sum()),
    prev20_quantiles={str(q): float(pd.Series(list(p20.values())).quantile(q)) for q in (0.1, 0.5, 0.9)},
    e0_exit_at_k_share=float(E['e0_exit_at_k'].mean()),
    note='持股月＝官方訊號列 k0 收盤時 CUR 持有的 (月份, 股票)；held_at_r＝新營收在 k0 前已首次可見、且可見時已持有（可提早賣的對象）。lead＝k0 − r（提早賣比現行早幾個交易日）。e0_exit_at_k＝官方日本來就會被 E0 賣掉（跌出 L_MA 前 20 或 ≤ 季線）。')
trig = {}
for rid, cond in [('XA_THR|0.0（格點外，參考）', lambda d: d['surge_fp'] < 0.0), ('XA_THR|0.5', lambda d: d['surge_fp'] < 0.5), ('XA_THR|1.0', lambda d: d['surge_fp'] < 1.0),
                  ('XA_THR|1.5', lambda d: d['surge_fp'] < 1.5), ('XA_THR|2.0', lambda d: d['surge_fp'] < 2.0), ('XA_NEG_MA20|0.0', lambda d: (d['surge_fp'] < 0.0) & (d['ma20_at_r'] == False)),
                  ('XA_PREV20|prev20', lambda d: d['surge_fp'] < d['prev20'])]:
    o = {}
    for per in ('train', 'valid'):
        d = E[E['period'] == per]
        x = d[cond(d).fillna(False).astype(bool)]
        o[per] = dict(n_flag=int(len(x)), n_months=int(x['month'].nunique()), e0_exit_at_k_anyway=int(x['e0_exit_at_k'].sum()),
                      share_e0_anyway=(float(x['e0_exit_at_k'].mean()) if len(x) else None))
    trig[rid] = o
chk['trigger_preview'] = trig
chk['trigger_note'] = '只用 CUR 持股估計（加入規則後持股會變，實際筆數以引擎為準）；不看任何報酬。share_e0_anyway 高＝規則多半只是把 E0 的賣出提早幾天。'
chk['data_sufficiency'] = dict(months=28, caveats=['只有上市（現行股池本來就只上市）', 'Tasi 時間是快照上界，週末前後公布的會晚 1～2 天才賣（保守）',
                                                    '28 個月切兩段各 14 個月，提早賣只影響少數筆交易，檢定力很低',
                                                    '首次值 SURGE 用 s39 FPM（當月 D 用首次值；μ、σ 用目前值）'])
chk['runtime_s'] = round(time.time() - t0, 1); chk['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
print(json.dumps(chk, ensure_ascii=False, indent=1, default=str))

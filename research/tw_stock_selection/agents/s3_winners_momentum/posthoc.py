#!/usr/bin/env python3
"""S3 事後分析（看過報酬後才加，不影響分類）。python3 -I -B posthoc.py → results_posthoc.json。要先跑 run.py。
A. 全期禁買 7610（E3 最大一筆 +1,291%）後各格重跑。
B. 起始資金微調（80／90／110／120 萬）看同一條規則的結果散布（路徑依賴）。
C. E2／E3 的交易依進場當下 L_MA 名次分組；D1 的交易依前 20 日漲幅 × 名次。
D. E1 規則檢查：每一筆規則出場，出場前一個訊號日的 L_MA 名次都不在前 20 名。
E. 各格持有 7610 的交易。
F. 截到 2025-11-28（7610 進場前）的年化與配對差。
G. 各格最大一筆占已平倉損益的比例；E2 買進名單第 10 名的 SURGE 分布（XS 出場門檻用）。"""
import sys, json, time, gc, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3lib
from s3lib import s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
P = s3lib.load()
U = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
DX = P.IDX[P.IDX < pd.Timestamp('2025-12-01')][-1]
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE)}
KW = {'D1': dict(), 'E1': dict(hold_n=20), 'E2': dict(mom=0.10), 'E3': dict(mom=0.10, hold_n=20), 'E4': dict(hold_n=30), 'E5a': dict(mom=0.05), 'E5b': dict(mom=0.15)}
CELLS = [(g, 1e6) for g in KW] + [('D1', 3e5), ('E1', 3e5), ('D1', 3e6), ('E1', 3e6)]
name = lambda g, c: g if c == 1e6 else f'{g}_{int(c / 1e4)}'
out = {}
j7610 = P.ci['7610']

# ---- A. 全期禁買 7610
A = {}; navA = {}
for g, cap in CELLS:
    S = s3lib.make_signal(P, U, RT, **KW[g])
    nav, lg, st = s2lib.simulate(P, S, U, idle='cash', cost=COSTS['strict'], capital=cap, block={j7610: [(0, P.T)]})
    w = {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}
    navA[name(g, cap)] = nav
    A[name(g, cap)] = dict(cagr={k: w[k]['cagr'] for k in w}, diff_0050={k: w[k]['diff'] for k in w}, maxdd=w['full']['maxdd'], n_closed=int((lg['why'] != 'open_end').sum()))
    del S; gc.collect()
for g, cap in CELLS:
    n = name(g, cap); b = name('D1', cap)
    if g != 'D1':
        pr = s2lib.paired(navA[n], navA[b], '2016-03')
        five = dict(paired_mean=pr['mean'], **{k: A[n]['cagr'][k] - A[b]['cagr'][k] for k in ('since2019', 'full', 'half1', 'half2')})
        A[n]['vs_D1_five'] = five; A[n]['paired'] = pr
        A[n]['vs_D1'] = '改善' if all(v > 0 for v in five.values()) else ('變差' if all(v < 0 for v in five.values()) else '沒差')
out['A_ban_7610'] = A

# ---- B. 起始資金微調
B = {}
for g in ('D1', 'E1', 'E2', 'E3'):
    S = s3lib.make_signal(P, U, RT, **KW[g])
    B[g] = {}
    for cap in (8e5, 9e5, 1e6, 1.1e6, 1.2e6):
        nav, lg, st = s2lib.simulate(P, S, U, idle='cash', cost=COSTS['strict'], capital=cap)
        x = lg[lg['code'] == '7610']
        B[g][str(int(cap / 1e4))] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), since2024=stlib.cagr(nav, D24, DE), maxdd=stlib.maxdd(nav, D0, DE),
                                         best_7610_ret=float(x['ret'].max()) if len(x) else None, pnl_7610_share=float(x['pnl'].sum() / lg['pnl'].sum()) if len(x) else 0.0)
    del S; gc.collect()
out['B_capital_jitter'] = B

# ---- 以下用 run.py 已存的交易與淨值
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
nv = pd.read_csv(H / 'navs.csv.gz', parse_dates=['date']).set_index('date')
d = lambda x: dict(n=int(len(x)), mean=float(x.mean()), median=float(x.median()), win=float((x > 0).mean())) if len(x) else dict(n=0)

C = {}
for g in ('D1', 'E1', 'E2', 'E3'):
    x = tr[(tr['run'] == f'{g}|cash|strict') & (tr['why'] != 'open_end')]
    C[g] = {'pos_le10': d(x.loc[x['pos_ma'] <= 10, 'ret']), 'pos_11_20': d(x.loc[(x['pos_ma'] > 10) & (x['pos_ma'] <= 20), 'ret']), 'pos_gt20': d(x.loc[x['pos_ma'] > 20, 'ret']),
            'pos_le10_excl7610': d(x.loc[(x['pos_ma'] <= 10) & (x['code'] != '7610'), 'ret']),
            'by_months': {str(m): d(x.loc[x['n_months'] == m, 'ret']) for m in (1, 2, 3)} | {'4+': d(x.loc[x['n_months'] >= 4, 'ret'])},
            'pre20_gt10_pos_le10': d(x.loc[(x['pre20'] > 0.10) & (x['pos_ma'] <= 10), 'ret']), 'pre20_le10_pos_le10': d(x.loc[(x['pre20'] <= 0.10) & (x['pos_ma'] <= 10), 'ret'])}
out['C_by_rank_at_entry'] = C

# ---- D. E1 規則檢查
S = s3lib.make_signal(P, U, RT, hold_n=20)
ix = {pd.Timestamp(v): i for i, v in enumerate(P.IDX)}
x = tr[(tr['run'] == 'E1|cash|strict') & (tr['why'] == 'rule')]
ks = np.array(sorted(S['pos_ma']))
bad = 0; late = 0
for r in x.itertuples():
    et = ix[r.exit_date]; j = P.ci[r.code]
    k = ks[ks < et][-1]                                      # 出場前最近一個訊號日
    if S['pos_ma'][k].get(j, 99999) <= 20:
        bad += 1
    if et != k + 1:
        late += 1
out['D_E1_rule_check'] = dict(rule_exits=int(len(x)), exits_while_in_top20=bad, exits_not_on_day_after_signal=late)
q = pd.DataFrame(s3lib.make_signal(P, U, RT, mom=0.10)['info'])
s10 = q['scores'].apply(lambda v: v[9] if len(v) >= 10 else np.nan)
out['G_E2_10th_surge'] = dict(q25=float(s10.quantile(0.25)), median=float(s10.median()), q75=float(s10.quantile(0.75)), n=int(s10.notna().sum()))
del S; gc.collect()

# ---- E. 各格持有 7610 的交易
E = {}
for run in sorted(set(tr['run'])):
    if run.endswith('cash|strict'):
        x = tr[(tr['run'] == run) & (tr['code'] == '7610')]
        E[run.split('|')[0]] = [dict(ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), ret=float(r.ret), pnl=float(r.pnl), cost=float(r.cost), why=r.why) for r in x.itertuples()]
out['E_7610_trades'] = E

# ---- F. 截到 2025-11-28
F = {'end': str(DX.date()), '0050': dict(since2019=stlib.cagr(bench, D19, DX), full=stlib.cagr(bench, D0, DX))}
for g, cap in CELLS:
    n = name(g, cap); b = name('D1', cap)
    s = nv[f'{n}|cash|strict']
    F[n] = dict(since2019=stlib.cagr(s, D19, DX), full=stlib.cagr(s, D0, DX))
    if g != 'D1':
        dd = (s2lib.monthly_ret(s) - s2lib.monthly_ret(nv[f'{b}|cash|strict'])).dropna()
        dd = dd[(dd.index >= pd.Period('2016-03', 'M')) & (dd.index <= pd.Period('2025-11', 'M'))]
        F[n]['paired_mean'] = float(dd.mean()); F[n]['paired_t'] = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))); F[n]['n_months'] = int(len(dd))
out['F_to_2025_11'] = F

# ---- G. 最大一筆占比
G = {}
for g, cap in CELLS:
    n = name(g, cap)
    x = tr[(tr['run'] == f'{n}|cash|strict') & (tr['why'] != 'open_end')].sort_values('pnl', ascending=False)
    G[n] = dict(top1=dict(code=x.iloc[0]['code'], ent=str(x.iloc[0]['ent_date'].date()), ret=float(x.iloc[0]['ret']), share=float(x.iloc[0]['pnl'] / x['pnl'].sum())),
                top5_share=float(x.head(5)['pnl'].sum() / x['pnl'].sum()), top10_share=float(x.head(10)['pnl'].sum() / x['pnl'].sum()),
                pnl_share_2026=float(x.loc[x['exit_date'].dt.year == 2026, 'pnl'].sum() / x['pnl'].sum()))
out['G_concentration'] = G
# ---- H. 禁買／位移後的 E1、E4、E3 有沒有抱到 7610（解釋為什麼 E1 禁買最賺 5 筆後反而變高）
Hh = {}
for g, cap, mode, arg in [('E1', 1e6, 'block', 5), ('E1', 1e6, 'block', 10), ('E1', 1e6, 'shift', 3), ('E1', 1e6, 'shift', 5), ('E1', 1e6, 'shift', 10), ('E4', 1e6, 'shift', 3), ('E4', 1e6, 'shift', 5),
                          ('E4', 1e6, 'shift', 10), ('E1', 3e5, 'block', 5), ('E1', 3e5, 'shift', 3), ('E3', 1e6, 'shift', 3), ('E3', 1e6, 'shift', 5), ('E3', 1e6, 'shift', 10), ('E3', 1e6, 'block', 5),
                          ('E1', 3e6, 'shift', 3), ('E1', 3e6, 'shift', 5), ('E1', 3e6, 'shift', 10)]:
    n = name(g, cap)
    if mode == 'block':
        S = s3lib.make_signal(P, U, RT, **KW[g])
        base = tr[tr['run'] == f'{n}|cash|strict'].copy()
        base['j'] = [P.ci[c] for c in base['code']]; base['ent_t'] = [ix[v] for v in base['ent_date']]; base['exit_t'] = [ix[v] for v in base['exit_date']]
        blk, _ = s2lib.block_from(base, arg)
        nav, lg, st = s2lib.simulate(P, S, U, idle='cash', cost=COSTS['strict'], capital=cap, block=blk)
    else:
        S = s3lib.make_signal(P, U, RT, shift=arg, **KW[g])
        nav, lg, st = s2lib.simulate(P, S, U, idle='cash', cost=COSTS['strict'], capital=cap)
    x = lg[lg['code'] == '7610']
    Hh[f'{n}|{mode}{arg}'] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), best_7610_ret=float(x['ret'].max()) if len(x) else None,
                                  pnl_7610_share=float(x['pnl'].sum() / lg['pnl'].sum()) if len(x) else 0.0)
    del S; gc.collect()
out['H_7610_in_block_shift_runs'] = Hh
for k, v in Hh.items():
    print('H', k, f"{v['since2019']*100:.2f} {v['full']*100:.2f}", None if v['best_7610_ret'] is None else round(v['best_7610_ret'] * 100), round(v['pnl_7610_share'] * 100))
out['meta'] = dict(maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
print(out['meta'])
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
pc = lambda v: f'{v * 100:.2f}'
for k, v in A.items():
    print('A', k, {a: pc(b) for a, b in v['cagr'].items()}, pc(v['maxdd']), v.get('vs_D1'), {a: round(b * 100, 3) for a, b in v.get('vs_D1_five', {}).items()}, f"t {v['paired']['t']:.2f}" if 'paired' in v else '')
for g, v in B.items():
    print('B', g, {c: (pc(x['since2019']), pc(x['full']), pc(x['since2024']), None if x['best_7610_ret'] is None else round(x['best_7610_ret'] * 100), round(x['pnl_7610_share'] * 100)) for c, x in v.items()})
for g, v in C.items():
    print('C', g, json.dumps(v))
print('D', out['D_E1_rule_check'], out['G_E2_10th_surge'])
for g, v in E.items():
    print('E', g, [(x['ent'], x['exit'], round(x['ret'] * 100), round(x['pnl'] / 1e4), round(x['cost'] / 1e4)) for x in v])
for g, v in F.items():
    print('F', g, v if isinstance(v, str) else {a: round(b * 100, 3) if a != 'paired_t' and a != 'n_months' else round(b, 2) for a, b in v.items()})
for g, v in G.items():
    print('G', g, v['top1']['code'], v['top1']['ent'], round(v['top1']['ret'] * 100), pc(v['top1']['share']), pc(v['top5_share']), pc(v['top10_share']), pc(v['pnl_share_2026']))

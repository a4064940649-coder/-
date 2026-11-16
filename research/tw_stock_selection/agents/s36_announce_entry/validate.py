#!/usr/bin/env python3
"""S36 驗證期（凍結後只跑一次）：凍結設定 FZ 與現行 CUR，只開驗證月份（2025-08～2026-09）早進場，驗證窗 2025-08-29 → 2026-10-22。
判定 V1～V4（20 條路徑）；描述：主跑交易、搶先報酬、成本加倍、全期開所有 28 個月的連續模擬。
輸出 results.json、trades.csv.gz。PREREG／FROZEN sha256 須相符；results.json 已存在就拒絕。python3 -I -B validate.py"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s36lib
from s36lib import s17lib, s10lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes(); sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG 不符'
fz = (H / 'FROZEN.json').read_bytes(); fsha = hashlib.sha256(fz).hexdigest()
assert fsha == (H / 'FROZEN.sha256').read_text().split()[0], 'FROZEN 不符'
assert not (H / 'results.json').exists(), 'results.json 已存在：驗證期只跑一次'
FZ = json.loads(fz)
RULE, THR = FZ['frozen_rule'], FZ['frozen_thr']
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s36lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s36lib.surge(P)
s17lib.e50_fix_suspension(P)
_, seen = s36lib.load_seen(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
VA, VB = lastb('2025-09-01'), P.IDX[-1]
TA = lastb('2024-07-01')
D19 = lastb('2019-01-01')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
MON = s36lib.VALID_MONTHS
ALLM = s36lib.TRAIN_MONTHS + s36lib.VALID_MONTHS
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])


def sim(S1, cap=1e6, cost='strict', block=None):
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, block=block, **CELL, **ekw)


def build(S, months, s):
    ev = s36lib.early_events(P, U, RT, seen, months, s)
    thr = s36lib.prev10_thr(P, U, RT, months, s) if THR == 'prev10' else float(THR)
    return s36lib.build_early(P, U, RT, S, RULE, thr, months, seen, s, ev)


S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
SF, addF = build(S0, MON, 0)
res = dict(PREREG_sha256=sha, FROZEN_sha256=fsha, frozen=FZ['frozen_id'], T0_pass=FZ['T0_pass'], window_valid=[str(VA.date()), str(VB.date())],
           bench_0050_valid=dict(cagr=stlib.cagr(bench, VA, VB), maxdd=stlib.maxdd(bench, VA, VB)))

# ---------------------------------------------------------------- 主跑（位移 0、100 萬）
main = {}
for c, S in (('CUR', S0), ('FZ', SF)):
    nav, lg, st = sim(S)
    nx = sim(S, cost='strict_x2')[0]
    main[c] = dict(nav=nav, lg=lg)
    res.setdefault('main', {})[c] = dict(valid=stlib.cagr(nav, VA, VB), valid_dd=stlib.maxdd(nav, VA, VB), valid_costx2=stlib.cagr(nx, VA, VB),
                                         since2019=stlib.cagr(nav, D19, VB), n_buy=int(st['n_buy']),
                                         n_trades_valid=int((lg['ent_date'] > VA).sum()), avg_npos=float(st['avg_npos']),
                                         yearly=stlib.yearly(nav.loc[VA:], bench))
assert np.array_equal(main['CUR']['nav'].loc[:VA].to_numpy(), main['FZ']['nav'].loc[:VA].to_numpy()), 'A3 失敗'
res['acceptance_A3'] = True
blkU = {}
top5 = {}
for c in ('CUR', 'FZ'):
    lg = main[c]['lg']
    b, top = s2lib.block_from(lg[lg['ent_date'] > VA], 5)
    top5[c] = [dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
    for j, v in b.items():
        blkU.setdefault(j, []).extend(v)
res['top5'] = top5
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in blkU.items()}

# 早進場逐筆描述：搶先報酬（r＋1 開盤 → 官方 k＋1 開盤），對 0050 超額，不扣成本
lgF = main['FZ']['lg']
f1set = {(x['month'], c) for x in S0['info'] for c in x['targets']}
ad = {(int(e), str(code)): m for m, code, e in addF}
rows = []
for y in lgF[lgF['ent_date'] > VA].itertuples():
    key = (int(y.ent_t) - 1, str(y.code))
    if key not in ad:
        continue
    m = pd.Period(ad[key], 'M'); k1 = RT['av'][m] + 1; j = P.ci[y.code]; e = int(y.ent_t)
    if k1 >= P.T:
        continue
    r_head = (P.CF[e, j] / P.O[e, j]) * (P.G[k1 - 1, j] / P.G[e, j]) * (P.O[k1, j] / P.REF[k1, j]) - 1
    r50 = P.E50O[k1] / P.E50O[e] - 1
    rows.append(dict(month=str(m), code=y.code, ent=str(y.ent_date.date()), lead_days=int(k1 - e), head_ret=float(r_head), head_excess=float(r_head - r50),
                     in_F1=bool((str(m), y.code) in f1set), why=y.why, trade_ret=float(y.ret), exit=str(y.exit_date.date())))
E = pd.DataFrame(rows)
desc = dict(n=int(len(E)))
if len(E):
    for nm, g in (('all', E), ('in_F1', E[E['in_F1']]), ('not_F1', E[~E['in_F1']])):
        desc[nm] = dict(n=int(len(g)), head_excess_mean=float(g['head_excess'].mean()) if len(g) else None, head_excess_median=float(g['head_excess'].median()) if len(g) else None,
                        lead_days_mean=float(g['lead_days'].mean()) if len(g) else None, trade_ret_mean=float(g['trade_ret'].mean()) if len(g) else None,
                        win=float((g['head_excess'] > 0).mean()) if len(g) else None)
res['early_trades_describe'] = desc
res['early_trades'] = rows
print('主跑完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 全期開 28 個月（描述）
SA, addA = build(S0, ALLM, 0)
navA, lgA, stA = sim(SA)
navC = main['CUR']['nav']
res['describe_all28'] = dict(n_signals=len(addA), cagr_FZ=stlib.cagr(navA, TA, VB), cagr_CUR=stlib.cagr(navC, TA, VB),
                             dd_FZ=stlib.maxdd(navA, TA, VB), dd_CUR=stlib.maxdd(navC, TA, VB),
                             since2019_FZ=stlib.cagr(navA, D19, VB), since2019_CUR=stlib.cagr(navC, D19, VB))
del navA, lgA, SA, main
gc.collect()

# ---------------------------------------------------------------- 20 條路徑
PATH = {k: {} for k in ('CUR', 'FZ', 'CUR_blk', 'FZ_blk', 'CUR_x2', 'FZ_x2')}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    Sf = SF if s == 0 else build(S, MON, s)[0]
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c, X in (('CUR', S), ('FZ', Sf)):
            nav = sim(X, cap)[0]
            PATH[c][key] = dict(valid=stlib.cagr(nav, VA, VB), valid_dd=stlib.maxdd(nav, VA, VB))
            PATH[c + '_blk'][key] = dict(valid=stlib.cagr(sim(X, cap, block=blkU)[0], VA, VB))
            PATH[c + '_x2'][key] = dict(valid=stlib.cagr(sim(X, cap, cost='strict_x2')[0], VA, VB))
    if s:
        del S, Sf
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR'])


def pair(a, b, f):
    d = np.array([PATH[a][k][f] - PATH[b][k][f] for k in keys]) * 100
    return dict(npos=int((d > 0).sum()), median_pp=float(np.median(d)), min_pp=float(d.min()), max_pp=float(d.max()))


pv = dict(valid=pair('FZ', 'CUR', 'valid'), valid_dd=pair('FZ', 'CUR', 'valid_dd'), valid_blk=pair('FZ_blk', 'CUR_blk', 'valid'), valid_costx2=pair('FZ_x2', 'CUR_x2', 'valid'))
res['pairs'] = pv
res['paths_median'] = {c: float(np.median([PATH[c][k]['valid'] for k in keys])) for c in PATH}
V = dict(V1=pv['valid']['npos'] >= 15, V2=pv['valid']['median_pp'] >= 2.0, V3=pv['valid_blk']['median_pp'] > 0, V4=pv['valid_dd']['median_pp'] >= -2.0)
V = {k: bool(v) for k, v in V.items()}
passed = bool(FZ['T0_pass'] and all(V.values()))
res['verdict'] = dict(T0=FZ['T0_pass'], **V, n_pass_V=int(sum(V.values())), overall='通過' if passed else '不通過')
res['runtime_s'] = round(time.time() - t_all, 1); res['maxrss_mb'] = rss()
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
E.to_csv(H / 'trades.csv.gz', index=False)
print(json.dumps({k: v for k, v in res.items() if k not in ('paths', 'early_trades', 'V3_block_union')}, ensure_ascii=False, indent=1, default=str)[:7000])

#!/usr/bin/env python3
"""S43 驗證期（凍結後只跑一次）：凍結設定 FZ 與現行 CUR，只開驗證月份（2025-08～2026-09）提早出場，驗證窗 2025-08-29 → 2026-10-22。
判定 V1～V4（20 條路徑）；描述：提早賣出筆數、被提早賣掉的股票「賣出日開盤 → CUR 原本出場日開盤」報酬（對 0050 超額）、成本加倍、全期開 28 個月連續模擬。
輸出 results.json、early_exits.csv.gz。PREREG／FROZEN sha256 須相符；results.json 已存在就拒絕。python3 -I -B validate.py"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s43lib
from s43lib import s17lib, s10lib, s4lib, s2lib, stlib, COSTS

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
P = s43lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT, SFP, okcur = s43lib.surges(P)
assert okcur
s17lib.e50_fix_suspension(P)
_, seen = s43lib.s36lib.load_seen(P)
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
MON = s43lib.VALID_MONTHS
ALLM = s43lib.TRAIN_MONTHS + s43lib.VALID_MONTHS
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])


def sim(S1, cap=1e6, cost='strict', block=None):
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, block=block, **CELL, **ekw)


def build(S, months, s):
    thr = s43lib.prev20_thr(P, U, RT, months, s) if THR == 'prev20' else float(THR)
    return s43lib.build_exit(P, U, RT, S, RULE, thr, months, seen, SFP, s)


def after_returns(lgX, lgC, marks, lo, hi):
    """lgX 中 why＝'ma_daily'（提早賣）且賣出日在 (lo, hi] 的每筆：找 CUR 同 (code, ent_t) 的交易 → 原本出場列 x。
    後來報酬＝賣出日 a 開盤 → x 開盤（CUR 若到期末仍持有：→ 最後一日收盤）的含息報酬；對 0050 同期超額（提早賣的錢放 0050）。正＝賣掉後來還漲（賣錯）。不扣成本。"""
    cur = {(str(c), int(e)): (int(x), w) for c, e, x, w in zip(lgC['code'], lgC['ent_t'], lgC['exit_t'], lgC['why'])}
    mk = {(str(c), int(e)): m for m, c, e in marks}
    rebuy = {(str(c), int(e)) for c, e in zip(lgX['code'], lgX['ent_t'])}
    rows = []
    for y in lgX[(lgX['why'] == 'ma_daily') & (lgX['exit_date'] > lo) & (lgX['exit_date'] <= hi)].itertuples():
        a = int(y.exit_t); j = int(y.j); c = str(y.code)
        m = mk.get((c, a - 1))
        if m is None:                                         # 跌停／無開盤順延：往前找標記
            cand = [(e, mm) for (cc, e), mm in mk.items() if cc == c and e < a]
            m = max(cand)[1] if cand else None
        hit = cur.get((c, int(y.ent_t)))
        row = dict(month=m, code=c, ent=str(y.ent_date.date()), sell=str(y.exit_date.date()), ret_to_sell=float(y.ret), found_in_CUR=hit is not None)
        if hit is not None:
            x, w = hit
            if w == 'open_end':
                r_stk = (P.CF[a, j] / P.O[a, j]) * (P.G[x, j] / P.G[a, j]) - 1; r50 = P.E50[x] / P.E50O[a] - 1
            else:
                r_stk = (P.CF[a, j] / P.O[a, j]) * (P.G[x - 1, j] / P.G[a, j]) * (P.O[x, j] / P.REF[x, j]) - 1; r50 = P.E50O[x] / P.E50O[a] - 1
            row.update(cur_exit=str(P.IDX[x].date()), cur_why=w, days_early=int(x - a), after_ret=float(r_stk), after_0050=float(r50), after_excess=float(r_stk - r50),
                       rebought_at_cur_exit=bool((c, x) in rebuy), cur_trade_ret=float(lgC.loc[(lgC['code'] == c) & (lgC['ent_t'] == y.ent_t), 'ret'].iloc[0]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(E):
    if not len(E) or 'after_excess' not in E:
        return dict(n=int(len(E)))
    g = E[E['found_in_CUR']]
    return dict(n=int(len(E)), n_matched=int(len(g)), days_early_mean=float(g['days_early'].mean()), days_early_median=float(g['days_early'].median()),
                after_excess_mean=float(g['after_excess'].mean()), after_excess_median=float(g['after_excess'].median()),
                after_excess_sum=float(g['after_excess'].sum()), share_sold_then_rose_vs0050=float((g['after_excess'] > 0).mean()),
                n_after_excess_gt_10pct=int((g['after_excess'] > 0.10).sum()), n_after_excess_lt_m10pct=int((g['after_excess'] < -0.10).sum()),
                worst_misses=g.sort_values('after_excess', ascending=False).head(5)[['code', 'sell', 'cur_exit', 'after_excess']].to_dict('records'),
                best_saves=g.sort_values('after_excess').head(5)[['code', 'sell', 'cur_exit', 'after_excess']].to_dict('records'),
                n_rebought=int(g['rebought_at_cur_exit'].sum()))


S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
SF, mkF = build(S0, MON, 0)
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
                                         n_early_sold_valid=int(((lg['why'] == 'ma_daily') & (lg['exit_date'] > VA)).sum()),
                                         yearly=stlib.yearly(nav.loc[VA:], bench))
assert np.array_equal(main['CUR']['nav'].loc[:VA].to_numpy(), main['FZ']['nav'].loc[:VA].to_numpy()), 'A3 失敗'
res['acceptance_A3'] = True
res['n_marks_valid_shift0'] = len(mkF)
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
EV = after_returns(main['FZ']['lg'], main['CUR']['lg'], mkF, VA, VB)
res['early_exits_valid'] = summarize(EV)
res['early_exits_valid_rows'] = EV.to_dict('records')
print('主跑完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 全期開 28 個月（描述）
SA, mkA = build(S0, ALLM, 0)
navA, lgA, stA = sim(SA)
navC = main['CUR']['nav']
EA = after_returns(lgA, main['CUR']['lg'], mkA, TA, VB)
EA['period'] = np.where(pd.to_datetime(EA['sell']) <= VA, 'train', 'valid')
res['describe_all28'] = dict(n_marks=len(mkA), n_early_sold=int(((lgA['why'] == 'ma_daily') & (lgA['exit_date'] > TA)).sum()),
                             cagr_FZ=stlib.cagr(navA, TA, VB), cagr_CUR=stlib.cagr(navC, TA, VB),
                             dd_FZ=stlib.maxdd(navA, TA, VB), dd_CUR=stlib.maxdd(navC, TA, VB),
                             since2019_FZ=stlib.cagr(navA, D19, VB), since2019_CUR=stlib.cagr(navC, D19, VB),
                             early_exits=summarize(EA), early_exits_train=summarize(EA[EA['period'] == 'train']), early_exits_valid=summarize(EA[EA['period'] == 'valid']))
EA.to_csv(H / 'early_exits.csv.gz', index=False)
del navA, lgA, SA, main
gc.collect()

# ---------------------------------------------------------------- 20 條路徑
PATH = {k: {} for k in ('CUR', 'FZ', 'CUR_blk', 'FZ_blk', 'CUR_x2', 'FZ_x2')}
NE = {}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    Sf = SF if s == 0 else build(S, MON, s)[0]
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c, X in (('CUR', S), ('FZ', Sf)):
            nav, lg, _ = sim(X, cap)
            PATH[c][key] = dict(valid=stlib.cagr(nav, VA, VB), valid_dd=stlib.maxdd(nav, VA, VB))
            if c == 'FZ':
                NE[key] = int(((lg['why'] == 'ma_daily') & (lg['exit_date'] > VA)).sum())
            PATH[c + '_blk'][key] = dict(valid=stlib.cagr(sim(X, cap, block=blkU)[0], VA, VB))
            PATH[c + '_x2'][key] = dict(valid=stlib.cagr(sim(X, cap, cost='strict_x2')[0], VA, VB))
    if s:
        del S, Sf
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
res['n_early_sold_valid_paths'] = NE
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
print(json.dumps({k: v for k, v in res.items() if k not in ('paths', 'early_exits_valid_rows', 'V3_block_union')}, ensure_ascii=False, indent=1, default=str)[:9000])

#!/usr/bin/env python3
"""S80 驗證（只跑一次；results.json 已存在就拒跑）：全期 2008-01-02～2026-10-22 連續帳戶，66 格＋現行 × 20 條路徑。
先核對 PREREG／FROZEN／程式 sha256，並核對每格訓練段數字與 results_train.json 逐位相同。
python3 -I -B validate.py → results.json、nav_std.csv.gz、yearly.csv、population.csv、placebo.csv"""
import sys, json, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
assert not (HERE / 'results.json').exists(), 'results.json 已存在：驗證只跑一次'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('PREREG', 'FROZEN'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f'{f} 與登記的 sha256 不符'
import numpy as np, pandas as pd
import s80lib as S
from s80lib import K
t0 = time.time()
FZ = json.loads((HERE / 'FROZEN.json').read_text()); TR = json.loads((HERE / 'results_train.json').read_text())
REF = FZ['reference_cell']; FROZEN = FZ['frozen']
P, U = K.load_cont()
F, RT = K.surge(P, 'B2')
IND, br = S.indicators(P, U); WE = S.week_ends(P)
tend = int(np.searchsorted(P.IDX.values, np.datetime64(S.TRAIN_END), side='right'))
A, M, E = P.IDX[P.t0], P.IDX[tend - 1], P.IDX[-1]
WIN = dict(train=(A, M), valid=(M, E), full=(A, E))
bench = pd.Series(P.E50, index=P.IDX).loc[A:]; bench = bench / bench.iloc[0] * 1e6
ids = [S.CUR] + [c['id'] for c in S.CELLS]
PATH = {i: {} for i in ids}; KEEP = {}; FRESH = {S.CUR: {}, REF: {}}; mism = 0
for s in S.SHIFTS:
    sg = K.signal(P, U, RT, s)
    for cid in ids:
        if cid == S.CUR:
            sg2, cm, szm, st = sg, None, None, None
        else:
            sg2, cm, szm, st, ch = S.apply_rule(P, U, sg, S.CELL[cid], IND, WE)
        for cap in S.CAPS:
            key = f'{int(cap / 1e4)}萬|+{s}'
            nav, lg, sx = S.sim(P, sg2, U, cap=cap, cashmode=cm, szm=szm)
            o = {}
            for w, (a, b) in WIN.items():
                x = S.window(nav, a, b); c, d = S.cagr(x), S.maxdd(x)
                o[w] = dict(cagr=c, maxdd=d, calmar=c / abs(d))
            o['n_closed'] = int((lg['why'] != 'open_end').sum())
            PATH[cid][key] = o
            tp = TR['paths'][cid][key]
            mism += int(not (tp['cagr'] == o['train']['cagr'] and tp['maxdd'] == o['train']['maxdd']))
            if s == 0 and cap == 1e6 and cid in (S.CUR, REF):
                KEEP[cid] = (nav, lg, sx, st)
            if cid in (S.CUR, REF):                           # 補充：2015-12-31 重新起跑
                n2, l2, _ = S.sim(P, sg2, U, cap=cap, cashmode=cm, szm=szm, tstart=tend - 1)
                c, d = S.cagr(n2), S.maxdd(n2)
                FRESH[cid][key] = dict(cagr=c, maxdd=d, calmar=c / abs(d))
    print('shift', s, f'{time.time() - t0:.0f}s', flush=True)
keys = list(PATH[S.CUR])
g = lambda cid, w, f: np.array([PATH[cid][k][w][f] for k in keys])
POP = []
for cid in ids:
    row = dict(cell=cid)
    for w in WIN:
        row[f'{w}_cagr'] = float(np.median(g(cid, w, 'cagr'))); row[f'{w}_maxdd'] = float(np.median(g(cid, w, 'maxdd'))); row[f'{w}_calmar'] = float(np.median(g(cid, w, 'calmar')))
        if cid != S.CUR:
            dc = (g(cid, w, 'cagr') - g(S.CUR, w, 'cagr')) * 100; dd = (g(cid, w, 'maxdd') - g(S.CUR, w, 'maxdd')) * 100; dk = g(cid, w, 'calmar') - g(S.CUR, w, 'calmar')
            row.update({f'{w}_dCAGR': float(np.median(dc)), f'{w}_dCAGR_min': float(dc.min()), f'{w}_dCAGR_max': float(dc.max()), f'{w}_nCAGR_pos': int((dc > 1e-9).sum()),
                        f'{w}_dMDD': float(np.median(dd)), f'{w}_nMDD_shallower': int((dd > 1e-9).sum()), f'{w}_dCalmar': float(np.median(dk)), f'{w}_nCalmar_pos': int((dk > 1e-12).sum())})
    POP.append(row)
pop = pd.DataFrame(POP).set_index('cell')
r = pop.loc[REF]
gates = dict(V1=bool(r['valid_dCAGR'] >= -2.0), V2=bool(r['valid_nMDD_shallower'] >= 15 and r['valid_dMDD'] >= 3.0), V3=bool(r['valid_nCalmar_pos'] >= 15),
             V4=bool(r['full_dCAGR'] >= -1.0 and r['full_dMDD'] >= 5.0))
allpass = bool(all(gates.values()) and FROZEN != S.CUR)
others = pop.drop(index=S.CUR)
dist = {}
for col in ('valid_dCAGR', 'valid_dMDD', 'valid_dCalmar', 'full_dCAGR', 'full_dMDD', 'full_dCalmar', 'train_dCAGR', 'train_dMDD', 'train_dCalmar'):
    v = others[col].to_numpy()
    dist[col] = dict(min=float(v.min()), p25=float(np.percentile(v, 25)), median=float(np.median(v)), p75=float(np.percentile(v, 75)), max=float(v.max()), n_pos=int((v > 0).sum()), n=len(v),
                     ref=float(r[col]), ref_rank_from_top=int((v > r[col]).sum()) + 1)
dist['n_cells_pass_all_gates'] = int(((others['valid_dCAGR'] >= -2) & (others['valid_nMDD_shallower'] >= 15) & (others['valid_dMDD'] >= 3) & (others['valid_nCalmar_pos'] >= 15)
                                      & (others['full_dCAGR'] >= -1) & (others['full_dMDD'] >= 5)).sum())
dist['cells_pass_all_gates'] = list(others.index[(others['valid_dCAGR'] >= -2) & (others['valid_nMDD_shallower'] >= 15) & (others['valid_dMDD'] >= 3) & (others['valid_nCalmar_pos'] >= 15)
                                                 & (others['full_dCAGR'] >= -1) & (others['full_dMDD'] >= 5)])
dist['corr_train_valid_dCalmar'] = float(np.corrcoef(others['train_dCalmar'], others['valid_dCalmar'])[0, 1])
dist['corr_train_valid_dCAGR'] = float(np.corrcoef(others['train_dCAGR'], others['valid_dCAGR'])[0, 1])
# ---- 標準路徑詳細（現行 vs 參考格）
DET = {}
for cid in (S.CUR, REF):
    nav, lg, sx, st = KEEP[cid]
    b = bench.reindex(nav.index); o = {}
    for w, (a, e) in WIN.items():
        x = S.win_stats(nav, b, a, e); x.update(S.flat_stats(P, lg, sx, a, e))
        if st is not None:
            ia, ie = P.IDX.get_loc(a), P.IDX.get_loc(e)
            x['bear_state_day_frac'] = float(st[ia:ie + 1].mean())
        if cid != S.CUR:
            x['vs_current'] = S.paired_t(nav, KEEP[S.CUR][0], a, e)
        x['paths_median'] = dict(cagr=float(pop.loc[cid, f'{w}_cagr']), maxdd=float(pop.loc[cid, f'{w}_maxdd']), calmar=float(pop.loc[cid, f'{w}_calmar']))
        o[w] = x
    o['yearly'] = K.yearly_table(nav, b)
    bears = {}
    for k, (a, e) in K.BEARS.items():
        x = nav.loc[a:e].dropna(); q = b.loc[a:e].dropna(); pre = nav.loc[:a].iloc[-1] if nav.index[0] < pd.Timestamp(a) else x.iloc[0]; preq = b.loc[:a].iloc[-1] if b.index[0] < pd.Timestamp(a) else q.iloc[0]
        bears[k] = dict(window=[a, e], seg_maxdd=K.dd_info(x)['maxdd'], seg_maxdd_0050=K.dd_info(q)['maxdd'], period_ret=float(x.iloc[-1] / pre - 1), period_ret_0050=float(q.iloc[-1] / preq - 1))
    o['bears'] = bears
    o['dd_full'] = K.dd_info(nav); o['final_nav'] = float(nav.iloc[-1])
    o['fresh_2016'] = dict(cagr_med=float(np.median([v['cagr'] for v in FRESH[cid].values()])), maxdd_med=float(np.median([v['maxdd'] for v in FRESH[cid].values()])))
    DET[cid] = o
fr = lambda f: np.array([FRESH[REF][k][f] for k in keys]) - np.array([FRESH[S.CUR][k][f] for k in keys])
fresh = dict(dCAGR_med=float(np.median(fr('cagr')) * 100), dMDD_med=float(np.median(fr('maxdd')) * 100), nCAGR_pos=int((fr('cagr') > 0).sum()), nMDD_shallower=int((fr('maxdd') > 0).sum()),
             nCalmar_pos=int((fr('calmar') > 0).sum()))
# ---- 安慰劑：參考格的弱勢旗標循環平移
PL = []
if S.CELL[REF]['ind'] is not None:
    rng = np.random.default_rng(80); sg = K.signal(P, U, RT, 0); raw = IND[S.CELL[REF]['ind']]
    cur = KEEP[S.CUR][0]
    for i, off in enumerate(rng.integers(250, P.T - 250, size=30)):
        sg2, cm, szm, st, ch = S.apply_rule(P, U, sg, S.CELL[REF], IND, WE, raw_override=np.roll(raw, int(off)))
        nav, lg, sx = S.sim(P, sg2, U, cap=1e6, cashmode=cm, szm=szm)
        row = dict(i=i, offset=int(off))
        for w, (a, e) in WIN.items():
            x, y = S.window(nav, a, e), S.window(cur, a, e)
            row[f'{w}_dCAGR'] = (S.cagr(x) - S.cagr(y)) * 100; row[f'{w}_dMDD'] = (S.maxdd(x) - S.maxdd(y)) * 100; row[f'{w}_dCalmar'] = S.cagr(x) / abs(S.maxdd(x)) - S.cagr(y) / abs(S.maxdd(y))
        PL.append(row)
    pl = pd.DataFrame(PL); pl.to_csv(HERE / 'placebo.csv', index=False)
    nav = KEEP[REF][0]; real = {}
    for w, (a, e) in WIN.items():
        x, y = S.window(nav, a, e), S.window(cur, a, e)
        real[f'{w}_dCAGR'] = (S.cagr(x) - S.cagr(y)) * 100; real[f'{w}_dMDD'] = (S.maxdd(x) - S.maxdd(y)) * 100; real[f'{w}_dCalmar'] = S.cagr(x) / abs(S.maxdd(x)) - S.cagr(y) / abs(S.maxdd(y))
    placebo = {k: dict(real=float(v), p10=float(pl[k].quantile(.1)), p50=float(pl[k].median()), p90=float(pl[k].quantile(.9)), pct_of_real=float((pl[k] < v).mean() * 100)) for k, v in real.items()}
else:
    placebo = None
out = dict(task='S80 市場狀態層', sha=dict(PREREG=sha(HERE / 'PREREG.json'), FROZEN=sha(HERE / 'FROZEN.json')), frozen=FROZEN, reference_cell=REF, windows={k: [str(a.date()), str(b.date())] for k, (a, b) in WIN.items()},
           train_recheck_mismatch=mism, gates=gates, verdict=('值得獨立複製' if allpass else '沒有比現行好，維持現行'), ref_population_row=r.to_dict(), detail=DET, fresh_start_2016=fresh,
           population_dist=dist, placebo=placebo, paths={S.CUR: PATH[S.CUR], REF: PATH[REF]}, fresh_paths=FRESH, runtime_s=round(time.time() - t0, 1))
(HERE / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
pop.to_csv(HERE / 'population.csv')
df = pd.DataFrame({'CURRENT': KEEP[S.CUR][0], REF: KEEP[REF][0], '0050': bench.reindex(KEEP[S.CUR][0].index)}).round(2); df.index.name = 'date'; df.to_csv(HERE / 'nav_std.csv.gz')
yc, yr = DET[S.CUR]['yearly'], DET[REF]['yearly']
pd.DataFrame([dict(year=a['year'], current=a['strat'], rule=b_['strat'], e0050=a['e0050'], rule_minus_current_pp=(b_['strat'] - a['strat']) * 100) for a, b_ in zip(yc, yr)]).to_csv(HERE / 'yearly.csv', index=False)
print(json.dumps(dict(gates=gates, verdict=out['verdict'], mism=mism, ref=REF), ensure_ascii=False)); print('done', out['runtime_s'])

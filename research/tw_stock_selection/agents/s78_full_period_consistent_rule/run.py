#!/usr/bin/env python3
"""S78 正式執行（只跑一次；results.json 已存在就拒絕）。先核對 PREREG 與程式的 sha256。
288 格 × 2008-01-02～2026-10-22 連續帳戶 → 區塊統計、4 折交叉驗證、全期一起挑、CSCV、虛無自助法、位移診斷、事前判定。
python3 -I -B run.py → results.json、grid.csv、grid_monthly.csv.gz、nav_key.csv.gz、cv_folds.csv"""
import sys, json, time, hashlib, itertools
from pathlib import Path
import os
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DRY = os.environ.get('S78_DRY') == '1'      # 乾跑：除現行格外，其餘 287 格的淨值一律換成「0050 × 隨機雜訊」的假資料，只用來測程式流程，輸出寫到 scratchpad，不看任何真實格子的成績
OUT = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s78/dry') if DRY else HERE
OUT.mkdir(parents=True, exist_ok=True)
assert DRY or not (HERE / 'results.json').exists(), 'results.json 已存在：本題只跑一次'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與登記的 sha256 不符'
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f'{f} 與登記的 sha256 不符'
import numpy as np, pandas as pd
import s78lib as Q
from s78lib import K, R

t0 = time.time()
P, U = K.load_cont(); Q.prep_panel(P)
D0, DE = P.IDX[P.t0], P.IDX[-1]
assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]; bench = bench / bench.iloc[0] * 1e6
F = Q.scores(P)
GRID = Q.grid(); IDS = [g['id'] for g in GRID]; NT = len(GRID)
assert NT == 288 and IDS.count(Q.CUR) == 1
ic = IDS.index(Q.CUR)

# ---------------------------------------------------------------- 288 格
NAV = np.empty((len(bench), NT)); meta = []
sgc = {}
for i, g in enumerate(GRID):
    key = (g['score'], g['trend'], g['entry'], g['exit'], g['regime'])
    if key not in sgc:
        sgc.clear(); sgc[key] = Q.make_signal(P, U, F, *key)
    sg = sgc[key]
    if DRY and g['id'] != Q.CUR:
        if i > 3:
            NAV[:, i] = bench.to_numpy() * np.exp(np.cumsum(np.random.default_rng(i).normal(0.0002, 0.004, len(bench))))
            meta.append(dict(n_closed=0, avg_npos=0.0, avg_idle_frac=0.0, mean_targets=0.0, n_signal_months=0)); continue
    nav, lg, st = Q.simulate(P, sg, U, g['weight'])
    if DRY and g['id'] != Q.CUR:
        nav = nav * 0 + bench * np.exp(np.cumsum(np.random.default_rng(i).normal(0.0002, 0.004, len(bench))))
    assert nav.index.equals(bench.index)
    NAV[:, i] = nav.to_numpy()
    x = lg[lg['why'] != 'open_end']
    meta.append(dict(n_closed=int(len(x)), avg_npos=float(st['avg_npos']), avg_idle_frac=float(st['avg_idle_frac']), mean_targets=float(np.mean([q['n_tgt'] for q in sg['info']])),
                     n_signal_months=len(sg['info'])))
    if i % 48 == 0:
        print(i, g['id'], f'{time.time() - t0:.0f}s', flush=True)
NAVdf = pd.DataFrame(NAV, index=bench.index, columns=IDS)
ref = pd.read_csv(K.HERE / 'nav_2008_2026.csv.gz', index_col=0, parse_dates=True)['V_BRIDGE']
a_cur = float((NAVdf[Q.CUR].round(2) - ref.reindex(NAVdf.index)).abs().max())
assert a_cur < 0.011, f'現行格與 S75 V_BRIDGE 不同：{a_cur}'

# ---------------------------------------------------------------- 月報酬、區塊
RS = {}
for c in IDS:
    rs, rb = Q.monthly(NAVdf[c], bench); RS[c] = rs
RS = pd.DataFrame(RS); EXdf = RS.sub(rb, axis=0); EX = EXdf.to_numpy()
MI = RS.index
bm = {b: np.asarray((MI >= pd.Period(a, 'M')) & (MI <= pd.Period(e, 'M'))) for b, (a, e) in Q.BLOCKS.items()}
hm = {b: np.asarray((MI >= pd.Period(a, 'M')) & (MI <= pd.Period(e, 'M'))) for b, (a, e) in Q.HALVES.items()}
BN = list(Q.BLOCKS); assert sum(m.sum() for m in bm.values()) == len(MI)
BEX = np.stack([EX[bm[b]].mean(axis=0) * 1200 for b in BN])           # 4×288 區塊超額年化
HEX = np.stack([EX[hm[b]].mean(axis=0) * 1200 for b in hm])
FULL = EX.mean(axis=0) * 1200


def describe(nav, label):
    """一條日淨值（index 同 bench）→ 全期、4 塊、前後段統計。"""
    rs, rb_ = Q.monthly(nav, bench)
    o = dict(label=label, final_nav=float(nav.iloc[-1]), cagr=K.cagr(nav), cagr_0050=K.cagr(bench), maxdd=R.maxdd(nav), maxdd_0050=R.maxdd(bench))
    o['full'] = Q.block_stats(rs, rb_, nav, str(MI[0]), str(MI[-1]))
    o['blocks'] = {b: Q.block_stats(rs, rb_, nav, a, e) for b, (a, e) in Q.BLOCKS.items()}
    o['halves'] = {b: Q.block_stats(rs, rb_, nav, a, e) for b, (a, e) in Q.HALVES.items()}
    o['min_block_ex'] = float(min(v['ex_ann_pp'] for v in o['blocks'].values()))
    o['gap_back_minus_front_pp'] = float(o['halves']['後段 2016-2026']['ex_ann_pp'] - o['halves']['前段 2008-2015']['ex_ann_pp'])
    for b, (a, e) in Q.BLOCKS.items():
        o['blocks'][b]['maxdd_0050'] = Q.dd_window(bench, a, e)
    return o, rs


# ---------------------------------------------------------------- 4 折交叉驗證
folds = []; pieces = []
drets = NAVdf.pct_change()
for bt in BN:
    tr = [bm[b] for b in BN if b != bt]
    j, mn, av = Q.pick(EX, tr)
    te = EX[bm[bt]].mean(axis=0) * 1200
    a, e = Q.BLOCKS[bt]
    dmask = (NAVdf.index >= pd.Period(a, 'M').to_timestamp()) & (NAVdf.index <= pd.Period(e, 'M').to_timestamp(how='end'))
    pieces.append(drets.loc[dmask, IDS[j]])
    st_ = Q.block_stats(RS[IDS[j]], rb, NAVdf[IDS[j]], a, e)
    folds.append(dict(test_block=bt, picked=IDS[j], picked_index=j, train_min_ex=float(mn[j]), train_mean_ex=float(av[j]),
                      train_block_ex={b: float(BEX[BN.index(b), j]) for b in BN if b != bt},
                      test_ex=float(te[j]), test_t_nw6=st_['t_nw6'], test_maxdd=st_['maxdd'], test_cagr=st_['cagr'], test_cagr_0050=st_['cagr_0050'],
                      test_rank_pct=float((te < te[j]).mean()), test_grid_median_ex=float(np.median(te)), test_grid_max_ex=float(te.max()),
                      test_current_ex=float(te[ic]), train_current_min_ex=float(mn[ic]), shrink_pp=float(mn[j] - te[j]),
                      train_top5=[dict(id=IDS[q], train_min=float(mn[q]), test_ex=float(te[q])) for q in np.lexsort((np.arange(NT), -av, -mn))[:5]]))
dr = pd.concat(pieces).sort_index().reindex(bench.index).fillna(0.0)
oos_nav = (1 + dr).cumprod() * 1e6
OOS, oos_rs = describe(oos_nav, '區塊外接起來（4 折）')
CURd, cur_rs = describe(NAVdf[Q.CUR], '現行規則')
jf, mnf, avf = Q.pick(EX, [bm[b] for b in BN])
FS, fs_rs = describe(NAVdf[IDS[jf]], '全期一起挑（看過全部資料）')
FS['id'] = IDS[jf]
top10 = [dict(id=IDS[q], min_block_ex=float(mnf[q]), block_ex=[float(x) for x in BEX[:, q]], full_ex=float(FULL[q]), cagr=K.cagr(NAVdf[IDS[q]]), maxdd=R.maxdd(NAVdf[IDS[q]]))
         for q in np.lexsort((np.arange(NT), -avf, -mnf))[:10]]


def paired(a, b):
    d = (a - b).to_numpy()
    return dict(diff_ann_pp=float(d.mean() * 1200), t_nw6=R.nw_t(d, 6), by_block={k: float(d[bm[k]].mean() * 1200) for k in BN},
                by_half={k: float(d[hm[k]].mean() * 1200) for k in hm})


pairs = dict(oos_minus_current=paired(oos_rs, cur_rs), fullpick_minus_current=paired(fs_rs, cur_rs))

# ---------------------------------------------------------------- 全體分布與各維度邊際
grid_summary = dict(
    median_block_ex={b: float(np.median(BEX[i])) for i, b in enumerate(BN)}, median_full_ex=float(np.median(FULL)),
    median_half_ex={b: float(np.median(HEX[i])) for i, b in enumerate(hm)},
    p10_p90_block_ex={b: [float(np.percentile(BEX[i], 10)), float(np.percentile(BEX[i], 90))] for i, b in enumerate(BN)},
    median_min_block_ex=float(np.median(BEX.min(axis=0))), max_min_block_ex=float(BEX.min(axis=0).max()),
    share_all4_positive=float((BEX > 0).all(axis=0).mean()), share_all4_ge3=float((BEX >= 3).all(axis=0).mean()), n_all4_ge3=int((BEX >= 3).all(axis=0).sum()),
    share_front_ge3=float((HEX[0] >= 3).mean()), share_back_ge3=float((HEX[1] >= 3).mean()),
    current_rank_full=int((FULL > FULL[ic]).sum() + 1), current_rank_min_block=int((BEX.min(axis=0) > BEX.min(axis=0)[ic]).sum() + 1),
    current_rank_by_block={b: int((BEX[i] > BEX[i, ic]).sum() + 1) for i, b in enumerate(BN)},
    median_cagr=float(np.median([K.cagr(NAVdf[c]) for c in IDS])), median_maxdd=float(np.median([R.maxdd(NAVdf[c]) for c in IDS])),
    block_ex_rank_corr={f'{BN[i]} vs {BN[j]}': float(pd.Series(BEX[i]).corr(pd.Series(BEX[j]), method='spearman')) for i, j in itertools.combinations(range(4), 2)})
marg = {}
for dim in ('score', 'trend', 'entry', 'exit', 'regime', 'weight'):
    marg[dim] = {}
    for lv in sorted(set(str(g[dim]) for g in GRID)):
        m = np.array([str(g[dim]) == lv for g in GRID])
        marg[dim][lv] = dict(n=int(m.sum()), block_ex_mean={b: float(BEX[i, m].mean()) for i, b in enumerate(BN)}, front_ex_mean=float(HEX[0, m].mean()), back_ex_mean=float(HEX[1, m].mean()),
                             full_ex_mean=float(FULL[m].mean()), min_block_mean=float(BEX[:, m].min(axis=0).mean()))

# ---------------------------------------------------------------- M1 CSCV（每塊切 4 段，任取 2 段挑、另 2 段驗；6^4＝1,296）
sub = {}
for b in BN:
    ix = np.flatnonzero(bm[b]); sub[b] = np.array_split(ix, 4)
c2 = list(itertools.combinations(range(4), 2))
SUBM = {b: np.stack([EX[s].mean(axis=0) * 1200 for s in sub[b]]) for b in BN}       # 每塊 4×288 子段年化超額
SUBN = {b: np.array([len(s) for s in sub[b]]) for b in BN}
rec = []
for combo in itertools.product(c2, repeat=4):
    IS, OS = [], []
    for b, sel in zip(BN, combo):
        oth = [q for q in range(4) if q not in sel]
        w1 = SUBN[b][list(sel)]; w2 = SUBN[b][oth]
        IS.append((SUBM[b][list(sel)] * w1[:, None]).sum(axis=0) / w1.sum()); OS.append((SUBM[b][oth] * w2[:, None]).sum(axis=0) / w2.sum())
    IS, OS = np.stack(IS), np.stack(OS)
    mn, av = IS.min(axis=0), IS.mean(axis=0)
    j = int(np.lexsort((np.arange(NT), -av, -mn))[0])
    omn, oav = OS.min(axis=0), OS.mean(axis=0)
    rec.append((j, mn[j], av[j], omn[j], oav[j], (omn < omn[j]).mean(), (oav < oav[j]).mean(), np.median(omn), np.median(oav), omn[ic], oav[ic]))
rec = np.array(rec)
cnt = pd.Series(rec[:, 0].astype(int)).value_counts()
cscv = dict(n_splits=len(rec), is_min_block_mean=float(rec[:, 1].mean()), oos_min_block_mean=float(rec[:, 3].mean()), shrink_min_block_pp=float((rec[:, 1] - rec[:, 3]).mean()),
            is_mean_ex_mean=float(rec[:, 2].mean()), oos_mean_ex_mean=float(rec[:, 4].mean()), shrink_mean_ex_pp=float((rec[:, 2] - rec[:, 4]).mean()),
            oos_min_block_p10_p50_p90=[float(np.percentile(rec[:, 3], q)) for q in (10, 50, 90)], oos_mean_ex_p10_p50_p90=[float(np.percentile(rec[:, 4], q)) for q in (10, 50, 90)],
            PBO_min_block=float((rec[:, 5] < 0.5).mean()), PBO_mean_ex=float((rec[:, 6] < 0.5).mean()),
            oos_grid_median_min_block=float(rec[:, 7].mean()), oos_grid_median_mean_ex=float(rec[:, 8].mean()),
            oos_current_min_block=float(rec[:, 9].mean()), oos_current_mean_ex=float(rec[:, 10].mean()),
            share_pick_beats_current_oos_min_block=float((rec[:, 3] > rec[:, 9]).mean()), share_pick_beats_current_oos_mean=float((rec[:, 4] > rec[:, 10]).mean()),
            share_oos_min_block_ge3=float((rec[:, 3] >= 3).mean()), share_oos_min_block_gt0=float((rec[:, 3] > 0).mean()),
            n_distinct_picks=int(len(cnt)), most_picked=[dict(id=IDS[i], share=float(n / len(rec))) for i, n in cnt.head(5).items()])

# ---------------------------------------------------------------- M2 虛無自助法（各塊內 stationary bootstrap，平均區段 6 個月）
rng = np.random.default_rng(78); NB = 2000; PB = 1 / 6
EXc = EX.copy()
for b in BN:
    EXc[bm[b]] -= EX[bm[b]].mean(axis=0)
bix = {b: np.flatnonzero(bm[b]) for b in BN}


def sb_index(n):
    out = np.empty(n, int); out[0] = rng.integers(n)
    jump = rng.random(n) < PB; rnd = rng.integers(n, size=n)
    for i in range(1, n):
        out[i] = rnd[i] if jump[i] else (out[i - 1] + 1) % n
    return out


nm = np.empty(NB); nf = np.empty(NB)
for q in range(NB):
    v = []; tot = np.zeros(NT)
    for b in BN:
        s = EXc[bix[b][sb_index(len(bix[b]))]]
        v.append(s.mean(axis=0) * 1200); tot += s.sum(axis=0)
    nm[q] = np.stack(v).min(axis=0).max(); nf[q] = (tot / len(MI) * 1200).max()
obs_m = float(BEX.min(axis=0).max()); obs_f = float(FULL.max())
null = dict(B=NB, seed=78, mean_block_months=6, observed_best_min_block=obs_m, null_best_min_block_mean=float(nm.mean()), null_best_min_block_p95=float(np.percentile(nm, 95)),
            p_best_min_block=float((nm >= obs_m).mean()), observed_best_full_ex=obs_f, null_best_full_mean=float(nf.mean()), null_best_full_p95=float(np.percentile(nf, 95)),
            p_best_full=float((nf >= obs_f).mean()), best_full_id=IDS[int(FULL.argmax())],
            current_min_block=float(BEX.min(axis=0)[ic]), p_current_min_block_vs_bestnull=float((nm >= BEX.min(axis=0)[ic]).mean()),
            current_full_ex=float(FULL[ic]), p_current_full_vs_bestnull=float((nf >= FULL[ic]).mean()))

# ---------------------------------------------------------------- 位移診斷（不參與挑選）
diag_ids = []
for c in [Q.CUR, IDS[jf]] + [f['picked'] for f in folds]:
    if c not in diag_ids:
        diag_ids.append(c)
diag = {}
for c in diag_ids:
    g = GRID[IDS.index(c)]; d = {}
    for s in (0, 1, 2, 3):
        if s == 0:
            nav = NAVdf[c]
        else:
            sg = Q.make_signal(P, U, F, g['score'], g['trend'], g['entry'], g['exit'], g['regime'], shift=s)
            nav, _, _ = Q.simulate(P, sg, U, g['weight'])
            if DRY:
                nav = NAVdf[c]
        rs, rb_ = Q.monthly(nav, bench); ex = (rs - rb_).to_numpy()
        d[f'+{s}'] = dict(cagr=K.cagr(nav), full_ex=float(ex.mean() * 1200), block_ex={b: float(ex[bm[b]].mean() * 1200) for b in BN}, min_block_ex=float(min(ex[bm[b]].mean() * 1200 for b in BN)),
                          front_ex=float(ex[hm['前段 2008-2015']].mean() * 1200), back_ex=float(ex[hm['後段 2016-2026']].mean() * 1200), maxdd=R.maxdd(nav))
    diag[c] = d
n_diag_runs = 3 * len(diag_ids)

# ---------------------------------------------------------------- 事前判定
H1, H2 = '前段 2008-2015', '後段 2016-2026'
q1 = dict(oos_block_ex={b: OOS['blocks'][b]['ex_ann_pp'] for b in BN}, all_ge3=bool(all(OOS['blocks'][b]['ex_ann_pp'] >= 3 for b in BN)), oos_full_t=OOS['full']['t_nw6'])
q1['pass'] = bool(q1['all_ge3'] and q1['oos_full_t'] >= 2.0)


def q2(new):
    o = dict(new_min_block=new['min_block_ex'], cur_min_block=CURd['min_block_ex'], i_pass=bool(new['min_block_ex'] >= CURd['min_block_ex'] + 2),
             new_gap=new['gap_back_minus_front_pp'], cur_gap=CURd['gap_back_minus_front_pp'],
             front_diff=new['halves'][H1]['ex_ann_pp'] - CURd['halves'][H1]['ex_ann_pp'], back_diff=new['halves'][H2]['ex_ann_pp'] - CURd['halves'][H2]['ex_ann_pp'])
    o['ii_pass'] = bool(o['new_gap'] <= o['cur_gap'] * (2 / 3)) if o['cur_gap'] > 0 else None
    o['front_improved'] = bool(o['front_diff'] >= 2); o['back_worse'] = bool(o['back_diff'] <= -3)
    return o


q2o, q2f = q2(OOS), q2(FS)
if q1['pass'] and q2o['i_pass'] and cscv['PBO_min_block'] < 0.5 and null['p_best_min_block'] < 0.10:
    rec_ = '值得進一步獨立複製的新規則'
elif grid_summary['share_all4_ge3'] < 0.10 and grid_summary['median_half_ex'][H1] < 3:
    rec_ = '這類訊號在早期本來就賺得少，不必硬改'
else:
    rec_ = '維持現行規則'
out = dict(task='S78 全期一致性規則（區塊交叉驗證）', sha=dict(PREREG=sha(HERE / 'PREREG.json'), s78lib=sha(HERE / 's78lib.py'), run=sha(HERE / 'run.py')),
           start=str(D0.date()), end=str(DE.date()), months=len(MI), n_trials=NT, n_diag_runs=n_diag_runs, accept_current_vs_S75_max_abs_twd=a_cur,
           bench=dict(final=float(bench.iloc[-1]), cagr=K.cagr(bench), maxdd=R.maxdd(bench), block_cagr={b: OOS['blocks'][b]['cagr_0050'] for b in BN}, block_maxdd={b: OOS['blocks'][b]['maxdd_0050'] for b in BN}),
           current=CURd, oos_stitched=OOS, full_sample_pick=FS, full_sample_top10=top10, folds=folds, pairs=pairs, grid_summary=grid_summary, marginals=marg,
           M1_cscv=cscv, M2_null_bootstrap=null, shift_diagnostics=diag, verdict=dict(Q1=q1, Q2_oos=q2o, Q2_fullpick=q2f, recommendation=rec_),
           runtime_s=round(time.time() - t0, 1))
(OUT / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
rows = []
for i, g in enumerate(GRID):
    rows.append(dict(**{k: g[k] for k in ('id', 'score', 'trend', 'entry', 'exit', 'regime', 'weight')}, cagr=K.cagr(NAVdf[g['id']]), maxdd=R.maxdd(NAVdf[g['id']]), full_ex=FULL[i],
                     **{f'ex_{b[:2]}': BEX[k, i] for k, b in enumerate(BN)}, min_block_ex=BEX[:, i].min(), front_ex=HEX[0, i], back_ex=HEX[1, i], **meta[i]))
pd.DataFrame(rows).to_csv(OUT / 'grid.csv', index=False)
EXdf.round(6).to_csv(OUT / 'grid_monthly_excess.csv.gz')
pd.DataFrame({'0050': bench, 'current': NAVdf[Q.CUR], 'oos_stitched': oos_nav, 'full_sample_pick': NAVdf[IDS[jf]]}).round(2).to_csv(OUT / 'nav_key.csv.gz')
pd.DataFrame([{k: v for k, v in f.items() if k not in ('train_top5', 'train_block_ex')} for f in folds]).to_csv(OUT / 'cv_folds.csv', index=False)
print('done', out['runtime_s'], 's')

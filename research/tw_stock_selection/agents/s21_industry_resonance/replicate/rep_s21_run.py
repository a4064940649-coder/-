#!/usr/bin/env python3
"""S21 獨立複製主程式：驗收 A0～A2 → 訓練期 7 設定 × 20 路徑（全期連續切訓練窗）→ 自行選規則凍結 →
驗證期（重新起跑）CUR／凍結規則／FILL_ALL × 20 路徑、V1～V4 → 全期連續描述 → 隨機換股對照（20 次）。
輸出 replicate/results_rep.json。python3 -I -B rep_s21_run.py"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rep_s21_lib as L
from rep_s21_lib import s20lib, s17lib, s10lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
G = s20lib.growth_features(P)
RT = dict(S=G['S'], av=G['av'])
del G
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
PI = L.PITInd(P)
print('載入', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
TA, TB, T18 = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01')
D19, D24 = T18, lastb('2024-01-01')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CANDS = ['FLT_P2', 'FLT_REL', 'FILL_P2', 'SWAP_P2', 'RANK_BR']
TRAIN_CFG = ['CUR'] + CANDS + ['FILL_ALL']
SEGV = ('2022', '2024-07~2025-04')
cg = stlib.cagr; mdd = stlib.maxdd


def run(S0, V, rule, cap, fresh=False, block=None, cost='strict'):
    sig, rank = V[rule]
    S2, ekw = L.variant_S(S0, sig, rank)
    if fresh:
        S2 = L.fresh(S2, None)
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, block=block, **CELL, **ekw)


def cont_metrics(nav, lg, cap):
    sg = s17lib.seg_dd(nav)
    return dict(train=cg(nav, TA, TB), train_dd=mdd(nav, TA, TB), y1618=cg(nav, TA, T18), y1921=cg(nav, T18, TB),
                since2019=cg(nav, D19, DE), full=cg(nav, D0, DE), valid=cg(nav, TB, DE), since2024=cg(nav, D24, DE), maxdd=mdd(nav, D0, DE),
                seg=sg, top5=s17lib.top5_share(lg, nav, cap))


def fresh_metrics(nav):
    sg = s17lib.seg_dd(nav)
    return dict(valid=cg(nav, TB, DE), valid_dd=mdd(nav, TB, DE), y2022_2023=cg(nav, TB, D24), since2024=cg(nav, D24, DE), seg={k: sg[k] for k in SEGV})


def keyof(cap, s):
    return f'{int(cap/1e4)}萬|+{s}'


res = {'note': '獨立複製：未 import／執行 s21lib.py、train.py、validate.py；引擎 s20lib→s17lib→s10lib', 'acceptance': {}}

# ---------------------------------------------------------------- 第一輪：全期連續（訓練分數＋連續描述），7 設定 × 20 路徑
CONT = {c: {} for c in TRAIN_CFG}
S0_0 = None; V_0 = None; rows0 = None
for s in range(4):
    S0 = s10lib.make_signal_s10(P, U, RT, shift=s)
    V, rows = L.build_variants(P, U, RT, PI, S0, s, TRAIN_CFG + (['PASSALL'] if s == 0 else []))
    if s == 0:
        navc, lgc, _ = run(S0, V, 'CUR', 1e6)
        a0 = dict(since2019=cg(navc, D19, DE), full=cg(navc, D0, DE), maxdd=mdd(navc, D0, DE))
        a0['ok'] = bool(abs(a0['since2019'] - 0.4813) < 0.00025 and abs(a0['full'] - 0.3408) < 0.00025 and abs(a0['maxdd'] + 0.3340) < 0.00025)
        nava, lga, _ = run(S0, V, 'PASSALL', 1e6)
        cols = ['code', 'ent_t', 'exit_t', 'why']
        a2 = dict(trades_identical=bool(len(lga) == len(lgc) and (lga[cols].reset_index(drop=True) == lgc[cols].reset_index(drop=True)).all().all()),
                  nav_identical=bool(np.allclose(nava.to_numpy(), navc.to_numpy(), rtol=0, atol=1e-6)))
        res['acceptance'] = dict(A0=a0, A1='每個訊號列自寫迴圈的 F1 與 make_signal_s10 info.targets 相同（4 個位移全部 assert 通過）', A2=a2)
        print('驗收', res['acceptance'], flush=True)
        del V['PASSALL'], navc, lgc, nava, lga
    for cap in CAPS:
        for c in TRAIN_CFG:
            nav, lg, _ = run(S0, V, c, cap)
            CONT[c][keyof(cap, s)] = cont_metrics(nav, lg, cap)
            del nav, lg
    if s == 0:
        S0_0, V_0, rows0 = S0, V, rows
    else:
        del S0, V, rows
    gc.collect()
    print('連續 位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
keys = list(CONT['CUR'])
med = lambda v: float(np.median(v))


def pair(D, a, b, f, sub=None):
    g = (lambda d: d[f]) if sub is None else (lambda d: d[f][sub])
    d = np.array([g(D[a][k]) - g(D[b][k]) for k in keys]) * 100
    return dict(median_pp=med(d), npos=int((d > 0).sum()), min_pp=float(d.min()), max_pp=float(d.max()))


# 改動檔數（位移 0，全期）：同分時用
nchg = {c: sum(len(set(x['f1']) ^ set(x['new'][c])) for x in rows0) for c in TRAIN_CFG}
nchg_tr = {c: sum(len(set(x['f1']) ^ set(x['new'][c])) for x in rows0 if pd.Period(x['month'], 'M') <= L.TRAIN_LAST) for c in TRAIN_CFG}
train = {}
for c in TRAIN_CFG:
    o = dict(train_median=med([CONT[c][k]['train'] for k in keys]), train_dd_median=med([CONT[c][k]['train_dd'] for k in keys]))
    if c != 'CUR':
        p = pair(CONT, c, 'CUR', 'train')
        o.update(score_median_pp=p['median_pp'], npos=p['npos'], diff_min_pp=p['min_pp'], diff_max_pp=p['max_pp'],
                 dd_diff_median_pp=pair(CONT, c, 'CUR', 'train_dd')['median_pp'],
                 y1618_median_pp=pair(CONT, c, 'CUR', 'y1618')['median_pp'], y1618_npos=pair(CONT, c, 'CUR', 'y1618')['npos'],
                 y1921_median_pp=pair(CONT, c, 'CUR', 'y1921')['median_pp'], y1921_npos=pair(CONT, c, 'CUR', 'y1921')['npos'],
                 n_changed_shift0_train=nchg_tr[c], n_changed_shift0_all=nchg[c])
    train[c] = o
best = sorted(CANDS, key=lambda c: (-train[c]['score_median_pp'], -train[c]['npos'], train[c]['n_changed_shift0_train']))[0]
T0 = bool(train[best]['score_median_pp'] >= 1.0 and train[best]['npos'] >= 14)
res['train'] = dict(window=[str(TA.date()), str(TB.date())], summary=train, order=sorted(CANDS, key=lambda c: -train[c]['score_median_pp']),
                    frozen_rep=best, T0_pass=T0)
FZ = best
print('訓練', json.dumps(train, ensure_ascii=False), '\n凍結', FZ, 'T0', T0, flush=True)

# ---------------------------------------------------------------- 主跑（位移 0）：重新起跑＋全期連續
S0, V = S0_0, V_0
nav1, lg1, _ = run(S0, V, 'CUR', 1e6, fresh=True)
ratio = float(nav1.loc[TB] / 1e6)
res['acceptance']['A3_fresh_no_trade_before_2022'] = bool((lg1['ent_date'] >= pd.Timestamp('2022-01-01')).all())
FCAP = {cap: cap / ratio for cap in CAPS}
MAIN = {}; mainlg = {}
for c in ('CUR', FZ, 'FILL_ALL'):
    for mode in ('fresh', 'cont'):
        fr = mode == 'fresh'
        cap = FCAP[1e6] if fr else 1e6
        nav, lg, st = run(S0, V, c, cap, fresh=fr)
        b0 = TB if fr else D0
        R = dict(valid=cg(nav, TB, DE), valid_dd=mdd(nav, TB, DE), since2024=cg(nav, D24, DE), maxdd=mdd(nav, b0, DE), seg=s17lib.seg_dd(nav),
                 top5_share=s17lib.top5_share(lg, nav.loc[b0:], float(nav.loc[b0])), n_trades=int(len(lg)))
        if not fr:
            R.update(since2019=cg(nav, D19, DE), full=cg(nav, D0, DE))
        MAIN[f'{c}|{mode}'] = R
        mainlg[f'{c}|{mode}'] = lg
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
res['main'] = MAIN
res['main_bench_0050_since2019'] = cg(bench, D19, DE)
blkU = {}
for nm in ('CUR|fresh', f'{FZ}|fresh'):
    b, _ = s2lib.block_from(mainlg[nm], 5)
    for j, v in b.items():
        blkU.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b].date())] for a, b in v] for j, v in blkU.items()}
# 換股名單差（全期連續主跑）
lc, lf = mainlg['CUR|cont'], mainlg[f'{FZ}|cont']
kc = set(zip(lc['code'], lc['ent_t'])); kf = set(zip(lf['code'], lf['ent_t']))
only_c = lc[[k in kc - kf for k in zip(lc['code'], lc['ent_t'])]]; only_f = lf[[k in kf - kc for k in zip(lf['code'], lf['ent_t'])]]
res['trade_diff_cont_main'] = dict(only_CUR_n=int(len(only_c)), only_CUR_pnl=float(only_c['pnl'].sum()), only_FZ_n=int(len(only_f)), only_FZ_pnl=float(only_f['pnl'].sum()),
                                   only_CUR_top=[dict(code=r.code, ent=str(r.ent_date.date()), ret=float(r.ret)) for r in only_c.sort_values('pnl', ascending=False).head(8).itertuples()])
del mainlg, nav1, lg1
print('主跑', json.dumps(MAIN, ensure_ascii=False, default=str)[:1500], flush=True)

# ---------------------------------------------------------------- 第二輪：重新起跑 × 20 路徑（CUR、FZ、FILL_ALL；禁買聯集 CUR、FZ）
FR = {f'{c}|fresh': {} for c in ('CUR', FZ, 'FILL_ALL')}
FR['CUR|fresh_blk'] = {}; FR[f'{FZ}|fresh_blk'] = {}
for s in range(4):
    if s == 0:
        S0, V = S0_0, V_0
    else:
        S0 = s10lib.make_signal_s10(P, U, RT, shift=s)
        V, _ = L.build_variants(P, U, RT, PI, S0, s, ['CUR', FZ, 'FILL_ALL'])
    for cap in CAPS:
        k = keyof(cap, s)
        for c in ('CUR', FZ, 'FILL_ALL'):
            nav = run(S0, V, c, FCAP[cap], fresh=True)[0]
            FR[f'{c}|fresh'][k] = fresh_metrics(nav)
            if c in ('CUR', FZ):
                nav = run(S0, V, c, FCAP[cap], fresh=True, block=blkU)[0]
                FR[f'{c}|fresh_blk'][k] = dict(valid=cg(nav, TB, DE))
            del nav
    if s:
        del S0, V
    gc.collect()
    print('重新起跑 位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

pv = dict(valid=pair(FR, f'{FZ}|fresh', 'CUR|fresh', 'valid'), valid_dd=pair(FR, f'{FZ}|fresh', 'CUR|fresh', 'valid_dd'),
          y2022_2023=pair(FR, f'{FZ}|fresh', 'CUR|fresh', 'y2022_2023'), since2024=pair(FR, f'{FZ}|fresh', 'CUR|fresh', 'since2024'),
          seg={sg: pair(FR, f'{FZ}|fresh', 'CUR|fresh', 'seg', sg) for sg in SEGV},
          blk=pair(FR, f'{FZ}|fresh_blk', 'CUR|fresh_blk', 'valid'))
V_ = dict(V1=pv['valid']['npos'] >= 15, V2=pv['valid']['median_pp'] >= 2.0, V3=pv['blk']['median_pp'] > 0, V4=all(pv['seg'][g]['median_pp'] >= -2.0 for g in SEGV))
V_ = {k: bool(v) for k, v in V_.items()}
res['valid'] = dict(pairs_FZ_vs_CUR=pv, FILL_ALL_vs_CUR=dict(valid=pair(FR, 'FILL_ALL|fresh', 'CUR|fresh', 'valid')),
                    medians={nm: {f: med([D[k][f] for k in keys]) for f in D[keys[0]] if f != 'seg'} for nm, D in FR.items()},
                    fresh_capital_ratio=ratio)
res['verdict'] = dict(T0=T0, **V_, overall='通過' if (T0 and all(V_.values())) else '不通過')
cp = {c: {f: pair(CONT, c, 'CUR', f) for f in ('since2019', 'full', 'valid', 'since2024', 'maxdd', 'top5')} for c in (FZ, 'FILL_ALL')}
res['cont_paths'] = dict(pairs=cp, medians={c: {f: med([CONT[c][k][f] for k in keys]) for f in ('since2019', 'full', 'valid', 'since2024', 'maxdd', 'top5')} for c in TRAIN_CFG})
print('判定', res['verdict'], json.dumps(pv, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- 隨機換股對照（主跑：位移 0、100 萬）
S0 = S0_0
spec = {x['k']: (len(set(x['f1']) - set(x['new'][FZ])), len(set(x['new'][FZ]) - set(x['f1']))) for x in rows0}
del V_0
gc.collect()
rng = np.random.default_rng(20261116)
RAND = []
fz_cont_valid = MAIN[f'{FZ}|cont']['valid']; fz_cont_19 = MAIN[f'{FZ}|cont']['since2019']; fz_fresh_valid = MAIN[f'{FZ}|fresh']['valid']
for i in range(20):
    Vr, _ = L.build_variants(P, U, RT, PI, S0, 0, ['RAND'], rng=rng, rand_spec=spec)
    nav = run(S0, Vr, 'RAND', 1e6)[0]
    navf = run(S0, Vr, 'RAND', FCAP[1e6], fresh=True)[0]
    RAND.append(dict(cont_valid=cg(nav, TB, DE), since2019=cg(nav, D19, DE), fresh_valid=cg(navf, TB, DE)))
    del Vr, nav, navf
    gc.collect()
rv = pd.DataFrame(RAND)
q = lambda x: dict(median=float(x.median()), p10=float(x.quantile(0.1)), p90=float(x.quantile(0.9)))
res['random_swap'] = dict(seed=20261116, n=20, cont_valid=q(rv['cont_valid']), since2019=q(rv['since2019']), fresh_valid=q(rv['fresh_valid']),
                          FZ_main=dict(cont_valid=fz_cont_valid, since2019=fz_cont_19, fresh_valid=fz_fresh_valid),
                          FZ_beats=dict(cont_valid=int((fz_cont_valid > rv['cont_valid']).sum()), since2019=int((fz_cont_19 > rv['since2019']).sum()),
                                        fresh_valid=int((fz_fresh_valid > rv['fresh_valid']).sum())), draws=RAND)
print('隨機', json.dumps({k: v for k, v in res['random_swap'].items() if k != 'draws'}, ensure_ascii=False), flush=True)
res['paths_cont'] = CONT; res['paths_fresh'] = FR
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(),
                   code_sha256={f: hashlib.sha256((H / f).read_bytes()).hexdigest() for f in ('rep_s21_lib.py', 'rep_s21_run.py', 'rep_s21_prep.py')})
(H / 'results_rep.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print('完成', res['meta'])

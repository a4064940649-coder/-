#!/usr/bin/env python3
"""S88 第 0 步（PREREG 之前）：驗收＋資料涵蓋＋旗標出現率（全期，不看報酬）＋訓練期（訊號 2016-02～2021-12）報酬分組。
驗證期（2022 起）的報酬在本步驟完全不計算、不輸出。輸出 data_check.json。python3 -I -B prep.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s88lib as L
from s88lib import s10lib, stlib

H = Path(__file__).resolve().parent
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = L.base(P)
L.s17lib.e50_fix_suspension(P)
L.slim(P); gc.collect()
FN = L.Fin(P)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
res = dict(flags=dict(EXRIGHT_FIX=L.cslib.EXRIGHT_FIX, DEADLINE_FIX=L.cslib.DEADLINE_FIX, INS15_MASK=L.cslib.INS15_MASK, REDUCE_FIX=L.cslib.REDUCE_FIX, E50_SOURCE=L.cslib.E50_SOURCE))

# ---- 驗收：現行標準路徑與 20 條路徑
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
nav, lg, st = L.sim(P, S0, U, cap=1e6)
lg['code'] = lg['code'].astype(str)
got = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()), n_all=int(len(lg)))
paths = {}
for s in L.SHIFTS:
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in L.CAPS:
        paths[f'{int(cap/1e4)}萬|+{s}'] = stlib.cagr(L.sim(P, S, U, cap=cap)[0], D19, DE)
got['paths_median'] = float(np.median(list(paths.values())))
off_now = json.loads((L.AG / 'official_nav' / 'out' / 'res_official.json').read_text())
off_old = json.loads((L.AG / 'official_nav' / 'archive_2026-11-21_pre_v2' / 'res_official.json').read_text())
res['acceptance'] = dict(got=got, target_3flags=L.ACC, target_paths_median_3flags=off_old['strats']['E0|W10_70']['paths_summary']['cagr19_median'],
                         res_official_now=dict(flags=off_now['flags'], cagr_since2019=off_now['strats']['E0|W10_70']['std']['cagr_since2019']),
                         res_official_archive_pre_v2=dict(flags=off_old['flags'], cagr_since2019=off_old['strats']['E0|W10_70']['std']['cagr_since2019']),
                         ok=bool(abs(got['since2019'] - L.ACC['since2019']) < 1e-9 and abs(got['full'] - L.ACC['full']) < 1e-9 and abs(got['maxdd'] - L.ACC['maxdd']) < 1e-9 and got['n_closed'] == L.ACC['n_closed']))
print('驗收', res['acceptance'], flush=True)
assert res['acceptance']['ok']

# ---- 資料涵蓋與旗標出現率（不看報酬）
T = L.trade_features(P, U, FN, lg)
per = lambda d: ('pre' if d < pd.Timestamp('2016-02-01') else 'train' if d < pd.Timestamp('2022-01-01') else 'valid')
T['per'] = [per(d) for d in T['sig_date']]
cov = {}
for nm, z in (('all', T), ('train', T[T['per'] == 'train']), ('valid', T[T['per'] == 'valid'])):
    cov[nm] = dict(n=int(len(z)), has_fin=float(np.isfinite(z['kq']).mean()), age_days_median=float(z['age_days'].median()),
                   avail={c: float(np.isfinite(z[c]).mean()) for c in ('gm_d', 'opm_d', 'rev_g', 'acc', 'ocf_a', 'ag_gap')},
                   flag_rate={c: float(z[c].mean()) for c in L.FLAGS},
                   rule_rate={r: float(np.asarray(fn(z)).mean()) for r, fn in L.RULES.items()})
res['bought_trades_flag_rate'] = cov
# 同期股池出現率（各訊號月平均）
ur = {c: [] for c in L.FLAGS}
for x in S0['info']:
    F = L.features(P, U, FN, x['k']); u = U[x['k']] & np.isfinite(F['kq'])
    for c in L.FLAGS:
        ur[c].append(float(F[c][u].mean()))
res['universe_flag_rate_mean'] = {c: float(np.mean(v)) for c, v in ur.items()}
# 規則之間在買進交易上的重疊（全期，不看報酬）
R = pd.DataFrame({r: np.asarray(fn(T), bool) for r, fn in L.RULES.items()})
res['rule_overlap_jaccard'] = {a: {b: (float((R[a] & R[b]).sum() / max(1, (R[a] | R[b]).sum()))) for b in R} for a in R}
# 每條規則對名單的機械影響（全期 shift 0；不看報酬）：擋掉幾檔、遞補幾檔
me = {}
for cell in L.CELLS:
    _, rec = L.apply_cell(P, U, RT, FN, S0, cell)
    me[cell] = dict(n_orig=sum(len(r['orig']) for r in rec), n_removed=sum(len(r['removed']) for r in rec), n_added=sum(len(r['added']) for r in rec),
                    months_changed=sum(1 for r in rec if r['removed']))
res['mechanics_shift0_full'] = me

# ---- 訓練期報酬分組（只有訊號日 2016-02-01～2021-12-31 的交易）
Z = T[T['per'] == 'train'].copy()
Z['ex'] = Z['ret'].to_numpy() - (P.E50[Z['exit_t'].to_numpy()] / P.E50O[Z['ent_t'].to_numpy()] - 1)
hi, lo = Z['ret'].quantile(0.95), Z['ret'].quantile(0.05)
tp = float(Z['pnl'].sum())


def gstat(z):
    if not len(z):
        return dict(n=0)
    return dict(n=int(len(z)), win=float((z['ret'] > 0).mean()), mean=float(z['ret'].mean()), median=float(z['ret'].median()), mean_ex=float(z['ex'].mean()),
                big_win=float((z['ret'] >= hi).mean()), big_lose=float((z['ret'] <= lo).mean()), n_big_win=int((z['ret'] >= hi).sum()), n_big_lose=int((z['ret'] <= lo).sum()),
                pnl_share=float(z['pnl'].sum() / tp))


G = dict(ALL=gstat(Z), thresholds=dict(big_win_ret_ge=float(hi), big_lose_ret_le=float(lo)))
for c in L.FLAGS:
    G[c + '=1'] = gstat(Z[Z[c]])
    G[c + '=0'] = gstat(Z[~Z[c]])
for r, fn in L.RULES.items():
    m = np.asarray(fn(Z), bool)
    G[r + '=1'] = gstat(Z[m])
G['無財報'] = gstat(Z[~np.isfinite(Z['kq'])])
res['train_returns_by_group'] = G
res['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'data_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(cov, ensure_ascii=False, indent=1))
print(json.dumps(res['universe_flag_rate_mean'], ensure_ascii=False))
print(json.dumps(me, ensure_ascii=False))
for k, v in G.items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
print(res['meta'])

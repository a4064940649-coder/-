#!/usr/bin/env python3
"""S59 第 3 步：依 SPEC／PREREG 做 A1～A6，輸出 results.json。用法：python3 -I -B run.py
先驗 SPEC／PREREG sha256；讀 env_monthly.csv（env.py）與暫存交易／訊號（engine.py）。"""
import json, math, hashlib, resource
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s59')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記不符'
eng = json.loads((HERE / 'engine_check.json').read_text())
assert eng['oos']['ok'] and eng['is']['ok']

E = pd.read_csv(HERE / 'env_monthly.csv', index_col=0)
E.index = pd.PeriodIndex(E.index, freq='M')
E = E.loc['2008-02':'2026-10']
assert len(E) == 225
seg = pd.Series('IS', index=E.index)
seg[(E.index >= pd.Period('2008-02', 'M')) & (E.index <= pd.Period('2013-01', 'M'))] = 'OOS'
seg[(E.index >= pd.Period('2013-02', 'M')) & (E.index <= pd.Period('2015-06', 'M'))] = 'STUCK'
E['seg'] = seg
E['y'] = E['excess'] * 1200                                     # 年化百分點
PRIM = E['seg'] != 'STUCK'
assert PRIM.sum() == 196
VARS = ['MKT_MOM12', 'MKT_VOL', 'SMB_MOM12', 'TURN_CHG', 'REV_DISP', 'ELEC_W', 'SURGE_TOP']
VARS6 = VARS[:6]
SEC = ['MKT_RET', 'SMB_RET']
mu = E.loc[PRIM, VARS + SEC].mean(); sd = E.loc[PRIM, VARS + SEC].std(ddof=1)
Z = (E[VARS + SEC] - mu) / sd


def nw_ols(y, X, L=6):
    y = np.asarray(y, float); X = np.asarray(X, float)
    T, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    t = b / se
    p = 2 * stats.t.sf(np.abs(t), T - k)
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return dict(b=b, se=se, t=t, p=p, V=V, T=T, r2=float(r2))


def nw_mean(x, L=6):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    r = nw_ols(x, np.ones((len(x), 1)), L)
    return dict(T=int(len(x)), mean=float(r['b'][0]), t_nw=float(r['t'][0]))


def uni(mask, v):
    m = mask & Z[v].notna()
    r = nw_ols(E.loc[m, 'y'], np.column_stack([np.ones(m.sum()), Z.loc[m, v]]))
    return dict(T=r['T'], slope_pp_per_sd=float(r['b'][1]), t_nw=float(r['t'][1]), p=float(r['p'][1]), intercept_pp=float(r['b'][0]), r2=r['r2'])


res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), engine_check=eng,
           env_check=json.loads((HERE / 'env_check.json').read_text()))
res['segments'] = {s: dict(months=int((E['seg'] == s).sum()), ann_excess_pp=nw_mean(E.loc[E['seg'] == s, 'y'])) for s in ('OOS', 'STUCK', 'IS')}
res['standardization'] = dict(mean={k: float(v) for k, v in mu.items()}, sd={k: float(v) for k, v in sd.items()})

# A1
A1 = {}
for v in VARS + SEC:
    A1[v] = dict(primary=uni(PRIM, v), all225=uni(pd.Series(True, index=E.index), v), OOS=uni(E['seg'] == 'OOS', v), IS=uni(E['seg'] == 'IS', v))
ps = np.array([A1[v]['primary']['p'] for v in VARS])
order = np.argsort(ps); holm = np.empty(len(ps)); run_max = 0
for rank, i in enumerate(order):
    run_max = max(run_max, min(1.0, (len(ps) - rank) * ps[i])); holm[i] = run_max
for v, h in zip(VARS, holm):
    A1[v]['primary']['p_holm7'] = float(h)
    A1[v]['primary']['verdict'] = '與超額有統計關聯' if h < 0.05 else ('探索性關聯' if A1[v]['primary']['p'] < 0.05 else '看不出關聯')
for v in SEC:
    A1[v]['primary']['verdict'] = '同期共變（探索）'
res['A1_univariate'] = A1
# 三分位
TER = {}
for v in VARS + SEC:
    m = PRIM & Z[v].notna()
    q = pd.qcut(E.loc[m, v], 3, labels=['低', '中', '高'])
    TER[v] = {str(k): dict(n=int((q == k).sum()), ann_excess_pp=float(E.loc[m, 'y'][q == k].mean()),
                           range=[float(E.loc[m, v][q == k].min()), float(E.loc[m, v][q == k].max())],
                           n_OOS=int(((q == k) & (E.loc[m, 'seg'] == 'OOS')).sum())) for k in ['低', '中', '高']}
res['A1_terciles'] = TER

# A2
def multi(mask, vs, dummy=False):
    m = mask & Z[vs].notna().all(axis=1)
    cols = [np.ones(m.sum())]
    names = ['const']
    if dummy:
        cols.append((E.loc[m, 'seg'] == 'IS').astype(float).to_numpy()); names.append('IS')
    for v in vs:
        cols.append(Z.loc[m, v].to_numpy()); names.append(v)
    r = nw_ols(E.loc[m, 'y'], np.column_stack(cols))
    return dict(T=r['T'], r2=r['r2'], coef={n: dict(b=float(b), t_nw=float(t), p=float(p)) for n, b, t, p in zip(names, r['b'], r['t'], r['p'])}), r, m


A2 = {}
A2['six'], r6, m6 = multi(PRIM, VARS6)
A2['seven'], _, _ = multi(PRIM, VARS)
A2['gap_no_env'], _, _ = multi(PRIM & Z[VARS6].notna().all(axis=1), [], dummy=True)
A2['gap_with_env6'], _, _ = multi(PRIM, VARS6, dummy=True)
A2['gap_with_env7'], _, _ = multi(PRIM, VARS, dummy=True)
d0 = A2['gap_no_env']['coef']['IS']['b']; d1 = A2['gap_with_env6']['coef']['IS']['b']
A2['gap_shrink_frac_env6'] = float(1 - d1 / d0)
d1b = A2['gap_with_env7']['coef']['IS']['b']
A2['gap_no_env_same_sample7'], _, _ = multi(PRIM & Z[VARS].notna().all(axis=1), [], dummy=True)
A2['gap_shrink_frac_env7'] = float(1 - d1b / A2['gap_no_env_same_sample7']['coef']['IS']['b'])
sh = A2['gap_shrink_frac_env6']
A2['verdict_env6'] = '樣本內超額大半可由環境解釋' if sh >= 0.5 else ('環境解釋不了樣本內較高的超額' if sh < 0.25 else '環境只解釋一部分')
res['A2_multivariate'] = A2
res['corr_primary'] = E.loc[PRIM, VARS].corr().round(3).to_dict()

# A3
A3 = {}
for v in VARS + SEC + ['NTGT', 'W_ACT', 'NPOS']:
    A3[v] = {}
    for s in ('OOS', 'STUCK', 'IS'):
        x = E.loc[E['seg'] == s, v].dropna()
        if len(x) == 0:
            A3[v][s] = None; continue
        A3[v][s] = dict(n=int(len(x)), mean=float(x.mean()), median=float(x.median()), q25=float(x.quantile(.25)), q75=float(x.quantile(.75)),
                        smd_vs_primary=float((x.mean() - mu[v]) / sd[v]) if v in mu.index else None)
    x = E.loc['2011-01':'2014-12', v].dropna()
    A3[v]['Y2011_2014'] = dict(n=int(len(x)), mean=float(x.mean()) if len(x) else None,
                               smd_vs_primary=float((x.mean() - mu[v]) / sd[v]) if (v in mu.index and len(x)) else None)
res['A3_segments'] = A3
yr = E.groupby(E.index.year)
res['by_year'] = {str(y): dict(ann_excess_pp=float(g['y'].mean()), **{v: (float(g[v].mean()) if g[v].notna().any() else None) for v in VARS + ['SMB_RET', 'NTGT', 'W_ACT']})
                  for y, g in yr}

# A4 交易層
to = pd.read_csv(SCR / 'oos_trades.csv', dtype={'code': str}); ti = pd.read_csv(SCR / 'is_trades.csv', dtype={'code': str})
to['seg'] = 'OOS'; ti['seg'] = 'IS'
last_oos_sig = pd.read_csv(SCR / 'oos_signals.csv')['sig_date'].max()


def tstats(t):
    r = t['ret'].to_numpy(); n = len(r)
    if n == 0:
        return None
    srt = np.sort(r)[::-1]; k10 = max(1, int(math.ceil(0.1 * n)))
    pos = r[r > 0].sum()
    return dict(n=int(n), win_rate=float((r > 0).mean()), mean_ret=float(r.mean()), median_ret=float(np.median(r)), mean_hold_days=float(t['hold_days'].mean()),
                top10pct_share_of_positive=float(srt[:k10].sum() / pos) if pos > 0 else None,
                top5_share_of_total=float(srt[:5].sum() / r.sum()) if r.sum() != 0 else None,
                mean_ret_ex_top5=float(srt[5:].mean()) if n > 5 else None, sum_ret=float(r.sum()),
                mean_win=float(r[r > 0].mean()) if (r > 0).any() else None, mean_loss=float(r[r <= 0].mean()) if (r <= 0).any() else None,
                share_ret_gt_50pct=float((r > 0.5).mean()))


A4 = {}
A4['trades_OOS_all'] = tstats(to)
A4['trades_OOS_closed_before_stuck'] = tstats(to[pd.to_datetime(to['exit_date']) <= pd.Timestamp('2013-02-28')])
A4['trades_OOS_entered_last_signal_held_into_stuck'] = tstats(to[pd.to_datetime(to['exit_date']).isna() | (pd.to_datetime(to['exit_date']) > pd.Timestamp('2013-02-28'))])
A4['trades_IS_all'] = tstats(ti)
A4['trades_IS_closed'] = tstats(ti[ti['why'] != 'open_end'])
for y0, y1, nm in (('2008-01-01', '2010-12-31', 'OOS_2008_2010'), ('2011-01-01', '2013-01-31', 'OOS_2011_2013'),
                   ('2016-01-01', '2018-12-31', 'IS_2016_2018'), ('2019-01-01', '2026-10-31', 'IS_2019_2026')):
    tt = pd.concat([to, ti]); e_ = pd.to_datetime(tt['ent_date'])
    A4[f'trades_{nm}'] = tstats(tt[(e_ >= y0) & (e_ <= y1)])
ntg = {}
for s in ('OOS', 'IS'):
    x = E.loc[(E['seg'] == s), 'NTGT'].dropna()
    ntg[s] = dict(n_signal_months=int(len(x)), mean=float(x.mean()), share_0=float((x == 0).mean()), share_1_2=float(x.between(1, 2).mean()),
                  share_3_5=float(x.between(3, 5).mean()), share_6p=float((x >= 6).mean()))
A4['ntgt_distribution'] = ntg
WA = {}
for s in ('OOS', 'STUCK', 'IS'):
    g = E[E['seg'] == s]
    gg = g[g['W_ACT'] >= 0.05]
    WA[s] = dict(mean_w_act=float(g['W_ACT'].mean()), mean_npos=float(g['NPOS'].mean()),
                 per_unit_ann_pp_ratio_of_sums=float(gg['y'].sum() / gg['W_ACT'].sum()), n_months_w5=int(len(gg)))
A4['active_weight'] = WA
WA_is = WA['IS']['mean_w_act']
A4['oos_rescaled_to_IS_weight_ann_pp'] = float(WA['OOS']['per_unit_ann_pp_ratio_of_sums'] * WA_is)
g11 = E.loc['2011-01':'2013-01']; g11 = g11[g11['W_ACT'] >= 0.05]
A4['per_unit_2011_01_2013_01'] = dict(ann_pp=float(g11['y'].sum() / g11['W_ACT'].sum()), n=int(len(g11)), rescaled_to_IS_weight=float(g11['y'].sum() / g11['W_ACT'].sum() * WA_is))
mm = PRIM & E['NTGT'].notna()
r = nw_ols(E.loc[mm, 'y'], np.column_stack([np.ones(mm.sum()), E.loc[mm, 'NTGT']]))
A4['excess_on_NTGT_primary'] = dict(T=r['T'], slope_pp_per_pick=float(r['b'][1]), t_nw=float(r['t'][1]), p=float(r['p'][1]))
r = nw_ols(E.loc[PRIM, 'y'], np.column_stack([np.ones(PRIM.sum()), E.loc[PRIM, 'W_ACT']]))
A4['excess_on_W_ACT_primary'] = dict(T=r['T'], slope_pp_per_unit=float(r['b'][1]), t_nw=float(r['t'][1]), p=float(r['p'][1]))
# 同期大盤與個股部位：主樣本 excess 對 MKT_RET×W_ACT（部位的 beta 偏差）
res['A4_trades'] = A4

# A5 情境：六變數模型代入 2011-01～2014-12 環境平均
b6 = r6['b']; V6 = r6['V']


def predict(zbar):
    x = np.concatenate([[1.0], zbar])
    return dict(pred_ann_pp=float(x @ b6), se=float(math.sqrt(x @ V6 @ x)))


A5 = {}
for nm, (a, b) in dict(Y2011_2014=('2011-01', '2014-12'), Y2011_2012=('2011-01', '2012-12'), OOS=('2008-02', '2013-01'),
                       IS=('2015-07', '2026-10'), Y2008_2010=('2008-02', '2010-12'), Y2019_2026=('2019-01', '2026-10')).items():
    zb = Z.loc[a:b, VARS6].mean().to_numpy()
    A5[nm] = dict(zbar={v: float(x) for v, x in zip(VARS6, zb)}, **predict(zb), realized_ann_pp=float(E.loc[a:b, 'y'].mean()))
A5['realized_2011_01_2013_01_active'] = nw_mean(E.loc['2011-01':'2013-01', 'y'])
A5['realized_2011_01_2014_12_incl_stuck'] = nw_mean(E.loc['2011-01':'2014-12', 'y'])
res['A5_scenario'] = A5

# A6 燈號
med_rev = E.loc[PRIM, 'REV_DISP'].median(); med_vol = E.loc[PRIM, 'MKT_VOL'].median()
C = pd.DataFrame(dict(c1=E['MKT_MOM12'] > 0, c2=E['SMB_MOM12'] > 0, c3=E['TURN_CHG'] > 0,
                      c4=(E['REV_DISP'] > med_rev).where(E['REV_DISP'].notna()), c5=E['MKT_VOL'] < med_vol))
score = C[['c1', 'c2', 'c3', 'c5']].sum(axis=1) + C['c4'].fillna(0.5)    # REV_DISP 缺值（2013 營收月）算半分
light = pd.Series(np.where(score >= 4, '綠', np.where(score >= 2, '黃', '紅')), index=E.index)
E['light'] = light; E['score'] = score
A6 = dict(thresholds=dict(REV_DISP_median=float(med_rev), MKT_VOL_median=float(med_vol)))
for nm, msk in (('primary', PRIM), ('OOS', E['seg'] == 'OOS'), ('STUCK', E['seg'] == 'STUCK'), ('IS', E['seg'] == 'IS'),
                ('Y2011_2014', (E.index >= pd.Period('2011-01', 'M')) & (E.index <= pd.Period('2014-12', 'M')))):
    A6[nm] = {}
    for c in ('綠', '黃', '紅'):
        x = E.loc[msk & (E['light'] == c), 'y']
        A6[nm][c] = dict(n=int(len(x)), ann_excess_pp=float(x.mean()) if len(x) else None)
# 各條件單獨：符合／不符合的主樣本平均
A6['each_condition_primary'] = {k: dict(yes=float(E.loc[PRIM & (C[k] == True), 'y'].mean()), no=float(E.loc[PRIM & (C[k] == False), 'y'].mean()),
                                        n_yes=int((PRIM & (C[k] == True)).sum())) for k in C.columns}
x = E.loc[PRIM & E['score'].notna()]
r = nw_ols(x['y'], np.column_stack([np.ones(len(x)), x['score']]))
A6['excess_on_score_primary'] = dict(T=r['T'], slope_pp_per_condition=float(r['b'][1]), t_nw=float(r['t'][1]), p=float(r['p'][1]))
A6['latest'] = {str(m): dict(score=float(E.loc[m, 'score']), light=str(E.loc[m, 'light'])) for m in E.index[-6:]}
res['A6_light'] = A6
E[['seg', 'y', 'score', 'light']].to_csv(HERE / 'light_monthly.csv')
res['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print('ok', res['maxrss_mb'])

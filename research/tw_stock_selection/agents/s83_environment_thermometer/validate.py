#!/usr/bin/env python3
"""S83 第 3 步：驗證段（2016-01～2026-10）。先核對 PREREG sha256。python3 -I -B validate.py → results.json＋csv"""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s83lib as E
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0]
PRE = json.load(open(HERE / 'PREREG.json', encoding='utf-8'))
I = pd.read_csv(HERE / 'indicators_monthly.csv', index_col=0); O = pd.read_csv(HERE / 'outcomes_monthly.csv', index_col=0)
mon = pd.PeriodIndex(I.index, freq='M'); VS = pd.Period(E.VALID_START, 'M'); TE = pd.Period(E.TRAIN_END, 'M')
va = np.asarray(mon >= VS); tr = np.asarray(mon <= TE)
ANN = dict(sync1=12, sync3=4, lead1=12, lead3=4, lead6=2, sig_lead1=12, sig_lead3=4)
thr = PRE['thresholds']
OUT = dict(prereg_sha256=sha(HERE / 'PREREG.json'), n_valid_months=int(va.sum()))

def family(inds, outs, mask):
    X = {i: I.loc[mask, i].to_numpy() for i in inds}; Y = {o: O.loc[mask, o].to_numpy() for o in outs}
    sign = {(i, o): PRE['pairs'][f'{i}|{o}']['sign'] for i in inds for o in outs}
    return E.shift_null(X, Y, sign)

rows = []
for fam, inds, outs in (('lead', E.IND, E.LEAD_OUT), ('sync', E.SYNC_IND, E.SYNC_OUT), ('secondary', E.IND, E.SEC_OUT)):
    obs, p_ind, p_fw, ns = family(inds, outs, va)
    for (i, o), v in obs.items():
        pp = PRE['pairs'][f'{i}|{o}']
        x = I.loc[va, i].to_numpy(); y = O.loc[va, o].to_numpy()
        tm = E.tercile_means(x, y, thr[i]['lo'], thr[i]['hi']); tt = pp['train_terciles']
        d_tr = (tt['high']['mean'] - tt['low']['mean']) if tt['high']['mean'] is not None and tt['low']['mean'] is not None else np.nan
        enough = tm['high']['n'] >= 12 and tm['low']['n'] >= 12
        d_va = (tm['high']['mean'] - tm['low']['mean']) if tm['high']['mean'] is not None and tm['low']['mean'] is not None else np.nan
        # 驗證段內部三等分（描述用；門檻取自驗證段本身，不列入過關）
        m = np.isfinite(x) & np.isfinite(y); q = np.quantile(x[m], [1 / 3, 2 / 3]); tin = E.tercile_means(x, y, q[0], q[1])
        a = ANN[o] * 100
        cond_c = bool(enough and np.isfinite(d_tr) and np.sign(d_va) == np.sign(d_tr))
        rows.append(dict(family=fam, indicator=i, outcome=o, train_rho=pp['train_rho'], train_n=pp['n'], valid_rho=v * pp['sign'], valid_n=int(m.sum()), same_sign=bool(v > 0),
                         p_single=p_ind[(i, o)], p_fwer=p_fw[(i, o)], n_shifts=ns,
                         tr_low_ann_pp=tt['low']['mean'] * a if tt['low']['mean'] is not None else None, tr_mid_ann_pp=tt['mid']['mean'] * a if tt['mid']['mean'] is not None else None, tr_high_ann_pp=tt['high']['mean'] * a if tt['high']['mean'] is not None else None,
                         va_low_n=tm['low']['n'], va_mid_n=tm['mid']['n'], va_high_n=tm['high']['n'],
                         va_low_ann_pp=tm['low']['mean'] * a if tm['low']['mean'] is not None else None, va_mid_ann_pp=tm['mid']['mean'] * a if tm['mid']['mean'] is not None else None, va_high_ann_pp=tm['high']['mean'] * a if tm['high']['mean'] is not None else None,
                         groups_enough=bool(enough), tercile_same_dir=cond_c,
                         vin_low_ann_pp=tin['low']['mean'] * a, vin_mid_ann_pp=tin['mid']['mean'] * a, vin_high_ann_pp=tin['high']['mean'] * a,
                         PASS=bool(fam != 'secondary' and v > 0 and p_fw[(i, o)] < 0.05 and cond_c)))
Rz = pd.DataFrame(rows); Rz.to_csv(HERE / 'validation_pairs.csv', index=False)
OUT['pass'] = {f: Rz[(Rz.family == f) & Rz.PASS][['indicator', 'outcome']].values.tolist() for f in ('lead', 'sync')}
OUT['summary'] = {f: dict(n_tests=int((Rz.family == f).sum()), n_same_sign=int(((Rz.family == f) & Rz.same_sign).sum()), n_p_single_lt05=int(((Rz.family == f) & (Rz.p_single < 0.05) & Rz.same_sign).sum()),
                          n_fwer_lt05=int(((Rz.family == f) & (Rz.p_fwer < 0.05)).sum()), min_p_fwer=float(Rz[Rz.family == f].p_fwer.min()),
                          corr_train_valid_rho=float(np.corrcoef(Rz[Rz.family == f].train_rho, Rz[Rz.family == f].valid_rho)[0, 1])) for f in ('lead', 'sync', 'secondary')}

# 訓練段自己的檢定（描述；方向取自訓練段本身 → 偏樂觀）
trows = []
for fam, inds, outs in (('lead', E.IND, E.LEAD_OUT), ('sync', E.SYNC_IND, E.SYNC_OUT)):
    obs, p_ind, p_fw, ns = family(inds, outs, tr)
    for (i, o), v in obs.items():
        trows.append(dict(family=fam, indicator=i, outcome=o, abs_rho=v, p_single=p_ind[(i, o)], p_fwer=p_fw[(i, o)]))
pd.DataFrame(trows).to_csv(HERE / 'train_pairs_insample.csv', index=False)

# 全期（描述；含年代間差異）
frows = []
for i in E.IND:
    for o in E.SYNC_OUT + E.LEAD_OUT:
        frows.append(dict(indicator=i, outcome=o, full_rho=E.spearman(I[i].to_numpy(), O[o].to_numpy())))
pd.DataFrame(frows).to_csv(HERE / 'full_period_rho.csv', index=False)

# 年代描述
ERAS = [('2008–2012', '2008-01', '2012-12'), ('2013–2015', '2013-01', '2015-12'), ('2016–2018', '2016-01', '2018-12'), ('2019–2023', '2019-01', '2023-12'), ('2024–2026-10', '2024-01', '2026-10')]
er = []
for nm, a, b in ERAS:
    m = np.asarray((mon >= pd.Period(a, 'M')) & (mon <= pd.Period(b, 'M')))
    r = dict(era=nm, months=int(m.sum()), ex_ann_pp_simple=float(O.loc[m, 'sync1'].mean() * 1200), stockfrac=float(O.loc[m, 'stockfrac'].mean()))
    for i in E.IND:
        x = I.loc[m, i]
        r[i] = float(x.mean()); r[i + '_median'] = float(x.median())
        r[i + '_share_high'] = float((x.dropna() > thr[i]['hi']).mean()); r[i + '_share_low'] = float((x.dropna() < thr[i]['lo']).mean())
    r['n_surge_ge3'] = float(I.loc[m, 'n_surge_ge3'].mean())
    er.append(r)
ER = pd.DataFrame(er); ER.to_csv(HERE / 'era_means.csv', index=False)
ex_era = ER['ex_ann_pp_simple'].to_numpy()
OUT['era_rank_corr_with_excess(5 個年代，僅描述)'] = {i: E.spearman(np.r_[ER[i].to_numpy()] * 1.0, ex_era) if False else float(pd.Series(ER[i]).rank().corr(pd.Series(ex_era).rank())) for i in E.IND}
# 36 個月移動
ex36 = O['sync1'].rolling(36, min_periods=36).mean() * 1200
roll = pd.DataFrame({'ex36_ann_pp': ex36})
r36 = {}
for i in E.IND:
    roll[i + '_36'] = I[i].rolling(36, min_periods=30).mean()
    r36[i] = E.spearman(roll[i + '_36'].to_numpy(), ex36.to_numpy())
roll.to_csv(HERE / 'rolling36.csv'); OUT['rolling36_spearman(高度重疊，僅描述)'] = r36
# 年度
yr = pd.concat([I[E.IND + ['n_surge_ge3']], O[['sync1']]], axis=1).groupby(mon.year).mean(); yr['ex_ann_pp_simple'] = yr.pop('sync1') * 1200
yr.to_csv(HERE / 'yearly_means.csv')
OUT['yearly_rank_corr_with_excess(19 個年度，僅描述)'] = {i: float(yr[i].rank().corr(yr['ex_ann_pp_simple'].rank())) for i in E.IND}

# 歷史分佈與目前讀數
dist = []
for i in E.IND + ['n_surge_ge3']:
    x = I[i].dropna(); cur = float(I[i].iloc[-1]); xt = I.loc[tr, i].dropna(); x19 = I.loc[np.asarray(mon >= pd.Period('2019-01', 'M')), i].dropna(); x18 = I.loc[np.asarray(mon <= pd.Period('2018-12', 'M')), i].dropna()
    dist.append(dict(indicator=i, p10=float(x.quantile(.1)), p25=float(x.quantile(.25)), p50=float(x.median()), p75=float(x.quantile(.75)), p90=float(x.quantile(.9)), min=float(x.min()), max=float(x.max()),
                     train_lo=thr.get(i, {}).get('lo'), train_hi=thr.get(i, {}).get('hi'), median_2008_2018=float(x18.median()), median_2019_on=float(x19.median()),
                     current=cur, current_month=I.index[-1], current_date=I['date'].iloc[-1], current_pctile_full=float((x <= cur).mean()), current_pctile_2008_2018=float((x18 <= cur).mean()), current_pctile_2019_on=float((x19 <= cur).mean()),
                     last3=[float(v) for v in I[i].tail(3)], mean_last12=float(I[i].tail(12).mean())))
pd.DataFrame(dist).to_csv(HERE / 'distribution_and_current.csv', index=False)
# 指標彼此相關（驗證段）
I.loc[va, E.IND].corr(method='spearman').to_csv(HERE / 'indicator_corr_valid.csv')
# 自我相關（指標黏不黏）
OUT['autocorr_lag1'] = {i: float(I[i].autocorr(1)) for i in E.IND}; OUT['excess_autocorr'] = {f'lag{l}': float(O['sync1'].autocorr(l)) for l in (1, 2, 3, 6, 12)}
json.dump(OUT, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200); pd.set_option('display.max_columns', 40)
print(json.dumps({k: OUT[k] for k in ('pass', 'summary', 'excess_autocorr')}, ensure_ascii=False, indent=1))
print(Rz[['family', 'indicator', 'outcome', 'train_rho', 'valid_rho', 'p_single', 'p_fwer', 'va_low_n', 'va_high_n', 'va_low_ann_pp', 'va_mid_ann_pp', 'va_high_ann_pp', 'vin_low_ann_pp', 'vin_high_ann_pp', 'PASS']].round(3).to_string())

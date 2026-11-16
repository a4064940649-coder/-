#!/usr/bin/env python3
"""S25 獨立複製 A2（產業動能 vs 因子動能）＋ A3（扣同業的選股報酬）。輸出 scratchpad/s25rep/a23.json。
依 SPEC.json 文字自寫；未讀、未 import、未執行 s25lib.py／prep.py／describe.py／train.py／validate.py／posthoc.py。
python3 -I -B rep_s25_a23.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rep_s25_lib as L
from rep_s25_lib import s10lib, s17lib, stlib, COSTS
from rep_s25_stats import tstat, ann_stats, ols_nw

H = Path(__file__).resolve().parent
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, INNOV, RT, fix = L.load_panel()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'NEARHI', 'V60'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
MC = L.mcap_panel(P)
print('載入', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res = {'note': '獨立複製：A2、A3 依 SPEC 文字自寫；引擎 s20lib→s17lib→s10lib 預設旗標'}

# ================================================================ A2
me = L.month_end_rows(P, '2015-01')
months = list(me.index)
IND_ME, names = L.industry_rows(P, me.values)
Gme = P.G[me.values]
M = pd.read_csv(L.AG / 's24_return_attribution' / 'monthly.csv', index_col='month'); M.index = pd.PeriodIndex(M.index, freq='M')
TS = pd.read_csv(L.REP / 'tsfm_series.csv', index_col=0); TS.index = pd.PeriodIndex(TS.index, freq='M')
NI = len(names)
indret = {w: pd.DataFrame(np.nan, index=months[1:], columns=range(NI)) for w in ('VW', 'EW')}
nind = {}
for i in range(len(months) - 1):
    t = me.iloc[i]; m1 = months[i + 1]
    mc = MC[t]
    ret = Gme[i + 1] / Gme[i] - 1
    elig = (P.MK[t] == 1) & P.has[t] & P.not91 & ~INNOV[t] & np.isfinite(mc) & (mc > 0) & np.isfinite(ret)
    ind = IND_ME[int(t)]
    ok = elig & (ind >= 0)
    cnt = np.bincount(ind[ok], minlength=NI)
    sw = np.bincount(ind[ok], weights=mc[ok], minlength=NI)
    swr = np.bincount(ind[ok], weights=mc[ok] * ret[ok], minlength=NI)
    sr = np.bincount(ind[ok], weights=ret[ok], minlength=NI)
    good = cnt >= 5
    with np.errstate(invalid='ignore', divide='ignore'):
        indret['VW'].loc[m1] = np.where(good, swr / sw, np.nan)
        indret['EW'].loc[m1] = np.where(good, sr / cnt, np.nan)
    nind[str(m1)] = int(good.sum())


def indmom(R, form=6, nlong=3):
    """形成期 t−5…t（以報酬月份計：形成月底 t＝報酬月 t 結束），前 3 名等權做多、後 3 名等權做空，持有下一個月。回傳 Series（index＝持有月）。"""
    cum = (1 + R).rolling(form, min_periods=form).apply(np.prod, raw=True) - 1
    out = {}
    idx = list(R.index)
    for i in range(len(idx) - 1):
        m, m1 = idx[i], idx[i + 1]
        c = cum.loc[m]; nxt = R.loc[m1]
        ok = c.notna() & nxt.notna()
        if ok.sum() < 2 * nlong:
            continue
        cs = c[ok].sort_values(kind='stable')
        lo, hi = cs.index[:nlong], cs.index[-nlong:]
        out[m1] = nxt[hi].mean() - nxt[lo].mean()
    return pd.Series(out)


a2 = {}
samp_ind = [v for k, v in nind.items() if pd.Period(k, 'M') >= pd.Period('2017-02', 'M')]
a2['n_industries'] = dict(min=int(min(samp_ind)), median=float(np.median(samp_ind)), max=int(max(samp_ind)))
for w in ('VW', 'EW'):
    IM = indmom(indret[w])
    D = pd.DataFrame({'IND': IM, 'WML': TS['WML'], 'WIN': TS['WIN'], 'MKT': M['MKT_VW'], 'MOM': M['MOM']}).dropna()
    D = D[D.index >= pd.Period('2017-02', 'M')]
    o = dict(months=[str(D.index[0]), str(D.index[-1])], n=int(len(D)), INDMOM=ann_stats(D['IND']), FMOM_WML=ann_stats(D['WML']),
             corr_IND_WML=float(D['IND'].corr(D['WML'])), corr_IND_MOM=float(D['IND'].corr(D['MOM'])), corr_WML_MOM=float(D['WML'].corr(D['MOM'])))
    rg = {}
    for nm, y, xs in (('IND_on_const', 'IND', []), ('IND_on_WML', 'IND', ['WML']), ('IND_on_WML_MKT', 'IND', ['WML', 'MKT']),
                      ('IND_on_WML_MKT_MOM', 'IND', ['WML', 'MKT', 'MOM']), ('IND_on_WIN_MKT', 'IND', ['WIN', 'MKT']),
                      ('WML_on_const', 'WML', []), ('WML_on_IND', 'WML', ['IND']), ('WML_on_IND_MKT', 'WML', ['IND', 'MKT']),
                      ('WML_on_IND_MKT_MOM', 'WML', ['IND', 'MKT', 'MOM']), ('IND_on_MOM_MKT', 'IND', ['MOM', 'MKT'])):
        rg[nm] = ols_nw(D[y].to_numpy(), D[xs].to_numpy() if xs else None)
    o['regs'] = rg
    a2[w] = o
    print('A2', w, {k: (round(v['alpha_ann_pct'], 2), round(v['alpha_t'], 2)) for k, v in rg.items()}, flush=True)
res['A2'] = a2
del IND_ME
gc.collect()

# ================================================================ A3
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
nav, lg, st = s10lib.simulate_s10(P, S0, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
acc = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_trades=int(len(lg)))
acc['ok'] = bool(abs(acc['since2019'] - 0.4813) < 0.00025 and abs(acc['full'] - 0.3408) < 0.00025 and abs(acc['maxdd'] + 0.3340) < 0.00025)
res['A3_acceptance'] = acc
print('A0', acc, flush=True)
del nav, st
gc.collect()

O, CF, G, REF, E50, E50O, MK = P.O, P.CF, P.G, P.REF, P.E50, P.E50O, P.MK
T = P.T
anyl = MK > 0
last_listed = np.where(anyl.any(axis=0), T - 1 - np.argmax(anyl[::-1], axis=0), -1)   # 每檔最後掛牌列
okO = lambda t, j: np.isfinite(O[t, j]) & (O[t, j] > 0)


def hold_ret(js, e, x, mode):
    """js：股票欄位陣列；e＝進場列（開盤買）。mode='open'：x 日開盤賣（x 無開盤用 x−1 收盤）；mode='close'：x 日收盤。
    中途下市（最後掛牌列 < 結束列）用最後掛牌日收盤。"""
    js = np.asarray(js, int)
    base = CF[e, js] / O[e, js]
    end_close = x - 1 if mode == 'open' else x
    ll = last_listed[js]
    out = np.empty(len(js))
    for i, j in enumerate(js):
        if ll[i] < end_close:
            out[i] = base[i] * G[ll[i], j] / G[e, j] - 1
        elif mode == 'open' and okO(x, j):
            out[i] = base[i] * G[x - 1, j] / G[e, j] * O[x, j] / REF[x, j] - 1
        else:
            out[i] = base[i] * G[end_close, j] / G[e, j] - 1
    return out


sigks = sorted(set(int(k) for k in lg['sig_t']))
IND_K, _ = L.industry_rows(P, sigks)
rows = []
for r in lg.itertuples():
    j, k, e, x = int(r.j), int(r.sig_t), int(r.ent_t), int(r.exit_t)
    if r.why == 'open_end':
        mode, xe = 'close', T - 1
    elif r.why == 'delist':
        mode, xe = 'close', int(last_listed[j])
    else:
        mode, xe = 'open', x
    ret = hold_ret([j], e, xe, mode)[0]
    r50 = (E50O[xe] / E50O[e] - 1) if mode == 'open' else (E50[xe] / E50O[e] - 1)
    ind = IND_K[k]
    rec = dict(code=r.code, j=j, sig_date=P.IDX[k], month=P.IDX[k].to_period('M'), e=e, x=xe, why=r.why, eng_ret=r.ret, ret=ret, r50=r50,
               industry=names[ind[j]] if ind[j] >= 0 else None)
    for tag, pool in (('U', U[k]), ('BROAD', (MK[k] == 1) & ~INNOV[k])):
        if ind[j] < 0:
            rec['n_' + tag] = 0; rec['peer_' + tag] = np.nan; continue
        same = pool & (ind == ind[j])
        same_incl = same.copy(); same_incl[j] = True                         # 產業（含自己）在股池內
        cand_all = np.flatnonzero(same_incl)
        mcv = np.where(np.isfinite(MC[k, cand_all]), MC[k, cand_all], -np.inf)
        big = cand_all[int(np.argmax(mcv))]
        peers = np.flatnonzero(same & np.isfinite(O[e]) & (O[e] > 0))
        peers = peers[(peers != j) & (peers != big)]
        rec['n_' + tag] = int(len(peers))
        if tag == 'U':
            rec['big'] = str(P.COLS[big])
        if len(peers) >= 5:
            pr = hold_ret(peers, e, xe, mode)
            rec['peer_' + tag] = float(np.nanmean(pr))
            if tag == 'U':
                w = MC[k, peers]; okw = np.isfinite(w) & (w > 0) & np.isfinite(pr)
                rec['peer_VW_U'] = float(np.sum(w[okw] * pr[okw]) / np.sum(w[okw]))
        else:
            rec['peer_' + tag] = np.nan
            if tag == 'U':
                rec['peer_VW_U'] = np.nan
    rows.append(rec)
TR = pd.DataFrame(rows)
TR['ex50'] = TR['ret'] - TR['r50']
TR['exP_U'] = TR['ret'] - TR['peer_U']
TR['peer50_U'] = TR['peer_U'] - TR['r50']
TR['exP_BROAD'] = TR['ret'] - TR['peer_BROAD']
TR['exP_VW_U'] = TR['ret'] - TR['peer_VW_U']


def summ(df, col):
    d = df[np.isfinite(df[col])]
    v = d[col].to_numpy()
    mm = d.groupby('month')[col].mean()
    top = d.nlargest(5, col)
    rest = d.drop(top.index)
    mr = rest.groupby('month')[col].mean()
    return dict(n=int(len(v)), n_months=int(len(mm)), mean_pct=v.mean() * 100, median_pct=float(np.median(v)) * 100, win_rate=float((v > 0).mean()),
                t=tstat(v), t_cluster_month=tstat(mm.to_numpy()), drop_top5_mean_pct=rest[col].mean() * 100, drop_top5_t_cluster=tstat(mr.to_numpy()),
                top5=[dict(code=c, sig=str(s.date()), val_pct=float(x) * 100) for c, s, x in zip(top['code'], top['sig_date'], top[col])])


a3 = dict(n_trades=int(len(TR)), why=TR['why'].value_counts().to_dict(), gross_vs_engine_ret_corr=float(np.corrcoef(TR['ret'], TR['eng_ret'])[0, 1]),
          gross_minus_engine_mean_pct=float((TR['ret'] - TR['eng_ret']).mean() * 100), industry_missing=int(TR['industry'].isna().sum()))
per = {'all': TR, 'train': TR[TR['sig_date'] < pd.Timestamp('2022-01-01')], 'valid': TR[TR['sig_date'] >= pd.Timestamp('2022-01-01')]}
for nm, df in per.items():
    g5 = df[df['n_U'] >= 5]; l5 = df[df['n_U'] < 5]
    a3[nm] = dict(n_peer_ge5=int(len(g5)), n_peer_lt5=int(len(l5)),
                  main={c: summ(g5, c) for c in ('ret', 'r50', 'peer_U', 'ex50', 'exP_U', 'peer50_U')},
                  peer_lt5=dict(n=int(len(l5)), ex50_mean_pct=float(l5['ex50'].mean() * 100) if len(l5) else None,
                                ex50_median_pct=float(l5['ex50'].median() * 100) if len(l5) else None),
                  all_trades_ex50=summ(df, 'ex50'),
                  robust_BROAD=dict(exP=summ(df[df['n_BROAD'] >= 5], 'exP_BROAD')),
                  robust_VW_U=summ(g5, 'exP_VW_U'))
g5 = TR[TR['n_U'] >= 5]
a3['peer_count_U'] = dict(median=float(g5['n_U'].median()), q10=float(g5['n_U'].quantile(0.1)), q90=float(g5['n_U'].quantile(0.9)))
a3['biggest_excluded_top'] = g5['big'].value_counts().head(10).to_dict()
bi = {}
for ind, g in g5.groupby('industry'):
    bi[ind] = dict(n=int(len(g)), ex50_mean_pct=g['ex50'].mean() * 100, exP_mean_pct=g['exP_U'].mean() * 100, peer50_mean_pct=g['peer50_U'].mean() * 100)
a3['by_industry'] = dict(sorted(bi.items(), key=lambda kv: -kv[1]['n']))
sh = g5[(g5['industry'] == '航運業') & (g5['sig_date'].dt.year.isin([2020, 2021]))]
a3['shipping_2020_2021'] = dict(n=int(len(sh)), peer50_mean_pct=float(sh['peer50_U'].mean() * 100), exP_mean_pct=float(sh['exP_U'].mean() * 100))
res['A3'] = a3
TR.drop(columns=['sig_date']).assign(sig_date=TR['sig_date'].dt.strftime('%Y-%m-%d'), month=TR['month'].astype(str)).to_csv(L.REP / 'peer_trades_rep.csv', index=False)
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(L.REP / 'a23.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
m = a3['all']['main']
print('A3 all', {c: (round(v['mean_pct'], 3), round(v['t_cluster_month'], 3), round(v['drop_top5_mean_pct'], 3), round(v['drop_top5_t_cluster'], 3)) for c, v in m.items()})
for s in ('train', 'valid'):
    print(s, {c: round(v['mean_pct'], 3) for c, v in a3[s]['main'].items()}, round(a3[s]['main']['exP_U']['drop_top5_mean_pct'], 3))
print('n', a3['all']['n_peer_ge5'], a3['all']['n_peer_lt5'], a3['why'], a3['peer_count_U'], 'BROAD', round(a3['all']['robust_BROAD']['exP']['mean_pct'], 3), a3['all']['robust_BROAD']['exP']['n'],
      'VW', round(a3['all']['robust_VW_U']['mean_pct'], 3), 'corr', a3['gross_vs_engine_ret_corr'], a3['gross_minus_engine_mean_pct'], a3['shipping_2020_2021'], res['meta'])

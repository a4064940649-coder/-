#!/usr/bin/env python3
"""S84 正式執行：大盤急跌後那一次換股的歷史。先核對 PREREG／程式 sha256。
python3 -I -B run.py → results.json、signal_months.csv、events.csv、first_batch_trades.csv、daily_analog.csv"""
import sys, json, time, resource, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與登記的 sha256 不符'
if (HERE / 'code.sha256').exists():
    for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
        h, f = line.split(); assert sha(HERE / f) == h, f'{f} 與登記的 sha256 不符'
import numpy as np, pandas as pd
import s84lib as L
from s84lib import K, D, dist, hpos

t0c = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U = K.load_cont()
F, RT = K.surge(P, 'B2')
sg = K.signal(P, U, RT, 0)
nav_s, lg, st = K.sim(P, sg, U, 1e6)
S2, ekw = L.s10lib.variant_S(sg, 'E0'); assert ekw == {}
T = P.T; IDX = pd.DatetimeIndex(P.IDX); E = np.asarray(P.E50, float)
NAV = nav_s.reindex(IDX).to_numpy(float); SF = st['stockfrac'].reindex(IDX).to_numpy(float)
# ---------------------------------------------------------------- 驗收
acc = dict(cagr=K.cagr(nav_s), final=float(nav_s.iloc[-1]), n_trades=int(len(lg)), n_sig=len(sg['info']))
acc['A1'] = bool(abs(acc['cagr'] - 0.2351900697944238) < 1e-12 and round(acc['final'], 2) == 53078883.65 and acc['n_trades'] == 632 and acc['n_sig'] == 225)
n2 = L.sim_new(P, S2, U, None, None, 'hist')[0]
acc['A2'] = bool(np.array_equal(n2.to_numpy(), nav_s.to_numpy(), equal_nan=True))
TF = pd.read_csv(K.AG / 's82_big_winner_dependence' / 'trades_features.csv.gz', dtype={'code': str}, parse_dates=['ent_date'])
lg = lg.copy(); lg['code'] = lg['code'].astype(str)
mg = lg.merge(TF[['code', 'ent_date', 'ret', 'ex', 'top5', 'b0050']].rename(columns={'ret': 'ret82'}), on=['code', 'ent_date'], how='left', validate='1:1')
acc['A3'] = bool(len(TF) == 632 and mg['ret82'].notna().all() and np.allclose(mg['ret'], mg['ret82'], rtol=0, atol=1e-12))
print('驗收', acc, f'{time.time()-t0c:.0f}s', rss(), 'MB', flush=True)
assert acc['A1'] and acc['A2'] and acc['A3'], '驗收失敗'

# ---------------------------------------------------------------- 訊號月表
info = sg['info']; ks = [x['k'] for x in info]; n = len(ks)
ent = lg['ent_t'].to_numpy(); ext = lg['exit_t'].to_numpy(); opn = (lg['why'] == 'open_end').to_numpy()
FW = {h: D.fwd_matrix(P, ks, h) for h in (1, 3, 6, 12)}
rows = []
for i, x in enumerate(info):
    k = x['k']; kn = ks[i + 1] if i + 1 < n else T - 1
    r = dict(i=i, rev_month=x['month'], sig_date=x['sig_date'], exec_date=str(IDX[k + 1].date()) if k + 1 < T else None, k=k)
    r.update(L.e50_metrics(E, k, ks[i - 1] if i else None))
    r['c_r10'] = bool(r['r10'] <= -0.10); r['c_r20'] = bool(r['r20'] <= -0.10); r['c_gap60'] = bool(r['gap60'] <= -0.08)
    r['MAIN'] = bool(r['c_r10'] or r['c_r20'] or r['c_gap60']); r['STRICT'] = bool(r['r10'] <= -0.15 or r['r20'] <= -0.15)
    r['WIDE'] = bool(r['MAIN'] or r['dd60'] <= -0.15 or (np.isfinite(r['m2m']) and r['m2m'] <= -0.10))
    s = RT['S'].loc[pd.Period(x['month'], 'M')].to_numpy(); s = L.cslib.ins15_mask_row(P, pd.Period(x['month'], 'M'), k, s)
    fin = U[k] & np.isfinite(s)
    r['n_fin'] = int(fin.sum()); r['n_ma'] = int(x['n_ma']); assert r['n_ma'] == int((fin & P.F_MA60[k]).sum())
    r['ma_frac'] = r['n_ma'] / r['n_fin']; r['breadth'] = float((U[k] & P.F_MA60[k]).sum() / U[k].sum()); r['n_tgt'] = int(x['n_tgt'])
    held = (ent <= k) & ((ext > k) | opn); sold = held & (ext == k + 1) & ~opn
    r['held'] = int(held.sum()); r['sold'] = int(sold.sum()); r['sold_frac'] = (r['sold'] / r['held']) if r['held'] else np.nan
    r['bought'] = int((ent == k + 1).sum())
    r['sf_sig'] = SF[k]; r['sf_exec'] = SF[k + 1] if k + 1 < T else np.nan; r['sf_month_avg'] = float(np.nanmean(SF[k + 1:kn + 1])) if kn > k else np.nan
    r['npos_after'] = r['held'] - r['sold'] + r['bought']
    for m in (1, 3, 6, 12):
        h = hpos(IDX, k, m)
        r[f'acct_{m}'] = NAV[h] / NAV[k] - 1 if h is not None else np.nan
        r[f'e50_{m}'] = E[h] / E[k] - 1 if h is not None else np.nan
        r[f'acct_x{m}'] = r[f'acct_{m}'] - r[f'e50_{m}']
    h12 = hpos(IDX, k, 12)
    r['acct_fall12'] = NAV[k:h12 + 1].min() / NAV[k] - 1 if h12 is not None else np.nan
    r['e50_fall12'] = E[k:h12 + 1].min() / E[k] - 1 if h12 is not None else np.nan
    tg = np.flatnonzero(sg['sig'][k])
    for h in (1, 3, 6, 12):
        Fm, b = FW[h]; v = Fm[i, tg]; v = v[np.isfinite(v)]
        r[f'list_{h}'] = float(v.mean()) if len(v) and np.isfinite(b[i]) else np.nan
        r[f'list_b{h}'] = b[i] if len(v) else np.nan; r[f'list_x{h}'] = r[f'list_{h}'] - r[f'list_b{h}']
    rows.append(r)
M = pd.DataFrame(rows)
ep = np.zeros(n, int); first = np.zeros(n, bool); e = 0
for i in range(n):
    if M.MAIN[i]:
        if i == 0 or not M.MAIN[i - 1]:
            e += 1; first[i] = True
        ep[i] = e
M['episode'] = ep; M['FIRST'] = first
M['PRE'] = False
for i in np.flatnonzero(first):
    if i > 0:
        M.loc[i - 1, 'PRE'] = True
assert not (M.PRE & M.MAIN).any()

# ---------------------------------------------------------------- Q5 新帳戶
for odd in ('all', 'hist'):
    cols = {c: np.full(n, np.nan) for c in ['n1', 'n3', 'n6', 'n12', 'nx1', 'nx3', 'nx6', 'nx12', 'ndeep12', 'nmdd12', 'nsf_m1', 'nbuy1']}
    for i, k in enumerate(ks):
        if k + 1 >= T:
            continue
        h12 = hpos(IDX, k, 12); end = h12 if h12 is not None else T - 1
        nv, lg_, st_ = L.sim_new(P, S2, U, k, end, odd)
        W = nv.reindex(IDX).to_numpy(float); assert W[k] == L.CAP and np.isfinite(W[k:end + 1]).all()
        for m in (1, 3, 6, 12):
            h = hpos(IDX, k, m)
            if h is not None:
                cols[f'n{m}'][i] = W[h] / L.CAP - 1; cols[f'nx{m}'][i] = cols[f'n{m}'][i] - (E[h] / E[k] - 1)
        if h12 is not None:
            seg = W[k:h12 + 1]; cols['ndeep12'][i] = seg.min() / L.CAP - 1; cols['nmdd12'][i] = (seg / np.maximum.accumulate(seg) - 1).min()
        kn = ks[i + 1] if i + 1 < n else T - 1
        sf = st_['stockfrac'].reindex(IDX).to_numpy(float)
        cols['nsf_m1'][i] = float(np.nanmean(sf[k + 1:min(kn, end) + 1])) if kn > k else np.nan
        cols['nbuy1'][i] = sum(1 for f in st_['fills'] if f[0] == k + 1 and f[2] == 'B')
    for c, v in cols.items():
        M[f'{c}_{odd}'] = v
    print('新帳戶', odd, f'{time.time()-t0c:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 彙總
rng = np.random.default_rng(84)
G = dict(FIRST=M.FIRST, MAIN_all=M.MAIN, STRICT=M.STRICT, WIDE=M.WIDE, PRE=M.PRE, normal=~M.MAIN, all=pd.Series(True, index=M.index))
out = dict(task='S84 大盤急跌後那一次換股的歷史（描述）', prereg_sha=sha(HERE / 'PREREG.json'), accept=acc, n_signal_months=n,
           counts={g: int(m.sum()) for g, m in G.items()}, n_episodes=int(e),
           cond_counts=dict(r10=int(M.c_r10.sum()), r20=int(M.c_r20.sum()), gap60=int(M.c_gap60.sum())),
           event_months={g: M.loc[G[g], 'sig_date'].tolist() for g in ('MAIN_all', 'FIRST', 'STRICT', 'PRE')},
           wide_only=M.loc[M.WIDE & ~M.MAIN, 'sig_date'].tolist())
q2c = ['n_fin', 'n_ma', 'ma_frac', 'breadth', 'n_tgt', 'held', 'sold', 'sold_frac', 'bought', 'npos_after', 'sf_sig', 'sf_exec', 'sf_month_avg']
out['Q2'] = {g: dict({c: dist(M.loc[m, c]) for c in q2c}, n_tgt_zero=int((M.loc[m, 'n_tgt'] == 0).sum()), n_tgt_le2=int((M.loc[m, 'n_tgt'] <= 2).sum()),
                     sold_all=int(((M.loc[m, 'held'] > 0) & (M.loc[m, 'sold'] == M.loc[m, 'held'])).sum()), n=int(m.sum())) for g, m in G.items()}
q3c = [f'{a}{m}' for a in ('acct_', 'e50_', 'acct_x') for m in (1, 3, 6, 12)] + ['acct_fall12', 'e50_fall12'] + [f'list_{p}{h}' for p in ('', 'b', 'x') for h in (1, 3, 6, 12)]
out['Q3'] = {g: {c: dist(M.loc[m, c]) for c in q3c} for g, m in G.items()}
out['Q3_ruler_FIRST_vs_normal'] = {c: dict(first_median=out['Q3']['FIRST'][c].get('median'), band=L.rand_median_band(M.loc[~M.MAIN, c], int(M.loc[M.FIRST, c].notna().sum()), rng))
                                   for c in ('acct_12', 'acct_x12', 'acct_3', 'acct_x3', 'e50_12', 'list_x1', 'list_x3')}
# Q4
mg['i'] = mg['sig_t'].map({k: i for i, k in enumerate(ks)}); assert mg['i'].notna().all() and (mg['ent_t'] == mg['sig_t'] + 1).all()
mg['i'] = mg['i'].astype(int)
for g in ('MAIN', 'FIRST', 'STRICT', 'WIDE', 'PRE'):
    mg[g] = mg['i'].map(M[g])
mg['era'] = np.where(mg['ent_date'] < '2016-01-01', '2008-2015', '2016-2026')


def tb(x):
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), n_months=int(x['i'].nunique()), win=float((x.ret > 0).mean()), mean=float(x.ret.mean()), median=float(x.ret.median()), best=float(x.ret.max()), worst=float(x.ret.min()),
                ge50=float((x.ret >= 0.5).mean()), n_ge50=int((x.ret >= 0.5).sum()), top5=float(x.top5.mean()), n_top5=int(x.top5.sum()), le_m15=float((x.ret <= -0.15).mean()),
                ex_mean=float(x.ex.mean()), ex_median=float(x.ex.median()), ex_win=float((x.ex > 0).mean()), hold_median=float(x.hold_days.median()), pnl_sum=float(x.pnl.sum()))


out['Q4'] = dict(FIRST=tb(mg[mg.FIRST]), MAIN_all=tb(mg[mg.MAIN]), STRICT=tb(mg[mg.STRICT]), PRE=tb(mg[mg.PRE]), normal=tb(mg[~mg.MAIN]), all=tb(mg),
                 by_era={er: dict(MAIN_all=tb(mg[mg.MAIN & (mg.era == er)]), normal=tb(mg[~mg.MAIN & (mg.era == er)])) for er in ('2008-2015', '2016-2026')},
                 per_event_month=[dict(sig_date=M.sig_date[i], **tb(mg[mg.i == i])) for i in np.flatnonzero(M.MAIN)])
# Q5
q5 = {}
for odd in ('all', 'hist'):
    q5[odd] = {g: {c: dist(M.loc[m, f'{c}_{odd}']) for c in ('n1', 'n3', 'n6', 'n12', 'nx1', 'nx3', 'nx6', 'nx12', 'ndeep12', 'nmdd12', 'nsf_m1', 'nbuy1')} for g, m in G.items()}
    q5[odd]['loss12_frac'] = {g: float((M.loc[m, f'n12_{odd}'].dropna() < 0).mean()) if M.loc[m, f'n12_{odd}'].notna().any() else None for g, m in G.items()}
    q5[odd]['ruler'] = {c: dict(first_median=q5[odd]['FIRST'][c].get('median'), pre_median=q5[odd]['PRE'][c].get('median'),
                                band=L.rand_median_band(M[f'{c}_{odd}'], int(M.loc[M.FIRST, f'{c}_{odd}'].notna().sum()), rng)) for c in ('n12', 'nx12', 'ndeep12')}
    d = (M.loc[M.FIRST, f'n12_{odd}'].to_numpy() - M.loc[np.flatnonzero(M.FIRST) - 1, f'n12_{odd}'].to_numpy()) if not M.FIRST[0] else None
    # 同一段：急跌前一個月起跑的人到「FIRST 起跑後 12 個月那一天」的報酬不在此算；這裡只比各自起跑後 12 個月
    q5[odd]['first_minus_pre_12m'] = dist(d) if d is not None else None
out['Q5'] = q5
# Q6
bk = dict(tgt0=M.n_tgt == 0, tgt1_2=M.n_tgt.between(1, 2), tgt3_5=M.n_tgt.between(3, 5), tgt6_10=M.n_tgt >= 6, ma_low10=M.ma_frac <= M.ma_frac.quantile(0.10),
          sold_all=(M.held > 0) & (M.sold == M.held))
bk.update({f'MAIN_and_{a}': M.MAIN & bk[a] for a in ('tgt0', 'tgt1_2', 'ma_low10', 'sold_all')}); bk['MAIN_tgt_le2'] = M.MAIN & (M.n_tgt <= 2); bk['tgt_le2'] = M.n_tgt <= 2
out['Q6'] = {b: dict(n=int(m.sum()), months=M.loc[m, 'sig_date'].tolist() if m.sum() <= 40 else None, ma_frac=dist(M.loc[m, 'ma_frac']), sf_month_avg=dist(M.loc[m, 'sf_month_avg']),
                     new12=dist(M.loc[m, 'n12_all']), new12_x=dist(M.loc[m, 'nx12_all']), new3=dist(M.loc[m, 'n3_all']), new3_x=dist(M.loc[m, 'nx3_all']), new_deep12=dist(M.loc[m, 'ndeep12_all']),
                     new_sf_m1=dist(M.loc[m, 'nsf_m1_all']), e50_12=dist(M.loc[m, 'e50_12']), acct_12=dist(M.loc[m, 'acct_12']), acct_x12=dist(M.loc[m, 'acct_x12'])) for b, m in bk.items()}
out['Q6']['ma_frac_p10_threshold'] = float(M.ma_frac.quantile(0.10))
# 0050 日資料類比
ar = []
r7 = np.full(T, np.nan); r7[7:] = E[7:] / E[:-7] - 1; r1 = np.full(T, np.nan); r1[1:] = E[1:] / E[:-1] - 1
for name, hit in (('r7_le_18.4', r7 <= -0.184), ('r1_le_6.4', r1 <= -0.064), ('r7_le_15（事後加，非登記）', r7 <= -0.15)):
    idx = np.flatnonzero(hit); last = -999; epi = 0
    for t in idx:
        new = t - last > 20
        epi += int(new); last = t
        if new:
            rr = dict(kind=name, episode=epi, first_day=str(IDX[t].date()), r7=r7[t], r1=r1[t], n_days=None)
            for m in (1, 3, 6, 12):
                h = hpos(IDX, t, m); rr[f'e50_{m}'] = E[h] / E[t] - 1 if h is not None else np.nan
            h12 = hpos(IDX, t, 12); rr['e50_fall12'] = E[t:h12 + 1].min() / E[t] - 1 if h12 is not None else np.nan
            ar.append(rr)
    out.setdefault('daily_analog', {})[name] = dict(n_days=int(len(idx)), n_episodes=epi, days=[str(IDX[t].date()) for t in idx][:60])
A = pd.DataFrame(ar)
out['daily_analog']['panel_range'] = [str(IDX[0].date()), str(IDX[-1].date())]
out['runtime_s'] = round(time.time() - t0c, 1); out['maxrss_mb'] = rss()
(HERE / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
M.round(6).to_csv(HERE / 'signal_months.csv', index=False)
M[M.WIDE | M.PRE].round(6).to_csv(HERE / 'events.csv', index=False)
mg[mg.WIDE | mg.PRE].drop(columns=['j', 'ret82']).round(6).to_csv(HERE / 'first_batch_trades.csv', index=False)
A.round(6).to_csv(HERE / 'daily_analog.csv', index=False)
print('done', out['runtime_s'], 's', out['maxrss_mb'], 'MB', out['counts'])

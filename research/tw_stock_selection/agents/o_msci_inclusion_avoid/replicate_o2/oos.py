import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import mainlib as M
V, K = M.V, M.K
PL = M.plans(); t0, t1, sd = PL['t0'], PL['t1'], PL['sd']
ew = PL['ew']
ev = pd.read_csv(M.HERE / 'events_rep.csv', dtype={'code': str}, parse_dates=['E'])
ev = ev[ev.family == 'MSCI']
ev = ev[ev.code.isin(M.COLS)]
CI = {c: j for j, c in enumerate(M.COLS)}
def sellmap(classes):
    d = {}
    for _, r in ev[ev.cls.isin(classes)].iterrows(): d.setdefault(int(r.Ei), set()).add(CI[r.code])
    return d
SELL_STD = sellmap(['STD'])
EXCL = sellmap(['STD', 'SC'])   # 本次納入股（遞補排除）
# ---------- 受影響持股計數 ----------
def hits(plan, classes):
    ex = [p[0] for p in plan]; out = []
    for i, (e, pk, _, d) in enumerate(plan):
        e2 = ex[i + 1] if i + 1 < len(ex) else t1 + 1
        for _, r in ev[ev.cls.isin(classes)].iterrows():
            if e <= r.Ei <= e2 - 1 and CI[r.code] in set(pk):
                out.append(dict(sig=str(d.date()), E=str(r.E.date()), code=r.code, name=r['name'], cls=r.cls))
    return pd.DataFrame(out)
res = {'hits': {}}
for n in (10, 30):
    h = hits(PL[n], ['STD', 'SC'])
    h['per'] = np.where(pd.to_datetime(h.E) <= pd.Timestamp('2018-12-31'), 'select', 'oos')
    # 只計主策略評價期（訊號 2010-11 起才算挑選期）
    res['hits'][n] = dict(all=len(h), by_per=h.per.value_counts().to_dict(), std=int((h.cls == 'STD').sum()),
                          std_by_per=h[h.cls == 'STD'].per.value_counts().to_dict(),
                          pre2010_11=int((pd.to_datetime(h.sig) < pd.Timestamp('2010-11-01')).sum()))
    print(n, res['hits'][n]); print(h.to_string())
# ---------- 模擬 ----------
SEG = {'select': ('2010-11-01', '2018-12-31'), 'valid': ('2019-01-01', '2023-12-31'), 'hold': ('2024-01-01', '2026-10-12'),
       'judge': ('2019-01-01', '2026-10-12'), 'cross': ('2006-04-01', '2026-10-12')}
def wret(nav):
    D = pd.DatetimeIndex(sd); x = nav.reindex(D); return (x.shift(-1) / x - 1).iloc[:-1]
def paired(navB, navO, log, a, b):
    rB, rO = wret(navB), wret(navO); df = rB - rO
    df = df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b))]
    st = set()
    for L_ in log:
        if L_['kind'] != 'sell': continue
        ts = M.IDX[L_['t']]
        D = pd.DatetimeIndex(sd)
        k = D.searchsorted(ts, side='left') - 1      # 窗 (D[k], D[k+1]]
        if 0 <= k < len(D) - 1 and D[k] in df.index: st.add(D[k])
    aff = df[df.index.isin(sorted(st))]
    o = dict(n_aff=int(len(aff)), mean_aff=round(float(aff.mean()) * 100, 2) if len(aff) else None, t_aff=round(K.tstat(aff.values), 2) if len(aff) > 2 else None,
             sd_aff=round(float(aff.std(ddof=1)) * 100, 2) if len(aff) > 1 else None,
             t_all=round(K.tstat(df.values), 2), aff_list={str(k.date()): round(float(v) * 100, 2) for k, v in aff.items()})
    if len(aff) > 1:
        o['drop_best'] = round(float(aff.drop(aff.idxmax()).mean()) * 100, 2)
    return o, aff
def run(n, kind, mode, plan=None, sells=SELL_STD):
    plan = PL[n] if plan is None else plan
    return M.simulate(plan, kind, t0, t1, sells=sells, mode=mode, cand=PL['cand'], excl=EXCL)
out = {}
for n in (10, 30):
    for kind in ('normal', 'stress'):
        navO, _ = run(n, kind, 'none')
        navB, log = run(n, kind, '0050')
        r = {'sales': [dict(t=str(M.IDX[x['t']].date()), code=M.COLS[x['j']], E=str(M.IDX[x['tE']].date())) for x in log if x['kind'] == 'sell']}
        for s, (a, b) in SEG.items():
            sb = V.seg_stats(navB, ew, a, b, sd); so = V.seg_stats(navO, ew, a, b, sd)
            p, aff = paired(navB, navO, log, a, b)
            r[s] = dict(B2=round(sb['CAGR'], 2), orig=round(so['CAGR'], 2), dCAGR=round(sb['CAGR'] - so['CAGR'], 2), ddB=round(sb['maxDD'], 2), ddO=round(so['maxDD'], 2),
                        t0050_B=round(sb['t_0050'], 2), t0050_O=round(so['t_0050'], 2), E50=round(sb['E50_CAGR'], 2), EW=round(sb['EW_CAGR'], 2), **p)
        out[f'n{n}|{kind}'] = r
        print(n, kind, json.dumps({k: v for k, v in r.items() if k != 'sales'}, ensure_ascii=False))
        print('  sales', r['sales'])
        if n == 10 and kind == 'normal':
            navB10, log10, navO10 = navB, log, navO
res['B2'] = out
# 檢定力：受影響月配對差 SD 與 MDE（2.8×sd/√n_oos）
pj = out['n10|normal']
allaff = list(pj['cross']['aff_list'].values())
sd_all = float(np.std(allaff, ddof=1))
res['power'] = dict(sd_all_aff=round(sd_all, 2), n_oos=pj['judge']['n_aff'], MDE=round(2.8 * sd_all / np.sqrt(pj['judge']['n_aff']), 2),
                    sd_oos=pj['judge']['sd_aff'], MDE_oos_sd=round(2.8 * pj['judge']['sd_aff'] / np.sqrt(pj['judge']['n_aff']), 2))
print('power', res['power'])
# ---------- P6：跨循環單股貢獻 ----------
con, per = V.stock_contrib(PL[10], t1)
top = sorted(con.items(), key=lambda x: -x[1])[:20]
stdcodes = {M.COLS[x['j']] for x in log10 if x['kind'] == 'sell'}
res['P6_top20'] = [(i + 1, c, V.SL['name'].get(c, ''), round(v * 100, 1), c in stdcodes) for i, (c, v) in enumerate(top)]
print('top20', res['P6_top20'])
# 每筆賣出：賣出日收盤 → 下一成交日開盤（原版會拿到的）vs 0050
rows = []
for x in log10:
    if x['kind'] != 'sell': continue
    j, t, last = x['j'], x['t'], x['last']
    e2 = last + 1
    stock = np.prod(M.G[t + 1:e2 + 1, j]) * (M.On[e2, j] / M.Cn[e2, j] if np.isfinite(M.On[e2, j]) else 1) - 1 if e2 <= t1 else np.prod(M.G[t + 1:t1 + 1, j]) - 1
    e50 = M.E50[min(last, t1)] / M.E50[t] - 1
    stock_last = np.prod(M.G[t + 1:min(last, t1) + 1, j]) - 1
    rows.append(dict(code=M.COLS[j], sell=str(M.IDX[t].date()), stock=round(stock * 100, 1), stock_last=round(stock_last * 100, 1), e50=round(e50 * 100, 1), diff=round((e50 - stock) * 100, 1), diff_last=round((e50 - stock_last) * 100, 1)))
P6 = pd.DataFrame(rows); print(P6.to_string())
res['P6_sales'] = rows
res['P6_sum'] = dict(save=round(P6['diff'][P6['diff'] > 0].sum(), 1), loss=round(P6['diff'][P6['diff'] < 0].sum(), 1),
                     stock_neg=round(-P6['stock'][P6['stock'] < 0].sum(), 1), stock_pos=round(-P6['stock'][P6['stock'] > 0].sum(), 1),
                     last_neg=round(-P6['stock_last'][P6['stock_last'] < 0].sum(), 1), last_pos=round(-P6['stock_last'][P6['stock_last'] > 0].sum(), 1),
                     dl_save=round(P6['diff_last'][P6['diff_last'] > 0].sum(), 1), dl_loss=round(P6['diff_last'][P6['diff_last'] < 0].sum(), 1))
print(res['P6_sum'])
json.dump(res, open(M.HERE / 'oos_rep.json', 'w'), indent=1, ensure_ascii=False, default=str)
pd.DataFrame({'B2': navB10, 'orig': navO10}).to_parquet(M.HERE / 'navs_rep.parquet')

#!/usr/bin/env python3
"""樣本外（凍結後；斷言 PREREG 存在）：P1 價差（主規格＋次要規格）、獨立 IMOM 腿條件、P2 描述、當沖子樣本、日均描述。
結合版已在凍結前判定不可用，本程式不計算任何結合版的驗證期數字。輸出 results_oos.json。"""
import sys, json, time, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as L
PR, SH, PT = L.assert_prereg()
K, V = L.K, L.V
IDX, COLS, N = L.IDX, L.COLS, L.N
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)

sp = L.Split()
S52, V60 = V.rolling_signals('gapfix')
OUT = {'PREREG_sha256': SH, 'PREREG_time': PT, 'run_time': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}

def periods(a, b):
    reb = [d for d in L.REB_ALL if a <= d <= b]
    ti = [IDX.get_loc(d) for d in reb]
    allreb = L.REB_ALL
    out = []
    for d, t in zip(reb, ti):
        nx = [x for x in allreb if x > d]
        if not nx: continue
        e2 = IDX.get_loc(nx[0]) + 1
        if e2 >= len(IDX): continue
        out.append((d, t, t + 1, e2))
    return out
SELP = periods(L.SEL_FIRST, pd.Timestamp('2019-03-31'))
VALP = periods(L.VAL_FIRST, pd.Timestamp('2026-09-30'))
log('periods', len(SELP), len(VALP), VALP[-1][0].date(), IDX[VALP[-1][3]].date())

def build(PER):
    cache = []
    for (d, t, e, e2) in PER:
        el, rk = L.base_pool(t, S52, V60)
        r = L.hold_returns(e, e2)
        fI, fO = L.split_future(e, e2, sp)
        w = {k: sp.window(t, *k) for k in ((126, 0), (126, 21), (252, 0), (252, 21))}
        cache.append(dict(d=d, t=t, e=e, el=el, rk=rk, r=r, fI=fI, fO=fO, w=w, s52=S52[t]))
    return cache
CS, CV = build(SELP), build(VALP)
log('cache')

def qsort(x, el, q=5):
    g = np.full(N, -1); m = el & np.isfinite(x)
    if m.sum() < 5 * q: return g
    rk = pd.Series(x[m]).rank(method='first').to_numpy()
    g[np.where(m)[0]] = np.minimum(((rk - 1) / m.sum() * q).astype(int), q - 1)
    return g

def spread(cache, getx, f3key, excl=True):
    rows = []; prev = None
    for c in cache:
        if excl and c['w'][f3key][3] > 0.20: continue
        el = c['el'] & np.isfinite(c['r'])
        g = qsort(getx(c), el)
        if (g >= 0).sum() < 50: continue
        rq = [np.nanmean(c['r'][g == k]) for k in range(5)]
        top = set(np.where(g == 4)[0]); bot = set(np.where(g == 0)[0])
        to = None if prev is None else 0.5 * (len(top - prev[0]) / max(len(top), 1) + len(bot - prev[1]) / max(len(bot), 1))
        prev = (top, bot)
        rows.append(dict(d=c['d'], hml=rq[4] - rq[0], q=rq, to=to, fI=np.nanmean(c['fI'][g == 4]) - np.nanmean(c['fI'][g == 0]),
                         fO=np.nanmean(c['fO'][g == 4]) - np.nanmean(c['fO'][g == 0])))
    return pd.DataFrame(rows).set_index('d')

def summ(df, cuts=()):
    h = df['hml'].to_numpy(); to = df['to'].dropna().mean()
    o = {'n': int(len(h)), 'hml_pct': float(np.nanmean(h) * 100), 't': L.tstat(h), 't_nw6': L.nw_t(h),
         'quintiles_pct': [round(float(v), 3) for v in np.nanmean(np.vstack(df['q'].to_numpy()), axis=0) * 100],
         'turnover': float(to), 'net_hml_pct_approx': float(np.nanmean(h - 2 * to * 0.005) * 100),
         'future_intraday_hml_logpct': float(df['fI'].mean() * 100), 'future_overnight_hml_logpct': float(df['fO'].mean() * 100)}
    for nm, a, b in cuts:
        m = (df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b))
        o[nm] = {'n': int(m.sum()), 'hml_pct': float(np.nanmean(h[m]) * 100) if m.any() else None, 't': L.tstat(h[m])}
    o['yearly_hml_pct'] = {int(y): round(float(g.mean() * 100), 3) for y, g in df['hml'].groupby(df.index.year)}
    return o
VCUTS = (('V_2019_2023', '2019-05-01', '2023-12-31'), ('V_2024_2026', '2024-01-01', '2026-09-30'))
SCUTS = (('S_2005_2011', '2005-01-01', '2011-12-31'), ('S_2012_2019', '2012-01-01', '2019-04-30'))

P1 = {}
series = {}
for nm, k in (('primary_L252_S21', (252, 21)), ('secondary_L126_S0', (126, 0))):
    for side, gx in (('IMOM', lambda c, k=k: c['w'][k][0]), ('OMOM', lambda c, k=k: c['w'][k][1])):
        dS = spread(CS, gx, k)
        dV = spread(CV, gx, k, excl=True); dVk = spread(CV, gx, k, excl=False)
        P1[f'{nm}|{side}'] = {'selection': summ(dS, SCUTS), 'validation_main': summ(dV, VCUTS), 'validation_keep_src3': summ(dVk, VCUTS),
                              'excluded_signal_months': [str(x.date()) for x in sorted(set(dVk.index) - set(dV.index))]}
        series[(nm, side)] = (dS['hml'], dV['hml'], dVk['hml'])
        log(nm, side, 'S', round(P1[f'{nm}|{side}']['selection']['hml_pct'], 3), round(P1[f'{nm}|{side}']['selection']['t'], 2),
            'V', round(P1[f'{nm}|{side}']['validation_main']['hml_pct'], 3), round(P1[f'{nm}|{side}']['validation_main']['t'], 2),
            'Vk', round(P1[f'{nm}|{side}']['validation_keep_src3']['hml_pct'], 3), round(P1[f'{nm}|{side}']['validation_keep_src3']['t'], 2))
# 其他 10 個規格：只當描述（凍結時沒有預登記為判定）
desc12 = {}
for Lw, S in ((126, 0), (126, 21), (252, 0), (252, 21)):
    for f in ('cum', 'share', 'diff'):
        dV = spread(CV, lambda c: L.form_signal(c['w'][(Lw, S)][0], c['w'][(Lw, S)][1], f), (Lw, S))
        desc12[f'L{Lw}_S{S}_{f}'] = {'hml_pct': float(dV['hml'].mean() * 100), 't': L.tstat(dV['hml']), 'n': int(len(dV))}
OUT['P1'] = P1; OUT['P1_other_specs_validation_descriptive'] = desc12

def p1_pass(o):
    return (o['IMOM']['hml_pct'] >= 0.40) and (o['IMOM']['t'] >= 1.0) and (o['OMOM']['hml_pct'] < 0)
verd = {}
for nm in ('primary_L252_S21', 'secondary_L126_S0'):
    s = {'IMOM': P1[f'{nm}|IMOM']['selection'], 'OMOM': P1[f'{nm}|OMOM']['selection']}
    v = {'IMOM': P1[f'{nm}|IMOM']['validation_main'], 'OMOM': P1[f'{nm}|OMOM']['validation_main']}
    vk = {'IMOM': P1[f'{nm}|IMOM']['validation_keep_src3'], 'OMOM': P1[f'{nm}|OMOM']['validation_keep_src3']}
    verd[nm] = {'selection_pass': p1_pass(s), 'validation_pass': p1_pass(v), 'validation_keep_src3_pass': p1_pass(vk)}
OUT['P1_verdict'] = verd
log('P1 verdict', verd)

# ---------------- 主策略淨值（全期）與相關 ----------------
YL, Y5, _ = V.load_revenue()
SM = V.V2('gapfix', YL, Y5, exfin=True)
t0, t1, sd = SM.bounds()
plm = SM.plan(10, 'official')
bb, ss = K.rates(0.001)
navm, _ = K.simulate([(p[0], p[1]) for p in plm], bb, ss, t0, t1)
ewm, _ = K.simulate([(p[0], p[2]) for p in plm], 0, 0, t0, t1)
rm = L.sig_window_returns(navm, sd); r50 = L.sig_window_returns(K.E50.reindex(navm.index), sd).reindex(rm.index)
exm = rm - r50
cors = {}
for (nm, side), (sS, sV, sVk) in series.items():
    if side != 'IMOM': continue
    cors[nm] = {'selection': float(pd.concat([sS, exm], axis=1).dropna().corr().iloc[0, 1]),
                'validation_main': float(pd.concat([sV, exm], axis=1).dropna().corr().iloc[0, 1]),
                'validation_keep_src3': float(pd.concat([sVk, exm], axis=1).dropna().corr().iloc[0, 1])}
OUT['corr_IMOM_spread_vs_main_excess_0050'] = cors
# 主策略本身的對照數字（驗證期，與 main_v2 一致性核對）
OUT['main_check'] = {k: L.r4(V.seg_stats(navm, ewm, a, b, sd, ad=ad)) for k, (a, b, ad) in V.SEGS.items()}
log('main check ALL', OUT['main_check']['ALL']['CAGR'], OUT['main_check']['ALL']['t_0050'])

leg = {}
for nm in ('primary_L252_S21', 'secondary_L126_S0'):
    ok = verd[nm]['selection_pass'] and verd[nm]['validation_pass'] and all(v < 0.5 for kk, v in cors[nm].items() if kk != 'validation_keep_src3')
    leg[nm] = {'eligible': bool(ok)}
    if ok:
        k = (252, 21) if nm.startswith('primary') else (126, 0)
        plq = []
        for (e, pk, pool, d) in plm:
            tt = IDX.get_loc(d)
            el, _ = L.base_pool(tt, S52, V60)
            CI, CO, okw, f3 = sp.window(tt, *k)
            g = qsort(CI, el)
            plq.append((e, np.where(g == 4)[0]))
        navq, _ = K.simulate(plq, bb, ss, t0, t1)
        leg[nm]['stats'] = {kk: L.r4(V.seg_stats(navq, ewm, a, b, sd, ad=ad)) for kk, (a, b, ad) in V.SEGS.items()}
OUT['standalone_leg'] = leg
log('leg', leg)

# ---------------- P2 描述：主策略前 30 名，252 日占比 ----------------
per_by_d = {c['d']: c for c in CS + CV}
def p2(getx, ds, sub=None, excl=True, f3key=(252, 0)):
    rows = []
    for d in ds:
        if d not in per_by_d: continue
        c = per_by_d[d]
        if excl and c['w'][f3key][3] > 0.20: continue
        cs = SM.cross_section(d)
        top = SM.order(cs, 'official')[:int(cs['el'].sum())][:30]
        ids = np.array([j for j in top if np.isfinite(c['r'][j]) and (sub is None or sub(c, j))])
        if len(ids) < 6: continue
        x = getx(c)[ids]; ok = np.isfinite(x)
        if ok.sum() < 6: continue
        ids, x = ids[ok], x[ok]; med = np.median(x)
        hi, lo = ids[x > med], ids[x <= med]
        if len(hi) == 0 or len(lo) == 0: continue
        rows.append((d, np.nanmean(c['r'][hi]) - np.nanmean(c['r'][lo])))
    if not rows: return None
    s = pd.Series(dict(rows))
    return {'n': int(len(s)), 'diff_pct': float(s.mean() * 100), 't': L.tstat(s.to_numpy())}
gx = lambda c: L.form_signal(c['w'][(252, 0)][0], c['w'][(252, 0)][1], 'share')
sdS = [d for d in SM.sig_days if d <= L.SEL_LAST_SIG]; sdV = [d for d in SM.sig_days if d >= L.VAL_FIRST]
OUT['P2_L252_S0_share'] = {
    'selection': {'top30': p2(gx, sdS), 'liq_1_100': p2(gx, sdS, lambda c, j: c['rk'][j] <= 100), 'liq_101_300': p2(gx, sdS, lambda c, j: c['rk'][j] > 100)},
    'validation_main': {'top30': p2(gx, sdV), 'liq_1_100': p2(gx, sdV, lambda c, j: c['rk'][j] <= 100), 'liq_101_300': p2(gx, sdV, lambda c, j: c['rk'][j] > 100)},
    'validation_keep_src3': {'top30': p2(gx, sdV, excl=False)}}
log('P2', OUT['P2_L252_S0_share'])

# ---------------- 當沖子樣本（2021 起）----------------
DT = pd.read_csv('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/scr_extra/daytrade_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
DTw = DT.pivot_table(index='date', columns='code', values='dt_ratio').reindex(columns=COLS)
dtr = DTw.rolling(20, min_periods=10).mean()
def dt_at(d):
    s = dtr[dtr.index <= d]
    return s.iloc[-1].to_numpy() if len(s) else np.full(N, np.nan)
sd21 = [d for d in sdV if d >= pd.Timestamp('2021-01-01')]
DTAT = {d: dt_at(d) for d in sd21}
def dt_split(hi):
    def sub(c, j):
        d = c['d']; x = DTAT[d]
        cs = SM.cross_section(d); top = SM.order(cs, 'official')[:30]
        m = np.nanmedian(x[top])
        return np.isfinite(x[j]) and ((x[j] > m) if hi else (x[j] <= m))
    return sub
OUT['P2_daytrade_2021'] = {'main': {'high_dt': p2(gx, sd21, dt_split(True)), 'low_dt': p2(gx, sd21, dt_split(False)), 'all': p2(gx, sd21)},
                           'keep_src3': {'high_dt': p2(gx, sd21, dt_split(True), excl=False), 'low_dt': p2(gx, sd21, dt_split(False), excl=False), 'all': p2(gx, sd21, excl=False)}}
log('dt', OUT['P2_daytrade_2021'])

# ---------------- 描述：日均盤中／隔夜（驗證期年份）----------------
desc = {}
liq_rank = np.full((len(IDX), N), np.nan)
for (d, t, e, e2) in VALP:
    _, rk = L.base_pool(t, S52, V60); liq_rank[e:e2] = rk
grp = {'top100': liq_rank <= 100, 'r101_300': (liq_rank > 100) & (liq_rank <= 300)}
for y in range(2019, 2027):
    m = (IDX.year == y) & (IDX >= L.VAL_FIRST)
    o = {}
    for gn, G in grp.items():
        a = sp.ID[m][G[m]]; b = sp.ON[m][G[m]]
        o[gn] = {'ID_bp': float(np.nanmean(a) * 1e4), 'ON_bp': float(np.nanmean(b) * 1e4)}
    srcm = K.SRC.reindex(index=IDX, columns=COLS).to_numpy()[m]
    o['src3_share'] = float(np.nanmean(srcm == 3))
    desc[y] = o
OUT['desc_daily_mean_validation_years'] = desc

# ---------------- 主策略驗證期入選股的占比分布（描述）----------------
recs = []
for (e, pk, pool, d) in plm:
    if d < L.VAL_FIRST or d not in per_by_d: continue
    c = per_by_d[d]; CI, CO, _, f3 = c['w'][(252, 0)]
    for j in pk:
        tot = CI[j] + CO[j]
        recs.append(dict(r=c['r'][j], CI=CI[j], CO=CO[j], share=CI[j] / tot if tot > 0 else np.nan, f3=f3))
R = pd.DataFrame(recs); R = R[R.f3 <= 0.2]
q90, q10 = R['r'].quantile(.9), R['r'].quantile(.1)
OUT['main_picks_share_validation'] = {'n': int(len(R)), 'share_median': float(R['share'].median()), 'frac_CI_negative': float((R['CI'] < 0).mean()),
                                      'winners_share_median': float(R[R.r >= q90]['share'].median()), 'losers_share_median': float(R[R.r <= q10]['share'].median()),
                                      'CI_median_logpct': float(R['CI'].median() * 100), 'CO_median_logpct': float(R['CO'].median() * 100)}
(L.HERE / 'results_oos.json').write_text(L.jdump(OUT), encoding='utf-8')
log('done')

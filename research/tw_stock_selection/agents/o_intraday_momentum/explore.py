#!/usr/bin/env python3
"""探索（只用挑選期，lock_selection 強制）：描述統計、P1 五分位價差、P2 主策略前 30 名的盤中占比分組、機制檢查。
輸出 explore.json。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as L
L.lock_selection()
K, V = L.K, L.V
IDX, COLS, N = L.IDX, L.COLS, L.N
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)

sp = L.Split()
S52, V60 = V.rolling_signals('gapfix')
assert np.all(~np.isfinite(sp.ID[L.CUT + 1:])), '鎖失效'

REB = [d for d in L.REB_ALL if L.SEL_FIRST <= d <= L.SEL_LAST_SIG]
TI = [IDX.get_loc(d) for d in REB]
# 持有期：訊號日 t → 成交 e=t+1 開盤 → 下一訊號成交日 e2 開盤。最後一期（2019-04 訊號）e2 落在驗證期 → 不用
PER = [(REB[i], TI[i], TI[i] + 1, TI[i + 1] + 1) for i in range(len(REB) - 1) if TI[i + 1] + 1 <= L.CUT]
log('periods', len(PER), PER[0][0].date(), PER[-1][0].date())
OUT = {'note': '挑選期 2005-01～2019-03 訊號（持有期開盤→開盤；2019-04 訊號期跨入驗證期，不用）。所有報酬為毛報酬（扣成本前），另附近似淨額。'}

# ---------------- (2) 描述：全池日均盤中／隔夜，分年、分流動性 ----------------
desc = {}
liq_rank = np.full((len(IDX), N), np.nan)
pool_mask = np.zeros((len(IDX), N), bool)
for (d, t, e, e2) in PER:
    el, rk = L.base_pool(t, S52, V60)
    # 下一期每一天都用這個月的分組
    liq_rank[e:e2] = rk; pool_mask[e:e2] = el
grp = {'top100': (liq_rank <= 100), 'r101_300': (liq_rank > 100) & (liq_rank <= 300), 'rest': (liq_rank > 300)}
yrs = IDX.year
for y in range(2005, 2020):
    m = (yrs == y) & (np.arange(len(IDX)) <= L.CUT)
    o = {}
    for gn, G in grp.items():
        gi = G[m]
        a = sp.ID[m][gi]; b = sp.ON[m][gi]
        o[gn] = {'ID_bp': float(np.nanmean(a) * 1e4), 'ON_bp': float(np.nanmean(b) * 1e4), 'n': int(np.isfinite(a).sum())}
    a = sp.ID[m]; b = sp.ON[m]
    o['all'] = {'ID_bp': float(np.nanmean(a) * 1e4), 'ON_bp': float(np.nanmean(b) * 1e4)}
    desc[y] = o
OUT['desc_daily_mean_by_year_liq'] = desc
log('desc done')

# ---------------- 訊號規格 ----------------
SPECS = [(Lw, S, f) for Lw in (126, 252) for S in (0, 21) for f in ('cum', 'share', 'diff')]
def spec_name(s): return f'L{s[0]}_S{s[1]}_{s[2]}'

# 預先算每期：池、報酬、未來盤中／隔夜
cache = []
for (d, t, e, e2) in PER:
    el, rk = L.base_pool(t, S52, V60)
    r = L.hold_returns(e, e2)
    fI, fO = L.split_future(e, e2, sp)
    w = {}
    for Lw in (126, 252):
        for S in (0, 21):
            w[(Lw, S)] = sp.window(t, Lw, S)
    cache.append(dict(d=d, t=t, el=el, rk=rk, r=r, fI=fI, fO=fO, w=w, s52=S52[t]))
log('cache done')

def qsort(x, el, q=5):
    """回傳每檔的分位（0..q-1），不在池或缺值為 -1。"""
    g = np.full(N, -1)
    m = el & np.isfinite(x)
    if m.sum() < 5 * q: return g
    rk = pd.Series(x[m]).rank(method='first').to_numpy()
    g[np.where(m)[0]] = np.minimum(((rk - 1) / m.sum() * q).astype(int), q - 1)
    return g

def spread_series(getx, q=5, cond=None):
    rows = []; prev = None
    for c in cache:
        x = getx(c)
        el = c['el'] & np.isfinite(c['r'])
        if cond is not None:
            # 條件排序：在 s52 三分位內各自分五分位
            g = np.full(N, -1)
            tg = qsort(c['s52'], el, 3)
            for k in range(3):
                gg = qsort(x, el & (tg == k), q)
                g = np.where(gg >= 0, gg, g)
        else:
            g = qsort(x, el, q)
        if (g >= 0).sum() < 50: continue
        rq = [np.nanmean(c['r'][g == k]) for k in range(q)]
        fi = [np.nanmean(c['fI'][g == k]) for k in range(q)]
        fo = [np.nanmean(c['fO'][g == k]) for k in range(q)]
        top = set(np.where(g == q - 1)[0]); bot = set(np.where(g == 0)[0])
        to = None
        if prev is not None:
            to = 0.5 * (len(top - prev[0]) / max(len(top), 1) + len(bot - prev[1]) / max(len(bot), 1))
        prev = (top, bot)
        rows.append(dict(d=c['d'], hml=rq[-1] - rq[0], q=rq, fI=fi[-1] - fi[0], fO=fo[-1] - fo[0], to=to))
    df = pd.DataFrame(rows).set_index('d')
    return df

def summ(df):
    h = df['hml'].to_numpy()
    to = df['to'].dropna().mean()
    net = h - 2 * to * 0.005        # 多空兩邊各換掉 to 比例、每換一檔來回 0.5%
    a = df.index < pd.Timestamp('2012-01-01')
    qm = np.nanmean(np.vstack(df['q'].to_numpy()), axis=0) * 100
    return {'n': int(len(h)), 'hml_pct': float(np.nanmean(h) * 100), 't': L.tstat(h), 't_nw6': L.nw_t(h),
            'hml_2005_2011_pct': float(np.nanmean(h[a]) * 100), 't_2005_2011': L.tstat(h[a]),
            'hml_2012_2019_pct': float(np.nanmean(h[~a]) * 100), 't_2012_2019': L.tstat(h[~a]),
            'quintiles_pct': [round(float(v), 3) for v in qm], 'turnover_avg': float(to),
            'net_hml_pct_approx': float(np.nanmean(net) * 100), 't_net': L.tstat(net),
            'future_intraday_hml_logpct': float(np.nanmean(df['fI']) * 100), 'future_overnight_hml_logpct': float(np.nanmean(df['fO']) * 100)}

P1 = {}
for Lw in (126, 252):
    for S in (0, 21):
        k = (Lw, S)
        for f in ('cum', 'share', 'diff'):
            nm = spec_name((Lw, S, f))
            df = spread_series(lambda c: L.form_signal(c['w'][k][0], c['w'][k][1], f))
            dfc = spread_series(lambda c: L.form_signal(c['w'][k][0], c['w'][k][1], f), cond=True)
            P1[nm] = {'uncond': summ(df), 'within_s52_terciles': summ(dfc)}
        # OMOM 與總報酬
        df = spread_series(lambda c: c['w'][k][1]); dfc = spread_series(lambda c: c['w'][k][1], cond=True)
        P1[f'L{Lw}_S{S}_OMOM'] = {'uncond': summ(df), 'within_s52_terciles': summ(dfc)}
        df = spread_series(lambda c: c['w'][k][0] + c['w'][k][1])
        P1[f'L{Lw}_S{S}_TOTAL'] = {'uncond': summ(df)}
        log('P1', Lw, S, {f: round(P1[spec_name((Lw, S, f))]['uncond']['hml_pct'], 3) for f in ('cum', 'share', 'diff')},
            'OMOM', round(P1[f'L{Lw}_S{S}_OMOM']['uncond']['hml_pct'], 3))
# 52 週高點本身（對照）
df = spread_series(lambda c: c['s52']); P1['S52_reference'] = {'uncond': summ(df)}
OUT['P1_quintile_spreads'] = P1

# ---------------- 機制：IMOM 價差的延續（持有 1..9 個月後的第 k 個月）----------------
evt = {}
for f, getx in (('IMOM_L252_S21', lambda c: c['w'][(252, 21)][0]), ('OMOM_L252_S21', lambda c: c['w'][(252, 21)][1])):
    res = {}
    for kk in range(1, 10):
        hs = []
        for i, c in enumerate(cache):
            j = i + kk - 1
            if j >= len(cache): break
            el = c['el']; g = qsort(getx(c), el)
            r = cache[j]['r']
            hs.append(np.nanmean(r[g == 4]) - np.nanmean(r[g == 0]))
        res[kk] = {'hml_pct': float(np.nanmean(hs) * 100), 't': L.tstat(hs)}
    evt[f] = res
OUT['mechanism_event_time'] = evt
log('event-time done')

# ---------------- P2：主策略前 30 名的盤中占比分組 ----------------
YL, Y5, _ = V.load_revenue()
SM = V.V2('gapfix', YL, Y5, exfin=True)
SM.sig_days = [d for d in SM.sig_days if d <= L.SEL_LAST_SIG]
SM.end_day = L.CUT_DAY
P2 = {}
per_by_d = {c['d']: c for c in cache}
main_top = {}
for d in SM.sig_days:
    if d not in per_by_d: continue
    cs = SM.cross_section(d)
    od = SM.order(cs, 'official')[:int(cs['el'].sum())]
    main_top[d] = od[:30]
log('main top30 months', len(main_top))
def p2_series(getx, sub=None, nhalf=None):
    rows = []
    for d, top in main_top.items():
        c = per_by_d[d]
        x = getx(c)
        ids = np.array([j for j in top if np.isfinite(c['r'][j])])
        if sub is not None:
            ids = np.array([j for j in ids if sub(c['rk'][j])])
        if len(ids) < 6: continue
        xv = x[ids]
        ok = np.isfinite(xv)
        if ok.sum() < 6: continue
        ids2 = ids[ok]; xv = xv[ok]
        med = np.median(xv)
        hi = ids2[xv > med]; lo = ids2[xv <= med]
        if len(hi) == 0 or len(lo) == 0: continue
        rows.append(dict(d=d, diff=np.nanmean(c['r'][hi]) - np.nanmean(c['r'][lo]), hi=np.nanmean(c['r'][hi]), lo=np.nanmean(c['r'][lo]),
                         nna=int((~ok).sum())))
    df = pd.DataFrame(rows).set_index('d')
    h = df['diff'].to_numpy(); a = df.index < pd.Timestamp('2012-01-01')
    return {'n': int(len(h)), 'diff_pct': float(np.nanmean(h) * 100), 't': L.tstat(h), 'hi_pct': float(df['hi'].mean() * 100), 'lo_pct': float(df['lo'].mean() * 100),
            'diff_2006_2011_pct': float(np.nanmean(h[a]) * 100), 't_2006_2011': L.tstat(h[a]), 'diff_2012_2019_pct': float(np.nanmean(h[~a]) * 100), 't_2012_2019': L.tstat(h[~a]),
            'avg_nan_in_top30': float(df['nna'].mean())}
for s in SPECS:
    k = (s[0], s[1]); f = s[2]
    gx = lambda c, k=k, f=f: L.form_signal(c['w'][k][0], c['w'][k][1], f)
    P2[spec_name(s)] = {'top30': p2_series(gx), 'liq_1_100': p2_series(gx, sub=lambda r: r <= 100), 'liq_101_300': p2_series(gx, sub=lambda r: r > 100)}
for k in ((126, 0), (126, 21), (252, 0), (252, 21)):
    gx = lambda c, k=k: -c['w'][k][1]
    P2[f'L{k[0]}_S{k[1]}_negOMOM'] = {'top30': p2_series(gx), 'liq_1_100': p2_series(gx, sub=lambda r: r <= 100), 'liq_101_300': p2_series(gx, sub=lambda r: r > 100)}
OUT['P2_main_top30_split'] = P2
log('P2', {k: (round(v['top30']['diff_pct'], 3), round(v['top30']['t'], 2)) for k, v in P2.items()})

# ---------------- 主策略實際 n10：入選前 252 日盤中占比分布、右尾／左尾 ----------------
pl = SM.plan(10, 'official')
recs = []
for (e, pk, pool, d) in pl:
    if d not in per_by_d: continue
    c = per_by_d[d]
    CI, CO, _, _ = c['w'][(252, 0)]
    for j in pk:
        tot = CI[j] + CO[j]
        recs.append(dict(d=d, code=COLS[j], r=c['r'][j], CI=CI[j], CO=CO[j], share=(CI[j] / tot if tot > 0 else np.nan), liq=c['rk'][j]))
R = pd.DataFrame(recs)
sh = R['share'].dropna()
dist = {'n_picks': int(len(R)), 'share_pcts': {str(q): float(sh.quantile(q)) for q in (0.05, .1, .25, .5, .75, .9, .95)},
        'frac_CI_negative': float((R['CI'] < 0).mean()), 'frac_share_gt_1': float((sh > 1).mean()),
        'CI_median_logpct': float(R['CI'].median() * 100), 'CO_median_logpct': float(R['CO'].median() * 100)}
q90, q10 = R['r'].quantile(.9), R['r'].quantile(.1)
win = R[R['r'] >= q90]; los = R[R['r'] <= q10]; mid = R[(R['r'] > q10) & (R['r'] < q90)]
dist['winners_top10pct'] = {'n': int(len(win)), 'share_median': float(win['share'].median()), 'CI_mean_logpct': float(win['CI'].mean() * 100), 'CO_mean_logpct': float(win['CO'].mean() * 100)}
dist['losers_bottom10pct'] = {'n': int(len(los)), 'share_median': float(los['share'].median()), 'CI_mean_logpct': float(los['CI'].mean() * 100), 'CO_mean_logpct': float(los['CO'].mean() * 100)}
dist['middle'] = {'n': int(len(mid)), 'share_median': float(mid['share'].median())}
# 依盤中占比五分位的下月報酬（實際入選股）
R['sq'] = pd.qcut(R['share'].rank(method='first'), 5, labels=False)
dist['next_ret_by_share_quintile_pct'] = {int(k): float(v * 100) for k, v in R.groupby('sq')['r'].mean().items()}
dist['next_ret_by_share_quintile_median_pct'] = {int(k): float(v * 100) for k, v in R.groupby('sq')['r'].median().items()}
dist['corr_share_vs_next_ret_spearman'] = float(R[['share', 'r']].corr(method='spearman').iloc[0, 1])
OUT['main_picks_share_distribution'] = dist
log('picks dist', dist['share_pcts'], dist['winners_top10pct'], dist['losers_bottom10pct'])

# ---------------- 訊號之間的相關（池內橫斷面 Spearman，月平均）----------------
cor = {}
for k in ((126, 0), (252, 0), (252, 21)):
    vals = {'IMOM_s52': [], 'OMOM_s52': [], 'IMOM_OMOM': [], 'share_s52': [], 'diff_s52': []}
    for c in cache:
        el = c['el']
        CI, CO = c['w'][k][0], c['w'][k][1]
        sh_ = L.form_signal(CI, CO, 'share'); df_ = CI - CO
        def sc(a, b):
            m = el & np.isfinite(a) & np.isfinite(b)
            return pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()) if m.sum() > 30 else np.nan
        vals['IMOM_s52'].append(sc(CI, c['s52'])); vals['OMOM_s52'].append(sc(CO, c['s52'])); vals['IMOM_OMOM'].append(sc(CI, CO))
        vals['share_s52'].append(sc(sh_, c['s52'])); vals['diff_s52'].append(sc(df_, c['s52']))
    cor[f'L{k[0]}_S{k[1]}'] = {kk: float(np.nanmean(v)) for kk, v in vals.items()}
OUT['signal_correlations'] = cor
log('corr', cor)

(L.HERE / 'explore.json').write_text(L.jdump(OUT), encoding='utf-8')
log('done')

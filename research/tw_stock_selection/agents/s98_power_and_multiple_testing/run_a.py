#!/usr/bin/env python3
"""S98 任務 A：最新淨值的月超額統計、DSR、Harvey-Liu、Hansen SPA（可取得族群內）。用法：python3 -I -B run_a.py"""
import sys, json, math, time, resource, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
AG = HERE.parent
ROOT = AG.parent.parent.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0]
sys.path.insert(0, str(AG / 'o_main_spa_breaks'))
import spa_lib  # noqa: E402
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
EMC = 0.5772156649015329
B = 5000; SEED = 20261126

# ---------- 試驗數（BACKLOG 逐行加總，見 trial_count）
N_S52 = 3006
INC = {'S54': 7, 'S53': 6, 'S56': 6, 'S57': 5, 'S58': 3, 'S67': 3, 'S68': 2, 'S70': 3, 'S78': 288, 'S80': 66, 'S77': 146, 'S81': 192, 'S79': 109,
       'S85': 28, 'S87': 16, 'S88': 19, 'S89': 26, 'S90': 30, 'S91': 28, 'S92': 12, 'S93': 12, 'S94': 2, 'S95': 1, 'S96': 10}
import re
bl = (ROOT / 'research/tw_stock_selection/BACKLOG.md').read_text(encoding='utf8').split('\n')
auto = {}
for l in bl:
    m = re.findall(r'(?:試驗數|資料窺探次數|N)\s*[+＋]\s*([0-9,]+)', l)
    k = re.search(r'(S\d+)', l)
    if m and k: auto[k.group(1)] = auto.get(k.group(1), 0) + sum(int(x.replace(',', '')) for x in m)
pre = {k: auto[k] for k in ('S45', 'S46', 'S47', 'S48', 'S49', 'S50')}
post = {k: v for k, v in auto.items() if k not in pre and v > 0}
assert post == INC, (post, INC)
N_ALL = N_S52 + sum(INC.values())
trial_count = dict(anchor_S52=N_S52, already_in_anchor=pre, increments_after_S52=INC, sum_increments=sum(INC.values()), N_all=N_ALL,
                   unlogged_notes='未記數：S51（3 格，與 S47 重疊）、S71（B1/B2 兩規則＋n 與下限兩輪）、S72 安慰劑（隨機排名，不是挑規則）、S76/S82/S83/S84/S97（描述性；S83 有 45 個領先指標檢定、S82 有 1 個事後題庫規則）。若全計約 +50～+60 → 約 4,080；S52 註記的錨點模糊 ±15。')
print('N_all', N_ALL, flush=True)
N_TENTH = round(N_ALL / 10)

# ---------- 月超額
def month_ret(nav):
    me = nav.groupby(nav.index.to_period('M')).tail(1)
    mm = pd.concat([nav.iloc[[0]], me[me.index > nav.index[0]]])
    r = mm.pct_change().iloc[1:]; r.index = r.index.to_period('M'); return r


def nw_t(x, lag):
    x = np.asarray(x, float); T = len(x); e = x - x.mean(); s = (e @ e) / T
    for i in range(1, lag + 1): s += 2 * (1 - i / (lag + 1)) * (e[i:] @ e[:-i]) / T
    return float(x.mean() / math.sqrt(s / T))


def desc(x):
    x = np.asarray(x, float); T = len(x); mu, sd = x.mean(), x.std(ddof=1)
    lag2 = int(math.floor(4 * (T / 100) ** (2 / 9)))
    t = mu / sd * math.sqrt(T); tn = nw_t(x, 6)
    return dict(months=T, ann_excess_pp=mu * 1200, te_pp=sd * math.sqrt(12) * 100, IR=mu / sd * math.sqrt(12), t_iid=t, t_nw6=tn, t_nw_auto=nw_t(x, lag2), nw_auto_lag=lag2,
                p1_iid=float(stats.t.sf(t, T - 1)), p1_nw6=float(stats.norm.sf(tn)), p2_iid=float(2 * stats.t.sf(abs(t), T - 1)),
                skew=float(stats.skew(x)), kurt=float(stats.kurtosis(x, fisher=False)), rho1=float(np.corrcoef(x[1:], x[:-1])[0, 1]), SR_m=mu / sd, sd_m=sd, mean_m=mu,
                hit=float((x > 0).mean()))


off = pd.read_csv(ROOT / 'xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)
v3 = pd.read_csv(AG / 's75_continuous_2008_2026/v3_2026-11-25_limit7/nav_2008_2026_v3.csv.gz', index_col=0, parse_dates=True)
Ro = month_ret(off); Rv = month_ret(v3)
P = lambda s: pd.Period(s, 'M')
cg = lambda r: float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
SER = {}
def add(name, R, col, a, b, src):
    m = (R.index >= P(a)) & (R.index <= P(b)); x = (R.loc[m, col] - R.loc[m, '0050'])
    d = desc(x.to_numpy()); d.update(period=f'{a}～{b}', source=src, cagr_strategy_pp=cg(R.loc[m, col].to_numpy()), cagr_0050_pp=cg(R.loc[m, '0050'].to_numpy()))
    SER[name] = dict(x=x, d=d)
add('W_official', Ro, 'E0|W10_70', '2015-07', '2026-10', 'navs_official E0|W10_70')
add('W_official_2019', Ro, 'E0|W10_70', '2019-01', '2026-10', 'navs_official E0|W10_70')
add('W_v3_full', Rv, 'V_BRIDGE', '2008-01', '2026-10', 'v3 V_BRIDGE')
add('W_v3_pre', Rv, 'V_BRIDGE', '2008-01', '2015-06', 'v3 V_BRIDGE')
add('W_v3_post', Rv, 'V_BRIDGE', '2015-07', '2026-10', 'v3 V_BRIDGE')
add('V_BRIDGE_2008_2013', Rv, 'V_BRIDGE', '2008-01', '2013-01', 'v3 V_BRIDGE')
add('V_BRIDGE_2013_2015', Rv, 'V_BRIDGE', '2013-02', '2015-06', 'v3 V_BRIDGE（B2 段）')
add('V_MASK_2008_2013', Rv, 'V_MASK', '2008-01', '2013-01', 'v3 V_MASK')
add('V_MASK_2008_2015', Rv, 'V_MASK', '2008-01', '2015-06', 'v3 V_MASK')
add('V_MASK_full', Rv, 'V_MASK', '2008-01', '2026-10', 'v3 V_MASK')
assert SER['W_official']['d']['months'] == 136 and SER['W_v3_pre']['d']['months'] == 90
for k, v in SER.items(): print(k, {a: round(b, 3) for a, b in v['d'].items() if a in ('months', 'ann_excess_pp', 'te_pp', 'IR', 't_iid', 't_nw6', 'p1_iid')}, flush=True)

# ---------- 族群
FE = pd.read_parquet(AG / 'o_main_spa_breaks/data/family_excess.parquet')
END = pd.Timestamp('2026-10-12')


def broad(start, end, nav, col):
    sub = FE[(FE.index >= pd.Timestamp(start)) & (FE.index < pd.Timestamp(end))]
    sub = sub.loc[:, sub.notna().all()]
    sub = sub.T.round(12).drop_duplicates().T
    sub = sub.loc[:, sub.std(ddof=1) >= 1e-10]
    allg = list(FE.index) + [END]
    i0 = allg.index(sub.index[0]); grid = allg[i0:i0 + len(sub) + 1]
    ns = nav[col].reindex(grid, method='ffill').to_numpy(); nb = nav['0050'].reindex(grid, method='ffill').to_numpy()
    fx = (ns[1:] / ns[:-1] - 1) - (nb[1:] / nb[:-1] - 1)
    return sub, fx


S78 = pd.read_csv(AG / 's78_full_period_consistent_rule/grid_monthly_excess.csv.gz', index_col=0); S78.index = pd.PeriodIndex(S78.index, freq='M')
NEARNAV = pd.read_csv(AG / 's52_snooping_adjusted_significance/family_monthly_nav.csv.gz', index_col=0, parse_dates=True)
NR = NEARNAV.pct_change().iloc[1:]; NR.index = NR.index.to_period('M')
near_cols = [c for c in NR.columns if c.split(':')[0] in ('S45', 'S46', 'S47', 'S48')]
NEAR = NR[near_cols].sub(NR['0050'], axis=0)


def fam_stats(X, fx, per_year):
    X = np.asarray(X, float); sr = X.mean(0) / X.std(0, ddof=1); T = X.shape[0]
    Z = (X - X.mean(0)) / X.std(0, ddof=1); n = X.shape[1]
    s = Z.sum(1); rbar = float(((s @ s) / (T - 1) - n) / (n * (n - 1)))
    fsr = fx.mean() / fx.std(ddof=1)
    cf = float(np.mean([np.corrcoef(fx, X[:, j])[0, 1] for j in range(n)]))
    return dict(n_members=int(n), T=int(T), V_SR=float(np.var(sr, ddof=1)), SD_SR_ann=float(np.std(sr, ddof=1) * math.sqrt(per_year)), SR_ann_max=float(sr.max() * math.sqrt(per_year)),
                SR_ann_median=float(np.median(sr) * math.sqrt(per_year)), focus_SR_ann=float(fsr * math.sqrt(per_year)), focus_rank=int(1 + (sr > fsr).sum()),
                mean_pairwise_corr=rbar, mean_corr_with_focus=cf, eff=spa_lib.eff_trials(X), share_t_gt2=float((sr * math.sqrt(T) > 2).mean())), sr


WIN = {  # 窗名: (焦點序列鍵, 廣族群起訖, nav, col, 月起訖)
    'A_2015_official': ('W_official', '2015-07-01', '2027-01-01', off, 'E0|W10_70', '2015-07', '2026-10'),
    'B_2008_v3_full': ('W_v3_full', '2008-01-01', '2027-01-01', v3, 'V_BRIDGE', '2008-01', '2026-10'),
    'C_2008_2015_v3_pre': ('W_v3_pre', '2008-01-01', '2015-07-01', v3, 'V_BRIDGE', '2008-01', '2015-06'),
}
FAM = {}
for w, (fk, bs, be, nav, col, ma, mb) in WIN.items():
    sub, fx = broad(bs, be, nav, col)
    per_year = 12.0 * 1.0  # 窗頻約每月一格
    st, sr = fam_stats(sub.to_numpy(), fx, 12)
    st['windows'] = f'{sub.index[0].date()}～{sub.index[-1].date()}（{len(sub)} 窗）'
    x = SER[fk]['x']
    g = S78.loc[(S78.index >= P(ma)) & (S78.index <= P(mb))]
    g = g.T.round(6).drop_duplicates().T; g = g.loc[:, g.std(ddof=1) > 1e-10]
    assert (g.index == x.index).all()
    st78, sr78 = fam_stats(g.to_numpy(), x.to_numpy(), 12)
    FAM[w] = dict(broad=dict(sub=sub, fx=fx, sr=sr, st=st), s78=dict(g=g, sr=sr78, st=st78))
    if w == 'A_2015_official':
        nn = NEAR.loc[x.index]; nn = nn.T.round(6).drop_duplicates().T
        stn, srn = fam_stats(nn.to_numpy(), x.to_numpy(), 12)
        FAM[w]['near'] = dict(g=nn, sr=srn, st=stn)
    print(w, 'broad', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items() if k != 'eff'}, st['eff'], flush=True)
    print(w, 's78', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st78.items() if k != 'eff'}, st78['eff'], flush=True)

# ---------- DSR
def psr(sr, sr_star, T, g3, g4):
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(T - 1) / math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))))


def sr0(V, N):
    N = max(float(N), 2.0)
    return math.sqrt(V) * ((1 - EMC) * stats.norm.ppf(1 - 1 / N) + EMC * stats.norm.ppf(1 - 1 / (N * math.e)))


DSR = {}
for w, (fk, *_r) in WIN.items():
    d = SER[fk]['d']; T, SRm, g3, g4 = d['months'], d['SR_m'], d['skew'], d['kurt']
    fb, f78 = FAM[w]['broad']['st'], FAM[w]['s78']['st']
    Vs = {'V1_broad_empirical': fb['V_SR'], 'V2_null_theory_1_over_T': 1.0 / (T - 1), 'V3_s78_surge_family': f78['V_SR']}
    if 'near' in FAM[w]: Vs['V3b_near97'] = FAM[w]['near']['st']['V_SR']
    Ns = {'N_all': N_ALL, 'N_s52_3006': N_S52, 'N_tenth': N_TENTH,
          'N_rho_broad': fb['mean_pairwise_corr'] + (1 - fb['mean_pairwise_corr']) * N_ALL,
          'N_rho_s78': f78['mean_pairwise_corr'] + (1 - f78['mean_pairwise_corr']) * N_ALL,
          'N_eig_LiJi_broad': fb['eff']['M_eff_LiJi'], 'N_eig_Nyholt_broad': fb['eff']['M_eff_Nyholt'], 'N_eig_LiJi_s78': f78['eff']['M_eff_LiJi'], 'N_10': 10, 'N_1': 1}
    tab = {}
    for vk, V in Vs.items():
        tab[vk] = {}
        for nk, N in Ns.items():
            z = sr0(V, N) if N > 1 else 0.0
            tab[vk][nk] = dict(N=float(N), SR0_ann=z * math.sqrt(12), DSR=psr(SRm, z, T, g3, g4), pass95=bool(psr(SRm, z, T, g3, g4) >= 0.95))
    DSR[w] = dict(focus_SR_ann=d['IR'], T=T, PSR0=psr(SRm, 0, T, g3, g4), SD_SR_ann_used={k: math.sqrt(v * 12) for k, v in Vs.items()}, N_used=Ns, table=tab)
    print('DSR', w, 'SR', round(d['IR'], 3), {vk: {nk: round(tab[vk][nk]['DSR'], 3) for nk in ('N_all', 'N_tenth', 'N_rho_s78', 'N_eig_LiJi_broad')} for vk in tab}, flush=True)

# ---------- Harvey-Liu
p2 = lambda t: float(2 * stats.norm.sf(abs(t)))
t_of_p = lambda p: float(stats.norm.isf(p / 2)) if p < 1 else 0.0


def holm_bhy(pf, po):
    allp = np.concatenate([[pf], po]); N = len(allp); order = np.argsort(allp, kind='stable'); ps = allp[order]
    r = int(np.where(order == 0)[0][0])
    holm = max(min((N - j) * ps[j], 1.0) for j in range(r + 1))
    cN = float(np.sum(1.0 / np.arange(1, N + 1)))
    bhy = float(np.min(np.minimum(N * cN / np.arange(1, N + 1) * ps, 1.0)[r:]))
    return float(holm), bhy, r + 1


HL = {}
for w, (fk, *_r) in WIN.items():
    d = SER[fk]['d']; T = d['months']; tf = d['t_iid']; pf = p2(tf); SRa = d['IR']
    srb = FAM[w]['broad']['sr']; pemp = np.sort(np.array([p2(t) for t in srb * math.sqrt(FAM[w]['broad']['st']['T'])]))
    out = dict(t_single=tf, p2_single=pf, ann_excess_pp=d['ann_excess_pp'], by_N={})
    for nk in ('N_all', 'N_s52_3006', 'N_tenth', 'N_rho_s78', 'N_eig_LiJi_broad', 'N_10'):
        N = max(int(round(DSR[w]['N_used'][nk])), 2)
        po = np.quantile(pemp, (np.arange(N - 1) + 0.5) / (N - 1))
        ph, pb, rk = holm_bhy(pf, po)
        row = dict(N=N, focus_rank=rk)
        for nm, pa in (('Bonferroni', min(N * pf, 1.0)), ('Holm', ph), ('BHY', pb)):
            ta = t_of_p(pa); ratio = max(ta / math.sqrt(T / 12) / SRa, 0.0) if SRa > 0 else 0.0
            row[nm] = dict(p_adj=pa, t_adj=ta, haircut=1 - ratio, ann_excess_adj_pp=d['ann_excess_pp'] * ratio)
        out['by_N'][nk] = row
    out['t_needed_bonferroni_5pct'] = {nk: t_of_p(0.05 / max(DSR[w]['N_used'][nk], 1)) for nk in ('N_all', 'N_tenth', 'N_rho_s78', 'N_eig_LiJi_broad', 'N_10')}
    HL[w] = out
    print('HL', w, round(tf, 2), {nk: {m: (round(v[m]['t_adj'], 2), round(v[m]['haircut'], 3)) for m in ('Bonferroni', 'Holm', 'BHY')} for nk, v in out['by_N'].items()}, flush=True)

# ---------- SPA
KEYS = ('n', 'N', 'max_t', 'p_RC', 'p_RC_focus', 'p_SPA_l', 'p_SPA_c', 'p_SPA_u', 'p_SPA_l_focus', 'p_SPA_c_focus', 'p_SPA_u_focus', 'focus_rank', 'focus_t', 'p_StepM_focus', 'p_single_focus')


def run_spa(fx, X, tag):
    D = np.ascontiguousarray(np.column_stack([np.asarray(fx, float), np.asarray(X, float)]))
    out = {}
    for L in (6, 3, 12):
        W = spa_lib.sb_counts(D.shape[0], L, B, SEED + L)
        r = spa_lib.mcp_tests(D, W, L, var='pr', focus=0)
        out[f'L{L}'] = {k: r[k] for k in KEYS}; del W
        print('SPA', tag, f'L{L}', 'focus p_c', round(r['p_SPA_c_focus'], 4), 'family p_c', round(r['p_SPA_c'], 4), 'rank', r['focus_rank'], '/', r['N'], 'focus_t', round(r['focus_t'], 2), f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    return out


SPA = {}
for w, (fk, *_r) in WIN.items():
    x = SER[fk]['x'].to_numpy()
    SPA[w] = dict(F_broad=run_spa(FAM[w]['broad']['fx'], FAM[w]['broad']['sub'].to_numpy(), w + ' broad'),
                  F_s78=run_spa(x, FAM[w]['s78']['g'].to_numpy(), w + ' s78'))
    if 'near' in FAM[w]:
        SPA[w]['F_near'] = run_spa(x, FAM[w]['near']['g'].to_numpy(), w + ' near')
        SPA[w]['F_near_plus_s78'] = run_spa(x, np.column_stack([FAM[w]['near']['g'].to_numpy(), FAM[w]['s78']['g'].to_numpy()]), w + ' near+s78')

# 月切點敏感度（S52 作法的簡化版）：正式淨值改用每月第 d 個交易日切點的焦點單一 t
cut = {}
for dday in (1, 6, 11, 16, 21, 26):
    g = off.groupby(off.index.to_period('M')).apply(lambda q: q[q.index.day >= dday].iloc[0] if (q.index.day >= dday).any() else q.iloc[-1])
    r = g.pct_change().iloc[1:]; xx = (r['E0|W10_70'] - r['0050']).to_numpy()
    cut[str(dday)] = dict(n=len(xx), t_iid=float(xx.mean() / xx.std(ddof=1) * math.sqrt(len(xx))), IR=float(xx.mean() / xx.std(ddof=1) * math.sqrt(12)))
print('cut', cut, flush=True)

# ---------- 判定（沿用 S52 三項）
dec = {}
for w in WIN:
    c1 = DSR[w]['table']['V1_broad_empirical']['N_all']['DSR']; c2 = SPA[w]['F_broad']['L6']['p_SPA_c_focus']; c3 = HL[w]['by_N']['N_all']['BHY']['p_adj']
    ps = dict(C1_DSR=bool(c1 >= 0.95), C2_SPA=bool(c2 < 0.05), C3_BHY=bool(c3 < 0.05)); k = sum(ps.values())
    dec[w] = dict(values=dict(DSR=c1, SPA_c_focus_p=c2, BHY_p=c3), passed=ps, label='校正後仍顯著' if k == 3 else ('部分通過（證據不一致）' if k else '校正後不顯著'))
print('判定', json.dumps(dec, ensure_ascii=False), flush=True)

# ---------- 2008–2015 樣本外
oos = {}
for k in ('W_v3_pre', 'V_MASK_2008_2013', 'V_MASK_2008_2015', 'V_BRIDGE_2008_2013', 'V_BRIDGE_2013_2015'):
    d = SER[k]['d']
    oos[k] = dict(period=d['period'], months=d['months'], ann_excess_pp=d['ann_excess_pp'], te_pp=d['te_pp'], IR=d['IR'], t_iid=d['t_iid'], t_nw6=d['t_nw6'], p1_iid=d['p1_iid'], p1_nw6=d['p1_nw6'],
                  p1_iid_bonf3=min(3 * d['p1_iid'], 1.0), p1_nw6_bonf3=min(3 * d['p1_nw6'], 1.0), cagr_strategy_pp=d['cagr_strategy_pp'], cagr_0050_pp=d['cagr_0050_pp'])
    # 區塊拔靴單尾 p（H0 平均=0）
    x = SER[k]['x'].to_numpy(); rng = np.random.default_rng(SEED + len(x)); n = len(x); e = x - x.mean(); L = 6; nb = math.ceil(n / L)
    st_ = rng.integers(0, n, size=(20000, nb)); idx = ((st_[:, :, None] + np.arange(L)[None, None, :]) % n).reshape(20000, nb * L)[:, :n]
    oos[k]['p1_block_bootstrap_L6'] = float((e[idx].mean(1) >= x.mean()).mean())
print('oos', json.dumps({k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} for k, v in oos.items()}, ensure_ascii=False), flush=True)

strip = lambda f: {k: v['st'] for k, v in f.items()}
res = dict(prereg_sha256=sha(HERE / 'PREREG.json'), inputs=dict(navs_official_sha256=sha(ROOT / 'xq_scripts/site/navs_official.csv.gz'),
           v3_nav_sha256=sha(AG / 's75_continuous_2008_2026/v3_2026-11-25_limit7/nav_2008_2026_v3.csv.gz'), spa_lib_sha256=sha(AG / 'o_main_spa_breaks/spa_lib.py')),
           trial_count=trial_count, series={k: v['d'] for k, v in SER.items()}, families={w: strip(f) for w, f in FAM.items()}, DSR=DSR, HarveyLiu=HL, SPA=SPA,
           cutpoint_sensitivity_official=cut, decision_S52_rule=dec, oos_2008_2015=oos, bootstrap=dict(B=B, L=[6, 3, 12], seed=SEED),
           runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'results_a.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
pd.DataFrame({k: v['x'] for k, v in SER.items() if k in ('W_official', 'W_v3_full', 'V_MASK_full')}).round(8).to_csv(HERE / 'monthly_excess.csv')
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')

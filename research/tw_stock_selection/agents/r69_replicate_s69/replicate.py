# -*- coding: utf-8 -*-
"""R69：獨立重現 S69（只依 SPEC/PREREG/PLAN_FROZEN/FINDINGS；未讀 S69 的 run.py 等程式）。
執行：python3 -I -B replicate.py
"""
import json, os, csv
import numpy as np, pandas as pd
from scipy import stats

ROOT = '/home/user/-'
HERE = os.path.dirname(os.path.abspath(__file__))
S69 = os.path.join(ROOT, 'research/tw_stock_selection/agents/s69_forward_validation_plan')
SEED = 690069          # 刻意與 S69（20261112）不同
NSIM = 200000
L = 6
out = {'seed': SEED, 'n_sim': NSIM}

# ---------- 1. 月超額 ----------
d = pd.read_csv(os.path.join(ROOT, 'xq_scripts/site/navs_official.csv.gz'), parse_dates=['date'])
d = d[['date', '0050', 'E0|W10_70']].set_index('date')
me = d.groupby(d.index.to_period('M')).last()
base = d.iloc[[0]].copy(); base.index = [me.index[0] - 1]
lv = pd.concat([base, me])
r = lv.pct_change().iloc[1:]
x = (r['E0|W10_70'] - r['0050']).values
n = len(x); mu = x.mean(); sd = x.std(ddof=1)
xc = x - mu
out['monthly'] = dict(
    n_months=int(n), mean_pct=mu * 100, sd_pct=sd * 100,
    ann_excess_pp=12 * mu * 100, te_pp=sd * np.sqrt(12) * 100,
    skew=float(stats.skew(x)), kurt_pearson=float(stats.kurtosis(x, fisher=False)),
    acf1=float((xc[1:] * xc[:-1]).sum() / (xc ** 2).sum()),
    ir=float(mu / sd * np.sqrt(12)))
TE = sd * np.sqrt(12) * 100

# ---------- 3. 序貫界線：遞迴數值積分（Armitage-McPherson-Rowe） ----------
def gs_bounds(spend_inc, K, upper=True, lim=12.0, npts=6001):
    """等距資訊 K 次；每次花掉 spend_inc[k] 的單邊機率。回傳 Z 界線（上界）。
    以部分和 S_k = sum of iid N(0,1) 增量，Z_k = S_k/sqrt(k)。"""
    from scipy.optimize import brentq
    bounds = []
    # 第一次
    b1 = stats.norm.isf(spend_inc[0])
    bounds.append(b1)
    # 尚未越界的次密度 g(s) on grid
    grid = None; g = None
    for k in range(2, K + 1):
        prevb = bounds[-1] * np.sqrt(k - 1)            # S 尺度上界
        lo = -lim * np.sqrt(k - 1)
        s = np.linspace(lo, prevb, npts)
        h = s[1] - s[0]
        w = np.full(npts, h); w[0] = w[-1] = h / 3
        w[1:-1:2] = 4 * h / 3; w[2:-1:2] = 2 * h / 3   # Simpson
        if k == 2:
            dens = stats.norm.pdf(s)
        else:
            # g_{k-1}(s) = ∫ g_{k-2}(u) φ(s-u) du
            dens = (stats.norm.pdf(s[:, None] - grid[None, :]) * (g * wg)[None, :]).sum(1)
        grid, g, wg = s, dens, w
        def cross(b):   # P(未越界到 k-1，且 S_k >= b*sqrt(k))
            return (stats.norm.sf(b * np.sqrt(k) - grid) * g * wg).sum()
        target = spend_inc[k - 1]
        bk = brentq(lambda b: cross(b) - target, 0.0, 30.0, xtol=1e-12)
        bounds.append(bk)
    return np.array(bounds)

def obf_classic_c(K, alpha):
    """原始 O'Brien-Fleming：Z_k >= c*sqrt(K/k)，單尾 alpha（驗證用）。"""
    from scipy.optimize import brentq
    def tot(c, npts=4001, lim=12.0):
        b = c * np.sqrt(K / np.arange(1, K + 1))
        p = stats.norm.sf(b[0]); grid = g = wg = None
        for k in range(2, K + 1):
            prevb = b[k - 2] * np.sqrt(k - 1)
            s = np.linspace(-lim * np.sqrt(k - 1), prevb, npts); h = s[1] - s[0]
            w = np.full(npts, h); w[0] = w[-1] = h / 3; w[1:-1:2] = 4 * h / 3; w[2:-1:2] = 2 * h / 3
            dens = stats.norm.pdf(s) if k == 2 else (stats.norm.pdf(s[:, None] - grid[None, :]) * (g * wg)[None, :]).sum(1)
            grid, g, wg = s, dens, w
            p += (stats.norm.sf(b[k - 1] * np.sqrt(k) - grid) * g * wg).sum()
        return p
    return brentq(lambda c: tot(c) - alpha, 1.5, 3.0, xtol=1e-6)

K = 10
tt = np.arange(1, K + 1) / K
a_cum = 2 - 2 * stats.norm.cdf(stats.norm.ppf(0.975) / np.sqrt(tt))   # 單尾整體 5%
a_inc = np.diff(np.concatenate([[0], a_cum]))
u = gs_bounds(a_inc, K)
df = 12 * np.arange(1, K + 1) - 1
ut = stats.t.isf(stats.norm.sf(u), df)
g_inc = np.full(K, 0.10 / K)
f = gs_bounds(g_inc, K)                       # 對稱：下界 = −f
ft = stats.t.isf(stats.norm.sf(f), df)
out['validation_obf_classic'] = dict(c_K5=obf_classic_c(5, 0.025), c_K10=obf_classic_c(10, 0.025),
                                     ref='Jennison & Turnbull 表 2.3：2.040、2.087')
out['boundaries'] = dict(alpha_cum=a_cum.tolist(), z_eff=u.tolist(), t_eff=ut.tolist(),
                         z_fut=f.tolist(), t3_fut=(-ft).tolist())

# 多變量常態模擬交叉檢查 z 界線的整體型一誤差
rng = np.random.default_rng(SEED + 1)
cross_any = 0; NM = 2_000_000
for _ in range(4):
    S = np.cumsum(rng.standard_normal((NM // 4, K)), axis=1) / np.sqrt(np.arange(1, K + 1))
    cross_any += (S >= u).any(1).sum()
out['boundaries']['mvn_sim_type1_z'] = cross_any / NM

# ---------- 2. 檢定力 + 4. 照規則走 ----------
def paths_block(resid, nsim, T, L, rng):
    m = len(resid); nb = int(np.ceil(T / L))
    st = rng.integers(0, m, size=(nsim, nb))
    idx = (st[:, :, None] + np.arange(L)[None, None, :]) % m
    return resid[idx.reshape(nsim, nb * L)[:, :T]]

def evaluate(e, deltas_pp, years=(1, 2, 3, 5, 7, 10)):
    """e：去均值殘差路徑 (nsim,120)，小數。"""
    res = {}
    nmon = 12 * np.arange(1, 11)
    cs = np.cumsum(e, 1)[:, nmon - 1]; cs2 = np.cumsum(e * e, 1)[:, nmon - 1]
    me0 = cs / nmon
    var = (cs2 - nmon * me0 ** 2) / (nmon - 1)      # 平移不改變異數
    se = np.sqrt(var / nmon)
    tcrit = stats.t.ppf(0.95, nmon - 1)
    for dl in deltas_pp:
        mean = me0 + dl / 1200.0
        t = mean / se; t3 = (mean - 3 / 1200.0) / se; m_pp = mean * 1200
        pw = {}
        for y in years:
            k = y - 1
            pw[str(y)] = dict(p_sig5=float((t[:, k] >= tcrit[k]).mean()),
                              p_m_ge5=float((m_pp[:, k] >= 5).mean()),
                              p_m_lt0=float((m_pp[:, k] < 0).mean()))
        eff = t >= ut; fut = t3 <= -ft
        prag = (m_pp < 0) & (np.arange(1, 11) >= 5)
        big = 99
        def first(b): return np.where(b.any(1), b.argmax(1), big)
        fe, ff, fp = first(eff), first(fut), first(prag)
        fstop = np.minimum(ff, fp)
        # 口徑 A：照規則走，停用後不再判定有效（先觸者為結局）
        effA = fe < fstop                      # 同一次不可能同時（t>0 與 m<0）；t3 無效與有效亦互斥
        stopA_formal = (~effA) & (ff <= fp) & (ff < big)
        stopA_prag = (~effA) & (fp < ff)
        undec = (~effA) & (fstop == big)
        term_stop = undec & (m_pp[:, 9] < 3); owner = undec & (m_pp[:, 9] >= 3)
        seq = dict(
            eff_rule_following=float(effA.mean()),
            stop_formal=float(stopA_formal.mean()), stop_pragmatic=float(stopA_prag.mean()),
            stop_terminal=float(term_stop.mean()),
            stop_total=float((stopA_formal | stopA_prag | term_stop).mean()),
            owner_decides=float(owner.mean()),
            eff_ever_nonbinding=float((fe < big).mean()),
            fut_formal_ever=float((ff < big).mean()),
            prag_ever=float((fp < big).mean()),
            eff_year_median=(float(np.median(fe[fe < big]) + 1) if (fe < big).any() else None),
            eff_first_by_year=[float((fe == k).mean()) for k in range(10)])
        res[str(dl)] = dict(power=pw, sequential=seq)
    return res

rng = np.random.default_rng(SEED)
e = paths_block(xc, NSIM, 120, L, rng)

# ---- size_contingency（SPEC part2，事前寫死）：D1_main 模擬檢查，必要時 z 界線乘同一常數 ----
def _t_null(e):
    nmon = 12 * np.arange(1, 11)
    cs = np.cumsum(e, 1)[:, nmon - 1]; cs2 = np.cumsum(e * e, 1)[:, nmon - 1]
    m0 = cs / nmon
    return m0 / np.sqrt((cs2 - nmon * m0 ** 2) / (nmon - 1) / nmon)
t_null = _t_null(e)                       # δ＝0 的 t，也等於 δ＝3 的 t3
type1_raw = float((t_null >= ut).any(1).mean())
falsefut_raw = float((t_null <= -ft).any(1).mean())
def _bisect(fn, target):
    lo, hi = 1.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if fn(mid) <= target: hi = mid
        else: lo = mid
    return hi
c_eff = 1.0; c_fut = 1.0
if type1_raw > 0.055:
    c_eff = _bisect(lambda c: (t_null >= stats.t.isf(stats.norm.sf(c * u), df)).any(1).mean(), 0.05)
    ut = stats.t.isf(stats.norm.sf(c_eff * u), df)
if falsefut_raw > 0.11:
    c_fut = _bisect(lambda c: (t_null <= -stats.t.isf(stats.norm.sf(c * f), df)).any(1).mean(), 0.10)
    ft = stats.t.isf(stats.norm.sf(c_fut * f), df)
out['size_contingency'] = dict(
    type1_sim_delta0_theory_bounds=type1_raw, efficacy_triggered=bool(type1_raw > 0.055), c_eff=c_eff,
    false_futility_sim_delta3_theory_bounds=falsefut_raw, futility_triggered=bool(falsefut_raw > 0.11), c_fut=c_fut,
    false_futility_after=float((t_null <= -ft).any(1).mean()),
    t3_fut_theory=(-stats.t.isf(stats.norm.sf(f), df)).tolist(), t3_fut_final=(-ft).tolist(), t_eff_final=ut.tolist())
del t_null
out['D1_main'] = evaluate(e, [-5, 0, 3, 5, 8, 12])
del e
for nm, LL in (('D1_L1', 1), ('D1_L12', 12)):
    e = paths_block(xc, NSIM, 120, LL, np.random.default_rng(SEED + LL))
    out[nm] = evaluate(e, [0, 5, 12]); del e
# D5 常態獨立：模擬 + 解析（非中心 t）
e = np.random.default_rng(SEED + 5).standard_normal((NSIM, 120)) * sd
e -= 0  # 母體平均 0
out['D5_normal_sim'] = evaluate(e, [0, 5, 12]); del e
an = {}
for dl in (0, 3, 5, 8, 12):
    for y in (1, 2, 3, 5, 7, 10):
        nn = 12 * y; ncp = (dl / 1200.0) / (sd / np.sqrt(nn))
        an[f'{dl}|{y}'] = float(stats.nct.sf(stats.t.ppf(0.95, nn - 1), nn - 1, ncp))
out['D5_normal_analytic_p_sig5'] = an

json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1)

# ---------- 對照表 ----------
plan = json.load(open(os.path.join(S69, 'PLAN_FROZEN.json')))
D = out['D1_main']
rows = []
def add(item, s69, r69, tol, unit):
    diff = r69 - s69
    rows.append(dict(項目=item, S69=s69, R69=round(r69, 4), 差=round(diff, 4), 容許=tol, 單位=unit,
                     結果='符合' if abs(diff) <= tol else '不符'))
mo = out['monthly']
add('月數', 136, mo['n_months'], 0, '個月')
add('年化超額', 11.68, mo['ann_excess_pp'], 0.005, '個百分點')
add('追蹤誤差', 16.21, mo['te_pp'], 0.005, '個百分點')
add('追蹤誤差（PLAN_FROZEN planning_te_pp）', plan['planning_te_pp'], mo['te_pp'], 1e-9, '個百分點')
add('偏態', 1.48, mo['skew'], 0.005, '')
add('一階自相關', -0.117, mo['acf1'], 0.0005, '')
P = lambda dl, y, kk: D[str(dl)]['power'][str(y)][kk] * 100
add('檢定力 真實+5／10年 單尾5%顯著', 17.4, P(5, 10, 'p_sig5'), 1.5, '%')
add('檢定力 真實+12／10年', 84.2, P(12, 10, 'p_sig5'), 1.5, '%')
add('檢定力 真實+12／5年', 49.9, P(12, 5, 'p_sig5'), 1.5, '%')
add('真實0／1年 看起來≥+5', 32.6, P(0, 1, 'p_m_ge5'), 1.5, '%')
add('真實0／5年 看起來≥+5', 20.5, P(0, 5, 'p_m_ge5'), 1.5, '%')
add('真實0／10年 看起來≥+5', 13.0, P(0, 10, 'p_m_ge5'), 1.5, '%')
add('真實+12／1年 為負', 19.6, P(12, 1, 'p_m_lt0'), 1.5, '%')
tab = {0: (0.4, 1.0, 1.2, 1.6), 3: (0.7, 2.6, 4.0, 7.5), 5: (1.2, 4.7, 8.2, 17.4), 8: (2.6, 11.0, 20.6, 44.4), 12: (5.9, 27.8, 49.9, 84.2)}
for dl, vals in tab.items():
    for y, v in zip((1, 3, 5, 10), vals):
        add(f'檢定力表 真實{dl:+d}／{y}年', v, P(dl, y, 'p_sig5'), 1.5, '%')
add('常態獨立 +5／10年（解析）', 24.9, an['5|10'] * 100, 1.5, '%')
add('常態獨立 +5／10年（模擬）', 24.9, out['D5_normal_sim']['5']['power']['10']['p_sig5'] * 100, 1.5, '%')
for k, lk in enumerate(plan['looks']):
    add(f'有效 t 界線 第{k+1}次', lk['efficacy_t_min'], ut[k], 0.02, 't')
for k, lk in enumerate(plan['looks']):
    add(f'正式無效 t3 界線 第{k+1}次', lk['futility_t3_max'], -ft[k], 0.02, 't')
S = lambda dl, kk: D[str(dl)]['sequential'][kk] * 100
ftab = {0: (1.0, 81.4, 17.6), 3: (5.3, 59.8, 34.9), 5: (13.1, 42.3, 44.6), 8: (36.9, 18.8, 44.3), 12: (79.1, 3.2, 17.7)}
for dl, (a, b, c) in ftab.items():
    add(f'結局 真實{dl:+d} 判定有效', a, S(dl, 'eff_rule_following'), 1.5, '%')
    add(f'結局 真實{dl:+d} 停用合計', b, S(dl, 'stop_total'), 1.5, '%')
    add(f'結局 真實{dl:+d} 第10年擁有者決定', c, S(dl, 'owner_decides'), 1.5, '%')
add('無效界線放大常數 c（PLAN_FROZEN 反推 1.3283）', 1.3283, c_fut, 0.02, '倍')
add('真實+5 被務實停用線誤停', 26.9, S(5, 'stop_pragmatic'), 1.5, '%')
with open(os.path.join(HERE, 'compare.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
bad = [r for r in rows if r['結果'] != '符合']
for r_ in rows: print(r_)
print('不符：', len(bad), '/', len(rows))
import resource
print('峰值記憶體 MB：', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

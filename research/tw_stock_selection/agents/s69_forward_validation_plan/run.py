# S69 前瞻驗證計畫：檢定力、序貫界線、貝氏表、白話表（SPEC.json／PREREG.json 先寫；本程式只讀資料）
# 用法：python3 -I -B run.py
import json, hashlib, math, time, resource, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize

T0 = time.time()
H = Path(__file__).resolve().parent
ROOT = H.parent.parent                                   # tw_stock_selection
NAV = Path('/home/user/-/xq_scripts/site/navs_official.csv.gz')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(H / 'SPEC.json') == (H / 'SPEC.sha256').read_text().split()[0]
assert sha(H / 'PREREG.json') == (H / 'PREREG.sha256').read_text().split()[0]
SPEC = json.loads((H / 'SPEC.json').read_text(encoding='utf-8'))
res = dict(spec_sha256=sha(H / 'SPEC.json'), prereg_sha256=sha(H / 'PREREG.json'), spec_deviations=[], inputs={'navs_official_sha256': sha(NAV)})
N_SIM = SPEC['part1_power']['n_sim']; SEED = SPEC['part1_power']['seed']
K = 10; LOOK_N = np.arange(1, K + 1) * 12

# ---------------- 0. 資料
nv = pd.read_csv(NAV, parse_dates=['date']).set_index('date')
me = nv.groupby(nv.index.to_period('M')).tail(1)
base = nv.iloc[[0]]
assert str(base.index[0].date()) == '2015-07-01'
mm = pd.concat([base, me[me.index > base.index[0]]]) if me.index[0] != base.index[0] else me
r = mm[['E0|W10_70', '0050']].pct_change().iloc[1:]
x_all = (r['E0|W10_70'] - r['0050']).to_numpy()
per = r.index.to_period('M')
assert len(x_all) == 136, len(x_all)
s52 = json.loads((ROOT / 'agents/s52_snooping_adjusted_significance/results.json').read_text())['focus']['full']['excess_vs_0050']
assert abs(x_all.mean() - s52['mean_m']) < 1e-9 and abs(x_all.std(ddof=1) - s52['sd_m']) < 1e-9, '與 S52 月超額不符'
oos = pd.read_csv(ROOT / 'agents/s55_pre2015_oos/oos_monthly.csv', index_col=0)
x_oos = oos.loc['2008-01':'2013-01', 'excess'].to_numpy(); assert len(x_oos) == 61


def desc(x):
    x = np.asarray(x, float); n = len(x); mu = x.mean(); sd = x.std(ddof=1)
    ac = lambda k: float(np.corrcoef(x[k:], x[:-k])[0, 1])
    return dict(months=n, ann_excess_pp=mu * 1200, ann_te_pp=sd * math.sqrt(12) * 100, IR=mu / sd * math.sqrt(12), t_iid=mu / sd * math.sqrt(n),
                skew=float(stats.skew(x)), kurt=float(stats.kurtosis(x, fisher=False)), rho1=ac(1), rho2=ac(2), rho3=ac(3), rho6=ac(6), rho12=ac(12),
                min_m_pp=x.min() * 100, max_m_pp=x.max() * 100, hit_rate=float((x > 0).mean()))


x19 = x_all[np.asarray(per >= pd.Period('2019-01', 'M'))]; assert len(x19) == 94
res['monthly_excess'] = dict(official_full=desc(x_all), official_since2019=desc(x19), s55_primary=desc(x_oos))
TE_MAIN = res['monthly_excess']['official_full']['ann_te_pp']          # 約 16.21
TE_OOS = res['monthly_excess']['s55_primary']['ann_te_pp']             # 約 5.63
print('月超額', json.dumps(res['monthly_excess'], ensure_ascii=False), flush=True)

# ---------------- 1. 模擬路徑（只存各檢視點的累積平均與標準差；δ 用平移）
def sim_paths(resid, L, seed, normal=False):
    rng = np.random.default_rng(seed)
    n = 120; nb = math.ceil(n / L); M = len(resid)
    mean_b = np.empty((N_SIM, K)); sd_b = np.empty((N_SIM, K))
    CH = 25000
    for a in range(0, N_SIM, CH):
        if normal:
            X = rng.standard_normal((CH, n)) * resid.std(ddof=1)
        else:
            st = rng.integers(0, M, size=(CH, nb))
            idx = (st[:, :, None] + np.arange(L)[None, None, :]) % M
            X = resid[idx.reshape(CH, nb * L)[:, :n]]
        c1 = np.cumsum(X, 1)[:, LOOK_N - 1]; c2 = np.cumsum(X * X, 1)[:, LOOK_N - 1]
        mean_b[a:a + CH] = c1 / LOOK_N
        sd_b[a:a + CH] = np.sqrt(np.maximum(c2 - c1 * c1 / LOOK_N, 0) / (LOOK_N - 1))
    return mean_b, sd_b


e_all = x_all - x_all.mean()
VAR = {
    'D1_main': dict(resid=e_all, L=6), 'D1_L1': dict(resid=e_all, L=1), 'D1_L12': dict(resid=e_all, L=12),
    'D2_te56_scaled': dict(resid=e_all * (TE_OOS / TE_MAIN), L=6), 'D3_s55_resid': dict(resid=x_oos - x_oos.mean(), L=6),
    'D4_since2019': dict(resid=x19 - x19.mean(), L=6), 'D5_normal': dict(resid=e_all, L=1, normal=True),
}
PATH = {}
for i, (k, v) in enumerate(VAR.items()):
    PATH[k] = sim_paths(v['resid'], v['L'], SEED + i, v.get('normal', False))
    v['te_pp'] = float(v['resid'].std(ddof=1) * math.sqrt(12) * 100)
    print('路徑', k, round(v['te_pp'], 3), round(time.time() - T0, 1), flush=True)

# ---------------- 2. 檢定力（固定期限、單次檢定）
DEL = SPEC['part1_power']['grid']['delta_pp']; YRS = SPEC['part1_power']['grid']['years']
power = {}
for k, (mb, sb) in PATH.items():
    out = {}
    for y in YRS:
        j = y - 1; n = 12 * y; crit = stats.t.isf(0.05, n - 1)
        row = {}
        for d in DEL:
            m = mb[:, j] + d / 1200; t = m / (sb[:, j] / math.sqrt(n))
            row[str(d)] = dict(p_sig_1sided5=float((t >= crit).mean()), p_obs_ge5=float((m * 1200 >= 5).mean()), p_obs_lt0=float((m < 0).mean()),
                               obs_p05_pp=float(np.quantile(m, 0.05) * 1200), obs_p50_pp=float(np.median(m) * 1200), obs_p95_pp=float(np.quantile(m, 0.95) * 1200))
        out[str(y)] = row
    power[k] = dict(te_pp=VAR[k]['te_pp'], L=VAR[k]['L'], table=out)
res['power'] = power
zz = stats.norm.isf(0.05) + stats.norm.isf(0.20)
res['power_analytic'] = {
    'normal_power': {f'TE{te:.2f}': {str(y): {str(d): float(stats.norm.sf(stats.norm.isf(0.05) - d / (te / math.sqrt(y)))) for d in DEL} for y in YRS} for te in (TE_MAIN, TE_OOS)},
    'years_for_80pct_power': {f'TE{te:.2f}': {str(d): float((zz * te / d) ** 2) for d in DEL if d > 0} for te in (TE_MAIN, TE_OOS)},
    'years_for_50pct_power': {f'TE{te:.2f}': {str(d): float((stats.norm.isf(0.05) * te / d) ** 2) for d in DEL if d > 0} for te in (TE_MAIN, TE_OOS)},
    'se_pp_by_year': {f'TE{te:.2f}': {str(y): te / math.sqrt(y) for y in range(1, 11)} for te in (TE_MAIN, TE_OOS)},
}

# ---------------- 3. 序貫界線（遞迴數值積分；分數尺度 S_k＝k 個 N(0,1) 增量之和，Z_k＝S_k/√k）
GM = 3001


def gs_exit(c_list, upto=None):
    """給定分數尺度上界 c_1..c_k，回傳各次首次越界機率（H0）。"""
    probs = []; g = None; xg = None
    for k, c in enumerate(c_list, 1):
        if k == 1:
            probs.append(float(stats.norm.sf(c)))
        else:
            w = np.full(GM, xg[1] - xg[0]); w[0] *= .5; w[-1] *= .5
            probs.append(float(np.sum(w * g * stats.norm.sf(c - xg))))
        xn = np.linspace(-9 * math.sqrt(k) - 2, c, GM)
        if k == 1:
            gn = stats.norm.pdf(xn)
        else:
            gn = (stats.norm.pdf(xn[:, None] - xg[None, :]) * (w * g)[None, :]).sum(1)
        g, xg = gn, xn
    return probs


def gs_bounds(increments):
    """給定每次要花掉的 α，逐次解界線（分數尺度）。"""
    cs = []; g = None; xg = None
    for k, inc in enumerate(increments, 1):
        if k == 1:
            c = float(stats.norm.isf(inc))
        else:
            w = np.full(GM, xg[1] - xg[0]); w[0] *= .5; w[-1] *= .5
            f = lambda c: float(np.sum(w * g * stats.norm.sf(c - xg))) - inc
            c = optimize.brentq(f, -5.0, 12 * math.sqrt(k), xtol=1e-12)
        xn = np.linspace(-9 * math.sqrt(k) - 2, c, GM)
        gn = stats.norm.pdf(xn) if k == 1 else (stats.norm.pdf(xn[:, None] - xg[None, :]) * (w * g)[None, :]).sum(1)
        g, xg = gn, xn; cs.append(c)
    return np.array(cs)


def obf_const(alpha, KK):
    f = lambda c: sum(gs_exit([c * math.sqrt(KK)] * KK)) - alpha
    return optimize.brentq(f, 1.0, 4.0, xtol=1e-10)


val = {'OBF_K5_one_sided_0.025': obf_const(0.025, 5), 'OBF_K10_one_sided_0.025': obf_const(0.025, 10)}
assert abs(val['OBF_K5_one_sided_0.025'] - 2.040) < 0.002 and abs(val['OBF_K10_one_sided_0.025'] - 2.087) < 0.002, val
tk = np.arange(1, K + 1) / K
ALPHA = 0.05; GAMMA = 0.10; DELTA1 = 3.0
spend_e = 2 - 2 * stats.norm.cdf(stats.norm.isf(ALPHA / 2) / np.sqrt(tk))
spend_f = GAMMA * tk
inc_e = np.diff(np.concatenate([[0], spend_e])); inc_f = np.diff(np.concatenate([[0], spend_f]))
ce = gs_bounds(inc_e); cf = gs_bounds(inc_f)
u_z = ce / np.sqrt(np.arange(1, K + 1)); f_z = cf / np.sqrt(np.arange(1, K + 1))
chk_e = gs_exit(list(ce)); chk_f = gs_exit(list(cf))
assert abs(sum(chk_e) - ALPHA) < 1e-6 and abs(sum(chk_f) - GAMMA) < 1e-6
c_obf = obf_const(ALPHA, K)
val.update({'OBF_K10_one_sided_0.05_const': c_obf, 'efficacy_total_alpha_check': sum(chk_e), 'futility_total_gamma_check': sum(chk_f)})
res['boundary_validation'] = val
print('界線驗證', val, flush=True)


def to_t(z, scale=1.0):
    p = stats.norm.sf(z * scale)
    return stats.t.isf(p, LOOK_N - 1), p


# ---------------- 4. 序貫模擬
def seq_eval(mb, sb, d, ut, ft):
    m = mb + d / 1200; se = sb / np.sqrt(LOOK_N)
    t = m / se; t3 = (m - DELTA1 / 1200) / se
    eff = t >= ut; fut = t3 <= -ft
    prag = (m < 0) & (np.arange(1, K + 1) >= 5)[None, :]
    N = len(m); big = K + 1
    first = lambda a: np.where(a.any(1), a.argmax(1) + 1, big)
    fe, ff, fp = first(eff), first(fut), first(prag)
    fstop = np.minimum(ff, fp)
    fany = np.minimum(fe, fstop)
    typ = np.where(fany == big, 0, np.where(fe == fany, 1, np.where(ff == fany, 2, 3)))    # 同年優先序：有效 > 正式無效 > 務實停用
    m10 = m[:, -1] * 1200
    by_year = {str(k): dict(efficacy=float(((typ == 1) & (fany == k)).mean()), futility_formal=float(((typ == 2) & (fany == k)).mean()),
                            pragmatic_stop=float(((typ == 3) & (fany == k)).mean())) for k in range(1, K + 1)}
    und = typ == 0
    dec_year = np.where(und, K, fany)                          # 未判定者在第 10 年以終局規則處理
    out = dict(
        by_year_first_decision=by_year,
        p_efficacy_first=float((typ == 1).mean()), p_futility_formal_first=float((typ == 2).mean()), p_pragmatic_stop_first=float((typ == 3).mean()),
        p_undecided_at_10=float(und.mean()), p_terminal_owner_decides_m_ge3=float((und & (m10 >= DELTA1)).mean()), p_terminal_stop_m_lt3=float((und & (m10 < DELTA1)).mean()),
        p_any_stop_incl_terminal=float(((typ >= 2) | (und & (m10 < DELTA1))).mean()),
        p_efficacy_ever_nonbinding=float((fe <= K).mean()), p_futility_formal_ever=float((ff <= K).mean()), p_pragmatic_ever=float((fp <= K).mean()),
        p_efficacy_after_stop=float(((typ >= 2) & (fe <= K)).mean()),
        efficacy_year_median_given_efficacy=(float(np.median(fe[fe <= K])) if (fe <= K).any() else None),
        efficacy_by_year_cum_nonbinding={str(k): float((fe <= k).mean()) for k in range(1, K + 1)},
        any_early_decision_by_year_cum={str(k): float((fany <= k).mean()) for k in range(1, K + 1)},
        decision_year_median=float(np.median(dec_year)), decision_year_mean=float(dec_year.mean()),
        p_decision_by_year5=float((fany <= 5).mean()),
    )
    return out


ut, pe = to_t(u_z); ft, pf = to_t(f_z)
SEQ_DEL = [-5, 0, 3, 5, 8, 12]
size_e = seq_eval(*PATH['D1_main'], 0, ut, ft)['p_efficacy_ever_nonbinding']
size_f = seq_eval(*PATH['D1_main'], DELTA1, ut, ft)['p_futility_formal_ever']
cont = dict(sim_size_efficacy_D1_before=size_e, sim_size_futility_D1_before=size_f, efficacy_scale=1.0, futility_scale=1.0, triggered_efficacy=False, triggered_futility=False)
if size_e > 0.055:
    g = lambda c: seq_eval(*PATH['D1_main'], 0, to_t(u_z, c)[0], ft)['p_efficacy_ever_nonbinding'] - 0.05
    cont['efficacy_scale'] = float(optimize.brentq(g, 1.0, 2.0, xtol=1e-4)) + 1e-4; cont['triggered_efficacy'] = True
    u_z = u_z * cont['efficacy_scale']; ut, pe = to_t(u_z)
if size_f > 0.11:
    g = lambda c: seq_eval(*PATH['D1_main'], DELTA1, ut, to_t(f_z, c)[0])['p_futility_formal_ever'] - 0.10
    cont['futility_scale'] = float(optimize.brentq(g, 1.0, 2.0, xtol=1e-4)) + 1e-4; cont['triggered_futility'] = True
    f_z = f_z * cont['futility_scale']; ft, pf = to_t(f_z)
cont['sim_size_efficacy_D1_after'] = seq_eval(*PATH['D1_main'], 0, ut, ft)['p_efficacy_ever_nonbinding']
cont['sim_size_futility_D1_after'] = seq_eval(*PATH['D1_main'], DELTA1, ut, ft)['p_futility_formal_ever']
res['size_contingency'] = cont
print('size', cont, flush=True)

yrs = np.arange(1, K + 1)
bounds = []
for i in range(K):
    k = i + 1
    bounds.append(dict(
        look=k, review_month=f'{2026 + k}-10', n_months=int(12 * k), info_frac=k / K,
        efficacy=dict(alpha_spent_cum=float(spend_e[i]), z=float(u_z[i]), nominal_p=float(pe[i]), t=float(ut[i]),
                      m_pp_at_te_main=float(ut[i] * TE_MAIN / math.sqrt(k)), m_pp_at_te_oos=float(ut[i] * TE_OOS / math.sqrt(k)),
                      obf_classic_z=float(c_obf * math.sqrt(K / k))),
        futility=dict(gamma_spent_cum=float(spend_f[i]), z=float(-f_z[i]), nominal_p=float(pf[i]), t3=float(-ft[i]),
                      m_pp_at_te_main=float(DELTA1 - ft[i] * TE_MAIN / math.sqrt(k)), m_pp_at_te_oos=float(DELTA1 - ft[i] * TE_OOS / math.sqrt(k))),
        pragmatic_stop=('m < 0' if k >= 5 else None)))
res['boundaries'] = bounds
res['sequential_sim'] = {k: dict(te_pp=VAR[k]['te_pp'], L=VAR[k]['L'], by_delta={str(d): seq_eval(*PATH[k], d, ut, ft) for d in SEQ_DEL}) for k in PATH}
# 務實停用線的代價：不設務實停用線時的口徑（描述）
res['sequential_sim_note'] = '『p_efficacy_first』＝照規則（停用後不再判定）的檢定力；『p_efficacy_ever_nonbinding』＝紙上紀錄做滿 10 年、不論是否停用的口徑（型一誤差以此計）。'

# ---------------- 5. 貝氏後驗表（描述）
B = SPEC['part3_bayes']
PRI = {'main': (B['prior_main']['mu'], B['prior_main']['sd']), 'conservative': (B['prior_sens']['conservative']['mu'], B['prior_sens']['conservative']['sd']),
       'optimistic': (B['prior_sens']['optimistic']['mu'], B['prior_sens']['optimistic']['sd'])}


def post(mu0, s0, m, T, te):
    pr = 1 / s0 ** 2 + T / te ** 2; v = 1 / pr; mu = v * (mu0 / s0 ** 2 + m * T / te ** 2); s = math.sqrt(v)
    return dict(post_mean=mu, post_sd=s, p_gt0=float(stats.norm.sf(-mu / s)), p_gt3=float(stats.norm.sf((3 - mu) / s)))


bay = {}
for pn, (mu0, s0) in PRI.items():
    for tn, te in (('te_main', TE_MAIN), ('te_oos', TE_OOS)):
        bay[f'{pn}|{tn}'] = dict(prior=dict(mu=mu0, sd=s0, p_gt0=float(stats.norm.sf(-mu0 / s0)), p_gt3=float(stats.norm.sf((3 - mu0) / s0))), te_pp=te,
                                 table={str(T): {str(m): post(mu0, s0, m, T, te) for m in B['grid']['observed_m_pp']} for T in B['grid']['years']})
res['bayes'] = bay
res['bayes_breakeven'] = {f'{pn}': {str(T): dict(m_for_post_mean_0=float(-(mu0 / s0 ** 2) * TE_MAIN ** 2 / T), m_for_post_mean_1=float((1 * (1 / s0 ** 2 + T / TE_MAIN ** 2) - mu0 / s0 ** 2) * TE_MAIN ** 2 / T),
                                                     m_for_p_gt0_below_25pct=float(optimize.brentq(lambda m: post(mu0, s0, m, T, TE_MAIN)['p_gt0'] - 0.25, -300, 50)),
                                                     m_for_p_gt0_above_90pct=float(optimize.brentq(lambda m: post(mu0, s0, m, T, TE_MAIN)['p_gt0'] - 0.90, -50, 300)))
                                     for T in (1, 3, 5, 10)} for pn, (mu0, s0) in PRI.items()}

# ---------------- 6. 凍結計畫檔
plan = dict(
    plan='S69 前瞻驗證計畫（凍結版 v1）', spec_sha256=res['spec_sha256'], prereg_sha256=res['prereg_sha256'],
    start_date='2026-11-12', start_nav=1000000, benchmark='0050（含息）', series='網站正式產品實盤逐日淨值（策略、0050）',
    month_rule='日曆月；第 1 個月＝2026-11-12 開盤（100 萬）到 2026-11 最後交易日；之後以每月最後交易日淨值計',
    statistic='x＝策略月報酬 − 0050 月報酬；m＝12×平均(x)（個百分點）；t＝平均(x)/(標準差(x, ddof=1)/√n)；t3＝(平均(x) − 0.25 個百分點)/(同一標準誤)',
    delta1_pp=DELTA1, alpha_one_sided=ALPHA, gamma_futility=GAMMA, efficacy_spending='Lan-DeMets O\'Brien-Fleming 型', futility_spending='Kim-DeMets 線性（non-binding）',
    looks=[dict(look=b['look'], review_month=b['review_month'], n_months=b['n_months'], efficacy_t_min=b['efficacy']['t'], futility_t3_max=b['futility']['t3'],
                pragmatic_stop_if_m_lt0=(b['look'] >= 5)) for b in bounds],
    terminal='第 10 次仍未判定：m ≥ +3 → 擁有者決定（未證實、降低期望）；m < +3 → 停用改 0050',
    middle_zones='m ≥ +3 續做；0 ≤ m < 3 續做、降低期望；m < 0 且第 1～4 次 續做、降低期望並看 S27 警戒線',
    governance='自 2026-11-12 起算；本檔產出後不得更改，要改必須另題事前登記並保留本檔與 sha256；正式判定只在 10 個檢視日做；淨值須為含息總報酬口徑',
    planning_te_pp=TE_MAIN, prior_main=dict(mu=PRI['main'][0], sd=PRI['main'][1]))
ptxt = json.dumps(plan, ensure_ascii=False, indent=1) + '\n'
pp = H / 'PLAN_FROZEN.json'
if pp.exists():
    assert pp.read_text(encoding='utf-8') == ptxt, 'PLAN_FROZEN.json 已存在且內容不同：不得覆寫'
else:
    pp.write_text(ptxt, encoding='utf-8'); (H / 'PLAN_FROZEN.sha256').write_text(f'{sha(pp)}  PLAN_FROZEN.json\n')
res['plan_frozen_sha256'] = sha(pp)
res['run_sha256'] = sha(__file__)
res['runtime_s'] = round(time.time() - T0, 1); res['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('完成', res['runtime_s'], 's', res['maxrss_mb'], 'MB', flush=True)

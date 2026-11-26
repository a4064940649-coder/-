#!/usr/bin/env python3
"""S98 任務 B：十年實盤驗證檢定力（常態＋區塊拔靴）、S69 規則模擬、大贏家數量不確定性。用法：python3 -I -B run_b.py（先跑 run_a.py）"""
import json, math, time, resource, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize

HERE = Path(__file__).resolve().parent
AG = HERE.parent
ROOT = AG.parent.parent.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0]
PR = json.loads((HERE / 'PREREG.json').read_text())['task_B']
SEEDS = PR['seeds']; NSIM = PR['n_sim']; IRS = PR['IR_grid']
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
PLAN = json.loads((AG / 's69_forward_validation_plan/PLAN_FROZEN.json').read_text())
assert sha(AG / 's69_forward_validation_plan/PLAN_FROZEN.json') == (AG / 's69_forward_validation_plan/PLAN_FROZEN.sha256').read_text().split()[0]
EFF = np.array([l['efficacy_t_min'] for l in PLAN['looks']]); FUT = np.array([l['futility_t3_max'] for l in PLAN['looks']])
PRAG = np.array([l['pragmatic_stop_if_m_lt0'] for l in PLAN['looks']]); D1 = PLAN['delta1_pp'] / 1200
K = 10; LOOK_N = np.arange(1, K + 1) * 12; N = 120

# ---------- 歷史月超額
off = pd.read_csv(ROOT / 'xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)
me = off.groupby(off.index.to_period('M')).tail(1); mm = pd.concat([off.iloc[[0]], me[me.index > off.index[0]]])
r = mm[['E0|W10_70', '0050']].pct_change().iloc[1:]
x_hist = (r['E0|W10_70'] - r['0050']).to_numpy(); rb_hist = r['0050'].to_numpy(); assert len(x_hist) == 136
RA = json.loads((HERE / 'results_a.json').read_text())
TE = RA['series']['W_official']['te_pp'] / 100          # 年化
assert abs(TE - x_hist.std(ddof=1) * math.sqrt(12)) < 1e-9
e_hist = x_hist - x_hist.mean()
print('TE', TE, 'hist IR', RA['series']['W_official']['IR'], flush=True)


def block_idx(seed, L, M=136):
    rng = np.random.default_rng(seed); nb = math.ceil(N / L)
    st = rng.integers(0, M, size=(NSIM, nb))
    return ((st[:, :, None] + np.arange(L)[None, None, :]) % M).reshape(NSIM, nb * L)[:, :N]


def paths(kind, seed, scale=1.0, L=6):
    """回傳零平均的 NSIM×120 月超額（年化 TE＝TE×scale）"""
    if kind == 'normal':
        return np.random.default_rng(seed).standard_normal((NSIM, N)) * (TE * scale / math.sqrt(12)), None
    idx = block_idx(seed, L)
    return e_hist[idx] * scale, idx


def summ(X):
    c1 = np.cumsum(X, 1)[:, LOOK_N - 1]; c2 = np.cumsum(X * X, 1)[:, LOOK_N - 1]
    mean = c1 / LOOK_N; sd = np.sqrt(np.maximum(c2 - c1 * c1 / LOOK_N, 0) / (LOOK_N - 1))
    return mean, sd


def rules_main(mean, sd):
    """PLAN_FROZEN 逐條：回傳各模擬的結局型別與發生的檢視次：1 通過、2 正式無效、3 務實停用、4 第 10 次 m<3 停用、5 擁有者決定"""
    se = sd / np.sqrt(LOOK_N); t = mean / se; t3 = (mean - D1) / se
    typ = np.zeros(len(mean), int); when = np.full(len(mean), 99)
    for k in range(K):
        live = typ == 0
        a = live & (t[:, k] >= EFF[k]); typ[a] = 1; when[a] = k + 1; live &= ~a
        b = live & (t3[:, k] <= FUT[k]); typ[b] = 2; when[b] = k + 1; live &= ~b
        if PRAG[k]:
            c = live & (mean[:, k] < 0); typ[c] = 3; when[c] = k + 1; live &= ~c
    live = typ == 0
    d = live & (mean[:, K - 1] < D1); typ[d] = 4; when[d] = 10
    typ[typ == 0] = 5; when[when == 99] = 10
    return typ, when


def rules_alt(mean, sd):
    """README 文字版：無提前有效；1～4 年 t3 界線；5～9 年 m<0 停用；第 10 年 m<3 停用、3≤m<9 未定、m≥9 通過"""
    se = sd / np.sqrt(LOOK_N); t3 = (mean - D1) / se
    typ = np.zeros(len(mean), int); when = np.full(len(mean), 99)
    for k in range(K - 1):
        live = typ == 0
        if k < 4: b = live & (t3[:, k] <= FUT[k]); typ[b] = 2; when[b] = k + 1
        else: c = live & (mean[:, k] < 0); typ[c] = 3; when[c] = k + 1
    live = typ == 0; m10 = mean[:, K - 1] * 1200
    typ[live & (m10 < 3)] = 4; typ[live & (m10 >= 9)] = 1; typ[live & (m10 >= 3) & (m10 < 9)] = 5; when[when == 99] = 10
    return typ, when


def status_table(typ, when, years=(1, 3, 5, 10)):
    out = {}
    for y in years:
        p = float(((typ == 1) & (when <= y)).mean()); s = float(((typ >= 2) & (typ <= 4) & (when <= y)).mean())
        out[str(y)] = dict(passed=p, stopped=s, undecided=1 - p - s)
    out['detail_10y'] = dict(pass_efficacy=float((typ == 1).mean()), stop_formal_futility=float((typ == 2).mean()), stop_pragmatic_m_lt0=float((typ == 3).mean()),
                             stop_year10_m_lt3=float((typ == 4).mean()), owner_decides=float((typ == 5).mean()),
                             pass_by_look={str(k): float(((typ == 1) & (when == k)).mean()) for k in range(1, 11)},
                             stop_by_look={str(k): float(((typ >= 2) & (typ <= 4) & (when == k)).mean()) for k in range(1, 11)})
    return out


# ---------- 0. 驗證：重現 S69 結局表（區塊拔靴、真實 0／+5／+12 個百分點）
Xv, _ = paths('boot', 20261118, 1.0, 6)
mv, sv = summ(Xv)
S69 = {0: (1.0, 81.4, 17.6), 5: (13.1, 42.3, 44.6), 12: (79.1, 3.2, 17.7)}
val = {}
for d, ref in S69.items():
    typ, when = rules_main(mv + d / 1200, sv)
    mine = (float((typ == 1).mean()) * 100, float(((typ >= 2) & (typ <= 4)).mean()) * 100, float((typ == 5).mean()) * 100)
    val[str(d)] = dict(mine=mine, s69=ref, max_abs_diff_pp=max(abs(a - b) for a, b in zip(mine, ref)))
    assert val[str(d)]['max_abs_diff_pp'] <= 1.5, val
print('驗證 S69', val, flush=True)

# ---------- 1. 主模擬
CONFIG = {'normal_TE100': ('normal', SEEDS['normal'], 1.0, 6), 'bootstrap_TE100': ('boot', SEEDS['bootstrap'], 1.0, 6),
          'normal_TE125': ('normal', SEEDS['normal_te125'], 1.25, 6), 'bootstrap_TE125': ('boot', SEEDS['bootstrap_te125'], 1.25, 6),
          'bootstrap_TE100_L1': ('boot', SEEDS['bootstrap_L1'], 1.0, 1), 'bootstrap_TE100_L12': ('boot', SEEDS['bootstrap_L12'], 1.0, 12)}
YI = {1: 0, 3: 2, 5: 4, 10: 9}
SIM = {}
for name, (kind, seed, scale, L) in CONFIG.items():
    X, idx = paths(kind, seed, scale, L); mean0, sd = summ(X); te = TE * scale
    out = dict(te_pp=te * 100, kind=kind, L=L if kind == 'boot' else None, seed=seed, by_IR={})
    for ir in IRS:
        mu = ir * te / 12; mean = mean0 + mu
        row = dict(true_ann_excess_pp=ir * te * 100, looks={})
        for y, j in YI.items():
            n = 12 * y; t = mean[:, j] / (sd[:, j] / math.sqrt(n))
            row['looks'][str(y)] = dict(p_cum_excess_pos=float((mean[:, j] > 0).mean()), p_sig_t165=float((t >= 1.65).mean()), p_sig_tcrit=float((t >= stats.t.isf(0.05, n - 1)).mean()),
                                        obs_ann_excess_pp_p05_p50_p95=[float(np.quantile(mean[:, j], q) * 1200) for q in (0.05, 0.5, 0.95)])
        typ, when = rules_main(mean, sd); row['S69_main'] = status_table(typ, when)
        typ2, when2 = rules_alt(mean, sd); row['S69_alt_readme_literal'] = status_table(typ2, when2)
        out['by_IR'][str(ir)] = row
    SIM[name] = out
    print(name, {ir: (round(v['looks']['10']['p_cum_excess_pos'], 3), round(v['looks']['10']['p_sig_t165'], 3), {k: round(x, 3) for k, x in v['S69_main']['10'].items()}) for ir, v in out['by_IR'].items()}, f'{time.time()-t0:.0f}s', rss(), flush=True)
    del X

# ---------- 1b. 成對拔靴：複利淨值「策略期末 > 0050」
idx = block_idx(SEEDS['pairs_nav'], 6)
E = e_hist[idx]; RB = rb_hist[idx]
pairs = {}
for ir in IRS:
    X = E + ir * TE / 12; ls = np.cumsum(np.log1p(RB + X), 1)[:, LOOK_N - 1]; lb = np.cumsum(np.log1p(RB), 1)[:, LOOK_N - 1]
    c1 = np.cumsum(X, 1)[:, LOOK_N - 1]
    pairs[str(ir)] = {str(y): dict(p_nav_strategy_gt_0050=float((ls[:, j] > lb[:, j]).mean()), p_sum_x_pos=float((c1[:, j] > 0).mean()),
                                  p_disagree=float(((ls[:, j] > lb[:, j]) != (c1[:, j] > 0)).mean())) for y, j in YI.items()}
print('pairs', {ir: (round(v['10']['p_nav_strategy_gt_0050'], 3), round(v['10']['p_sum_x_pos'], 3)) for ir, v in pairs.items()}, flush=True)
del E, RB, idx

# ---------- 2. 80% 檢定力所需年數（單次單尾 5% t 檢定，iid 常態；非中心 t）
def power_t(ir, years):
    n = int(round(12 * years)); nc = ir / math.sqrt(12) * math.sqrt(n)
    return float(stats.nct.sf(stats.t.isf(0.05, n - 1), n - 1, nc))


def years_for(ir, target):
    if ir <= 0: return None
    y = 1
    while power_t(ir, y) < target and y < 2000: y += 1
    return y


zz = stats.norm.isf(0.05) + stats.norm.isf(0.2)
power80 = {str(ir): dict(years_80_exact_nct=years_for(ir, 0.8), years_80_normal_approx=(zz / ir) ** 2 if ir > 0 else None, years_50_exact_nct=years_for(ir, 0.5),
                         power_at_10y=power_t(ir, 10), power_at_20y=power_t(ir, 20)) for ir in IRS}
power80['IR_needed_for_80pct_power_at_10y_simple_t'] = float(optimize.brentq(lambda q: power_t(q, 10) - 0.8, 0.05, 3))
# 十年內照 PLAN「通過」達 80%／50% 所需 IR（常態、拔靴）
for nm, (kind, seed) in {'normal': ('normal', SEEDS['normal']), 'bootstrap': ('boot', SEEDS['bootstrap'])}.items():
    X, _ = paths(kind, seed, 1.0, 6); m0, s0 = summ(X)
    f = lambda q, tg: float((rules_main(m0 + q * TE / 12, s0)[0] == 1).mean()) - tg
    power80[f'IR_needed_PLAN_pass_80pct_{nm}'] = float(optimize.brentq(f, 0.0, 3.0, args=(0.8,), xtol=1e-3))
    power80[f'IR_needed_PLAN_pass_50pct_{nm}'] = float(optimize.brentq(f, 0.0, 3.0, args=(0.5,), xtol=1e-3))
    del X
print('power80', json.dumps(power80, default=float)[:900], flush=True)

# ---------- 3. 大贏家
TR = pd.read_csv(AG / 's75_continuous_2008_2026/v3_2026-11-25_limit7/trades_v3.csv.gz', parse_dates=['ent_date', 'exit_date', 'sig_date'])
TR = TR[TR.version == 'V_BRIDGE'].reset_index(drop=True)
v3 = pd.read_csv(AG / 's75_continuous_2008_2026/v3_2026-11-25_limit7/nav_2008_2026_v3.csv.gz', index_col=0, parse_dates=True)
THR = 0.576
TR['big'] = TR.ret >= THR
TR['ey'] = TR.ent_date.dt.year; TR['xy'] = TR.exit_date.dt.year
years = list(range(2008, 2027))
cnt_e = TR[TR.big].groupby('ey').size().reindex(years, fill_value=0); cnt_x = TR[TR.big].groupby('xy').size().reindex(years, fill_value=0)
ntr = TR.groupby('ey').size().reindex(years, fill_value=0)


def cstat(c):
    c = np.asarray(c, float); m = c.mean(); v = c.var(ddof=1)
    out = dict(years=len(c), mean=float(m), var=float(v), fano=float(v / m), min=int(c.min()), max=int(c.max()), median=float(np.median(c)), zero_years=int((c == 0).sum()),
               p0_poisson=float(math.exp(-m)))
    if v > m: k = m * m / (v - m); out.update(nb_k=float(k), p0_negbin=float((k / (k + m)) ** k))
    return out


nav_b = v3['V_BRIDGE']; nav_0 = v3['0050']
TR['nav_exit'] = nav_b.reindex(TR.exit_date).to_numpy(); TR['nav_ent'] = nav_b.reindex(TR.ent_date).to_numpy()
TR['r0050'] = nav_0.reindex(TR.exit_date).to_numpy() / nav_0.reindex(TR.ent_date).to_numpy() - 1
assert TR[['nav_exit', 'nav_ent', 'r0050']].notna().all().all()
TR['w_ent'] = TR.cost / TR.nav_ent
big = TR[TR.big].copy(); non = TR[~TR.big]
winners = dict(threshold=THR, n_trades=int(len(TR)), n_big=int(len(big)), by_entry_year=cnt_e.to_dict(), by_exit_year=cnt_x.to_dict(), trades_by_entry_year=ntr.to_dict(),
               stats_entry_year_all=cstat(cnt_e), stats_entry_2008_2018=cstat(cnt_e.loc[2008:2018]), stats_entry_2019_2026=cstat(cnt_e.loc[2019:2026]),
               stats_entry_2016_2026=cstat(cnt_e.loc[2016:2026]), stats_exit_year_all=cstat(cnt_x),
               big_ret_mean=float(big.ret.mean()), big_ret_median=float(big.ret.median()), big_weight_at_entry_mean=float(big.w_ent.mean()),
               big_share_of_trade_pnl=float(big.pnl.sum() / TR.pnl.sum()), nonbig_ret_mean=float(non.ret.mean()), big_hold_days_median=float(big.hold_days.median()),
               big_r0050_same_period_mean=float(big.r0050.mean()))
print('winners', json.dumps({k: v for k, v in winners.items() if 'stats' in k or k in ('n_big', 'by_entry_year', 'big_share_of_trade_pnl', 'big_ret_mean', 'big_weight_at_entry_mean')}, default=float), flush=True)

END_NAV = float(nav_b.iloc[-1]); YRS_FULL = (nav_b.index[-1] - nav_b.index[0]).days / 365.25
cagr = lambda end, start, yrs: (end / start) ** (1 / yrs) - 1
base_full = cagr(END_NAV, float(nav_b.iloc[0]), YRS_FULL); b0_full = cagr(float(nav_0.iloc[-1]), float(nav_0.iloc[0]), YRS_FULL)
d0 = pd.Timestamp('2015-07-01'); n15 = float(nav_b.asof(d0)); yrs15 = (nav_b.index[-1] - d0).days / 365.25
base_15 = cagr(END_NAV, n15, yrs15); b0_15 = cagr(float(nav_0.iloc[-1]), float(nav_0.asof(d0)), yrs15)
d19 = pd.Timestamp('2019-01-01'); n19 = float(nav_b.asof(d19)); yrs19 = (nav_b.index[-1] - d19).days / 365.25; base_19 = cagr(END_NAV, n19, yrs19)

rng = np.random.default_rng(SEEDS['winners'])
bi = big.index.to_numpy(); nb_ = len(bi)
loss_R1 = ((big.pnl - big.cost * big.r0050) / big.nav_exit).to_numpy()          # 改放 0050
pool = {y: non[non.ey == y].ret.to_numpy() for y in years}
drop = rng.random((NSIM, nb_)) < 0.5
repl_ret = np.column_stack([rng.choice(pool[y], size=NSIM) for y in big.ey])       # 同年隨機非大贏家
loss_R2 = (big.pnl.to_numpy()[None, :] - big.cost.to_numpy()[None, :] * repl_ret) / big.nav_exit.to_numpy()[None, :]
m15 = (big.ent_date >= d0).to_numpy(); m19 = (big.ent_date >= d19).to_numpy()


def cagr_after(lossmat, dropmat, mask, base_ratio, yrs):
    f = np.where(dropmat & mask[None, :], 1 - lossmat, 1.0).prod(1)
    return (base_ratio * f) ** (1 / yrs) - 1


def q(a): return dict(mean=float(a.mean() * 100), p05=float(np.quantile(a, .05) * 100), p50=float(np.median(a) * 100), p95=float(np.quantile(a, .95) * 100))


allm = np.ones(nb_, bool); alld = np.ones((1, nb_), bool)
halving = {}
for tag, mask, ratio, yrs, base in (('full_2008_2026', allm, END_NAV / float(nav_b.iloc[0]), YRS_FULL, base_full), ('since_2015_07', m15, END_NAV / n15, yrs15, base_15), ('since_2019', m19, END_NAV / n19, yrs19, base_19)):
    L1 = np.broadcast_to(loss_R1, (NSIM, nb_))
    h1 = cagr_after(L1, drop, mask, ratio, yrs) - base; h2 = cagr_after(loss_R2, drop, mask, ratio, yrs) - base
    a1 = cagr_after(loss_R1[None, :], alld, mask, ratio, yrs)[0]; a2 = cagr_after(loss_R2, np.ones((NSIM, nb_), bool), mask, ratio, yrs)
    halving[tag] = dict(base_cagr_pp=base * 100, n_big_in_window=int(mask.sum()), big_per_year=float(mask.sum() / yrs),
                        half_R1_0050_delta_cagr_pp=q(h1), half_R2_random_nonbig_delta_cagr_pp=q(h2),
                        all_removed_R1_cagr_pp=float(a1 * 100), all_removed_R1_delta_pp=float((a1 - base) * 100), all_removed_R2_delta_pp=q(a2 - base))
halving['benchmarks'] = dict(cagr_0050_full_pp=b0_full * 100, cagr_0050_since_2015_07_pp=b0_15 * 100, s82_all_removed_cagr_pp_reference=13.4, s82_base_reference=23.5)
print('halving', json.dumps(halving, default=float), flush=True)

res = dict(prereg_sha256=sha(HERE / 'PREREG.json'), plan_frozen_sha256=sha(AG / 's69_forward_validation_plan/PLAN_FROZEN.json'), TE_pp=TE * 100, hist_IR=RA['series']['W_official']['IR'],
           n_sim=NSIM, validation_vs_S69=val, sim=SIM, pairs_compound_nav=pairs, power80=power80, winners=winners, winners_halving=halving,
           runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'results_b.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')

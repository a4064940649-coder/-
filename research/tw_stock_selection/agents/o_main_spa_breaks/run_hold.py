#!/usr/bin/env python3
"""步驟 5：2024～2026 的落後（PREREG 後才可執行；只跑一次）。
(a) 預測分布：用 2006-04～2023-12 的 AB_V4 對 0050 窗超額做平穩區塊 bootstrap（L＝3/6/12，主 6），
    抽出與保留期相同窗數的路徑，統計量＝Σ log((1+r_main)/(1+r_0050))（主）與平均算術超額（副）；
    p_lower＝模擬值 ≤ 實際值的比例；判斷是否落在 10～90% 區間。
(b) MDE：80% 檢定力、α=0.05 單尾，「保留期均值比 2006～2023 低」的最小可偵測下降（HAC 與 bootstrap 兩法），
    以及只靠保留期 32 窗本身的 MDE。
(c) 落後拆解：r_main − r_0050 ＝ (r_main − r_ex) − w·(r2330 − r_ex)；台積電缺口 G＝Σ w(r2330 − r_ex)，
    選股部分 A＝Σ (r_main − r_ex)；log 版同理（以 0050 相對財富）。集中度因子回歸：x50_net ~ CF（HAC）。
(d) 同樣的預測分布也對「對同池等權」與「五分位價差」做（判斷選股力本身是否掉出預測區間）。
寫入 RESULTS_HOLD。"""
import sys, math, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C, breaks_lib as BL, spa_lib as S
RES = C.check_prereg()
PR = RES['PREREG']; HP = PR['hold']
MAIN_ID = PR['main_id']
T0 = time.time()

def sb_paths(n_src, n_out, L, Bn, seed):
    rng = np.random.default_rng(seed); p = 1 / L
    idx = np.empty((Bn, n_out), np.int64); idx[:, 0] = rng.integers(0, n_src, Bn)
    for t in range(1, n_out):
        new = rng.random(Bn) < p
        idx[:, t] = np.where(new, rng.integers(0, n_src, Bn), (idx[:, t - 1] + 1) % n_src)
    return idx

def hac_ols(y, x):
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    n = len(y); lag = BL.nw_lag(n); Xe = X * e[:, None]
    Sg = Xe.T @ Xe / n
    for i in range(1, lag + 1):
        G = Xe[i:].T @ Xe[:-i] / n; Sg += (1 - i / (lag + 1)) * (G + G.T)
    Qi = np.linalg.inv(X.T @ X / n); V = Qi @ Sg @ Qi / n
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return {'alpha_pp': float(b[0] * 100), 't_alpha': float(b[0] / math.sqrt(V[0, 0])), 'beta': float(b[1]), 't_beta': float(b[1] / math.sqrt(V[1, 1])), 'R2': float(r2), 'n': int(n)}

def main():
    D = pd.read_parquet(C.DATA / 'family_excess.parquet'); Dew = pd.read_parquet(C.DATA / 'family_excess_ew.parquet')
    B = pd.read_parquet(C.DATA / 'bench.parquet'); BI = pd.read_parquet(C.DATA / 'break_inputs.parquet')
    T = pd.read_parquet(C.DATA / 'tsmc_inputs.parquet')
    F = pd.read_parquet(C.DATA / 'family_ret.parquet')
    rm = F[MAIN_ID]; r50 = B['r0050']
    pre = C.wmask(rm.index, *HP['pre']); hold = C.wmask(rm.index, *HP['hold'])
    nh = int(hold.sum()); out = {'n_pre': int(pre.sum()), 'n_hold': nh}
    ser = {'x50_net': (rm, r50), 'xEW_net': (rm, B['rEW_ABV4']), 'Q5mQ1_gross': (BI['Q5|zero'], BI['Q1|zero'])}
    pdist = {}
    for name, (ra, rb) in ser.items():
        lx = np.log1p(ra) - np.log1p(rb); ax = ra - rb
        act = {'cumlog': float(lx[hold].sum()), 'mean_arith_pp': float(ax[hold].mean() * 100)}
        src_l = lx[pre].to_numpy(); src_a = ax[pre].to_numpy()
        res = {'actual': act, 'pre_mean_arith_pp': float(src_a.mean() * 100), 'pre_cumlog_per_window': float(src_l.mean())}
        for L in HP['block_lengths']:
            idx = sb_paths(len(src_l), nh, L, HP['B'], HP['seed'] + L)
            sl = src_l[idx].sum(axis=1); sa = src_a[idx].mean(axis=1)
            res[f'L{L}'] = {'p_lower_cumlog': float((sl <= act['cumlog']).mean()), 'p_lower_mean': float((sa <= act['mean_arith_pp'] / 100).mean()),
                            'q10_cumlog': float(np.quantile(sl, 0.10)), 'q50_cumlog': float(np.quantile(sl, 0.50)), 'q90_cumlog': float(np.quantile(sl, 0.90)),
                            'q10_mean_pp': float(np.quantile(sa, 0.10) * 100), 'q90_mean_pp': float(np.quantile(sa, 0.90) * 100)}
            res[f'L{L}']['inside_10_90_cumlog'] = bool(res[f'L{L}']['q10_cumlog'] <= act['cumlog'] <= res[f'L{L}']['q90_cumlog'])
        # MDE（均值下降）
        a_pre, a_h = src_a, ax[hold].to_numpy()
        om_pre = BL.lrv1(a_pre - a_pre.mean()); om_h = BL.lrv1(a_h - a_h.mean())
        z = stats.norm.ppf(0.95) + stats.norm.ppf(0.80)
        res['MDE80_one_sided_pp'] = {'two_sample_HAC': float(z * math.sqrt(om_pre / len(a_pre) + om_h / len(a_h)) * 100),
                                     'hold_only_HAC': float(z * math.sqrt(om_h / len(a_h)) * 100),
                                     'hold_only_iid': float(z * a_h.std(ddof=1) / math.sqrt(len(a_h)) * 100)}
        # bootstrap 版 MDE：從挑選＋驗證期抽 32 窗，平均下降 d 時被判「低於 2006～2023 的 5% 分位」的機率 = 80% 的 d
        idx = sb_paths(len(src_a), nh, 6, HP['B'], HP['seed'] + 99)
        sa = src_a[idx].mean(axis=1); q05 = np.quantile(sa, 0.05)
        res['MDE80_boot_pp'] = float((np.quantile(sa, 0.80) - q05) * 100)
        res['diff_hold_minus_pre_pp'] = float((a_h.mean() - a_pre.mean()) * 100)
        pdist[name] = res
    out['predictive'] = pdist
    # 落後拆解
    h = hold
    w = T['w2330_used']; rex = T['r_ex']; r23 = T['r2330']
    A = (rm - rex)[h]; G = (w * (r23 - rex))[h]; X50 = (rm - r50)[h]
    lw_main = float(np.log1p(rm[h]).sum()); lw_50 = float(np.log1p(r50[h]).sum()); lw_ex = float(np.log1p(rex[h]).sum())
    out['decomp'] = {
        'arith_sum_pp': {'excess_vs_0050': float(X50.sum() * 100), 'A_vs_ex_tsmc_0050': float(A.sum() * 100), 'G_tsmc_gap': float(G.sum() * 100),
                         'identity_check': float((A.sum() - G.sum() - X50.sum()) * 100)},
        'log_wealth': {'main': lw_main, '0050': lw_50, '0050_ex_tsmc': lw_ex, 'shortfall_vs_0050': float(lw_50 - lw_main),
                       'main_vs_ex_tsmc': float(lw_main - lw_ex), 'tsmc_gap_log': float(lw_50 - lw_ex)},
        'cagr': {'main': float(math.exp(lw_main / (nh / 12)) - 1), '0050': float(math.exp(lw_50 / (nh / 12)) - 1), '0050_ex_tsmc': float(math.exp(lw_ex / (nh / 12)) - 1)},
        'w2330_mean_hold': float(w[h].mean()), 'w2330_source': 'actual（2025-10 起）＋近似校準（之前）',
        'r2330_minus_0050_mean_pp': float((r23 - r50)[h].mean() * 100), 't_2330_vs_0050_hold': C.tstat((r23 - r50)[h]),
        't_2330_vs_0050_all': C.tstat((r23 - r50)[C.wmask(r50.index, *C.WINDOWS['W_long'])]),
    }
    sf = out['decomp']['log_wealth']['shortfall_vs_0050']; gap = out['decomp']['log_wealth']['tsmc_gap_log']
    out['decomp']['share_explained_by_tsmc_gap'] = float(gap / sf) if sf > 0 else None
    # 集中度因子回歸
    x50 = (rm - r50); cf = T['CF']
    allm = C.wmask(x50.index, *C.WINDOWS['W_long'])
    regs = {}
    for lab, m in (('ALL', allm), ('PRE', pre), ('HOLD', h)):
        ok = m & x50.notna() & cf.notna()
        regs[lab] = hac_ols(x50[ok].to_numpy(), cf[ok].to_numpy())
        regs[lab]['mean_CF_pp'] = float(cf[ok].mean() * 100); regs[lab]['mean_x50_pp'] = float(x50[ok].mean() * 100)
    # 保留期：用全期 β 預測的超額缺口 = β_ALL × mean(CF_hold)
    regs['HOLD_explained_by_CF_pp'] = float(regs['ALL']['beta'] * regs['HOLD']['mean_CF_pp'])
    out['concentration_regression'] = regs
    C.save_key('RESULTS_HOLD', out)
    print(f'[{time.time() - T0:.0f}s] done', {k: v['L6'] for k, v in pdist.items()}, out['decomp'], flush=True)

if __name__ == '__main__':
    main()

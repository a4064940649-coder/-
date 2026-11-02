#!/usr/bin/env python3
"""步驟 6（樣本外）：凍結規則的偽前瞻。
- 起點 2019-05（驗證期）：先驗＝CV_SEL（挑選期 CV 選版、s、k）＋族群 SEL 窗；σ＝SEL 的 HAC sd（當時只知道這些）。
- 起點 2024-01（保留期）：先驗＝CV_PRE2024＋族群 < 2024-01 窗；σ＝max(HAC <2024, HAC 2019-05～2023-12)。
- 規則＝OC_FINAL 選定者（z 單位門檻與 σ 無關，直接沿用）；決策線＝主格 j 中位數（一般成本）；壓力成本線另跑。
- 配置政策：P30＝起始主策略 30%（其餘 0050），升級→100%，下修→0%；P100＝起始 100%，下修→0%。決策在第 n 個前瞻月資料完成後，
  從錨定月 n＋2 生效（JMED(m) 要到 m＋1 月下旬才定稿）。主策略月報酬＝0050 官方窗報酬＋j 中位數超額（近似）。
- 對手：0050、同池等權（V7 AB_V7_B10|EW）、主策略 100%（j 中位數可執行）、主策略官方日零衝擊（AB_V7_B10）。
輸出 results.json 的 PSEUDO_FORWARD。用法：python3 -I -B s6_pseudo_forward.py"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fslib as F, seq2 as Q, prior as PR
from s5_oc_final import decision_line


def stats_(r, r50):
    r = np.asarray(r, float); r50 = np.asarray(r50, float); n = len(r)
    nav = np.cumprod(1 + r); mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1))
    ex = (r - r50) * 100
    return dict(n=n, ann=float(nav[-1] ** (12 / n) - 1) * 100, mdd=mdd * 100, ex_mean_pp=float(ex.mean()),
                t_vs_0050=float(ex.mean() / ex.std(ddof=1) * math.sqrt(n)) if ex.std(ddof=1) > 0 else 0.0)


def run_rule(x, sigma, prior, th, par, rule):
    X = np.asarray(x, float)[None, :]
    up, dn = Q.raw_triggers(rule, X, sigma, par, prior, th)
    u, d = Q.first_events(up, dn)
    z = np.cumsum(X[0]) / (sigma * np.sqrt(np.arange(1, X.shape[1] + 1)))
    pp = Q.post_probs(X, sigma, prior, th, list(range(1, X.shape[1] + 1)), par['alpha_min'])
    return int(u[0]), int(d[0]), z, {n: (float(v[0][0]), float(v[1][0])) for n, v in pp.items()}


def policy(x0, up_m, dn_m, n):
    """回傳長度 n 的主策略權重序列（決策月 k → 從第 k＋2 月生效）。"""
    w = np.full(n, x0)
    if dn_m: w[min(dn_m + 2, n):] = 0.0
    if up_m: w[min(up_m + 2, n):] = 1.0
    return w


def main():
    PRE = F.check_prereg('PREREG'); cfg = PRE['config']
    res = F.load_res(); oc = res['OC_FINAL']; rule = oc['chosen'] or 'R2'
    par = dict(oc['params'])
    X, lab, med, anc = F.family()
    line = decision_line(); line_s = decision_line('stress')
    O = F.official_series()                                   # 零衝擊 AB_V7_B10 官方窗：r、r50
    EW = F.official_series('AB_V7_B10|EW')
    O.index = pd.PeriodIndex(O.index, freq='M'); EW.index = pd.PeriodIndex(EW.index, freq='M')
    out = {'rule': rule, 'params': par}
    starts = {
        'VAL_2019-05': dict(start=F.VAL[0], cv=res['CV_SEL'], est=F.per_mask(anc, None, F.SEL_END), sig_pre=F.SEL_END),
        'HOLD_2024-01': dict(start=F.HOLD[0], cv=res['CV_PRE2024'], est=F.per_mask(anc, None, pd.Period('2023-12', 'M')), sig_pre=pd.Period('2023-12', 'M')),
    }
    for nm, s in starts.items():
        hist = line[line.index <= s['sig_pre']]
        sig = F.hac_sd(hist)
        if nm.startswith('HOLD'): sig = max(sig, F.hac_sd(hist[hist.index >= F.VAL[0]]))
        sig *= cfg['sig_mult']
        b = PR.build(s['est'], s['cv']['selected'], s['cv']['s'], s['cv']['k'], hist)
        fwd = line[line.index >= s['start']]; fwd_s = line_s.reindex(fwd.index)
        u, d, z, pp = run_rule(fwd.to_numpy(), sig, b['prior'], b['th'], par, rule)
        n = len(fwd)
        idx = fwd.index
        r50 = O.r50.reindex(idx).to_numpy()
        rm = r50 + fwd.to_numpy() / 100; rms = r50 + fwd_s.to_numpy() / 100
        res_p = {}
        for pname, x0 in (('P30', cfg['policy_x0']), ('P100', 1.0)):
            w = policy(x0, u, d, n)
            rp = w * rm + (1 - w) * r50; rps = w * rms + (1 - w) * r50
            res_p[pname] = dict(normal=stats_(rp, r50), stress=stats_(rps, r50), vs_main_t=float(((rp - rm) * 100).mean() / max(((rp - rm) * 100).std(ddof=1), 1e-12) * math.sqrt(n)))
        comp = dict(E50=stats_(r50, r50), MAIN_JMED_EXEC=stats_(rm, r50), MAIN_JMED_EXEC_STRESS=stats_(rms, r50),
                    MAIN_OFFICIAL_ZERO=stats_(O.r.reindex(idx).to_numpy(), r50), EW_POOL=stats_(EW.r.reindex(idx).to_numpy(), r50))
        yearly = {}
        for y in sorted(set(p.year for p in idx)):
            m = np.array([p.year == y for p in idx])
            yearly[y] = dict(n=int(m.sum()), jmed_ex_mean=float(fwd.to_numpy()[m].mean()), cum_z_end=float(z[np.where(m)[0][-1]]))
        big = int(np.argmax(np.abs(fwd.to_numpy())))
        x_drop = np.delete(fwd.to_numpy(), big)
        out[nm] = dict(n_months=n, sigma=sig, prior_cal=b['cal'], prior_raw=b['raw'], cv=dict(selected=s['cv']['selected'], s=s['cv']['s'], k=s['cv']['k']),
                       first_up_month=u, first_down_month=d, first_up_anchor=str(idx[u - 1]) if u else None, first_down_anchor=str(idx[d - 1]) if d else None,
                       z_end=float(z[-1]), z_at_36=float(z[35]) if n >= 36 else None, P_gt0_end=1 - pp[n][1], P_gt_min_end=pp[n][0],
                       ex_mean=float(fwd.mean()), ex_sd=float(fwd.std(ddof=1)), ex_hac_sd=F.hac_sd(fwd), ex_mean_stress=float(fwd_s.mean()),
                       z_end_drop_largest=float(x_drop.sum() / (sig * math.sqrt(len(x_drop)))), largest_month=str(idx[big]), largest_value=float(fwd.iloc[big]),
                       policies=res_p, comparators=comp, yearly=yearly,
                       z_path={str(idx[i]): float(z[i]) for i in range(n)})
        print(nm, 'n', n, 'sig', round(sig, 3), 'prior', {k: round(v, 3) for k, v in b['cal'].items() if k in ('mean', 'sd', 'p_pos')},
              'up', u, 'down', d, 'z_end', round(z[-1], 3), 'mean', round(fwd.mean(), 3))
        for k, v in comp.items(): print('  ', k, {a: round(b_, 2) for a, b_ in v.items()})
        for k, v in res_p.items(): print('  ', k, {a: round(b_, 2) for a, b_ in v['normal'].items()}, 'stress', round(v['stress']['ann'], 2), 'vs_main_t', round(v['vs_main_t'], 2))
    out['utc'] = F.now_utc()
    F.save_key('PSEUDO_FORWARD', out, need=['OC_FINAL'])


if __name__ == '__main__':
    main()

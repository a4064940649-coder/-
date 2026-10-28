#!/usr/bin/env python3
"""步驟 2b～3（PREREG 之後一次計算）：選定版本下的事後分布、配置表、新策略、前瞻各線先驗、敏感度、族群與焦點層級樣本外、歷史比較。
必須先有 results.json 的 PREREG（內容雜湊、登記檔案 sha256 都要吻合）。
用法：python3 -I -B s4_formal.py [--inputs v6|v7] [--paths 20000]
輸出：results.json 的 RESULTS（v6）或 RESULTS_V7；data/post_cells.npz、data/forward_priors.npz（v6）。
"""
import sys, json, math, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
import tlib as T, hmethods as H, s3_cells as C3

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
TIERS = C3.TIERS; MODES = C3.MODES
XS = [i / 10 for i in range(11)]

# ---------------------------------------------------------------- 密度工具
def mix_shifts(base, th, shifts):
    acc = np.zeros_like(base)
    for s in shifts: acc += T.shift_dens(base, th, s)
    return acc / acc.sum()

def scale_dens(d, th, c):
    """Y＝cX 的格點密度（c>0）。"""
    y = np.interp(th / c, th, d, left=0.0, right=0.0)
    return y / y.sum()

def add_dens(d1, d2, th, step):
    """獨立相加 Y＝X1＋X2（兩者都在同一格點 th）。"""
    full = np.convolve(d1, d2, mode='full')
    gx = 2 * th[0] + np.arange(len(full)) * step
    out = np.interp(th, gx, full, left=0.0, right=0.0)
    return out / out.sum()

def normal_dens(th, m, s):
    d = stats.norm.pdf(th, m, s); return d / d.sum()

def inv_cdf_sample(d, th, u, step):
    c = np.cumsum(d); c = c / c[-1]
    j = np.searchsorted(c, u).clip(0, len(th) - 1)
    return th[j] + (u - np.where(j > 0, c[j - 1], 0.0)) / np.maximum(d[j] / d.sum(), 1e-300) * step - step / 2

# ---------------------------------------------------------------- 配置模擬
class Alloc:
    def __init__(self, r50, B, L=6, seed=20261101, Hs=(120, 240)):
        self.r50 = np.asarray(r50, float); self.n = len(self.r50)
        self.Hs = Hs; self.B = B
        self.idx = T.sb_index(self.n, max(Hs), L, B, seed)
        self.u = np.random.default_rng(seed + 1).random(B)
    def run(self, dens, th, step, resid_pp, xs=XS):
        a = inv_cdf_sample(dens, th, self.u, step) / 100.0
        e = (np.asarray(resid_pp, float) - np.mean(resid_pp)) / 100.0
        out = {}
        for H_ in self.Hs:
            I = self.idx[:, :H_]; r50p = self.r50[I]; ep = e[I]
            res = {}
            for x in xs:
                o = T.simulate(r50p, ep, a, x, 'M')
                R = o['Wp'] / o['W50']
                res[x] = {'Elog_W_per_year': float(o['logW'].mean() / (H_ / 12)), 'Elog_ratio_per_year': float(o['logR'].mean() / (H_ / 12)),
                          'P_win': float((R > 1).mean()), 'P_ratio_lt_0.75': float((R < 0.75).mean()),
                          'ratio_p10': float(np.percentile(R, 10)), 'ratio_p50': float(np.percentile(R, 50)), 'mdd_p50': float(np.percentile(o['mdd'], 50))}
            xstar = max(res, key=lambda x: res[x]['Elog_W_per_year'])
            out[f'{H_ // 12}y'] = {'xstar': int(round(xstar * 100)), 'grid': {int(round(x * 100)): v for x, v in res.items()}}
        return out

# ---------------------------------------------------------------- 主流程
def posterior_family(fam, ver, X, lab, med, member):
    cols = list(X.columns); k = cols.index(member); c = int(lab[member])
    return fam.posterior(ver, X.values[:, k], c, is_medoid=(k == med[c]), seed=c)

def cell_posts(base_d, cells, th, shifts=None):
    out = {}
    for nm, c in cells.items():
        S = shifts[nm] if shifts else c['S']
        out[nm] = mix_shifts(base_d[c['base']], th, S)
    return out

def run(self, dens, th, step, resid_pp, xs=XS):
        a = inv_cdf_sample(dens, th, self.u, step) / 100.0
        e = (np.asarray(resid_pp, float) - np.mean(resid_pp)) / 100.0
        out = {}
        for H_ in self.Hs:
            I = self.idx[:, :H_]; r50p = self.r50[I]; ep = e[I]
            res = {}
            for x in xs:
                o = T.simulate(r50p, ep, a, x, 'M')
                R = o['Wp'] / o['W50']
                res[x] = {'Elog_W_per_year': float(o['logW'].mean() / (H_ / 12)), 'Elog_ratio_per_year': float(o['logR'].mean() / (H_ / 12)),
                          'P_win': float((R > 1).mean()), 'P_ratio_lt_0.75': float((R < 0.75).mean()),
                          'ratio_p10': float(np.percentile(R, 10)), 'ratio_p50': float(np.percentile(R, 50)), 'mdd_p50': float(np.percentile(o['mdd'], 50))}
            xstar = max(res, key=lambda x: res[x]['Elog_W_per_year'])
            out[f'{H_ // 12}y'] = {'xstar': int(round(xstar * 100)), 'grid': {int(round(x * 100)): v for x, v in res.items()}}
        return out

# ---------------------------------------------------------------- 主流程
def posterior_family(fam, ver, X, lab, med, member):
    cols = list(X.columns); k = cols.index(member); c = int(lab[member])
    return fam.posterior(ver, X.values[:, k], c, is_medoid=(k == med[c]), seed=c)

def cell_posts(base_d, cells, th, shifts=None):
    out = {}
    for nm, c in cells.items():
        S = shifts[nm] if shifts else c['S']
        out[nm] = mix_shifts(base_d[c['base']], th, S)
    return out

def summarize_cells(posts, cells, th, raw=None):
    out = {}
    for nm, d in posts.items():
        s = T.dens_summary(d, th)
        c = cells[nm]
        ra = c.get('raw_alpha_hat', c.get('raw_alpha_hat_jmean'))
        if c['kind'] == 'jmix' and nm.startswith('EXEC'): ra = c['raw_alpha_hat_jmean']
        elif c['kind'] == 'jmix': ra = c['raw_alpha_hat'] + float(np.mean(c['S']) - c['S'][0] * 0) - (0 if True else 0)
        s['raw_alpha_hat_ref'] = ra
        out[nm] = s
    return out

def run(inputs='v6', paths=20000, sens=True, smoke=False):
    if smoke:
        # 冒煙：真實族群殘差＋隨機真值的合成族群（不碰真實焦點的事後），輸出不存入 results.json
        ver = 'V5_HT'; sfx = '' if inputs == 'v6' else '_' + inputs
        X, lab, med = H.load_real()
        rng = np.random.default_rng(99)
        E = X.values - X.values.mean(axis=0, keepdims=True)
        X = pd.DataFrame(E[T.sb_index(len(X), len(X), 6, 1, 99)[0]] + rng.normal(-0.3, 0.6, X.shape[1])[None, :], index=X.index, columns=X.columns)
    else:
        pre = T.check_block('PREREG')
        P = pre['PREREG']
        ver = P['selected_version']
        sfx = '' if inputs == 'v6' else '_' + inputs
        X, lab, med = H.load_real()
    fam = H.Fam(X.values, lab, med, need_g3=(ver == 'V6_HNG3'))
    th, step = fam.th, fam.G.step
    cj = json.loads((T.DATA / f'cells{sfx}.json').read_text(encoding='utf-8'))
    cells = cj['cells']; sig_cell = cj['sigma_lr_pp']
    CS = pd.read_parquet(T.DATA / f'cell_series{sfx}.parquet')
    LS = pd.read_parquet(T.DATA / f'lines_series{sfx}.parquet')
    lmeta = json.loads((T.DATA / f'lines_meta{sfx}.json').read_text(encoding='utf-8'))
    r50 = T.win_ret(T.e50_nav())
    R = {'selected_version': ver, 'inputs': inputs, 'sigma_w': fam.sigma_w, 'sigma_w_c114': fam.sw_c.get(114), 'vP': fam.vP}
    # ---------------- 1. 焦點（AB、B10）：選定版本＋全部版本對照
    base, allv = {}, {}
    for v in H.VERSIONS:
        famv = fam if (v != 'V6_HNG3' or fam.g3 is not None) else H.Fam(X.values, lab, med, need_g3=True)
        dd = {b: posterior_family(famv, v, X, lab, med, m) for b, m in (('AB', T.MAIN_ID), ('B10', T.B10_ID))}
        allv[v] = {b: T.dens_summary(d, th) for b, d in dd.items()}
        if v == ver: base = dd
    raw = {b: float(X[m].mean()) for b, m in (('AB', T.MAIN_ID), ('B10', T.B10_ID))}
    for v in allv:
        for b in allv[v]: allv[v][b]['post_over_raw'] = allv[v][b]['mean'] / raw[b]
    R['focus_all_versions'] = allv; R['raw_alpha_hat'] = raw
    log('focus posteriors', {b: round(allv[ver][b]['mean'], 3) for b in raw})
    # ---------------- 2. 各格事後
    posts = cell_posts(base, cells, th)
    cs = {}
    for nm, d in posts.items():
        s = T.dens_summary(d, th); c = cells[nm]
        s['raw_ref'] = raw[c['base']] + float(np.mean(c['S']))          # 不收縮的對照：原始 α̂＋平均平移
        s['post_over_raw'] = s['mean'] / s['raw_ref'] if s['raw_ref'] > 0 else None
        cs[nm] = s
    R['cells_posterior'] = cs
    if not smoke: np.savez_compressed(T.DATA / f'post_cells{sfx}.npz', theta=th, **{k.replace('|', '__'): v for k, v in posts.items()},
                        base_AB=base['AB'], base_B10=base['B10'])
    log('cells posterior done')
    # ---------------- 3. 配置模擬
    A = Alloc(r50.values, paths)
    alloc = {}
    for nm in cells:
        alloc[nm] = A.run(posts[nm], th, step, CS[nm].values)
    R['allocation'] = alloc
    log('allocation done')
    # 兩模型輸出規則
    tab = {}
    for t in TIERS:
        for m in MODES:
            for cost in ('normal', 'stress'):
                k = f'EXEC|{t}|{m}|{cost}'
                xk, xt = alloc[k + '|KAPPA']['20y']['xstar'], alloc[k + '|TICK']['20y']['xstar']
                xk10, xt10 = alloc[k + '|KAPPA']['10y']['xstar'], alloc[k + '|TICK']['10y']['xstar']
                diff = abs(xk - xt)
                tab[k] = {'xstar_KAPPA_20y': xk, 'xstar_TICK_20y': xt, 'xstar_KAPPA_10y': xk10, 'xstar_TICK_10y': xt10, 'diff_pp': diff,
                          'write': (f'{min(xk, xt)}%' if diff <= 30 else f'{min(xk, xt)}～{max(xk, xt)}%'), 'is_number': diff <= 30,
                          'P_pos_KAPPA': cs[k + '|KAPPA']['p_pos'], 'P_pos_TICK': cs[k + '|TICK']['p_pos'],
                          'mean_KAPPA': cs[k + '|KAPPA']['mean'], 'mean_TICK': cs[k + '|TICK']['mean'],
                          'win10_x100_KAPPA': alloc[k + '|KAPPA']['10y']['grid'][100]['P_win'], 'win10_x100_TICK': alloc[k + '|TICK']['10y']['grid'][100]['P_win'],
                          'win20_x100_KAPPA': alloc[k + '|KAPPA']['20y']['grid'][100]['P_win'], 'win20_x100_TICK': alloc[k + '|TICK']['20y']['grid'][100]['P_win'],
                          'Plt075_10y_x100_KAPPA': alloc[k + '|KAPPA']['10y']['grid'][100]['P_ratio_lt_0.75'],
                          'Plt075_10y_x100_TICK': alloc[k + '|TICK']['10y']['grid'][100]['P_ratio_lt_0.75']}
    R['tier_rule'] = tab
    grow = [cs[f'EXEC|{t}|GROW|{c}|{mo}']['p_pos'] for t in TIERS for c in ('normal', 'stress') for mo in ('KAPPA', 'TICK')]
    R['sop_conditional_0050_primary'] = {'all_GROW_P_lt_0.5': bool(max(grow) < 0.5), 'max_GROW_P': max(grow), 'min_GROW_P': min(grow)}
    # ---------------- 4. 新策略與前瞻各線先驗
    nl = {}
    vers_cm = lmeta['capmatched']['versions']; fz = lmeta['capmatched']['frozen']
    M = LS[vers_cm].values; fi = vers_cm.index(fz)
    nl['CAPM_F'], mi = H.posterior_newcluster(fam, ver, M, fi, seed=11)
    nl_meta = {'CAPM_F': {'medoid': vers_cm[mi], 'raw': float(LS[fz].mean()), 't_iid': T.tstat(LS[fz]),
                          't_hac': float(LS[fz].mean() / math.sqrt(T.hac_var_mean(LS[fz].values, 6)))}}
    vers_rg = lmeta['regime']['versions']; fzr = lmeta['regime']['frozen']
    Mr = LS[vers_rg].dropna().values
    nl['REGIME'], mir = H.posterior_newcluster(fam, ver, LS[vers_rg].fillna(0.0).values, vers_rg.index(fzr), seed=12)
    nl_meta['REGIME'] = {'medoid': vers_rg[mir], 'raw': float(LS[fzr].mean()), 't_iid': T.tstat(LS[fzr]),
                         't_hac': float(LS[fzr].mean() / math.sqrt(T.hac_var_mean(LS[fzr].fillna(0).values, 6)))}
    # 登記焦點 d（輪動 − 靜態）：N(0, 0.5²) 先驗，d̂ 減去 8 版 d 序列的最大值偏誤
    stat = LS['RG:V5'] - LS['RG_D']                         # 靜態對 0050 的超額
    Dm = np.column_stack([(LS[v] - stat).fillna(0.0).values for v in vers_rg])
    bq = np.quantile(T.max_noise_boot(Dm, B=400, L=6, seed=13), (np.arange(H.NB_Q) + 0.5) / H.NB_Q)
    dh = float(LS['RG_D'].mean()); s2d = float(T.hac_var_mean(LS['RG_D'].fillna(0).values, 6))
    acc = np.zeros_like(th)
    for b in bq:
        pv = 1 / (1 / 0.25 + 1 / s2d); pm = pv * ((dh - b) / s2d)
        acc += normal_dens(th, pm, math.sqrt(pv))
    nl['REGIME_D'] = acc / acc.sum()
    nl_meta['REGIME_D'] = {'raw': dh, 't_hac': dh / math.sqrt(s2d), 'maxbias_mean': float(bq.mean())}
    se2330 = math.sqrt(T.hac_var_mean(LS['TSMC'].values, 6))
    gB = T.convolve_dens(fam.g2['g'] / fam.g2['g'].sum(), math.sqrt(fam.vP), step)
    newsum = {k: dict(T.dens_summary(d, th), **nl_meta[k]) for k, d in nl.items()}
    for k in newsum:
        t_ = newsum[k].get('t_hac')
        newsum[k]['P_from_raw_t_hac'] = float(stats.norm.cdf(t_)) if t_ is not None else None
        newsum[k]['P_from_raw_t_iid'] = float(stats.norm.cdf(newsum[k]['t_iid'])) if 't_iid' in newsum[k] else None
    # 前瞻各線
    pri, sig, meta_l = {}, {}, {}
    for t in TIERS:
        for m in MODES:
            mix = 0.5 * posts[f'EXEC|{t}|{m}|normal|KAPPA'] + 0.5 * posts[f'EXEC|{t}|{m}|normal|TICK']
            pri[f'L1_MAIN_B10_EXEC@{t}|{m}'] = mix / mix.sum()
            sig[f'L1_MAIN_B10_EXEC@{t}|{m}'] = 0.5 * (sig_cell[f'EXEC|{t}|{m}|normal|KAPPA'] + sig_cell[f'EXEC|{t}|{m}|normal|TICK'])
    d1 = pri['L1_MAIN_B10_EXEC@100萬|FIX']; s1 = sig['L1_MAIN_B10_EXEC@100萬|FIX']
    pri['L2_MAIN80_DISP20'] = add_dens(scale_dens(d1, th, 0.8), scale_dens(gB, th, 0.2), th, step); sig['L2_MAIN80_DISP20'] = math.sqrt(0.84) * s1
    pri['L3_MAIN70_E50_30'] = scale_dens(d1, th, 0.7); sig['L3_MAIN70_E50_30'] = C3.lrsd(LS['M_E50_30'])
    pri['L4_MAIN70_2330_30'] = add_dens(scale_dens(d1, th, 0.7), normal_dens(th, 0.0, 0.3 * se2330), th, step); sig['L4_MAIN70_2330_30'] = C3.lrsd(LS['M_2330_30'])
    pri['L5_CAPM_F'] = nl['CAPM_F']; sig['L5_CAPM_F'] = C3.lrsd(LS[fz])
    pri['L6_CAPM_C50'] = scale_dens(nl['CAPM_F'], th, 0.5); sig['L6_CAPM_C50'] = C3.lrsd(LS['CM_C50'])
    pri['L7_CAPM_C70'] = scale_dens(nl['CAPM_F'], th, 0.3); sig['L7_CAPM_C70'] = C3.lrsd(LS['CM_C70'])
    pri['L8_REGIME'] = nl['REGIME']; sig['L8_REGIME'] = C3.lrsd(LS[fzr])
    pri['L9_REGIME_CORE50'] = scale_dens(nl['REGIME'], th, 0.5); sig['L9_REGIME_CORE50'] = C3.lrsd(LS['RG_CORE50'])
    pri['L10_REGIME_D'] = nl['REGIME_D']; sig['L10_REGIME_D'] = C3.lrsd(LS['RG_D'])
    fsum = {}
    for k, d in pri.items():
        s = T.dens_summary(d, th)
        s['sigma_pp'] = sig[k]; s['alpha_min'] = 0.5 * max(s['mean'], 0.0)
        fsum[k] = s
    R['new_strategies'] = newsum; R['forward_priors'] = fsum; R['se2330_pp'] = se2330
    if inputs == 'v6' and not smoke:
        np.savez_compressed(T.DATA / 'forward_priors.npz', theta=th, **{k.replace('|', '__').replace('@', '_AT_'): v for k, v in pri.items()})
    log('new strategies & forward priors done')
    # ---------------- 5. 敏感度
    if sens:
        R['sensitivity'] = sensitivity(X, lab, med, ver, cells, CS, r50, fam, base, posts, inputs, sfx)
        log('sensitivity done')
        R['oos'] = oos(X, lab, med, ver)
        log('oos done')
        R['history'] = history(CS, r50, alloc, cells)
    return R

def sensitivity(X, lab, med, ver, cells, CS, r50, fam0, base0, posts0, inputs, sfx):
    th0 = fam0.th
    keys = ['B10', 'B10_J', 'EXEC|30萬|FIX|normal|KAPPA', 'EXEC|30萬|FIX|normal|TICK', 'EXEC|100萬|FIX|normal|KAPPA', 'EXEC|100萬|FIX|normal|TICK',
            'EXEC|100萬|GROW|normal|KAPPA', 'EXEC|100萬|GROW|normal|TICK', 'EXEC|1000萬|GROW|normal|KAPPA', 'EXEC|1000萬|GROW|normal|TICK',
            'EXEC|300萬|FIX|stress|KAPPA', 'EXEC|300萬|FIX|stress|TICK']
    r50v = r50.values
    A5 = Alloc(r50v, 5000)
    def evalp(posts, th, step, mask=None, A=None):
        A = A or A5
        out = {}
        for k in keys:
            s = T.dens_summary(posts[k], th)
            rs = CS[k].values if mask is None else CS[k].values[mask]
            if mask is not None:
                A = Alloc(r50v[mask], 5000)
            a = A.run(posts[k], th, step, rs, xs=XS)
            out[k] = {'mean': s['mean'], 'p_pos': s['p_pos'], 'xstar_20y': a['20y']['xstar'], 'xstar_10y': a['10y']['xstar'],
                      'win10_x100': a['10y']['grid'][100]['P_win']}
        return out
    S = {'base_B5000': evalp(posts0, th0, fam0.G.step)}
    for c0 in (0.01, 0.05, 0.2, 1.0):
        f = H.Fam(X.values, lab, med, c0=c0, need_g3=(ver == 'V6_HNG3'))
        b = {bb: posterior_family(f, ver, X, lab, med, m) for bb, m in (('AB', T.MAIN_ID), ('B10', T.B10_ID))}
        S[f'c0_{c0}'] = evalp(cell_posts(b, cells, f.th), f.th, f.G.step)
    for lo, hi in ((-6.0, 6.0), (-10.0, 6.0)):
        G = T.Grid(lo, hi, 0.02)
        f = H.Fam(X.values, lab, med, G=G, need_g3=(ver == 'V6_HNG3'))
        b = {bb: posterior_family(f, ver, X, lab, med, m) for bb, m in (('AB', T.MAIN_ID), ('B10', T.B10_ID))}
        S[f'grid_{lo}_{hi}'] = evalp(cell_posts(b, cells, f.th), f.th, f.G.step)
    for scl in (0.5, 2.0):
        f = H.Fam(X.values, lab, med, sw_scale=scl, need_g3=(ver == 'V6_HNG3'))
        b = {bb: posterior_family(f, ver, X, lab, med, m) for bb, m in (('AB', T.MAIN_ID), ('B10', T.B10_ID))}
        S[f'sigma_w_x{scl}'] = evalp(cell_posts(b, cells, f.th), f.th, f.G.step)
    for how in ('mean', 'median'):
        sh = {k: ([float(np.mean(c['S'])) if how == 'mean' else float(np.median(c['S']))] if c['kind'] == 'jmix' else c['S']) for k, c in cells.items()}
        S[f'j_point_{how}'] = evalp(cell_posts(base0, cells, th0, sh), th0, fam0.G.step)
    S['block_L12'] = evalp(posts0, th0, fam0.G.step, A=Alloc(r50v, 5000, L=12))
    # 拿掉 2021、拿掉 B10 最好 10 窗
    cj = json.loads((T.DATA / f'cells{sfx}.json').read_text(encoding='utf-8'))
    V4 = pd.read_parquet(T.HERE / C3.INPUTS['v6']['v4'])
    basew = {'AB': T.win_ret(V4['AB_V4|normal']), 'B10': T.win_ret(V4['AB_V4_B10|normal'])}
    DJW = pd.read_parquet(T.DATA / f'dJ_windows{sfx}.parquet')
    DJWd = {tuple(k.split('|')): DJW[k] for k in sorted(set(c[0] for c in DJW.columns))}
    DJWd = {k: DJW[f'{k[0]}|{k[1]}'] for k in [('AB', 'normal'), ('AB', 'stress'), ('B10', 'normal'), ('B10', 'stress')]}
    for k in DJWd: DJWd[k].columns = [int(c) for c in DJWd[k].columns]
    exb10 = ((basew['B10'] - r50) * 100)
    masks = {'drop_2021': ~((X.index >= '2021-01-01') & (X.index <= '2021-12-31')),
             'drop_best10_B10': ~X.index.isin(exb10.sort_values(ascending=False).index[:10])}
    for nm, mk in masks.items():
        mk = np.asarray(mk, bool)
        Xm = X[mk]
        f = H.Fam(Xm.values, lab, med, need_g3=(ver == 'V6_HNG3'))
        b = {bb: posterior_family(f, ver, Xm, lab, med, m) for bb, m in (('AB', T.MAIN_ID), ('B10', T.B10_ID))}
        sh = C3.cell_shifts(cells, CS, DJWd, basew, r50, mk)
        S[nm] = evalp(cell_posts(b, cells, f.th, sh), f.th, f.G.step, mask=mk)
    return S

def oos(X, lab, med, ver):
    """SEL 估、VAL／HOLD 看（兩期都已看過，只當描述）。"""
    sel = T.wmask(X.index, *T.SEL)
    out = {}
    labv = np.asarray(lab, int)
    fam = H.Fam(X.values[sel], lab, med, need_g3=(ver == 'V6_HNG3'))
    for per in ('VAL', 'HOLD'):
        pm = T.wmask(X.index, *getattr(T, per))
        rows = []
        for c in sorted(med):
            ii = np.where(labv == c)[0]
            if len(ii) < 5: continue
            aS = X.values[sel][:, ii].mean(axis=0); k = int(ii[np.argmax(aS)])
            y = float(X.values[pm][:, k].mean()); s2 = float(T.hac_var_mean(X.values[pm][:, k], 6))
            for v in ('V1_o6', ver):
                d = fam.posterior(v, X.values[sel][:, k], c, is_medoid=(k == med[c]), seed=c)
                pred = T.convolve_dens(d, math.sqrt(s2), fam.G.step)
                pit = float(np.interp(y, fam.th, np.cumsum(pred)))
                rows.append(dict(cluster=int(c), ver=v, ahat_sel=float(aS.max()), post_mean=T.dens_summary(d, fam.th)['mean'], y=y, pit=pit,
                                 logpd=math.log(max(float(np.interp(y, fam.th, pred)) / fam.G.step, 1e-300))))
        D = pd.DataFrame(rows)
        out[per] = D.groupby('ver').agg(n=('y', 'size'), mean_ahat_sel=('ahat_sel', 'mean'), mean_post=('post_mean', 'mean'), mean_y=('y', 'mean'),
                                         cov80=('pit', lambda p: float(((p >= 0.1) & (p <= 0.9)).mean())), sum_logpd=('logpd', 'sum'),
                                         pit_mean=('pit', 'mean')).reset_index().to_dict('records')
    # 焦點層級：B10（SEL 估）預測 VAL、HOLD
    foc = {}
    cols = list(X.columns); k = cols.index(T.B10_ID); c = int(lab[T.B10_ID])
    for v in ('V1_o6', ver):
        d = fam.posterior(v, X.values[sel][:, k], c, is_medoid=False, seed=c)
        foc[v] = {'post_mean_SEL': T.dens_summary(d, fam.th)['mean']}
        for per in ('VAL', 'HOLD'):
            pm = T.wmask(X.index, *getattr(T, per))
            y = float(X.values[pm][:, k].mean()); s2 = float(T.hac_var_mean(X.values[pm][:, k], 6))
            pred = T.convolve_dens(d, math.sqrt(s2), fam.G.step)
            foc[v][per] = {'realized': y, 'pct': float(np.interp(y, fam.th, np.cumsum(pred)))}
    out['focus_B10'] = foc
    return out

def history(CS, r50, alloc, cells):
    """歷史描述：各格 x*（20 年）組合、主策略 100%、0050、同池等權，在 SEL／VAL／HOLD／ALL 的年化、窗頻 MDD、對 0050 t、對主策略 t。"""
    V4 = pd.read_parquet(T.HERE / C3.INPUTS['v6']['v4'])
    ew = (T.win_ret(V4['AB_V4|EW']) - r50) * 100
    pers = {'SEL': T.SEL, 'VAL': T.VAL, 'HOLD': T.HOLD, 'ALL': ('2006-04-01', '2026-10-12')}
    out = {}
    def stats_(ex, nm):
        r = r50.values + ex.values / 100
        o = {}
        for p, (a, b) in pers.items():
            m = T.wmask(ex.index, a, b)
            g = np.cumprod(1 + r[m]); yrs = m.sum() / 12.0
            o[p] = {'cagr': float(g[-1] ** (1 / yrs) - 1) * 100, 'mdd': float((g / np.maximum.accumulate(g) - 1).min()) * 100,
                    't_vs_0050': T.tstat(ex.values[m]), 'cagr_0050': float(np.prod(1 + r50.values[m]) ** (1 / yrs) - 1) * 100}
        return o
    out['0050'] = stats_(pd.Series(0.0, index=r50.index), '0050'); out['pool_EW'] = stats_(ew, 'EW')
    for nm in ['AB', 'B10', 'EXEC|100萬|FIX|normal|KAPPA', 'EXEC|100萬|FIX|normal|TICK', 'EXEC|1000萬|GROW|normal|KAPPA', 'EXEC|1000萬|GROW|normal|TICK',
               'EXEC|30萬|FIX|normal|KAPPA', 'EXEC|300萬|GROW|normal|KAPPA']:
        xs = alloc[nm]['20y']['xstar'] / 100
        ex = CS[nm]
        o = {'x100': stats_(ex, nm)}
        if 0 < xs < 1:
            o[f'x{int(xs * 100)}'] = stats_(xs * ex, nm)
            d = xs * ex - ex
            o['vs_main_t'] = T.tstat(d.values)
        o['xstar_20y'] = int(xs * 100)
        out[nm] = o
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', default='v6'); ap.add_argument('--paths', type=int, default=20000)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        R = run(a.inputs, 2000, sens=True, smoke=True)
        (T.DATA / 'smoke_ok.txt').write_text('ok ' + str(len(R)))
        log('smoke ok（合成族群；結果不保存、不列印）'); return
    res = T.load_res()
    key = 'RESULTS' if a.inputs == 'v6' else 'RESULTS_' + a.inputs.upper()
    assert key not in res, f'{key} 已存在（一次計算）'
    R = run(a.inputs, a.paths)
    R['utc'] = T.now_utc(); R['runtime_s'] = time.time() - T0
    T.save_key(key, R, need=['PREREG'])
    log('saved', key)

if __name__ == '__main__':
    main()

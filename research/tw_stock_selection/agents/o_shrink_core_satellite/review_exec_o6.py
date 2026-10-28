#!/usr/bin/env python3
"""反方審查（成本與可成交角度）：不改研究員任何檔案，只讀資料、在記憶體內重算。
用法：python3 -I -B review_exec_o6.py
檢查：
 1) 複製 EXEC_J（B10＋100萬 FIX 拖累＋j 中位數平移）。
 2) 把可執行口徑換成：GROW（10～20 年複利後帳戶已不是 100 萬）、300 萬／1,000 萬、手續費 B、壓力、j 用平均而非中位數。
 3) 各口徑在 MAIN 族群 G1/G2/G3 下的事後均值與 P(α>0)，G2 下的 x*（10pp 格點、L6、10 年／20 年）與 x 100%／80% 勝率、P(財富比<0.75)。
不寫任何檔案。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

AG = S.AG
FC = S.FC
w = FC.win_ret

def build():
    F0 = S.load_focus()
    V4 = pd.read_parquet(AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet')
    EX = pd.read_parquet(AG / 'o_exec_capacity_tiers' / 'navs_abavg.parquet')
    EB = pd.read_parquet(AG / 'o_exec_capacity_tiers' / 'data' / 'bench.parquet')
    meta = json.loads((S.DATA / 'focus_meta.json').read_text(encoding='utf-8'))
    zi = w(EB['AB_V4|normal'])
    B10 = w(V4['AB_V4_B10|normal']); B10s = w(V4['AB_V4_B10|stress'])
    jm = np.array(list(meta['j_shift']['j_means_pp'].values())) / 100
    off = meta['j_shift']['AB_V5_official_mean_pp'] / 100
    dj_med = float(np.median(jm) - off); dj_mean = float(jm.mean() - off)
    dj_late = float(np.median(jm[5:]) - off)   # j≥5（散戶無法換股日當天下單時）
    d = lambda c: w(EX[c]) - zi
    F = pd.DataFrame(index=F0.index); F['r0050'] = F0['r0050']
    F['EXEC_J'] = B10 + d('main|100萬|FIX|A') + dj_med
    F['EXEC_J_meanj'] = B10 + d('main|100萬|FIX|A') + dj_mean
    F['EXEC_J_j5plus'] = B10 + d('main|100萬|FIX|A') + dj_late
    F['EXEC_J_30F'] = B10 + d('main|30萬|FIX|A') + dj_med
    F['EXEC_J_30G'] = B10 + d('main|30萬|GROW|A') + dj_med
    F['EXEC_J_feeB'] = B10 + d('main|100萬|FIX|B') + dj_med
    F['EXEC_J_100G'] = B10 + d('main|100萬|GROW|A') + dj_med
    F['EXEC_J_300F'] = B10 + d('main|300萬|FIX|A') + dj_med
    F['EXEC_J_300G'] = B10 + d('main|300萬|GROW|A') + dj_med
    F['EXEC_J_1000F'] = B10 + d('main|1000萬|FIX|A') + dj_med
    F['EXEC_J_1000G'] = B10 + d('main|1000萬|GROW|A') + dj_med
    F['EXEC_J_stress'] = B10s + d('stress|100萬|FIX|B') + dj_med
    F['EXEC_J_stress100G'] = B10s + d('stress|100萬|GROW|B') + dj_med
    rep = float(((F['EXEC_J'] - F0['EXEC_J']).abs()).max())
    # GROW 拖累的時期結構（100 萬）：前 10 年 vs 後 10 年
    g = d('main|100萬|GROW|A') * 100; f = d('main|100萬|FIX|A') * 100
    half = F.index < pd.Timestamp('2016-04-01')
    drag_split = {'100G_first10y': float(g[half].mean()), '100G_last10y': float(g[~half].mean()),
                  '100F_first10y': float(f[half].mean()), '100F_last10y': float(f[~half].mean())}
    return F, meta, {'dj_med_pp': dj_med * 100, 'dj_mean_pp': dj_mean * 100, 'dj_j5plus_pp': dj_late * 100,
                     'replicate_EXEC_J_maxabs': rep, 'drag_split_pp': drag_split}

def main():
    F, meta, info = build()
    print(json.dumps(info, ensure_ascii=False, indent=1))
    sd_cost = float(meta['cost_uncertainty']['sd_cost_pp'])
    X, _ = S.load_family()
    fam = json.loads((S.DATA / 'families.json').read_text(encoding='utf-8'))
    foci = [c for c in F.columns if c != 'r0050']
    fx = {c: (F[c] - F['r0050']).values * 100 for c in foci}
    dec = S.decompose(X[fam['MAIN']], fx, mode='decomp')
    priors = {m: S.fit_prior(m, dec['xhat'], dec['s2']) for m in ('G1', 'G2', 'G3')}
    r50 = F['r0050'].values; n = len(r50); B = 6000
    idx = S.sb_index(n, 240, 6, B, 20261029)
    out = {}
    for c in foci:
        s2f = dec['focus_s2'][c] + sd_cost ** 2
        row = {'raw': fx[c].mean(), 't': S.tstat(fx[c])}
        dens = {}
        for m, pr in priors.items():
            p = S.posterior(pr, m, dec['focus_xhat'][c], s2f, dec['vP'])
            dens[m] = p.pop('dens'); row[m] = (round(p['mean'], 3), round(p['p_pos'], 3))
        e = F[c].values - r50; resid = e - e.mean()
        for m in ('G2', 'G3'):
            a = S.sample_post(dens[m], B, np.random.default_rng(7)) / 100
            for H in (120, 240):
                I = idx[:, :H]; r50p = r50[I]; ep = resid[I]
                el = {}; st = {}
                for x in [i / 10 for i in range(11)]:
                    o = S.simulate(r50p, ep, a, x, 'M')
                    el[x] = float(o['logR'].mean() / (H / 12))
                    if x in (0.3, 0.8, 1.0) and m == 'G2':
                        R = o['Wp'] / o['W50']
                        st[x] = (round(float((R > 1).mean()), 3), round(float((R < 0.75).mean()), 3), round(float(np.percentile(R, 10)), 2))
                xs = max(el, key=el.get)
                row[f'{m}_xstar_{H // 12}y'] = int(round(xs * 100))
                if m == 'G2': row[f'G2_{H // 12}y_win/P<.75/p10@30,80,100'] = st
        out[c] = row
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""步驟 5：序貫規則的檢定力與誤判率模擬（PREREG_FORWARD 凍結前；只用歷史波動，不用歷史 α）。
每條前瞻線：歷史窗超額去均值的殘差與 0050 窗報酬同列平穩 bootstrap（L6，保留自相關與厚尾），加上真實 α ∈ {−1, 0, 0.2, 0.5, 0.76}（pp／月），
4,000 條路徑 × 120 個月，計算觸發時間分布、各期限內升級／下修／審查機率。若任何線在 α＝0 的 120 個月誤升級率 > 5%，
依登記把 E_up 門檻提高（20→40→80）後重跑，直到 ≤ 5%。
輸出：results.json 的 STEP5_SEQSIM。
用法：python3 -I -B s5_seqsim.py
"""
import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tlib as T, seqlib as Q

THR0 = {'E_up': 20.0, 'up_min_months': 24, 'E_down': 10.0, 'P_down': 0.25, 'P_down_min_months': 18, 'change_min_months': 12,
        'E_review': 20.0, 'ratio_review': 0.75}
ALPHAS = [-1.0, 0.0, 0.2, 0.5, 0.76]
NPATH, HOR = 4000, 120
LINE_RESID = {'L1_MAIN_B10_EXEC@100萬|FIX': ('cell', 'EXEC|100萬|FIX|normal|KAPPA', 1.0),
              'L1_MAIN_B10_EXEC@1000萬|GROW': ('cell', 'EXEC|1000萬|GROW|normal|KAPPA', 1.0),
              'L2_MAIN80_DISP20': ('cell', 'EXEC|100萬|FIX|normal|KAPPA', math.sqrt(0.84)),
              'L3_MAIN70_E50_30': ('line', 'M_E50_30', 1.0), 'L4_MAIN70_2330_30': ('line', 'M_2330_30', 1.0),
              'L5_CAPM_F': ('line', 'CM:F1_K5_N20_sqrt|normal', 1.0), 'L6_CAPM_C50': ('line', 'CM_C50', 1.0), 'L7_CAPM_C70': ('line', 'CM_C70', 1.0),
              'L8_REGIME': ('line', 'RG:V5', 1.0), 'L9_REGIME_CORE50': ('line', 'RG_CORE50', 1.0), 'L10_REGIME_D': ('line', 'RG_D', 1.0)}

def run(thr):
    res = T.load_res(); FP = res['RESULTS']['forward_priors']
    Z = np.load(T.DATA / 'forward_priors.npz'); th = Z['theta']
    CS = pd.read_parquet(T.DATA / 'cell_series.parquet'); LS = pd.read_parquet(T.DATA / 'lines_series.parquet')
    r50 = T.win_ret(T.e50_nav()).values
    n = len(r50)
    idx = T.sb_index(n, HOR, 6, NPATH, 20261102)
    out = {}
    for line, (src, col, scl) in LINE_RESID.items():
        key = line.replace('|', '__').replace('@', '_AT_')
        prior = Z[key]; sig = FP[line]['sigma_pp']; amin = FP[line]['alpha_min']
        s = (CS[col] if src == 'cell' else LS[col]).fillna(0.0).values
        e = (s - s.mean()) * scl
        E = e[idx]; R = r50[idx]
        o = {'sigma_pp': sig, 'alpha_min': amin, 'prior_mean': FP[line]['mean'], 'prior_P': FP[line]['p_pos'], 'by_alpha': {}}
        for a in ALPHAS:
            st = Q.path_stats(E + a, prior, th, sig, amin, r50=R)
            tg = Q.triggers(st, thr)
            r = {}
            for k, b in tg.items():
                ft = Q.first_time(b)
                r[k] = {f'P_by_{m}': float(((ft > 0) & (ft <= m)).mean()) for m in (12, 18, 24, 36, 60, 120)}
                r[k]['median_time_if_any'] = float(np.median(ft[ft > 0])) if (ft > 0).any() else None
            r['post_P_median_at'] = {m: float(np.median(st['post_P'][:, m - 1])) for m in (12, 24, 36, 60, 120)}
            o['by_alpha'][str(a)] = r
        out[line] = o
    return out

def main():
    T.check_block('PREREG')
    assert 'RESULTS' in T.load_res()
    t0 = time.time()
    thr = dict(THR0); hist = []
    while True:
        out = run(thr)
        fu = {l: o['by_alpha']['0.0']['upgrade']['P_by_120'] for l, o in out.items()}
        hist.append({'E_up': thr['E_up'], 'max_false_upgrade_120m': max(fu.values()), 'by_line': fu})
        print('E_up', thr['E_up'], 'max false upgrade', round(max(fu.values()), 4), flush=True)
        if max(fu.values()) <= 0.05 or thr['E_up'] >= 80: break
        thr['E_up'] *= 2
    R = {'thresholds_final': thr, 'threshold_history': hist, 'sim': out, 'paths': NPATH, 'horizon_months': HOR, 'alphas_pp': ALPHAS,
         'runtime_s': time.time() - t0, 'utc': T.now_utc()}
    l1 = out['L1_MAIN_B10_EXEC@100萬|FIX']['by_alpha']
    R['prediction4'] = {'P_up36_alpha0.2': l1['0.2']['upgrade']['P_by_36'], 'P_up36_alpha0.5': l1['0.5']['upgrade']['P_by_36'],
                        'P_down24_alpha-1': l1['-1.0']['down']['P_by_24'], 'P_false_up_120_alpha0_maxline': hist[-1]['max_false_upgrade_120m'],
                        'holds': {'up0.2_lt_0.10': l1['0.2']['upgrade']['P_by_36'] < 0.10, 'up0.5_lt_0.25': l1['0.5']['upgrade']['P_by_36'] < 0.25,
                                  'down-1_ge_0.70': l1['-1.0']['down']['P_by_24'] >= 0.70, 'false_up_le_0.05': hist[-1]['max_false_upgrade_120m'] <= 0.05}}
    T.save_key('STEP5_SEQSIM', R, need=['PREREG', 'RESULTS'])
    print(json.dumps(R['prediction4'], indent=1))
    for l, o in out.items():
        print(l, {a: (round(v['upgrade']['P_by_36'], 3), round(v['down']['P_by_24'], 3), round(v['review']['P_by_24'], 3), round(v['upgrade']['P_by_120'], 3))
                  for a, v in o['by_alpha'].items()})

if __name__ == '__main__':
    main()

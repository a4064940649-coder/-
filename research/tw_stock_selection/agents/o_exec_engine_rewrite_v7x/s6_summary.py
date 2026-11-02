#!/usr/bin/env python3
"""步驟五～六：讀 s5_oos 的淨值，產出 RESULTS_OOS（對帳、四項修正逐步、正式表、判定、否證檢查）寫進 results.json。只讀結果，不改規格。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x_data as D, x_stats as S, x_run as XR

OUT = HERE / 'out'
R = json.loads((HERE / 'results.json').read_text())
assert 'PREREG' in R
NX = pd.read_parquet(D.O7 / 'navs_exec7.parquet')
TJ = json.loads((D.O7 / 'results.json').read_text())['RESULTS_EXEC']['table']
CELLS16 = [f'{m}|normal|{t}|{v}' for m in ('K5T', 'TICK') for t in XR.TIERS for v in ('FIX', 'GROW')]
CELLS_ST = [c.replace('normal', 'stress') for c in CELLS16]
KEYS = ['CAGR', 'exc', 't_0050', 'CAGR_SEL', 'exc_SEL', 't_SEL', 'CAGR_Y19on', 'exc_Y19on', 't_Y19on', 'CAGR_VAL', 'exc_VAL', 't_VAL',
        'CAGR_HOLD', 'exc_HOLD', 't_HOLD', 'CAGR_PREODD', 'exc_PREODD', 'CAGR_POSTODD', 'exc_POSTODD', 'maxDD', 'E50_CAGR']


def per_j(df, prefix='B10'):
    out = {}
    for c in df.columns:
        st, j, m, co, t, v = c.split('|')
        if j == 'OFF': continue
        out.setdefault(f'{m}|{co}|{t}|{v}', {})[int(j[1:])] = S.stats(df[c], 'AB_V7_B10', int(j[1:]))
    return out


def summ(pj):
    o = {}
    for k, d in pj.items():
        L = [d[j] for j in sorted(d)]
        o[k] = {kk: float(np.median([x[kk] for x in L])) for kk in KEYS}
        o[k].update({kk + '_p10': float(np.percentile([x[kk] for x in L], 10)) for kk in ('CAGR', 'exc', 'exc_Y19on')})
        o[k]['CAGR_by_j'] = [x['CAGR'] for x in L]
        o[k]['n_j_exc_pos'] = int(sum(x['exc'] > 0 for x in L)); o[k]['n_j_exc_Y19on_pos'] = int(sum(x['exc_Y19on'] > 0 for x in L))
    return o


def verdict(s):
    if s['exc'] > 0 and s['t_0050'] >= 1.5 and s['exc_Y19on'] >= 0: return '贏'
    if s['exc'] <= 0: return '輸'
    return '無可辨識差異'


def formal_table(SM, ACC, rn):
    formal = {}
    for t in XR.TIERS:
        for v in ('FIX', 'GROW'):
            k = f'K5T|normal|{t}|{v}'; s = SM[rn][k]; s0 = SM['R0'][k]
            kt = f'TICK|normal|{t}|{v}'; st = SM[rn][f'K5T|stress|{t}|{v}']
            pa = ACC[rn]
            part = np.median([pa[c]['part_median'] for c in pa if c.endswith(f'|K5T|normal|{t}|{v}') and '|J' in c])
            formal[f'{t}|{v}'] = dict(CAGR_K5T=s['CAGR'], CAGR_p10_K5T=s['CAGR_p10'], exc_med=s['exc'], exc_p10=s['exc_p10'], t_med=s['t_0050'],
                                      exc_SEL=s['exc_SEL'], t_SEL=s['t_SEL'], exc_Y19on=s['exc_Y19on'], t_Y19on=s['t_Y19on'], CAGR_Y19on=s['CAGR_Y19on'],
                                      exc_VAL=s['exc_VAL'], t_VAL=s['t_VAL'], exc_HOLD=s['exc_HOLD'], CAGR_HOLD=s['CAGR_HOLD'], maxDD=s['maxDD'],
                                      E50_CAGR=s['E50_CAGR'], n_j_exc_Y19on_pos=s['n_j_exc_Y19on_pos'],
                                      stress_CAGR=st['CAGR'], stress_exc=st['exc'], TICK_sens_CAGR=SM[rn][kt]['CAGR'], TICK_sens_exc=SM[rn][kt]['exc'],
                                      TICK_sens_exc_Y19on=SM[rn][kt]['exc_Y19on'],
                                      part_Q_over_OA_median=float(part), TICK_not_lower_bound=bool(part > 0.10),
                                      orig_setting_CAGR=s0['CAGR'], delta_vs_orig=s['CAGR'] - s0['CAGR'],
                                      verdict=verdict(s), verdict_orig_setting=verdict(s0), verdict_TICK_sens=verdict(SM[rn][kt]))
    return formal


def load(name):
    p = OUT / f'navs_oos_{name}.parquet'
    return pd.read_parquet(p) if p.exists() else None


if __name__ == '__main__':
    res = {}
    runs = {n: load(n) for n in ('R0', 'R1', 'R2', 'R3', 'R4', 'S_F3', 'S_F4', 'R4_z025', 'R4_z100', 'R2c', 'R4c')}
    PJ = {n: per_j(df) for n, df in runs.items() if df is not None}
    SM = {n: summ(pj) for n, pj in PJ.items()}
    ACC = {n: json.loads((OUT / f'acc_oos_{n}.json').read_text()) for n in runs if (OUT / f'acc_oos_{n}.json').exists()}
    # ---- 對帳：R0 對 V7 原表 ----
    orig = summ(per_j(NX[[c for c in NX.columns if c.startswith('B10|J')]]))
    rec = {}
    for k in CELLS16 + CELLS_ST:
        a, b = SM['R0'][k], orig[k]
        rec[k] = dict(orig_CAGR_med=b['CAGR'], v7x_CAGR_med=a['CAGR'], diff_pp=a['CAGR'] - b['CAGR'],
                      max_abs_diff_by_j=float(np.max(np.abs(np.array(a['CAGR_by_j']) - np.array(b['CAGR_by_j'])))),
                      orig_exc_Y19on=b['exc_Y19on'], v7x_exc_Y19on=a['exc_Y19on'], orig_t=b['t_0050'], v7x_t=a['t_0050'],
                      unfavourable_CAGR=min(a['CAGR'], b['CAGR']))
    d16 = np.array([abs(rec[k]['diff_pp']) for k in CELLS16]); dst = np.array([abs(rec[k]['diff_pp']) for k in CELLS_ST])
    res['RECON'] = dict(cells=rec, max_abs_diff_normal16=float(d16.max()), max_abs_diff_stress16=float(dst.max()),
                        n_over_0p10_normal=int((d16 > 0.10).sum()), n_over_0p10_stress=int((dst > 0.10).sum()),
                        live_table_check={k: dict(orig_K5T=TJ.get(f'B10|J0|{k}') is not None) for k in []})
    # 官方日對帳
    off = {}
    for nm in ('R0',):
        f = OUT / f'navs_oos_{nm}_OFF.parquet'
        if f.exists():
            df = pd.read_parquet(f)
            for c in df.columns:
                st, j, m, co, t, v = c.split('|')
                a = S.stats(df[c], 'AB_V7_B10', 'OFF'); b = S.stats(NX[c], 'AB_V7_B10', 'OFF')
                off[c] = dict(orig=b['CAGR'], v7x=a['CAGR'], diff=a['CAGR'] - b['CAGR'])
    res['RECON_OFF'] = dict(cells=off, max_abs_diff=float(max(abs(x['diff']) for x in off.values())) if off else None)
    # ---- 逐步修正 ----
    steps = [('F1', 'R0', 'R1'), ('F2', 'R1', 'R2'), ('F3', 'R2', 'R3'), ('F4', 'R3', 'R4'), ('ALL', 'R0', 'R4'),
             ('F3_alone', 'R0', 'S_F3'), ('F2c_posthoc', 'R1', 'R2c'), ('ALL_c_posthoc', 'R0', 'R4c'), ('R4c_vs_R4', 'R4', 'R4c'), ('F4_alone', 'R0', 'S_F4'),
             ('F3_z025_vs_R4', 'R4', 'R4_z025'), ('F3_z100_vs_R4', 'R4', 'R4_z100')]
    dec = {}
    for nm, a, b in steps:
        if a not in SM or b not in SM: continue
        dec[nm] = {k: {kk: float(np.median([PJ[b][k][j][kk] - PJ[a][k][j][kk] for j in range(13)]))
                       for kk in ('CAGR', 'CAGR_SEL', 'CAGR_Y19on', 'CAGR_PREODD', 'CAGR_POSTODD', 't_0050')}
                   for k in CELLS16 + CELLS_ST if k in PJ[a] and k in PJ[b]}
        for k in dec[nm]:
            v = [PJ[b][k][j]['CAGR'] - PJ[a][k][j]['CAGR'] for j in range(13)]
            dec[nm][k]['CAGR_min_j'] = float(min(v)); dec[nm][k]['CAGR_max_j'] = float(max(v)); dec[nm][k]['n_j_positive'] = int(sum(x > 0 for x in v))
    res['DECOMP'] = dec
    res['FALSIFY'] = {nm: [k for k, v in d.items() if v['CAGR'] > 0.05] for nm, d in dec.items() if nm in ('F1', 'F2', 'F3', 'F4', 'ALL')}
    # ---- 每步 16 格摘要 ----
    res['STEP_TABLES'] = {n: {k: {kk: s[k][kk] for kk in ('CAGR', 'CAGR_p10', 'exc', 'exc_p10', 't_0050', 'CAGR_SEL', 'exc_SEL', 't_SEL', 'CAGR_Y19on', 'exc_Y19on',
                                                         't_Y19on', 'exc_VAL', 't_VAL', 'CAGR_HOLD', 'exc_HOLD', 'maxDD', 'E50_CAGR', 'n_j_exc_Y19on_pos')}
                              for k in CELLS16 + CELLS_ST if k in s} for n, s in SM.items()}
    # ---- 正式表與判定（R4 = V7X）----
    formal = formal_table(SM, ACC, 'R4')
    if 'R4c' in SM: res['FORMAL_V7Xc_posthoc'] = formal_table(SM, ACC, 'R4c')
    res['FORMAL_V7X'] = formal
    res['Y19_ALL16_LOSE'] = {n: all(SM[n][k]['exc_Y19on'] < 0 for k in CELLS16) for n in SM if all(k in SM[n] for k in CELLS16)}
    # 1,000 萬 FIX：原設定 vs V7X
    res['P4_1000FIX'] = {n: dict(K5T=verdict(SM[n]['K5T|normal|1000萬|FIX']), TICK=verdict(SM[n]['TICK|normal|1000萬|FIX']),
                                 exc_K5T=SM[n]['K5T|normal|1000萬|FIX']['exc'], t=SM[n]['K5T|normal|1000萬|FIX']['t_0050']) for n in SM if 'TICK|normal|1000萬|FIX' in SM[n]}
    # 逐年（V7X 正式 30萬 FIX、1000萬 GROW；13 j 中位數）
    yr = {}
    for k in ('K5T|normal|30萬|FIX', 'K5T|normal|1000萬|GROW', 'K5T|normal|100萬|FIX'):
        Y = [S.yearly_exc(runs['R4'][f'B10|J{j}|{k}'], 'AB_V7_B10', j) for j in range(13)]
        yr[k] = {y: float(np.median([d[y] for d in Y if y in d])) for y in sorted(set().union(*Y))}
    res['YEARLY_EXC_MED'] = yr
    # 成本記帳（j 中位數，pp／年的近似：總成本÷平均權益÷年數 由 acc 給的元數換算略；這裡報次數與參與率）
    res['ACC_R4'] = {k: {kk: float(np.median([ACC['R4'][c][kk] for c in ACC['R4'] if c.endswith('|' + k) and '|J' in c]))
                         for kk in ('n_cap_hit', 'n_zcap_hit', 'deferred_days', 'n_disp_sells', 'n_disp_cap_hit', 'part_median', 'part_p90', 'part_gt10')}
                     for k in CELLS16}
    res['ACC_R0'] = {k: {kk: float(np.median([ACC['R0'][c][kk] for c in ACC['R0'] if c.endswith('|' + k) and '|J' in c]))
                         for kk in ('n_cap_hit', 'n_zcap_hit', 'deferred_days', 'n_disp_sells', 'part_median', 'part_p90', 'part_gt10')}
                     for k in CELLS16}
    # 對照：0050、同池等權、主策略零衝擊（V7 官方檔）
    NV = pd.read_parquet(D.O7 / 'navs_v7.parquet'); JS = pd.read_parquet(D.O7 / 'navs_jscan_v7.parquet')
    zj = [S.stats(JS[f'AB_V7_B10|normal|J{j}'], 'AB_V7_B10', j) for j in range(13)]
    res['BENCH'] = dict(E50_CAGR_jmed=float(np.median([x['E50_CAGR'] for x in zj])),
                        zero_impact_B10_jmed=dict(CAGR=float(np.median([x['CAGR'] for x in zj])), exc=float(np.median([x['exc'] for x in zj])),
                                                  t=float(np.median([x['t_0050'] for x in zj])), exc_Y19on=float(np.median([x['exc_Y19on'] for x in zj]))),
                        EW_official=S.stats(NV['V7_B10|EW'].dropna(), 'AB_V7_B10', 'OFF')['CAGR'],
                        main_official_AB_V7=S.stats(NV['AB_V7|normal'].dropna(), 'AB_V7', 'OFF')['CAGR'] if 'AB_V7|normal' in NV else None)
    R['RESULTS_OOS'] = res
    (HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print(json.dumps(dict(RECON={k: v for k, v in res['RECON'].items() if k != 'cells'}, RECON_OFF=res['RECON_OFF'].get('max_abs_diff'),
                          FALSIFY=res['FALSIFY'], Y19=res['Y19_ALL16_LOSE'], P4=res['P4_1000FIX']), ensure_ascii=False, indent=1))
    for k, v in formal.items(): print(k, {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()})

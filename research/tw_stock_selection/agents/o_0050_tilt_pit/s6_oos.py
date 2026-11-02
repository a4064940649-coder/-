#!/usr/bin/env python3
"""樣本外（PREREG 凍結後才可執行；斷言 data/PREREG.json 存在且 sha 相符）。
驗證期 VAL：訊號窗起點 2019-01～2023-12；保留期 HOLD：2024-01～2026-10；POST2019＝兩者合併。挑選期 SEL 2014-01～2018-12 一併列出。
凍結版：T1＝T1_lam1.5_nocap（主）、T2＝T2_lam1.5_nocap。13 個 j 的中位數為主欄；官方日另列。寫 results.json['oos'] 與 data/oos_series.parquet。"""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tcore as T, tlib as L
PR = L.DATA / 'PREREG.json'
assert PR.exists(), 'PREREG 不存在：不得計算樣本外'
PRE = json.loads(PR.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(PRE['spec'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == PRE['spec_sha256'], 'PREREG 內容被改過'
for f, h in PRE['code_sha256'].items():
    assert L.sha(L.HERE / f) == h, f'程式 {f} 在凍結後被改過'
SPEC = PRE['spec']
JS = list(range(13))
PER = {'SEL': ('2014-01-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31'), 'HOLD': ('2024-01-01', '2026-10-31'), 'POST2019': ('2019-01-01', '2026-10-31')}
VERS = {'T1': ('T1', 1.5, False), 'T2': ('T2', 1.5, False)}
MAIN = pd.read_parquet(L.AG / 'o_v7_listing_exec_engine' / 'navs_jscan_v7.parquet')
MAIN_OFF = pd.read_parquet(L.AG / 'o_v7_listing_exec_engine' / 'navs_v7.parquet')
def main_win(days, j):
    col = f'AB_V7|normal|J{j}' if j != 'off' else 'AB_V7|normal'
    nav = (MAIN if j != 'off' else MAIN_OFF)[col].reindex(L.IDX).ffill()
    v = nav.reindex(pd.DatetimeIndex(days)).to_numpy()
    return pd.Series(v[1:] / v[:-1] - 1, index=pd.DatetimeIndex(days[:-1]))
def ew_plan(pre):
    return {d: (u, np.full(len(u), 1.0 / len(u))) for d, u, w, tz in pre}
OUT = {'spec_sha256': PRE['spec_sha256'], 'periods': PER, 'by_version': {}}
SER = {}
for vname, (kind, lam, cap) in VERS.items():
    res = {}
    for j in JS + ['off']:
        ds = T.sig_days(j, '2014-01-01', '2026-10-31')
        pre = T.precompute(ds, kind)
        PB = {d: (u, w) for d, u, w, tz in pre}; PT = {}
        for d, U, wb, tz in pre:
            wt = wb.copy()
            for nm, (idx, z) in tz.items(): wt[idx] = T.tilt(wb[idx], z, lam, cap)
            PT[d] = (U, wt)
        out = {}
        for cost in ('normal', 'stress', 'zero'):
            rT, iT = T.run(ds, PT, cost=cost); rB, iB = T.run(ds, PB, cost=cost)
            out[cost] = (rT, rB, iT, iB)
        rEW, _ = T.run(ds, ew_plan(pre)); b = T.bench_win(ds); m = main_win(ds, j)
        R, Rb0 = T.gross_matrix(ds, kind, lam, cap, P=1000, seed=7000 + (j if j != 'off' else 99))
        Rr, _ = T.gross_matrix(ds, kind, lam, cap, real=True)
        plc = pd.Series(R.mean(0), index=Rb0.index)
        df = pd.DataFrame({'T': out['normal'][0], 'B': out['normal'][1], 'T_st': out['stress'][0], 'B_st': out['stress'][1],
                           'T_g': out['zero'][0], 'B_g': out['zero'][1], 'PLC_g': plc, 'E50': b, 'EW': rEW, 'MAIN': m,
                           'turn_T': pd.Series(out['normal'][2]['turnover'], index=out['normal'][0].index), 'cost_T': pd.Series(out['normal'][2]['cost'], index=out['normal'][0].index)})
        SER[(vname, str(j))] = df
        per = {}
        for pn, (a, z) in PER.items():
            x = df[(df.index >= a) & (df.index <= z)]
            if len(x) < 3: continue
            yr = (x['T'] - x['B']).groupby(x.index.year).sum()
            tot = (x['T'] - x['B']).sum()
            xs = (x['T'] - x['B']).sort_values()
            per[pn] = {
                'n': int(len(x)),
                'vsB': T.stats(x['T'] - x['B']), 'vs0050': T.stats(x['T'] - x['E50']), 'B_vs0050': T.stats(x['B'] - x['E50']),
                'vsB_stress': T.stats(x['T_st'] - x['B_st']), 'vs0050_stress': T.stats(x['T_st'] - x['E50']),
                'placebo_vsB_gross': T.stats(x['PLC_g'] - x['B_g']), 'vs_random_gross': T.stats(x['T_g'] - x['PLC_g']),
                'drop_best5_vsB': T.stats(xs.iloc[:-5]) if len(xs) > 8 else None,
                'max_year_share': float(yr.max() / tot) if tot > 0 else float('nan'), 'year_excess_pp': (yr * 100).round(3).to_dict(),
                'CAGR': {'T': T.cagr(x['T']), 'B': T.cagr(x['B']), 'E50': T.cagr(x['E50']), 'EW': T.cagr(x['EW']), 'MAIN': T.cagr(x['MAIN'])},
                'MDD': {'T': T.mdd(x['T']), 'E50': T.mdd(x['E50']), 'MAIN': T.mdd(x['MAIN'])},
                'turnover': float(x['turn_T'].iloc[1:].mean()), 'cost_pp': float(x['cost_T'].iloc[1:].mean() * 100),
                'vsEW': T.stats(x['T'] - x['EW']), 'vsMAIN': T.stats(x['T'] - x['MAIN']), 'MAIN_vs0050': T.stats(x['MAIN'] - x['E50']),
                'corr_active_T_vs_main': float(np.corrcoef(x['T'] - x['E50'], x['MAIN'] - x['E50'])[0, 1]),
                'corr_activeB_T_vs_main': float(np.corrcoef(x['T'] - x['B'], x['MAIN'] - x['E50'])[0, 1]),
                'mix_T70_main30_vs0050': T.stats(0.7 * x['T'] + 0.3 * x['MAIN'] - x['E50']), 'mix_E70_main30_vs0050': T.stats(0.7 * x['E50'] + 0.3 * x['MAIN'] - x['E50']),
                'mix_pair_T70_minus_E70': T.stats(0.7 * (x['T'] - x['E50'])),
            }
        res[str(j)] = per
        print(vname, j, {pn: (round(v['vsB']['mean_pp'], 3), round(v['vsB']['t'], 2), round(v['vs0050']['mean_pp'], 3), round(v['vs0050']['t'], 2)) for pn, v in per.items()}, flush=True)
    # j 中位數彙總
    def jmed(pn, path):
        vals = []
        for j in JS:
            v = res[str(j)].get(pn)
            for p in path: v = v[p] if v is not None else None
            if v is not None and np.isfinite(v): vals.append(v)
        return {'median': float(np.median(vals)), 'p10': float(np.percentile(vals, 10)), 'p90': float(np.percentile(vals, 90)), 'n_j': len(vals)} if vals else None
    summ = {}
    for pn in PER:
        summ[pn] = {k: jmed(pn, p) for k, p in {
            'vsB_mean_pp': ('vsB', 'mean_pp'), 'vsB_t': ('vsB', 't'), 'vsB_IR': ('vsB', 'IR'), 'vsB_TE': ('vsB', 'TE'),
            'vs0050_mean_pp': ('vs0050', 'mean_pp'), 'vs0050_t': ('vs0050', 't'), 'vs0050_IR': ('vs0050', 'IR'),
            'B_vs0050_mean_pp': ('B_vs0050', 'mean_pp'), 'vsB_stress_mean_pp': ('vsB_stress', 'mean_pp'), 'vs0050_stress_mean_pp': ('vs0050_stress', 'mean_pp'),
            'vs0050_stress_t': ('vs0050_stress', 't'),
            'placebo_mean_pp': ('placebo_vsB_gross', 'mean_pp'), 'placebo_t': ('placebo_vsB_gross', 't'),
            'vs_random_mean_pp': ('vs_random_gross', 'mean_pp'), 'vs_random_t': ('vs_random_gross', 't'),
            'drop_best5_t': ('drop_best5_vsB', 't'), 'max_year_share': ('max_year_share',),
            'CAGR_T': ('CAGR', 'T'), 'CAGR_B': ('CAGR', 'B'), 'CAGR_E50': ('CAGR', 'E50'), 'CAGR_EW': ('CAGR', 'EW'), 'CAGR_MAIN': ('CAGR', 'MAIN'),
            'MDD_T': ('MDD', 'T'), 'MDD_E50': ('MDD', 'E50'), 'MDD_MAIN': ('MDD', 'MAIN'),
            'turnover': ('turnover',), 'cost_pp': ('cost_pp',), 'vsEW_mean_pp': ('vsEW', 'mean_pp'), 'vsMAIN_mean_pp': ('vsMAIN', 'mean_pp'), 'vsMAIN_t': ('vsMAIN', 't'),
            'MAIN_vs0050_mean_pp': ('MAIN_vs0050', 'mean_pp'), 'MAIN_vs0050_t': ('MAIN_vs0050', 't'),
            'corr_active_T_vs_main': ('corr_active_T_vs_main',), 'corr_activeB_T_vs_main': ('corr_activeB_T_vs_main',),
            'IR_mix_T70_main30': ('mix_T70_main30_vs0050', 'IR'), 'IR_mix_E70_main30': ('mix_E70_main30_vs0050', 'IR'),
            'mix_pair_t': ('mix_pair_T70_minus_E70', 't')}.items()}
    OUT['by_version'][vname] = {'per_j': res, 'jmedian': summ}
# 判定（PREREG 門檻）
G = {}
for vname in VERS:
    s = OUT['by_version'][vname]['jmedian']; P = s['POST2019']
    a = abs(P['placebo_t']['median']) < 1 and abs(s['SEL']['placebo_t']['median']) < 1
    b = (P['vsB_mean_pp']['median'] >= 0.08 and P['vsB_t']['median'] >= 2.0 and P['vs0050_mean_pp']['median'] >= 0.08 and P['vs0050_t']['median'] >= 2.0)
    c = P['vs_random_t']['median'] >= 2.0
    d = P['drop_best5_t']['median'] >= 1.0 and P['max_year_share']['median'] <= 0.40
    e = s['SEL']['vsB_mean_pp']['median'] >= 0
    G[vname] = {'a_placebo': bool(a), 'b_excess': bool(b), 'c_vs_random': bool(c), 'd_robust': bool(d), 'e_selection_nonneg': bool(e)}
OUT['gates_a_to_e'] = G
R = json.loads((L.HERE / 'results.json').read_text(encoding='utf-8')); R['oos'] = OUT
(L.HERE / 'results.json').write_text(L.dumps(R), encoding='utf-8')
pd.concat({f'{k[0]}|{k[1]}': v for k, v in SER.items()}, axis=1).to_parquet(L.DATA / 'oos_series.parquet')
print(json.dumps(G, ensure_ascii=False))
for vname in VERS:
    print(vname, json.dumps({pn: {k: (round(v['median'], 3) if v else None) for k, v in OUT['by_version'][vname]['jmedian'][pn].items()} for pn in PER}, ensure_ascii=False))

#!/usr/bin/env python3
"""穩健性與機制子樣本（PREREG 凍結後）：主動報酬拆解（層、產業、個股；毛報酬近似：Σ(w'−w)×e 開盤→下一訊號日收盤報酬）、
拿掉最大貢獻股、子樣本（營收年增率分散度高／低、電子權重高／低、2330 相對 0050 大漲月、金融層）。門檻用 explore.json E6（挑選期凍結）。
期間 POST2019、SEL；13 個 j 的中位數。寫 results.json['robust']。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tcore as T, tlib as L
PRE = json.loads((L.DATA / 'PREREG.json').read_text(encoding='utf-8'))
AM = json.loads((L.DATA / 'PREREG_AMEND.json').read_text(encoding='utf-8')) if (L.DATA / 'PREREG_AMEND.json').exists() else {'code_sha256': {}}
for f, h in PRE['code_sha256'].items(): assert L.sha(L.HERE / f) in (h, AM['code_sha256'].get(f)), f
TH = json.loads((L.HERE / 'explore.json').read_text(encoding='utf-8'))['E6_subsample_thresholds']
SER = pd.read_parquet(L.DATA / 'oos_series.parquet')
OUT = {}
for vname, (kind, lam, cap) in {'T1': ('T1', 1.5, False), 'T2': ('T2', 1.5, False)}.items():
    for pname, (a, z) in {'POST2019': ('2019-01-01', '2026-10-31'), 'SEL': ('2014-01-01', '2018-12-31')}.items():
        agg = []
        for j in range(13):
            ds = T.sig_days(j, a, z); pre = T.precompute(ds, kind); ti = [T.IDX.get_loc(d) for d in ds]
            df = SER[f'{vname}|{j}'].dropna(how='all'); df = df[(df.index >= a) & (df.index <= z)]
            rows = []; stk = {}; ind = {}; tiers = {}
            for k, (d, U, wb, tz) in enumerate(pre):
                f2 = T.factor_open_to_close(ti[k] + 1, ti[k + 1]) - 1
                wt = wb.copy(); tmap = {}
                for nm, (idx, zz) in tz.items():
                    wt[idx] = T.tilt(wb[idx], zz, lam, cap)
                    for i in idx: tmap[i] = nm
                act = wt - wb; r = np.nan_to_num(f2[U], nan=0.0)
                c = act * r
                for i, cc in enumerate(c):
                    if cc == 0: continue
                    code = T.COLS[U[i]]; stk[code] = stk.get(code, 0) + cc
                    ind[T.IND[U[i]]] = ind.get(T.IND[U[i]], 0) + cc
                    tn = tmap.get(i, 'FIX'); tiers.setdefault(tn, {})[d] = tiers.get(tn, {}).get(d, 0) + cc
                fin_act = float(np.abs(act[T.FINMASK[U]]).sum()); a2330 = float(np.abs(act[U == T.J2330]).sum())
                y = T.y3_ab(d)[U][~T.FINMASK[U]]; y = y[np.isfinite(y)]
                iqr = float(np.subtract(*np.percentile(y, [75, 25]))) if len(y) >= 4 else float('nan')   # 修正 v1：該月無已公告營收時記缺值
                elec = float(wb[T.ELECMASK[U]].sum())
                rel = float(T.factor_close_to_close(ti[k], ti[k + 1])[T.J2330] - 1 - (L.E50.iloc[ti[k + 1]] / L.E50.iloc[ti[k]] - 1))
                rows.append({'d': d, 'gross_active': float(c.sum()), 'iqr': iqr, 'elec': elec, 'rel2330': rel, 'fin_abs_active': fin_act, 'tsmc_abs_active': a2330})
            M = pd.DataFrame(rows).set_index('d')
            M['vsB'] = (df['T'] - df['B']).reindex(M.index)
            top = max(stk, key=lambda c: stk[c]); worst = min(stk, key=lambda c: stk[c])
            x = M['vsB'].dropna()
            res = {
                'tier_contrib_pp_per_month': {tn: float(pd.Series(v).reindex(M.index).fillna(0).mean() * 100) for tn, v in tiers.items()},
                'tier_contrib_t': {tn: L.tstat(pd.Series(v).reindex(M.index).fillna(0)) for tn, v in tiers.items()},
                'industry_contrib_pp_total': {k: float(v * 100) for k, v in sorted(ind.items(), key=lambda kv: -abs(kv[1]))[:8]},
                'top_stock': [top, float(stk[top] * 100)], 'worst_stock': [worst, float(stk[worst] * 100)],
                'top5_stocks': {c: float(stk[c] * 100) for c in sorted(stk, key=lambda c: -stk[c])[:5]},
                'ex_top_stock_t': L.tstat(M['gross_active'] - 0) if False else None,
                'gross_active_mean_pp': float(M['gross_active'].mean() * 100), 'gross_active_t': L.tstat(M['gross_active']),
                'fin_abs_active_max': float(M['fin_abs_active'].max()), 'tsmc_abs_active_max': float(M['tsmc_abs_active'].max()),
                'sub_hi_dispersion': T.stats(x[M['iqr'].reindex(x.index) > TH['rev_yoy_IQR_T1_median']]), 'sub_lo_dispersion': T.stats(x[M['iqr'].reindex(x.index) <= TH['rev_yoy_IQR_T1_median']]),
                'sub_hi_elec': T.stats(x[M['elec'].reindex(x.index) > TH['elec_weight_T1_median']]), 'sub_lo_elec': T.stats(x[M['elec'].reindex(x.index) <= TH['elec_weight_T1_median']]),
                'sub_tsmc_up': T.stats(x[M['rel2330'].reindex(x.index) > TH['tsmc_minus_0050_top_tercile']]), 'sub_tsmc_other': T.stats(x[M['rel2330'].reindex(x.index) <= TH['tsmc_minus_0050_top_tercile']]),
                'corr_vsB_with_rel2330': float(np.corrcoef(x, M['rel2330'].reindex(x.index))[0, 1]),
            }
            # 拿掉最大貢獻股：以其每窗主動貢獻自 vsB 扣除
            tc = {}
            for k, (d, U, wb, tz) in enumerate(pre):
                if top in set(T.COLS[U]):
                    i = list(T.COLS[U]).index(top)
                    f2 = T.factor_open_to_close(ti[k] + 1, ti[k + 1]) - 1
                    wt = wb.copy()
                    for nm, (idx, zz) in tz.items(): wt[idx] = T.tilt(wb[idx], zz, lam, cap)
                    tc[d] = (wt[i] - wb[i]) * np.nan_to_num(f2[U[i]])
            res['ex_top_stock'] = T.stats(x - pd.Series(tc).reindex(x.index).fillna(0))
            agg.append(res)
        def med(f):
            v = [f(r) for r in agg]; v = [q for q in v if q is not None and np.isfinite(q)]
            return float(np.median(v)) if v else None
        tiers_all = sorted({k for r in agg for k in r['tier_contrib_pp_per_month']})
        OUT[f'{vname}|{pname}'] = {
            'tier_contrib_pp_jmed': {tn: med(lambda r: r['tier_contrib_pp_per_month'].get(tn, 0.0)) for tn in tiers_all},
            'tier_contrib_t_jmed': {tn: med(lambda r: r['tier_contrib_t'].get(tn)) for tn in tiers_all},
            'gross_active_mean_pp_jmed': med(lambda r: r['gross_active_mean_pp']), 'gross_active_t_jmed': med(lambda r: r['gross_active_t']),
            'ex_top_stock_t_jmed': med(lambda r: r['ex_top_stock']['t']), 'ex_top_stock_mean_pp_jmed': med(lambda r: r['ex_top_stock']['mean_pp']),
            'top_stock_j0': agg[0]['top_stock'], 'worst_stock_j0': agg[0]['worst_stock'], 'top5_j0': agg[0]['top5_stocks'], 'industry_j0': agg[0]['industry_contrib_pp_total'],
            'sub': {s: {'mean_pp': med(lambda r: r[s]['mean_pp']), 't': med(lambda r: r[s]['t']), 'n': med(lambda r: r[s]['n'])}
                    for s in ['sub_hi_dispersion', 'sub_lo_dispersion', 'sub_hi_elec', 'sub_lo_elec', 'sub_tsmc_up', 'sub_tsmc_other']},
            'corr_vsB_with_rel2330_jmed': med(lambda r: r['corr_vsB_with_rel2330']),
            'fin_abs_active_max': max(r['fin_abs_active_max'] for r in agg), 'tsmc_abs_active_max': max(r['tsmc_abs_active_max'] for r in agg)}
        print(vname, pname, json.dumps(OUT[f'{vname}|{pname}'], ensure_ascii=False, default=str)[:1500], flush=True)
R = json.loads((L.HERE / 'results.json').read_text(encoding='utf-8')); R['robust'] = OUT
(L.HERE / 'results.json').write_text(L.dumps(R), encoding='utf-8')

#!/usr/bin/env python3
"""事後彙整（PREREG 後、描述用，不參與判定）：版本 (a)/(b) 實際名單的年化、月頻最大回撤、對 0050、對主策略 AB_V4（o7 raw/navs/navs_v4.parquet）的配對 t，
各期間（MAIN、PRE、合併），j 中位數。python3 -I -B s7_post.py → results.json['post']"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from s4_stats import load, tstat, JK
import pcore_lite as PL
M = pd.read_parquet(PL.O7 / 'raw' / 'navs' / 'navs_v4.parquet')['AB_V4|normal']
E = pd.read_csv(PL.O7 / 'raw' / 'bench' / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
def wr(nav, sd, de):
    ds = list(pd.DatetimeIndex([str(x) for x in sd])) + [pd.Timestamp(str(de))]
    v = nav.reindex(nav.index.union(ds)).ffill().reindex(ds).to_numpy(); return v[1:] / v[:-1] - 1
def cagr(r, sd, de):
    yrs = (pd.Timestamp(str(de)) - pd.Timestamp(str(sd[0]))).days / 365.25; return float(np.prod(1 + r) ** (1 / yrs) - 1)
def mdd(r):
    v = np.cumprod(1 + r); return float((v / np.maximum.accumulate(v) - 1).min())
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
OUT = {}
for cfg in ['a', 'b']:
    for per in ['MAIN', 'PRE']:
        D = load(f'run_{cfg}_{per}'); rows = []
        for k in ['OFF'] + JK:
            d = D[k]; sd, de = d['sd'], str(d['de']); ra, r5 = d['ra'], d['r5']
            rm = wr(M, sd, de); ok = np.isfinite(rm)
            rows.append(dict(k=k, cagr=cagr(ra, sd, de), cagr5=cagr(r5, sd, de), cagr_main=cagr(rm[ok], sd, de) if ok.all() else np.nan,
                             mdd=mdd(ra), mdd5=mdd(r5), t5=tstat(ra - r5), m5=float((ra - r5).mean() * 100),
                             tmain=tstat((ra - rm)[ok]), mmain=float((ra - rm)[ok].mean() * 100), nmain=int(ok.sum()),
                             t_rand=tstat(ra - d['Rr'].mean(axis=0))))
        T = pd.DataFrame(rows).set_index('k'); J = T.loc[JK]
        OUT[f'{cfg}|{per}'] = dict(CAGR_jmed=float(J.cagr.median()), CAGR_0050_jmed=float(J.cagr5.median()), CAGR_main_AB_V4_jmed=float(J.cagr_main.median()),
                                    maxDD_monthly_jmed=float(J.mdd.median()), maxDD_0050_monthly=float(J.mdd5.median()),
                                    vs0050_t_jmed=float(J.t5.median()), vs0050_mean_pp_jmed=float(J.m5.median()), vs0050_t_p10=float(np.percentile(J.t5, 10)),
                                    vs_main_t_jmed=float(J.tmain.median()), vs_main_mean_pp_jmed=float(J.mmain.median()), vs_main_n=int(J.nmain.median()),
                                    off=dict(CAGR=float(T.loc['OFF', 'cagr']), vs0050_t=float(T.loc['OFF', 't5']), vs_main_t=float(T.loc['OFF', 'tmain'])))
    # 合併 2014-01～2026-10（j 中位數，窗報酬串接）
    c5, cr, cagr_c = [], [], []
    for k in JK:
        d1, d2 = load(f'run_{cfg}_PRE')[k], load(f'run_{cfg}_MAIN')[k]
        ra = np.concatenate([d1['ra'], d2['ra']]); r5 = np.concatenate([d1['r5'], d2['r5']])
        c5.append(tstat(ra - r5)); cagr_c.append(cagr(ra, d1['sd'], d2['de']))
    OUT[f'{cfg}|ALL_2014_2026'] = dict(vs0050_t_jmed=float(np.median(c5)), CAGR_jmed=float(np.median(cagr_c)))
R['post'] = OUT
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(OUT, ensure_ascii=False, indent=1))

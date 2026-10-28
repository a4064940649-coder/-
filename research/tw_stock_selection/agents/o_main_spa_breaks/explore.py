#!/usr/bin/env python3
"""步驟 2：探索（只用挑選期 2006-04～2019-04 的窗；不計算任何正式 p 值）。
內容：族群 t 分布、相關結構、有效試驗數、主策略在族群中的排名；斷點序列與五分位的挑選期描述；
機制可見性（五分位是否單調、對等權超額與對 0050 超額在挑選期內各段的均值）。輸出 explore.json。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C, spa_lib as S
MAIN_ID = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'
SEL = ('2006-04-01', '2019-04-30')

def tdist(D):
    t = np.array([C.tstat(D[c]) for c in D.columns])
    q = np.nanpercentile(t, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {'n': int(len(t)), 'pct': dict(zip(['p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'], [float(x) for x in q])),
            'max': float(np.nanmax(t)), 'n_t_ge2': int((t >= 2).sum()), 'n_t_ge2.5': int((t >= 2.5).sum())}, t

def main():
    D = pd.read_parquet(C.DATA / 'family_excess.parquet'); Dew = pd.read_parquet(C.DATA / 'family_excess_ew.parquet')
    MM = pd.read_csv(C.DATA / 'family_members.csv')
    idx = D.index; m = C.wmask(idx, *SEL)
    fl = MM[MM.F_long].member_id.tolist(); fm = MM[MM.F_main & MM.F_long].member_id.tolist()
    out = {'period': SEL, 'n_windows': int(m.sum())}
    for name, mem in (('F_long', fl), ('F_main_long', fm)):
        X = D.loc[m, mem].fillna(0.0)
        td, t = tdist(X)
        rank = int((t > t[mem.index(MAIN_ID)]).sum()) + 1
        et = S.eff_trials(X.to_numpy())
        Cm = np.corrcoef(X.to_numpy(), rowvar=False)
        iu = np.triu_indices_from(Cm, 1)
        cmain = Cm[mem.index(MAIN_ID)]
        o = {'t_dist': td, 'main_t_SEL': float(t[mem.index(MAIN_ID)]), 'main_rank': rank, 'eff_trials': et,
             'mean_pair_corr': float(np.nanmean(Cm[iu])), 'median_pair_corr': float(np.nanmedian(Cm[iu])),
             'main_corr_pct': {k: float(v) for k, v in zip(['p10', 'p50', 'p90'], np.nanpercentile(np.delete(cmain, mem.index(MAIN_ID)), [10, 50, 90]))},
             'top5': [(mem[j], float(t[j])) for j in np.argsort(-t)[:5]]}
        out[name] = o
    # 挑選期：對同池等權的族群 t
    X = Dew.loc[m, fl].fillna(0.0)
    td, t = tdist(X); out['F_long_vs_EW'] = {'t_dist': td, 'main_t_SEL': float(t[fl.index(MAIN_ID)]), 'main_rank': int((t > t[fl.index(MAIN_ID)]).sum()) + 1}
    # 斷點序列（挑選期描述）
    B = pd.read_parquet(C.DATA / 'bench.parquet'); BI = pd.read_parquet(C.DATA / 'break_inputs.parquet')
    ser = {'x50_net': D[MAIN_ID], 'xEW_net': Dew[MAIN_ID],
           'x50_gross': BI['AB_V4|zero'] - B.r0050, 'xEW_gross': BI['AB_V4|zero'] - B.rEW_ABV4,
           'Q5mQ1_gross': BI['Q5|zero'] - BI['Q1|zero'], 'Q5mQ1_net': BI['Q5|normal'] - BI['Q1|normal']}
    sub = {'2006-04～2010': ('2006-04-01', '2010-12-31'), '2011～2014': ('2011-01-01', '2014-12-31'), '2015～2019-04': ('2015-01-01', '2019-04-30')}
    desc = {}
    for k, s in ser.items():
        x = s[m]
        desc[k] = {'mean_pp': float(x.mean() * 100), 'sd_pp': float(x.std() * 100), 't': C.tstat(x),
                   'sub_mean_pp': {a: float(x[C.wmask(x.index, *b)].mean() * 100) for a, b in sub.items()},
                   'ac1': float(x.autocorr(1)), 'ac3': float(x.autocorr(3))}
    out['break_series_SEL'] = desc
    out['quintile_mean_pp_SEL'] = {q: float(BI[f'Q{q}|zero'][m].mean() * 100) for q in range(1, 6)}
    out['quintile_monotone_SEL'] = bool(np.all(np.diff([out['quintile_mean_pp_SEL'][q] for q in range(1, 6)]) > 0))
    out['corr_series_SEL'] = pd.DataFrame({k: v[m] for k, v in ser.items()}).corr().round(3).to_dict()
    json.dump(out, open(HERE / 'explore.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])

if __name__ == '__main__':
    main()

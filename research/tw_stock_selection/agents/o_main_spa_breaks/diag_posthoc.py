#!/usr/bin/env python3
"""事後診斷（不在 PREREG、不參與任何判定；寫入 results.json 的 POSTHOC_not_preregistered）：
1) 日曆口徑的 2024～2026（2024-01-01 起，日頻淨值，o4 的 S4 定義）落後量，與窗格點口徑（2024-02-15 起的 32 窗）的差異來源：
   2023-12-11 那一窗跨過 2024-01-02～02-14 面板缺口，在 PREREG 的窗格點上被算進「保留期之前」。
2) 同一段日曆期間的台積電缺口（日頻：2024-01-01～2026-10-12，用 0050 與 2330 的日報酬；權重用窗格點的 w2330_used 前值填補）。
3) 五分位價差在保留期的集中度：拿掉最好的 1／3 窗後的保留期均值。
4) 壓力成本的主策略對 0050 t（navs_v4 的 AB_V4|stress），供 SOP 參考。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C
RES = C.check_prereg()
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')

def main():
    nv = pd.read_parquet(C.V4DIR / 'navs_v4.parquet')
    nav = nv['AB_V4|normal']; e50 = C.e50_nav().reindex(nav.index.union(C.e50_nav().index)).ffill().reindex(nav.index)
    a, b = pd.Timestamp('2024-01-01'), pd.Timestamp('2026-10-12')
    pre = nav[nav.index < a].iloc[-1]; pre50 = e50[e50.index < a].iloc[-1]
    yrs = (b - nav[nav.index < a].index[-1]).days / 365.25
    lm = math.log(nav.loc[:b].iloc[-1] / pre); l50 = math.log(e50.loc[:b].iloc[-1] / pre50)
    out = {'calendar_S4': {'main_cagr': math.exp(lm / yrs) - 1, '0050_cagr': math.exp(l50 / yrs) - 1, 'log_shortfall': l50 - lm}}
    # 2024-01-02～2024-02-15 這段（窗格點上屬於 2023-12-11 那一窗）
    s, e = nav[nav.index < a].index[-1], pd.Timestamp('2024-02-15')
    out['gap_segment_2023-12-29_to_2024-02-15'] = {'main': float(nav.loc[e] / nav.loc[s] - 1), '0050': float(e50.loc[e] / e50.loc[s] - 1)}
    # 日頻台積電缺口
    adj = pd.read_parquet(P / 'adj.parquet')['2330']
    T = pd.read_parquet(C.DATA / 'tsmc_inputs.parquet')
    w = T['w2330_used'].reindex(nav.index.union(T.index)).ffill().reindex(nav.index)
    d = nav.index[(nav.index > s) & (nav.index <= b)]
    r50 = e50.pct_change().reindex(d); r23 = adj.reindex(adj.index.union(nav.index)).ffill().reindex(nav.index).pct_change().reindex(d)
    ww = w.shift(1).reindex(d).fillna(w.reindex(d).bfill())
    rex = (r50 - ww * r23) / (1 - ww)
    lex = float(np.log1p(rex.fillna(0)).sum()); l50d = float(np.log1p(r50.fillna(0)).sum())
    out['calendar_S4']['tsmc_gap_log_daily'] = l50d - lex
    out['calendar_S4']['main_vs_ex_tsmc_log'] = lm - lex
    out['calendar_S4']['share_explained_by_tsmc_gap'] = (l50d - lex) / (l50 - lm) if (l50 - lm) > 0 else None
    # 五分位價差保留期集中度
    BI = pd.read_parquet(C.DATA / 'break_inputs.parquet')
    q = (BI['Q5|zero'] - BI['Q1|zero'])
    qh = q[C.wmask(q.index, '2024-01-01', '2026-10-12')].sort_values(ascending=False)
    out['Q5mQ1_hold'] = {'mean_pp': float(qh.mean() * 100), 'drop_best1_pp': float(qh.iloc[1:].mean() * 100), 'drop_best3_pp': float(qh.iloc[3:].mean() * 100),
                         'best3': {str(k.date()): float(v * 100) for k, v in qh.iloc[:3].items()}}
    # 壓力成本 t
    sd = C.GRID
    x = C.win_ret(nv['AB_V4|stress']) - C.win_ret(C.e50_nav())
    out['AB_V4_stress_t_vs_0050'] = {'ALL': C.tstat(x), 'pre2019': C.tstat(x[x.index < pd.Timestamp('2019-05-01')]), 'post2019': C.tstat(x[x.index >= pd.Timestamp('2019-05-01')])}
    C.save_key('POSTHOC_not_preregistered', out)
    import json; print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()

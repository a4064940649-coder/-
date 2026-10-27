#!/usr/bin/env python3
"""第 0 步：缺口清單。不看任何策略報酬。
對每個換股日（2006-04～2026-10，gapfix 窗），在「營收以外」的合格池（前 300、52 週有值、非處置、隔日有開盤）中，
找出 y3（近 3 月年增率平均）缺值的股票，分類：
  NOLEVEL：該月 tw_revenue/rev_5y 完全沒有這檔的營收水準（＝存活者偏誤的候選）
  NOBASE ：有營收水準但沒有 12 個月前的基期（新上市）或基期 ≤ 0
輸出 gap_list.csv（逐檔）、gap_stockmonths.csv（逐股票月）、gap_summary.json。
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
K = V.K
sys.path.pop(0)

YL, Y5, Y5f = V.load_revenue()
# 營收水準（判斷 NOLEVEL）
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
LV = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='first')
r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str}); r5['p'] = pd.PeriodIndex(r5['ym'], freq='M')
LV5 = r5.pivot_table(index='p', columns='code', values='revenue', aggfunc='last')

S = V.V2('gapfix', YL, Y5, exfin=False)
rows = []
for dd in S.sig_days:
    t = K.IDX.get_loc(dd); e = t + 1
    v60 = S.V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(base, v60, -np.inf)
    order = np.argsort(-vv, kind='stable')
    top = np.zeros(K.N, bool); top[order[:min(300, base.sum())]] = True; top &= base
    s52 = S.S52[t]
    el0 = top & np.isfinite(s52) & ~K.disposal_mask(dd) & np.isfinite(K.On[e]) & K.TRADED[e]
    y3 = S.y3(dd)
    miss = el0 & ~np.isfinite(y3)
    m = pd.Period(dd, 'M')
    L = LV5 if dd >= V.SWITCH else LV
    for j in np.where(miss)[0]:
        c = K.COLS[j]
        lv = [L[c].get(m - k, np.nan) if c in L.columns else np.nan for k in (1, 2, 3)]
        typ = 'NOLEVEL' if not np.isfinite(lv).any() else 'NOBASE'
        rows.append((str(dd.date()), c, typ, int(el0.sum()), int(order[:300].tolist().index(j)) if j in order[:300] else -1))
G = pd.DataFrame(rows, columns=['sig', 'code', 'type', 'pool_ex_rev', 'liq_rank'])
G.to_csv(HERE / 'gap_stockmonths.csv', index=False)

# 逐檔：名稱、在池月數、最後交易日（面板）
lt = pd.Series({c: K.IDX[np.where(K.TRADED[:, j])[0][-1]] if K.TRADED[:, j].any() else pd.NaT for j, c in enumerate(K.COLS)})
names = {}
for src in (d[['code', 'name']], pd.read_csv(K.X.parent / 'stock_list.csv', dtype=str)[['code', 'name']]):
    for c, nm in zip(src['code'], src['name']): names.setdefault(c, nm)
lst = (G[G.type == 'NOLEVEL'].groupby('code').agg(first=('sig', 'min'), last=('sig', 'max'), months=('sig', 'size'))
       .reset_index())
lst['name'] = lst['code'].map(names)
lst['last_trade'] = lst['code'].map(lambda c: str(lt[c].date()) if pd.notna(lt[c]) else '')
lst['delisted'] = lst['code'].map(lambda c: bool(pd.notna(lt[c]) and lt[c] < K.IDX[-1] - pd.Timedelta(days=30)))
lst['ever_in_long_rev'] = lst['code'].isin(LV.columns)
lst['last_long_rev'] = lst['code'].map(lambda c: str(LV[c].dropna().index.max()) if c in LV.columns and LV[c].notna().any() else '')
lst = lst.sort_values('months', ascending=False)
lst.to_csv(HERE / 'gap_list.csv', index=False)

G['y'] = G['sig'].str[:4]
summ = {'n_sig_days': len(S.sig_days),
        'by_year_type_stockmonths': G.groupby(['y', 'type']).size().unstack(fill_value=0).to_dict(orient='index'),
        'NOLEVEL_stockmonths_2013_2026': int(((G.type == 'NOLEVEL') & (G.sig >= '2013-01-01')).sum()),
        'NOLEVEL_stockmonths_2013_2020': int(((G.type == 'NOLEVEL') & (G.sig >= '2013-01-01') & (G.sig < '2021-03-01')).sum()),
        'NOLEVEL_stockmonths_2021_2026': int(((G.type == 'NOLEVEL') & (G.sig >= '2021-03-01')).sum()),
        'NOLEVEL_stockmonths_pre2013': int(((G.type == 'NOLEVEL') & (G.sig < '2013-01-01')).sum()),
        'NOLEVEL_codes_2013_2026': int(G[(G.type == 'NOLEVEL') & (G.sig >= '2013-01-01')].code.nunique()),
        'NOLEVEL_codes_delisted': int(lst['delisted'].sum()),
        'per_day_NOLEVEL_2013_2018_mean': float(G[(G.type == 'NOLEVEL') & (G.sig >= '2013-01-01') & (G.sig < '2019-01-01')].groupby('sig').size().reindex(
            [str(x.date()) for x in S.sig_days if '2013-01-01' <= str(x.date()) < '2019-01-01'], fill_value=0).mean())}
(HERE / 'gap_summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(summ, ensure_ascii=False, indent=1))
print(lst.head(80).to_string())

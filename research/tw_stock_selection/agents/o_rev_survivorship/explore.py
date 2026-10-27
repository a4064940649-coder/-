#!/usr/bin/env python3
"""探索（只用挑選期：訊號日 ≤ 2019-04-30；不跑任何策略淨值）。描述統計：
 E1 缺口規模（每個換股日的池內缺營收檔數：國內／KY）
 E2 修補後，新補進的股票在分數前 10／前 30 名中出現的次數（不含開盤漲停遞補，不算報酬）
 E3 後來下市的國內公司：下市前 6～12 個月營收年增率型態（>100% 的比例），與池中其他股票比較
 E4 事件報酬：補進股票在池內時的下一個月報酬（訊號日收盤→下一訊號日收盤）減池平均；分「是否在分數前 30」
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import osurv_core as O
V, K = O.V, O.K
END_SEL = pd.Timestamp('2019-04-30')

YO, Y5 = O.load_yl('ORIG'); YD, _ = O.load_yl('R_DEL'); YA, _ = O.load_yl('R_ALL')
G = pd.read_csv(HERE / 'gap_stockmonths_filled.csv', dtype={'code': str})
G = G[pd.to_datetime(G['sig']) <= END_SEL]
out = {}
per = G.groupby(['sig', 'foreign']).size().unstack(fill_value=0)
per.columns = ['domestic' if not c else 'foreign' for c in per.columns]
out['E1_gap_per_signal_2013_04_2019_04'] = {k: {'mean': float(per.loc[per.index >= '2013-04-01', k].mean()), 'min': int(per.loc[per.index >= '2013-04-01', k].min()),
                                                 'max': int(per.loc[per.index >= '2013-04-01', k].max())} for k in per.columns}
out['E1_gap_per_signal_2013_04_2018_12_domestic_mean'] = float(per.loc[(per.index >= '2013-04-01') & (per.index < '2019-01-01'), 'domestic'].mean())

# 後來下市（panel 最後交易日早於資料終點 30 天以上）
lt = pd.Series({c: K.IDX[np.where(K.TRADED[:, j])[0][-1]] for j, c in enumerate(K.COLS) if K.TRADED[:, j].any()})
DELISTED = set(lt.index[lt < K.IDX[-1] - pd.Timedelta(days=30)])
filled_dom = set(G.loc[G['fill_del'], 'code']); filled_for = set(G.loc[G['fill_all'] & G['foreign'], 'code'])
out['n_filled_codes_selection'] = {'domestic': len(filled_dom), 'domestic_delisted': len(filled_dom & DELISTED), 'foreign': len(filled_for),
                                   'foreign_delisted': len(filled_for & DELISTED)}

# E2 / E4
SO = V.V2('gapfix', YO, Y5); SD = V.V2('gapfix', YD, Y5); SA = V.V2('gapfix', YA, Y5)
days = [d for d in SO.sig_days if pd.Timestamp('2013-01-01') <= d <= END_SEL]
nxt = {d: SO.sig_days[i + 1] for i, d in enumerate(SO.sig_days[:-1])}
e2 = {'R_DEL': {'top10': 0, 'top30': 0}, 'R_ALL': {'top10': 0, 'top30': 0}}
ev = []
for d in days:
    co = SO.cross_section(d)
    for nm, S, fset in (('R_DEL', SD, filled_dom), ('R_ALL', SA, filled_dom | filled_for)):
        cs = S.cross_section(d)
        od = S.order(cs)[:int(cs['el'].sum())]
        new = cs['el'] & ~co['el']
        e2[nm]['top10'] += int(new[od[:10]].sum()); e2[nm]['top30'] += int(new[od[:30]].sum())
        if nm == 'R_ALL' and d in nxt:
            t, t2 = K.IDX.get_loc(d), K.IDX.get_loc(nxt[d])
            A = K.MYADJ.to_numpy()
            r = A[t2] / A[t] - 1      # 收盤→下一訊號日收盤（下一日無值者 NaN；用凍結價處理過於細，探索只看有值者）
            el = cs['el']; poolm = np.nanmean(np.where(el, r, np.nan))
            rk = np.empty(K.N); rk[:] = np.nan; rk[od] = np.arange(len(od))
            for j in np.where(new)[0]:
                ev.append({'sig': str(d.date()), 'code': K.COLS[j], 'foreign': K.COLS[j] in O.FOREIGN, 'delisted': K.COLS[j] in DELISTED,
                           'rank': int(rk[j]), 'y3': float(cs['y3'][j]), 'ret': float(r[j]) if np.isfinite(r[j]) else None, 'pool_mean': float(poolm)})
out['E2_new_entrants_in_top_scores_2013_2019_04'] = e2
E = pd.DataFrame(ev)
E['ex'] = E['ret'] - E['pool_mean']
def summ(m):
    x = E.loc[m, 'ex'].dropna()
    return {'n': int(len(x)), 'mean_ex_pp': float(x.mean() * 100) if len(x) else None, 't_naive': float(K.tstat(x)) if len(x) > 2 else None,
            'n_months': int(E.loc[m, 'sig'].nunique())}
out['E4_next_month_excess_vs_pool'] = {
    'domestic_all': summ(~E['foreign']), 'domestic_top30': summ(~E['foreign'] & (E['rank'] < 30)),
    'domestic_delisted': summ(~E['foreign'] & E['delisted']), 'domestic_delisted_top30': summ(~E['foreign'] & E['delisted'] & (E['rank'] < 30)),
    'foreign_all': summ(E['foreign']), 'foreign_top30': summ(E['foreign'] & (E['rank'] < 30)),
    'foreign_delisted': summ(E['foreign'] & E['delisted'])}
# 事件 t 按事件日分群（方法教訓 1）：月平均再做 t
def clustered(m):
    x = E.loc[m].dropna(subset=['ex']).groupby('sig')['ex'].mean()
    return {'n_months': int(len(x)), 'mean_pp': float(x.mean() * 100) if len(x) else None, 't_by_month': float(K.tstat(x)) if len(x) > 2 else None}
out['E4_clustered_by_signal'] = {'domestic_all': clustered(~E['foreign']), 'domestic_top30': clustered(~E['foreign'] & (E['rank'] < 30)),
                                 'foreign_all': clustered(E['foreign']), 'foreign_top30': clustered(E['foreign'] & (E['rank'] < 30))}

# E3：下市前營收型態（用 R_ALL 年增率；下市前 6～12 個月內任一月 >100%）
def pre_delist_pattern(codes):
    rows = []
    for c in codes:
        if c not in YA.columns or c not in lt.index: continue
        L = pd.Period(lt[c], 'M')
        s = YA[c].reindex(pd.period_range(L - 12, L - 1, freq='M'))
        w = s.reindex(pd.period_range(L - 12, L - 6, freq='M'))
        rows.append({'code': c, 'last_trade': str(lt[c].date()), 'max_yoy_pre12_6': float(w.max()) if w.notna().any() else None,
                     'mean_yoy_pre12_1': float(s.mean()) if s.notna().any() else None, 'gt100': bool((w > 100).any())})
    return rows
dl = sorted((filled_dom | filled_for) & DELISTED)
pat = pre_delist_pattern(dl)
out['E3_pre_delisting_revenue'] = pat
# 池中比較：挑選期所有池內股票月，y3 > 100% 的比例
allp = []
for d in days[::3]:
    cs = SA.cross_section(d); allp.append(np.nanmean(cs['y3'][cs['el']] > 100))
out['E3_pool_share_y3_gt100'] = float(np.mean(allp))
out['E3_delisted_share_gt100'] = float(np.mean([r['gt100'] for r in pat])) if pat else None
(HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
E.to_csv(HERE / 'explore_events.csv', index=False)
print(json.dumps({k: v for k, v in out.items() if k != 'E3_pre_delisting_revenue'}, ensure_ascii=False, indent=1))
print(pd.DataFrame(pat).to_string())

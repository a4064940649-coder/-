#!/usr/bin/env python3
"""檢定 2：減資／公司行動恢復交易（面板 flag==3：參考價重設）。
事前固定規則：
* 事件日 t＝flag==3 的日子；t+1 收盤買進，持有 5／20 個交易日（t+6、t+21 收盤賣）。
* 進場日 t+1 收盤若鎖漲停（漲幅 ≥ 漲跌幅−0.5% 且收在最高）→ 視為買不到，剔除。
* 異常報酬對流動性合格上市股等權；另報對 0050。
* 資料清理版本（事前定義，非依結果挑選）：
  (a) all：全部 flag 3
  (b) clean：剔除「同一天 ≥ 5 檔 flag 3」的日子（全市場性資料錯置，如 2018-01-13/15、2017-08-04、2025-04-10、2025-09-24）
      且剔除 src==3（Yahoo 近似收盤，flag 3 由開盤偏離推定，常是誤判）
  (c) clean_official：clean 且事件日為官方來源 1／2（有證交所參考價），最可信的減資樣本
  (d) clean_liquid：clean 且流動性合格：事件日前「最近 60 筆有成交」的平均成交值 ≥ 1,000 萬、事件日收盤 ≥ 10
      （減資會停牌，日曆 60 日窗口內有缺值，故改用最近 60 筆有效觀測）
* 另報事件日當天報酬（收盤 / 參考價，僅描述，不可交易）與方向分組（事件日當天漲／跌）。
* 日曆時間組合（clean，持有 20 日）：閒置持現金，成本 0.5%。
挑選期 2005-01～2019-04、驗證期 2019-05～2026-10。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evlib as L

D = L.load(); idx = D['idx']; F = D['flag']; SRC = D['src']
SPL = {'sel_2005_2019-04': ('2005-01-01', '2019-04-30'), 'val_2019-05_2026': ('2019-05-01', '2026-12-31')}
st = F.stack(); st = st[st == 3]
ev = pd.DataFrame({'d': st.index.get_level_values(0), 'code': st.index.get_level_values(1)})
ev = ev[ev['d'] >= pd.Timestamp('2005-01-01')].reset_index(drop=True)
cnt = ev.groupby('d').size(); ev['same_day'] = ev['d'].map(cnt)
pos = {d: i for i, d in enumerate(idx)}
cols = {c: j for j, c in enumerate(D['adj'].columns)}
src = SRC.to_numpy(); liq = D['liq'].to_numpy(); lu = D['limit_up_close'].to_numpy(); R = D['ret'].to_numpy()
car = L.Car(D, 'mkt_liq')
rows = []
for d, code, sd in zip(ev['d'], ev['code'], ev['same_day']):
    i = pos[d]; j = cols[code]; i0 = i + 1
    if i0 >= len(idx): continue
    vv = D['value'][code].iloc[:i].dropna().tail(60)
    lp = bool(len(vv) >= 20 and vv.mean() >= 1e7 and D['close_raw'][code].iat[i] >= 10)
    r = {'event': d, 'code': code, 'same_day': sd, 'src': src[i, j], 'liq_prev': lp,
         'day0_ret': R[i, j], 'limit_up_entry': bool(lu[i0, j])}
    r['date'] = idx[i0]
    for h in (5, 20):
        raw, ab, ab50 = car(code, i0, i0 + h) if not r['limit_up_entry'] else (np.nan,) * 3
        r[f'raw{h}'], r[f'ab{h}'], r[f'ab50_{h}'] = raw, ab, ab50
    rows.append(r)
E = pd.DataFrame(rows)
clean = (E['same_day'] < 5) & (E['src'] != 3)
sets = {'all': E, 'clean': E[clean], 'clean_official': E[clean & E['src'].isin([1, 2])], 'clean_liquid': E[clean & E['liq_prev']],
        'clean_day0_up': E[clean & (E['day0_ret'] > 0)], 'clean_day0_down': E[clean & (E['day0_ret'] <= 0)]}
res = {'n_flag3_since_2005': int(len(E)), 'n_clean': int(clean.sum()), 'n_limit_up_at_entry_removed_clean': int((E[clean]['limit_up_entry']).sum()),
       'clustered_dates_removed': {str(k.date()): int(v) for k, v in cnt[cnt >= 5].items()},
       'clean_by_src': {int(k): int(v) for k, v in E[clean]['src'].value_counts().items()},
       'clean_day0_ret_median_pct': round(float(E[clean]['day0_ret'].median() * 100), 2)}
mk = lambda g, h: pd.DataFrame({'date': g['date'], 'raw': g[f'raw{h}'], 'ab': g[f'ab{h}'], 'ab50': g[f'ab50_{h}']})
res['event'] = {k: {f'h{h}': L.by_period(mk(g, h), SPL) for h in (5, 20)} for k, g in sets.items()}
res['yearly_clean_ab20'] = L.yearly_ab(mk(sets['clean'], 20))
g = sets['clean'].dropna(subset=['ab20'])
tr = [(c, pos[d], min(pos[d] + 20, len(idx) - 1)) for c, d in zip(g['code'], g['date'])]
res['calendar_clean_h20'] = {p: L.calendar_portfolio(D, tr, a, min(pd.Timestamp(b), idx[-1])) for p, (a, b) in SPL.items()}
L.save('2_flag3_capital_reduction', res)
print({k: v for k, v in res.items() if k not in ('event', 'calendar_clean_h20')})
for k, v in res['event'].items():
    for h, e in v.items(): print(k, h, {p: (x.get('n'), x.get('mean_ab_pct'), x.get('median_ab_pct'), x.get('t_month'), x.get('net_ab_pct')) for p, x in e.items()})
for p, v in res['calendar_clean_h20'].items(): print(p, v['strategy'], v['0050'], v['share_days_invested'], v['t_vs_EW_invested'])

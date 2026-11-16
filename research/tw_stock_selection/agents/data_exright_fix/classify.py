#!/usr/bin/env python3
"""第 1 步：確認問題。全部「漲跌價差＝0 但收盤有變」的格子對 TWT49U（證交所除權除息計算結果表，公開鏡像）分類；
另抽 20 個案例（固定亂數種子，上市 16＋上櫃 4，另加稽核報告點名的 5 筆）逐筆列出。
輸出 classify.json（本資料夾）。python3 -I -B classify.py"""
import json, time, resource, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from exr_common import HERE, load_twt49u, raw_panel, classify

t0 = time.time()
Z, f = raw_panel()
ev = load_twt49u()
cells, e = classify(Z, ev)
IDX = pd.DatetimeIndex(Z['IDX'])
END = pd.Timestamp('2026-10-22')
out = dict(panel=dict(first=str(IDX[0].date()), last=str(IDX[-1].date()), T=int(len(IDX)), N=int(len(Z['COLS']))),
           twt49u=dict(rows=int(len(ev)), first=str(ev['date'].min().date()), last=str(ev['date'].max().date())))
out['n_cells'] = int(len(cells))
out['n_cells_by_market'] = cells['mk'].value_counts().to_dict()
out['n_cells_to_backtest_end'] = int((cells['date'] <= END).sum())
out['by_cat'] = cells['cat'].value_counts().to_dict()
out['by_cat_twse_to_END'] = cells[(cells['mk'] == 'twse') & (cells['date'] <= END)]['cat'].value_counts().to_dict()
out['by_cat_market'] = cells.groupby('mk')['cat'].value_counts().unstack(fill_value=0).astype(int).to_dict(orient='index')
out['twse_other_examples'] = cells[cells['cat'] == '上市不在除權息表_其他'].head(20)[['date', 'code', 'prev', 'close', 'ret_close']].astype(str).to_dict(orient='records')
out['by_month'] = cells.groupby(cells['date'].dt.month)['cat'].value_counts().unstack(fill_value=0).astype(int).to_dict(orient='index')
out['by_year_twse'] = cells[cells['mk'] == 'twse'].groupby(cells['date'].dt.year)['cat'].value_counts().unstack(fill_value=0).astype(int).to_dict(orient='index')
tw = cells[cells['cat'] == '真除權息_漲跌價差漏填']
out['events_close_unchanged_twse'] = int(((e['mk'] == 1) & (e['chg'] == 0) & (np.abs(e['close'] - e['prevC']) < 1e-9)).sum())
out['missed_effect'] = dict(n=int(len(tw)), mean_ret_ref_pct=float(tw['ret_ref'].mean() * 100), mean_div_yield_pct=float((tw['prev'] / tw['ref_price'] - 1).mean() * 100),
                            median_abs_ret_ref_pct=float(tw['ret_ref'].abs().median() * 100),
                            mean_ret_close_pct=float(tw['ret_close'].mean() * 100))
# 事件面：TWT49U 事件日的官方漲跌價差狀態（上市）
ee = e[e['mk'] == 1]
out['twt49u_events_on_panel_twse'] = int(len(ee))
out['twt49u_event_chg_status'] = ee['ecat'].value_counts().to_dict()
out['twt49u_event_chg_status_by_year'] = ee.groupby(ee['date'].dt.year)['ecat'].value_counts().unstack(fill_value=0).astype(int).to_dict(orient='index')
oth = ee[ee['ecat'] == '其他']
out['twt49u_event_other_examples'] = oth.head(15)[['date', 'code', 'prev_close', 'ref_price', 'type', 'close', 'chg', 'ref_official', 'prevC']].astype(str).to_dict(orient='records')
# 參考價口徑：已反映參考價的事件，官方參考價 vs TWT49U 參考價
fil = ee[ee['ecat'] == '已反映參考價']
out['ref_match_exact_share'] = float((np.abs(fil['ref_official'] - fil['ref_price']) < 1e-6).mean())
# 抽樣
rng = np.random.default_rng(20261116)
pick = []
tws = cells[cells['mk'] == 'twse'].reset_index(drop=True); tps = cells[cells['mk'] == 'tpex'].reset_index(drop=True)
pick += tws.iloc[np.sort(rng.choice(len(tws), 16, replace=False))].assign(src='隨機（上市）').to_dict(orient='records')
pick += tps.iloc[np.sort(rng.choice(len(tps), 4, replace=False))].assign(src='隨機（上櫃）').to_dict(orient='records')
for d, c in [('2026-08-27', '4736'), ('2026-08-31', '3209'), ('2026-09-01', '3416'), ('2026-09-08', '9933'), ('2026-10-07', '6108')]:
    r = cells[(cells['date'] == pd.Timestamp(d)) & (cells['code'] == c)]
    if len(r):
        pick.append(dict(r.iloc[0].to_dict(), src='稽核報告點名'))
    else:
        pick.append(dict(date=d, code=c, src='稽核報告點名', cat='不在 0 格清單'))
keep = ['src', 'date', 'code', 'mk', 'prev_traded', 'prev', 'close', 'ref_price', 'value', 'type', 'ret_close', 'ret_ref', 'cat']
out['sample'] = [{k: (str(x[k].date()) if k == 'date' and hasattr(x.get(k), 'date') else (None if (isinstance(x.get(k), float) and not np.isfinite(x.get(k))) else x.get(k))) for k in keep} for x in pick]
for s in out['sample']:
    for k in ('prev', 'close', 'ref_price', 'value'):
        if s.get(k) is not None:
            s[k] = round(float(s[k]), 4)
    for k in ('ret_close', 'ret_ref'):
        if s.get(k) is not None:
            s[k] = round(float(s[k]) * 100, 2)
out['peak_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
out['secs'] = round(time.time() - t0)
json.dump(out, open(HERE / 'classify.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ('sample', 'by_month', 'by_year_twse', 'twt49u_event_chg_status_by_year')}, ensure_ascii=False, indent=1, default=str))

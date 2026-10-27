#!/usr/bin/env python3
"""事後診斷（不改判定）：
(1) 積極版事件報酬改用「每單位等金額加權」（加碼事件權重 2）——事前公式把未加碼事件算滿額、加碼事件兩單位平均，會高估積極版；
(2) 隨機抽 300 筆 PRIMARY 事件，用獨立程式路徑重算訊號條件、進場價、出場日。
python3 -I diag_agg_unit.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402
from btlib import tstat  # noqa: E402
ev = pd.read_csv(HERE / 'primary_events.csv.gz', dtype={'code': str}, parse_dates=['sig', 'entry'])
COSTS = {'gross': (0.0, 0.0), 'rt0.5': (0.0025, 0.0025), 'slip0.1': L.cost_rates(0.001), 'stress': L.cost_rates('stress')}
W = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'), 'y2024-01_2025-12': ('2024-01-01', '2025-12-31')}
out = {'unit_weighted_AGG_NT': {}}
ev['units'] = np.where(ev.added, 2, 1)
for k, (a, b) in W.items():
    x = ev[(ev.sig >= a) & (ev.sig <= b)]; em = x.entry.dt.to_period('M'); o = {'n_events': int(len(x)), 'n_units': int(x.units.sum())}
    for cn, (bu, se) in COSTS.items():
        ab = ((1 + x.raw) * (1 - se) / (1 + bu) - 1 - x.ew) * x.units
        g = ab.groupby(em).sum() / x.units.groupby(em).sum()
        o[cn] = {'mean_ab_pct_per_unit': round(float(ab.sum() / x.units.sum() * 100), 3), 't_month': tstat(g)}
    # 固定資金版：每事件資金 K，首單位 K/2，未加碼的另一半放現金（0 報酬）
    half = np.where(x.added, x.raw, x.raw1 / 2)
    ewh = np.where(x.added, x.ew, x.ew / 2)
    bu, se = COSTS['slip0.1']
    nh = np.where(x.added, (1 + x.raw) * (1 - se) / (1 + bu) - 1, ((1 + x.raw1) * (1 - se) / (1 + bu) - 1) / 2)
    abh = pd.Series(nh - ewh, index=x.index)
    o['fixed_capital_half_units_slip0.1'] = {'mean_ab_pct': round(float(abh.mean() * 100), 3), 't_month': tstat(abh.groupby(em).mean())}
    out['unit_weighted_AGG_NT'][k] = o
print(json.dumps(out, ensure_ascii=False, indent=1))

# (2) 獨立重算抽樣
P = L.P
adj = pd.read_parquet(P / 'adj.parquet'); op = pd.read_parquet(P / 'open.parquet'); cr = pd.read_parquet(P / 'close_raw.parquet'); fl = pd.read_parquet(P / 'flag.parquet')
good = adj.notna().sum(axis=1) >= 300
adj = adj[good]; op = op[good]; cr = cr[good]; fl = fl[good]
rng = np.random.default_rng(7); smp = ev.sample(300, random_state=7)
bad = []
for r in smp.itertuples():
    s = adj[r.code].mask(fl[r.code] == 5).ffill(limit=10); i = s.index.get_loc(r.sig)
    w = s.iloc[:i + 1]
    c1 = w.iloc[-1] > w.iloc[-200:].mean(); c2 = w.iloc[-1] < w.iloc[-5:].mean()
    c3 = int((w.iloc[-6:].diff().iloc[1:] < 0).sum()) >= 4
    e = s.index[i + 1]; p0 = op[r.code].loc[e] * adj[r.code].loc[e] / cr[r.code].loc[e]
    # 出場：e 起第一個收盤 > SMA5 的日子 + 1
    sma5 = s.rolling(5).mean(); after = (s > sma5).iloc[i + 1:]
    first = after.idxmax() if after.any() else None
    xd = s.index[s.index.get_loc(first) + 1] if first is not None and s.index.get_loc(first) + 1 < len(s) else None
    ok = c1 and c2 and c3 and abs(p0 / r.p0 - 1) < 1e-9 and e == r.entry and (xd is None or str(xd.date()) >= str(adj.index[0].date()))
    sx = adj.index[int(r.x)] if int(r.x) < len(adj.index) else None
    if not (c1 and c2 and c3) or abs(p0 / r.p0 - 1) > 1e-9 or e != r.entry or (xd is not None and sx is not None and xd != sx and r.why == 'sma5'):
        bad.append((r.code, str(r.sig.date()), bool(c1), bool(c2), bool(c3), str(xd), str(sx), r.why))
print('spot-check mismatches', len(bad), bad[:10])
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
res['diag_post_hoc'] = {**out, 'spot_check_300_events_mismatch': len(bad), 'spot_check_examples': bad[:10],
                        'note': '事後診斷，不改判定。PREREG 的 AGG 事件公式（加碼事件取兩單位平均、未加碼事件滿額）會高估積極版；改用每單位等金額加權或固定資金版才是可比口徑。'}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')

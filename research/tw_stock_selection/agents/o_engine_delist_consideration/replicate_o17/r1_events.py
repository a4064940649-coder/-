#!/usr/bin/env python3
"""o17 獨立複製 r1：對價表 v2 計數、現金溢價、換股偏差、d、價格缺口（只用凍結規格＋面板＋delist_prices＋raw 證據）。"""
import hashlib, json, math, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; T = HERE.parent; AG = T.parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
EV = T / 'events_v2.csv'
assert hashlib.sha256(EV.read_bytes()).hexdigest() == '66e80725a4e48387690ded34fc1ebb5b4b7a71ca2b501448c81b30f583b0afbe'
d = pd.read_csv(EV, dtype=str)
C = pd.read_parquet(P / 'close_raw.parquet')
DP = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet')
DPc = DP.pivot(index='date', columns='code', values='close'); DPc.index = pd.to_datetime(DPc.index)
def close(code, day):
    day = pd.Timestamp(day)
    for src in (C, DPc):
        if code in src.columns and day in src.index and np.isfinite(src.at[day, code]): return float(src.at[day, code])
    return float('nan')
out = {}
out['n_rows'] = len(d)
h = d[d.held == 'True']; nh = d[d.held != 'True']
out['held'] = len(h); out['held_verified'] = h.verified.value_counts().to_dict(); out['nonheld_verified'] = nh.verified.value_counts().to_dict()
out['held_U'] = int((d.held_U == 'True').sum()); out['held_IA_only'] = sorted(d.code[d.held_IA_only == 'True'])
# 現金與換股：自己由面板／delist_prices 價格與 raw 證據的數字重算
f = lambda x: float(x) if isinstance(x, str) and x != '' else 0.0
rows = []
for _, r in h.iterrows():
    if r.verified != '已核實（新聞）': continue
    lt = r.last_trade; lc = close(r.code, lt)
    cons = f(r.cash) + f(r.pref_ratio) * f(r.pref_value)
    sc = close(r.successor, lt) if isinstance(r.successor, str) else float('nan')
    if f(r.ratio): cons += f(r.ratio) * sc
    raw = ''.join((T / 'raw' / x).read_text(encoding='utf-8') for x in (r.raw_files or '').split(';') if x and (T / 'raw' / x).exists())
    nums = [x for x in (r.cash, r.ratio, r.pref_ratio) if isinstance(x, str)]
    found = {x: (re.sub(r'0+$', '', x).rstrip('.') in raw) for x in nums}
    rows.append(dict(code=r.code, type=r.type, lt=lt, last_close_mine=lc, last_close_tab=f(r.last_close), succ=r.successor, succ_close_lt=sc,
                     cons=cons, dev_mine=cons / lc - 1, dev_tab=r.dev_vs_lastclose_lt, nums_in_raw=found, raw_files=r.raw_files))
out['verified_rows'] = rows
cashdev = [x['dev_mine'] for x in rows if x['type'] == 'cash']
swapdev = {x['code']: x['dev_mine'] for x in rows if x['type'] in ('mixed', 'swap')}
out['cash_dev_range'] = [min(cashdev), max(cashdev)]
out['swap_dev_lt'] = swapdev; out['swap_dev_median'] = float(np.median(list(swapdev.values())))
mx = max(abs(v) for v in list(swapdev.values()))
out['d'] = math.ceil(round(mx * 1000, 6)) / 1000
# 價格面存活者缺口
ia_last = {c: C[c].last_valid_index() for c in C.columns}
gap = [c for c in DP.code.unique() if c not in C.columns or (ia_last[c] is not None and ia_last[c] < DP[DP.code == c].date.max())]
out['delist_prices_codes'] = int(DP.code.nunique()); out['survivor_gap_codes'] = len(gap)
# IA 面板價格缺口：真實最後交易日比 ia 面板最後收盤晚 ≥ 7 天（日曆）
pg = []
for _, r in d.iterrows():
    if isinstance(r.last_trade, str) and r.code in C.columns:
        li = C[r.code].last_valid_index()
        if li is not None and (pd.Timestamp(r.last_trade) - li).days >= 7: pg.append((r.code, str(li.date()), r.last_trade, r.held))
out['price_gap_rows'] = pg
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
(HERE / 'r1_events.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

#!/usr/bin/env python3
"""事後診斷與抽樣核對（不影響判定）：獨立重算訊號、手算事件、成本／跳空拆解。輸出寫入 results.json 的 diagnostics_posthoc。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402

P = R.P
ev = pd.read_csv(Path(P).parent / 'w4_chu_PRIMARY_events.csv.gz', dtype={'code': str}, parse_dates=['sig', 'entry'])
rng = np.random.default_rng(7)
out = {}

# 1) 獨立重算訊號：直接讀原始 parquet，不用 run.py 的中間變數
adj = pd.read_parquet(P / 'adj.parquet'); cr = pd.read_parquet(P / 'close_raw.parquet'); hi = pd.read_parquet(P / 'high.parquet')
op = pd.read_parquet(P / 'open.parquet'); val = pd.read_parquet(P / 'value.parquet')
keep = adj.notna().sum(axis=1) >= 300
adj, cr, hi, op, val = [x.loc[keep] for x in (adj, cr, hi, op, val)]
def sig_ok(c, d):
    a = adj[c].dropna(); i = a.index.get_loc(d)
    s = adj[c].loc[:d]
    ma = {n: s.iloc[-n:].mean() for n in (5, 10, 20, 60)}
    ma20_5 = s.iloc[-25:-5].mean()
    r = s.iloc[-1] / s.iloc[-2] - 1
    k = s.index[-2]
    ph = hi.at[k, c] * adj.at[k, c] / cr.at[k, c]
    vol = (val[c] / cr[c]).loc[:d]
    v5 = vol.iloc[-6:-1].mean()
    v60 = val[c].loc[:d].fillna(0).iloc[-60:].mean()
    return dict(c1=bool(ma[5] > ma[10] > ma[20] > ma[60]), c2=bool(ma[20] > ma20_5), c3=bool(r >= 0.035), c4=bool(s.iloc[-1] > ph),
                c5=bool(vol.iloc[-1] >= 1.5 * v5), univ=bool(v60 >= 1e7 and cr.at[d, c] >= 10))
smp = ev.sample(200, random_state=1)
bad = []
for _, r in smp.iterrows():
    o = sig_ok(r['code'], r['sig'])
    if not all(o.values()): bad.append((r['code'], str(r['sig'].date()), o))
out['recheck_signal_sample200_failures'] = len(bad)
out['recheck_signal_failures_examples'] = [str(b) for b in bad[:5]]
print('signal recheck failures', len(bad), bad[:3])

# 2) 手算 3 筆事件
traces = []
for _, r in ev.sample(3, random_state=3).iterrows():
    c = r['code']; a = adj[c]; o_adj = op[c] * adj[c] / cr[c]
    ma5 = a.rolling(5).mean()
    e = r['entry']; ie = a.index.get_loc(e); t = r['sig']
    stop = o_adj.loc[t]; p0 = o_adj.loc[e]
    xs = None
    for s in range(ie, ie + 60):
        if a.iloc[s] < stop or a.iloc[s] < ma5.iloc[s]: xs = s + 1; break
    px = o_adj.iloc[xs]
    traces.append({'code': c, 'sig': str(t.date()), 'entry': str(e.date()), 'p0': round(float(p0), 3), 'exit_day': str(a.index[xs].date()),
                   'exit_px': round(float(px), 3), 'raw_hand': round(float(px / p0 - 1), 5), 'raw_run': round(float(r['raw']), 5), 'why_run': r['why']})
out['hand_traces'] = traces
print(json.dumps(traces, ensure_ascii=False, indent=0))

# 3) 拆解：訊號日收盤（買不到）→ 進場開盤的跳空；持有天數分布；成本
for w, (a_, b_) in R.WIN.items():
    d = ev[(ev['sig'] >= a_) & (ev['sig'] <= b_)]
    o = {'n': len(d), 'hold_dist': {str(k): round(float(v), 3) for k, v in d['hold'].clip(upper=11).value_counts(normalize=True).sort_index().items()},
         'mean_gap_close_to_open_pct': round(d['gap'].mean() * 100, 3),
         'mean_gross_ab_pct': round(d['ab_gross'].mean() * 100, 3), 'mean_cost_pct': round((d['raw'] - d['net']).mean() * 100, 3),
         'gross_ab_if_bought_at_signal_close_pct(unattainable)': round(((1 + d['raw']) * (1 + d['gap']) - 1 - d['ew']).mean() * 100, 3),
         'mean_raw_by_exit': {k: round(g['raw'].mean() * 100, 2) for k, g in d.groupby('why')},
         'top1pct_contrib_raw': round(float(d['raw'].nlargest(max(1, len(d) // 100)).sum() / len(d) * 100), 3)}
    out[f'decomp_{w}'] = o
    print(w, o)

# 4) 逐年事件淨異常
out['yearly_ab_net_pct'] = {int(y): round(g['ab'].mean() * 100, 2) for y, g in ev.groupby(ev['sig'].dt.year)}
out['yearly_ab_gross_pct'] = {int(y): round(g['ab_gross'].mean() * 100, 2) for y, g in ev.groupby(ev['sig'].dt.year)}
print('yearly net', out['yearly_ab_net_pct'])
print('yearly gross', out['yearly_ab_gross_pct'])

f = HERE / 'results.json'
dd = json.loads(f.read_text(encoding='utf-8')); dd['diagnostics_posthoc'] = out
f.write_text(json.dumps(dd, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

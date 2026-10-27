#!/usr/bin/env python3
"""事後診斷與抽樣核對（不影響判定）：獨立重算寶塔線狀態與事件、成本／跳空拆解、逐年。輸出寫入 results.json 的 diagnostics_posthoc。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
ev = pd.read_csv(P.parent / 'w4_tower_PRIMARY_events.csv.gz', dtype={'code': str}, parse_dates=['sig', 'entry'])
out = {}
adj = pd.read_parquet(P / 'adj.parquet'); cr = pd.read_parquet(P / 'close_raw.parquet'); op = pd.read_parquet(P / 'open.parquet')
val = pd.read_parquet(P / 'value.parquet'); fl = pd.read_parquet(P / 'flag.parquet')
keep = adj.notna().sum(axis=1) >= 300
adj, cr, op, val, fl = [x.loc[keep] for x in (adj, cr, op, val, fl)]

def state_naive(a, n=3):
    """逐日迴圈（不用 rolling）算寶塔線狀態。"""
    v = a.to_numpy(); st = np.zeros(len(v)); cur = 0
    for k in range(len(v)):
        if k >= n:
            w = v[k - n:k]
            if v[k] > w.max(): cur = 1
            elif v[k] < w.min(): cur = -1
        st[k] = cur
    return pd.Series(st, index=a.index)

def hand_event(c, sig):
    a = adj[c].mask(fl[c] == 5).dropna(); st = state_naive(a); ma = a.rolling(20).mean()
    k = a.index.get_loc(sig)
    ok_sig = st.iloc[k] == 1 and st.iloc[k - 1] == -1 and a.iloc[k] >= ma.iloc[k]
    v60 = val[c].fillna(0).where(val[c].notna().cumsum() > 0).rolling(60).mean().loc[sig]
    ok_univ = v60 >= 1e7 and cr.at[sig, c] >= 10
    o_adj = (op[c] * adj[c] / cr[c])
    e = adj.index[adj.index.get_loc(sig) + 1]; p0 = o_adj.loc[e]
    ke = a.index.get_loc(e)
    s1 = next(kk for kk in range(ke, len(a)) if st.iloc[kk] == -1)
    u = next((kk for kk in range(s1 + 1, len(a)) if st.iloc[kk] == -1), None)
    def xpx(kk):
        d = a.index[kk]; x = adj.index[adj.index.get_loc(d) + 1]
        return x, o_adj.loc[x:].dropna().iloc[0]
    x1, p1 = xpx(s1); x2, p2 = xpx(u)
    raw = 0.5 * p1 / p0 + 0.5 * p2 / p0 - 1
    return dict(code=c, sig=str(sig.date()), signal_ok=bool(ok_sig), univ_ok=bool(ok_univ), entry=str(e.date()), p0=round(float(p0), 3),
                exit1=str(x1.date()), exit2=str(x2.date()), raw_hand=round(float(raw), 5))

rng = np.random.default_rng(11)
smp = ev.sample(150, random_state=5)
bad = []; traces = []
for _, r in smp.iterrows():
    h = hand_event(r['code'], r['sig'])
    if not (h['signal_ok'] and h['univ_ok'] and abs(h['raw_hand'] - r['raw']) < 6e-6): bad.append((h, round(r['raw'], 5)))
    if len(traces) < 4: traces.append({**h, 'raw_run': round(float(r['raw']), 5)})
out['recheck_sample150_failures'] = len(bad); out['recheck_failures_examples'] = [str(b) for b in bad[:5]]
out['hand_traces'] = traces
print('recheck failures', len(bad), bad[:3]); print(json.dumps(traces, ensure_ascii=False, indent=0))

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31')}
for w, (a_, b_) in WIN.items():
    d = ev[(ev['sig'] >= a_) & (ev['sig'] <= b_)]
    o = {'n': len(d), 'hold_last_dist': {str(k): round(float(v), 3) for k, v in d['hold_last'].clip(upper=11).value_counts(normalize=True).sort_index().items()},
         'mean_gap_close_to_open_pct': round(d['gap'].mean() * 100, 3),
         'mean_gross_ab_pct': round(d['ab_gross'].mean() * 100, 3), 'mean_cost_pct': round((d['raw'] - d['net']).mean() * 100, 3),
         'gross_ab_if_bought_at_signal_close_pct(unattainable)': round(((1 + d['raw']) * (1 + d['gap']) - 1 - d['ew']).mean() * 100, 3),
         'top1pct_contrib_raw': round(float(d['raw'].nlargest(max(1, len(d) // 100)).sum() / len(d) * 100), 3),
         'mean_raw_pct': round(d['raw'].mean() * 100, 3), 'hit_raw': round(float((d['raw'] > 0).mean()), 3)}
    out[f'decomp_{w}'] = o
    print(w, o)
out['yearly_ab_net_pct'] = {int(y): round(g['ab'].mean() * 100, 2) for y, g in ev.groupby(ev['sig'].dt.year)}
out['yearly_ab_gross_pct'] = {int(y): round(g['ab_gross'].mean() * 100, 2) for y, g in ev.groupby(ev['sig'].dt.year)}
print('yearly net', out['yearly_ab_net_pct']); print('yearly gross', out['yearly_ab_gross_pct'])
f = HERE / 'results.json'
dd = json.loads(f.read_text(encoding='utf-8')); dd['diagnostics_posthoc'] = out
f.write_text(json.dumps(dd, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

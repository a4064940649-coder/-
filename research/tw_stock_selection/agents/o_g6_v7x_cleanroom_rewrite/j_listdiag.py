# J 診斷（不屬凍結程式）：V60 口徑對名單層（官方日 V7_B10）逐槽一致率的影響
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_g6_v7x_cleanroom_rewrite/')
import xlists
def v60(X, mode):
    V = X['V']; GOOD = X['GOOD']; dates = X['dates']; T, N = V.shape; gi = np.where(GOOD)[0]
    EX = [pd.Timestamp('2017-08-03'), pd.Timestamp('2018-07-12')] + list(pd.date_range('2024-02-15', '2024-03-15'))
    ex = np.isin(dates[gi], pd.DatetimeIndex(EX)); Vg = V[gi]; nanv = np.isnan(Vg); bad = nanv & (~ex[:, None])
    s = pd.DataFrame(np.where(nanv, 0, Vg)).rolling(60, min_periods=60).sum().values
    n = pd.DataFrame((~nanv).astype(float)).rolling(60, min_periods=60).sum().values
    b = pd.DataFrame(bad.astype(float)).rolling(60, min_periods=60).sum().values
    with np.errstate(invalid='ignore', divide='ignore'): g = np.where((b == 0) & (n > 0), s / n, np.nan)
    out = np.full((T, N), np.nan); out[gi] = g
    if mode == 'lim3': return pd.DataFrame(out).ffill(limit=3).values
    if mode == 'strict':
        for t in np.where(~GOOD)[0]:
            if t > 0: out[t] = out[t - 1]
        return out
    return pd.DataFrame(out).ffill().values
ref = pd.read_csv(open('/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine/plan_V7_B10.csv'), dtype=str)
L = xlists.Lists(); base = L.X['V60'].copy(); res = {}
for mode in ['asfrozen', 'strict', 'lim3']:
    L.X['V60'] = v60(L.X, mode) if mode != 'asfrozen' else base
    p = L.plan(L.official_days(), seed=None)
    mine = {str(L.dates[s].date()): set(L.codes[c] for c in pk) for s, e, pk in p}
    tot = same = dd = 0; diffs = []
    for r in ref.itertuples():
        a = set(r.picks.split(',')); b = mine.get(r.signal, set()); tot += len(a); same += len(a & b)
        if a != b: dd += 1; diffs.append((r.signal, sorted(a - b), sorted(b - a)))
    res[mode] = dict(overlap=same / tot, days_diff=dd, diffs=diffs)
    print(mode, same / tot, dd)
json.dump(res, open('/home/user/-/research/tw_stock_selection/agents/o_g6_v7x_cleanroom_rewrite/logs/j_listdiag.json', 'w'), ensure_ascii=False, indent=0)

# R4 名單層再比對（F1／F2／F6 之後；J 角色）：官方日 V7_B10 逐槽、AB_V7_B10 j1～j12 seed0（無亂數日）逐槽 vs 原名單；G6_Y3PCT=rev 為規格缺口另一讀法
import sys, os, json, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import xdata, xlists
L = xlists.Lists(); dates = L.dates
ref = pd.read_csv(open('/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine/plan_V7_B10.csv'), dtype=str)
p = L.plan(L.official_days(), seed=None); mine = {str(dates[s].date()): [L.codes[c] for c in pk] for s, e, pk in p}
tot = same = dd = ex = 0; diffs = []
for r in ref.itertuples():
    a = r.picks.split(','); b = mine.get(r.signal, []); tot += len(a); same += len(set(a) & set(b)); ex += a == b
    if set(a) != set(b): dd += 1; diffs.append((r.signal, sorted(set(a) - set(b)), sorted(set(b) - set(a))))
KD = pickle.load(open(xdata.SCR.replace('g6A/', 'g6K_lists/') + 'k_all_lists.pkl', 'rb'))
CUT = pd.Timestamp('2026-10-12'); t2 = s2 = n2 = d2 = 0; d2l = []
for j in range(1, 13):
    days = [x for x in L.jdays(j) if dates[x[1]] < CUT]
    pm = {str(dates[s].date()): [L.codes[c] for c in pk] for s, e, pk in L.plan(days, seed=0)}
    a = {s: pk for s, e, pk in KD['variants']['orig'][(j, 0)]}
    for s in sorted(set(a) | set(pm)):
        x, y = a.get(s, []), pm.get(s, []); t2 += max(len(x), len(y)); s2 += len(set(x) & set(y)); n2 += 1
        if set(x) != set(y): d2 += 1; d2l.append((j, s, sorted(set(x) - set(y)), sorted(set(y) - set(x))))
out = dict(mode=os.environ.get('G6_Y3PCT', 'frozen'), official=dict(slot_overlap=same / tot, days_setdiff=dd, days_exact=ex, n_days_ref=len(ref), n_days_mine=len(mine), diffs=diffs),
           ab_j1to12_seed0=dict(slot_overlap=s2 / t2, days_setdiff=d2, days=n2, first_diffs=d2l[:40]))
json.dump(out, open(os.path.join(HERE, 'logs', f"list_compare2_{out['mode']}.json"), 'w'), ensure_ascii=False, indent=1)
print(out['mode'], 'official', round(same / tot, 4), dd, ex, len(mine), '| AB j1-12 s0', round(s2 / t2, 4), d2, n2); print(diffs[:12])

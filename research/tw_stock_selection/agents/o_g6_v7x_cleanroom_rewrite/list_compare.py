# R4 名單層比對（解封後；J 角色）：
# (a) 官方日確定性 V7_B10：本題 xlists 重建 vs o_v7_listing_exec_engine/plan_V7_B10.csv（逐槽）
# (b) AB_V7_B10 13 j × 20 seeds：本題自建（formal run 存的 my_plans）vs K 以原程式產生的 orig 名單（＝plans_j7）
#     亂數只影響 J＝D 日（j0 與特殊月），其餘日應逐槽相同。
import sys, json, pickle
import numpy as np, pandas as pd
HERE = '/home/user/-/research/tw_stock_selection/agents/o_g6_v7x_cleanroom_rewrite/'
sys.path.insert(0, HERE)
import xdata, xlists
SCR = xdata.SCR
out = {}
L = xlists.Lists(); dates = L.dates
od = L.official_days(); p = L.plan(od, seed=None)
ref = pd.read_csv(open('/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine/plan_V7_B10.csv'), dtype=str)
mine = {str(dates[s].date()): [L.codes[c] for c in pk] for s, e, pk in p}
rows = []; nsl = 0; nsame = 0; ndays_exact = 0
for r in ref.itertuples():
    a = r.picks.split(','); b = mine.get(r.signal)
    if b is None:
        rows.append(dict(signal=r.signal, status='missing_in_mine')); nsl += len(a); continue
    sa, sb = set(a), set(b); nsl += len(a); nsame += len(sa & sb)
    if a == b: ndays_exact += 1
    elif sa != sb: rows.append(dict(signal=r.signal, only_ref=sorted(sa - sb), only_mine=sorted(sb - sa)))
extra = sorted(set(mine) - set(ref.signal))
out['official_V7_B10'] = dict(ref_days=len(ref), mine_days=len(mine), slots=nsl, slot_overlap=nsame, slot_overlap_rate=nsame / nsl,
                              days_exact_order=ndays_exact, days_set_diff=len([x for x in rows if 'only_ref' in x]), extra_days_mine=extra, diffs=rows[:60])
# (b)
KD = pickle.load(open(SCR.replace('g6A/', 'g6K_lists/') + 'k_all_lists.pkl', 'rb'))
MP = pickle.load(open(SCR + 'my_plans_AB_V7_B10.pkl', 'rb'))
perj = {}
for j in range(13):
    tot = same = dexact = dn = 0; dset = 0; days_m = days_k = 0
    for sd in range(20):
        a = {s: pk for s, e, pk in KD['variants']['orig'][(j, sd)]}
        b = {s: pk for s, e, pk in MP[(j, sd)]}
        days_k += len(a); days_m += len(b)
        for s in sorted(set(a) | set(b)):
            x, y = a.get(s, []), b.get(s, [])
            tot += max(len(x), len(y)); same += len(set(x) & set(y)); dn += 1
            if x == y: dexact += 1
            elif set(x) != set(y): dset += 1
    perj[j] = dict(days_orig=days_k, days_mine=days_m, slot_overlap_rate=same / tot if tot else None, days_exact=dexact, days_setdiff=dset, days=dn)
out['AB_V7_B10_vs_orig'] = perj
# 第一個 j≥1 的不同日（無亂數）逐筆
ex = []
for j in range(1, 13):
    a = {s: pk for s, e, pk in KD['variants']['orig'][(j, 0)]}; b = {s: pk for s, e, pk in MP[(j, 0)]}
    for s in sorted(set(a) | set(b)):
        if set(a.get(s, [])) != set(b.get(s, [])):
            ex.append(dict(j=j, signal=s, only_orig=sorted(set(a.get(s, [])) - set(b.get(s, []))), only_mine=sorted(set(b.get(s, [])) - set(a.get(s, [])))))
out['nonrandom_diffs_seed0'] = ex[:200]; out['n_nonrandom_diffs_seed0'] = len(ex)
json.dump(out, open(HERE + 'list_compare.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: (v if k != 'official_V7_B10' else {kk: vv for kk, vv in v.items() if kk != 'diffs'}) for k, v in out.items() if k != 'nonrandom_diffs_seed0'}, ensure_ascii=False, indent=1, default=str)[:4000])
print('n_nonrandom_diffs_seed0', len(ex)); print(ex[:15])

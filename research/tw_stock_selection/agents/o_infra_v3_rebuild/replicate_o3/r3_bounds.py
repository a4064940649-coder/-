#!/usr/bin/env python3
"""界限（以 V3 為底）：X_upper／X_lower、B05／B10（含 X、B10 無 X）、期望拖累、每年最壞 1 檔；SOP 預期表；併購對價覆蓋。"""
import sys, json, math
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/replicate_o3')
import r3_lib as L, r3_bounds_lib as B
sys.path.pop(0)
import numpy as np, pandas as pd
K, V = L.K, L.V
L.apply_patch()
ref = pd.read_parquet(L.O3 / 'navs_P1.parquet')
mine = pd.read_parquet(L.HERE / 'my_navs_P1.parquet')
YL, Y5 = L.load_rev(1)
bN, sN = K.rates(); bS, sS = K.rates('stress')
out = {}
def seg(nav, ew, sd, g='ALL'):
    a, b, ad = V.SEGS[g]; return V.seg_stats(nav, ew, a, b, sd, ad=ad)
s3 = L.S3(YL, Y5, True, 'hi'); t0, t1, sd = s3.bounds(); pl10 = s3.plan(10)
ew = mine['P1|R1|X1|Dhi|EXFIN|EW']; v3 = mine['P1|R1|X1|Dhi|EXFIN|n10|normal']
# 1. X_upper／X_lower
for m in ('upper', 'lower'):
    for n in (10, 30):
        xr = L.XRules(m); pl = s3.plan(n)
        nav, lg = L.simulate(pl, bN, sN, t0, t1, xr)
        o = seg(nav, ew, sd); out[f'X_{m}|n{n}'] = {'CAGR': o['CAGR'], 't': o['t_0050'], 'conv': lg['conv']}
        print('X', m, n, round(o['CAGR'], 2), round(o['t_0050'], 2), lg['conv'])
# 2. B05／B10
xr = L.XRules('main')
Bobj = {}
for nm, pct in (('B05', 0.5), ('B10', 1.0)):
    for U in ('EXFIN', 'ALL'):
        sb = B.SB(YL, Y5, U == 'EXFIN', 'hi', pct); Bobj[(nm, U)] = sb
        ewU = mine[f'P1|R1|X1|Dhi|{U}|EW']
        for n in (10, 30):
            pl = sb.plan(n)
            nav, lg = L.simulate(pl, bN, sN, t0, t1, xr)
            r = ref[f'P1|R1|Dhi|{U}|{nm}|n{n}|normal']
            o = seg(nav, ewU, sd)
            rec = {'CAGR': o['CAGR'], 't': o['t_0050'], 'maxdiff_vs_ref': float((nav / r - 1).abs().max()), 'frozen_n': len(lg['frozen']),
                   'n_exits': len(lg['conv']), 'gap_picks': int(sum(sum(K.COLS[j] in sb.added.get(p[3], []) for j in p[1]) for p in pl)),
                   'picked_added_codes': sorted({K.COLS[j] for p in pl for j in p[1] if K.COLS[j] in sb.added.get(p[3], [])})}
            if nm == 'B10':
                nav0, lg0 = L.simulate(pl, bN, sN, t0, t1, None)
                o0 = seg(nav0, ewU, sd)
                rec['noX'] = {'CAGR': o0['CAGR'], 't': o0['t_0050'], 'frozen_n': len(lg0['frozen']), 'frozen_max': max([f[2] for f in lg0['frozen']], default=0)}
            out[f'{nm}|{U}|n{n}'] = rec
            print(nm, U, n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != 'picked_added_codes'}, flush=True)
# 併購對價覆蓋：B10 情境中被選中、之後下市的放回股
codes = set()
for U in ('EXFIN', 'ALL'):
    for n in (10, 30): codes |= set(out[f'B10|{U}|n{n}']['picked_added_codes'])
mc = L.MC.set_index('code')
dead = sorted(c for c in codes if c in mc.index or c in B.TERM)
def usable(c):
    if c not in mc.index: return False
    t = mc.at[c, 'type']
    return (t in ('swap', 'failure')) or (t == 'cash' and np.isfinite(mc.at[c, 'cash']))
out['B10_merger_cov'] = {'n_codes': len(dead), 'in_csv': sum(c in mc.index for c in dead), 'type_known': sum(c in mc.index and mc.at[c, 'type'] != 'unknown' for c in dead),
                         'usable_strict': sum(usable(c) for c in dead), 'news': sum(c in mc.index and mc.at[c, 'verified'] == 'news' for c in dead), 'codes': dead}
print('B10 merger cov', out['B10_merger_cov'])
# 3. 期望拖累與 4. 每年最壞 1 檔（排除金融 n10）
sb10 = Bobj[('B10', 'EXFIN')]
ex_days = [p[0] for p in pl10]; sig = [p[3] for p in pl10]
xr = L.XRules('main')
def hold_ret(j, e, e2):
    """e 開盤買進 → e2 開盤（最後一期到 t1 收盤）。期間若永久停止交易：最後收盤 × X 乘數（經營惡化 0）。"""
    v = K.Cn[e, j] / K.On[e, j]
    last = e2 if e2 is not None else t1
    v *= np.prod(K.GROW[e + 1:last + 1, j])
    if e2 is not None:
        if K.TRADED[e2, j] and np.isfinite(K.On[e2, j]): v *= K.On[e2, j] / K.Cn[e2, j]
        elif xr.lt[j] < e2:
            c = K.COLS[j]; typ = xr.mc.at[c, 'type'] if c in xr.mc.index else 'unknown'
            if typ == 'failure': v = 0.0
            elif typ == 'cash' and c in xr.mc.index and np.isfinite(xr.mc.at[c, 'cash']): v *= xr.mc.at[c, 'cash'] / K.Cn[xr.lt[j], j]
    return v - 1
navd = v3.copy()
D = pd.DatetimeIndex([d for d in sd if d in v3.index])
win = (v3.reindex(D).shift(-1) / v3.reindex(D) - 1).iloc[:-1]
delta = {}; worst = {}
for i, (e, pk, pool, d) in enumerate(pl10):
    e2 = ex_days[i + 1] if i + 1 < len(ex_days) else None
    cs = sb10.cross_section(d); add = cs.get('add')
    if add is None or not add.any(): continue
    rp = np.array([hold_ret(j, e, e2) for j in pool]); rew = float(np.mean(rp))
    adds = np.where(add)[0]
    ra = {K.COLS[j]: hold_ret(j, e, e2) for j in adds}
    delta[d] = sum(r - rew for r in ra.values()) / 300
    for j in adds:
        y = d.year
        if y not in worst or cs['sc'][j] > worst[y][2]: worst[y] = (d, K.COLS[j], float(cs['sc'][j]), ra[K.COLS[j]])
def adj_nav(adds):
    """把每個訊號窗的報酬加上 adds[d]（在窗終點以乘數調整）。"""
    f = pd.Series(1.0, index=v3.index)
    for d, x in adds.items():
        if d not in win.index: continue
        r = win[d]; g = 1 + x / (1 + r)
        nxt = D[D.get_loc(d) + 1]
        f[f.index >= nxt] *= g
    return v3 * f
nav_drag = adj_nav(delta)
o = seg(nav_drag, ew, sd); out['expected_drag'] = {'CAGR': o['CAGR'], 't': o['t_0050'], 'n_months': len(delta), 'mean_delta_pp': float(np.mean(list(delta.values())) * 100)}
print('expected drag', out['expected_drag'])
wadd = {}
for y, (d, c, scv, r) in worst.items():
    wadd[d] = wadd.get(d, 0.0) + (r - win[d]) / 10
nav_w = adj_nav(wadd)
o = seg(nav_w, ew, sd); out['worst1'] = {'CAGR': o['CAGR'], 't': o['t_0050'], 'picks': {str(y): (str(d.date()), c, round(scv, 4), round(r * 100, 1)) for y, (d, c, scv, r) in worst.items()}}
print('worst1', out['worst1'])
# 5. SOP：0050 ＋ 0.5×(V3 − 0050)
sop = {}
for g in ('ALL', 'P1', 'S3', 'S4'):
    o = seg(v3, ew, sd, g); sop[g] = {'V3': o['CAGR'], '0050': o['E50_CAGR'], 'exp': o['E50_CAGR'] + 0.5 * (o['CAGR'] - o['E50_CAGR'])}
o = seg(mine['P1|R1|X1|Dhi|EXFIN|n10|stress'], ew, sd); sop['ALL_stress'] = {'V3': o['CAGR'], '0050': o['E50_CAGR'], 'exp': o['E50_CAGR'] + 0.5 * (o['CAGR'] - o['E50_CAGR'])}
b10 = out['B10|EXFIN|n10']['CAGR']; sop['ALL_B10'] = {'V3': b10, 'exp': sop['ALL']['0050'] + 0.5 * (b10 - sop['ALL']['0050'])}
out['SOP'] = sop; print('SOP', {k: {a: round(b, 2) for a, b in v.items()} for k, v in sop.items()})
# 6. 處置剔除名額（2006～2021-09-23 訊號日）：v2 名單中被 D_lo／D_hi 剔除的 (訊號日, 股票)
for n in (10, 30):
    s_v2 = L.S3(YL, Y5, True, 'none'); pv = s_v2.plan(n)
    cnt_lo = cnt_hi = 0
    for (e, pk, pool, d) in pv:
        if d >= L.DSWITCH: continue
        hm = L.hist_mask(d); ext = set(s3.extra_removed.get(d, [])) if d in s3._cs or True else set()
        s3.cross_section(d)
        ext = set(s3.extra_removed.get(d, []))
        for j in pk:
            if hm[j]: cnt_lo += 1; cnt_hi += 1
            elif K.COLS[j] in ext: cnt_hi += 1
    out[f'removed_slots_n{n}'] = {'approx_only': cnt_lo, 'approx_plus_extra': cnt_hi}
    print('removed slots', n, cnt_lo, cnt_hi)
(L.HERE / 'my_bounds.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

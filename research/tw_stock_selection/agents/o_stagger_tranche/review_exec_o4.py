#!/usr/bin/env python3
"""反方審查（成本與可成交）：只讀研究員檔案與模組，不寫任何檔案。python3 -I -B review_exec_o4.py"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import st_core as C
import iterate as IT
T0 = time.time()
st = C.setup(); K = C.K
off_days, off_end = C.official_list()
WO = C.windows_from(off_days, off_end)
E50 = C.E50
out = {}

def ex_series(nav, W=WO):
    return C.wret(nav, W) - C.wret(E50.reindex(nav.index), W)

def limit_down_price(ref, lim):
    raw = ref * (1 - lim); t = K.tick(raw)
    return np.ceil(raw / t - 1e-9) * t

def gap_down_open(t):
    lim = 0.07 if C.IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    ref = K.ref_price(t); ld = limit_down_price(ref, lim)
    return np.isfinite(K.On[t]) & np.isfinite(ref) & (K.On[t] <= ld + 1e-6)

# ---------- 1) 官方日 V3 與 J+1 / V08 的計畫 ----------
plOFF = C.plan_single(off_days, ab=False)
plOFFab = C.plan_single(off_days, ab=True)
d1, e1 = C.version_days(1)
plJ1 = C.plan_single(d1)
SPEC = IT.SPECS['V08']
plsV08 = C.plan_tranches(SPEC['js'], SPEC['a'], SPEC['b'])
print('plans', round(time.time() - T0), flush=True)

# ---------- 2) 跌停開盤賣出（引擎照開盤價成交）、開盤漲停遞補（需盤中盯盤）----------
def exec_stats(plan):
    n_sell = n_ld = 0; n_sub = n_buy = 0; prev = set()
    for (e, pk, pool, d, m) in plan:
        cur = set(pk.tolist())
        sold = prev - cur
        if sold:
            gd = gap_down_open(e)
            n_sell += len(sold); n_ld += sum(bool(gd[j]) for j in sold)
        newb = cur - prev; n_buy += len(newb)
        gu = K.gap_up_open(e)
        # 原始前 10 名裡有幾檔因開盤漲停被跳過（要開盤後才知道，遞補股實際只能盤中買）
        n_sub += 0  # computed below
        prev = cur
    return dict(n_sell=n_sell, n_sell_limitdown_open=n_ld, share=n_ld / max(n_sell, 1), n_newbuy=n_buy)

def subst_stats(days, ab=True):
    n_ev = n_ev_sub = n_sub = 0
    held = set()
    for m, d in days:
        c = C.cross(d, m, 0, ab)
        if c['el'].sum() < 50: continue
        cand = C.order_of(c['sc'], c['y3'], c['el'])
        top = [int(j) for j in cand[:10]]
        s = sum(1 for j in top if c['gu'][j] and j not in held)
        n_ev += 1; n_sub += s; n_ev_sub += s > 0
        pick = []
        for j in cand:
            if c['gu'][j] and j not in held: continue
            pick.append(int(j))
            if len(pick) == 10: break
        held = set(pick)
    return dict(events=n_ev, events_with_limitup_skip=n_ev_sub, share_events=n_ev_sub / n_ev, skipped_names=n_sub)

out['limitdown_sell'] = {'OFF': exec_stats(plOFF), 'J+1': exec_stats(plJ1), 'V08_tranches': [exec_stats(p) for p in plsV08]}
out['limitup_subst'] = {'OFF': subst_stats(off_days, ab=False), 'J+1': subst_stats(d1)}
print('exec', round(time.time() - T0), flush=True)

# ---------- 3) 小帳戶：整股可買性（2020-10-26 前沒有盤中零股，開盤只能整股）、最低手續費 ----------
ODD_INTRADAY = pd.Timestamp('2020-10-26')
def lot_stats(plans, cap, k):
    alloc = cap / k / 10
    n = n_bad = 0; n_pre = n_bad_pre = 0
    for pl in plans:
        for (e, pk, pool, d, m) in pl:
            px = K.On[e][pk] * 1000
            ok = np.isfinite(px)
            bad = (px[ok] > alloc)
            n += ok.sum(); n_bad += bad.sum()
            if C.IDX[e] < ODD_INTRADAY: n_pre += ok.sum(); n_bad_pre += bad.sum()
    return dict(alloc=alloc, share_1lot_gt_alloc=n_bad / n, share_pre2020_10=n_bad_pre / max(n_pre, 1))
out['lot'] = {}
for cap in (3e5, 1e6, 3e6):
    out['lot'][int(cap)] = {'k1_OFF': lot_stats([plOFF], cap, 1), 'k4_V08': lot_stats(plsV08, cap, 4)}
# 最低手續費 20 元（整股）：每筆金額 < 20/FEE 時的額外成本（每邊，占該筆）
FEE = C.FEE
for cap in (3e5, 1e6):
    for k in (1, 4):
        a = cap / k / 10
        out.setdefault('minfee', {})[f'{int(cap)}_k{k}'] = dict(per_trade=a, extra_cost_pct_per_side=max(20 - a * FEE, 0) / a * 100)
print('lot', round(time.time() - T0), flush=True)

# ---------- 4) V3 正式的兩個前視月 ----------
NV = pd.read_parquet(HERE / 'navs_oos.parquet')
exO = ex_series(NV['OFF'].dropna()); exA = ex_series(NV['OFF_AB0'].dropna())
lk = []
for m, d in off_days:
    D = C.deadline(m)
    if d < D: lk.append(str(m))
out['lookahead_months'] = {m: dict(OFF_ex_pp=float(exO[pd.Period(m, 'M')] * 100), AB_ex_pp=float(exA[pd.Period(m, 'M')] * 100)) for m in lk}
x = exO.copy()
for m in lk: x[pd.Period(m, 'M')] = exA[pd.Period(m, 'M')]
out['OFF_t_lookahead_months_replaced_by_AB'] = C.tstat(x.dropna())
out['OFF_t'] = C.tstat(exO.dropna()); out['OFF_AB0_t'] = C.tstat(exA.dropna())
for key in ('SEL', 'POST', 'HOLD'):
    out[f'OFF_t_{key}'] = C.tstat(C.sel(exO, key).dropna()); out[f'OFF_AB0_t_{key}'] = C.tstat(C.sel(exA, key).dropna())
print('lookahead', out['lookahead_months'], round(time.time() - T0), flush=True)

# ---------- 5) 成本壓力：官方 AB 與 V08 在真實成本下 ----------
def blk(nav):
    b = C.block(nav, WO, ['ALL', 'POST', 'HOLD']); return {k: dict(CAGR=round(b[k]['CAGR'], 2), t=round(b[k]['t_0050'], 3), ex=round(b[k]['ex_mean_pp'], 3)) for k in b}
t0O = C.IDX.get_loc(off_days[0][1]); t1O = C.IDX.get_loc(off_end)
FULL = (0.001425 + 0.002, 0.001425 + 0.003 + 0.002)   # 無折扣手續費＋單邊 0.2% 滑價
for lab, cost in (('normal', C.NORMAL), ('slip0.2', C.rates_slip(0.002)), ('fullfee_slip0.2', FULL)):
    navA = C.tsim([plOFFab], *cost, t0O, t1O, xrules=st['XR'])[0]
    navV = IT.run_spec(SPEC, cost, max_m=None, t_end=off_end)[0]
    out.setdefault('cost', {})[lab] = {'OFF_AB0': blk(navA), 'V08': blk(navV)}
    print('cost', lab, out['cost'][lab], round(time.time() - T0), flush=True)

# ---------- 6) 研究員第 9 節建議：同一張名單在官方成交日起 3 個開盤分次下單（未經測試） ----------
def shifted_plan(days, q, ab):
    outp = []; held = set()
    for m, d in days:
        c = C.cross(d, m, 0, ab)
        if c['el'].sum() < 50: continue
        e = c['e']
        for _ in range(q):
            e2 = C.next_good(e)
            if e2 is None: break
            e = e2
        gu = K.gap_up_open(e)
        cand = C.order_of(c['sc'], c['y3'], c['el'])
        pick = []
        for j in cand:
            if gu[j] and j not in held: continue
            if not (np.isfinite(K.On[e][j]) and K.TRADED[e][j]): continue
            pick.append(int(j))
            if len(pick) == 10: break
        outp.append((e, np.array(pick, int), np.where(c['el'])[0], d, m))
        held = set(pick)
    return outp
for ab in (False, True):
    pls3 = [shifted_plan(off_days, q, ab) for q in range(3)]
    nav3, navs3, lg3 = C.tsim(pls3, *C.NORMAL, t0O, t1O, xrules=st['XR'], log=True)
    navS = C.tsim([plOFFab if ab else plOFF], *C.NORMAL, t0O, t1O, xrules=st['XR'], log=True)
    r = dict(split3=blk(nav3), single=blk(navS[0]),
             paired={k: C.paired(nav3, navS[0], WO, k) for k in ('ALL', 'SEL', 'POST', 'HOLD')},
             cap_split3=C.capacity(lg3['trades'], nav3, t0O), cap_single=C.capacity(navS[2]['trades'], navS[0], t0O))
    out[f'split3_sameList_ab{int(ab)}'] = r
    print('split3', ab, r['split3']['ALL'], r['single']['ALL'], {k: round(v['mean_pp'], 3) for k, v in r['paired'].items()}, round(time.time() - T0), flush=True)

# ---------- 7) 容量：POST 期 J+1 單日也很少超過 1%（研究員只報 SEL 38.6%）----------
o = json.loads((HERE / 'oos.json').read_text(encoding='utf-8'))
out['capacity_researcher'] = o['selected_detail']['capacity']
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))

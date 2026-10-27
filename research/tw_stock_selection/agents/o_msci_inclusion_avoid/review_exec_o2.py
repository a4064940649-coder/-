#!/usr/bin/env python3
"""反方審查（成本與可成交角度）o_msci_inclusion_avoid。只讀 import 研究員程式，不寫入研究員檔案。
輸出：scratchpad/review_exec_o2_out.json
檢查：
 1. 重現 B2 vs 原版（n10、一般成本）判定窗／跨循環。
 2. 成本拆解：零成本、壓力成本、E 日收盤額外滑價 0.5%。
 3. 0050 進場時點：研究員以 E 收盤價近似 E+1 開盤買進（多拿 E→E+1 隔夜）；改成 E+1 收盤才進場（下界）。
 4. 每筆 B2 賣出：下一換股期是否又買回（額外來回成本），E 日是否漲停／跌停、E 日成交值。
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import oa_lib as O
import vers as W
K, V = O.K, O.V
IDX, COLS, CI, N = O.IDX, O.COLS, O.CI, O.N
E50 = O.E50
OUT = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_exec_o2_out.json'
T00 = time.time()
out = {}


def sim(plan, buy, sell, t0, t1, actions, b50, e50_delay=False, extra_E_slip=0.0):
    """oa_lib.simulate_mod 的複本，加兩個開關（只影響 E 日賣出與 0050 進場）。"""
    reb = {p[0]: p for p in plan}; reb_days = sorted(reb)
    val = np.zeros(N); cash = 1.0; e50v = 0.0
    nav = np.empty(t1 - t0 + 1); log = []
    pend_sell = {}; pend_buy = []
    for k, v in actions.items(): pend_sell.setdefault(k, []).extend(v)
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t][1]
            if e50v > 0: cash += e50v * (1 - b50[1]); e50v = 0.0
            if pend_buy: cash += sum(x[0] for x in pend_buy); pend_buy = []
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            gap = np.where(tr, K.GROW[t] * K.On[t] / np.where(tr, K.Cn[t], 1.0), 1.0)
            h = val * gap; frz = (h > 0) & ~tr; htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash; kk = len(pk)
            def cost_of(T):
                tgt = np.zeros(N); tgt[pk] = T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt / kk
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * kk + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            newval = np.where(frz, h, 0.0); newval[pk] += lo
            oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc); cash = 0.0
        else:
            if t > t0:
                val = val * K.GROW[t]
                if e50v > 0: e50v *= E50[t] / E50[t - 1]
            if pend_buy:
                for (amt, mode, ranked, excl) in pend_buy:
                    g = 1.0 if e50_delay else E50[t] / E50[t - 1]
                    e50v += amt / (1 + b50[0]) * g
                    log.append(('buy_0050', str(IDX[t].date()), mode))
                pend_buy = []
        if t in pend_sell:
            nxt = [d for d in reb_days if d > t]; nxt = nxt[0] if nxt else t1 + 1
            for (j, mode, ranked, excl) in pend_sell.pop(t):
                if val[j] <= 0: continue
                if O.limit_dn_locked(t, j):
                    if t + 1 < nxt: pend_sell.setdefault(t + 1, []).append((j, mode, ranked, excl))
                    continue
                amt = val[j] * (1 - sell - extra_E_slip); val[j] = 0.0
                log.append(('sell_E', str(IDX[t].date()), COLS[j], mode))
                if t + 1 < nxt and t + 1 <= t1: pend_buy.append((amt, mode, ranked, excl))
                else: cash += amt
        nav[t - t0] = val.sum() + cash + e50v + sum(x[0] for x in pend_buy)
    return pd.Series(nav, index=IDX[t0:t1 + 1]), log


n = 10
T0, T1 = W.T0, W.T1_FULL
ew = W.ew_nav(n, T1)
pl, _ = O.plan_mod(W.S, n)
acts, hits = O.build_actions(pl, W.INC, ('STD',), 'e50')
out['hits'] = hits.astype(str).values.tolist()
WINS = {'oos_2019_2026': ('2019-01-01', str(IDX[T1].date())), 'all': (str(IDX[T0].date()), str(IDX[T1].date()))}

def cagr(nav, a, b):
    return W.seg(nav, ew, a, b)['CAGR']

variants = {}
for cname in ('normal', 'stress', 'zero'):
    if cname == 'zero': (b, s), b50 = (0.0, 0.0), (0.0, 0.0)
    else: (b, s), b50 = W.COST[cname]
    nb, _ = sim(pl, b, s, T0, T1, {}, b50)
    for vname, kw in [('B2', {}), ('B2_e50_delay', {'e50_delay': True}), ('B2_extraEslip0.5', {'extra_E_slip': 0.005}), ('B2_delay+slip', {'e50_delay': True, 'extra_E_slip': 0.005})]:
        if cname == 'zero' and vname != 'B2': continue
        if cname == 'stress' and vname not in ('B2', 'B2_delay+slip'): continue
        nm, lg = sim(pl, b, s, T0, T1, acts, b50, **kw)
        variants[f'{vname}|{cname}'] = {w: {'B2': round(cagr(nm, *ab), 3), 'base': round(cagr(nb, *ab), 3), 'diff_pp': round(cagr(nm, *ab) - cagr(nb, *ab), 3)} for w, ab in WINS.items()}
        print(vname, cname, variants[f'{vname}|{cname}'], round(time.time() - T00), flush=True)
out['variants'] = variants

# 每筆事件：下一期是否買回、E 日狀態
ex = [p[0] for p in pl]
rows = []
for h in hits.itertuples():
    k = [i for i, p in enumerate(pl) if p[0] == IDX.get_loc(h.exec)][0]
    j = CI[h.code]; iE = IDX.get_loc(h.E)
    nxt_pick = COLS[j] in {COLS[x] for x in pl[k + 1][1]} if k + 1 < len(pl) else None
    ref = K.ref_price(iE)[j]; c = K.Cn[iE, j]
    lim = 0.07 if h.E < pd.Timestamp('2015-06-01') else 0.10
    e1 = ex[k + 1] if k + 1 < len(ex) else None
    rows.append(dict(E=str(h.E.date()), code=h.code, name=h.name, rebought_next_period=nxt_pick,
                     E_ret=round(float(c / ref - 1), 4), near_limit_up=bool(c / ref - 1 > lim - 0.015), near_limit_dn=bool(c / ref - 1 < -lim + 0.015),
                     E_value_億=round(float(K.V.iat[iE, j]) / 1e8, 1), tdays_E_to_next_exec=(int(e1 - iE) if e1 else None),
                     src3=bool(O.SRC3[0] <= h.E <= O.SRC3[1])))
out['per_event_exec'] = rows
for r in rows: print(r)
# 額外成本：每筆 B2 相對原版多付的交易成本（一般成本）
b_, s_ = W.COST['normal'][0]; b50 = W.COST['normal'][1]
for r in rows:
    r['extra_cost%'] = round(100 * ((b50[0] + b50[1]) + ((s_ + b_) if r['rebought_next_period'] else 0.0)), 3)
out['extra_cost_mean%'] = round(float(np.mean([r['extra_cost%'] for r in rows])), 3)
out['spec_assumed_extra_cost%'] = 0.69
json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=str)
print('extra cost mean', out['extra_cost_mean%'], 'done', round(time.time() - T00))

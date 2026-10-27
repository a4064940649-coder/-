"""主策略：用 main_v2/v2_core（正式版 gapfix、排除金融、official 同分）產生名單；自寫 B2 日頻引擎。"""
import sys, os, pickle, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parents[1]
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
Cn, On, TR, G = K.Cn, K.On, K.TRADED, K.GROW
E50 = K.E50.to_numpy()
FEE = K.FEE

def plans():
    f = HERE / 'plans.pkl'
    if f.exists(): return pickle.loads(f.read_bytes())
    YL, Y5, _ = V.load_revenue()
    S = V.V2('gapfix', YL, Y5, exfin=True)
    t0, t1, sd = S.bounds()
    out = {'t0': t0, 't1': t1, 'sd': sd}
    for n in (10, 30): out[n] = S.plan(n, 'official')
    # A 系列需要候選排序：存每個訊號日的完整排序與開盤漲停
    out['cand'] = {}
    for d in S.sig_days:
        cs = S.cross_section(d)
        if cs['el'].sum() < 50: continue
        out['cand'][cs['e']] = (S.order(cs, 'official')[:int(cs['el'].sum())], K.gap_up_open(cs['e']))
    ew, _ = K.simulate([(p[0], p[2]) for p in out[10]], 0, 0, t0, t1)
    out['ew'] = ew
    f.write_bytes(pickle.dumps(out))
    return out

def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)

def limit_down_price(ref, lim):
    raw = ref * (1 - lim); t = tick(raw)
    return np.ceil(raw / t - 1e-9) * t

def locked_down(t, j):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    ref = K.ref_price(t)[j]
    if not np.isfinite(ref) or not np.isfinite(Cn[t, j]): return False
    return Cn[t, j] <= limit_down_price(ref, lim) + 1e-9

def costs(kind):
    """回傳 (股買, 股賣, 0050買, 0050賣)。"""
    sb, ss = FEE + 0.001, FEE + 0.003 + 0.001
    eb, es = FEE, FEE + 0.001
    if kind == 'stress':
        sb += 0.003; ss += 0.003; eb += 0.001; es += 0.001
    return sb, ss, eb, es

def simulate(plan, kind, t0, t1, sells=None, mode='0050', cand=None, excl=None):
    """plan=[(e, picks,...)]；sells={t_E: set(col)}（E 日；不能賣則延後到下一成交日前一日為止）。
    mode='0050'（B2）：E 收盤賣出，以 E 收盤近似買 0050，持有到下一成交日前一日收盤賣出；
    'repl'（B1）：E+1 開盤用當月候選排序的下一名遞補（非本次納入股、非持股、非開盤漲停）；'none'：原版。
    excl={t_E: set(col)} 本次納入股（遞補排除用）。"""
    sb, ss, eb, es = costs(kind)
    ex = [p[0] for p in plan]
    reb = {p[0]: p[1] for p in plan}
    nxt = {ex[i]: (ex[i + 1] if i + 1 < len(ex) else None) for i in range(len(ex))}
    val = np.zeros(N); cash = 1.0; e50 = 0.0
    nav = np.empty(t1 - t0 + 1)
    pend = {}; q = []; log = []; cur = None; last = t1; e50_fee = 0.0
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t]; cur = t; nx = nxt[t]; last = (nx - 1) if nx is not None else t1
            tr = TR[t] & np.isfinite(On[t])
            gap = np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            cash -= e50_fee; e50_fee = 0.0
            htr = np.where(frz, 0.0, h); Vt = htr.sum() + cash
            k = len(pk)
            def cost_of(T):
                tgt = np.zeros(N); tgt[pk] = T
                return sb * np.clip(tgt - htr, 0, None).sum() + ss * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * k + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            newval = np.where(frz, h, 0.0); newval[pk] += lo
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc); cash = 0.0
            pend = {}; q = []
            if sells and mode != 'none':
                for tt in range(t, last + 1):
                    for j in sells.get(tt, ()):
                        pend.setdefault(j, tt)
        elif t > t0:
            val = val * G[t]
            if e50 > 0: e50 *= E50[t] / E50[t - 1]
            # B1 遞補：開盤買
            for (tb, amt, tE) in [x for x in q if x[0] == t]:
                order, _ = cand[cur]
                gu = K.gap_up_open(t)
                bad = excl.get(tE, set()) if excl else set()
                pick = None
                for j in order:
                    if val[j] > 0 or j in bad or gu[j] or not (TR[t, j] and np.isfinite(On[t, j])): continue
                    pick = j; break
                if pick is None: cash += amt; continue
                val[pick] += amt / (1 + sb) * Cn[t, pick] / On[t, pick]
                log.append(dict(t=t, j=pick, kind='buy', tE=tE))
            q = [x for x in q if x[0] != t]
        for j in sorted(pend):
            tE = pend[j]
            if t < tE: continue
            if val[j] <= 0: del pend[j]; continue
            if (not TR[t, j]) or locked_down(t, j):
                if t >= last: del pend[j]
                continue
            proceeds = val[j] * (1 - ss)
            log.append(dict(t=t, j=j, tE=tE, kind='sell', val=float(val[j]), last=last, cur=cur))
            val[j] = 0.0; del pend[j]
            if mode == '0050' and t < last: e50 += proceeds * (1 - eb)
            elif mode == 'repl' and t < last: q.append((t + 1, proceeds, tE))
            else: cash += proceeds
        if e50 > 0 and t == last:
            # 0050 在下一成交日開盤賣出（價格以前一日收盤近似）：成本記在成交日
            e50_fee = e50 * es; cash += e50; e50 = 0.0
        nav[t - t0] = val.sum() + cash + e50
    return pd.Series(nav, index=IDX[t0:t1 + 1]), log

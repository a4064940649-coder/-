# 紙上零衝擊引擎（clean-room；依 o4 FINDINGS 376～382 行文字＋cons_main 文字規格）
import numpy as np, pandas as pd, math
import xdata

def run_paper(W, plan, end, cost='normal', merger='cons_main', start_base=None):
    """plan: [(s,e,picks)]；回傳日淨值（W.dates 長度）；start_base：基期 idx（預設第一個訊號日）"""
    if isinstance(cost, tuple): bf, sf = cost[1], cost[2]
    elif cost == 'normal': bf = 0.001425 * 0.6 + 0.001; sf = bf + 0.003
    else: bf = sf = 0.005
    G, O, C, TR, CF = W.G, W.O, W.C, W.TR, W.CF
    emap = {e: pk for (s, e, pk) in plan}
    st = plan[0][0] if start_base is None else start_base
    vals = {}; cash = 1.0; claims = []
    nav = np.full(len(W.dates), np.nan); nav[st] = 1.0
    for t in range(st + 1, end + 1):
        keep = []
        for cl in claims:
            if cl['kind'] == 'cash' and t >= cl['pay']: cash += cl['v']
            elif cl['kind'] == 'stock' and cl['s'] is not None and t >= cl['s']:
                sj = cl['succ']; vals[sj] = vals.get(sj, 0.0) + cl['sh'] * C[t, sj] * (O[t, sj] / C[t, sj] if t in emap else 1.0)
                cl['added'] = t
            else: keep.append(cl)
        claims = keep
        if merger == 'cons_main':
            for j in list(vals):
                d = W.XC.get(j)
                if d is None or t != d['trig']: continue
                v = vals.pop(j); lc = d['last_close']; sh = v / lc; ty = d['type']
                if ty == 'cash': claims.append(dict(kind='cash', v=sh * (d['cash'] or lc), pay=d['pay']))
                elif ty in ('conversion', 'swap', 'swap_ur') and d['succ'] is not None and d['ratio']:
                    claims.append(dict(kind='stock', v=v, sh=sh * d['ratio'], succ=d['succ'], s=d['s']))
                elif ty == 'mixed' and d['succ'] is not None:
                    if d['cash']: claims.append(dict(kind='cash', v=sh * d['cash'], pay=d['pay']))
                    if d['ratio']: claims.append(dict(kind='stock', v=sh * max(lc - (d['cash'] or 0), 0), sh=sh * d['ratio'], succ=d['succ'], s=d['s']))
                    if d['pref_ratio'] and d['pref_value']: claims.append(dict(kind='cash', v=sh * d['pref_ratio'] * d['pref_value'], pay=(d['s'] or d['pay'])))
                elif ty == 'distress': pass
                else: claims.append(dict(kind='cash', v=v, pay=d['pay']))
        if t in emap:
            pk = emap[t]
            for j in list(vals):
                if TR[t, j]: vals[j] *= G[t, j] * O[t, j] / C[t, j]
            if merger in ('orig', 'freeze_x'):
                for j in list(vals):
                    if TR[t, j]: continue
                    x = W.XO.get(j)
                    if x is None or not (x['rel'] <= t < x['vu']): continue
                    v = vals.pop(j)
                    if x['kind'] in ('mcash', 'one'): cash += v * (x['ratio'] if x['kind'] == 'mcash' else 1.0)
            trad = {j: v for j, v in vals.items() if TR[t, j]}
            total = sum(trad.values()) + cash
            k = len(pk); pks = set(pk)
            def need(T):
                c = 0.0
                for j in pk:
                    v = trad.get(j, 0.0)
                    c += bf * max(T - v, 0) + sf * max(v - T, 0)
                for j, v in trad.items():
                    if j not in pks: c += sf * v
                return k * T + c
            lo, hi = 0.0, total / k
            for _ in range(60):
                mid = (lo + hi) / 2
                if need(mid) <= total: lo = mid
                else: hi = mid
            T = lo
            cash = total - need(T)
            for j in list(trad):
                if j not in pks: vals.pop(j)
            for j in pk: vals[j] = T
            for j in list(vals):
                if TR[t, j]: vals[j] *= C[t, j] / O[t, j]
        else:
            for j in list(vals):
                if TR[t, j]: vals[j] *= G[t, j]
        nav[t] = cash + sum(vals.values()) + sum(cl['v'] for cl in claims)
    return nav

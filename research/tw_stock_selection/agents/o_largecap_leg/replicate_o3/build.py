"""建所有組合的目標權重與淨值（一般、壓力、零成本）。"""
import sys, pickle, json, time
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
from rlib import *

SC = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
D = pickle.load(open(SC + 'rep_o3_data.pkl', 'rb'))
OUT = '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3/'

sigs = [t for t in D.sigs if D.next_day(t) <= END and t < END]
main, L, Lx, pool, poolx, top1 = {}, {}, {}, {}, {}, {}
hm, hl, hlx, hp, hpx = set(), set(), set(), set(), set()
for t in sigs:
    d = D.next_day(t)
    pm = D.pick(D.rank(D.main_pool(t)), 10, hm, d); main[t] = pm; hm = set(pm)
    P = D.lc_pool(t); rk = D.rank(P)
    pl = D.pick(rk, 5, hl, d); L[t] = pl; hl = set(pl)
    pool[t] = list(P.index); top1[t] = P.index[0]
    PX = D.lc_pool(t, drop=('2330',)); rkx = D.rank(PX)
    plx = D.pick(rkx, 5, hlx, d); Lx[t] = plx; hlx = set(plx)
    poolx[t] = list(PX.index)


def ew(codes, held, d):
    ok = [c for c in codes if c in held or not D.lu_blocked(c, d)]
    return ok


def build_pool_ew(pmap):
    out = {}; held = set()
    for t in sigs:
        ok = ew(pmap[t], held, D.next_day(t)); out[t] = ok; held = set(ok)
    return out


poolE = build_pool_ew(pool); poolxE = build_pool_ew(poolx)


def mix(*parts):
    tg = {}
    for t in sigs:
        w = {}
        for wt, m in parts:
            v = m[t] if isinstance(m[t], list) else [m[t]]
            for c in v:
                w[c] = w.get(c, 0) + wt / len(v)
        tg[t] = w
    return tg


T2330 = {t: ['2330'] for t in sigs}
TG = {
    'MAIN': mix((1.0, main)),
    'L': mix((1.0, L)),
    'LX': mix((1.0, Lx)),
    'POOL': mix((1.0, poolE)),
    'POOLX': mix((1.0, poolxE)),
    'COMBO': mix((0.7, main), (0.3, L)),
    'TSMC30': mix((0.7, main), (0.3, T2330)),
    'PLACEBO': mix((0.7, main), (0.3, poolE)),
    'TOP1': mix((0.7, main), (0.3, top1)),
}
pickle.dump({'sigs': sigs, 'main': main, 'L': L, 'Lx': Lx, 'pool': pool, 'poolx': poolx, 'poolE': poolE, 'top1': top1, 'TG': TG},
            open(SC + 'rep_o3_plans.pkl', 'wb'))
navs, turns = {}, {}
for k, tg in TG.items():
    for cost in (['normal', 'stress3', 'zero', 'old05'] if k in ('MAIN', 'COMBO', 'TSMC30', 'L', 'POOL', 'LX', 'POOLX') else ['normal', 'stress3']):
        t0 = time.time()
        nav, to = D.simulate(tg, cost)
        navs[f'{k}|{cost}'] = nav; turns[f'{k}|{cost}'] = to
        print(k, cost, round(nav.iloc[-1], 3), round(to, 3), round(time.time() - t0, 1), flush=True)
pd.DataFrame(navs).to_parquet(OUT + 'navs_rep.parquet')
json.dump(turns, open(OUT + 'turns_rep.json', 'w'), indent=1)

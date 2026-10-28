import sys, pickle, time; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/main_v2'); import v2_core as V2; sys.path.pop(0)
assert list(V2.COLS) == list(K.COLS)
SP = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
def avg_nav(navs):
    r = pd.concat([n.pct_change() for n in navs], axis=1).mean(1).fillna(0)
    return (1 + r).cumprod()
def ab(kw=None, cost='normal', worst=False, seeds=range(40), tag=''):
    kw = kw or {}
    b, s = K.rates(cost) if isinstance(cost, str) else cost
    navs = []; plans = []; nun = []
    for sd in ([None] if worst else seeds):
        U = V2.make_unavail('worst') if worst else V2.make_unavail('mc', sd)
        pl = K.build_plan(unavail=U, **kw); plans.append(pl)
        nun += [p['nunav'] for p in pl]
        nav, st = K.run(pl, b, s); navs.append(nav)
    return avg_nav(navs), plans, np.mean(nun)
if __name__ == '__main__':
    t0 = time.time()
    D = pickle.load(open(SP + 'rep_o4_main.pkl', 'rb'))
    nav, plans, nun = ab()
    print('AB nunav', nun, time.time() - t0)
    o = K.summary(nav, plans[0], D['ew'], full=True)
    for k, v in o.items():
        if k != 'yearly': print(k, v)
    pickle.dump(dict(nav=nav, plans=plans), open(SP + 'rep_o4_ab.pkl', 'wb'))

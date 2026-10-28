import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ind_core as C, numpy as np, pandas as pd
H = pathlib.Path(__file__).resolve().parent
C.OPT['rev'] = 'anchor'
VERS = {'V01': ([1, 11], 0, 0, 0), 'V02': ([1, 8, 15], 0, 0, 0), 'V03': ([1, 6, 11, 16], 0, 0, 0), 'V04': ([1, 11], 1, 0, 0), 'V05': ([1, 8, 15], 1, 0, 0),
        'V06': ([1, 6, 11, 16], 1, 0, 0), 'V07': ([1, 11], 0, 0, 1), 'V08': ([1, 6, 11, 16], 0, 0, 1), 'V09': ([1, 11], 0, 1, 0), 'V10': ([1, 6, 11, 16], 0, 1, 0),
        'V11': ([1, 6, 11, 16], 0, 1, 1), 'V12': ([1, 3, 5], 0, 0, 0)}
def plans_for(js, b1=False, seed=0, w=0.5):
    out = []
    ref = {}
    if b1:
        ab1 = C.AB(seed)
        for p in C.ANCH:
            s = C.J(p, 1)
            if s in C.POS:
                ref[p] = C.rank(p, s, ab1(p, s), ab1.tag, w)['prr']
    for j in js:
        ab = C.AB(seed)
        days = [(p, C.J(p, j)) for p in C.ANCH]
        rf = (lambda p: ref.get(p)) if (b1 and j > 1) else None
        if rf is not None:
            rf0 = rf; rf = lambda p, _r=rf0: _r(p)
        out.append(C.plan(days, ab, w=w, revfix_fn=rf) if rf is None else plan_rf(days, ab, ref, w))
    return out
def plan_rf(days, ab, ref, w):
    # b1：該錨定月沒有 j=+1 橫斷面時照常重排
    out = []; held = set()
    for p, s in days:
        if s is None or s not in C.POS: continue
        U = ab(p, s); rf = ref.get(p)
        r = C.rank(p, s, U, ab.tag, w, rf)
        if r['nel'] < 50: continue
        gu = C.gapup(r['e']); pk = []
        for j in r['order']:
            if gu[j] and j not in held: continue
            pk.append(int(j))
            if len(pk) == 10: break
        out.append(dict(e=r['e'], pk=np.array(pk, int), p=p, s=s, t=r['t'], order=r['order'])); held = set(pk)
    return out
def run(js, a1=0, b1=0, c1=0, costs=C.NORMAL, seed=0, w=0.5):
    pls = plans_for(js, b1, seed, w)
    t0 = min(pl[0]['t'] for pl in pls)
    return C.tsim(pls, *costs, t0, C.TEND, c1=bool(c1), a1=bool(a1))
def wstats(nav, ref=None, a=None, b=None):
    ex = C.exw(nav, C.OFFSD); am = {C.OFF[p]: p for p in C.ANCH}
    pr = pd.PeriodIndex([am[d] for d in ex.index], freq='M')
    m = np.ones(len(ex), bool)
    if a: m &= pr >= pd.Period(a, 'M')
    if b: m &= pr <= pd.Period(b, 'M')
    o = dict(mean=float(ex[m].mean() * 100), sd=float(ex[m].std(ddof=1) * 100), t=C.tst(ex[m]), n=int(m.sum()))
    if ref is not None:
        d = (C.wret(nav, C.OFFSD) - C.wret(ref, C.OFFSD))[m]
        o.update(pd_mean=float(d.mean() * 100), pd_t=C.tst(d))
    return o
if __name__ == '__main__':
    base, _ = run([1])
    navs = {'J+1': base}
    sb = wstats(base, None, None, '2018-12')
    print('J+1 SEL', sb)
    res = {'J+1': sb}
    for v, (js, a1, b1, c1) in VERS.items():
        nav, lg = run(js, a1, b1, c1)
        navs[v] = nav
        s = wstats(nav, base, None, '2018-12'); s['sd_chg_pct'] = (s['sd'] / sb['sd'] - 1) * 100; s['turn'] = lg['turnover']
        res[v] = s
        print(v, {k: round(x, 4) if isinstance(x, float) else x for k, x in s.items()}, flush=True)
    json.dump(res, open(H / 'ind_tr.json', 'w'), indent=1)
    pd.DataFrame(navs).to_parquet(H / 'ind_tr_navs.parquet')

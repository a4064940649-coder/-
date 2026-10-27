import sys, pickle, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
from rlib import *
SC = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
OUT = '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3/'
D3 = pickle.load(open(SC + 'rep_o3_data_v3.pkl', 'rb'))
sigs = [t for t in D3.sigs if D3.next_day(t) <= END and t < END]
CUT = pd.Timestamp('2021-09-24')
VAR = sys.argv[1] if len(sys.argv) > 1 else 'a'


def disp_v3(t):
    if t < CUT:
        h = D3.dsp_hist
        return set(h.code[(h.start <= t) & (h.end >= t)])
    return in_disposal(D3.dsp, t)


def frame(t):
    nd = D3.next_day(t)
    df = pd.DataFrame({'v60': D3.v60.loc[t], 'close': D3.close.loc[t], 'r52': D3.r52.loc[t], 'nopen': D3.open.loc[nd]})
    df['yoy3'] = yoy3_at(t, D3.y, D3.r5).reindex(df.index)
    df['disp'] = df.index.isin(disp_v3(t))
    return df


def extra_excl(p, k):
    """D_hi：排除池中 52 週比最高的 k 檔（同值依池內成交值順序）。"""
    if k <= 0: return set()
    q = p.copy(); q['o'] = range(len(q))
    if VAR in ('a', 'b'):
        q = q.sort_values(['r52', 'o'], ascending=[False, True])
    else:
        q['code'] = q.index; q = q.sort_values(['r52', 'code'], ascending=[False, True])
    return set(q.index[:k])


def main_pool3(t):
    df = frame(t)
    p = df[(df.v60 >= 1e7) & (df.close >= 10)].sort_values('v60', ascending=False).head(300)
    p = p[~p.disp & (p.nopen > 0) & p.yoy3.notna() & p.r52.notna()]
    k = D3.dhi_extra.get(t, 0) if t < CUT else 0
    if VAR in ('b',):
        ex = extra_excl(p, k); p = p[~p.index.isin(D3.fin)]; p = p[~p.index.isin(ex)]
    else:
        p = p[~p.index.isin(D3.fin)]; ex = extra_excl(p, k); p = p[~p.index.isin(ex)]
    return p, ex, df


def lc_pool3(t, df, ex, n=50):
    ok = (df.v60 >= 1e7) & (df.close >= 10) & df.yoy3.notna() & df.r52.notna() & ~df.disp & (df.nopen > 0)
    ok &= ~df.index.isin(D3.fin) & ~df.index.isin(D3.foreign) & ~df.index.isin(list(ex))
    return df[ok].sort_values('v60', ascending=False).head(n)


# X 規則
mc = D3.mc
execs = [D3.next_day(t) for t in sigs]
def first_exec_after(day):
    for e in execs:
        if e >= day: return e
    return pd.Timestamp('2100-01-01')
xr = {}
for _, r in mc.iterrows():
    dd = pd.Timestamp(r.delist_date)
    if r.type == 'cash':
        mult = float(r.cash) / float(r.last_close) if pd.notna(r.cash) else 1.0
        rel = first_exec_after(dd + pd.Timedelta(days=14)) if pd.isna(r.pay_date) else first_exec_after(pd.Timestamp(r.pay_date))
    elif r.type == 'failure':
        mult = 0.0; rel = first_exec_after(dd)
    else:
        mult = 1.0; rel = first_exec_after(dd)
    xr[r.code] = (rel, mult)

main, L, top1, exs = {}, {}, {}, {}
hm, hl = set(), set()
for t in sigs:
    d = D3.next_day(t)
    p, ex, df = main_pool3(t)
    pm = D3.pick(D3.rank(p), 10, hm, d); main[t] = pm; hm = set(pm)
    P = lc_pool3(t, df, ex if VAR != 'c2' else set())
    pl = D3.pick(D3.rank(P), 5, hl, d); L[t] = pl; hl = set(pl)
    top1[t] = P.index[0]; exs[t] = (ex, list(P.index))


def mix(*parts):
    tg = {}
    for t in sigs:
        w = {}
        for wt, m in parts:
            v = m[t] if isinstance(m[t], list) else [m[t]]
            for c in v: w[c] = w.get(c, 0) + wt / len(v)
        tg[t] = w
    return tg


ref = pd.read_parquet(AG + 'o_infra_v3_rebuild/navs_P1.parquet')['P1|R1|X1|Dhi|EXFIN|n10|normal']
navM, _ = D3.simulate(mix((1.0, main)), 'normal', xrule=xr)
print(VAR, 'MAIN V3 maxrel', float((navM / ref - 1).abs().max()), navM.iloc[-1], ref.iloc[-1], flush=True)
if len(sys.argv) > 2:
    B = bench0050(D3.nidx); rb = mret(B, D3.sigs)
    res = {}
    navs = {'MAIN': navM}
    navs['COMBO'], _ = D3.simulate(mix((0.7, main), (0.3, L)), 'normal', xrule=xr)
    navs['TSMC30'], _ = D3.simulate(mix((0.7, main), (0.3, {t: ['2330'] for t in sigs})), 'normal', xrule=xr)
    navs['L'], _ = D3.simulate(mix((1.0, L)), 'normal', xrule=xr)
    navs['L0'], _ = D3.simulate(mix((1.0, L)), 'zero', xrule=xr)
    navs['POOL'], _ = D3.simulate(mix((1.0, {t: exs[t][1] for t in sigs})), 'zero', xrule=xr)
    for k, nv in navs.items():
        r = mret(nv, D3.sigs) - rb
        res[k] = {s: (round(seg_nav_cagr(nv, None, s) * 100, 2), round(tstat(seg_mask(r, s)), 3)) for s in ['SEL', 'VAL', 'VAL_A', 'HOLD', 'ALL']}
    for a, b in [('L', 'POOL'), ('L0', 'POOL')]:
        dd = mret(navs[a], D3.sigs) - mret(navs[b], D3.sigs)
        res[a + '-' + b] = {s: (round(seg_mask(dd, s).mean() * 100, 3), round(tstat(seg_mask(dd, s)), 2)) for s in ['SEL', 'VAL', 'VAL_A', 'HOLD', 'ALL']}
    dd = mret(navs['COMBO'], D3.sigs) - mret(navs['MAIN'], D3.sigs)
    res['COMBO-MAIN'] = {s: (round(seg_mask(dd, s).mean() * 100, 3), round(tstat(seg_mask(dd, s)), 2)) for s in ['SEL', 'VAL', 'HOLD', 'ALL']}
    for k, v in res.items(): print(k, v)
    json.dump(res, open(OUT + f'v3_rep_{VAR}.json', 'w'), indent=1)

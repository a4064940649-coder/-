import sys, pickle, json, math
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
from rlib import *
SC = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
OUT = '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3/'
D = pickle.load(open(SC + 'rep_o3_data.pkl', 'rb'))
PL = pickle.load(open(SC + 'rep_o3_plans.pkl', 'rb'))
N = pd.read_parquet(OUT + 'navs_rep.parquet')
sigs = D.sigs
B = bench0050(D.nidx)
ref = pd.read_parquet(AG + 'integrate_o2/navs.parquet')
res = {}
res['check_MAIN_vs_MAIN_R_maxrel'] = float((N['MAIN|normal'] / ref['MAIN_R'] - 1).abs().max())
res['check_TSMC30_vs_TSMC30_R_maxrel'] = float((N['TSMC30|normal'] / ref['TSMC30_R'] - 1).abs().max())


def sleeve_mix(navA, navB, wa=0.7):
    pts = [x for x in sigs if x <= END]
    out = pd.Series(index=navA.index, dtype=float); base = 1.0
    seg_starts = pts
    for i, s in enumerate(seg_starts):
        e = seg_starts[i + 1] if i + 1 < len(seg_starts) else END
        m = (navA.index >= s) & (navA.index <= e)
        v = base * (wa * navA[m] / navA[s] + (1 - wa) * navB[m] / navB[s])
        out[m] = v; base = v[e] if e in v.index else v.iloc[-1]
    out.iloc[0] = 1.0
    return out


N['M70_0050|normal'] = sleeve_mix(N['MAIN|normal'], B)
N['M70_0050|stress3'] = sleeve_mix(N['MAIN|stress3'], B)
segs = ['SEL', 'VAL', 'VAL_A', 'HOLD', 'ALL']
rb = mret(B, sigs)


def block(col):
    nav = N[col]; r = mret(nav, sigs); ex = r - rb
    o = {}
    for s in segs:
        e = seg_mask(ex, s)
        o[s] = {'ann': round(seg_nav_cagr(nav, None, s) * 100, 2), 't': round(tstat(e), 3), 'ex_pp': round(e.mean() * 100, 3),
                'te': round(e.std(ddof=1) * math.sqrt(12) * 100, 2), 'n': int(len(e)), 'mdd': round(seg_dd(nav, s) * 100, 2)}
    return o


res['0050'] = {s: round(seg_nav_cagr(B, None, s) * 100, 2) for s in segs}
for col in N.columns:
    res[col] = block(col)


def pdiff(a, b):
    ra, rbb = mret(N[a], sigs), mret(N[b], sigs); d = ra - rbb
    return {s: {'pp': round(seg_mask(d, s).mean() * 100, 3), 't': round(tstat(seg_mask(d, s)), 2)} for s in segs}


res['COMBO_minus_MAIN'] = pdiff('COMBO|normal', 'MAIN|normal')
res['L_vs_POOL_normal'] = pdiff('L|normal', 'POOL|normal')
res['L_vs_POOL_zero'] = pdiff('L|zero', 'POOL|zero')
res['L_vs_POOL_stress3'] = pdiff('L|stress3', 'POOL|stress3')
res['LX_vs_POOLX_normal'] = pdiff('LX|normal', 'POOLX|normal')
res['LX_vs_POOLX_zero'] = pdiff('LX|zero', 'POOLX|zero')
res['TSMC30_minus_MAIN'] = pdiff('TSMC30|normal', 'MAIN|normal')
res['POOL_vs_0050'] = {s: {'pp': round(seg_mask(mret(N['POOL|normal'], sigs) - rb, s).mean() * 100, 3),
                           't': round(tstat(seg_mask(mret(N['POOL|normal'], sigs) - rb, s)), 2)} for s in segs}

# Shapley t 拆解
def shap(a, b='MAIN|normal'):
    o = {}
    for s in ['SEL', 'VAL']:
        ea = seg_mask(mret(N[a], sigs) - rb, s); eb = seg_mask(mret(N[b], sigs) - rb, s)
        n = len(ea); f = lambda m, sd: m / sd * math.sqrt(n)
        m1, s1, m0, s0 = ea.mean(), ea.std(ddof=1), eb.mean(), eb.std(ddof=1)
        dm = 0.5 * ((f(m1, s0) - f(m0, s0)) + (f(m1, s1) - f(m0, s1)))
        dv = 0.5 * ((f(m0, s1) - f(m0, s0)) + (f(m1, s1) - f(m1, s0)))
        o[s] = {'dt': round(f(m1, s1) - f(m0, s0), 3), 'mean': round(dm, 3), 'vol': round(dv, 3)}
    return o


res['shapley_COMBO'] = shap('COMBO|normal'); res['shapley_TSMC30'] = shap('TSMC30|normal')
# 相關與 β
r2330 = mret(D.adjf['2330'].reindex(D.nidx), sigs)
x = r2330 - rb
eL = mret(N['L|normal'], sigs) - rb; eM = mret(N['MAIN|normal'], sigs) - rb; eC = mret(N['COMBO|normal'], sigs) - rb
cb = {}
for s in ['SEL', 'VAL', 'VAL_A', 'HOLD', 'ALL']:
    a, m, c, xx = seg_mask(eL, s), seg_mask(eM, s), seg_mask(eC, s), seg_mask(x, s)
    beta = lambda y: float(np.cov(y, xx)[0, 1] / np.var(xx, ddof=1))
    cb[s] = {'corr_L_M': round(a.corr(m), 3), 'corr_C_M': round(c.corr(m), 3), 'beta_L': round(beta(a), 2), 'beta_M': round(beta(m), 2)}
res['corr_beta'] = cb
# 事件數、周轉、2330 占比
L = PL['L']; ss = PL['sigs']
def segsigs(s):
    a, b = SEG[s]; return [t for t in ss if pd.Timestamp(a) <= t <= pd.Timestamp(b)]
ev = {s: len(segsigs(s)) for s in segs}
rep = []
for i in range(1, len(ss)):
    if ss[i] in segsigs('SEL'):
        rep.append(len(set(L[ss[i]]) - set(L[ss[i - 1]])) / 5)
ev['L_SEL_replace'] = round(float(np.mean(rep)), 3)
ev['L_2330_share'] = {s: round(float(np.mean(['2330' in L[t] for t in segsigs(s)])), 3) for s in segs}
ev['top1_2330_share'] = {s: round(float(np.mean([PL['top1'][t] == '2330' for t in segsigs(s)])), 3) for s in segs}
ev['turns'] = json.load(open(OUT + 'turns_rep.json'))
res['events'] = ev
json.dump(res, open(OUT + 'stats_rep.json', 'w'), indent=1, ensure_ascii=False)
N.to_parquet(OUT + 'navs_rep2.parquet')
print(json.dumps(res, ensure_ascii=False)[:200])

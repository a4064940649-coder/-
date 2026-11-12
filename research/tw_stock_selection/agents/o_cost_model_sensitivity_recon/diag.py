#!/usr/bin/env python3
"""分析函式（研究員；複製員不得 import）：會計拖累、4.4pp 橋段、Shapley、實際成交價 edge、事前預估成本 ĉ、成本五等分、月份區塊 bootstrap、MDE。
只讀 scratchpad/csr 的引擎輸出與 V7X 資料層陣列；不跑引擎。"""
import sys, math, itertools, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import csr_lib as L; sys.path.pop(0)
SCR = L.SCR
E = L.engine(); D = E['D']; S = E['S']; XE = E['XE']
IDX = D.IDX; COLS = D.COLS
with np.errstate(divide='ignore', invalid='ignore'):
    LA = np.cumsum(np.log(np.where(np.isfinite(D.ADJ) & (D.ADJ > 0), D.ADJ, 1.0)), axis=0)
FAC = ['T', 'X', 'F', 'K', 'I', 'C']
COMPS = ['tax', 'fee', 'tick_amt', 'imp_amt', 'odd_amt', 'stress_amt']

def key6(s): return ''.join(f for f in FAC if f in s) or 'none'
def med(x): return float(np.median(x))

# ---------- 期間 ----------
def perbounds(nav, j, per):
    e = S.end_day('AB_V7_B10', j); nv = nav.loc[:e].dropna(); a, b = L.PER[per]
    s0, s1 = L.seg(nv, a, b, xs=(per in L.XS_START)); return s0, s1, nv
def cagr_per(nav, j, per):
    s0, s1, nv = perbounds(nav, j, per); return S.cagr(nv, s0, s1)
def years_o20(per, j):
    """o20 analyze.years：a＝max(期間起點, 2006-04-01)、b＝min(期間終點, j 終點)。"""
    a, b = L.PER[per]; e = S.end_day('AB_V7_B10', j)
    a = max(pd.Timestamp(a), pd.Timestamp('2006-04-01')); b = min(pd.Timestamp(b), e)
    return (b - a).days / 365.25

# ---------- 成交表加工 ----------
def prep_fills(F, rebs, v, j):
    """一個 (變體, j) 的日誌加上換股歸屬 k、類別 cls、avail、成本。"""
    f = F[(F.variant == v) & (F.j_ == j) & (F.shares > 0)].copy()
    rt = np.array([r[0] for r in rebs]); k = np.searchsorted(rt, f.t.values, side='right') - 1
    assert (k >= 0).all()
    pk = [set(r[1]) for r in rebs]; hb = [set(r[2]) for r in rebs]; av = np.array([r[4] for r in rebs])
    f['k'] = k
    f['cls'] = np.where(f.side < 0, np.where([c not in pk[kk] for c, kk in zip(f.j, f.k)], 'exit', 'rw_sell'),
                        np.where([c in hb[kk] for c, kk in zip(f.j, f.k)], 'rw_buy', 'entry'))
    f['avail'] = av[f.k.values]; f['reb_t'] = rt[f.k.values]; f['date'] = IDX[f.t.values]
    f['cost'] = f[COMPS].sum(axis=1)
    return f

# ---------- 會計拖累（o20 口徑） ----------
def acct_drag(f, rebs, per, j):
    rt = np.array([r[0] for r in rebs]); dates = IDX[rt]; a, b = L.PER[per]
    ks = np.flatnonzero((dates >= a) & (dates <= b))
    g = f[f.k.isin(ks)]
    y = years_o20(per, j); o = {c: float((g[c] / g.avail).sum() / y * 100) for c in COMPS + ['cost']}
    o['turn1_exit'] = float((g.notional[g.cls == 'exit'] / g.avail[g.cls == 'exit']).sum() / y)
    return o

# ---------- B1：分母改成成交日成本前權益、歸屬改成成交日、年數改成年化口徑 ----------
def b1_drag(f, nav, daycost, per, j, c0):
    s0, s1, nv = perbounds(nav, j, per)
    y = (s1 - s0).days / 365.25
    dc = pd.Series(daycost, dtype=float)
    Epre = c0 * nav.reindex(IDX[f.t.values]).values + dc.reindex(f.t.values).fillna(0).values
    w = 1.0 / Epre
    rt_dates = IDX[f.reb_t.values]; a, b = L.PER[per]
    ink = (rt_dates >= a) & (rt_dates <= b)
    infd = (f.date > s0) & (f.date <= s1)
    o = dict(
        B1a=float((f.cost * w)[ink].sum() / years_o20(per, j) * 100),            # 只換分母
        B1ab=float((f.cost * w)[infd].sum() / years_o20(per, j) * 100),          # ＋成交日歸屬
        B1=float((f.cost * w)[infd].sum() / y * 100),                            # ＋年數
        B1_avgnav=float(f.cost[infd].sum() / (c0 * nv.loc[s0:s1].mean()) / y * 100))
    for c in COMPS: o['B1_' + c] = float((f[c] * w)[infd].sum() / y * 100)
    return o

def logdrag(nav, daycost, per, j, c0):
    """ℓ＝Σ_d log(1＋C_d/E_post,d)/Y（d∈(s0,s1]）；E_post＝收盤權益＝c0×nav。"""
    s0, s1, nv = perbounds(nav, j, per); y = (s1 - s0).days / 365.25
    dc = pd.Series(daycost, dtype=float); dc.index = IDX[dc.index.values]
    dc = dc[(dc.index > s0) & (dc.index <= s1)]
    Ep = c0 * nav.reindex(dc.index).values
    return float(np.log1p(dc.values / Ep).sum() / y), float((dc.values / (Ep + dc.values)).sum() / y)

# ---------- Shapley ----------
PERMS = list(itertools.permutations(FAC))
def shapley(vf):
    marg = {f: [] for f in FAC}; cc = []
    for pm in PERMS:
        cur = ''; tot = 0.0
        for f in pm:
            nx = key6(cur + f); d = vf[nx] - vf[key6(cur)]; marg[f].append(d)
            if f != 'T': tot += d
            cur = nx
        cc.append(-tot)
    return {f: -float(np.mean(marg[f])) for f in FAC}, dict(mean=float(np.mean(cc)), min=float(np.min(cc)), max=float(np.max(cc)))

# ---------- 橋段 ----------
BR = ['B1', 'B2', 'U', 'B4', 'B5']
def chain(q):
    """q：dict of per-j arrays：c_acc, c_B1, G, q3(g^R−g^A), q4(g^T−g^A), CCj（逐 j Shapley 成本類）；CCmed（中位數值函數 Shapley）。
    主順序：起點 c_acc → B1 → B2 → U → B4 → B5 → B3 → 終點 CCmed。回傳中位數層各階、三層增量、全部 120 種逐 j 增量排列的範圍。"""
    lv = ['c_acc', 'c_B1', 'G', 'q3', 'q4', 'CCj']
    M = {k: med(q[k]) for k in lv}
    inc = {b: np.asarray(q[lv[i + 1]]) - np.asarray(q[lv[i]]) for i, b in enumerate(BR)}
    main = {b: M[lv[i + 1]] - M[lv[i]] for i, b in enumerate(BR)}
    main['B3'] = q['CCmed'] - M['CCj']
    layers = {b: dict(median_layer=main[b], med_of_inc=med(inc[b]), mean_inc=float(np.mean(inc[b])), maxabs_inc=float(inc[b][np.argmax(np.abs(inc[b]))])) for b in BR}
    layers['B3'] = dict(median_layer=main['B3'])
    rng = {b: [] for b in BR}
    c0 = np.asarray(q['c_acc'])
    for pm in itertools.permutations(BR):
        cur = c0.copy(); m0 = med(cur)
        for b in pm:
            cur = cur + inc[b]; m1 = med(cur); rng[b].append(m1 - m0); m0 = m1
    rngs = {b: dict(min=float(np.min(x)), max=float(np.max(x))) for b, x in rng.items()}
    total = q['CCmed'] - M['c_acc']
    explained = total - main['U']
    return dict(levels_median=M, start=M['c_acc'], end=q['CCmed'], total=total, bridges_main=main, layers=layers, perm_range=rngs,
                residual_U=main['U'], explained_excl_U=explained, j_median_idx={k: int(np.argsort(q[k])[6]) for k in lv})

# ---------- 換股表（開盤價 edge 與實際成交價 edge） ----------
def px_at(t, j):
    o = D.O[t, j]; return o if np.isfinite(o) else D.CF[t - 1, j]
def pend_at(e2, j):
    b = D.O[e2, j]
    return b if np.isfinite(b) else D.CF[e2, j]
def wret(j, e, e2): return float(pend_at(e2, j) * np.exp(LA[e2, j] - LA[e, j]) / px_at(e, j) - 1)

def swaps(f, rebs, v, jj, sigmap, c0, nav, tier):
    """o20 build_tables 的 exit 層換股表（B 配對）＋實際成交價 edge＋ĉ。"""
    rt = np.array([r[0] for r in rebs]); endi = D.TPOS[S.end_day('AB_V7_B10', int(jj))]
    agg = f.groupby(['k', 'j', 'cls'])
    A = {}
    for (kk, c, cl), g in agg:
        nP = int((g.ctx == 'P').sum())
        A[(kk, c, cl)] = dict(notional=float(g.notional.sum()), cost=float(g.cost.sum()), shares=float(g.shares.sum()), nP=nP,
                              dl_td=int((g.t - g.reb_t).max()), dl_cd=int((g.date.values - IDX[g.reb_t.values].values).astype('timedelta64[D]').astype(int).max()),
                              kinds=''.join(sorted(set(g.kind))), t=g.t.values, ref=g.ref.values, px=g.px.values, sh=g.shares.values, nt=g.notional.values)
    BYK = {}
    for (k_, c, cl) in A: BYK.setdefault(k_, []).append((c, cl))
    rows = []
    for kk, r in enumerate(rebs):
        e = r[0]; e2 = rt[kk + 1] if kk + 1 < len(rt) else endi
        if e2 <= e: continue
        sg = sigmap[e]; s = D.TPOS[pd.Timestamp(sg)]
        exits = sorted({c for (c, cl) in BYK.get(kk, []) if cl == 'exit'})
        ents = sorted({c for (c, cl) in BYK.get(kk, []) if cl == 'entry'})
        if not exits or not ents:
            for c in exits: rows.append(dict(variant=v, j=jj, k=kk, signal=sg, e=e, code=COLS[c], edge_B=np.nan))
            continue
        def legret(a, c, use='ref'):
            p = a['ref'] if use == 'ref' else a['px']
            ri = pend_at(e2, c) * np.exp(LA[e2, c] - LA[a['t'], c]) / p - 1
            return float(np.sum(a['nt'] * ri) / np.sum(a['nt']))
        Eq_d = c0 * float(nav.iloc[nav.index.get_loc(IDX[s])]) if IDX[s] in nav.index else np.nan
        nk = len(r[1])
        er = []
        for c in ents:
            a = A[(kk, c, 'entry')]
            er.append(dict(r=wret(c, e, e2), rf=legret(a, c), rx=legret(a, c, 'px'), rate=a['cost'] / a['notional'], nP=a['nP'], dl=a['dl_td'],
                           chat=chat_rate(s, c, +1, Eq_d / nk * (1 - XE.SP['frictions']['cash']), None)))
        mr = np.mean([q['r'] for q in er]); mrf = np.mean([q['rf'] for q in er]); mrx = np.mean([q['rx'] for q in er])
        mb = np.mean([q['rate'] for q in er]); mch = np.mean([q['chat'] for q in er])
        entP = int(sum(q['nP'] > 0 for q in er))
        for c in exits:
            a = A[(kk, c, 'exit')]; rx_ = wret(c, e, e2)
            sh_before = r[2].get(c, np.nan)
            rows.append(dict(variant=v, j=jj, k=kk, signal=sg, e=e, code=COLS[c], tier=tier, price=px_at(e, c), price_d=D.C[s, c],
                             sell_notional=a['notional'], sell_rate=a['cost'] / a['notional'], deferred=a['nP'], dl_td=a['dl_td'], dl_cd=a['dl_cd'], kinds=a['kinds'],
                             r=rx_, r_fill=legret(a, c), r_allin=legret(a, c, 'px'), n_entries=len(er), n_exits=len(exits), entries_deferred=entP,
                             edge_B=mr - rx_, edge_fill=mrf - legret(a, c), edge_allin=mrx - legret(a, c, 'px'),
                             rtc_B=a['cost'] / a['notional'] + mb, chat=chat_rate(s, c, -1, None, sh_before) + mch,
                             disp=bool(D.DISP[e, c])))
    return rows

def chat_rate(s, c, side, tval, shares):
    """事前預估單邊成本率 ĉ（訊號日 d＝s 收盤可得）。side>0：買進目標金額 tval；side<0：賣出持有股數 shares。"""
    t1 = s + 1   # 事前陣列（V60p、SIGp、PC、VRp）在 s+1 列＝截至 s 收盤
    C = D.C[s, c]
    if not (np.isfinite(C) and C > 0): return np.nan
    if side > 0:
        if not np.isfinite(tval) or tval <= 0: return np.nan
        Lw = math.floor(tval / (1000.0 * C) + 1e-12); q = math.floor((tval - 1000.0 * Lw * C) / C + 1e-9)
    else:
        if not np.isfinite(shares) or shares <= 0: return np.nan
        n = int(math.floor(shares + 1e-9)); Lw, q = n // 1000, n % 1000
    fr = 0.001425 * 0.6 if t1 < XE.I_FEE else 0.001425 * 0.28
    tot = 0.0; vtot = 0.0
    if Lw > 0:
        v = 1000.0 * Lw * C
        oa = XE.OASH[t1, c] * D.V60p[t1, c]; sg = D.SIGp[t1, c]; r = XE.BASE
        if np.isfinite(oa) and oa > 0 and np.isfinite(sg): r = r + XE.KAPPA * sg * math.sqrt(v / oa)
        r = max(r, float(D.tick(np.array([C]))[0]) / C)
        tot += v * r + max(20.0, fr * v) + (0.003 * v if side < 0 else 0.0); vtot += v
    if q > 0:
        v = q * C
        oc = XE.OCB[t1, c] if side > 0 else XE.OCS[t1, c]
        tot += v * oc + max(20.0, fr * v) + (0.003 * v if side < 0 else 0.0); vtot += v
    return tot / vtot if vtot > 0 else np.nan

# ---------- bootstrap ----------
def month_ids(dates):
    m = pd.to_datetime(pd.Series(dates)).dt.to_period('M').astype(str).to_numpy()
    u = np.unique(m); return [np.flatnonzero(m == x) for x in u]
def boot(df, stat, B=2000, seed=20261112, col='signal'):
    df = df.reset_index(drop=True); ids = month_ids(df[col]); rng = np.random.default_rng(seed); n = len(ids)
    obs = stat(df); out = np.empty(B)
    for b in range(B):
        out[b] = stat(df.iloc[np.concatenate([ids[i] for i in rng.integers(0, n, n)])])
    out = out[np.isfinite(out)]
    return dict(obs=float(obs), se=float(np.std(out, ddof=1)), ci90=[float(np.percentile(out, 5)), float(np.percentile(out, 95))],
                p_le0=float(np.mean(out <= 0)), p_ge0=float(np.mean(out >= 0)), mde80=float(2.80 * np.std(out, ddof=1)), B=int(len(out)))
def neff_mean(x, dates, B=1000, seed=7):
    x = np.asarray(x, float); ok = np.isfinite(x); x = x[ok]; dts = np.asarray(dates)[ok]
    if len(x) < 3: return dict(n=int(len(x)))
    df = pd.DataFrame(dict(x=x, signal=dts)); b = boot(df, lambda d: d.x.mean(), B=B, seed=seed)
    vi = np.var(x, ddof=1) / len(x)
    return dict(n=int(len(x)), mean=float(x.mean()), se_boot=b['se'], se_iid=float(math.sqrt(vi)), deff=float(b['se'] ** 2 / vi), n_eff=float(len(x) * vi / b['se'] ** 2), mde80=b['mde80'], ci90=b['ci90'])
def quint(d, col, gcol='gap'):
    d = d[np.isfinite(d[col]) & np.isfinite(d[gcol])].copy()
    d['q'] = pd.qcut(d[col], 5, labels=False, duplicates='drop')
    tab = {int(k): dict(n=int(len(g)), x_mean=float(g[col].mean()), rtc_mean=float(g.rtc_B.mean()), edge_mean=float(g[gcol.replace('gap', 'edge')].mean()) if gcol.replace('gap', 'edge') in g else None,
                        gap_mean=float(g[gcol].mean()), gap_med=float(g[gcol].median()), frac_below=float((g[gcol] < 0).mean())) for k, g in d.groupby('q')}
    def st(z):
        a = z[gcol][z.q == 0]; b = z[gcol][z.q == 4]
        return a.mean() - b.mean() if len(a) and len(b) else np.nan
    return tab, d, st

# ---------- 橋段的月度輸入（點估計＝全部月份權重 1；bootstrap＝月份多項抽樣權重） ----------
COALS = [''.join(f for f, b in zip(FAC, bits) if b) or 'none' for bits in itertools.product([0, 1], repeat=6)]
def _shap_w():
    """Shapley(T) 權重：對不含 T 的組合 S，w(|S|)＝|S|!(5−|S|)!/6!，乘在 v(S∪T)−v(S)。"""
    n = 6; W = []
    for c in COALS:
        if c != 'none' and 'T' in c: continue
        s = 0 if c == 'none' else len(c)
        W.append((c, key6(('' if c == 'none' else c) + 'T'), math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)))
    return W
SHW = _shap_w()
def monthly(nav, s0, s1):
    x = nav.loc[s0:s1].dropna(); lr = np.log(x).diff().iloc[1:]
    m = lr.index.to_period('M'); g = lr.groupby(m).sum()
    last = x.groupby(x.index.to_period('M')).apply(lambda z: z.index[-1]); first = pd.Series([s0] + list(last.values[:-1]), index=last.index)
    days = pd.Series([(pd.Timestamp(b) - pd.Timestamp(a)).days for a, b in zip(first.values, last.values)], index=last.index)
    days = days.loc[g.index]
    return g, days

def bridge_inputs(per, v, NC, NR, NCOAL, RB, DC, FL, c0, refund_key):
    """回傳 13 個 j 的月度陣列（共用月份索引）：lrA、lrR、lrC[64]、days、cacc（會計，換股日月份）、cb1（C/E_pre，成交日月份）、ld（log1p(C/E_post)）、以及 years_o20。"""
    out = []
    for j in range(13):
        key = f'{v}|{j}|normal|100萬|std'
        navA = NC[key]; s0, s1, _ = perbounds(navA, j, per)
        gA, days = monthly(navA, s0, s1); M = gA.index
        gR, _ = monthly(NR[f'{refund_key}|{v}|{j}'], s0, s1)
        lrC = np.array([monthly(NCOAL[f'{c}|{v}|{j}'], s0, s1)[0].reindex(M).fillna(0).values for c in COALS])
        f = prep_fills(FL, RB[key], v, j)
        dc = pd.Series(DC[key], dtype=float)
        Epre = c0 * navA.reindex(IDX[f.t.values]).values + dc.reindex(f.t.values).fillna(0).values
        infd = (f.date > s0) & (f.date <= s1)
        cb1 = (f.cost / Epre)[infd].groupby(f.date[infd].dt.to_period('M')).sum().reindex(M).fillna(0).values
        a, b = L.PER[per]; rd = IDX[f.reb_t.values]; ink = (rd >= a) & (rd <= b)
        cacc = (f.cost / f.avail)[ink].groupby(rd[ink].to_period('M')).sum()
        cacc_tot = float(cacc.sum()); cacc = cacc.reindex(M).fillna(0).values   # 會計歸屬月份若落在 M 外（極少），點估計用總和
        dcs = dc.copy(); dcs.index = IDX[dcs.index.values]; dcs = dcs[(dcs.index > s0) & (dcs.index <= s1)]
        ld = pd.Series(np.log1p(dcs.values / (c0 * navA.reindex(dcs.index).values)), index=dcs.index).groupby(dcs.index.to_period('M')).sum().reindex(M).fillna(0).values
        out.append(dict(M=M, lrA=gA.values, lrR=gR.reindex(M).fillna(0).values, lrC=lrC, days=days.values.astype(float), cacc=cacc, cacc_tot=cacc_tot,
                        cb1=cb1, ld=ld, y_o20=years_o20(per, j), s0=s0, s1=s1))
    return out

def bridge_levels(BI, w_by_j=None):
    """由月度輸入算各 j 的橋段量（pp）；w_by_j＝每個 j 的月份權重（None＝1）。"""
    q = {k: [] for k in ('c_acc', 'c_B1', 'ell', 'G', 'G_lin', 'gA', 'gR', 'gT', 'q3', 'q4', 'CCj')}
    VC = []
    for j, b in enumerate(BI):
        w = np.ones(len(b['M'])) if w_by_j is None else w_by_j[j]
        Y = float(w @ b['days']) / 365.25; Y0 = float(b['days'].sum()) / 365.25
        ca = (b['cacc_tot'] if w_by_j is None else float(w @ b['cacc'])) / (b['y_o20'] * Y / Y0) * 100
        cb = float(w @ b['cb1']) / Y * 100; el = float(w @ b['ld']) / Y
        g = lambda lr: (math.exp(float(w @ lr) / Y) - 1) * 100
        gA, gR = g(b['lrA']), g(b['lrR'])
        vc = {c: (math.exp(float(w @ b['lrC'][i]) / Y) - 1) * 100 for i, c in enumerate(COALS)}
        gT = vc['T']
        shT = sum(wt * (vc[cT] - vc[c]) for c, cT, wt in SHW)   # Shapley 邊際（年化差，負＝下降）；成本類＝v(none)−v(all)＋mean marg_T
        CCj = vc['none'] - vc['TXFKIC'] + shT
        q['c_acc'].append(ca); q['c_B1'].append(cb); q['ell'].append(el * 100)
        q['G'].append((1 + gR / 100) * (1 - math.exp(-el)) * 100); q['G_lin'].append((1 + gR / 100) * cb)
        q['gA'].append(gA); q['gR'].append(gR); q['gT'].append(gT); q['q3'].append(gR - gA); q['q4'].append(gT - gA); q['CCj'].append(CCj)
        VC.append(vc)
    q = {k: np.array(x) for k, x in q.items()}
    vmed = {c: med([vc[c] for vc in VC]) for c in COALS}
    q['CCmed'] = vmed['none'] - vmed['TXFKIC'] + sum(wt * (vmed[cT] - vmed[c]) for c, cT, wt in SHW)
    q['Tfirst_med'] = vmed['T'] - vmed['TXFKIC']
    q['vmed'] = vmed
    return q

def bridge_boot(BI, B=1000, seed=20261112):
    """月份區塊 bootstrap：共用月份（以 Period 對齊），每次抽樣同一組月份套用到 13 j 與全部淨值。"""
    allM = sorted(set().union(*[set(b['M']) for b in BI])); pos = {m: i for i, m in enumerate(allM)}
    rng = np.random.default_rng(seed); reps = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, len(allM), len(allM)), minlength=len(allM)).astype(float)
        w = [cnt[[pos[m] for m in b['M']]] for b in BI]
        q = bridge_levels(BI, w); ch = chain(q)
        reps.append(dict(levels=ch['levels_median'], bridges=ch['bridges_main'], end=ch['end']))
    out = {}
    for k in BR + ['B3']:
        x = np.array([r['bridges'][k] for r in reps]); out[k] = dict(se=float(np.std(x, ddof=1)), ci90=[float(np.percentile(x, 5)), float(np.percentile(x, 95))], mde80=float(2.8 * np.std(x, ddof=1)))
    for k in ('c_acc', 'CCj'):
        x = np.array([r['levels'][k] for r in reps]); out['level_' + k] = dict(se=float(np.std(x, ddof=1)))
    x = np.array([r['end'] for r in reps]); out['level_CCmed'] = dict(se=float(np.std(x, ddof=1)))
    x = np.array([r['end'] - r['levels']['c_acc'] for r in reps]); out['total_gap'] = dict(se=float(np.std(x, ddof=1)), ci90=[float(np.percentile(x, 5)), float(np.percentile(x, 95))])
    out['n_months'] = len(allM); out['B'] = B
    return out

# s5_analysis.py：由 s4_oos.py analysis 階段在雜湊斷言後 exec（共用 s4 的命名空間：X, Rn, E, PL, KEY, GRIDS, BODY, SCR, HERE, RES）。
import importlib.util
PAP = pickle.loads((SCR / 'paper.pkl').read_bytes())
EXE = pickle.loads((SCR / 'exec.pkl').read_bytes())
PLS = pickle.loads((SCR / 'plans_key.pkl').read_bytes())
R = json.loads(RES.read_text(encoding='utf-8'))
PERS = {'SEL': ('2014-01-01', '2016-12-31'), 'VAL': ('2017-01-01', '2026-12-31'), 'Y19': ('2019-01-01', '2026-12-31'), 'PIT': ('2014-01-01', '2026-12-31'),
        'FULL': ('2006-04-01', '2026-12-31'), 'SEG1': ('2006-04-01', '2012-12-31'), 'SEG2': ('2013-01-01', '2018-12-31'), 'SEG3': ('2019-01-01', '2026-12-31'),
        'DESC': ('2006-04-01', '2013-12-31'), 'Y24': ('2024-01-01', '2026-12-31')}
DAYS = {j: X.days_for(j, 'FULL') for j in Rn.JS}
WD = {j: pd.DatetimeIndex(DAYS[j][:-1]) for j in Rn.JS}
BEN = {j: pd.Series(X.bench_win(DAYS[j]), index=WD[j]) for j in Rn.JS}
def S(nm, j, cost='normal'): return pd.Series(PAP[(nm, j, cost)]['r'], index=WD[j])
def EX(nm, j, tn='100萬', mode='GROW', cost='normal'): return pd.Series(EXE[(nm, j, tn, mode, cost)]['r'], index=WD[j])
def sl(s, p): a, b = PERS[p]; return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
def jsum(f, p, b=None):
    """f(j)→策略窗報酬；b(j)→基準（預設 0050）。回傳 j 中位數統計＋年化。"""
    st = []; cg = []; cb = []; md = []
    for j in Rn.JS:
        r = sl(f(j), p); bb = sl(BEN[j] if b is None else b(j), p).reindex(r.index)
        x = (r - bb).to_numpy(); st.append(X.stats(x)); cg.append(X.cagr(r.to_numpy())); cb.append(X.cagr(bb.to_numpy())); md.append(X.mdd(r.to_numpy()))
    a = lambda k: float(np.median([s[k] for s in st]))
    return dict(mean_pp=a('mean_pp'), mean_pp_p10=float(np.percentile([s['mean_pp'] for s in st], 10)), t=a('t'), t_p10=float(np.percentile([s['t'] for s in st], 10)),
                TE=a('TE'), IR=a('IR'), n=int(a('n')), CAGR=float(np.median(cg)), CAGR_bench=float(np.median(cb)), MDD_win=float(np.median(md)))
def geo_ex(r, b):
    return X.cagr(r) - X.cagr(b)
def seg_geo(f, p, b=None):
    v = []
    for j in Rn.JS:
        r = sl(f(j), p); bb = sl(BEN[j] if b is None else b(j), p).reindex(r.index); v.append(geo_ex(r.to_numpy(), bb.to_numpy()))
    return float(np.median(v) * 100)
def drop3(f, p, b=None):
    ts = []
    for j in Rn.JS:
        r = sl(f(j), p); bb = sl(BEN[j] if b is None else b(j), p).reindex(r.index)
        t, yrs = X.drop_best_years_t((r - bb).to_numpy(), list(r.index), 3, bb.to_numpy()); ts.append(t)
    return float(np.median(ts)), yrs
def yearly(f, b=None):
    out = {}
    for y in range(2006, 2027):
        v = []; vb = []; vr = []
        for j in Rn.JS:
            r = f(j); r = r[r.index.year == y]; bb = (BEN[j] if b is None else b(j)).reindex(r.index)
            if len(r) < 3: continue
            vr.append(np.prod(1 + r) - 1); vb.append(np.prod(1 + bb) - 1)
        if vr: out[str(y)] = dict(ret=float(np.median(vr) * 100), bench=float(np.median(vb) * 100), ex=float(np.median(np.array(vr) - np.array(vb)) * 100))
    return out

out = {'s5_sha256': X.sha(HERE / 's5_analysis.py'), 'note': '主欄＝13 個 j 中位數；2014 前市值非 PIT（只作描述）'}
PL_ = ['SEL', 'VAL', 'Y19', 'Y24', 'PIT', 'FULL', 'SEG1', 'SEG2', 'SEG3']
# ---------- 1. 紙上：關鍵規格 ----------
paper = {}
for nm in KEY:
    d = {}
    for cost in ('normal', 'stress', 'zero'):
        d[cost] = {p: jsum(lambda j: S(nm, j, cost), p) for p in PL_}
        d[cost + '_vsB'] = {p: jsum(lambda j: S(nm, j, cost), p, lambda j: S('B', j, cost)) for p in PL_}
    d['seg_geo_ex_pp'] = {p: seg_geo(lambda j: S(nm, j), p) for p in ('SEG1', 'SEG2', 'SEG3')}
    d['drop3_t_PIT'], d['drop3_years_PIT'] = drop3(lambda j: S(nm, j), 'PIT')
    d['drop3_t_FULL'], d['drop3_years_FULL'] = drop3(lambda j: S(nm, j), 'FULL')
    d['yearly'] = yearly(lambda j: S(nm, j))
    d['turnover_1side'] = float(np.median([PAP[(nm, j, 'normal')]['to'].mean() for j in Rn.JS]))
    paper[nm] = d
out['paper_key'] = paper
print('paper key done', flush=True)
# 鄰域熱圖
heat = {}
for gsp in GRIDS:
    nm = Rn.name(gsp)
    heat[nm] = {p: round(jsum(lambda j: S(nm, j), p)['IR'], 3) for p in ('SEL', 'VAL', 'Y19', 'PIT')}
    heat[nm].update({'t_' + p: round(jsum(lambda j: S(nm, j), p)['t'], 2) for p in ('VAL', 'PIT')})
    heat[nm]['vsB_t_PIT'] = round(jsum(lambda j: S(nm, j), 'PIT', lambda j: S('B', j))['t'], 2)
out['grid_heat'] = heat
print('heat done', flush=True)

# ---------- 2. 可執行（K5T）----------
ex = {}
for nm in ['SEL', 'DEF', 'SEL_PURE', 'CTRL_SEL', 'CTRL_SEL_PURE', 'CTRL_DEF', 'IND_ONLY_PURE', 'IND_ONLY_SAT', 'B']:
    d = {}
    for tn in E.TIERS:
        for mode in ('GROW', 'FIX'):
            for cost in ('normal', 'stress'):
                k = f'{tn}|{mode}|{cost}'
                d[k] = {p: jsum(lambda j: EX(nm, j, tn, mode, cost), p) for p in ('VAL', 'Y19', 'PIT', 'FULL', 'SEG1', 'SEG2', 'SEG3')}
                d[k]['cost_pp_month'] = float(np.median([EXE[(nm, j, tn, mode, cost)]['cost_pp'] for j in Rn.JS]))
                d[k]['odd_share'] = float(np.median([EXE[(nm, j, tn, mode, cost)]['odd_share'] for j in Rn.JS]))
                d[k]['pre2020_odd_share'] = float(np.median([EXE[(nm, j, tn, mode, cost)]['pre_odd'] for j in Rn.JS]))
    d['seg_geo_ex_pp_100萬GROW'] = {p: seg_geo(lambda j: EX(nm, j), p) for p in ('SEG1', 'SEG2', 'SEG3')}
    d['drop3_t_PIT_100萬GROW'], d['drop3_years'] = drop3(lambda j: EX(nm, j), 'PIT')
    d['drop3_t_FULL_100萬GROW'], _ = drop3(lambda j: EX(nm, j), 'FULL')
    d['yearly_100萬GROW'] = yearly(lambda j: EX(nm, j))
    ex[nm] = d
out['exec'] = ex
print('exec done', flush=True)

# ---------- 3. 主策略（V7X 可執行 B10）----------
NAVM = pd.read_parquet(X.AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet')
def main_ser(j, cell='100萬', cost='normal'):
    nav = NAVM[f'B10|J{j}|K5T|{cost}|{cell}|GROW'].dropna()
    v = nav.reindex(nav.index.union(pd.DatetimeIndex(DAYS[j]))).ffill().reindex(pd.DatetimeIndex(DAYS[j])).to_numpy()
    r = v[1:] / v[:-1] - 1
    s = pd.Series(r, index=WD[j]); s[s.index < nav.index[0]] = np.nan
    return s
MAIN = {j: main_ser(j) for j in Rn.JS}
out['main_exec_100萬GROW'] = {p: jsum(lambda j: MAIN[j].dropna(), p) for p in ('VAL', 'Y19', 'PIT', 'FULL')}
out['main_yearly'] = yearly(lambda j: MAIN[j].dropna())
cmp = {}
for nm in ('SEL', 'SEL_PURE', 'CTRL_SEL', 'IND_ONLY_SAT'):
    dd = {}
    for p in ('Y19', 'PIT'):
        pt, cr, cr2, irm, irc, pb = [], [], [], [], [], []
        for j in Rn.JS:
            a = sl(EX(nm, j), p); m = sl(MAIN[j], p).reindex(a.index); b = sl(BEN[j], p).reindex(a.index); ok = m.notna()
            xa = (a - b)[ok].to_numpy(); xm = (m - b)[ok].to_numpy()
            pt.append(X.tstat(xa - xm)); cr.append(np.corrcoef(xa, xm)[0, 1])
            xc = 0.5 * xa + 0.5 * xm; irm.append(X.stats(xm)['IR']); irc.append(X.stats(xc)['IR'])
            rng = np.random.default_rng(j); n = len(xa); cnt = 0
            for bsi in range(2000):
                idx = [];
                while len(idx) < n:
                    s0 = rng.integers(0, n); idx += [(s0 + q) % n for q in range(6)]
                idx = np.array(idx[:n]); d_ir = X.stats(xc[idx])['IR'] - X.stats(xm[idx])['IR']
                cnt += d_ir <= 0
            pb.append(cnt / 2000)
        dd[p] = dict(paired_t_vs_main=float(np.median(pt)), corr_excess=float(np.median(cr)), IR_main=float(np.median(irm)), IR_combo5050=float(np.median(irc)),
                     boot_p_combo_le_main=float(np.median(pb)))
    cmp[nm] = dd
out['vs_main'] = cmp
print('main cmp done', flush=True)

# ---------- 4. Q3 拆解（紙上一般成本）----------
def pdiff(a, b, p):
    st = [X.stats((sl(S(a, j), p) - sl(S(b, j), p)).to_numpy()) for j in Rn.JS]
    return dict(mean_pp=float(np.median([s['mean_pp'] for s in st])), t=float(np.median([s['t'] for s in st])))
q3 = {}
for p in ('Y19', 'PIT', 'VAL', 'FULL'):
    q3[p] = dict(sector_effect_IND_ONLY_PURE_minus_B=pdiff('IND_ONLY_PURE', 'B', p), within_sector_SEL_PURE_minus_IND_ONLY_PURE=pdiff('SEL_PURE', 'IND_ONLY_PURE', p),
                 standalone_stock_CTRL_PURE_minus_B=pdiff('CTRL_SEL_PURE', 'B', p), IND_minus_CTRL_PURE=pdiff('SEL_PURE', 'CTRL_SEL_PURE', p),
                 IND_minus_CTRL_SAT=pdiff('SEL_SATv', 'CTRL_SEL_SAT', p), IND_ONLY_SAT_minus_B=pdiff('IND_ONLY_SAT', 'B', p))
out['Q3'] = q3

# ---------- 5. Q1 差中差、IC、Q2 ----------
def q1(j, p):
    days = DAYS[j]; SR = X.stock_win_ret(days); rows = []
    a, b = PERS[p]
    for k, d in enumerate(days[:-1]):
        if not (pd.Timestamp(a) <= d <= pd.Timestamp(b)): continue
        cs = X.cs_cached(d); ST = cs['ST']; r = SR[k][cs['U']]
        if len(ST) < 10: continue
        def spread(secs):
            diffs = []; ics = []
            for s in secs:
                m = [i for i in cs['secs'][s]['members'] if np.isfinite(cs['score'][i]) and np.isfinite(r[i])]
                if len(m) < 2: continue
                o = X.stock_order(cs, m); h = max(len(o) // 2, 1)
                diffs.append(np.mean(r[o[:h]]) - np.mean(r[o[-h:]]))
                if len(m) >= 3: ics.append(pd.Series(cs['score'][m]).corr(pd.Series(r[m]), method='spearman'))
            return (np.mean(diffs) if diffs else np.nan), (np.nanmean(ics) if ics else np.nan)
        KK = X.mode_params()['k']; dt, it_ = spread(list(ST.index[:KK])); db, ib = spread(list(ST.index[-KK:]))
        rows.append(dict(did=dt - db, top=dt, bot=db, ic_top=it_, ic_bot=ib))
    return pd.DataFrame(rows)
Q1 = {}
for p in ('SEL', 'VAL', 'Y19', 'PIT'):
    st = [q1(j, p) for j in Rn.JS]
    Q1[p] = dict(did_pp=float(np.median([q.did.mean() * 100 for q in st])), t_did=float(np.median([X.tstat(q.did) for q in st])),
                 top_pp=float(np.median([q.top.mean() * 100 for q in st])), bot_pp=float(np.median([q.bot.mean() * 100 for q in st])),
                 ic_top=float(np.median([q.ic_top.mean() for q in st])), ic_bot=float(np.median([q.ic_bot.mean() for q in st])))
out['Q1'] = Q1
def q2(j, p, n):
    days = DAYS[j]; SR = X.stock_win_ret(days); v = []
    a, b = PERS[p]; s_ = Rn.make(dict(KEY['SEL_PURE'], n=n))
    for k, d in enumerate(days[:-1]):
        cs = X.cs_cached(d); sel, w, info = s_.sleeve(cs)
        if not (pd.Timestamp(a) <= d <= pd.Timestamp(b)): continue
        r = SR[k][cs['U']]; dif = []
        for sname in info['sectors']:
            mem = cs['secs'][sname]['members']; ok = np.isfinite(r[mem])
            if ok.sum() == 0: continue
            rs = (cs['mc'][mem][ok] * r[mem][ok]).sum() / cs['mc'][mem][ok].sum()
            ss = [i for i in sel if i in set(mem) and np.isfinite(r[i])]
            if ss: dif.append(np.mean(r[ss]) - rs)
        if dif: v.append(np.mean(dif))
    return np.array(v)
out['Q2'] = {f'n{n}': {p: dict(mean_pp=float(np.median([q2(j, p, n).mean() * 100 for j in Rn.JS])), t=float(np.median([X.tstat(q2(j, p, n)) for j in Rn.JS])))
                         for p in ('SEL', 'VAL', 'PIT')} for n in (1, 2, 3)}
print('Q1 Q2 done', flush=True)

# ---------- 6. Brinson（紙上毛報酬、收盤到收盤）----------
def brinson(nm, p):
    res = {k: [] for k in ('alloc', 'select', 'inter')}
    for j in Rn.JS:
        days = DAYS[j]; SR = X.stock_win_ret(days); pl, info = PLS[(nm, j)]; al, se, it_ = [], [], []
        a, b = PERS[p]
        for k, d in enumerate(days[:-1]):
            if not (pd.Timestamp(a) <= d <= pd.Timestamp(b)): continue
            cs = X.cs_cached(d); U = cs['U']; wB = cs['mc'] / cs['mc'].sum(); lab = X.ind_at(d) if X.MODE['m'] == 'COMP' else X.t1_module().ind_at(d)
            idx, w = pl[k]; r = np.nan_to_num(SR[k], nan=0.0)
            wp = pd.Series(w, index=idx).groupby(level=0).sum()
            secP = pd.Series(w, index=[lab[i] for i in idx]); secB = pd.Series(wB, index=[lab[i] for i in U])
            rP = pd.Series(w * r[idx], index=secP.index); rB = pd.Series(wB * r[U], index=secB.index)
            WP = secP.groupby(level=0).sum(); WB = secB.groupby(level=0).sum()
            RP = rP.groupby(level=0).sum() / WP; RB = rB.groupby(level=0).sum() / WB
            allsec = WP.index.union(WB.index); WP = WP.reindex(allsec).fillna(0); WB = WB.reindex(allsec).fillna(0)
            RB = RB.reindex(allsec); RP = RP.reindex(allsec); RB = RB.fillna(RP); RP = RP.fillna(RB)
            rb_tot = (WB * RB).sum()
            al.append(((WP - WB) * (RB - rb_tot)).sum()); se.append((WB * (RP - RB)).sum()); it_.append(((WP - WB) * (RP - RB)).sum())
        res['alloc'].append(np.array(al)); res['select'].append(np.array(se)); res['inter'].append(np.array(it_))
    return {k: dict(mean_pp=float(np.median([x.mean() * 100 for x in v])), t=float(np.median([X.tstat(x) for x in v]))) for k, v in res.items()}
out['Brinson'] = {nm: {p: brinson(nm, p) for p in ('Y19', 'PIT', 'VAL')} for nm in ('SEL', 'SEL_PURE', 'CTRL_SEL_PURE', 'IND_ONLY_PURE')}
print('Brinson done', flush=True)

# ---------- 7. 右尾 ----------
def contrib(nm, j):
    days = DAYS[j]; SR = X.stock_win_ret(days); pl, _ = PLS[(nm, j)]; b = BEN[j].to_numpy(); rows = []
    for k, d in enumerate(days[:-1]):
        idx, w = pl[k]; r = SR[k][idx]
        c = w * (np.nan_to_num(r, nan=b[k]) - b[k])
        lab = X.ind_at(d) if X.MODE['m'] == 'COMP' else X.t1_module().ind_at(d)
        for i, ww, cc in zip(idx, w, c): rows.append((d, X.COLS[i], lab[i], ww, cc))
    return pd.DataFrame(rows, columns=['d', 'code', 'ind', 'w', 'c'])
rt = {}
for nm in ('SEL', 'SEL_PURE'):
    agg = []; tt = {k: [] for k in ('base', 'ex_2021ship_2025AI', 'ex_top10')}; tt19 = {k: [] for k in tt}
    for j in Rn.JS:
        C = contrib(nm, j); agg.append(C.groupby('code').c.sum())
        x = (S(nm, j) - BEN[j])
        rm1 = C[((C.d.dt.year == 2021) & (C.ind == '航運業')) | ((C.d.dt.year == 2025) & C.ind.isin(['電子零組件業', '電腦及週邊設備業']))].groupby('d').c.sum()
        top10 = C[C.d >= pd.Timestamp('2014-01-01')].groupby('code').c.sum().sort_values(ascending=False).index[:10]
        rm2 = C[C.code.isin(top10)].groupby('d').c.sum()
        for key, xx in (('base', x), ('ex_2021ship_2025AI', x - rm1.reindex(x.index).fillna(0)), ('ex_top10', x - rm2.reindex(x.index).fillna(0))):
            tt[key].append(X.tstat(sl(xx, 'PIT').to_numpy())); tt19[key].append(X.tstat(sl(xx, 'Y19').to_numpy()))
    A = pd.concat(agg, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    rt[nm] = dict(t_PIT={k: float(np.median(v)) for k, v in tt.items()}, t_Y19={k: float(np.median(v)) for k, v in tt19.items()},
                  top10_contrib_FULL_pp={k: round(float(v) * 100, 3) for k, v in A.head(10).items()}, bottom5_contrib_pp={k: round(float(v) * 100, 3) for k, v in A.tail(5).items()})
out['right_tail'] = rt
print('right tail done', flush=True)

# ---------- 8. Q4 子樣本 ----------
def q4(nm):
    rows = []
    for j in Rn.JS:
        x = S(nm, j) - BEN[j]
        for d, v in x.items():
            cs = X.cs_cached(d); disp = float(cs['ST']['REV'].std()) if len(cs['ST']) else np.nan
            rows.append(dict(j=j, d=d, x=v, disp=disp))
    D = pd.DataFrame(rows); P = D[D.d >= pd.Timestamp('2014-01-01')].copy()
    med = P.disp.median(); hi = P[P.disp > med]; lo = P[P.disp <= med]
    def mm(Z): return float(Z.groupby('j').x.mean().median() * 100)
    e50 = pd.read_csv(X.O8 / 'raw' / 'bench' / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    dd = e50 / e50.cummax() - 1
    troughs = []; inb = False; lowv = 0; lowd = None
    for dt_, v in dd.items():
        if v <= -0.20: inb = True
        if inb:
            if v < lowv: lowv = v; lowd = dt_
            if v > -0.05: troughs.append(lowd); inb = False; lowv = 0
    def rebound_mask(d):
        return any(td is not None and td < d <= td + pd.DateOffset(months=3) for td in troughs)
    D['rebound'] = [rebound_mask(d) for d in D.d]
    D['theme_end'] = [(d.year == 2022) or (pd.Timestamp('2024-07-01') <= d <= pd.Timestamp('2024-08-31')) for d in D.d]
    D['theme'] = [(2020 <= d.year <= 2021) or (2023 <= d.year <= 2025) for d in D.d]
    D['theme_0910'] = [2009 <= d.year <= 2010 for d in D.d]
    P = D[D.d >= pd.Timestamp('2014-01-01')]
    res = dict(disp_hi_pp=mm(hi), disp_lo_pp=mm(lo), j0_4_pp=float(P[P.j <= 4].x.mean() * 100), j8_12_pp=float(P[P.j >= 8].x.mean() * 100),
               theme_2020_21_23_25_pp=mm(P[P.theme]), non_theme_PIT_pp=mm(P[~P.theme]), theme_2009_10_desc_pp=float(D[D.theme_0910].x.mean() * 100) if D.theme_0910.any() else None,
               rebound_after_bear_pp=float(P[P.rebound].x.mean() * 100) if P.rebound.any() else None, n_rebound_win=int(P.rebound.sum() / 13),
               theme_end_pp=float(P[P.theme_end].x.mean() * 100), troughs=[str(t.date()) for t in troughs if t is not None])
    # 主題結束期回撤：策略 vs 0050（j 中位數，窗頻）
    def dd_in(a, b):
        vs, vb = [], []
        for j in Rn.JS:
            r = S(nm, j); r = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]; bb = BEN[j].reindex(r.index)
            vs.append(X.mdd(r.to_numpy())); vb.append(X.mdd(bb.to_numpy()))
        return dict(strat=float(np.median(vs)), e50=float(np.median(vb)))
    res['mdd_2022'] = dd_in('2022-01-01', '2022-12-31'); res['mdd_2024_07_08'] = dd_in('2024-06-01', '2024-09-30')
    return res
out['Q4'] = {nm: q4(nm) for nm in ('SEL', 'SEL_PURE')}
print('Q4 done', flush=True)

# ---------- 9. Q5 主策略右尾歸因、R² ----------
pj = pd.read_csv(X.AG / 'o_v7_listing_exec_engine' / 'data' / 'plans_j7.csv', dtype={'picks': str})
pj = pj[(pj.strat == 'AB_V7_B10') & (pj.seed == 0)]
agg = []
for j in Rn.JS:
    g_ = pj[pj.j == j].sort_values('signal'); sd = list(pd.to_datetime(g_.signal)) + [X.IDX[-1]]
    ti = [X.IDX.get_loc(d) for d in sd]; e50 = X.L.E50.to_numpy()
    for k in range(len(ti) - 1):
        codes = g_.picks.iloc[k].split(','); r = X.cumret(ti[k], ti[k + 1]); rb = e50[ti[k + 1]] / e50[ti[k]] - 1
        d = sd[k]; mymon = {x.to_period('M'): x for x in DAYS[j][:-1]}
        dm = mymon.get(d.to_period('M'))
        if dm is None: continue
        cs = X.cs_cached(dm); top = set(cs['ST'].index[:X.mode_params()['k']]); lab = X.ind_at(dm) if X.MODE['m'] == 'COMP' else X.t1_module().ind_at(dm)
        for c in codes:
            i = X.CI[c]; agg.append((j, d.year, c, (np.nan_to_num(r[i], nan=rb) - rb) / len(codes), lab[i] in top, lab[i], len(cs['ST'])))
A = pd.DataFrame(agg, columns=['j', 'year', 'code', 'c', 'intop', 'ind', 'nsec'])
SY = A.groupby(['code', 'year']).agg(c=('c', 'sum'), intop=('intop', 'mean'), ind=('ind', 'first')).reset_index()
SY['c'] = SY['c'] / 13
q5 = {}
for lab_, sub in (('FULL', SY), ('PIT', SY[SY.year >= 2014])):
    top20 = sub.sort_values('c', ascending=False).head(20)
    q5[lab_] = dict(share_intop_majority=float((top20.intop >= 0.5).mean()), mean_intop_fraction=float(top20.intop.mean()),
                    random_expectation=float((X.mode_params()['k'] / A[A.year >= (2014 if lab_ == 'PIT' else 2006)].nsec).mean()),
                    rows=[dict(code=r.code, year=int(r.year), contrib_pp=round(r.c * 100, 2), intop=round(r.intop, 2), ind=r.ind) for r in top20.itertuples()])
r2 = []
for j in Rn.JS:
    y = (MAIN[j] - BEN[j]); f = S('IND_ONLY_PURE', j) - S('B', j)
    ok = y.notna() & (y.index >= pd.Timestamp('2014-01-01')); yy, ff = y[ok].to_numpy(), f[ok].to_numpy()
    r2.append(np.corrcoef(yy, ff)[0, 1] ** 2)
q5['R2_main_excess_on_sector_factor_PIT'] = float(np.median(r2))
# 與主策略名單重疊（選定版強勢股腿 vs 主策略 10 檔，同 j、同訊號月）
ovl = []; ovl_ctrl = []
for j in Rn.JS:
    g_ = pj[pj.j == j]; mp = {pd.Timestamp(x).to_period('M'): set(y.split(',')) for x, y in zip(g_.signal, g_.picks)}
    for nm_, store in (('SEL', ovl), ('CTRL_SEL', ovl_ctrl)):
        pl, info = PLS[(nm_, j)]
        for k, d in enumerate(DAYS[j][:-1]):
            if d < pd.Timestamp('2014-01-01'): continue
            idx, w = pl[k]
            cs = X.cs_cached(d); wB = dict(zip(cs['U'], cs['mc'] / cs['mc'].sum()))
            sleeve = [X.COLS[i] for i, ww in zip(idx, w) if ww > 0.7 * wB.get(i, 0) + 1e-9] if KEY[nm_]['carrier'] == 'SAT' else [X.COLS[i] for i in idx]
            m = mp.get(d.to_period('M'))
            if m and sleeve: store.append(len(set(sleeve) & m) / len(sleeve))
q5['overlap_with_main_picks'] = dict(SEL=float(np.mean(ovl)) if ovl else None, CTRL_SEL=float(np.mean(ovl_ctrl)) if ovl_ctrl else None)
out['Q5'] = q5
# ---------- 10. Q6 ----------
q6 = {}
r2330 = {j: pd.Series([X.cumret(a, b)[X.CI['2330']] for a, b in zip([X.IDX.get_loc(d) for d in DAYS[j][:-1]], [X.IDX.get_loc(d) for d in DAYS[j][1:]])], index=WD[j]) for j in Rn.JS}
for nm in ('SEL', 'SEL_PURE', 'CTRL_SEL'):
    cr, be = {}, {}
    for p in ('PIT', 'Y19'):
        c_, b_ = [], []
        for j in Rn.JS:
            x = sl(S(nm, j) - BEN[j], p); m = sl(MAIN[j] - BEN[j], p).reindex(x.index); f = sl(r2330[j] - BEN[j], p).reindex(x.index)
            ok = m.notna(); c_.append(np.corrcoef(x[ok], m[ok])[0, 1])
            b_.append(float(np.polyfit(f.to_numpy(), x.to_numpy(), 1)[0]))
        cr[p] = float(np.median(c_)); be[p] = float(np.median(b_))
    q6[nm] = dict(corr_with_main_excess=cr, beta_on_2330_minus_0050=be)
out['Q6'] = q6
print('Q5 Q6 done', flush=True)

# ---------- 11. Q7 安慰劑 ----------
q7 = {}
for kind in ('a', 'b', 'c'):
    f = SCR / f'placebo_{kind}.npz'
    if not f.exists(): q7[kind] = None; continue
    Z = np.load(f); res = {}
    for p in ('VAL', 'PIT', 'Y19'):
        pn, pg = [], []
        for j in Rn.JS:
            days = [pd.Timestamp(str(x)) for x in Z[f'days{j}']]; wd = pd.DatetimeIndex(days[:-1])
            m = (wd >= pd.Timestamp(PERS[p][0])) & (wd <= pd.Timestamp(PERS[p][1]))
            b = X.bench_win(days)
            rn = Rn.run(KEY['SEL'], j, 'PIT', 'normal')['r']; rz = Rn.run(KEY['SEL'], j, 'PIT', 'zero')['r']
            sn = (Z[f'net{j}'][:, m] - b[m]).mean(1); sg = (Z[f'gross{j}'][:, m] - b[m]).mean(1)
            pn.append(float((sn >= (rn[m] - b[m]).mean()).mean())); pg.append(float((sg >= (rz[m] - b[m]).mean()).mean()))
        res[p] = dict(p_net_jmed=float(np.median(pn)), p_gross_jmed=float(np.median(pg)), p_net_per_j=pn, p_gross_per_j=pg)
    q7[kind] = res
out['Q7'] = q7
print('Q7 done', flush=True)

# ---------- 12. G3 SPA／DSR ----------
spec = importlib.util.spec_from_file_location('g2lib_ro', X.AG / 'o_g2_feasibility_stoprule' / 'g2lib.py'); G2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(G2)
Xf, meta = G2.family()
WIN = G2.WIN; R50 = G2.R50
mine = {}
for gsp in GRIDS + [KEY['IND_ONLY_PURE'], KEY['IND_ONLY_SAT']]:
    nm = Rn.name(gsp); o = PAP[(nm, 'off', 'normal')]
    s = pd.Series(o['r'] - o['b'], index=pd.DatetimeIndex(o['days'][:-1]))
    mine['LCSS:' + nm] = s.reindex(WIN)
o = EXE[('SEL', 'off', '100萬', 'GROW', 'normal')]; offd = X.sig_days('off', '2006-04-01', '2026-10-31')
sx = pd.Series(o['r'] - X.bench_win(offd), index=pd.DatetimeIndex(offd[:-1])).reindex(WIN)
mine['LCSS:SELECTED_EXEC_K5T_100萬GROW'] = sx
M = pd.DataFrame(mine)
align = dict(n_win=len(WIN), matched=int(M.notna().any(axis=1).sum()), first=str(WIN[0].date()), last=str(WIN[-1].date()))
def spa(Dall, focus):
    Df = Dall.fillna(0.0).to_numpy(); n = Df.shape[0]
    W = G2.SP.sb_counts(n, 6, 10000, 20261103); Mb = G2.SP.boot_means(W, Df)
    om2 = G2.SP.lrv(Df, 'pr', 6); om = np.sqrt(om2); sn = np.sqrt(n)
    dbar = Df.mean(0); tst = sn * dbar / om
    thr = -np.sqrt(om2 / n * 2 * np.log(np.log(n))); gc = np.where(dbar >= thr, dbar, 0.0)
    Zb = (Mb - gc[None, :].astype(np.float32)) * (sn / om).astype(np.float32)[None, :]
    Tb = np.maximum(Zb.max(axis=1), 0.0)
    k = list(Dall.columns).index(focus); tf = tst[k]
    sr_all = Df.mean(0) / Df.std(0, ddof=1); sr_all = sr_all[np.isfinite(sr_all)]
    dsr = G2.SP.dsr(Df[:, k], sr_all, Df.shape[1])
    return dict(N=int(Df.shape[1]), t_focus=float(tf), p_SPA_c=float((Tb >= tf).mean()) if tf > 0 else 1.0, Tb_q95=float(np.quantile(Tb, 0.95)), DSR=float(dsr['DSR']), SR0=float(dsr['SR0']))
fam_all = pd.concat([Xf, M], axis=1)
fmain_ids = meta.loc[meta.F_main == True, 'member_id'].tolist()
fam_fm = pd.concat([Xf[fmain_ids], M], axis=1)
out['G3'] = dict(alignment=align, family_all=spa(fam_all, 'LCSS:SELECTED_EXEC_K5T_100萬GROW'), family_Fmain=spa(fam_fm, 'LCSS:SELECTED_EXEC_K5T_100萬GROW'),
                 note='官方日窗（AB_V4 訊號日格點）；本題 2014 前為非 PIT 股數描述段，仍放入以對齊族群全期')
print('G3 done', out['G3'], flush=True)

# ---------- 13. G1～G8 ----------
e = out['exec']['SEL']; pk = out['paper_key']['SEL']
gN = e['100萬|GROW|normal']; gS = e['100萬|GROW|stress']
G = {}
G['G1'] = dict(normal_PIT_jmed_pp=gN['PIT']['mean_pp'], normal_PIT_p10_pp=gN['PIT']['mean_pp_p10'], stress_PIT_jmed_pp=gS['PIT']['mean_pp'], stress_PIT_p10_pp=gS['PIT']['mean_pp_p10'],
               normal_FULL_jmed_pp=gN['FULL']['mean_pp'], normal_FULL_p10_pp=gN['FULL']['mean_pp_p10'], stress_FULL_jmed_pp=gS['FULL']['mean_pp'], stress_FULL_p10_pp=gS['FULL']['mean_pp_p10'])
G['G1']['pass'] = bool(min(G['G1']['normal_PIT_jmed_pp'], G['G1']['normal_PIT_p10_pp'], G['G1']['stress_PIT_jmed_pp'], G['G1']['stress_PIT_p10_pp'],
                           G['G1']['normal_FULL_jmed_pp'], G['G1']['normal_FULL_p10_pp'], G['G1']['stress_FULL_jmed_pp'], G['G1']['stress_FULL_p10_pp']) > 0)
G['G2'] = dict(t_FULL=gN['FULL']['t'], t_PIT=gN['PIT']['t'], t_Y19=gN['Y19']['t'], pass_=bool(gN['FULL']['t'] >= 3 and gN['Y19']['t'] >= 2))
G['G3'] = dict(p_all=out['G3']['family_all']['p_SPA_c'], DSR_all=out['G3']['family_all']['DSR'], p_Fmain=out['G3']['family_Fmain']['p_SPA_c'], DSR_Fmain=out['G3']['family_Fmain']['DSR'])
G['G3']['pass_'] = bool(G['G3']['p_all'] < 0.05 and G['G3']['DSR_all'] >= 0.95)
sg = e['seg_geo_ex_pp_100萬GROW']
nbr = {k: v for k, v in heat.items() if k.startswith('IND')}
selnm = Rn.name(KEY['SEL']).replace('_b1', '')
G['G4'] = dict(seg_geo_ex_pp=sg, drop3_t_PIT=e['drop3_t_PIT_100萬GROW'], drop3_t_FULL=e['drop3_t_FULL_100萬GROW'],
               neighborhood_IR_PIT_signs={k: v['PIT'] for k, v in nbr.items()}, pass_=bool(all(v > 0 for v in sg.values()) and e['drop3_t_FULL_100萬GROW'] >= 2 and all(v['PIT'] > 0 for v in nbr.values())))
G['G5'] = dict(pass_=False, why='PIT 市值只有 2014 年起；2006～2013 用非 PIT 股數；個別公司 2007～2023 改類無法 PIT；下市電子股 11 檔手動補')
G['G6'] = dict(pass_=False, why='待獨立複製員')
G['G7'] = {tn: dict(PIT_t=e[f'{tn}|GROW|normal']['PIT']['t'], FULL_t=e[f'{tn}|GROW|normal']['FULL']['t'], Y19_t=e[f'{tn}|GROW|normal']['Y19']['t'],
                    PIT_mean_pp=e[f'{tn}|GROW|normal']['PIT']['mean_pp'], cost_pp=e[f'{tn}|GROW|normal']['cost_pp_month']) for tn in E.TIERS}
G['G7']['pass_'] = bool(all(G['G7'][tn]['FULL_t'] >= 3 and G['G7'][tn]['Y19_t'] >= 2 for tn in ('100萬', '300萬', '1000萬')))
G['G8'] = dict(pass_=False, why='前瞻未開始')
val_t = gN['VAL']['t']
qa = out['Q7'].get('a') or {}; qb = out['Q7'].get('b') or {}
pa = qa.get('VAL', {}).get('p_net_jmed', 1.0) if qa else 1.0; pbb = qb.get('VAL', {}).get('p_net_jmed', 1.0) if qb else 1.0
if all(G[k].get('pass', G[k].get('pass_')) for k in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7')): verdict = '歷史最高標準通過，前瞻驗證中'
elif val_t >= 2.0 and pa < 0.05 and pbb < 0.05 and G['G1']['pass']: verdict = '待觀察'
else: verdict = '不可用'
G['verdict'] = verdict; G['verdict_inputs'] = dict(VAL_t_exec100萬GROW=val_t, Q7a_p_net_VAL=pa, Q7b_p_net_VAL=pbb, G1=G['G1']['pass'])
mech_q1 = out['Q1']['VAL']['t_did']; ind_ctrl = out['Q3']['VAL']['IND_minus_CTRL_PURE']['mean_pp']
G['mechanism'] = dict(Q1_VAL_t=mech_q1, IND_minus_CTRL_VAL_pp=ind_ctrl, sector_x_stock=bool(mech_q1 >= 2 and ind_ctrl > 0),
                      IND_ONLY_t_VAL=out['paper_key']['IND_ONLY_PURE']['normal']['VAL']['t'], CTRL_t_VAL=out['paper_key']['CTRL_SEL_PURE']['normal']['VAL']['t'])
out['G'] = G

# ---------- 14. 規則書（最新訊號日）----------
lastd = DAYS[0][-2]
s2_ = Rn.make(KEY['SEL_PURE']); hist = X.days_for(0, 'PIT')[:-1]
for d in hist[:-1]: s2_.sleeve(X.cs_cached(d))
sel_, w_, info_ = s2_.sleeve(X.cs_cached(hist[-1]))
cs = X.cs_cached(hist[-1]); t = cs['t']
book = dict(signal_day=str(hist[-1].date()), sectors_top5=list(cs['ST'].index[:5]), sector_table=cs['ST'][['COMP', 'r_REV', 'r_RS', 'r_H52', 'r_NH', 'REV', 'RS126', 'H52', 'NH']].round(3).head(10).reset_index().to_dict('records'),
            picks=[])
for i, w in zip(sel_, w_):
    c = X.COLS[cs['U'][i]]; px = float(X.L.Cn[t, cs['U'][i]])
    row = dict(code=c, ind=str(cs['ind'][i]), sleeve_w=round(float(w), 4), close=px, s52=round(float(cs['s52'][i]), 3), rs61=round(float(cs['rs'][i]), 3),
               y3=None if not np.isfinite(cs['y3'][i]) else round(float(cs['y3'][i]), 1), score=round(float(cs['score'][i]), 3))
    for acct in (1e6, 1e7):
        amt = acct * 0.30 * w; sh = int(amt // px); row[f'{int(acct/1e4)}萬_amount'] = round(amt); row[f'{int(acct/1e4)}萬_lots'] = sh // 1000; row[f'{int(acct/1e4)}萬_odd'] = sh % 1000
    book['picks'].append(row)
book['skeleton'] = '70%：回測骨架是 B（PIT 市值前 100 市值加權）；實務可用 0050 代替（B 與 0050 不同，見 SAT50 描述）'
out['rule_book'] = book
# SAT50：0.7×0050＋0.3×PURE（描述，非事前格點）
sat50 = {p: jsum(lambda j: 0.7 * BEN[j] + 0.3 * EX('SEL_PURE', j), p) for p in ('VAL', 'Y19', 'PIT', 'FULL')}
out['SAT50_desc_exec100萬GROW'] = sat50
# ---------- 切換前版本（COMP、k=5）----------
fc = SCR / 'paper_comp.pkl'
if fc.exists():
    PC = pickle.loads(fc.read_bytes())
    def SC(nm, j, cost='normal'): return pd.Series(PC[(nm, j, cost)]['r'], index=WD[j])
    out['preswitch_COMP'] = {nm: {p: dict(vs0050=jsum(lambda j: SC(nm, j), p), vsB=jsum(lambda j: SC(nm, j), p, lambda j: SC('B', j)),
                                         gross_vs0050=jsum(lambda j: SC(nm, j, 'zero'), p)) for p in ('SEL', 'VAL', 'Y19', 'PIT', 'FULL')}
                             for nm in ('DEF', 'CTRL_DEF', 'IND_ONLY_PURE', 'IND_ONLY_SAT')}
    out['preswitch_COMP']['DEF_minus_CTRL_DEF'] = {p: dict(mean_pp=float(np.median([X.stats((sl(SC('DEF', j), p) - sl(SC('CTRL_DEF', j), p)).to_numpy())['mean_pp'] for j in Rn.JS])),
                                                           t=float(np.median([X.stats((sl(SC('DEF', j), p) - sl(SC('CTRL_DEF', j), p)).to_numpy())['t'] for j in Rn.JS]))) for p in ('VAL', 'PIT')}
R['OOS'] = out
RES.write_text(X.dumps(R), encoding='utf-8')
print('analysis done', G['verdict'], flush=True)

#!/usr/bin/env python3
"""探索（只用挑選期：訊號日 2006-04～2021-08）。輸出 explore.json、explore_events.csv。
(a) 每年被處置剔除的主策略名額（n10、n30）、名次、剩餘天數、第幾次
(b) 剔除股持有期報酬 − 同月其他入選股；子群 × 事件窗；Holm
(c) 對 52 週比、營收、log 成交值、電子業取殘差
(d) 安慰劑（近似未達標組）
另：事件後逐日累積超額路徑。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import drj_core as R
K, V = R.K, R.V
HERE = R.HERE
assert 'PLAN_sha256' in json.loads((HERE / 'results.json').read_text(encoding='utf-8')), '先寫 PLAN'
LISTV = sys.argv[1] if len(sys.argv) > 1 else 'A'
D = R.load_list(use_proxy_gapyears=(LISTV == 'AP'))
DS = R.Disp(D)
Cn, On, G, TR = K.Cn, K.On, K.GROW, K.TRADED
IDX, COLS, N = R.IDX, R.COLS, R.N

tw = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
ma200 = tw.rolling(200).mean()
BULL = (tw > ma200).reindex(IDX, method='ffill')
SL = V.SL
ELEC = {'半導體業', '電腦及週邊設備業', '光電業', '通信網路業', '電子零組件業', '電子通路業', '資訊服務業', '其他電子業'}
ISEL = np.array([SL['industry'].get(c) in ELEC for c in COLS])
A6 = (K.MYADJ / K.MYADJ.shift(6) - 1).to_numpy()

def r_oc(j, a, b):
    """a 開盤 → b 收盤。"""
    if not (TR[a, j] and np.isfinite(On[a, j])): return np.nan
    r = Cn[a, j] / On[a, j]
    if b > a: r *= np.prod(G[a + 1:b + 1, j])
    return r - 1
def r_oo(j, a, b):
    """a 開盤 → b 開盤（b 無成交則用 b 收盤）。"""
    x = r_oc(j, a, b)
    if not np.isfinite(x): return np.nan
    if TR[b, j] and np.isfinite(On[b, j]): return (1 + x) * On[b, j] / Cn[b, j] - 1
    return x

def run(n):
    S0 = R.make_strategy('E0', DS); S1 = R.make_strategy('E1', DS)
    p0 = S0.plan(n); p1 = S1.plan(n)
    m1 = {p[3]: p for p in p1}
    rows = []; slots_by_year = {}
    for i, (e, pk, pool, d) in enumerate(p0):
        if not (R.SEL_A <= d <= R.SEL_B): continue
        e2 = p0[i + 1][0] if i + 1 < len(p0) else None
        if e2 is None: continue
        cs = S0.cross_section(d)
        info = DS.at(d)
        el = cs['el']
        y3p = pd.Series(np.where(el, cs['y3'], np.nan)).rank(pct=True).to_numpy()
        s5p = pd.Series(np.where(el, cs['s52'], np.nan)).rank(pct=True).to_numpy()
        lv = np.log(np.where(cs['t'] is not None, S0.V60[cs['t']], np.nan))
        t = cs['t']
        pk1 = set(m1[d][1].tolist()) if d in m1 else set()
        repl = [j for j in pk1 if j not in set(pk.tolist())]
        with np.errstate(all='ignore'):
            import warnings; warnings.simplefilter('ignore')
            r6max = np.nanmax(A6[t - 9:t + 1], axis=0)
        for rank, j in enumerate(pk):
            j = int(j)
            inf = info.get(j)
            # 過去 30 個交易日內有處置（安慰劑要排除）
            recent = DS.in_disp_day(j, t) is not None
            if not recent:
                g = DS.by_code.get(COLS[j])
                if g is not None: recent = bool(((g.i1 >= t - 30) & (g.i0 <= t)).any())
            row = dict(sig=d, e=e, e2=e2, code=COLS[j], rank=rank + 1, disp=inf is not None,
                       r_full=r_oo(j, e, e2), s52p=s5p[j], y3p=y3p[j], lv60=lv[j], elec=bool(ISEL[j]), r6max=float(r6max[j]) if np.isfinite(r6max[j]) else np.nan,
                       recent_disp=recent, bull=bool(BULL.iloc[t]) if pd.notna(BULL.iloc[t]) else None, lim10=d >= pd.Timestamp('2015-06-01'))
            if inf:
                ei = inf['end_i']
                row.update(nth=inf['nth'], rem=inf['rem'], L=inf['L'], thr=inf['thr'], regime=inf['regime'], kind=inf['kind'], match_min=inf['match_min'], end_i=ei,
                           r_in=r_oc(j, e, ei) if ei >= e else np.nan, r_post=r_oo(j, ei + 1, e2) if ei + 1 < e2 else np.nan)
            rows.append(row)
        if repl:
            rr = [r_oo(j, e, e2) for j in repl]
            rows.append(dict(sig=d, e=e, e2=e2, code='REPL', rank=0, disp=False, r_full=float(np.nanmean(rr)), repl_n=len(repl), repl_codes=','.join(COLS[j] for j in repl)))
    ev = pd.DataFrame(rows)
    return ev, p0, p1

def month_t(x, by):
    s = pd.Series(np.asarray(x, float), index=by).dropna()
    g = s.groupby(level=0).mean()
    return dict(mean_pp=float(g.mean() * 100) if len(g) else np.nan, t=R.tstat(g.values), n_months=int(len(g)), n_events=int(len(s)))

def excess(ev, col='r_full', win_other=None):
    """事件 − 同月其他入選股（非處置、非 REPL）在同一窗的平均。"""
    base = ev[(ev.code != 'REPL')]
    out = []
    for d, g in base.groupby('sig'):
        oth = g[~g.disp]
        for _, r in g[g.disp].iterrows():
            if col == 'r_full':
                o = oth.r_full.mean()
            else:
                # 同窗：其他股 e 開盤 → end 收盤 或 end+1 開盤 → e2 開盤
                js = [K.CI[c] for c in oth.code]
                if col == 'r_in':
                    if not np.isfinite(r.get('r_in', np.nan)): continue
                    o = np.nanmean([r_oc(j, int(r.e), int(r.end_i)) for j in js])
                else:
                    if not np.isfinite(r.get('r_post', np.nan)): continue
                    o = np.nanmean([r_oo(j, int(r.end_i) + 1, int(r.e2)) for j in js])
            out.append(dict(sig=d, code=r.code, x=r[col] - o))
    return pd.DataFrame(out)

def main():
    out = {'list_version': LISTV, 'period': 'signal days 2006-04-01..2021-08-31 only'}
    ev10, p0, p1 = run(10)
    ev30, _, _ = run(30)
    assert ev10.sig.max() <= R.SEL_B and ev30.sig.max() <= R.SEL_B
    ev10.to_csv(HERE / f'explore_events_{LISTV}.csv', index=False)
    # ---- (a) ----
    a = {}
    for n, ev in ((10, ev10), (30, ev30)):
        x = ev[(ev.code != 'REPL') & ev.disp]
        a[f'n{n}'] = dict(total_slots=int(len(x)), total_slot_months=int(ev[ev.code != 'REPL'].sig.nunique()),
                          by_year={int(y): int(c) for y, c in x.groupby(x.sig.dt.year).size().items()},
                          rank_median=float(x['rank'].median()) if len(x) else None, rem_dist={int(k): int(v) for k, v in x.rem.value_counts().sort_index().items()},
                          nth_dist={int(k): int(v) for k, v in x.nth.value_counts().items()}, regime={k: int(v) for k, v in x.regime.value_counts().items()},
                          kind={k: int(v) for k, v in x.kind.value_counts().items()}, n_months_with_affected=int(x.sig.nunique()),
                          codes=sorted(x.code.unique().tolist()))
    out['a_affected_slots'] = a
    # ---- (b) 子群 × 窗 ----
    x = ev10[(ev10.code != 'REPL')].copy()
    x['disp'] = x.disp.astype(bool); x['recent_disp'] = x.recent_disp.astype(bool)
    dsp = x[x.disp].copy()
    dsp['lim10'] = dsp.lim10.astype(bool)
    dsp['Rg'] = np.where(dsp.rem == 0, 'R0', np.where(dsp.rem >= dsp.thr, 'R2', 'R1'))
    subs = {'all': dsp.index, 'nth1': dsp.index[dsp.nth == 1], 'nth2+': dsp.index[dsp.nth >= 2], 'R0': dsp.index[dsp.Rg == 'R0'], 'R1': dsp.index[dsp.Rg == 'R1'],
            'R2': dsp.index[dsp.Rg == 'R2'], 'bull': dsp.index[dsp.bull == True], 'bear': dsp.index[dsp.bull == False], 'rev_hi': dsp.index[dsp.y3p >= 0.5],
            'rev_lo': dsp.index[dsp.y3p < 0.5], 'lim7': dsp.index[~dsp.lim10], 'lim10': dsp.index[dsp.lim10], 'prepay_era': dsp.index[dsp.regime == 'prepay_only'],
            'periodic_era': dsp.index[dsp.regime != 'prepay_only']}
    tests = {}; pv = {}
    for w in ('r_full', 'r_in', 'r_post'):
        ex = excess(x, w)
        ex = ex.set_index(['sig', 'code'])
        for sname, ix in subs.items():
            keys = list(zip(dsp.loc[ix, 'sig'], dsp.loc[ix, 'code']))
            exs = ex.reindex([k for k in keys if k in ex.index])
            if len(exs) < 3: continue
            mt = month_t(exs.x.values, exs.index.get_level_values(0))
            raw = month_t(dsp.loc[ix, w].values, dsp.loc[ix, 'sig'].values)
            tests[f'{w}|{sname}'] = dict(excess=mt, raw_mean_pp=raw['mean_pp'])
            pv[f'{w}|{sname}'] = R.p_from_t(mt['t'], mt['n_months'])
    H = R.holm({k: v for k, v in pv.items() if np.isfinite(v)})
    for k, (p, ph) in H.items():
        tests[k]['p'] = p; tests[k]['p_holm'] = ph
    out['b_event_tests'] = tests
    out['b_n_tests_holm'] = len(H)
    # 替補股
    rp = ev10[ev10.code == 'REPL']
    oth = x[~x.disp].groupby('sig').r_full.mean()
    out['b_replacements'] = dict(n_months=int(len(rp)), repl_mean_minus_others_pp=month_t((rp.set_index('sig').r_full - oth.reindex(rp.sig).values).values, rp.sig.values))
    # 剔除股 − 替補股（＝組合層差的來源）
    dm = dsp.groupby('sig').r_full.mean(); rm = rp.set_index('sig').r_full
    jj = dm.index.intersection(rm.index)
    out['b_affected_minus_replacement'] = dict(mean_pp=float((dm[jj] - rm[jj]).mean() * 100), t=R.tstat((dm[jj] - rm[jj]).values), n_months=int(len(jj)))
    # ---- (c) 殘差 ----
    def resid(df, flag):
        z = df.dropna(subset=['r_full', 's52p', 'y3p', 'lv60']).copy()
        z['f'] = z[flag].astype(float); z['elec'] = z.elec.astype(float)
        cols = ['s52p', 'y3p', 'lv60', 'elec', 'f']
        Z = z[['r_full'] + cols].groupby(z.sig).transform(lambda s: s - s.mean())
        Xm = Z[cols].to_numpy(); y = Z['r_full'].to_numpy()
        keep = np.isfinite(Xm).all(axis=1) & np.isfinite(y)
        Xm, y, gs = Xm[keep], y[keep], z.sig.to_numpy()[keep]
        b = np.linalg.lstsq(Xm, y, rcond=None)[0]
        u = y - Xm @ b
        XtX = np.linalg.inv(Xm.T @ Xm)
        meat = np.zeros((len(cols), len(cols)))
        for gv in np.unique(gs):
            m = gs == gv; sc = Xm[m].T @ u[m]; meat += np.outer(sc, sc)
        G_ = len(np.unique(gs)); kk = len(cols); nn = len(y)
        Vc = XtX @ meat @ XtX * (G_ / (G_ - 1)) * ((nn - 1) / (nn - kk))
        se = np.sqrt(np.diag(Vc))
        return {c: dict(coef_pp=float(b[i] * 100), t_cluster=float(b[i] / se[i])) for i, c in enumerate(cols)} | {'n': int(nn), 'months': int(G_)}
    x2 = x.copy()
    x2['plac'] = (~x2.disp) & (~x2.recent_disp) & (x2.r6max >= 0.25)
    out['c_residual_disp'] = resid(x2, 'disp')
    out['c_residual_placebo'] = resid(x2[~x2.disp], 'plac')
    # ---- (d) 安慰劑 ----
    pl_ = x2[x2.plac]
    ex_p = []
    for d, g in x2.groupby('sig'):
        o = g[(~g.disp) & (~g.plac)].r_full.mean()
        for _, r in g[g.plac].iterrows(): ex_p.append(dict(sig=d, x=r.r_full - o))
    ex_p = pd.DataFrame(ex_p)
    ex_d = excess(x, 'r_full')
    out['d_placebo'] = dict(n_events=int(len(pl_)), excess=month_t(ex_p.x.values, ex_p.sig.values) if len(ex_p) else None,
                            disposal_excess=month_t(ex_d.x.values, ex_d.sig.values),
                            r6max_median_placebo=float(pl_.r6max.median()), r6max_median_disposal=float(dsp.r6max.median()),
                            r6max_q25_disposal=float(dsp.r6max.quantile(0.25)))
    # ---- 路徑：e 開盤後第 k 日收盤的累積超額（vs 同月其他非處置入選股） ----
    path = {}
    for lab, sub in (('disposal', dsp), ('placebo', pl_)):
        acc = []
        for _, r in sub.iterrows():
            e = int(r.e); j = K.CI[r.code]
            oth = [K.CI[c] for c in x[(x.sig == r.sig) & (~x.disp)].code]
            v = []
            for k in range(0, 21):
                if e + k >= len(IDX): break
                a_ = r_oc(j, e, e + k); b_ = np.nanmean([r_oc(o, e, e + k) for o in oth]) if oth else np.nan
                v.append(a_ - b_)
            acc.append(v + [np.nan] * (21 - len(v)))
        A = np.array(acc, float)
        path[lab] = [round(float(np.nanmean(A[:, k])) * 100, 2) for k in range(21)] if len(A) else []
    out['path_cum_excess_pp_k0_20'] = path
    # 依 rem 的路徑終點（處置結束點）
    out['n10_events_disp_by_year'] = {int(y): int(c) for y, c in dsp.groupby(dsp.sig.dt.year).size().items()}
    (HERE / f'explore_{LISTV}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(out['a_affected_slots'], ensure_ascii=False, default=str)[:3000])
    for k, v in tests.items(): print(k, v)
    for k in ('b_replacements', 'b_affected_minus_replacement', 'c_residual_disp', 'c_residual_placebo', 'd_placebo', 'path_cum_excess_pp_k0_20'):
        print(k, json.dumps(out[k], ensure_ascii=False, default=str))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""strategy_ensemble 主程式：相關矩陣 → E1～E6 → 挑選與判定 → 窮舉子集（E1/E2）→ SPA／max-t → 邊際貢獻。
規則全部在 results.json 的 PREREG（prereg.py 寫入，elib 開頭比對 sha256）。執行：python3 -I -B run.py"""
import json, math, itertools, time, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import elib as E   # noqa: E402

PR = E.PR; H = E.H; CAL = E.CAL; T = E.T
ME_IDX = np.array(sorted(E.MONTH_END))
FIRST_ALLOC = {'S_long': '2006-05-11', 'S_recent': '2021-02-26'}
PER = PR['periods']
BEARW = PR['metrics']['bear']
t0 = time.time()

def bounds(a, b):
    ia = int(CAL.searchsorted(pd.Timestamp(a))) - 1; ib = int(CAL.searchsorted(pd.Timestamp(b), 'right')) - 1
    return max(ia, 0), ib

def full(series):
    a = np.full(T, np.nan); a[CAL.get_indexer(series.index)] = series.to_numpy(); return a

def fstats(x, a, b):
    ia, ib = bounds(a, b)
    seg = x[ia:ib + 1]
    if not np.isfinite(seg[0]):
        k = int(np.argmax(np.isfinite(seg))); ia += k; seg = seg[k:]
    yrs = (CAL[ib] - CAL[ia]).days / 365.25
    r = seg[1:] / seg[:-1] - 1
    cagr = (seg[-1] / seg[0]) ** (1 / yrs) - 1
    dd = float((seg / np.maximum.accumulate(seg) - 1).min())
    me = ME_IDX[(ME_IDX > ia) & (ME_IDX < ib)]
    pts = np.r_[ia, me, ib]
    mv = x[pts]; m = mv[1:] / mv[:-1] - 1
    return {'CAGR': float(cagr), 'vol': float(r.std(ddof=1) * math.sqrt(252)), 'maxDD': dd, 'Calmar': float(cagr / abs(dd)) if dd < 0 else np.nan, 'm': m}

def tst(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1) * math.sqrt(len(d))) if len(d) > 2 and d.std(ddof=1) > 0 else np.nan

REF = {k: E.P[:, E.CI[k]] for k in ('E0050', 'N10', 'N30')}
REFNAME = {'E0050': '0050', 'N10': 'N10', 'N30': 'N30'}
def report(x, a, b, extra_refs=None):
    s = fstats(x, a, b); o = {k: (round(v, 4) if k != 'Calmar' else round(v, 3)) for k, v in s.items() if k != 'm'}
    o['n_months'] = len(s['m'])
    refs = dict(REF); refs.update(extra_refs or {})
    for k, rx in refs.items():
        mr = fstats(rx, a, b)['m']; d = s['m'] - mr
        o[f't_vs_{REFNAME.get(k, k)}'] = round(tst(d), 2); o[f'mdiff_vs_{REFNAME.get(k, k)}_pct'] = round(float(d.mean() * 100), 2)
    return o

def yearly(x, a, b):
    ia, ib = bounds(a, b); seg = pd.Series(x[ia:ib + 1], index=CAL[ia:ib + 1]).dropna(); r = seg.pct_change().iloc[1:]
    return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

def bear(x):
    o = {}
    for k, (a, b) in BEARW.items():
        ia, ib = bounds(a, b); seg = x[ia:ib + 1]
        if not np.isfinite(seg).all(): o[k] = None; continue
        o[k] = {'ret_pct': round(float(seg[-1] / seg[0] - 1) * 100, 1), 'maxDD_pct': round(float((seg / np.maximum.accumulate(seg) - 1).min()) * 100, 1)}
    return o

OUT = {}
# ================= 1. 相關矩陣 =================
def mret(k, a, b):
    s = fstats(E.P[:, E.CI[k]], a, b)['m']; return s
corr = {}
for setname, names in PR['sets'].items():
    for w, (a, b) in PER[setname].items():
        M = pd.DataFrame({k: mret(k, a, b) for k in names})
        if 'DISP_REAL' in names:   # 2021-10 之前 DISP_REAL＝0050，相關矩陣中遮掉
            ia, ib = bounds(a, b); me = ME_IDX[(ME_IDX > ia) & (ME_IDX < ib)]; ends = np.r_[me, ib]
            M.loc[CAL[ends] < pd.Timestamp('2021-10-31'), 'DISP_REAL'] = np.nan
        corr[f'{setname}|{w}'] = M.corr().round(2).to_dict()
names_l = PR['sets']['S_long']
M = pd.DataFrame({k: mret(k, '2006-05-12', '2026-10-12') for k in names_l}); corr['S_long|full_2006-05_2026-10'] = M.corr().round(2).to_dict()
OUT['correlation_monthly'] = corr

# ================= 2. 單一策略 =================
single = {}
for setname, names in PR['sets'].items():
    for k in names:
        x = E.P[:, E.CI[k]]
        single.setdefault(k, {})
        for w, (a, b) in PER[setname].items():
            if setname == 'S_recent' and k in names_l and w in single[k]: pass
            single[k][f'{setname}|{w}'] = report(x, a, b)
OUT['standalone'] = single

# ================= 3. E1～E6 =================
def wf(setname, m):
    if m == 'E6':
        d = 'DISP_PX' if setname == 'S_long' else 'DISP_REAL'
        return ['N10', d], E.w_fixed({'N10': 0.8, d: 0.2})
    return PR['sets'][setname], E.METHODS[m]
ens = {}; navstore = {}; wlogs = {}
for setname in PR['sets']:
    for m in ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']:
        names, f = wf(setname, m)
        for cost in (0.005, 0.01):
            nav, wl = E.simulate(names, f, FIRST_ALLOC[setname], cost=cost, return_w=True)
            x = full(nav)
            key = f'{setname}|{m}' + ('' if cost == 0.005 else '|cost1pct')
            if cost == 0.005:
                navstore[key] = x; wlogs[key] = wl
            e6 = None
            r = {w: report(x, a, b) for w, (a, b) in PER[setname].items()}
            turn = np.mean([t for _, _, t in wl[1:]]) if len(wl) > 1 else 0
            ens[key] = {'windows': r, 'mean_monthly_turnover_sum_abs_dw': round(float(turn), 4)}
            if cost == 0.005:
                ens[key]['yearly_pct'] = yearly(x, PER[setname]['selection'][0], PER[setname][list(PER[setname])[-1]][1])
                ens[key]['bear'] = bear(x)
                ws = pd.DataFrame([w for _, w, _ in wl]); ens[key]['mean_weights'] = ws.mean().round(3).to_dict()
                ens[key]['mean_weights_by_period'] = {}
                dts = pd.DatetimeIndex([d for d, _, _ in wl])
                for w, (a, b) in PER[setname].items():
                    mk = (dts >= pd.Timestamp(a) - pd.Timedelta(days=31)) & (dts <= pd.Timestamp(b))
                    ens[key]['mean_weights_by_period'][w] = ws[mk].mean().round(3).to_dict()
# 對 E6 的配對
for setname in PR['sets']:
    e6 = navstore[f'{setname}|E6']
    for m in ['E1', 'E2', 'E3', 'E4', 'E5']:
        k = f'{setname}|{m}'
        ens[k]['vs_E6'] = {}
        for w, (a, b) in PER[setname].items():
            d = fstats(navstore[k], a, b)['m'] - fstats(e6, a, b)['m']
            ens[k]['vs_E6'][w] = {'t': round(tst(d), 2), 'mdiff_pct': round(float(d.mean() * 100), 2)}
OUT['ensembles'] = ens
OUT['weights_log_tail'] = {k: [(str(d.date()), w, t) for d, w, t in v[-3:]] for k, v in wlogs.items()}
# 參照：N10、N30、0050 的逐年與空頭
refs_y = {}
for setname in PR['sets']:
    a0 = PER[setname]['selection'][0]; b0 = PER[setname][list(PER[setname])[-1]][1]
    for k in ('N10', 'N30', 'E0050', 'DISP_PX', 'DISP_REAL'):
        refs_y[f'{setname}|{k}'] = {'yearly_pct': yearly(E.P[:, E.CI[k]], a0, b0), 'bear': bear(E.P[:, E.CI[k]])}
OUT['reference_yearly_bear'] = refs_y

# ---- 挑選與判定 ----
def pick_best(cands, get):
    """cands: list of keys；get(key)->(t, calmar)。t 最高；與第二名差 ≤0.05 → Calmar。"""
    sc = sorted(cands, key=lambda k: -get(k)[0])
    best = sc[0]
    if len(sc) > 1 and get(sc[0])[0] - get(sc[1])[0] <= 0.05 and get(sc[1])[1] > get(sc[0])[1]: best = sc[1]
    return best, sc
def passes(r, n10):
    return bool(r['Calmar'] > n10['Calmar'] and r['t_vs_0050'] >= n10['t_vs_0050'])
sel = {}
for setname in PR['sets']:
    keys = [f'{setname}|{m}' for m in ['E1', 'E2', 'E3', 'E4', 'E5']]
    best, order = pick_best(keys, lambda k: (ens[k]['windows']['selection']['t_vs_0050'], ens[k]['windows']['selection']['Calmar']))
    oos = [w for w in PER[setname] if w != 'selection']
    ok = {w: passes(ens[best]['windows'][w], single['N10'][f'{setname}|{w}']) for w in oos}
    n = sum(ok.values())
    if setname == 'S_long': verdict = '可用' if n == 2 else ('待觀察' if n == 1 else '不可用')
    else: verdict = '待觀察' if n == 1 else '不可用'
    e6ok = {w: passes(ens[f'{setname}|E6']['windows'][w], single['N10'][f'{setname}|{w}']) for w in oos}
    sel[setname] = {'selection_t_order': [(k.split('|')[1], ens[k]['windows']['selection']['t_vs_0050'], ens[k]['windows']['selection']['Calmar']) for k in order],
                    'picked': best.split('|')[1], 'pass': ok, 'verdict': verdict, 'E6_same_rule_info': e6ok}
OUT['selection_E1_E5'] = sel
print(json.dumps(sel, ensure_ascii=False), flush=True)
print('main done', round(time.time() - t0, 1), flush=True)

# ================= 4. 窮舉 =================
EXH = {}; exh_rows = []
MSTORE = {}
for setname, names in PR['sets'].items():
    wins = PER[setname]
    subsets = [c for r in range(1, len(names) + 1) for c in itertools.combinations(names, r)]
    metr = {}
    for c in subsets:
        for m in (['E1'] if len(c) == 1 else ['E1', 'E2']):
            x = full(E.simulate(c, E.METHODS[m], FIRST_ALLOC[setname]))
            rec = {}
            for w, (a, b) in wins.items():
                s = fstats(x, a, b)
                rec[w] = s
            metr[(c, m)] = rec
            if len(c) == 1: metr[(c, 'E2')] = rec
    MSTORE[setname] = metr
    print(setname, 'exhaustive sims', len(metr), round(time.time() - t0, 1), flush=True)
    refm = {w: {k: fstats(REF[k], a, b)['m'] for k in REF} for w, (a, b) in wins.items()}
    n10s = {w: fstats(REF['N10'], a, b) for w, (a, b) in wins.items()}
    combos = [(c, m) for (c, m) in metr if len(c) >= 2]
    def tv(cm, w, ref='E0050'): return tst(metr[cm][w]['m'] - refm[w][ref])
    tsel = {cm: tv(cm, 'selection') for cm in combos}
    order = sorted(combos, key=lambda cm: (-round(tsel[cm], 6), -metr[cm]['selection']['Calmar']))
    top3 = order[:3]
    oos = [w for w in wins if w != 'selection']
    n10t = {w: tst(n10s[w]['m'] - refm[w]['E0050']) for w in wins}
    def row(cm):
        o = {'subset': list(cm[0]), 'weighting': cm[1]}
        for w in wins:
            s = metr[cm][w]
            o[w] = {'CAGR': round(s['CAGR'], 4), 'vol': round(s['vol'], 4), 'maxDD': round(s['maxDD'], 4), 'Calmar': round(s['Calmar'], 3),
                    't_vs_0050': round(tv(cm, w), 2), 't_vs_N10': round(tst(s['m'] - refm[w]['N10']), 2), 't_vs_N30': round(tst(s['m'] - refm[w]['N30']), 2)}
        return o
    toprows = []
    for cm in top3:
        o = row(cm)
        ok = {w: bool(metr[cm][w]['Calmar'] > n10s[w]['Calmar'] and tv(cm, w) >= n10t[w]) for w in oos}
        n = sum(ok.values())
        o['pass'] = ok
        o['verdict'] = ('可用' if n == 2 else ('待觀察' if n == 1 else '不可用')) if setname == 'S_long' else ('待觀察' if n == 1 else '不可用')
        x = full(E.simulate(cm[0], E.METHODS[cm[1]], FIRST_ALLOC[setname]))
        o['yearly_pct'] = yearly(x, wins['selection'][0], wins[list(wins)[-1]][1]); o['bear'] = bear(x)
        toprows.append(o)
    # 分佈
    dist = {}
    for w in wins:
        cal = np.array([metr[cm][w]['Calmar'] for cm in combos]); tt = np.array([tv(cm, w) for cm in combos]); cg = np.array([metr[cm][w]['CAGR'] for cm in combos])
        dd = np.array([metr[cm][w]['maxDD'] for cm in combos])
        q = lambda v: [round(float(np.nanpercentile(v, p)), 3) for p in (25, 50, 75)]
        better_c = cal > n10s[w]['Calmar']; better_t = tt >= n10t[w]
        dist[w] = {'N10': {'CAGR': round(n10s[w]['CAGR'], 4), 'maxDD': round(n10s[w]['maxDD'], 4), 'Calmar': round(n10s[w]['Calmar'], 3), 't_vs_0050': round(n10t[w], 2)},
                   'CAGR_q25_50_75': q(cg), 'maxDD_q25_50_75': q(dd), 'Calmar_q25_50_75': q(cal), 't_vs_0050_q25_50_75': q(tt),
                   'share_Calmar_gt_N10': round(float(better_c.mean()), 3), 'share_t_ge_N10': round(float(better_t.mean()), 3),
                   'share_both': round(float((better_c & better_t).mean()), 3),
                   'by_weighting': {m: {'median_Calmar': round(float(np.median(cal[[c[1] == m for c in combos]])), 3),
                                        'median_t_vs_0050': round(float(np.median(tt[[c[1] == m for c in combos]])), 2),
                                        'share_both': round(float((better_c & better_t)[[c[1] == m for c in combos]].mean()), 3)} for m in ('E1', 'E2')}}
    # 挑選期排名前 10%（依 t）在驗證期的表現 vs 其餘（挑選期排名是否有預測力）
    rk = {cm: i for i, cm in enumerate(order)}
    persist = {}
    for w in oos:
        a1 = np.array([rk[cm] for cm in combos]); b1 = np.array([tv(cm, w) for cm in combos]); c1 = np.array([metr[cm][w]['Calmar'] for cm in combos])
        persist[w] = {'spearman_sel_t_vs_oos_t': round(float(pd.Series(-a1).corr(pd.Series(b1), method='spearman')), 3),
                      'spearman_sel_t_vs_oos_Calmar': round(float(pd.Series(-a1).corr(pd.Series(c1), method='spearman')), 3)}
    # 多重檢定
    mt = {}
    for w in wins:
        for bname, bref in (('N10', 'N10'), ('0050', 'E0050')):
            D = np.column_stack([metr[cm][w]['m'] - refm[w][bref] for cm in combos])
            r = E.spa_maxt(D)
            adj = r.pop('_adj_p'); tt_ = r.pop('_t')
            r['best_model'] = {'subset': list(combos[r['best_model_index']][0]), 'weighting': combos[r['best_model_index']][1]}
            r['top3_selection_models_adj_p'] = [{'subset': list(cm[0]), 'w': cm[1], 't': round(float(tt_[combos.index(cm)]), 2), 'maxt_adj_p': round(float(adj[combos.index(cm)]), 4)} for cm in top3]
            r['share_models_t_gt_0'] = round(float((tt_ > 0).mean()), 3)
            mt[f'{w}|vs_{bname}'] = r
        # 風險調整版（事後縮放，診斷）
        sN = n10s[w]['vol']
        D = np.column_stack([metr[cm][w]['m'] * (sN / metr[cm][w]['vol']) - refm[w]['N10'] for cm in combos])
        r = E.spa_maxt(D); r.pop('_adj_p'); r.pop('_t'); r['best_model'] = {'subset': list(combos[r['best_model_index']][0]), 'weighting': combos[r['best_model_index']][1]}
        mt[f'{w}|volmatched_vs_N10_diag'] = r
    print(setname, 'spa done', round(time.time() - t0, 1), flush=True)
    # 邊際貢獻
    marg = {}
    for s_ in names:
        for m in ('E1', 'E2'):
            o = {}
            for w in wins:
                dc, dt_ = [], []
                for c in subsets:
                    if s_ in c: continue
                    c2 = tuple(sorted(c + (s_,), key=names.index))
                    a_ = metr[(c, m)][w]; b_ = metr[(c2, m)][w]
                    dc.append(b_['Calmar'] - a_['Calmar'])
                    dt_.append(tv((c2, m), w) - tst(a_['m'] - refm[w]['E0050']) if len(c) > 1 or c[0] != 'E0050' else tv((c2, m), w))
                dc, dt_ = np.array(dc), np.array(dt_, float)
                o[w] = {'dCalmar_mean': round(float(np.nanmean(dc)), 3), 'dCalmar_median': round(float(np.nanmedian(dc)), 3), 'dCalmar_share_pos': round(float(np.nanmean(dc > 0)), 3),
                        'dt_mean': round(float(np.nanmean(dt_)), 2), 'dt_median': round(float(np.nanmedian(dt_)), 2), 'dt_share_pos': round(float(np.nanmean(dt_ > 0)), 3), 'n_pairs': int(len(dc))}
            marg[f'{s_}|{m}'] = o
    EXH[setname] = {'n_combos': len(combos), 'top3_by_selection_t': toprows, 'distribution': dist, 'rank_persistence': persist, 'multiple_testing': mt, 'marginal_contribution': marg}
    for cm in combos:
        r_ = {'set': setname, 'subset': '+'.join(cm[0]), 'k': len(cm[0]), 'w': cm[1]}
        for w in wins:
            s = metr[cm][w]; r_.update({f'{w}_CAGR': s['CAGR'], f'{w}_maxDD': s['maxDD'], f'{w}_Calmar': s['Calmar'], f'{w}_t0050': tv(cm, w), f'{w}_tN10': tst(s['m'] - refm[w]['N10'])})
        exh_rows.append(r_)
OUT['exhaustive'] = EXH
pd.DataFrame(exh_rows).to_csv(H / 'exhaustive_all.csv.gz', index=False, compression='gzip')
pd.DataFrame({k: v for k, v in navstore.items()}, index=CAL).to_parquet(H / 'ensemble_nav.parquet')
RES = E.RES; RES['RESULTS'] = OUT
(H / 'results.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda v: None if isinstance(v, float) and not np.isfinite(v) else (float(v) if isinstance(v, (np.floating,)) else (int(v) if isinstance(v, np.integer) else str(v)))), encoding='utf-8')
print('all done', round(time.time() - t0, 1))

#!/usr/bin/env python3
"""步驟 5b：容量表統計與事前登記判定（讀 run_cap.py 的輸出；PREREG 必須存在）。寫入 results.json 的 RESULTS。
每格（strat|variant|tier|ver）：40 seed 日報酬平均的「可執行 AB」淨值 → 年化、最大回撤、對 0050 t（訊號窗超額，一般 t）、分段、逐 seed 分布、
端點（拿掉最好 1／3／5／10 窗、起點 2008）、2019 年後 t、零股制度分段（PRE_ODD／POST_ODD）、對零衝擊 AB_V4（B10 另對零衝擊 AB_V4_B10）的配對差、
損益兩平滑價（每日報酬扣 x × 成交金額比例，t 降到 2.0、超額均值降到 0 的 x）、成本拆解（占權益 %／年）、執行計數；j 分布（20 seeds）中位數與 p10。
用法：python3 -I -B analyze.py [--dry]
"""
import sys, os, json, hashlib, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import ecore6 as E

DRY = '--dry' in sys.argv
SRC = E.DATA if not DRY else Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/cap_dry')
RES_F = HERE / 'results.json'
if not DRY:
    res = json.loads(RES_F.read_text(encoding='utf-8'))
    assert 'PREREG' in res and hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256']
NAV = pd.read_parquet(SRC / 'cap_navs.parquet'); TV = pd.read_parquet(SRC / 'cap_tv.parquet')
ACC = json.load(open(SRC / 'cap_acc.json'))
D = E.Data(load_j=True)
E50 = D.E50
BN = D.BENCH
TIERS = list(E.TIERS); VERS = ('GROW', 'FIX')

def groups():
    g = {}
    for k in NAV.columns:
        cell = k.rsplit('|', 1)[0]; g.setdefault(cell, []).append(k)
    return g
G = groups()

def avg_nav(cols):
    R = NAV[cols].pct_change().mean(axis=1)
    first = NAV[cols].iloc[:, 0].first_valid_index()
    R = R[R.index > first].fillna(0)
    nav = pd.concat([pd.Series([1.0], index=[first]), (1 + R).cumprod()])
    tv = TV[cols].mean(axis=1).reindex(nav.index).fillna(0)
    return nav, tv

def sd_for(strat):
    return D.bounds(strat)[2]

def tt(nav, sd, a=E.A_, b=E.B_, bench=None):
    return E.tstat(E.excess(nav, (E50 if bench is None else bench).reindex(nav.index), sd, a, b))

def drop_best(x, k):
    return E.tstat(x.drop(x.nlargest(k).index)) if len(x) > k + 2 else float('nan')

def breakeven(nav, tv, sd, target):
    r = nav.pct_change().fillna(0)
    def f(x):
        n2 = (1 + r - x * tv).cumprod()
        ex = E.excess(n2, E50.reindex(n2.index), sd)
        return E.tstat(ex) if target == 't2' else float(ex.mean())
    goal = 2.0 if target == 't2' else 0.0
    if f(0.0) < goal: return 0.0
    lo, hi = 0.0, 0.05
    if f(hi) >= goal: return float('inf')
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if f(m) >= goal: lo = m
        else: hi = m
    return lo

def cell_stats(cell, full=True):
    cols = G[cell]; strat = cell.split('|')[0]
    sd = sd_for(strat)
    nav, tv = avg_nav(cols)
    x = E.excess(nav, E50.reindex(nav.index), sd)
    out = dict(n_seeds=len(cols), CAGR=E.cagr_mdd(nav, E.A_, E.B_)[0], MDD=E.cagr_mdd(nav, E.A_, E.B_)[1], t=E.tstat(x),
               mean_excess_pp=float(x.mean() * 100), sd_excess_pp=float(x.std(ddof=1) * 100), n_windows=int(len(x)))
    if not full: return out, nav, tv
    for p, (a, b) in E.PER.items():
        if p == 'ALL': continue
        try: out[f'CAGR_{p}'] = E.cagr_mdd(nav, a, b)[0]
        except Exception: out[f'CAGR_{p}'] = float('nan')
        out[f't_{p}'] = tt(nav, sd, a, b)
    out.update({f't_drop_best{k}': drop_best(x, k) for k in (1, 3, 5, 10)})
    ts, cg = [], []
    for c in cols:
        n1 = NAV[c].dropna(); ts.append(tt(n1, sd)); cg.append(E.cagr_mdd(n1, E.A_, E.B_)[0])
    ts = np.array(ts)
    out['seed_t'] = dict(median=float(np.median(ts)), min=float(ts.min()), max=float(ts.max()), p10=float(np.percentile(ts, 10)),
                         n_lt_2=int((ts < 2.0).sum()), n_lt_1_8=int((ts < 1.8).sum()))
    out['seed_CAGR'] = dict(median=float(np.median(cg)), min=float(np.min(cg)), max=float(np.max(cg)))
    # 配對差：對零衝擊（AB → AB_V4|normal；B10 → AB_V4_B10|normal）
    ref = BN['AB_V4_B10|normal'] if strat.startswith('B10') else BN['AB_V4|normal']
    if strat in ('AB', 'B10'):
        dlt = E.win_ret(nav, sd) - E.win_ret(ref.reindex(nav.index), sd); dlt = dlt[(dlt.index >= pd.Timestamp(E.A_))].dropna()
        out['paired_vs_zero_impact'] = dict(mean_pp=float(dlt.mean() * 100), t=E.tstat(dlt), MDE80_pp_yr=float(2.8 * dlt.std(ddof=1) / math.sqrt(len(dlt)) * 100 * 12),
                                            CAGR_zero=E.cagr_mdd(ref.reindex(nav.index).dropna(), E.A_, E.B_)[0])
        out['breakeven_extra_slip_t2'] = breakeven(nav, tv, sd, 't2'); out['breakeven_extra_slip_excess0'] = breakeven(nav, tv, sd, 'ex0')
    # 成本與執行
    A = [ACC[c] for c in cols]
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    out['cost_pct_equity_per_yr'] = {k: float(np.mean([a.get('pct_' + k, 0.0) for a in A]) * 100 / yrs) for k in ('tax', 'fee', 'impact', 'base_slip', 'odd_spread', 'odd_timing', 'e2_drift')}
    out['exec_counts_per_seed'] = {k: float(np.mean([a.get(k, 0.0) for a in A])) for k in ('n_orders', 'n_lot_orders', 'n_odd_orders', 'n_cap_hit', 'n_zcap_hit', 'n_lim_block', 'n_e1', 'deferred_days')}
    out['avg_cash_frac'] = float(np.mean([a['cash_frac_sum'] for a in A]) / len(nav))
    yr = {}
    for y in range(2006, 2027):
        a, b = f'{y}-01-01', f'{y}-12-31'
        s = nav[(nav.index >= a) & (nav.index <= b)]; e = E50.reindex(nav.index)[(nav.index >= a) & (nav.index <= b)]
        if len(s) < 5: continue
        i0 = nav.index.get_loc(s.index[0]); base = nav.iloc[i0 - 1] if i0 > 0 else s.iloc[0]
        eb = E50.reindex(nav.index).iloc[i0 - 1] if i0 > 0 else e.iloc[0]
        yr[y] = float((s.iloc[-1] / base - e.iloc[-1] / eb) * 100)
    out['yearly_excess_vs_0050_pp'] = yr
    return out, nav, tv

def run():
    R = {}; navs_out = {}
    cells = sorted(G)
    main_cells = [c for c in cells if not ('j' in c.split('|')[0] and c.split('|')[0] not in ('AB', 'B10'))]
    for c in main_cells:
        st, nav, tv = cell_stats(c); R[c] = st; navs_out[c] = nav
    # j 分布
    J = {}
    for base in ('AB', 'B10'):
        for ti in TIERS:
            for ve in VERS:
                ts, cg, rows = [], [], {}
                for j in range(13):
                    c = f'{base}j{j}|M3V_point|{ti}|{ve}'
                    if c not in G: continue
                    st, nav, _ = cell_stats(c, full=False); ts.append(st['t']); cg.append(st['CAGR']); rows[j] = dict(t=st['t'], CAGR=st['CAGR'])
                if ts:
                    J[f'{base}|{ti}|{ve}'] = dict(per_j=rows, t_median=float(np.median(ts)), t_p10=float(np.percentile(ts, 10)), CAGR_median=float(np.median(cg)),
                                                  CAGR_p10=float(np.percentile(cg, 10)), n_j=len(ts))
    # 參照
    sd = D.sig_days
    ref = {}
    for k in ('AB_V4|normal', 'AB_V4_B10|normal', 'AB_V4|EW'):
        n = BN[k].dropna(); n = n[n.index >= D.IDX[D.t0]]
        ref[k] = dict(CAGR=E.cagr_mdd(n, E.A_, E.B_)[0], MDD=E.cagr_mdd(n, E.A_, E.B_)[1], t=tt(n, sd),
                      t_Y19_26=tt(n, sd, *E.PER['Y19_26']), t_FROM2008=tt(n, sd, *E.PER['FROM2008']))
    e = E50.dropna(); e = e[e.index >= D.IDX[D.t0]]
    ref['0050'] = dict(CAGR=E.cagr_mdd(e, E.A_, E.B_)[0], MDD=E.cagr_mdd(e, E.A_, E.B_)[1])
    jz = json.load(open(E.DATA / 'jscan_meta.json'))['meta']
    ref['jscan_zero_impact_t_median'] = {b: float(np.median([jz[f'{b}|j{j}']['t'] for j in range(13)])) for b in ('AB', 'B10')}
    return R, J, ref, navs_out

def verdict(R, J):
    out = {}
    tbl = {}
    for st in ('AB', 'B10'):
        for ti in TIERS:
            for ve in VERS:
                k = f'{st}|{ti}|{ve}'
                g = lambda v: R.get(f'{st}|{v}|{ti}|{ve}', {})
                tbl[k] = dict(CAGR=g('M3V_point').get('CAGR'), t=g('M3V_point').get('t'), t_p10=g('M3V_p10').get('t'), t_p90=g('M3V_p90').get('t'),
                              CAGR_p10=g('M3V_p10').get('CAGR'), CAGR_p90=g('M3V_p90').get('CAGR'),
                              t_stress=g('STRESS').get('t'), CAGR_stress=g('STRESS').get('CAGR'),
                              j_t_median=J.get(k, {}).get('t_median'), j_CAGR_median=J.get(k, {}).get('CAGR_median'))
    out['table'] = tbl
    e50 = None
    sweet = {}
    for ti in TIERS:
        for ve in VERS:
            b10p90 = tbl[f'B10|{ti}|{ve}']['t_p90']; abj = tbl[f'AB|{ti}|{ve}']['j_t_median']
            sweet[f'{ti}|{ve}'] = dict(B10_t_at_p90=b10p90, AB_j_median_t=abj, ok=bool(b10p90 is not None and abj is not None and b10p90 >= 2.0 and abj >= 1.8))
    out['sweet_spot_rule'] = sweet
    out['recommended_capital_range'] = [k for k, v in sweet.items() if v['ok']] or '沒有被驗證的甜蜜區'
    flips = {}
    for k, v in tbl.items():
        a, b = v['t_p10'], v['t_p90']
        flips[k] = dict(t_p10=a, t_p90=b, crosses_2=bool((a >= 2.0) != (b >= 2.0)) if a is not None and b is not None else None)
    out['kappa_p10_p90_flip'] = flips
    # P5：最佳級距（FIX、point）落在 100～300 萬的 seed 比例
    best = []
    for s in range(40):
        tv = {}
        for ti in TIERS:
            c = f'AB|M3V_point|{ti}|FIX|s{s:02d}'
            if c in NAV: tv[ti] = tt(NAV[c].dropna(), D.sig_days)
        if tv: best.append(max(tv, key=tv.get))
    out['P5_best_tier_share'] = {ti: best.count(ti) / len(best) for ti in TIERS} if best else None
    return out

if __name__ == '__main__':
    R, J, ref, navs_out = run()
    V = verdict(R, J)
    RESULTS = dict(cells=R, jscan=J, reference=ref, verdict=V)
    pd.DataFrame(navs_out).to_parquet((E.DATA if not DRY else SRC) / 'cap_avg_navs.parquet')
    if DRY:
        json.dump(dict(n_cells=len(R), keys=list(R)[:5]), open(SRC / 'analyze_dry.json', 'w'))
        print('dry ok', len(R), len(J))
    else:
        res = json.loads(RES_F.read_text(encoding='utf-8'))
        res['RESULTS'] = RESULTS
        json.dump(res, open(RES_F, 'w'), ensure_ascii=False, indent=1, default=float)
        print(json.dumps(V, ensure_ascii=False, indent=1, default=float)[:6000])

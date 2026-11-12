#!/usr/bin/env python3
"""凍結後正式分析（一次跑）：斷言 PREREG 與 PREREG_FREEZE 存在、sha 相符、程式與輸入 sha 相符。→ results.json['RESULTS']、data/*.csv.gz／parquet。
python3 -I -B analyze.py"""
import sys, json, pickle, hashlib, time, math, itertools
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R and 'PREREG_FREEZE' in R, '凍結前不得計算正式統計量'
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert hashlib.sha256(json.dumps(R['PREREG_FREEZE'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_FREEZE_sha256']
for f, h in R['PREREG_FREEZE']['files_sha256'].items(): assert hashlib.sha256((HERE / f).read_bytes()).hexdigest() == h, f
assert 'RESULTS' not in R, '只跑一次'
sys.path.insert(0, str(HERE)); import csr_lib as L; import diag as DG; sys.path.pop(0)
FZ = R['PREREG_FREEZE']; RK = FZ['refund_main']; RK2 = 'cash' if RK == 'prop' else 'prop'
SCR = L.SCR; t0 = time.time(); S = DG.S; D = DG.D
OUT = {'utc_start': pd.Timestamp.now('UTC').isoformat(), 'seeds20_sha256': L.sha256f(L.SCR / 'navs_seeds20.parquet')}
def log(*a): print(round(time.time() - t0), *a, flush=True)
M = pickle.load(open(SCR / 'center_meta.pkl', 'rb'))
NC = pd.read_parquet(SCR / 'navs_center.parquet'); NC.index = DG.IDX
NR = pd.read_parquet(SCR / 'navs_refund.parquet'); NR.index = DG.IDX
NK = pd.read_parquet(SCR / 'navs_coal.parquet'); NK.index = DG.IDX
NG = pd.read_parquet(SCR / 'navs_grid.parquet'); NG.index = DG.IDX
NT = pd.read_parquet(SCR / 'navs_tick.parquet'); NT.index = DG.IDX
N20 = pd.read_parquet(SCR / 'navs_seeds20.parquet'); N20.index = DG.IDX
GM = pickle.load(open(SCR / 'grid_meta.pkl', 'rb')); TM = pickle.load(open(SCR / 'tick_meta.pkl', 'rb')); RM = pickle.load(open(SCR / 'refund_meta.pkl', 'rb'))
OFF = json.loads((L.AG / 'integrate_o19/results.json').read_text(encoding='utf-8'))['RESULTS_V7X']['table']
TFC = json.loads((L.TFC / 'results.json').read_text(encoding='utf-8'))['RESULTS']
F100 = pd.read_parquet(SCR / 'fills_center_normal_100萬_std.parquet'); F100['variant'] = F100.variant.astype(str)
c0 = L.cash0(DG.E, '100萬')
KEYS = ['CAGR_Y19on_med', 'exc_Y19on_med', 'exc_med', 'exc_p10', 't_0050_med', 'maxDD_med', 'CAGR_med', 'exc_POSTODD_med', 'exc_PREODD_med', 'CAGR_POSTODD_med', 'CAGR_PREODD_med']
def nav_of(src, key): return src[key].dropna()

# ======== A. 正式數字重現 ========
A = {}
for c in L.CELLS:
    ck = L.CELLKEY[c]
    for v in L.VARS:
        st = S.jsumm([S.stats(nav_of(N20, f'1.0|1.0|{v}|{j}|{ck}'), 'AB_V7_B10', j) for j in range(13)])
        o = OFF[f'strict|{v}|{ck}']
        A[f'{v}|{ck}'] = {k: dict(mine=st[k], official=o[k], diff=st[k] - o[k]) for k in ('CAGR_Y19on_med', 'exc_Y19on_med', 'exc_med', 'exc_p10', 't_0050_med')}
OUT['A_reproduce_official_maxabs'] = float(max(abs(x['diff']) for d in A.values() for x in d.values()))
OUT['A_reproduce_official'] = A
log('A', OUT['A_reproduce_official_maxabs'])

# ======== B. 會計拖累與 4.4pp 對帳 ========
B = {}
for v in L.VARS:
    ad = []; b1 = []
    for j in range(13):
        key = f'{v}|{j}|normal|100萬|std'
        f = DG.prep_fills(F100, M['rebs'][key], v, j)
        ad.append(DG.acct_drag(f, M['rebs'][key], 'Y19on', j)); b1.append(DG.b1_drag(f, NC[key], M['daycost'][key], 'Y19on', j, c0))
    B[f'{v}|acct_Y19on'] = {k: float(np.median([x[k] for x in ad])) for k in ad[0]}
    B[f'{v}|acct_Y19on_vs_o20'] = {k: B[f'{v}|acct_Y19on'][k2] - TFC['B_drag']['normal|100萬|Y19on'][v][k2] for k, k2 in (('cost', 'cost'), ('tax', 'tax'), ('fee', 'fee'), ('tick', 'tick_amt'), ('imp', 'imp_amt'), ('odd', 'odd_amt'))}
    B[f'{v}|b1_Y19on'] = {k: float(np.median([x[k] for x in b1])) for k in b1[0]}
    for per in FZ['bridge_periods']:
        for rk in (RK, RK2):
            BI = DG.bridge_inputs(per, v, NC, NR, NK, M['rebs'], M['daycost'], F100, c0, rk)
            q = DG.bridge_levels(BI); ch = DG.chain(q)
            ch['Tfirst_med'] = q['Tfirst_med']; ch['per_j'] = {k: [float(x) for x in q[k]] for k in ('c_acc', 'c_B1', 'ell', 'G', 'G_lin', 'gA', 'gR', 'gT', 'q3', 'q4', 'CCj')}
            ch['G_lin_med'] = float(np.median(q['G_lin'])); ch['ell_med'] = float(np.median(q['ell']))
            if per == 'Y19on' and rk == RK: ch['boot'] = DG.bridge_boot(BI, B=FZ['bridge_boot_B'] if v == 'REAL_opt' else 300)
            B[f'{v}|{per}|{rk}'] = ch
    log('B', v)
sh_chk = {v: B[f'{v}|Y19on|{RK}']['end'] - TFC['I_shapley'][f'100萬|{v}|CAGR_Y19on']['cost_class']['mean'] for v in L.VARS}
OUT['B_shapley_vs_o20'] = sh_chk
def verdict_bridge(ch):
    U = ch['residual_U']; b2 = ch['bridges_main']['B2']
    if abs(U) <= 0.3: return '對帳完成'
    return '部分對帳' if b2 >= 2.6 else '未對帳'
B['verdicts'] = {f'{v}|{rk}': verdict_bridge(B[f'{v}|Y19on|{rk}']) for v in L.VARS for rk in (RK, RK2)}
OUT['B_bridge'] = B
log('B done', B['verdicts'])

# ======== C. 敏感度 ========
def cellstats(src, pref, ck, seeds20=False):
    o = {}
    for v in L.VARS:
        navs = [nav_of(src, f'{pref}|{v}|{j}|{ck}') for j in range(13)]
        st = S.jsumm([S.stats(n, 'AB_V7_B10', j) for j, n in enumerate(navs)])
        ps = [L.pstats(S, n, j, pers=['S0612', 'S1318', 'Y19on', 'POSTODD', 'PREODD']) for j, n in enumerate(navs)]
        st['seg_exc_med'] = {p: float(np.median([x[p]['exc'] for x in ps])) for p in ('S0612', 'S1318', 'Y19on', 'POSTODD', 'PREODD')}
        st['cagr_by_j_Y19on'] = [float(S.stats(n, 'AB_V7_B10', j)['CAGR_Y19on']) for j, n in enumerate(navs)]
        o[v] = st
    w = {}
    for k in KEYS:
        vals = {v: o[v][k] for v in L.VARS}
        wv = min(vals, key=vals.get) if k != 'maxDD_med' else min(vals, key=vals.get)
        w[k] = dict(value=vals[wv], variant=wv)
    for p in ('S0612', 'S1318', 'Y19on', 'POSTODD', 'PREODD'):
        vals = {v: o[v]['seg_exc_med'][p] for v in L.VARS}; wv = min(vals, key=vals.get); w['seg_exc_' + p] = dict(value=vals[wv], variant=wv)
    return o, w
def summ_cell(SM, pref, ck, per):
    rows = [SM[f'{pref}|REAL_opt|{j}|{ck}'][per] for j in range(13) if per in SM[f'{pref}|REAL_opt|{j}|{ck}']]
    if not rows: return None
    g = lambda f: float(np.median([f(r) for r in rows]))
    return dict(fill_days_share=g(lambda r: r['n_fill_days'] / r['n_tdays']), P_only_days_share=g(lambda r: r['n_P_days'] / r['n_tdays']),
                chain_td_p50=g(lambda r: r['chain_td_p50'] or 0), chain_td_p90=g(lambda r: r['chain_td_p90'] or 0), chain_td_max=g(lambda r: r['chain_td_max'] or 0),
                chain_cd_p50=g(lambda r: r['chain_cd_p50'] or 0), chain_cd_max=g(lambda r: r['chain_cd_max'] or 0), n_chains=g(lambda r: r['n_chains']),
                odd_share_n=g(lambda r: r['n_odd'] / max(1, r['n_fills'])), odd_share_not=g(lambda r: r['not_odd'] / max(1, r['not_tot'])),
                turn2_per_yr=g(lambda r: r['turn2'] / r['yrs']), drag_cost_pp=g(lambda r: r['drag_cost'] / r['yrs'] * 100), drag_odd_pp=g(lambda r: r['drag_odd_amt'] / r['yrs'] * 100),
                drag_odd_hi_pp=g(lambda r: r['drag_odd_amt_hi'] / r['yrs'] * 100), fee_min_share=g(lambda r: r['fee_min_share'] or 0))
CS = {}
SMC = {f'1.0|1.0|1.0|{k}': x for k, x in M['summ'].items()}
for c in L.CELLS:
    ck = L.CELLKEY[c]
    for (mo, mc) in L.GRID:
        if (mo, mc) == (1.0, 1.0):
            src = NC.rename(columns=lambda x: '1.0|1.0|1.0|' + x); SM = SMC
        else: src = NG; SM = GM['summ']
        pref = f'{mo}|{mc}|1.0'
        o, w = cellstats(src, pref, ck)
        ent = dict(worst=w, by_variant={v: {k: o[v][k] for k in KEYS} for v in L.VARS},
                   summ={p: summ_cell(SM, pref, ck, p) for p in ('Y19on', 'ALL', 'PREODD', 'POSTODD')})
        g = w['CAGR_Y19on_med']['value'] / 100; tn = ent['summ']['Y19on']['turn2_per_yr']; tna = ent['summ']['ALL']['turn2_per_yr']
        ent['breakeven_slip_pp_per_side'] = dict(Y19on=w['exc_Y19on_med']['value'] / (tn * (1 + g)), ALL=w['exc_med']['value'] / (tna * (1 + g)))
        if (mo, mc) in L.CORNERS:
            o20, w20 = cellstats(N20, f'{mo}|{mc}', ck); ent['seeds20_worst'] = w20; ent['seeds20_by_variant'] = {v: {k: o20[v][k] for k in KEYS} for v in L.VARS}
        CS[f'{ck}|mo={mo}|mc={mc}'] = ent
    o, w = cellstats(NT, '1.0|1.0|0.5', ck)
    CS[f'{ck}|tick=0.5'] = dict(worst=w, summ={p: summ_cell(TM['summ'], '1.0|1.0|0.5', ck, p) for p in ('Y19on', 'ALL')})
    log('C', ck)
# 正式格（20 seeds 中心）與各格差；假設區間
HI = {}
for c in L.CELLS:
    ck = L.CELLKEY[c]; base = CS[f'{ck}|mo=1.0|mc=1.0']
    d = {}
    for (mo, mc) in L.GRID:
        e = CS[f'{ck}|mo={mo}|mc={mc}']
        d[f'mo={mo}|mc={mc}'] = {k: e['worst'][k]['value'] - base['worst'][k]['value'] for k in ('CAGR_Y19on_med', 'exc_Y19on_med', 'exc_med', 'exc_p10', 't_0050_med', 'CAGR_POSTODD_med', 'CAGR_PREODD_med')}
        if 'seeds20_worst' in e: d[f'mo={mo}|mc={mc}']['seeds20_dCAGR_Y19on'] = e['seeds20_worst']['CAGR_Y19on_med']['value'] - base['seeds20_worst']['CAGR_Y19on_med']['value']
        d[f'mo={mo}|mc={mc}']['worst_variant_Y19on'] = e['worst']['CAGR_Y19on_med']['variant']
    d['tick=0.5'] = {k: CS[f'{ck}|tick=0.5']['worst'][k]['value'] - base['worst'][k]['value'] for k in ('CAGR_Y19on_med', 'exc_Y19on_med', 'exc_med', 'exc_p10')}
    rng = [x['CAGR_Y19on_med'] for k, x in d.items() if k.startswith('mo=')]
    HI[ck] = dict(deltas=d, range_dCAGR_Y19on=[float(min(rng)), float(max(rng))], range_seed20_corners=[float(min(x['seeds20_dCAGR_Y19on'] for x in d.values() if 'seeds20_dCAGR_Y19on' in x)), float(max(x['seeds20_dCAGR_Y19on'] for x in d.values() if 'seeds20_dCAGR_Y19on' in x))])
# 抽樣區間：正式格（20 seeds，中位 j）2019 起超額的月份區塊 bootstrap
def samp_ci(ck, v):
    j_med = int(np.argsort([S.stats(nav_of(N20, f'1.0|1.0|{v}|{j}|{ck}'), 'AB_V7_B10', j)['exc_Y19on'] for j in range(13)])[6])
    n = nav_of(N20, f'1.0|1.0|{v}|{j_med}|{ck}'); s0, s1, nv = DG.perbounds(n, j_med, 'Y19on')
    g, days = DG.monthly(nv, s0, s1); g5, _ = DG.monthly(S.E50.reindex(nv.index).ffill(), s0, s1)
    rng = np.random.default_rng(5); out = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g)).astype(float); Y = w @ days.values / 365.25
        out.append((math.exp(w @ g.values / Y) - math.exp(w @ g5.reindex(g.index).values / Y)) * 100)
    return dict(j_med=j_med, se=float(np.std(out, ddof=1)), ci90=[float(np.percentile(out, 5)), float(np.percentile(out, 95))])
HI['sampling_ci_exc_Y19on_100n_opt'] = samp_ci('normal|100萬|std', 'REAL_opt')
OUT['C_sensitivity'] = CS; OUT['C_hyp_interval'] = HI
log('C done')

# ======== D. 實際成交價 edge ========
X = pd.read_parquet(SCR / 'swaps_center.parquet'); X = X[np.isfinite(X.edge_B)].copy()
X['sig'] = pd.to_datetime(X.signal); X['gap'] = X.edge_B - X.rtc_B; X['gap_fill'] = X.edge_fill - X.rtc_B; X['d_fill'] = X.edge_fill - X.edge_B
X['dlb'] = pd.cut(X.dl_td, [-1, 0, 3, 10, 10 ** 6], labels=['0', '1~3', '4~10', '>10']).astype(str)
X['kb'] = np.where(X.kinds.str.contains('D'), 'D', np.where(X.kinds == 'W', 'W', np.where(X.kinds == 'O', 'O', 'OW')))
PERS = {'Y19on': ('2019-01-01', '2100-01-01'), 'S0618': ('1900-01-01', '2018-12-31'), 'V1520': ('2015-01-01', '2020-12-31'), 'H2126': ('2021-01-01', '2100-01-01'), 'ALL': ('1900-01-01', '2100-01-01'),
        'POSTODD': ('2020-10-26', '2100-01-01'), 'PREODD': ('1900-01-01', '2020-10-25')}
def per_cut(d, p): a, b = PERS[p]; return d[(d.sig >= a) & (d.sig <= b)]
def esumm(h, full=False):
    o = dict(n=int(len(h)), edge_open_mean=float(h.edge_B.mean()), edge_fill_mean=float(h.edge_fill.mean()), edge_allin_mean=float(h.edge_allin.mean()),
             edge_open_med=float(h.edge_B.median()), edge_fill_med=float(h.edge_fill.median()), d_fill_mean=float(h.d_fill.mean()), d_fill_med=float(h.d_fill.median()),
             rtc_mean=float(h.rtc_B.mean()), rtc_med=float(h.rtc_B.median()), gap_open_mean=float(h.gap.mean()), gap_fill_mean=float(h.gap_fill.mean()),
             frac_below_open=float((h.gap < 0).mean()), frac_below_fill=float((h.gap_fill < 0).mean()))
    if full and len(h) >= 30:
        o['uniq_events'] = int(h.assign(m=h.sig.dt.to_period('M')).groupby(['code', 'm']).ngroups)
        o['d_fill_stats'] = DG.neff_mean(h.d_fill.values, h.signal.values)
        o['edge_fill_stats'] = DG.neff_mean(h.edge_fill.values, h.signal.values)
    return o
DE = {}
for cell, g in X.groupby('cell'):
    for v, h in g.groupby('variant'):
        for p in PERS:
            hp = per_cut(h, p); full = (v == 'REAL_opt')
            ent = dict(all=esumm(hp, full))
            if full:
                ent['by_deferred'] = {('deferred' if k else 'not'): esumm(z, True) for k, z in hp.groupby(hp.deferred > 0)}
                ent['by_entries_deferred'] = {('any_entry_deferred' if k else 'none'): esumm(z) for k, z in hp.groupby(hp.entries_deferred > 0)}
                ent['by_delay'] = {k: esumm(z) for k, z in hp.groupby('dlb')}
                ent['by_kind'] = {k: esumm(z) for k, z in hp.groupby('kb')}
            DE[f'{cell}|{v}|{p}'] = ent
OUT['D_edges'] = DE
log('D done')

# ======== E. 平均數口徑兩平表與成本五等分 ========
QE = {}; BE = []
for cell, g in X.groupby('cell'):
    h0 = g[g.variant == 'REAL_opt']
    for p in ('Y19on', 'S0618', 'V1520', 'H2126', 'POSTODD', 'PREODD', 'ALL'):
        hp = per_cut(h0, p)
        if len(hp) < 50: continue
        for pb, z in [('ALL', hp)] + list(hp.groupby(pd.cut(hp.price, [0, 20, 50, 100, 500, 1e9], labels=['<20', '20~50', '50~100', '100~500', '>=500']).astype(str))):
            BE.append(dict(cell=cell, period=p, price=pb, **esumm(z)))
        if cell in ('normal|100萬|std', 'normal|30萬|std', 'normal|300萬|std', 'normal|1000萬|std', 'stress|100萬|std') and p in ('Y19on', 'S0618', 'V1520', 'H2126'):
            o = {}
            for col in ('chat', 'rtc_B'):
                for gc in ('gap', 'gap_fill'):
                    tab, dd, st = DG.quint(hp, col, gc)
                    o[f'{col}|{gc}'] = dict(tab=tab, diff_q0_minus_q4=DG.boot(dd, st, B=2000), n_eff_overall=DG.neff_mean(dd[gc].values, dd.signal.values, B=500)['n_eff'])
            o['spearman_chat_rtc'] = float(hp[['chat', 'rtc_B']].corr(method='spearman').iloc[0, 1])
            QE[f'{cell}|{p}'] = o
BEdf = pd.DataFrame(BE); BEdf.to_csv(HERE / 'data/breakeven_mean_table.csv.gz', index=False, compression='gzip', float_format='%.6g')
OUT['E_quintiles'] = QE
# 兩欄分母（會計算術 vs 年化幾何），各資金級距
TWO = {}
for c in L.CELLS:
    ck = L.CELLKEY[c]
    Fc = F100 if ck == 'normal|100萬|std' else pd.read_parquet(SCR / f'fills_center_{ck.replace("|", "_")}.parquet').assign(variant=lambda d: d.variant.astype(str))
    cc = L.cash0(DG.E, c[1])
    for v in L.VARS:
        acc = []; b1 = []; ell = []; gA = []
        for j in range(13):
            key = f'{v}|{j}|{ck}'
            f = DG.prep_fills(Fc, M['rebs'][key], v, j)
            acc.append(DG.acct_drag(f, M['rebs'][key], 'Y19on', j)['cost']); b1.append(DG.b1_drag(f, NC[key], M['daycost'][key], 'Y19on', j, cc)['B1'])
            el, _ = DG.logdrag(NC[key], M['daycost'][key], 'Y19on', j, cc); ell.append(el); gA.append(DG.cagr_per(NC[key], j, 'Y19on'))
        geo = [(1 + g_ / 100) * (math.exp(e_) - 1) * 100 for g_, e_ in zip(gA, ell)]
        TWO[f'{ck}|{v}'] = dict(acct_arith_o20=float(np.median(acc)), arith_B1=float(np.median(b1)), log_drag=float(np.median(ell)) * 100, geo_annualized=float(np.median(geo)),
                               formula='幾何年化拖累＝(1＋g_net)(e^ℓ−1)＝(1＋g_gross)(1−e^{−ℓ})，ℓ＝Σ log(1＋C_d/E_d)/Y ≈ 算術 c；近似 (1＋g)·c')
OUT['E_two_column_breakeven'] = TWO
log('E done')

# ======== F. 預測與判定 ========
P = {}
bo = B[f'REAL_opt|Y19on|{RK}']
P['P1'] = dict(B2=bo['bridges_main']['B2'], residual_U=bo['residual_U'], total=bo['total'], pass_B2=bo['bridges_main']['B2'] >= 2.6, pass_resid=abs(bo['residual_U']) <= 0.3,
               verdict=B['verdicts'][f'REAL_opt|{RK}'], all_variants={v: B['verdicts'][f'{v}|{RK}'] for v in L.VARS})
d = lambda ck, mo, mc, k='CAGR_Y19on_med': HI[ck]['deltas'][f'mo={mo}|mc={mc}'][k]
P['P2'] = dict(d30_mo2_Y19=d('normal|30萬|std', 2.0, 1.0), d30_mo2_POSTODD=d('normal|30萬|std', 2.0, 1.0, 'CAGR_POSTODD_med'), d100_mo2=d('normal|100萬|std', 2.0, 1.0), d1000_mo2=d('normal|1000萬|std', 2.0, 1.0),
               d30_mo05_Y19=d('normal|30萬|std', 0.5, 1.0), d100_mo05=d('normal|100萬|std', 0.5, 1.0), d1000_mo05=d('normal|1000萬|std', 0.5, 1.0),
               odd_drag_hi_30_POSTODD=dict(center=CS['normal|30萬|std|mo=1.0|mc=1.0']['summ']['POSTODD']['drag_odd_hi_pp'], mo2=CS['normal|30萬|std|mo=2.0|mc=1.0']['summ']['POSTODD']['drag_odd_hi_pp'],
                                           total_center=CS['normal|30萬|std|mo=1.0|mc=1.0']['summ']['POSTODD']['drag_odd_pp'], total_mo2=CS['normal|30萬|std|mo=2.0|mc=1.0']['summ']['POSTODD']['drag_odd_pp']))
P['P2']['pass'] = bool(P['P2']['d30_mo2_POSTODD'] <= -1.0 and P['P2']['d100_mo2'] <= -0.5 and abs(P['P2']['d1000_mo2']) < 0.3)
P['P3'] = dict(d1000_mc05=d('normal|1000萬|std', 1.0, 0.5), d30_mc05=d('normal|30萬|std', 1.0, 0.5), d100_mc05=d('normal|100萬|std', 1.0, 0.5), d300_mc05=d('normal|300萬|std', 1.0, 0.5),
               fill_days_1000=dict(center=CS['normal|1000萬|std|mo=1.0|mc=1.0']['summ']['Y19on']['fill_days_share'], mc05=CS['normal|1000萬|std|mo=1.0|mc=0.5']['summ']['Y19on']['fill_days_share']),
               chain_p50_1000=dict(center=CS['normal|1000萬|std|mo=1.0|mc=1.0']['summ']['Y19on']['chain_td_p50'], mc05=CS['normal|1000萬|std|mo=1.0|mc=0.5']['summ']['Y19on']['chain_td_p50']))
P['P3']['pass'] = bool(P['P3']['d1000_mc05'] <= -1.0 and abs(P['P3']['d30_mc05']) < 0.3 and abs(P['P3']['d1000_mc05']) == max(abs(P['P3'][k]) for k in ('d1000_mc05', 'd30_mc05', 'd100_mc05', 'd300_mc05')))
e19 = DE['normal|100萬|std|REAL_opt|Y19on']; e18 = DE['normal|100萬|std|REAL_opt|S0618']
P['P4'] = dict(Y19on=dict(open=e19['all']['edge_open_mean'], fill=e19['all']['edge_fill_mean'], diff=e19['all']['d_fill_mean'], diff_deferred=e19['by_deferred'].get('deferred', {}).get('d_fill_mean'),
                          mde80=e19['all']['d_fill_stats']['mde80']),
               S0618=dict(diff=e18['all']['d_fill_mean'], diff_deferred=e18['by_deferred'].get('deferred', {}).get('d_fill_mean')))
P['P4']['pass'] = bool(P['P4']['Y19on']['diff'] <= -0.001 and (P['P4']['Y19on']['diff_deferred'] or 0) <= -0.005 and np.sign(P['P4']['S0618']['diff']) == np.sign(P['P4']['Y19on']['diff']))
q19 = QE['normal|100萬|std|Y19on']['chat|gap_fill']['diff_q0_minus_q4']; q18 = QE['normal|100萬|std|S0618']['chat|gap_fill']['diff_q0_minus_q4']
P['P5'] = dict(chat_fill_Y19on=q19, chat_fill_S0618=q18, rtc_fill_Y19on=QE['normal|100萬|std|Y19on']['rtc_B|gap_fill']['diff_q0_minus_q4']['obs'], rtc_open_Y19on=QE['normal|100萬|std|Y19on']['rtc_B|gap']['diff_q0_minus_q4']['obs'],
               pass_ge1=bool(q19['obs'] >= 0.01), weak_lt05=bool(q19['obs'] < 0.005))
G = HI['normal|100萬|std']
P['P6'] = dict(range=G['range_dCAGR_Y19on'], within_1p5=bool(max(abs(x) for x in G['range_dCAGR_Y19on']) <= 1.5),
               any_G_flip=False, note='G1（j p10>0、壓力>0）、G2（t≥3）、G7（300 萬、1,000 萬>0）在任一格都要翻號才算改變；見 C_sensitivity 逐格 worst。')
flip = []
for c in L.CELLS:
    ck = L.CELLKEY[c]
    for (mo, mc) in L.GRID:
        w = CS[f'{ck}|mo={mo}|mc={mc}']['worst']
        if w['exc_p10']['value'] > 0 or w['t_0050_med']['value'] >= 3.0 or (ck in ('normal|300萬|std', 'normal|1000萬|std', 'stress|100萬|std') and w['exc_med']['value'] > 0 and w['exc_p10']['value'] > 0):
            flip.append(f'{ck}|{mo}|{mc}')
P['P6']['any_G_flip'] = bool(flip); P['P6']['flips'] = flip
P['P6']['pass'] = bool(P['P6']['within_1p5'] and not flip)
sens = any(max(abs(x) for x in HI[L.CELLKEY[c]]['range_dCAGR_Y19on']) > 1.5 for c in L.CELLS if c[0] == 'normal' and c[2] == 'std') or bool(flip)
OUT['F_predictions'] = P
OUT['F_verdicts'] = dict(reconciliation=B['verdicts'][f'REAL_opt|{RK}'], cost_assumption='敏感' if sens else '穩健',
                         official_numbers_changed=False, G_table_changed=False, investable=False)
log('F', OUT['F_verdicts'])

# ======== G. 匯出關鍵中間檔 ========
rows = []
for v in L.VARS:
    for per in FZ['bridge_periods']:
        for rk in (RK, RK2):
            ch = B[f'{v}|{per}|{rk}']
            for j in range(13): rows.append(dict(variant=v, period=per, refund=rk, j=j, **{k: ch['per_j'][k][j] for k in ch['per_j']}))
pd.DataFrame(rows).to_csv(HERE / 'data/bridge_per_j.csv.gz', index=False, compression='gzip', float_format='%.8g')
cr = []
for c_ in DG.COALS:
    for v in L.VARS:
        for j in range(13):
            n = NK[f'{c_}|{v}|{j}']; ps = L.pstats(S, n.dropna(), j, pers=FZ['bridge_periods'])
            cr.append(dict(coalition=c_, variant=v, j=j, **{f'CAGR_{p}': ps[p]['cagr'] for p in FZ['bridge_periods']}))
pd.DataFrame(cr).to_csv(HERE / 'data/coalitions_cagr_100n.csv.gz', index=False, compression='gzip', float_format='%.10g')
keep = {}
for v in L.VARS:
    for j in range(13):
        keep[f'A|{v}|{j}'] = NC[f'{v}|{j}|normal|100萬|std']; keep[f'T|{v}|{j}'] = NK[f'T|{v}|{j}']
        for rk in ('cash', 'prop'): keep[f'R_{rk}|{v}|{j}'] = NR[f'{rk}|{v}|{j}']
pd.DataFrame(keep).to_parquet(HERE / 'data/navs_bridge_100n_seed0.parquet', compression='zstd')
pd.DataFrame({k: N20[k] for k in N20.columns}).astype('float32').to_parquet(HERE / 'data/navs_seeds20_corners_center.parquet', compression='zstd')
pd.DataFrame({k: NG[k] for k in NG.columns}).astype('float32').to_parquet(HERE / 'data/navs_grid_seed0.parquet', compression='zstd')
dcrows = [dict(key=k, t=int(t), date=str(DG.IDX[t].date()), cost=c) for k, dd in M['daycost'].items() if '|100萬|' in k or '|30萬|std' in k for t, c in dd.items()]
pd.DataFrame(dcrows).to_csv(HERE / 'data/daycost_center_100n_30n.csv.gz', index=False, compression='gzip', float_format='%.10g')
for ck in ('normal|100萬|std', 'normal|30萬|std'):
    Fx = pd.read_parquet(SCR / f'fills_center_{ck.replace("|", "_")}.parquet'); Fx['code'] = [DG.COLS[x] for x in Fx.j]; Fx['date'] = DG.IDX[Fx.t.values].strftime('%Y-%m-%d')
    Fx.drop(columns=['j']).to_csv(HERE / f'data/tradelog_center_{ck.split("|")[1]}_seed0.csv.gz', index=False, compression='gzip', float_format='%.10g')
rb = [dict(key=k, k=i, t=r[0], date=str(DG.IDX[r[0]].date()), picks=','.join(DG.COLS[x] for x in r[1]), hold_before=';'.join(f'{DG.COLS[a]}:{b:.6f}' for a, b in r[2].items()), avail=r[4])
      for k, rr in M['rebs'].items() if '|100萬|std' in k or '|30萬|std' in k for i, r in enumerate(rr)]
pd.DataFrame(rb).to_csv(HERE / 'data/rebs_center_100n_30n.csv.gz', index=False, compression='gzip', float_format='%.10g')
OUT['G_files'] = {p.name: dict(bytes=p.stat().st_size, sha256=L.sha256f(p)) for p in sorted((HERE / 'data').iterdir())}
OUT['utc_end'] = pd.Timestamp.now('UTC').isoformat()
R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS' not in R
R['RESULTS'] = OUT; RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
log('done')

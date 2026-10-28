#!/usr/bin/env python3
"""integrate_o6 回測：照 results.json 的 PREREG，把 o6 通過審查（待觀察）的 o_v6_price_panel_official（V6 價格面板）和主策略（排除金融 n10）整合。
引擎全部只讀 import：o_v6_price_panel_official/v6env.py（把 independent_audit/ia_core 的面板路徑換成 V6 data/）→ o_v4_revenue_patch_official/v4lib.py
→ main_v2/v2_core、o_disposal_rejudge/drj_core、o_infra_v3_rebuild/o3_core；v6stats.py（窗、年化、HOLD 日曆歸屬）；o_main_spa_breaks/spa_lib.py。
成本、成交、漲停遞補、停牌凍結、併購出清、AB 抽樣、SPA bootstrap 全部沿用；本檔只新寫袖子混合（與 integrate_o4／o5 相同函式）、保留期預測分布（照抄 V6 s5 演算法並斷言重現）與統計彙整。
用法：python3 -I -B run.py（正式）；python3 -I -B run.py --smoke <輸出檔>（2 seeds、B=200，只測管線，不寫 results.json、不印數字）。"""
import sys, json, time, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
DATA = HERE / 'data'
V6D = AG / 'o_v6_price_panel_official'
SPD = AG / 'o_main_spa_breaks'
SMOKE = '--smoke' in sys.argv
import numpy as np, pandas as pd
from multiprocessing import Pool

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
RJ = HERE / 'results.json'
if not SMOKE:
    RES = json.loads(RJ.read_text(encoding='utf-8'))
    assert hashlib.sha256(json.dumps(RES['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256'], 'PREREG 被改過'
    assert 'RESULTS' not in RES, 'RESULTS 已存在：只跑一次'
    for f, h in RES['PREREG']['sha256'].items():
        assert sha(f) == h, f'{f} 與 PREREG 登記的 sha256 不符'
COPIES = {'navs_v6.parquet': V6D / 'navs_v6.parquet', 'seed_navs_v6.parquet': V6D / 'seed_navs_v6.parquet', 'navs_jscan_v6.parquet': V6D / 'navs_jscan_v6.parquet',
          'plan_V6.csv': V6D / 'plan_V6.csv', 'navs_v4.parquet': AG / 'o_v4_revenue_patch_official/navs_v4.parquet', 'plan_V4.csv': AG / 'o_v4_revenue_patch_official/plan_V4.csv',
          'integrate_o4_navs.parquet': AG / 'integrate_o4/navs.parquet', 'family_excess.parquet': SPD / 'data/family_excess.parquet',
          'family_members.csv': SPD / 'data/family_members.csv', 'spa_bench.parquet': SPD / 'data/bench.parquet',
          'exec_navs_abavg.parquet': AG / 'o_exec_capacity_tiers/navs_abavg.parquet', 'exec_bench.parquet': AG / 'o_exec_capacity_tiers/data/bench.parquet',
          'etf_0050.csv': Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'), 'taiex_twii.csv': Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')}
for k, f in COPIES.items(): assert sha(DATA / k) == sha(f), k
V6R = json.loads((V6D / 'results.json').read_text(encoding='utf-8'))
assert V6R['PREREG_sha256'] == '609f42d0064130d492cb931d267df15b48e104607061f260ce51266478e282a6'
fz = json.loads((V6D / 'data_freeze_v6.json').read_text(encoding='utf-8'))
for k, v in fz['v6_panel'].items(): assert sha(v['path']) == v['sha256'], k

sys.path.insert(0, str(V6D))
import v6env, v6stats as ST
sys.path.pop(0)
L = v6env.load('v6')
K, V = L.K, L.V
sys.path.insert(0, str(SPD))
import spa_lib as SP
sys.path.pop(0)
T0_ = time.time()
def log(*a):
    if SMOKE: print(f'[{time.time() - T0_:6.0f}s]', *[x for x in a if isinstance(x, str)], flush=True)
    else: print(f'[{time.time() - T0_:6.0f}s]', *a, flush=True)

fz4 = L.check_freeze()
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
SEEDS = list(range(2)) if SMOKE else list(range(40))
SLIPS = [0.0010, 0.0020] if SMOKE else [round(0.0010 + 0.0002 * i, 4) for i in range(16)]
SPEC = {'AB_V6': dict(), 'AB_V6_B10': dict(gap=GAP, pfix=1.0)}
COSTS = {'AB_V6': ['normal', 'stress'] + [('slip', s) for s in SLIPS], 'AB_V6_B10': ['normal', 'stress']}
S0 = L.V4(YFX, Y5F, DMT)
t0, t1, sd = S0.bounds()
pl0 = S0.plan(10, 'official')
EXEC = [p[0] for p in pl0]
VAL = {}

def job(a):
    spec, seed = a
    S = L.V4(YFX, Y5F, DMT, unavail=V.make_unavail('mc', seed), **SPEC[spec]); pl = S.plan(10, 'official')
    assert S.bounds()[:2] == (t0, t1)
    out = {str(c): L.sim_plan(pl, t0, t1, c, XR)[0] for c in COSTS[spec]}
    out['EW'] = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)[0]
    return spec, seed, out, [p[0] for p in pl]

# ---------------- AB_V6／AB_V6_B10：40 seeds 重跑（4 行程平行，fork 後共用已載入的 V6 面板）----------------
if __name__ == '__main__':
    with Pool(4) as P: OUT = P.map(job, [(s, sd_) for s in SPEC for sd_ in SEEDS], chunksize=1)
    by = {s: sorted([o for o in OUT if o[0] == s], key=lambda x: x[1]) for s in SPEC}
    for s in SPEC:
        for o in by[s]: assert o[3] == EXEC, (s, o[1])
    def mean_nav(lst): return (1 + pd.concat([x.pct_change() for x in lst], axis=1).mean(axis=1).fillna(0)).cumprod()
    BASEN = {}
    for s in SPEC:
        for c in COSTS[s] + ['EW']: BASEN[(s, str(c))] = mean_nav([o[2][str(c)] for o in by[s]])
    SEEDN = {(s, sd_): by[s][k][2]['normal'] for s in SPEC for k, sd_ in enumerate(SEEDS)}
    NV6 = pd.read_parquet(DATA / 'navs_v6.parquet'); NV4 = pd.read_parquet(DATA / 'navs_v4.parquet'); SN6 = pd.read_parquet(DATA / 'seed_navs_v6.parquet')
    if not SMOKE:
        for s in SPEC:
            for c in ('normal', 'stress', 'EW'):
                a = BASEN[(s, c)]; VAL[f'{s}_{c}_vs_navs_v6_maxrel'] = float((a / NV6[f'{s}|{c}'].reindex(a.index) - 1).abs().max())
                assert VAL[f'{s}_{c}_vs_navs_v6_maxrel'] <= 1e-9, VAL
            VAL[f'{s}_seed_navs_vs_seed_navs_v6_maxrel'] = float(max((SEEDN[(s, k)] / SN6[f'{s}|normal|s{k}'].reindex(SEEDN[(s, k)].index) - 1).abs().max() for k in SEEDS))
            assert VAL[f'{s}_seed_navs_vs_seed_navs_v6_maxrel'] <= 1e-9, VAL
    IX = K.IDX[t0:t1 + 1]
    for k_ in BASEN: assert BASEN[k_].index.equals(IX), k_
    sd6 = [pd.Timestamp(x) for x in pd.read_csv(DATA / 'plan_V6.csv')['signal']] + [pd.Timestamp(ST.B0)]
    assert [pd.Timestamp(x) for x in sd] == sd6 or [pd.Timestamp(x) for x in sd][:-1] == sd6[:-1], 'V6 格點與 plan_V6 不符'
    G6 = pd.DatetimeIndex(sd6)
    G4 = pd.DatetimeIndex([pd.Timestamp(x) for x in pd.read_csv(DATA / 'plan_V4.csv')['signal']] + [pd.Timestamp(ST.B0)])
    VAL['n_windows_V6_grid'] = len(G6) - 1; VAL['n_windows_V4_grid'] = len(G4) - 1
    VAL['V6_grid_days_not_in_V4_grid'] = [str(d.date()) for d in G6 if d not in set(G4)]
    log('AB rerun done', VAL)

    # ---------------- 淨值跳日檢查（o5 教訓 1）----------------
    tx = pd.read_csv(DATA / 'taiex_twii.csv', parse_dates=['date'])['date']; TX = pd.DatetimeIndex(sorted(set(tx)))
    TXw = TX[(TX >= IX[0]) & (TX <= IX[-1])]
    r6 = BASEN[('AB_V6', 'normal')].pct_change()
    JUMP = {'taiex_days_missing_in_V6_index': int((~TXw.isin(IX)).sum()), 'taiex_days_missing_2024': [str(d.date()) for d in TXw[~TXw.isin(IX)] if d.year == 2024],
            'V6_index_days_not_taiex': int((~IX.isin(TX)).sum()),
            'AB_V6_abs_daily_ret_gt_10pct': {str(d.date()): float(v) for d, v in r6[r6.abs() > 0.10].items()},
            'AB_V6_max_abs_daily_ret_2024_01_02_to_02_15': float(r6['2024-01-02':'2024-02-15'].abs().max()),
            'AB_V6_ret_2024_01_02': float(r6.loc['2024-01-02']), 'AB_V6_ret_2024_02_15': float(r6.loc['2024-02-15'])}
    i4 = NV4['AB_V4|normal'].dropna(); JUMP['AB_V4_ret_2024_02_15_(跨 2023-12-29→2024-02-15 缺口一日)'] = float(i4.pct_change().loc['2024-02-15'])
    JUMP['AB_V6_vs_AB_V4_identical_to_2023_12_29_maxrel'] = float((BASEN[('AB_V6', 'normal')][:'2023-12-29'] / i4[:'2023-12-29'] - 1).abs().max())
    VAL['jump_check'] = JUMP
    if not SMOKE: assert JUMP['AB_V6_vs_AB_V4_identical_to_2023_12_29_maxrel'] <= 1e-12

    # ---------------- 袖子混合（同 integrate_o4／o5）----------------
    def sleeve(navA, rB, w, cost, kindB, ix, exi):
        """navA：主策略腿淨值（已扣自身成本）；rB：另一腿日報酬（對齊 ix）；w：主策略腿固定比例；cost＝(買, 賣) 股票費率。
        kindB：'etf'（賣方＝買方＋0.1%）、'stock'（同股票費率）。主策略成交日 e 收盤調回比例，移轉金額扣成本。"""
        buy, sell = cost
        bB, sB = {'etf': (buy, buy + 0.001), 'stock': (buy, sell)}[kindB]
        ra = navA.reindex(ix).pct_change().fillna(0).to_numpy()
        a, b = w, 1 - w; out = np.empty(len(ix))
        for i in range(len(ix)):
            if i > 0: a *= 1 + ra[i]; b *= 1 + rB[i]
            if i in exi and i > 0:
                tot = a + b; ta = w * tot
                if a < ta: m = ta - a; c = m * (sB + buy)
                else: m = a - ta; c = m * (sell + bB)
                tot -= c; a, b = w * tot, (1 - w) * tot
            out[i] = a + b
        return pd.Series(out, index=ix)
    E50raw = pd.read_csv(DATA / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    def e50_on(ix): return E50raw.reindex(ix.union(E50raw.index)).ffill().reindex(ix)
    # V4 驗證：同一函式在 V4 日曆上重算 integrate_o4 的 L_E50_20/30/50、L_2330_30，逐日相對差 ≤ 1e-9
    PV4 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
    c4 = pd.read_parquet(PV4 / 'close_raw.parquet', columns=['2330']); rt4 = pd.read_parquet(PV4 / 'ret.parquet', columns=['2330'])
    IX4 = NV4['AB_V4|normal'].dropna().index
    g4 = pd.Series(np.where(np.isfinite(c4['2330'].to_numpy()), 1 + np.nan_to_num(rt4['2330'].to_numpy(), nan=0.0), 1.0), index=c4.index).reindex(IX4)
    R2330_4 = (g4 - 1).to_numpy().copy(); R2330_4[0] = 0.0
    RE50_4 = e50_on(IX4).pct_change().fillna(0).to_numpy()
    P4 = pd.read_csv(DATA / 'plan_V4.csv'); EXI4 = set(IX4.get_loc(pd.Timestamp(d)) for d in P4['exec'] if pd.Timestamp(d) in IX4)
    NI4 = pd.read_parquet(DATA / 'integrate_o4_navs.parquet')
    for nm_, w_, kind_, rB_ in (('L_E50_20', 0.8, 'etf', RE50_4), ('L_E50_30', 0.7, 'etf', RE50_4), ('L_E50_50', 0.5, 'etf', RE50_4), ('L_2330_30', 0.7, 'stock', R2330_4)):
        for c in ('normal', 'stress'):
            a = sleeve(NV4[f'AB_V4|{c}'].dropna(), rB_, w_, L.cost_rates(c), kind_, IX4, EXI4)
            VAL[f'sleeve_V4_{nm_}_{c}_vs_integrate_o4_maxrel'] = float((a / NI4[f'{nm_}|{c}'].reindex(IX4) - 1).abs().max())
            assert VAL[f'sleeve_V4_{nm_}_{c}_vs_integrate_o4_maxrel'] <= 1e-9, VAL
    log('sleeve V4 validation done', VAL)

    EXI = set(i - t0 for i in EXEC if t0 <= i <= t1)
    R_E50 = e50_on(IX).pct_change().fillna(0).to_numpy()
    R_2330 = (K.GROW[t0:t1 + 1, K.CI['2330']] - 1.0).copy(); R_2330[0] = 0.0
    MIX = {'M_E50_20': (0.8, 'etf'), 'M_E50_30': (0.7, 'etf'), 'M_E50_50': (0.5, 'etf'), 'M_2330_30': (0.7, 'stock')}
    LEG = {'etf': R_E50, 'stock': R_2330}
    ORDER = ['AB_V6'] + list(MIX)
    BASEK = {'AB': 'AB_V6', 'B10': 'AB_V6_B10'}
    def build(nm, base, c):
        nb = BASEN[(BASEK[base], str(c))]
        if nm == 'AB_V6': return nb
        w, kd = MIX[nm]; return sleeve(nb, LEG[kd], w, L.cost_rates(c), kd, IX, EXI)
    NAV = {f'{nm}|{b}|{c}': build(nm, b, c) for nm in ORDER for b in ('AB', 'B10') for c in ('normal', 'stress')}
    SLIPNAV = {(nm, s): build(nm, 'AB', ('slip', s)) for nm in ORDER for s in SLIPS}
    SEEDNAV = {(nm, s): (SEEDN[('AB_V6', s)] if nm == 'AB_V6' else sleeve(SEEDN[('AB_V6', s)], LEG[MIX[nm][1]], MIX[nm][0], L.cost_rates('normal'), MIX[nm][1], IX, EXI))
               for nm in ORDER for s in SEEDS}
    EW = {'AB': BASEN[('AB_V6', 'EW')], 'B10': BASEN[('AB_V6_B10', 'EW')]}
    log('combos built')

    # ---------------- 統計 ----------------
    E50 = K.E50
    PER = dict(L.PER)
    PK = ('S1', 'S2', 'S3', 'SEL', 'VAL', 'HOLD', 'ALL')
    CAL_A, CAL_B = pd.Period('2006-05', 'M'), pd.Period('2026-09', 'M')
    def tstat(x): return ST.tstat(x)
    def cal_ret(nav, drop=()):
        s = nav.dropna(); me = s.groupby(s.index.to_period('M')).last()
        me = me[~me.index.isin([pd.Period(x, 'M') for x in drop])]
        return me.pct_change()
    E50M = cal_ret(E50raw)
    def selp(r, key):
        a, b = PER[key]; pa = max(pd.Period(a, 'M'), CAL_A); pb = min(pd.Period(b, 'M'), CAL_B)
        return r[(r.index >= pa) & (r.index <= pb)].dropna()
    def cal_ex(nav, key='ALL'): return selp(cal_ret(nav) - E50M.reindex(cal_ret(nav).index), key)
    def pd_stats(d):
        d = d.dropna(); nz = d[d.abs() > 1e-12]
        o = dict(mean_pp=float(d.mean() * 100), t=tstat(d), n=int(len(d)), n_nonzero=int(len(nz)),
                 MDE80_pp=float(2.8 * d.std(ddof=1) / math.sqrt(len(d)) * 100) if len(d) > 2 else None)
        if len(nz) >= 3:
            big = d.abs().idxmax(); o['drop_maxabs_t'] = tstat(d.drop(big)); o['maxabs_period'] = str(big)[:10]; o['maxabs_pp'] = float(d[big] * 100)
        return o
    def wsel(s, a, b): return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))].dropna()
    def paired_cal(a, b, key, drop=()): return pd_stats(selp(cal_ret(a, drop) - cal_ret(b, drop), key))
    def paired_win(a, b, a_, b_, g=G6):
        return pd_stats(wsel(ST.win(a, g).r - ST.win(b, g).r, a_, b_))
    def corr_win(a, b, a_, b_, excess=False):
        ra, rb = wsel(ST.win(a, G6).r, a_, b_), wsel(ST.win(b, G6).r, a_, b_)
        if excess:
            e = wsel(ST.win(E50raw, G6).r, a_, b_); ra, rb = ra - e, rb - e
        z = pd.concat([ra, rb], axis=1).dropna(); return float(z.iloc[:, 0].corr(z.iloc[:, 1]))
    def corr_cal(a, b, key):
        z = pd.concat([selp(cal_ret(a), key), selp(cal_ret(b), key)], axis=1).dropna(); return float(z.iloc[:, 0].corr(z.iloc[:, 1]))
    def xs(nav, a=PER['ALL'][0], b=PER['ALL'][1]): return L.excess(nav, E50.reindex(nav.index), sd, a, b)
    def hold_cal(nav, ew):
        x = ST.xs(nav, E50raw, G6, ST.HOLD_CAL_END, ST.B0, attr='end'); xe = ST.xs(nav, ew, G6, ST.HOLD_CAL_END, ST.B0, attr='end')
        W = ST.win(nav, G6).r.reindex(x.index); Wb = ST.win(E50raw, G6).r.reindex(x.index)
        return {'t_0050': tstat(x), 'mean_pp': float(x.mean() * 100), 'n': int(len(x)), 't_EW': tstat(xe), 'cum_log_vs_0050': float(np.log1p(W).sum() - np.log1p(Wb).sum())}
    def endpoint(nav):
        x = xs(nav); o = {'t_all': tstat(x), 'drop_last1': tstat(x.iloc[:-1])}
        xsrt = x.sort_values(ascending=False)
        for k in (1, 3, 5, 10): o[f'drop_best{k}'] = tstat(x.drop(xsrt.index[:k]))
        yr = x.groupby(x.index.year).sum(); by_ = int(yr.idxmax())
        o['best_year'] = by_; o['drop_best_year'] = tstat(x[x.index.year != by_])
        o['drop_first_year_2006'] = tstat(x[x.index.year != 2006])
        o['start_2007'] = tstat(x[x.index >= pd.Timestamp('2007-01-01')]); o['start_2008'] = tstat(x[x.index >= pd.Timestamp('2008-01-01')])
        o['to_2025_end'] = tstat(x[x.index <= pd.Timestamp('2025-12-31')])
        c = cal_ex(nav); o['cal_month_t'] = tstat(c); o['cal_drop_best1'] = tstat(c.drop(c.idxmax()))
        return o
    def yearly(nav):
        r = nav.pct_change(); r50 = E50.reindex(nav.index).pct_change(); msk = r.index > nav.index[0]
        return {int(y): {'s': float((1 + g.fillna(0)).prod() - 1) * 100, '0050': float((1 + r50[g.index].fillna(0)).prod() - 1) * 100}
                for y, g in r[msk].groupby(r[msk].index.year)}
    def breakeven(nm):
        pts = [(s, L.seg(SLIPNAV[(nm, s)], EW['AB'], sd, *PER['ALL'])['t_0050']) for s in SLIPS]
        def cross(th):
            for (s1, a), (s2, b) in zip(pts, pts[1:]):
                if a >= th > b: return s1 + (s2 - s1) * (a - th) / (a - b)
            return '<0.10%' if pts[0][1] < th else '>0.40%'
        return {'slip_at_t2.3': cross(2.3), 'slip_at_t2.0': cross(2.0), 't_at_slip': {str(s): t for s, t in pts}}
    def seed_dist(nm):
        ts, tc, th = [], [], []
        for s in SEEDS:
            n_ = SEEDNAV[(nm, s)]
            ts.append(L.seg(n_, EW['AB'], sd, *PER['ALL'])['t_0050']); tc.append(tstat(cal_ex(n_))); th.append(L.seg(n_, EW['AB'], sd, *PER['HOLD'])['t_0050'])
        ts, tc, th = map(np.array, (ts, tc, th))
        return {'t_median': float(np.median(ts)), 't_min': float(ts.min()), 't_max': float(ts.max()), 't_p10': float(np.percentile(ts, 10)),
                'n_below_2.3': int((ts < 2.3).sum()), 'n_below_2.0': int((ts < 2.0).sum()), 'cal_t_median': float(np.median(tc)),
                'cal_t_min': float(tc.min()), 'HOLD_t_median': float(np.median(th)), 'n': len(ts)}

    STATS = {}
    for base in ('AB', 'B10'):
        for nm in ORDER:
            nav = NAV[f'{nm}|{base}|normal']; o = {'periods': {}}
            ref = NAV[f'AB_V6|{base}|normal']
            for p in PK:
                a, b = PER[p]; s_ = L.seg(nav, EW[base], sd, a, b)
                x = xs(nav, a, b); s_['TE'] = float(x.std(ddof=1) * math.sqrt(12) * 100)
                c_ = cal_ex(nav, p); s_['t_cal'], s_['n_cal'] = tstat(c_), int(len(c_))
                if nm != 'AB_V6':
                    s_['vs_AB_V6_cal'] = paired_cal(nav, ref, p); s_['vs_AB_V6_win'] = paired_win(nav, ref, a, b)
                    s_['corr_AB_V6_win'] = corr_win(nav, ref, a, b); s_['corr_AB_V6_excess'] = corr_win(nav, ref, a, b, True); s_['corr_AB_V6_cal'] = corr_cal(nav, ref, p)
                st_ = L.seg(NAV[f'{nm}|{base}|stress'], EW[base], sd, a, b); s_['stress_CAGR'] = st_['CAGR']; s_['stress_t_0050'] = st_['t_0050']; s_['stress_maxDD'] = st_['maxDD']
                o['periods'][p] = s_
            o['HOLD_cal'] = hold_cal(nav, EW[base])
            o['HOLD_from_2023_12_29_CAGR'] = L.seg(nav, EW[base], sd, *PER['HOLD'])['CAGR']
            if base == 'AB':
                o['endpoint'] = endpoint(nav); o['yearly'] = yearly(nav); o['breakeven'] = breakeven(nm); o['seed_dist'] = seed_dist(nm)
                o['years_lose_0050'] = int(sum(1 for y, v in o['yearly'].items() if v['s'] < v['0050']))
            STATS[f'{nm}|{base}'] = o
            log('stats', nm, base)

    # ---------------- R_V6（取代）對 AB_V4：同一主策略、兩種資料版本 ----------------
    a6, a4 = NAV['AB_V6|AB|normal'], NV4['AB_V4|normal'].dropna()
    b6, b4 = NAV['AB_V6|B10|normal'], NV4['AB_V4_B10|normal'].dropna()
    RV = {}
    for lab, x6, x4 in (('AB', a6, a4), ('B10', b6, b4)):
        ew4 = NV4[f"{'AB_V4' if lab == 'AB' else 'AB_V4_B10'}|EW"].dropna()
        v4s = {p: L.seg(x4, ew4, list(G4), *PER[p]) for p in PK}
        o = {'paired_V4grid_win': {p: pd_stats(wsel(ST.win(x6, G4).r - ST.win(x4, G4).r, *PER[p])) for p in PK},
             'paired_cal_merge_2024_01_02': {p: paired_cal(x6, x4, p, drop=('2024-01',)) for p in PK},
             'corr_V4grid_win_ALL': float(pd.concat([ST.win(x6, G4).r, ST.win(x4, G4).r], axis=1).dropna().corr().iloc[0, 1]),
             'V4_periods_on_V4_grid': v4s,
             'CAGR_diff': {p: L.seg(x6, EW[lab], sd, *PER[p])['CAGR'] - v4s[p]['CAGR'] for p in PK}}
        RV[lab] = o
    log('R_V6 paired done')

    # ---------------- 保留期預測分布（照抄 V6 s5 演算法；日曆歸屬，V6 格點）----------------
    HP = V6R['PREREG']['hold']
    def sb_paths(n_src, n_out, Lb, Bn, seed):
        rng = np.random.default_rng(seed); p = 1 / Lb
        idx = np.empty((Bn, n_out), np.int64); idx[:, 0] = rng.integers(0, n_src, Bn)
        for t in range(1, n_out):
            new = rng.random(Bn) < p
            idx[:, t] = np.where(new, rng.integers(0, n_src, Bn), (idx[:, t - 1] + 1) % n_src)
        return idx
    HB = 200 if SMOKE else HP['B']
    def predictive(ra, rb):
        ends = pd.Series(G6[1:], index=G6[:-1])
        pre = ((ends < ST.HOLD_CAL_END) & (ra.index >= pd.Timestamp(HP['pre'][0]))).to_numpy(); hold = (ends >= ST.HOLD_CAL_END).to_numpy()
        lx = np.log1p(ra) - np.log1p(rb); act = float(lx[hold].sum()); src = lx[pre].to_numpy(); nh = int(hold.sum())
        res = {'cumlog': act, 'n_hold': nh, 'n_pre': int(pre.sum())}
        for Lb in HP['block_lengths']:
            idx = sb_paths(len(src), nh, Lb, HB, HP['seed'] + Lb); sl = src[idx].sum(axis=1)
            res[f'L{Lb}'] = {'p_lower': float((sl <= act).mean()), 'q10': float(np.quantile(sl, 0.10)), 'q50': float(np.quantile(sl, 0.50)), 'q90': float(np.quantile(sl, 0.90))}
        pl = [res[f'L{Lb}']['p_lower'] for Lb in HP['block_lengths']]; res['p_lower_range'] = [min(pl), max(pl)]
        res['judgement_L6'] = '與雜訊相容' if 0.10 <= res['L6']['p_lower'] <= 0.90 else ('落後超出雜訊' if res['L6']['p_lower'] < 0.10 else '超前超出雜訊')
        return res
    r50_6 = ST.win(E50raw, G6).r
    HOLD = {}
    for base in ('AB', 'B10'):
        for nm in ORDER:
            for c in ('normal', 'stress'):
                if base == 'B10' and c == 'stress': continue
                rm = ST.win(NAV[f'{nm}|{base}|{c}'], G6).r
                o = {'x50': predictive(rm, r50_6)}
                if c == 'normal': o['xEW'] = predictive(rm, ST.win(EW[base], G6).r)
                HOLD[f'{nm}|{base}|{c}'] = o
    if not SMOKE:
        ref = V6R['RESULTS_HOLD_SPA']['hold']
        for k6, kk in (('AB_V6', 'AB_V6|AB|normal'), ('AB_V6_B10', 'AB_V6|B10|normal'), ('AB_V6_stress', 'AB_V6|AB|stress')):
            for Lb in HP['block_lengths']:
                r_ = ref[k6]['x50_calendar_v6grid'][f'L{Lb}']['p_lower_cumlog']; m_ = HOLD[kk]['x50'][f'L{Lb}']['p_lower']
                VAL[f'hold_repro_{k6}_L{Lb}'] = [m_, r_]; assert abs(m_ - r_) < 1e-12, (k6, Lb, m_, r_)
            if k6 != 'AB_V6_stress':
                r_ = ref[k6]['xEW_calendar_v6grid']['L6']['p_lower_cumlog']; m_ = HOLD[kk]['xEW']['L6']['p_lower']
                VAL[f'hold_repro_{k6}_xEW_L6'] = [m_, r_]; assert abs(m_ - r_) < 1e-12
    log('hold done')

    # ---------------- SPA（o_main_spa_breaks F_long × W_long；V6 版族群＝14 條 o_v4 成員換 V6，同 V6 研究）----------------
    SPR = json.loads((SPD / 'results.json').read_text(encoding='utf-8')); SPP = SPR['PREREG']
    D = pd.read_parquet(DATA / 'family_excess.parquet'); MM = pd.read_csv(DATA / 'family_members.csv'); BB = pd.read_parquet(DATA / 'spa_bench.parquet')
    pref = 'o_v4_revenue_patch_official/navs_v4.parquet::'
    v4mem = [m for m in MM.member_id if m.startswith(pref)]
    def wr(nav, g): return pd.Series(ST.win(nav, g).r.to_numpy(), index=g[:-1])
    def v6name(k):
        k = k.replace('V4', 'V6'); return 'O3REP_V6' + k[5:] if k.startswith('O3REP') else k
    D6 = D.copy()
    for m in v4mem: D6[m] = (wr(NV6[v6name(m[len(pref):])], G4) - BB['r0050']).to_numpy()
    mem = MM[MM['cov_W_long'] & MM['F_long']].member_id.tolist()
    wm = (D.index >= pd.Timestamp('2006-04-01')) & (D.index <= pd.Timestamp('2026-10-12')); n = int(wm.sum())
    def cand_ex(nav): return (wr(nav, G4) - BB['r0050']).to_numpy()[wm]
    VAL['AB_V6_mine_vs_family_member_maxabs'] = float(np.abs(cand_ex(NAV['AB_V6|AB|normal']) - D6.loc[wm, pref + 'AB_V4|normal'].to_numpy()).max())
    VAL['AB_V6_B10_mine_vs_family_member_maxabs'] = float(np.abs(cand_ex(NAV['AB_V6|B10|normal']) - D6.loc[wm, pref + 'AB_V4_B10|normal'].to_numpy()).max())
    if not SMOKE: assert VAL['AB_V6_mine_vs_family_member_maxabs'] < 1e-9 and VAL['AB_V6_B10_mine_vs_family_member_maxabs'] < 1e-9
    EXTRA = {'EXTRA::AB_V6|stress': cand_ex(NV6['AB_V6|stress'])}
    for nm in MIX:
        for b, c in (('AB', 'normal'), ('B10', 'normal'), ('AB', 'stress')):
            EXTRA[f'integrate_o6::{nm}|{b}|{c}'] = cand_ex(NAV[f'{nm}|{b}|{c}'])
    for k_, v_ in EXTRA.items(): assert np.isfinite(v_).all(), k_
    X0 = D6.loc[wm, mem].fillna(0.0).to_numpy()
    names = mem + list(EXTRA)
    X = np.hstack([X0] + [EXTRA[k_][:, None] for k_ in EXTRA])
    B = 200 if SMOKE else V6R['PREREG']['spa']['B']
    SPA = {'N_family': len(names), 'n_windows': n}
    def grade(p): return '第 2 級（通過資料窺探校正）' if p <= 0.05 else ('第 2 級（邊緣）' if p <= 0.10 else '第 3 級（有理論支持，但未通過資料窺探校正）')
    FOCUS = {'AB_V6|AB|normal': pref + 'AB_V4|normal', 'AB_V6|B10|normal': pref + 'AB_V4_B10|normal', 'AB_V6|AB|stress': 'EXTRA::AB_V6|stress'}
    for nm in MIX:
        for b, c in (('AB', 'normal'), ('B10', 'normal'), ('AB', 'stress')): FOCUS[f'{nm}|{b}|{c}'] = f'integrate_o6::{nm}|{b}|{c}'
    KEYS = ('N', 'max_t', 'p_SPA_c', 'p_SPA_c_focus', 'p_SPA_l_focus', 'p_SPA_u_focus', 'p_RC_focus', 'focus_rank', 'focus_t', 'p_StepM_focus', 'p_single_focus')
    for Lb in V6R['PREREG']['spa']['block_lengths']:
        Wc = SP.sb_counts(n, Lb, B, V6R['PREREG']['spa']['seed_base'] + 1000 * 0 + Lb)
        if not SMOKE:   # 重現 V6 研究的焦點 p（同族群、同種子）
            ref = V6R['RESULTS_HOLD_SPA']['spa']
            for foc, fid, mm_, XX in (('AB_V6', pref + 'AB_V4|normal', mem, X0), ('AB_V6_B10', pref + 'AB_V4_B10|normal', mem, X0),
                                      ('AB_V6_stress', 'EXTRA::AB_V6|stress', mem + ['EXTRA::AB_V6|stress'], np.hstack([X0, EXTRA['EXTRA::AB_V6|stress'][:, None]]))):
                o = SP.mcp_tests(XX, Wc, Lb, V6R['PREREG']['spa']['variance'], focus=mm_.index(fid))
                VAL[f'spa_repro_{foc}_L{Lb}'] = [o['p_SPA_c_focus'], ref[f'{foc}|L{Lb}']['p_SPA_c_focus'], o['focus_rank'], ref[f'{foc}|L{Lb}']['focus_rank']]
                assert abs(o['p_SPA_c_focus'] - ref[f'{foc}|L{Lb}']['p_SPA_c_focus']) < 1e-12 and o['focus_rank'] == ref[f'{foc}|L{Lb}']['focus_rank'], VAL
        Mb = SP.boot_means(Wc, X)
        for k_, fid in FOCUS.items():
            o = SP.mcp_tests(X, Wc, Lb, V6R['PREREG']['spa']['variance'], Mb=Mb, focus=names.index(fid))
            SPA[f'{k_}|L{Lb}'] = {kk: o[kk] for kk in KEYS}; SPA[f'{k_}|L{Lb}']['grade'] = grade(o['p_SPA_c_focus'])
        del Mb
        log('SPA', Lb)

    # ---------------- 實盤預期：可執行版 j 分布（AB_V6_B10 為 SOP 口徑；AB_V6 並列）----------------
    EXN = pd.read_parquet(DATA / 'exec_navs_abavg.parquet'); BEN = pd.read_parquet(DATA / 'exec_bench.parquet'); rzi = BEN['AB_V4|normal'].pct_change()
    JN = pd.read_parquet(DATA / 'navs_jscan_v6.parquet')
    TIERS = [f'main|{t}|{m}|A' for t in ('30萬', '100萬', '300萬', '1000萬') for m in ('FIX', 'GROW')]
    EXECD = [K.IDX[i] for i in EXEC]
    r2330_all = pd.Series(K.GROW[:, K.CI['2330']] - 1.0, index=K.IDX)
    LIVE = {}
    for base in ('AB_V6', 'AB_V6_B10'):
        jr = V6R['RESULTS_JSCAN'][f'{base}|normal']
        for col in TIERS:
            per = {nm: {'CAGR': [], 'maxDD': [], 'E50_CAGR': []} for nm in ORDER}
            for j in range(13):
                nav = JN[f'{base}|normal|J{j}'].dropna(); de = min(pd.Timestamp(jr['by_j'][str(j)]['end_day']), pd.Timestamp(ST.B0))
                drag = (EXN[col].pct_change() - rzi).reindex(nav.index).fillna(0.0)
                ne = (1 + nav.pct_change().fillna(0) + drag).cumprod()
                ix = ne.index; exi = set(ix.get_loc(d) for d in EXECD if d in ix)
                rE = e50_on(ix).pct_change().fillna(0).to_numpy(); r23 = r2330_all.reindex(ix).fillna(0).to_numpy().copy(); r23[0] = 0.0
                for nm in ORDER:
                    if nm == 'AB_V6': nn = ne
                    else:
                        w, kd = MIX[nm]; nn = sleeve(ne, rE if kd == 'etf' else r23, w, L.cost_rates('normal'), kd, ix, exi)
                    cg, md = ST.cagr_mdd(nn, ST.A0, de); per[nm]['CAGR'].append(cg); per[nm]['maxDD'].append(md)
                    per[nm]['E50_CAGR'].append(ST.cagr_mdd(e50_on(ix), ST.A0, de)[0])
                if not SMOKE:
                    ref_ = jr['by_j'][str(j)][f'exec|{col}']['CAGR']
                    assert abs(per['AB_V6']['CAGR'][-1] - ref_) < 1e-9, (base, col, j, per['AB_V6']['CAGR'][-1], ref_)
            LIVE[f'{base}|{col}'] = {nm: {'CAGR_median': float(np.median(v['CAGR'])), 'CAGR_p10': float(np.percentile(v['CAGR'], 10)),
                                          'maxDD_median': float(np.median(v['maxDD'])), 'E50_CAGR_median': float(np.median(v['E50_CAGR'])),
                                          'excess_median_pp': float(np.median(np.array(v['CAGR']) - np.array(v['E50_CAGR'])))} for nm, v in per.items()}
    VAL['live_exec_j_AB_V6_reproduces_V6_RESULTS_JSCAN'] = not SMOKE
    log('live done')

    # ---------------- 選版（只用 SEL）與判定 ----------------
    ab = STATS['AB_V6|AB']['periods']
    sel = []
    for nm in MIX:
        s_ = STATS[f'{nm}|AB']['periods']['SEL']
        ok = (s_['t_0050'] > ab['SEL']['t_0050']) and (ab['SEL']['CAGR'] - s_['CAGR'] <= 1.0)
        sel.append({'name': nm, 'SEL_t': s_['t_0050'], 'SEL_CAGR_loss_pp': ab['SEL']['CAGR'] - s_['CAGR'], 'eligible': bool(ok)})
    elig = [x for x in sel if x['eligible']]
    SELECTED = max(elig, key=lambda x: x['SEL_t'])['name'] if elig else 'AB_V6（主策略單獨）'
    JUDGE = {}
    for nm in MIX:
        P = STATS[f'{nm}|AB']['periods']
        pa = P['ALL']['vs_AB_V6_cal']['t']; ms, mv = P['SEL']['vs_AB_V6_cal']['mean_pp'], P['VAL']['vs_AB_V6_cal']['mean_pp']
        A = pa >= 1.0 and ms > 0 and mv > 0
        Bp = (P['ALL']['t_0050'] - ab['ALL']['t_0050'] >= 0.2 and P['SEL']['t_0050'] > ab['SEL']['t_0050'] and P['VAL']['t_0050'] > ab['VAL']['t_0050']
              and ab['ALL']['CAGR'] - P['ALL']['CAGR'] <= 1.0 and P['ALL']['maxDD'] >= ab['ALL']['maxDD'])
        W = pa > -1.0 and ab['ALL']['CAGR'] - P['ALL']['CAGR'] <= 2.0 and (P['SEL']['t_0050'] > ab['SEL']['t_0050'] or P['VAL']['t_0050'] > ab['VAL']['t_0050'])
        raw = '可用' if (A or Bp) else ('待觀察' if W else '不可用')
        JUDGE[nm] = {'path_A': bool(A), 'path_B': bool(Bp), 'watch_cond': bool(W), 'raw': raw, 'final': '待觀察' if raw == '可用' else raw,
                     'SPA_L6_p_c_focus': SPA[f'{nm}|AB|normal|L6']['p_SPA_c_focus'], 'SPA_grade': SPA[f'{nm}|AB|normal|L6']['grade']}
    rep = V6R['SUMMARY_posthoc']['replication']['pass']
    rep_ok = bool(rep['panel>=99.9%'] and rep['nav<=1e-8'] and all(v <= 0.02 for v in rep['spa_diff<=0.02'].values()) and all(v <= 0.02 for v in rep['hold_diff<=0.02'].values()))
    dC = STATS['AB_V6|AB']['periods']['ALL']['CAGR'] - RV['AB']['V4_periods_on_V4_grid']['ALL']['CAGR']
    cond = {'rerun_reproduces_navs_v6_le_1e-9': not SMOKE, 'identical_to_AB_V4_through_2023_12_29': JUMP['AB_V6_vs_AB_V4_identical_to_2023_12_29_maxrel'] <= 1e-12,
            'V6_gate_PASS': bool(V6R['GATE']['PASS']), 'V6_replication_pass': rep_ok, 'abs_dCAGR_ALL_le_1pp': bool(abs(dC) <= 1.0), 'dCAGR_ALL_pp': dC}
    JUDGE['R_V6'] = {'conditions': cond, 'decision': '取代（資料版本）：SOP 正式數字換成 V6' if all(v for k_, v in cond.items() if k_ != 'dCAGR_ALL_pp') else '不取代，維持 V4',
                     'SPA_L6': {k_: SPA[f'{k_}|L6']['p_SPA_c_focus'] for k_ in ('AB_V6|AB|normal', 'AB_V6|B10|normal', 'AB_V6|AB|stress')},
                     'grade': '可提議回到第 2 級（需總監確認）' if (SPA['AB_V6|AB|normal|L6']['p_SPA_c_focus'] <= 0.10 and SPA['AB_V6|B10|normal|L6']['p_SPA_c_focus'] <= 0.10) else '維持第 3 級',
                     'hold_L6': HOLD['AB_V6|AB|normal']['x50']['judgement_L6']}
    OUT = {'validation': VAL, 'stats': STATS, 'R_V6_vs_AB_V4': RV, 'hold': HOLD, 'spa': SPA, 'live_exec_j': LIVE,
           'selection': {'table': sel, 'selected': SELECTED}, 'judgement': JUDGE, 'run_seconds': time.time() - T0_}
    def js(o):
        if isinstance(o, dict): return {str(k): js(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [js(v) for v in o]
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        return o
    if SMOKE:
        Path(sys.argv[sys.argv.index('--smoke') + 1]).write_text(json.dumps(js(OUT), ensure_ascii=False))
        print('smoke ok', flush=True)
    else:
        pd.DataFrame({k: v for k, v in NAV.items()}).to_parquet(HERE / 'navs.parquet')
        RES['RESULTS'] = js(OUT); RES['RESULTS_written_at'] = pd.Timestamp.now(tz='UTC').isoformat()
        RJ.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding='utf-8')
        log('done', SELECTED, {k: v.get('final', v.get('decision')) for k, v in JUDGE.items()})

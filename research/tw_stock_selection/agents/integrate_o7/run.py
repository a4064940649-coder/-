#!/usr/bin/env python3
"""integrate_o7 回測：照 results.json 的 PREREG，把 o7 唯一通過審查（待觀察）的 o_v7_listing_exec_engine（V7：上市身分母體、颱風除權息、
2024-01 前視修補，加上逐名單可執行引擎）和主策略（排除金融 n10）整合。
引擎全部只讀 import：o_v7_listing_exec_engine/v7env.py（ia_core 面板路徑換成 V7 data/）→ v4lib → v2_core／drj_core／o3_core；
v7stats.py（窗、年化、HOLD 日曆歸屬）；o_main_spa_breaks/spa_lib.py、fam_common.py。
本檔只新寫：袖子混合（與 integrate_o4／o5／o6 同一函式）、保留期預測分布（照抄 V7 s5 演算法並斷言重現）、可執行 j 分布上的袖子混合與統計彙整。
用法：python3 -I -B run.py（正式）；python3 -I -B run.py --smoke <輸出檔>（1 seed、B=200、j 只跑 2 個，只測管線，不寫 results.json、不印數字）。"""
import sys, json, time, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
DATA = HERE / 'data'
V7D = AG / 'o_v7_listing_exec_engine'
SPD = AG / 'o_main_spa_breaks'
SMOKE = '--smoke' in sys.argv
import numpy as np, pandas as pd
from multiprocessing import get_context

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
RJ = HERE / 'results.json'
if not SMOKE:
    RES = json.loads(RJ.read_text(encoding='utf-8'))
    assert hashlib.sha256(json.dumps(RES['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256'], 'PREREG 被改過'
    assert 'RESULTS' not in RES, 'RESULTS 已存在：只跑一次'
    for f, h in RES['PREREG']['sha256'].items():
        assert sha(f) == h, f'{f} 與 PREREG 登記的 sha256 不符'
COPIES = {'navs_v7.parquet': V7D / 'navs_v7.parquet', 'seed_navs_v7.parquet': V7D / 'seed_navs_v7.parquet', 'navs_exec7.parquet': V7D / 'navs_exec7.parquet',
          'plan_V7.csv': V7D / 'plan_V7.csv', 'plans_j7.csv': V7D / 'data/plans_j7.csv', 'jscan_end_days.json': V7D / 'data/jscan_end_days.json',
          'navs_v4.parquet': AG / 'o_v4_revenue_patch_official/navs_v4.parquet', 'plan_V4.csv': AG / 'o_v4_revenue_patch_official/plan_V4.csv',
          'navs_v6.parquet': AG / 'o_v6_price_panel_official/navs_v6.parquet', 'plan_V6.csv': AG / 'o_v6_price_panel_official/plan_V6.csv',
          'integrate_o6_navs.parquet': AG / 'integrate_o6/navs.parquet',
          'family_excess.parquet': SPD / 'data/family_excess.parquet', 'family_members.csv': SPD / 'data/family_members.csv', 'spa_bench.parquet': SPD / 'data/bench.parquet',
          'etf_0050.csv': Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'), 'taiex_twii.csv': Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')}
for k, f in COPIES.items(): assert sha(DATA / k) == sha(f), k
V7R = json.loads((V7D / 'results.json').read_text(encoding='utf-8'))
assert V7R['PREREG_sha256'] == 'e84cf8bd00d3374e914fb3b168f2a79941ed59627f00bb11c06c42d3ccee6e71'
fz = json.loads((V7D / 'data_freeze_v7.json').read_text(encoding='utf-8'))
for k, v in fz['v7_panel'].items(): assert sha(v['path']) == v['sha256'], k

sys.path.insert(0, str(V7D))
import v7env, v7stats as ST
sys.path.pop(0)
L = v7env.load('v7')
K, V = L.K, L.V
sys.path.insert(0, str(SPD))
import spa_lib as SP, fam_common as FC
sys.path.pop(0)
T0_ = time.time()
def log(*a):
    if SMOKE: print(f'[{time.time() - T0_:6.0f}s]', *[x for x in a if isinstance(x, str)], flush=True)
    else: print(f'[{time.time() - T0_:6.0f}s]', *a, flush=True)

VAL = {}
NV7 = pd.read_parquet(DATA / 'navs_v7.parquet'); SN7 = pd.read_parquet(DATA / 'seed_navs_v7.parquet')
NV4 = pd.read_parquet(DATA / 'navs_v4.parquet'); NV6 = pd.read_parquet(DATA / 'navs_v6.parquet')
E50raw = pd.read_csv(DATA / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
def e50_on(ix): return E50raw.reindex(ix.union(E50raw.index)).ffill().reindex(ix)

# ---------------- (0) 重現：40 seeds 平均＝navs_v7；seeds 0、1 用 V7 引擎重跑＝seed_navs_v7 ----------------
fz4 = L.check_freeze()
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
S0 = L.V4(YFX, Y5F, DMT)
t0, t1, sd = S0.bounds()
pl0 = S0.plan(10, 'official')
EXEC = [p[0] for p in pl0]
IX = K.IDX[t0:t1 + 1]
assert NV7.index.equals(IX), 'navs_v7 日期與引擎不符'
SPEC = {'AB_V7': dict(), 'AB_V7_B10': dict(gap=GAP, pfix=1.0)}
RERUN_SEEDS = [0] if SMOKE else [0, 1]
for s in SPEC:
    for c in ('normal', 'stress'):
        m = (1 + pd.concat([SN7[f'{s}|{c}|s{k}'].pct_change() for k in range(40)], axis=1).mean(axis=1).fillna(0)).cumprod()
        VAL[f'{s}_{c}_mean40seeds_vs_navs_v7_maxrel'] = float((m / NV7[f'{s}|{c}'] - 1).abs().max())
        assert VAL[f'{s}_{c}_mean40seeds_vs_navs_v7_maxrel'] <= 1e-9, VAL
    for seed in RERUN_SEEDS:
        S = L.V4(YFX, Y5F, DMT, unavail=V.make_unavail('mc', seed), **SPEC[s]); pl = S.plan(10, 'official')
        assert S.bounds()[:2] == (t0, t1) and [p[0] for p in pl] == EXEC
        for c in ('normal', 'stress'):
            nav = L.sim_plan(pl, t0, t1, c, XR)[0]
            VAL[f'{s}_{c}_s{seed}_rerun_vs_seed_navs_v7_maxrel'] = float((nav / SN7[f'{s}|{c}|s{seed}'].reindex(nav.index) - 1).abs().max())
            assert VAL[f'{s}_{c}_s{seed}_rerun_vs_seed_navs_v7_maxrel'] <= 1e-9, VAL
sdp = [pd.Timestamp(x) for x in pd.read_csv(DATA / 'plan_V7.csv')['signal']] + [pd.Timestamp(ST.B0)]
sd = [pd.Timestamp(x) for x in sd]
assert sd == sdp or sd[:-1] == sdp[:-1], 'V7 格點與 plan_V7 不符'
G7 = pd.DatetimeIndex(sdp); sd = list(G7)
G4 = FC.GRID
EXECD = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(DATA / 'plan_V7.csv')['exec']))
assert list(EXECD) == [K.IDX[i] for i in EXEC], 'plan_V7 成交日與引擎不符'
EXI = set(IX.get_loc(d) for d in EXECD if d in IX)
VAL['n_windows_V7_grid'] = len(G7) - 1
log('rerun done', VAL)

# ---------------- (1) 淨值跳日檢查（o5 教訓 1）----------------
tx = pd.read_csv(DATA / 'taiex_twii.csv', parse_dates=['date'])['date']; TX = pd.DatetimeIndex(sorted(set(tx)))
TXw = TX[(TX >= IX[0]) & (TX <= IX[-1])]
r7 = NV7['AB_V7|normal'].pct_change()
i4 = NV4['AB_V4|normal'].dropna(); i6 = NV6['AB_V6|normal'].dropna()
JUMP = {'taiex_days_missing_in_V7_index': [str(d.date()) for d in TXw[~TXw.isin(IX)]], 'V7_index_days_not_taiex': [str(d.date()) for d in IX[~IX.isin(TX)]],
        'AB_V7_abs_daily_ret_gt_10pct': {str(d.date()): float(v) for d, v in r7[r7.abs() > 0.10].items()},
        'AB_V7_max_abs_daily_ret_2024_01_02_to_02_15': float(r7['2024-01-02':'2024-02-15'].abs().max()),
        'AB_V7_vs_AB_V6_maxrel_to_2023_12_29': float((NV7['AB_V7|normal'][:'2023-12-29'] / i6[:'2023-12-29'] - 1).abs().max()),
        'AB_V7_vs_AB_V4_maxrel_to_2023_12_29': float((NV7['AB_V7|normal'][:'2023-12-29'] / i4[:'2023-12-29'] - 1).abs().max())}
d76 = (NV7['AB_V7|normal'].pct_change() - i6.reindex(IX).pct_change()).abs()
JUMP['first_day_AB_V7_ne_AB_V6_daily_ret_gt_1e-12'] = str(d76[d76 > 1e-12].index[0].date()) if (d76 > 1e-12).any() else None
VAL['jump_check'] = JUMP

# ---------------- (2) 袖子混合（同 integrate_o4／o5／o6）----------------
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
# 驗證：同一函式在 V6 日曆上重算 integrate_o6 的 M_E50_20/30/50（AB、B10 × 一般、壓力）
NI6 = pd.read_parquet(DATA / 'integrate_o6_navs.parquet')
IX6 = i6.index; RE50_6 = e50_on(IX6).pct_change().fillna(0).to_numpy()
EXI6 = set(IX6.get_loc(pd.Timestamp(d)) for d in pd.read_csv(DATA / 'plan_V6.csv')['exec'] if pd.Timestamp(d) in IX6)
for nm_, w_ in (('M_E50_20', 0.8), ('M_E50_30', 0.7), ('M_E50_50', 0.5)):
    for b_, col_ in (('AB', 'AB_V6'), ('B10', 'AB_V6_B10')):
        for c in ('normal', 'stress'):
            a = sleeve(NV6[f'{col_}|{c}'].dropna(), RE50_6, w_, L.cost_rates(c), 'etf', IX6, EXI6)
            VAL[f'sleeve_V6_{nm_}_{b_}_{c}_vs_integrate_o6_maxrel'] = float((a / NI6[f'{nm_}|{b_}|{c}'].reindex(IX6) - 1).abs().max())
            assert VAL[f'sleeve_V6_{nm_}_{b_}_{c}_vs_integrate_o6_maxrel'] <= 1e-8, VAL
log('sleeve V6 validation done')

R_E50 = e50_on(IX).pct_change().fillna(0).to_numpy()
R_2330 = (K.GROW[t0:t1 + 1, K.CI['2330']] - 1.0).copy(); R_2330[0] = 0.0
MIX = {'M_E50_20': (0.8, 'etf'), 'M_E50_30': (0.7, 'etf'), 'M_E50_50': (0.5, 'etf'), 'C_E50_70': (0.3, 'etf'), 'M_2330_30': (0.7, 'stock')}
LEG = {'etf': R_E50, 'stock': R_2330}
ORDER = ['AB_V7'] + list(MIX)
BASEK = {'AB': 'AB_V7', 'B10': 'AB_V7_B10'}
def build(nm, base, c):
    nb = NV7[f'{BASEK[base]}|{c}']
    if nm == 'AB_V7': return nb
    w, kd = MIX[nm]; return sleeve(nb, LEG[kd], w, L.cost_rates(c), kd, IX, EXI)
NAV = {f'{nm}|{b}|{c}': build(nm, b, c) for nm in ORDER for b in ('AB', 'B10') for c in ('normal', 'stress')}
SEEDS = list(range(40))
SEEDNAV = {(nm, s): (SN7[f'AB_V7|normal|s{s}'] if nm == 'AB_V7' else sleeve(SN7[f'AB_V7|normal|s{s}'], LEG[MIX[nm][1]], MIX[nm][0], L.cost_rates('normal'), MIX[nm][1], IX, EXI))
           for nm in ORDER for s in SEEDS}
EW = {'AB': NV7['AB_V7|EW'], 'B10': NV7['AB_V7_B10|EW']}
log('combos built')

# ---------------- (3) 統計 ----------------
E50 = K.E50
PER = dict(ST.PER); PER['P19_22'] = ('2019-01-01', '2022-12-31'); PER['P23_26'] = ('2023-01-01', '2026-10-12')
PK = ('S1', 'S2', 'S3', 'SEL', 'VAL', 'HOLD', 'Y19on', 'P19_22', 'P23_26', 'ALL')
CAL_A, CAL_B = pd.Period('2006-05', 'M'), pd.Period('2026-09', 'M')
tstat = ST.tstat
def cal_ret(nav, drop=()):
    s = nav.dropna(); me = s.groupby(s.index.to_period('M')).last()
    me = me[~me.index.isin([pd.Period(x, 'M') for x in drop])]
    return me.pct_change()
E50M = cal_ret(E50raw)
def selp(r, key):
    a, b = PER[key]; pa = max(pd.Period(a, 'M'), CAL_A); pb = min(pd.Period(b, 'M'), CAL_B)
    return r[(r.index >= pa) & (r.index <= pb)].dropna()
def cal_ex(nav, key='ALL'): r = cal_ret(nav); return selp(r - E50M.reindex(r.index), key)
def pd_stats(d):
    d = d.dropna(); nz = d[d.abs() > 1e-12]
    o = dict(mean_pp=float(d.mean() * 100), t=tstat(d), n=int(len(d)), n_nonzero=int(len(nz)),
             MDE80_pp=float(2.8 * d.std(ddof=1) / math.sqrt(len(d)) * 100) if len(d) > 2 else None)
    if len(nz) >= 3:
        big = d.abs().idxmax(); o['drop_maxabs_t'] = tstat(d.drop(big)); o['maxabs_period'] = str(big)[:10]; o['maxabs_pp'] = float(d[big] * 100)
    return o
def wsel(s, a, b): return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))].dropna()
def paired_cal(a, b, key, drop=()): return pd_stats(selp(cal_ret(a, drop) - cal_ret(b, drop), key))
def paired_win(a, b, a_, b_, g=G7): return pd_stats(wsel(ST.win(a, g).r - ST.win(b, g).r, a_, b_))
def corr_win(a, b, a_, b_, excess=False):
    ra, rb = wsel(ST.win(a, G7).r, a_, b_), wsel(ST.win(b, G7).r, a_, b_)
    if excess:
        e = wsel(ST.win(E50raw, G7).r, a_, b_); ra, rb = ra - e, rb - e
    z = pd.concat([ra, rb], axis=1).dropna(); return float(z.iloc[:, 0].corr(z.iloc[:, 1]))
def corr_cal(a, b, key):
    z = pd.concat([selp(cal_ret(a), key), selp(cal_ret(b), key)], axis=1).dropna(); return float(z.iloc[:, 0].corr(z.iloc[:, 1]))
def seg(nav, ew, a, b):
    cg, md = ST.cagr_mdd(nav, a, b); x = ST.xs(nav, E50, sd, a, b); xe = ST.xs(nav, ew, sd, a, b)
    return {'CAGR': cg, 'maxDD': md, 'E50_CAGR': ST.cagr_mdd(E50.reindex(nav.index).ffill(), a, b)[0], 'E50_maxDD': ST.cagr_mdd(E50.reindex(nav.index).ffill(), a, b)[1],
            'EW_CAGR': ST.cagr_mdd(ew, a, b)[0], 't_0050': tstat(x), 'mean_pp': float(x.mean() * 100), 'n': int(len(x)),
            'TE': float(x.std(ddof=1) * math.sqrt(12) * 100), 't_EW': tstat(xe)}
def hold_cal(nav, ew):
    x = ST.xs(nav, E50, sd, ST.HOLD_CAL_END, ST.B0, attr='end'); xe = ST.xs(nav, ew, sd, ST.HOLD_CAL_END, ST.B0, attr='end')
    W = ST.win(nav, sd).r.reindex(x.index); Wb = ST.win(E50, sd).r.reindex(x.index)
    return {'t_0050': tstat(x), 'mean_pp': float(x.mean() * 100), 'n': int(len(x)), 't_EW': tstat(xe), 'cum_log_vs_0050': float(np.log1p(W).sum() - np.log1p(Wb).sum())}
# 先驗證統計函式重現 V7 研究的 AB_V7 表
for key_, col_ in (('AB_V7|normal', 'AB_V7|normal'), ('AB_V7_B10|normal', 'AB_V7_B10|normal'), ('AB_V7|stress', 'AB_V7|stress')):
    ref = V7R['RESULTS_RUN']['table'][key_]
    for p in ('ALL', 'SEL', 'VAL', 'HOLD', 'Y19on'):
        o = seg(NV7[col_], NV7[col_.split('|')[0] + '|EW'], *PER[p])
        for m in ('CAGR', 't_0050', 'maxDD', 't_EW'):
            VAL.setdefault('stats_repro_V7_table_maxabs', 0.0)
            VAL['stats_repro_V7_table_maxabs'] = max(VAL['stats_repro_V7_table_maxabs'], abs(o[m] - ref[p][m]))
assert VAL['stats_repro_V7_table_maxabs'] < 1e-9, VAL['stats_repro_V7_table_maxabs']

STATS = {}
E50S = {p: {'CAGR': ST.cagr_mdd(E50.reindex(IX), *PER[p])[0], 'maxDD': ST.cagr_mdd(E50.reindex(IX), *PER[p])[1]} for p in PK}
for base in ('AB', 'B10'):
    for nm in ORDER:
        nav = NAV[f'{nm}|{base}|normal']; o = {'periods': {}}
        ref = NAV[f'AB_V7|{base}|normal']
        for p in PK:
            a, b = PER[p]; s_ = seg(nav, EW[base], a, b)
            c_ = cal_ex(nav, p); s_['t_cal'], s_['n_cal'] = tstat(c_), int(len(c_))
            if nm != 'AB_V7':
                s_['vs_AB_V7_cal'] = paired_cal(nav, ref, p); s_['vs_AB_V7_win'] = paired_win(nav, ref, a, b)
                s_['corr_AB_V7_win'] = corr_win(nav, ref, a, b); s_['corr_AB_V7_excess'] = corr_win(nav, ref, a, b, True); s_['corr_AB_V7_cal'] = corr_cal(nav, ref, p)
            st_ = seg(NAV[f'{nm}|{base}|stress'], EW[base], a, b); s_['stress_CAGR'] = st_['CAGR']; s_['stress_t_0050'] = st_['t_0050']; s_['stress_maxDD'] = st_['maxDD']
            o['periods'][p] = s_
        o['HOLD_cal'] = hold_cal(nav, EW[base])
        if base == 'AB':
            o['endpoint'] = ST.endpoint(nav, E50, sd); o['yearly'] = ST.yearly(nav, E50)
            o['years_lose_0050'] = [y for y, v in o['yearly'].items() if v['strat'] < v['0050']]
            ts = np.array([tstat(ST.xs(SEEDNAV[(nm, s)], E50, sd)) for s in SEEDS]); t19 = np.array([tstat(ST.xs(SEEDNAV[(nm, s)], E50, sd, *PER['Y19on'])) for s in SEEDS])
            o['seed_dist'] = {'t_median': float(np.median(ts)), 't_min': float(ts.min()), 't_max': float(ts.max()), 'n_below_2.0': int((ts < 2.0).sum()),
                              'Y19on_t_median': float(np.median(t19)), 'Y19on_t_min': float(t19.min()), 'Y19on_t_max': float(t19.max())}
        STATS[f'{nm}|{base}'] = o
        log('stats', nm, base)

# ---------------- (4) R_V7：同一主策略、三種資料版本（V4／V6／V7）----------------
RV = {}
for lab, k7, k4, k6 in (('AB', 'AB_V7', 'AB_V4', 'AB_V6'), ('B10', 'AB_V7_B10', 'AB_V4_B10', 'AB_V6_B10')):
    x7, x4, x6 = NV7[f'{k7}|normal'], NV4[f'{k4}|normal'].dropna(), NV6[f'{k6}|normal'].dropna()
    o = {'vs_V4_V4grid_win': {p: pd_stats(wsel(ST.win(x7, G4).r - ST.win(x4, G4).r, *PER[p])) for p in PK},
         'vs_V4_cal_merge_2024_01_02': {p: paired_cal(x7, x4, p, drop=('2024-01',)) for p in PK},
         'vs_V6_V7grid_win': {p: pd_stats(wsel(ST.win(x7, G7).r - ST.win(x6, G7).r, *PER[p])) for p in PK},
         'vs_V6_cal': {p: paired_cal(x7, x6, p) for p in PK},
         'CAGR': {p: {'V7': ST.cagr_mdd(x7, *PER[p])[0], 'V6': ST.cagr_mdd(x6, *PER[p])[0], 'V4': ST.cagr_mdd(x4, *PER[p])[0]} for p in PK},
         't_0050': {p: {'V7': tstat(ST.xs(x7, E50, sd, *PER[p])), 'V6': tstat(ST.xs(x6, E50, sd, *PER[p])), 'V4': tstat(ST.xs(x4, E50, list(G4), *PER[p]))} for p in PK}}
    RV[lab] = o
log('R_V7 paired done')

# ---------------- (5) 保留期預測分布（照抄 V7 s5；日曆歸屬，V7 格點）----------------
HP = V7R['PREREG']['hold']
HB = 200 if SMOKE else HP['B']
def sb_paths(n_src, n_out, Lb, Bn, seed):
    rng = np.random.default_rng(seed); p = 1 / Lb
    idx = np.empty((Bn, n_out), np.int64); idx[:, 0] = rng.integers(0, n_src, Bn)
    for t in range(1, n_out):
        new = rng.random(Bn) < p
        idx[:, t] = np.where(new, rng.integers(0, n_src, Bn), (idx[:, t - 1] + 1) % n_src)
    return idx
ENDS = pd.Series(G7[1:], index=G7[:-1])
def predictive(ra, rb):
    pre = ((ENDS < ST.HOLD_CAL_END) & (ra.index >= pd.Timestamp(HP['pre'][0]))).to_numpy(); hold = (ENDS >= ST.HOLD_CAL_END).to_numpy()
    lx = np.log1p(ra) - np.log1p(rb); act = float(lx[hold].sum()); src = lx[pre].to_numpy(); nh = int(hold.sum()); mu = src.mean()
    res = {'cumlog': act, 'n_hold': nh, 'n_pre': int(pre.sum()), 'mean_arith_pp_hold': float((ra - rb)[hold].mean() * 100)}
    for Lb in HP['block_lengths']:
        idx = sb_paths(len(src), nh, Lb, HB, HP['seed'] + Lb)
        sl = src[idx].sum(axis=1); sh = (src - 0.5 * mu)[idx].sum(axis=1); sz = (src - mu)[idx].sum(axis=1)
        res[f'L{Lb}'] = {'p_lower': float((sl <= act).mean()), 'p_lower_shrink50': float((sh <= act).mean()), 'p_lower_zero': float((sz <= act).mean()),
                         'q10': float(np.quantile(sl, 0.10)), 'q50': float(np.quantile(sl, 0.50)), 'q90': float(np.quantile(sl, 0.90))}
    pl = [res[f'L{Lb}']['p_lower'] for Lb in HP['block_lengths']]; res['p_lower_range'] = [min(pl), max(pl)]
    res['judgement_L6'] = '與雜訊相容' if 0.10 <= res['L6']['p_lower'] <= 0.90 else ('落後超出雜訊' if res['L6']['p_lower'] < 0.10 else '超前超出雜訊')
    return res
r50_7 = FC.win_ret(E50raw, G7)
HOLD = {}
for base in ('AB', 'B10'):
    for nm in ORDER:
        for c in ('normal', 'stress'):
            rm = FC.win_ret(NAV[f'{nm}|{base}|{c}'], G7)
            o = {'x50': predictive(rm, r50_7)}
            if c == 'normal': o['xEW'] = predictive(rm, FC.win_ret(EW[base], G7))
            HOLD[f'{nm}|{base}|{c}'] = o
if not SMOKE:
    ref = V7R['RESULTS_HOLD_SPA']['hold']
    for k7, kk in (('AB_V7', 'AB_V7|AB|normal'), ('AB_V7_B10', 'AB_V7|B10|normal'), ('AB_V7_stress', 'AB_V7|AB|stress'), ('AB_V7_B10_stress', 'AB_V7|B10|stress')):
        for Lb in HP['block_lengths']:
            for mine, theirs in (('p_lower', 'p_lower_cumlog'), ('p_lower_shrink50', 'p_lower_cumlog_shrink50'), ('p_lower_zero', 'p_lower_cumlog_zero')):
                r_ = ref[k7]['x50_calendar_v6grid'][f'L{Lb}'][theirs]; m_ = HOLD[kk]['x50'][f'L{Lb}'][mine]
                assert abs(m_ - r_) < 1e-12, (k7, Lb, mine, m_, r_)
        if 'stress' not in k7:
            r_ = ref[k7]['xEW_calendar_v6grid']['L6']['p_lower_cumlog']; m_ = HOLD[kk]['xEW']['L6']['p_lower']; assert abs(m_ - r_) < 1e-12
    VAL['hold_repro_V7_exact'] = True
log('hold done')

# ---------------- (6) SPA（F_long × W_long；14 條 o_v4 成員換 V7 版，同 V7 研究）----------------
SPP = V7R['PREREG']['spa']
D = pd.read_parquet(DATA / 'family_excess.parquet'); MM = pd.read_csv(DATA / 'family_members.csv'); BB = pd.read_parquet(DATA / 'spa_bench.parquet')
pref = 'o_v4_revenue_patch_official/navs_v4.parquet::'
v4mem = [m for m in MM.member_id if m.startswith(pref)]
def v7name(k):
    k = k.replace('V4', 'V7'); return 'O3REP_V7' + k[5:] if k.startswith('O3REP') else k
D7 = D.copy()
for m in v4mem: D7[m] = (FC.win_ret(NV7[v7name(m[len(pref):])], G4) - BB['r0050']).to_numpy()
mem = MM[MM['cov_W_long'] & MM['F_long']].member_id.tolist()
wm = FC.wmask(D.index, *FC.WINDOWS['W_long']); n = int(wm.sum())
def cand_ex(nav): return (FC.win_ret(nav, G4) - BB['r0050']).to_numpy()[wm]
VAL['AB_V7_mine_vs_family_member_maxabs'] = float(np.abs(cand_ex(NAV['AB_V7|AB|normal']) - D7.loc[wm, pref + 'AB_V4|normal'].to_numpy()).max())
VAL['AB_V7_B10_mine_vs_family_member_maxabs'] = float(np.abs(cand_ex(NAV['AB_V7|B10|normal']) - D7.loc[wm, pref + 'AB_V4_B10|normal'].to_numpy()).max())
assert VAL['AB_V7_mine_vs_family_member_maxabs'] < 1e-12 and VAL['AB_V7_B10_mine_vs_family_member_maxabs'] < 1e-12
EXTRA = {'EXTRA::AB_V7|stress': cand_ex(NV7['AB_V7|stress'])}
for nm in MIX:
    for b, c in (('AB', 'normal'), ('B10', 'normal'), ('AB', 'stress')):
        EXTRA[f'integrate_o7::{nm}|{b}|{c}'] = cand_ex(NAV[f'{nm}|{b}|{c}'])
for k_, v_ in EXTRA.items(): assert np.isfinite(v_).all(), k_
X0 = D7.loc[wm, mem].fillna(0.0).to_numpy()
names = mem + list(EXTRA)
X = np.hstack([X0] + [EXTRA[k_][:, None] for k_ in EXTRA])
B = 200 if SMOKE else SPP['B']
SPA = {'N_family': len(names), 'n_windows': n}
def grade(p): return '第 2 級（通過資料窺探校正）' if p <= 0.05 else ('第 2 級（邊緣）' if p <= 0.10 else '第 3 級（有理論支持，但未通過資料窺探校正）')
FOCUS = {'AB_V7|AB|normal': pref + 'AB_V4|normal', 'AB_V7|B10|normal': pref + 'AB_V4_B10|normal', 'AB_V7|AB|stress': 'EXTRA::AB_V7|stress'}
for nm in MIX:
    for b, c in (('AB', 'normal'), ('B10', 'normal'), ('AB', 'stress')): FOCUS[f'{nm}|{b}|{c}'] = f'integrate_o7::{nm}|{b}|{c}'
KEYS = ('N', 'max_t', 'p_SPA_c', 'p_SPA_c_focus', 'p_SPA_l_focus', 'p_SPA_u_focus', 'p_RC_focus', 'focus_rank', 'focus_t', 'p_StepM_focus', 'p_single_focus')
for Lb in SPP['block_lengths']:
    Wc = SP.sb_counts(n, Lb, B, SPP['seed_base'] + 1000 * 0 + Lb)
    if not SMOKE:   # 重現 V7 研究（研究員版）的焦點 p（同族群、同種子）
        ref = V7R['RESULTS_HOLD_SPA']['spa']
        for foc, fid, mm_, XX in (('AB_V7', pref + 'AB_V4|normal', mem, X0), ('AB_V7_B10', pref + 'AB_V4_B10|normal', mem, X0),
                                  ('AB_V7_stress', 'EXTRA::AB_V7|stress', mem + ['EXTRA::AB_V7|stress'], np.hstack([X0, EXTRA['EXTRA::AB_V7|stress'][:, None]]))):
            o = SP.mcp_tests(XX, Wc, Lb, SPP['variance'], focus=mm_.index(fid))
            VAL[f'spa_repro_{foc}_L{Lb}'] = [o['p_SPA_c_focus'], ref[f'{foc}|L{Lb}']['p_SPA_c_focus'], o['focus_rank'], ref[f'{foc}|L{Lb}']['focus_rank']]
            assert abs(o['p_SPA_c_focus'] - ref[f'{foc}|L{Lb}']['p_SPA_c_focus']) < 1e-12 and o['focus_rank'] == ref[f'{foc}|L{Lb}']['focus_rank'], VAL
    Mb = SP.boot_means(Wc, X)
    for k_, fid in FOCUS.items():
        o = SP.mcp_tests(X, Wc, Lb, SPP['variance'], Mb=Mb, focus=names.index(fid))
        SPA[f'{k_}|L{Lb}'] = {kk: o[kk] for kk in KEYS}; SPA[f'{k_}|L{Lb}']['grade'] = grade(o['p_SPA_c_focus'])
    del Mb
    log('SPA', Lb)

# ---------------- (7) 實盤預期：V7 可執行引擎 j 分布上的袖子混合 ----------------
NE = pd.read_parquet(DATA / 'navs_exec7.parquet')
PJ = pd.read_csv(DATA / 'plans_j7.csv', usecols=['strat', 'j', 'seed', 'signal', 'exec'])
PJ = PJ[PJ.seed == 0]
JEND = json.loads((DATA / 'jscan_end_days.json').read_text(encoding='utf-8'))
TIERS = ['30萬', '100萬', '300萬', '1000萬']; VERS = ['FIX', 'GROW']; WS = [1.0, 0.8, 0.7, 0.5, 0.3]
JS = [0, 5] if SMOKE else list(range(13))
STRAT = {'AB': 'AB_V7', 'B10': 'AB_V7_B10'}
def exec_job(arg):
    st, j = arg
    g = PJ[(PJ.strat == STRAT[st]) & (PJ.j == j)]
    de = pd.Timestamp(JEND[f'{STRAT[st]}|J{j}'])
    grid = [pd.Timestamp(x) for x in g.signal] + [de]
    exd = [pd.Timestamp(x) for x in g['exec']]
    out = {}
    for model in ('K5T', 'TICK'):
        for cost in ('normal', 'stress'):
            for tier in TIERS:
                for ver in VERS:
                    col = f'{st}|J{j}|{model}|{cost}|{tier}|{ver}'
                    nav = NE[col].dropna(); ix = nav.index; exi = set(ix.get_loc(d) for d in exd if d in ix)
                    rE = e50_on(ix).pct_change().fillna(0).to_numpy()
                    a, b = ix[0], ix[-1]; e50n = E50.reindex(ix).ffill()
                    for w in WS:
                        nn = nav if w == 1.0 else sleeve(nav, rE, w, L.cost_rates(cost), 'etf', ix, exi)
                        cg, md = ST.cagr_mdd(nn, a, b); c50 = ST.cagr_mdd(e50n, a, b)[0]
                        o = {'CAGR': cg, 'maxDD': md, 'E50_CAGR': c50, 'excess': cg - c50, 't_0050': tstat(ST.xs(nn, E50, grid, a, b))}
                        for lab, (pa, pb) in (('Y19on', ('2019-01-01', b)), ('P19_22', ('2019-01-01', '2022-12-31')), ('P23_26', ('2023-01-01', b)), ('HOLD', ('2024-01-01', b)), ('SEL', ('2006-04-01', '2018-12-31'))):
                            o[f'CAGR_{lab}'] = ST.cagr_mdd(nn, pa, pb)[0]; o[f'E50_CAGR_{lab}'] = ST.cagr_mdd(e50n, pa, pb)[0]
                            o[f't_{lab}'] = tstat(ST.xs(nn, E50, grid, pa, pb))
                        out[(st, j, model, cost, tier, ver, w)] = o
    return out
with get_context('fork').Pool(4) as P:
    parts = P.map(exec_job, [(st, j) for st in STRAT for j in JS], chunksize=1)
EXJ = {}
for p_ in parts: EXJ.update(p_)
if not SMOKE:
    TAB = V7R['RESULTS_EXEC']['table']; mx = 0.0
    for (st, j, model, cost, tier, ver, w), o in EXJ.items():
        if w != 1.0: continue
        r_ = TAB['|'.join((st, f'J{j}', model, cost, tier, ver))]
        mx = max(mx, abs(o['CAGR'] - r_['CAGR']), abs(o['t_0050'] - r_['t_0050']), abs(o['maxDD'] - r_['maxDD']), abs(o['E50_CAGR'] - r_['E50_CAGR']), abs(o['t_Y19on'] - r_['t_Y19on']))
    VAL['exec_w1_vs_V7_RESULTS_EXEC_table_maxabs'] = mx; assert mx < 1e-9, mx
def q(v, p): return float(np.percentile(v, p))
LIVE = {}
for st in STRAT:
    for model in ('K5T', 'TICK'):
        for cost in ('normal', 'stress'):
            for tier in TIERS:
                for ver in VERS:
                    for w in WS:
                        rows = [EXJ[(st, j, model, cost, tier, ver, w)] for j in JS]
                        f = lambda k: [r[k] for r in rows]
                        LIVE[f'{st}|{model}|{cost}|{tier}|{ver}|w{w}'] = {
                            'leg_capital': tier, 'account_capital_wan': float(int(tier.replace('萬', '')) / w),
                            'CAGR_median': q(f('CAGR'), 50), 'CAGR_p10': q(f('CAGR'), 10), 'excess_median_pp': q(f('excess'), 50), 'excess_p10_pp': q(f('excess'), 10),
                            'E50_CAGR_median': q(f('E50_CAGR'), 50), 'maxDD_median': q(f('maxDD'), 50), 't_median': q(f('t_0050'), 50),
                            't_Y19on_median': q(f('t_Y19on'), 50), 'CAGR_Y19on_median': q(f('CAGR_Y19on'), 50), 'E50_CAGR_Y19on_median': q(f('E50_CAGR_Y19on'), 50),
                            'CAGR_P19_22_median': q(f('CAGR_P19_22'), 50), 'E50_CAGR_P19_22_median': q(f('E50_CAGR_P19_22'), 50),
                            'CAGR_P23_26_median': q(f('CAGR_P23_26'), 50), 'E50_CAGR_P23_26_median': q(f('E50_CAGR_P23_26'), 50),
                            'CAGR_HOLD_median': q(f('CAGR_HOLD'), 50), 'E50_CAGR_HOLD_median': q(f('E50_CAGR_HOLD'), 50), 't_HOLD_median': q(f('t_HOLD'), 50),
                            'CAGR_SEL_median': q(f('CAGR_SEL'), 50), 'n_j_beat_0050': int(sum(1 for r in rows if r['excess'] > 0)), 'n_j': len(rows)}
if not SMOKE:
    JSM = V7R['RESULTS_EXEC']['jsumm']; mx = 0.0
    for st in STRAT:
        for model in ('K5T', 'TICK'):
            for cost in ('normal', 'stress'):
                for tier in TIERS:
                    for ver in VERS:
                        r_ = JSM['|'.join((st, model, cost, tier, ver))]; m_ = LIVE[f'{st}|{model}|{cost}|{tier}|{ver}|w1.0']
                        mx = max(mx, abs(r_['CAGR_median'] - m_['CAGR_median']), abs(r_['CAGR_p10'] - m_['CAGR_p10']), abs(r_['t_median'] - m_['t_median']))
    VAL['exec_jsumm_w1_vs_V7_maxabs'] = mx; assert mx < 1e-9, mx
# 同帳戶資金配對（事前登記）：帳戶 100 萬＝全押（100 萬級距）vs 主策略 30%（30 萬級距）；帳戶 1,000 萬＝全押（1,000 萬）vs 30%（300 萬）
PAIRS = {'帳戶100萬': (('100萬', 1.0), ('30萬', 0.3)), '帳戶1000萬': (('1000萬', 1.0), ('300萬', 0.3))}
MATCH = {}
for st in STRAT:
    for cost in ('normal', 'stress'):
        for acct, ((ta, wa), (tb, wb)) in PAIRS.items():
            for ver in VERS:
                row = {}
                for model in ('K5T', 'TICK'):
                    da = np.array([EXJ[(st, j, model, cost, ta, ver, wa)]['CAGR'] for j in JS]); db = np.array([EXJ[(st, j, model, cost, tb, ver, wb)]['CAGR'] for j in JS])
                    dd = db - da
                    row[model] = {'alone_CAGR_median': float(np.median(da)), 'mix30_CAGR_median': float(np.median(db)), 'diff_median_pp': float(np.median(dd)),
                                  'n_j_mix_higher': int((dd > 0).sum()), 'alone_maxDD_median': float(np.median([EXJ[(st, j, model, cost, ta, ver, wa)]['maxDD'] for j in JS])),
                                  'mix30_maxDD_median': float(np.median([EXJ[(st, j, model, cost, tb, ver, wb)]['maxDD'] for j in JS])),
                                  'E50_CAGR_median': float(np.median([EXJ[(st, j, model, cost, ta, ver, wa)]['E50_CAGR'] for j in JS]))}
                su, sl_ = row['K5T']['diff_median_pp'] > 0, row['TICK']['diff_median_pp'] > 0
                row['conclusion'] = '主策略 30% 較高' if (su and sl_) else ('全押較高' if (not su and not sl_) else '無法判斷')
                MATCH[f'{st}|{cost}|{acct}|{ver}'] = row
log('live done')

# ---------------- (8) 選版（只用 SEL）與判定 ----------------
ab = STATS['AB_V7|AB']['periods']
sel = []
for nm in MIX:
    s_ = STATS[f'{nm}|AB']['periods']['SEL']
    ok = (s_['t_0050'] > ab['SEL']['t_0050']) and (ab['SEL']['CAGR'] - s_['CAGR'] <= 1.0) and not nm.startswith('M_2330')
    sel.append({'name': nm, 'SEL_t': s_['t_0050'], 'SEL_CAGR_loss_pp': ab['SEL']['CAGR'] - s_['CAGR'], 'eligible': bool(ok)})
elig = [x for x in sel if x['eligible']]
SELECTED = max(elig, key=lambda x: x['SEL_t'])['name'] if elig else 'AB_V7（主策略單獨）'
JUDGE = {}
for nm in MIX:
    P = STATS[f'{nm}|AB']['periods']
    pa = P['ALL']['vs_AB_V7_cal']['t']; ms, mv = P['SEL']['vs_AB_V7_cal']['mean_pp'], P['VAL']['vs_AB_V7_cal']['mean_pp']
    A = pa >= 1.0 and ms > 0 and mv > 0
    Bp = (P['ALL']['t_0050'] - ab['ALL']['t_0050'] >= 0.2 and P['SEL']['t_0050'] > ab['SEL']['t_0050'] and P['VAL']['t_0050'] > ab['VAL']['t_0050']
          and ab['ALL']['CAGR'] - P['ALL']['CAGR'] <= 1.0 and P['ALL']['maxDD'] >= ab['ALL']['maxDD'])
    if nm.startswith('M_2330'): Bp_eff = False
    else: Bp_eff = Bp
    W = pa > -1.0 and ab['ALL']['CAGR'] - P['ALL']['CAGR'] <= 2.0 and (P['SEL']['t_0050'] > ab['SEL']['t_0050'] or P['VAL']['t_0050'] > ab['VAL']['t_0050'])
    raw = '可用' if (A or Bp_eff) else ('待觀察' if W else '不可用')
    final = '待觀察' if raw == '可用' else raw
    if nm.startswith('M_2330'): final = '事後觀察' if raw != '不可用' else '不可用'
    JUDGE[nm] = {'path_A': bool(A), 'path_B_raw': bool(Bp), 'path_B_effective': bool(Bp_eff), 'watch_cond': bool(W), 'raw': raw, 'final': final,
                 'SPA_L6_p_c_focus': SPA[f'{nm}|AB|normal|L6']['p_SPA_c_focus'],
                 'SPA_grade': (grade(SPA[f'{nm}|AB|normal|L6']['p_SPA_c_focus']) if not nm.startswith('M_2330') else '不適用（含單一事後贏家 2330，不得升級）')}
dC = STATS['AB_V7|AB']['periods']['ALL']['CAGR'] - RV['AB']['CAGR']['ALL']['V4']
VD = V7R['VERDICT_posthoc']['decision']
cond = {'navs_v7_seed_mean_and_rerun_le_1e-9': not SMOKE, 'V7_gate_PASS': bool(VD['gate_pass']), 'V7_data_replication_pass': bool(VD['data_replication_pass(逐格＋淨值)']),
        'V7_exec_replication_pass': bool(VD['exec_replication_all_within_tol']), 'V7_stat_replication_within_tol(超出取不利值)': bool(VD['stat_replication_all_within_tol']),
        'abs_dCAGR_ALL_vs_V4_le_1pp': bool(abs(dC) <= 1.0), 'dCAGR_ALL_vs_V4_pp': dC}
adopted = VD['SPA_focus_L6_adopted']
JUDGE['R_V7'] = {'conditions': cond,
                 'decision': '取代（資料版本）：SOP 正式數字換成 V7' if all(cond[k_] for k_ in ('navs_v7_seed_mean_and_rerun_le_1e-9', 'V7_gate_PASS', 'V7_data_replication_pass', 'V7_exec_replication_pass', 'abs_dCAGR_ALL_vs_V4_le_1pp')) else '不取代',
                 'SPA_L6_mine': {k_: SPA[f'{k_}|L6']['p_SPA_c_focus'] for k_ in ('AB_V7|AB|normal', 'AB_V7|B10|normal', 'AB_V7|AB|stress')},
                 'SPA_L6_adopted_V7_unfavorable': adopted,
                 'grade': '可提議回到第 2 級（需總監確認）' if (adopted['AB'] <= 0.10 and adopted['B10'] <= 0.10) else '維持第 3 級',
                 'hold_L6': HOLD['AB_V7|AB|normal']['x50']['judgement_L6']}
SPA_ADJ = max(abs(v['researcher'] - v['replicator_same_family']) for k_, v in V7R['VERDICT_posthoc']['spa'].items() if isinstance(v, dict) and 'researcher' in v)
JUDGE['SPA_unfavorable_adjustment_pp'] = SPA_ADJ
OUT = {'validation': VAL, 'stats': STATS, 'E50_periods': E50S, 'R_V7_vs_V4_V6': RV, 'hold': HOLD, 'spa': SPA, 'live_exec_j': LIVE, 'live_matched_accounts': MATCH,
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
    RES = json.loads(RJ.read_text(encoding='utf-8'))
    assert 'RESULTS' not in RES
    RES['RESULTS'] = js(OUT); RES['RESULTS_written_at'] = pd.Timestamp.now(tz='UTC').isoformat()
    RJ.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding='utf-8')
    pd.DataFrame({k: v for k, v in NAV.items()}).to_parquet(HERE / 'navs.parquet')
    log('written', JUDGE)

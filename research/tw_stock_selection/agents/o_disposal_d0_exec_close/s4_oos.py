#!/usr/bin/env python3
"""樣本外與全部判定（PREREG 之後只正式跑一次）。乾跑：python3 -I -B s4_oos.py --dry（全部期間換成挑選期，只用來除錯）。
正式：python3 -I -B s4_oos.py"""
import sys, json, math, datetime
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import dlib as D

K = D.K
DRY = '--dry' in sys.argv
RES_F = HERE / 'results.json'
res = json.loads(RES_F.read_text()) if RES_F.exists() else {}
if not DRY:
    assert 'PREREG' in res and res.get('PREREG_sha256') == D.canon_sha(res['PREREG']), 'PREREG 不存在或被改過：先跑 s3b_prereg.py'
    for f, h in res['PREREG']['hashes'].items():
        assert D.sha(f if f.startswith('/') else HERE / f) == h, f'{f} 與 PREREG 雜湊不符'
    assert 'RESULTS' not in res, 'RESULTS 已存在：s4 只能跑一次'
SPEC = json.loads((HERE / 'EXEC_SPEC.json').read_text()) if (HERE / 'EXEC_SPEC.json').exists() else {}
PER = dict(D.PER)
if DRY:
    for k in list(PER): PER[k] = D.PER['SEL']
T0 = PER['FULL'][0]
TIM = [(en, xk) for en in ('O1', 'C1', 'O2') for xk in ('CL1', 'CL', 'OA')]
TN = [f'{a}x{b}' for a, b in TIM]
OUT = {'run_utc': datetime.datetime.utcnow().isoformat() + 'Z', 'dry': DRY}


def P(**kw):
    p = D.default_prm(); p.update(kw); return p


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.median(v)), 4) if v else None


def p10(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.percentile(v, 10)), 4) if v else None


# ======================= 事件表（9 時點） =======================
EV = {}
for (en, xk), nm in zip(TIM, TN):
    e, u, st = D.build_x(dict(entry=en, exit=xk))
    EV[nm] = (e, u, st)

# ======================= 逐筆可執行 =======================
SLOTS = {'30萬': 3e4, '100萬': 1e5, '300萬': 3e5, '1000萬': 1e6}
EX = {}
for nm in TN:
    e = EV[nm][0]
    for sl_nm, sl in SLOTS.items():
        EX[(nm, sl_nm, 'normal')] = D.event_exec(e, sl, P())
    EX[(nm, '100萬', 'stress')] = D.event_exec(e, 1e5, P(stress=True))
    EX[(nm, '30萬', 'stress')] = D.event_exec(e, 3e4, P(stress=True))
    EX[(nm, '300萬', 'stress')] = D.event_exec(e, 3e5, P(stress=True))


def trade_tab(slot='100萬', cost='normal', col='net_x', periods=('SEL', 'VAL', 'FULL', 'NEWREG', 'OOS_o9', 'P1_2006_2012', 'P2_2013_2018', 'P3_2019_2026', 'Y2006_2013', 'Y2014_', '2019+')):
    o = {}
    for pn in periods:
        a, b = PER[pn]
        per_t = {nm: D.S_ev(D.sub(EX[(nm, slot, cost)], a, b), col) for nm in TN}
        means = [per_t[nm].get('mean_pct') for nm in TN]
        o[pn] = {'by_timing': per_t, 'median_of_means_pct': med(means), 'p10_of_means_pct': p10(means), 'min_pct': med([min([m for m in means if m is not None])]) if any(m is not None for m in means) else None,
                 'median_t_month': med([per_t[nm].get('t_month') for nm in TN]), 'n_median': med([per_t[nm].get('n') for nm in TN])}
    return o


TR = {}
for sl in SLOTS:
    TR[f'{sl}|normal'] = trade_tab(sl, 'normal')
for sl in ('30萬', '100萬', '300萬'):
    TR[f'{sl}|stress'] = trade_tab(sl, 'stress')
TR['100萬|normal|vs0050'] = trade_tab('100萬', 'normal', 'net50_x')
TR['100萬|normal|gross_ideal'] = trade_tab('100萬', 'normal', 'gross')
OUT['TRADES'] = TR

# o9 V7X 錨點（E1）與 P2
o9ev, _ = K.build(dict(entry='C1', exit='CL'))
P2 = {}
for sl_nm, sl in (('30萬', 3e4), ('100萬', 1e5), ('1000萬', 1e6)):
    x9 = K.exec_cost_event(o9ev, sl)
    mine = EX[('C1xCL', sl_nm, 'normal')]
    row = {}
    for pn in ('FULL', 'VAL', 'OOS_o9'):
        a, b = PER[pn]
        m9 = D.S_ev(D.sub(x9, a, b), 'net_x'); mm = D.S_ev(D.sub(mine, a, b), 'net_x')
        row[pn] = {'o9_V7X': m9, 'exec_C1xCL': mm, 'reduction_pp': round(m9.get('mean_pct', np.nan) - mm.get('mean_pct', np.nan), 2) if m9.get('n') and mm.get('n') else None}
    P2[sl_nm] = row
OUT['P2'] = P2
OUT['P2']['verdict'] = '成立' if all((P2[s]['FULL']['reduction_pp'] or 0) >= 0.5 for s in P2 if isinstance(P2[s], dict) and 'FULL' in P2[s]) else '不成立'

# ======================= 衛星組合 =======================
NAVS = {}; STATS = {}


def run_sat(tag, tier=1e6, mode='FIX', prm=None, prepay='strict', k=2, K_=10, off=0.0, cfg=None):
    exs = {}
    for (en, xk), nm in zip(TIM, TN):
        if cfg:
            e, u, _ = D.build_x(dict(entry=en, exit=xk, **cfg))
        else:
            e, u, _ = EV[nm]
        nav, st, _ = D.sleeve(e, u, tier_2026=tier, mode=mode, K_=K_, prm=prm or P(), prepay=prepay, pool_k=k, t_start=T0, t_end=PER['FULL'][1], off=off)
        NAVS[f'{tag}|{nm}'] = nav; STATS[f'{tag}|{nm}'] = st
        exs[nm] = D.ex_vs_0050(nav)
    return pd.DataFrame(exs)


def sat_summary(df, tag):
    medser = df.median(axis=1)
    o = {}
    for pn in ('SEL', 'VAL', 'FULL', 'NEWREG', 'P1_2006_2012', 'P2_2013_2018', 'P3_2019_2026', '2019+', 'OOS_o9'):
        a, b = PER[pn]
        per = {nm: D.mstats(df[nm], a, b) for nm in df.columns}
        ts = [per[nm].get('t') for nm in df.columns]; ms = [per[nm].get('mean_pp') for nm in df.columns]
        o[pn] = {'median_t': med(ts), 'p10_t': p10(ts), 'median_mean_pp': med(ms), 'p10_mean_pp': p10(ms),
                 'median_series': D.mstats(medser, a, b), 'by_timing_t': dict(zip(df.columns, ts))}
    st = [STATS[f'{tag}|{nm}'] for nm in df.columns]
    o['ledger'] = {'cash_skip_rate_median': med([s['cash_skip_rate'] for s in st]), 'entered_median': med([s['entered'] for s in st]),
                   'no_slot_median': med([s['no_slot'] for s in st]), 'mean_cash_share_median': med([s['mean_cash_share'] for s in st]),
                   'mean_invested_median': med([s['mean_invested'] for s in st]), 'min_settled_cash_min': min(s['min_settled_cash'] for s in st)}
    return o, medser


SAT = {}; MED = {}
cfgs = [('MAIN', dict()), ('STRESS', dict(prm=P(stress=True))),
        ('FIX30', dict(tier=3e5)), ('FIX300', dict(tier=3e6)), ('FIX1000', dict(tier=1e7)),
        ('GROW100', dict(mode='GROW')), ('GROW300', dict(tier=3e6, mode='GROW')), ('GROW1000', dict(tier=1e7, mode='GROW')),
        ('k1', dict(k=1)), ('k3', dict(k=3)), ('THRESHOLD', dict(prepay='threshold')), ('NONE', dict(prepay='none')),
        ('ODD02', dict(prm=P(odd_spread=0.002))), ('ODD08', dict(prm=P(odd_spread=0.008))), ('KAPPA025', dict(prm=P(kappa=0.25))),
        ('NB_off2', dict(off=0.02, cfg=dict(off=0.02))), ('NB_off05', dict(off=0.005, cfg=dict(off=0.005))),
        ('NB_liq500', dict(cfg=dict(liq=5e6))), ('NB_liq2000', dict(cfg=dict(liq=2e7))), ('NB_K5', dict(K_=5)), ('NB_K20', dict(K_=20)),
        ('FIX30_STRESS', dict(tier=3e5, prm=P(stress=True))), ('FIX300_STRESS', dict(tier=3e6, prm=P(stress=True)))]
for tag, kw in cfgs:
    df = run_sat(tag, **kw)
    SAT[tag], MED[tag] = sat_summary(df, tag)
OUT['SAT'] = SAT

# ======================= 判定 K1～K3、可用 =======================
VAL_t = SAT['MAIN']['VAL']['median_t']
K2v = TR['100萬|normal']['FULL']
K3v = TR['100萬|stress']['VAL']['median_of_means_pct']
DEC = {'K1': {'VAL_median_t': VAL_t, 'VAL_median_series_t': SAT['MAIN']['VAL']['median_series'].get('t'), 'triggered': bool(VAL_t is None or VAL_t < 1.0)},
       'K2': {'FULL_median_pct': K2v['median_of_means_pct'], 'FULL_p10_pct': K2v['p10_of_means_pct'],
              'triggered': bool(K2v['median_of_means_pct'] is None or K2v['median_of_means_pct'] <= 0 or K2v['p10_of_means_pct'] <= 0)},
       'K3': {'VAL_stress_median_pct': K3v, 'triggered': bool(K3v is None or K3v <= 0)}}
use = {sl: TR[f'{sl}|normal']['VAL']['median_of_means_pct'] for sl in ('30萬', '100萬', '300萬')}
DEC['usable_tiers_VAL_pct'] = use
DEC['closed'] = DEC['K1']['triggered'] or DEC['K2']['triggered'] or DEC['K3']['triggered']
DEC['usable'] = (not DEC['closed']) and all(v is not None and v > 0 for v in use.values())
DEC['verdict'] = '結案（不可用）' if DEC['closed'] else ('可用（僅前瞻紙上，真錢 0）' if DEC['usable'] else '不可用（未達可用條件）')
OUT['DECISION'] = DEC

# ======================= 預測 P1、P3～P7 =======================
PR = {}
PR['P1'] = {'VAL_median_t': VAL_t, 'holds(t<1.0)': bool(VAL_t is not None and VAL_t < 1.0)}
PR['P3'] = {'cash_skip_rate_median_FULL': SAT['MAIN']['ledger']['cash_skip_rate_median'],
            'holds(≥15%)': bool((SAT['MAIN']['ledger']['cash_skip_rate_median'] or 0) >= 0.15),
            'by_k': {t: SAT[t]['ledger']['cash_skip_rate_median'] for t in ('k1', 'MAIN', 'k3')}}
ef = D.sub(EX[('C1xCL', '100萬', 'normal')], *PER['FULL']).copy()
cuts = K.run6_cuts(D.U)
ef['first'] = ~ef['second']
ef['post2014'] = ef['date'] >= '2014-01-01'
ef['lim10'] = ef['date'] >= '2015-06-01'
ef['whole_lot'] = ef['nl'] >= 1
ef['second_f'] = ef['second']
ef['mm20'] = ef['match_min'] >= 20
ef['run6_top'] = ef['run6'] >= cuts[3]
ef['pre2014'] = ef['date'] < '2014-01-01'
ef['vdrop_big'] = ef['vdrop'] <= ef['vdrop'].median()
ps = K.path_split(ef.reset_index(drop=True))
PR['P4'] = {f: D.cluster_t(ef.dropna(subset=['net_x']), f, 'net_x') for f in ('first', 'post2014', 'lim10', 'whole_lot')}
PR['P4']['late_minus_early'] = {'early_pct': round(float(ps['early'].mean() * 100), 2), 'late_pct': round(float(ps['late'].mean() * 100), 2),
                                'diff_pp': round(float(ps['late_m_early'].mean() * 100), 2),
                                't_month': D.tstat(ps['late_m_early'].groupby(ps['date'].dt.to_period('M')).mean())}
e5 = ef[ef['match_min'].isin([5.0, 20.0])]
PR['P5'] = {'second': D.cluster_t(ef.dropna(subset=['net_x']), 'second_f', 'net_x'),
            'mm20_vs_mm5': D.cluster_t(e5.dropna(subset=['net_x']), 'mm20', 'net_x'),
            'run6_top': D.cluster_t(ef.dropna(subset=['net_x']), 'run6_top', 'net_x'),
            'pre2014': D.cluster_t(ef.dropna(subset=['net_x']), 'pre2014', 'net_x'),
            'vdrop_big': D.cluster_t(ef.dropna(subset=['net_x', 'vdrop']), 'vdrop_big', 'net_x')}
MV = pd.read_parquet(D.AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet')
mcols = [c for c in MV.columns if c.startswith('B10|J') and c.endswith('|K5T|normal|100萬|FIX')]
e50m = D.monthly(pd.Series(D.E50, index=D.IDX))
mm = pd.concat([D.monthly(MV[c].dropna()) for c in mcols], axis=1).mean(axis=1)
mex = (mm - e50m.reindex(mm.index)).dropna()
sat = MED['MAIN']
com = pd.concat([sat.rename('sat'), mex.rename('main')], axis=1).dropna()
c19 = com[com.index >= '2019-01-01']; cv = com[(com.index >= PER['VAL'][0]) & (com.index <= PER['VAL'][1])]
PR['P6'] = {'corr_FULL': round(float(com.corr().iloc[0, 1]), 3), 'corr_2019+': round(float(c19.corr().iloc[0, 1]), 3) if len(c19) > 3 else None,
            'corr_VAL': round(float(cv.corr().iloc[0, 1]), 3) if len(cv) > 3 else None, 'n_months': int(len(com)),
            'corr_by_timing_FULL': {nm: round(float(pd.concat([pd.Series(D.ex_vs_0050(NAVS[f'MAIN|{nm}'])), mex], axis=1).dropna().corr().iloc[0, 1]), 3) for nm in TN}}
PR['P6']['holds(≥0.4)'] = bool(PR['P6']['corr_FULL'] >= 0.4)
PR['P7'] = {'C1xCL': D.S_ev(D.sub(EX[('C1xCL', '100萬', 'normal')], *PER['NEWREG']), 'net_x'), 'median_9': TR['100萬|normal']['NEWREG']['median_of_means_pct']}
PR['P7']['holds(≤0)'] = bool(PR['P7']['median_9'] is not None and PR['P7']['median_9'] <= 0)
PR['mechanism_falsified'] = {'FULL_median_pct': K2v['median_of_means_pct'], 'FULL_stress_median_pct': TR['100萬|stress']['FULL']['median_of_means_pct'],
                             'falsified': bool((K2v['median_of_means_pct'] or 0) <= 0 or (TR['100萬|stress']['FULL']['median_of_means_pct'] or 0) <= 0)}
OUT['PRED'] = PR

# ======================= 和主策略、0050、EW 比較 =======================
def pair(a, b):
    d = (a - b).dropna()
    return {'mean_pp': round(float(d.mean() * 100), 3), 't': D.tstat(d), 'IR_ann': round(float(d.mean() / d.std(ddof=1) * math.sqrt(12)), 3) if d.std(ddof=1) > 0 else None, 'n': int(len(d))}


satm = (MED['MAIN'] + e50m.reindex(MED['MAIN'].index))   # 衛星月報酬（中位數時點）
CMP = {}
for pn in ('FULL', 'VAL', '2019+'):
    a, b = PER[pn]
    msk = lambda s: s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]
    A_ = msk((0.8 * mm + 0.2 * satm).dropna()); B_ = msk(mm.reindex(A_.index))
    C_ = msk((0.8 * e50m + 0.2 * satm).dropna()); E_ = msk(e50m.reindex(C_.index))
    CMP[pn] = {'main80_d0_20_vs_main100': pair(A_, B_), '0050_80_d0_20_vs_0050': pair(C_, E_),
               'ann_main80_d0_20': round(float((1 + A_).prod() ** (12 / max(len(A_), 1)) - 1) * 100, 2), 'ann_main100': round(float((1 + B_).prod() ** (12 / max(len(B_), 1)) - 1) * 100, 2),
               'ann_0050_80_d0_20': round(float((1 + C_).prod() ** (12 / max(len(C_), 1)) - 1) * 100, 2), 'ann_0050': round(float((1 + E_).prod() ** (12 / max(len(E_), 1)) - 1) * 100, 2),
               'main_vs_0050': D.mstats(msk(mex))}
ewn = pd.Series(np.exp(D.LEW), index=D.IDX)
CMP['EW_pool_vs_0050'] = {pn: D.mstats(D.ex_vs_0050(ewn), *PER[pn]) for pn in ('FULL', 'VAL', '2019+')}
# 主欄衛星（中位數月報酬）年化、最大回撤
def nav_from_m(m):
    return (1 + m).cumprod()
for pn in ('FULL', 'VAL', 'SEL', '2019+'):
    a, b = PER[pn]
    s = satm[(satm.index >= pd.Timestamp(a)) & (satm.index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))].dropna()
    e = e50m.reindex(s.index)
    if len(s) > 2:
        ns, ne = nav_from_m(s), nav_from_m(e)
        CMP.setdefault('sat_main_median_timing', {})[pn] = {'ann': round(float(ns.iloc[-1] ** (12 / len(s)) - 1) * 100, 2), 'ann_0050': round(float(ne.iloc[-1] ** (12 / len(e)) - 1) * 100, 2),
                                                          'mdd_monthly': round(float((ns / ns.cummax() - 1).min()) * 100, 1), 'mdd_0050_monthly': round(float((ne / ne.cummax() - 1).min()) * 100, 1)}
# C1×CL 日頻（代表時點）年化與日頻最大回撤
for pn in ('FULL', 'VAL'):
    CMP.setdefault('sat_main_C1xCL_daily', {})[pn] = D.per_stats(NAVS['MAIN|C1xCL'], *PER[pn])
OUT['COMPARE'] = CMP

# ======================= G1～G8 =======================
G = {}
st_tr = TR['100萬|stress']['FULL']
G['G1'] = {'trade_FULL_median_pct': K2v['median_of_means_pct'], 'trade_FULL_p10_pct': K2v['p10_of_means_pct'],
           'trade_FULL_stress_median_pct': st_tr['median_of_means_pct'], 'trade_FULL_stress_p10_pct': st_tr['p10_of_means_pct'],
           'sat_FULL_median_pp': SAT['MAIN']['FULL']['median_mean_pp'], 'sat_FULL_p10_pp': SAT['MAIN']['FULL']['p10_mean_pp'],
           'sat_FULL_stress_median_pp': SAT['STRESS']['FULL']['median_mean_pp'], 'sat_FULL_stress_p10_pp': SAT['STRESS']['FULL']['p10_mean_pp']}
G['G1']['pass'] = bool(all((G['G1'][k] or -1) > 0 for k in G['G1']))
G['G2'] = {'FULL_median_t': SAT['MAIN']['FULL']['median_t'], '2019+_median_t': SAT['MAIN']['2019+']['median_t'],
           'pass': bool((SAT['MAIN']['FULL']['median_t'] or 0) >= 3.0 and (SAT['MAIN']['2019+']['median_t'] or 0) >= 2.0)}
sm = MED['MAIN'][(MED['MAIN'].index >= PER['FULL'][0])]
yr = sm.groupby(sm.index.year).sum().sort_values(ascending=False)
ex3 = sm[~sm.index.year.isin(yr.index[:3])]
segs = {pn: SAT['MAIN'][pn]['median_series'].get('mean_pp') for pn in ('P1_2006_2012', 'P2_2013_2018', 'P3_2019_2026')}
nb = {t: SAT[t]['FULL']['median_series'].get('mean_pp') for t in ('NB_off2', 'NB_off05', 'NB_liq500', 'NB_liq2000', 'NB_K5', 'NB_K20')}
G['G4'] = {'segments_mean_pp': segs, 'drop_best3': {'years': [int(y) for y in yr.index[:3]], 't': D.tstat(ex3), 'mean_pp': round(float(ex3.mean() * 100), 3)},
           'neighborhood_FULL_mean_pp': nb, 'neighborhood_FULL_t': {t: SAT[t]['FULL']['median_series'].get('t') for t in nb},
           'yearly_sum_pp': {int(k): round(float(v * 100), 2) for k, v in sm.groupby(sm.index.year).sum().items()}}
G['G4']['pass'] = bool(all((v or -1) > 0 for v in segs.values()) and (G['G4']['drop_best3']['t'] or 0) >= 2.0 and all((v or -1) > 0 for v in nb.values()))
gate = json.loads((HERE / 'data_gate.json').read_text())
G['G5'] = {'gate_pass': gate['gate']['pass'], 'coverage': gate['gate']['price_coverage'], 'announce_before_decision': gate['gate']['announce_before_decision_all'],
           'note': '價格含 o_infra_v3 下市股價；2019-05～2023 V7 上市身分面板已補下市股；R3／R5 改成交規則後無同日收盤決策', 'pass': bool(gate['gate']['pass'])}
G['G6'] = {'status': '待另一位研究員只看 frozen_spec.txt＋EXEC_SPEC.json 重寫；本題不自評', 'targets': {
    'FULL_C1xCL_net_x_mean_pct': TR['100萬|normal']['FULL']['by_timing']['C1xCL'].get('mean_pct'),
    'FULL_C1xCL_n': TR['100萬|normal']['FULL']['by_timing']['C1xCL'].get('n'),
    'VAL_MAIN_C1xCL_t': SAT['MAIN']['VAL']['by_timing_t'].get('C1xCL'), 'VAL_MAIN_median_t': VAL_t}, 'pass': None}
g7 = {t: {'mean_pp': SAT[t]['FULL']['median_series'].get('mean_pp'), 't': SAT[t]['FULL']['median_series'].get('t')} for t in ('GROW100', 'GROW300', 'GROW1000')}
G['G7'] = {'GROW_FULL': g7, 'pass': bool(all((v['mean_pp'] or -1) > 0 and (v['t'] or 0) >= 2.0 for v in g7.values()))}
G['G8'] = {'status': '未開始（結案則不啟動）', 'pass': None}
OUT['G'] = G

# ======================= 穩健性：分年、分產業、去最大貢獻股 =======================
RB = {}
x = ef.copy(); x['yy'] = x['date'].dt.year
RB['yearly_C1xCL_net_x'] = {int(k): {'n': int(len(g)), 'mean_pct': round(float(g['net_x'].mean() * 100), 2)} for k, g in x.groupby('yy')}
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str})
ind = dict(zip(sl['code'], sl['industry']))
x['ind'] = x['code'].map(ind).fillna('未知（已下市或缺）')
RB['industry_C1xCL_net_x（現行產業別，非 PIT）'] = {k: {'n': int(len(g)), 'mean_pct': round(float(g['net_x'].mean() * 100), 2)} for k, g in x.groupby('ind') if len(g) >= 15}
drop = {}
for nm in TN:
    xx = D.sub(EX[(nm, '100萬', 'normal')], *PER['FULL'])
    top = xx.groupby('code')['net_x'].sum().idxmax()
    drop[nm] = {'dropped': top, 'mean_pct': round(float(xx[xx['code'] != top]['net_x'].mean() * 100), 2)}
RB['drop_top_stock_FULL'] = {'by_timing': drop, 'median_pct': med([v['mean_pct'] for v in drop.values()])}
OUT['ROBUST'] = RB

# ======================= G3：SPA＋DSR =======================
sys.path.insert(0, str(D.AG / 'o_main_spa_breaks'))
import fam_common as FC, spa_lib as SPA
FD = pd.read_parquet(FC.DATA / 'family_excess.parquet'); MM_ = pd.read_csv(FC.DATA / 'family_members.csv')
a, b = FC.WINDOWS['W_long']; mskw = FC.wmask(FD.index, a, b)
mem = MM_[MM_['F_long'] & MM_['cov_W_long']]['member_id'].tolist()
e50w = FC.win_ret(FC.e50_nav())
o9n = pd.read_parquet(D.AG / 'o_disposal_official_fullhist' / 'navs_fixcorrected.parquet')
mine = {}
for c in o9n.columns:
    mine['O9|' + c] = (FC.win_ret(o9n[c]) - e50w).reindex(FD.index)
dr = pd.DataFrame({k: v.pct_change() for k, v in NAVS.items() if k.startswith('MAIN|')}).median(axis=1).fillna(0.0)
focus_nav = (1 + dr).cumprod()
NAVS['FOCUS_median_daily'] = focus_nav
for k, nav in NAVS.items():
    mine['THIS|' + k] = (FC.win_ret(nav) - e50w).reindex(FD.index)
MY = pd.DataFrame(mine)
X = pd.concat([FD.loc[mskw, mem], MY.loc[mskw]], axis=1).fillna(0.0)
names = list(X.columns); foc = names.index('THIS|FOCUS_median_daily')
if not DRY:
    Wc = SPA.sb_counts(len(X), 6, 10000, 20261103)
    sp = SPA.mcp_tests(X.to_numpy(), Wc, 6, 'pr', focus=foc)
    et = SPA.eff_trials(X.to_numpy())
    sr_all = (X.mean(axis=0) / X.std(axis=0, ddof=1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xf = X.iloc[:, foc].to_numpy()
    G3 = {'family_size': len(names), 'n_F_long': len(mem), 'n_o9': int(o9n.shape[1]), 'n_this': int(sum(1 for n in names if n.startswith('THIS|'))),
          'n_windows': int(len(X)), 'SPA_c_focus_p': sp.get('p_SPA_c_focus'), 'SPA_l_focus_p': sp.get('p_SPA_l_focus'), 'SPA_c_family_p': sp.get('p_SPA_c'),
          'focus_rank': sp.get('focus_rank'), 'focus_t_window': sp.get('focus_t'), 'max_t': sp.get('max_t'), 'eff_trials': et,
          'DSR_LiJi': SPA.dsr(xf, sr_all.to_numpy(), et['M_eff_LiJi']), 'DSR_Nyholt': SPA.dsr(xf, sr_all.to_numpy(), et['M_eff_Nyholt'])}
    G3['pass'] = bool((G3['SPA_c_focus_p'] or 1) < 0.05 and G3['DSR_LiJi']['DSR'] >= 0.95)
else:
    G3 = {'family_size': len(names), 'dry': True}
OUT['G']['G3'] = G3
OUT['n_variants_this_topic'] = int(sum(1 for k in NAVS if k != 'FOCUS_median_daily'))

# ======================= 輸出 =======================
if DRY:
    print(json.dumps({'DECISION': DEC, 'n_navs': len(NAVS)}, ensure_ascii=False, default=str, indent=1)[:3000])
    sys.exit(0)
res['RESULTS'] = OUT
RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame({k: v for k, v in NAVS.items() if k.startswith(('MAIN|', 'STRESS|', 'GROW', 'FOCUS'))}).to_parquet(HERE / 'navs.parquet')
tr = []
for (nm, sl, cost), x_ in EX.items():
    if sl == '100萬':
        y = x_[['code', 'date', 'exit_date', 'announce', 'second', 'nth', 'match_min', 'regime', 'raw', 'ew', 'e50', 'gross', 'ret_x', 'net_x', 'net50_x', 'fill', 'nl', 'no', 'capped']].copy()
        y['timing'] = nm; y['cost'] = cost; tr.append(y)
pd.concat(tr).to_csv(HERE / 'trades_all.csv.gz', index=False)
pd.DataFrame({'sat_median_excess': MED['MAIN'], 'main_V7X_excess': mex}).to_csv(HERE / 'monthly_excess.csv')
print(json.dumps({'DECISION': DEC, 'G': {k: v.get('pass') for k, v in G.items()}}, ensure_ascii=False, default=str, indent=1))

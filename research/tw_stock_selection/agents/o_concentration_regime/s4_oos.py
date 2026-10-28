#!/usr/bin/env python3
"""步驟 4：樣本外（驗證期 2023-01～2026-10）＋結構前對照（2006～2018，只檢驗 P1、P5）＋描述延伸＋穩健性。
必須在 PREREG 寫入 results.json 之後執行；開跑先核對 PREREG 的 sha256（腳本、引擎、raw/、面板、挑選期中間檔）。一次跑完，不得改規格。"""
import sys, json, time, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cr_lib as L, cr_eval as E

SMOKE = '--smoke' in sys.argv     # 冒煙測試：全部輸入截到 2023-01-11，VAL／PRE 換成挑選期內的假期間，不寫 results.json
CUT = pd.Timestamp('2023-01-11') if SMOKE else None
RES = L.load_json()
if SMOKE:
    L.VAL = ('2021-01-01', '2022-12-31'); L.PRE = ('2019-06-01', '2020-12-31')
    it0 = json.loads((L.HERE / 'iterations.json').read_text(encoding='utf-8'))
    PR = dict(versions=[o['spec'] for r in it0['rounds'] for o in r['full']], frozen=dict(main_version='V5', wbar_frozen=0.479), spec_sha256='smoke')
else:
    PR = RES.get('PREREG'); assert PR, 'PREREG 不存在：不得計算樣本外'
    bad = {f: h for f, h in PR['hashes'].items() if L.sha(f) != h}
    assert not bad, f'PREREG 雜湊不符：{list(bad)[:5]}'
    assert pd.Timestamp(PR['timestamp_utc']) < pd.Timestamp.utcnow(), 'PREREG 時戳異常'
def cut(x):
    return x if CUT is None else x[x.index <= CUT]
T0 = time.time()
OUT = dict(started_utc=pd.Timestamp.utcnow().isoformat(), prereg_sha_of_spec=PR['spec_sha256'])

SPECS = {s['id']: s for s in PR['versions']}
MAIN_ID = PR['frozen']['main_version']
WBAR_FROZEN = PR['frozen']['wbar_frozen']
SIM0 = '2018-03-01'

# ---------------- 全期訊號基礎表；與挑選期檔逐格核對 ----------------
P = L.Panel(end=CUT)
e50 = cut(L.e50_series())
dates = P.official_dates('2004-03-01')
W = L.add_e50(P.windows(dates), e50)
W.to_parquet(L.DATA / ('W_smoke.parquet' if SMOKE else 'W_full.parquet'))
Ws = pd.read_parquet(L.DATA / 'W_sel.parquet')
ov = W.reindex(Ws.index)
OUT['W_full_vs_W_sel_maxabs'] = {c: float((ov[c] - Ws[c]).abs().max()) for c in ['rEW', 'top5', 'r50', 'd']}
assert max(OUT['W_full_vs_W_sel_maxabs'].values()) < 1e-12, OUT['W_full_vs_W_sel_maxabs']
plan = pd.read_csv(L.RAW / 'plan_V4.csv', parse_dates=['signal'])
OUT['decision_dates_vs_plan_V4_full'] = dict(only_mine=[str(d.date()) for d in sorted(set(d for d in dates if pd.Timestamp('2006-04-01') <= d <= pd.Timestamp('2026-10-12')) - set(plan.signal))],
                                             only_plan=[str(d.date()) for d in sorted(set(plan.signal) - set(dates))])

NV4 = cut(pd.read_parquet(L.RAW / 'navs_v4.parquet')); NV6 = cut(pd.read_parquet(L.RAW / 'navs_v6.parquet'))
NIO6 = cut(pd.read_parquet(L.RAW / 'navs_integrate_o6.parquet')); NJ = cut(pd.read_parquet(L.RAW / 'navs_jscan_v6.parquet'))
NT = cut(pd.read_parquet(L.RAW / 'navs_exec_tiers_abavg.parquet'))
MAIN = {'AB': NV4['AB_V4|normal'].dropna(), 'AB_stress': NV4['AB_V4|stress'].dropna(), 'B10': NV4['AB_V4_B10|normal'].dropna(),
        'V6': NV6['AB_V6|normal'].dropna(), 'EW': NV4['AB_V4|EW'].dropna()}
END = MAIN['AB'].index[-1]           # 2026-10-12
Wv = W[W['end'] <= END]


def ev(spec, mainnav, per, cost='normal', wbar=None, e=e50, Wx=None, sim0=SIM0):
    Wx = Wv if Wx is None else Wx
    sd_end = Wx['end'][(Wx.index >= pd.Timestamp(per[0])) & (Wx.index <= pd.Timestamp(per[1]))].max()
    sd_end = min(sd_end, mainnav.index[-1])
    B = E.build(Wx, spec, mainnav, e, sim0, cost, wbar=wbar, per=per)
    return E.evaluate(B, Wx, mainnav, e, per, sd_end, label=spec['id']), B


def short(o):
    return dict(n=o['n'], switches=o['switches'], wbar=o['wbar'], frac_C=o['frac_C'], CAGR_rot=o['CAGR_rot'], MDD_rot=o['MDD_rot'], CAGR_static=o['CAGR_static'],
                CAGR_main=o['CAGR_main'], MDD_main=o['MDD_main'], CAGR_0050=o['CAGR_0050'], MDD_0050=o['MDD_0050'],
                vs_static_mean_pp=o['vs_static']['mean_pp'], vs_static_t=o['vs_static']['t'], vs_static_IR=o['vs_static']['IR_ann'],
                vs_static_boot_state=[o['vs_static']['boot_state']['p5'] * 100, o['vs_static']['boot_state']['p95'] * 100],
                vs_0050_mean_pp=o['vs_0050']['mean_pp'], vs_0050_t=o['vs_0050']['t'], vs_main_mean_pp=o['vs_main']['mean_pp'], vs_main_t=o['vs_main']['t'],
                vs_main_CAGR_diff_pp=o['vs_main']['CAGR_diff_pp'], TE_rot_vs_0050=o['TE_rot_vs_0050_ann'], TE_main_vs_0050=o['TE_main_vs_0050_ann'],
                TE_vs_main=o['TE_vs_main_ann'], whip_ratio=o['whipsaw']['ratio_whip_to_gain'], gain_long_pp=o['whipsaw']['gain_long_ge6_pp'],
                net_short_pp=o['whipsaw']['net_short_lt6_pp'], switch_cost_total_pp=o['whipsaw']['switch_cost_total_pp'])


# ---------------- 1. 全部版本：挑選期重算（應與 iterations 相同）、驗證期（兩種靜態對照）----------------
ALLV = {}
it = json.loads((L.HERE / 'iterations.json').read_text(encoding='utf-8'))
itmap = {o['label']: o for r in it['rounds'] for o in r['full']}
for vid, spec in SPECS.items():
    if spec.get('derived'): continue
    oS, BS = ev(spec, MAIN['AB'], L.SEL)
    oV_real, BV = ev(spec, MAIN['AB'], L.VAL)
    oV_froz, _ = ev(spec, MAIN['AB'], L.VAL, wbar=BS['wbar'])
    rep = abs(oS['vs_static']['mean_pp'] - itmap[vid]['vs_static']['mean_pp']) if vid in itmap else None
    ALLV[vid] = dict(spec=spec, SEL=short(oS), VAL_static_real=short(oV_real), VAL_static_frozen=short(oV_froz), sel_repro_absdiff_pp=rep,
                     VAL_whipsaw_episodes=oV_real['whipsaw']['episodes'])
    if vid == MAIN_ID:
        OMS, OMV, OMVF, BMS, BMV = oS, oV_real, oV_froz, BS, BV
OUT['versions'] = ALLV
assert all(v['sel_repro_absdiff_pp'] is None or v['sel_repro_absdiff_pp'] < 1e-9 for v in ALLV.values()), '挑選期重算與 iterations 不同'

# 衍生版：0050 50% 核心＋主版本 50% 衛星（w＝0.5＋0.5×w_rot）
spec_m = SPECS[MAIN_ID]
def build_core(per, mainnav=MAIN['AB'], cost='normal', wbar=None):
    S, st, w = E.version_states(Wv, spec_m)
    w2 = w.map(lambda x: None if x is None else 0.5 + 0.5 * x)
    dec = [d for d in Wv.index if d >= pd.Timestamp(SIM0) and w2.get(d) is not None]
    wb = float(L.sel(w2.dropna().astype(float), per).mean()) if wbar is None else wbar
    rot, lg = L.rotate(mainnav, e50, L.exec_map(dec, [w2[d] for d in dec], mainnav.index), cost)
    sta, lgs = L.rotate(mainnav, e50, L.exec_map(dec, [wb] * len(dec), mainnav.index), cost)
    B = dict(S=S, st=st, w=w2, wbar=wb, rot=rot, sta=sta, log=lg, logs=lgs, dec=dec)
    sd_end = min(Wv['end'][(Wv.index >= pd.Timestamp(per[0])) & (Wv.index <= pd.Timestamp(per[1]))].max(), mainnav.index[-1])
    return E.evaluate(B, Wv, mainnav, e50, per, sd_end, label='CORE50_SAT')
OUT['core50_satellite'] = {'SEL': short(build_core(L.SEL)), 'VAL': short(build_core(L.VAL))}

# ---------------- 2. 對照組（驗證期與挑選期）----------------
def comp_stats(nav, per, name):
    sd = list(Wv.index) + [END]
    r = L.sel(L.win_ret(nav, sd), per); e = L.on_cal(e50, nav.index); r50 = L.win_ret(e, sd).reindex(r.index)
    rm = L.win_ret(MAIN['AB'], sd).reindex(r.index)
    a = r.index[0]; b = min(Wv['end'][(Wv.index >= pd.Timestamp(per[0])) & (Wv.index <= pd.Timestamp(per[1]))].max(), nav.index[-1])
    cg, md = L.cagr_mdd(nav, a, b)
    return dict(name=name, CAGR=cg, MDD=md, vs_0050_mean_pp=float((r - r50).mean() * 100), vs_0050_t=L.tstat(r - r50),
                vs_main_mean_pp=float((r - rm).mean() * 100), vs_main_t=L.tstat(r - rm), TE_vs_0050=L.te_ann(r, r50), n=int(len(r)))
mix30 = L.rotate(MAIN['AB'], e50, L.exec_map([d for d in Wv.index if d >= pd.Timestamp(SIM0)], [0.3] * len([d for d in Wv.index if d >= pd.Timestamp(SIM0)]), MAIN['AB'].index), 'normal')[0]
e50nav = L.on_cal(e50, MAIN['AB'].index)
# 引擎核對：用本題 rotate() 重現 integrate_o6 的 M_E50_30|AB|normal（AB_V6 主腿、plan_V6 exec 日、integrate_o6 成本口徑）
p6x = pd.read_csv(L.RAW / 'plan_V6.csv', parse_dates=['signal', 'exec'])
nv6 = MAIN['V6']; ex6 = {d: 0.3 for d in p6x['exec'] if d in nv6.index}
chk, _ = L.rotate(nv6, e50, ex6, 'io6')
ref = NIO6['M_E50_30|AB|normal'].dropna().reindex(chk.index)
OUT['engine_check_vs_integrate_o6_M_E50_30'] = dict(maxrel=float((chk / chk.iloc[0] / (ref / ref.iloc[0]) - 1).abs().max()), n=int(len(chk)))
COMP = {}
for per_name, per in (('SEL', L.SEL), ('VAL', L.VAL)):
    COMP[per_name] = [comp_stats(e50nav, per, '0050'), comp_stats(MAIN['AB'], per, 'AB_V4（主策略）'), comp_stats(MAIN['B10'], per, 'AB_V4_B10'),
                      comp_stats(NIO6['M_E50_30|AB|normal'].dropna(), per, 'M_E50_30（integrate_o6，AB_V6 基底）'),
                      comp_stats(mix30, per, 'MIX30（本題引擎：0050 30%＋AB_V4 70%）'), comp_stats(MAIN['EW'], per, '同池等權（AB_V4|EW）'),
                      comp_stats(BMV['rot'], per, f'{MAIN_ID} 輪動'), comp_stats(BMV['sta'] if per_name == 'VAL' else BMS['sta'], per, '同平均權重靜態混合')]
OUT['comparators'] = COMP

# ---------------- 3. 主版本：成本、主腿口徑、端點 ----------------
MV = {}
for nm, mn, cost in (('stress', MAIN['AB_stress'], 'stress'), ('B10', MAIN['B10'], 'normal'), ('V6_main', MAIN['V6'], 'normal'), ('zero_switch_cost', MAIN['AB'], 'zero')):
    o_r, _ = ev(spec_m, mn, L.VAL, cost); o_f, _ = ev(spec_m, mn, L.VAL, cost, wbar=WBAR_FROZEN); o_s, _ = ev(spec_m, mn, L.SEL, cost)
    MV[nm] = dict(VAL_static_real=short(o_r), VAL_static_frozen=short(o_f), SEL=short(o_s))
OUT['main_variants'] = MV

def endpoint(o):
    x = (o['_series']['r_rot'] - o['_series']['r_sta']); x50 = o['_series']['r_rot'] - o['_series']['r50']
    xs = x.sort_values(ascending=False)
    res = dict(full_mean_pp=float(x.mean() * 100), t=L.tstat(x))
    for k in (1, 3):
        y = x.drop(xs.index[:k]); res[f'drop_best{k}_mean_pp'] = float(y.mean() * 100); res[f'drop_best{k}_t'] = L.tstat(y)
        y50 = x50.drop(x50.sort_values(ascending=False).index[:k]); res[f'vs0050_drop_best{k}_mean_pp'] = float(y50.mean() * 100)
    w24 = [d for d in x.index if d <= pd.Timestamp('2024-01-31') and Wv.loc[d, 'end'] >= pd.Timestamp('2024-01-31')]
    res['window_2024_01'] = [str(d.date()) for d in w24]
    if w24:
        res['drop_2024_01_mean_pp'] = float(x.drop(w24).mean() * 100); res['drop_2024_01_t'] = L.tstat(x.drop(w24))
        res['vs0050_drop_2024_01_mean_pp'] = float(x50.drop(w24).mean() * 100)
        res['window_2024_01_detail'] = dict(rot=float(o['_series']['r_rot'][w24[0]]), static=float(o['_series']['r_sta'][w24[0]]), r0050=float(o['_series']['r50'][w24[0]]),
                                            main=float(o['_series']['r_main'][w24[0]]), state=o['_series']['st'][w24[0]])
    yr = x.groupby(x.index.year).agg(['sum', 'count']); res['by_year_sum_pp'] = {int(k): float(v * 100) for k, v in yr['sum'].items()}
    yr50 = x50.groupby(x50.index.year).sum(); res['vs0050_by_year_sum_pp'] = {int(k): float(v * 100) for k, v in yr50.items()}
    return res
OUT['endpoints'] = {'VAL_static_real': endpoint(OMV), 'VAL_static_frozen': endpoint(OMVF), 'SEL': endpoint(OMS)}
OUT['main_VAL_static_real_full'] = E.strip(OMV); OUT['main_VAL_static_frozen_full'] = E.strip(OMVF); OUT['main_SEL_full'] = E.strip(OMS)

# ---------------- 4. 換股日 j＝0～12 ----------------
TCAL = pd.DatetimeIndex(sorted(set(pd.read_csv(L.RAW / 'taiex_twii.csv', parse_dates=['date'])['date'])))
SPECIAL = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12')}
p6 = pd.read_csv(L.RAW / 'plan_V6.csv', parse_dates=['signal'])
ANCH = sorted(set(pd.Period(d, 'M') for d in p6.signal) | set(pd.period_range('2016-12', '2026-10', freq='M')))
def jdays(j, nav_ix):
    out = []
    G = set(nav_ix)
    for m in ANCH:
        if m < pd.Period('2016-12', 'M'): continue
        D = SPECIAL.get(str(m), pd.Timestamp(m.year, m.month, 10)); k = TCAL.searchsorted(D)
        if k >= len(TCAL): continue
        k += j
        if k >= len(TCAL): continue
        s = TCAL[k]
        if s not in G: continue
        if nav_ix.searchsorted(s, side='right') >= len(nav_ix): continue
        out.append(s)
    return sorted(set(out))
JS = {}
for j in range(13):
    nav = NJ[f'AB_V6|normal|J{j}'].dropna()
    dj = jdays(j, nav.index)
    Wj = L.add_e50(P.windows(dj), e50)
    Wj = Wj[Wj['end'] <= nav.index[-1]]
    rr = {}
    for per_name, per in (('SEL', L.SEL), ('VAL', L.VAL)):
        o, B = ev(spec_m, nav, per, Wx=Wj)
        rr[per_name] = short(o)
    o_st, _ = ev(spec_m, NJ[f'AB_V6|stress|J{j}'].dropna(), L.VAL, 'stress', Wx=Wj)
    rr['VAL_stress'] = short(o_st)
    JS[f'J{j}'] = rr
def jmed(per, key): return float(np.median([JS[k][per][key] for k in JS]))
OUT['jscan'] = dict(by_j=JS, median=dict(VAL_vs_static_mean_pp=jmed('VAL', 'vs_static_mean_pp'), VAL_vs_0050_mean_pp=jmed('VAL', 'vs_0050_mean_pp'),
                                         VAL_vs_static_t=jmed('VAL', 'vs_static_t'), VAL_vs_0050_t=jmed('VAL', 'vs_0050_t'), VAL_switches=jmed('VAL', 'switches'),
                                         SEL_vs_static_mean_pp=jmed('SEL', 'vs_static_mean_pp'), SEL_vs_0050_mean_pp=jmed('SEL', 'vs_0050_mean_pp'),
                                         VAL_stress_vs_static_mean_pp=jmed('VAL_stress', 'vs_static_mean_pp'), VAL_stress_vs_0050_mean_pp=jmed('VAL_stress', 'vs_0050_mean_pp')),
                    n_pos_VAL_vs_static=int(sum(JS[k]['VAL']['vs_static_mean_pp'] > 0 for k in JS)), n_pos_VAL_vs_0050=int(sum(JS[k]['VAL']['vs_0050_mean_pp'] > 0 for k in JS)),
                    note='主腿＝navs_jscan_v6 的 AB_V6|normal|Jj；訊號在 J 日重算（2024-01 的 J 日在本面板無資料，同池等權與前 5 大占比用 2023-12-29 的值，0050 用真實值）。')

# ---------------- 5. 可執行 4 級距 × FIX／GROW ----------------
TI = {}
for tier in ('30萬', '100萬', '300萬', '1000萬'):
    for g in ('GROW', 'FIX'):
        nav = NT[f'main|{tier}|{g}|A'].dropna()
        o_r, _ = ev(spec_m, nav, L.VAL); o_s, _ = ev(spec_m, nav, L.SEL)
        TI[f'{tier}|{g}'] = dict(VAL=short(o_r), SEL=short(o_s))
OUT['exec_tiers'] = TI

# ---------------- 6. 0052／006208 取代 0050 持有腿 ----------------
def etf_ext(code):
    s = cut(L.etf_long(code)); last = s.index[-1]
    ext = e50[e50.index > last]
    if len(ext):
        base = L.on_cal(e50, pd.DatetimeIndex([last])).iloc[0]
        s = pd.concat([s, s.iloc[-1] * ext / base])
    return s
ETF = {}
for code in ('0052', '006208'):
    es = etf_ext(code)
    o_r, _ = ev(spec_m, MAIN['AB'], L.VAL, e=es); o_s, _ = ev(spec_m, MAIN['AB'], L.SEL, e=es)
    # 對 0050 的超額：用 0050 而非該 ETF
    sd = list(Wv.index) + [END]
    def x0050(o):
        r = o['_series']['r_rot']; r50 = L.win_ret(e50nav, sd).reindex(r.index); return dict(mean_pp=float((r - r50).mean() * 100), t=L.tstat(r - r50))
    ETF[code] = dict(VAL=short(o_r), SEL=short(o_s), VAL_vs_real_0050=x0050(o_r), SEL_vs_real_0050=x0050(o_s), note='etf_long 到 2026-10-08，之後以 0050 報酬延伸；分割已修補')
OUT['etf_leg_sensitivity'] = ETF

# ---------------- 7. 結構前對照 2006～2018（只檢驗 P1、P5）----------------
Wpre = W[W['end'] <= END]
PRE = {}
for kind, Lk in (('S1', 6), ('S1', 12), ('S1', 3), ('S1skip', 12), ('S2', 6)):
    S = L.signal(Wpre, kind, Lk); y = Wpre['d']
    for per_name, per in (('PRE_2006_2018', L.PRE), ('SEL_2019_2022', L.SEL), ('VAL_2023_2026', L.VAL)):
        m = (Wpre.index >= pd.Timestamp(per[0])) & (Wpre.index <= pd.Timestamp(per[1])) & S.notna().to_numpy()
        r = L.ols_hac(y[m].to_numpy(), S[m].to_numpy(), lag=max(Lk, 3))
        PRE[f'{kind}_L{Lk}|{per_name}'] = r
dpre = L.sel(Wpre['d'], L.PRE)
OUT['P1_by_period'] = PRE
OUT['d_acf_by_period'] = {pn: {f'lag{l}': float(L.sel(Wpre['d'], per).autocorr(l)) for l in (1, 2, 3, 6, 12)} for pn, per in (('PRE', L.PRE), ('SEL', L.SEL), ('VAL', L.VAL))}
o_pre, B_pre = ev(spec_m, MAIN['AB'], L.PRE, sim0='2006-04-01')
o_pre_f, _ = ev(spec_m, MAIN['AB'], L.PRE, sim0='2006-04-01', wbar=WBAR_FROZEN)
OUT['P5_pre_structure'] = dict(static_real=short(o_pre), static_frozen=short(o_pre_f))

# ---------------- 8. P3 行為、P6 ----------------
st_all = BMV['st']
OUT['P3'] = dict(states_2021={str(d.date()): st_all.get(d) for d in st_all.index if d.year == 2021},
                 state_window_covering_2024_01={str(d.date()): st_all.get(d) for d in st_all.index if pd.Timestamp('2023-12-01') <= d <= pd.Timestamp('2024-02-29')},
                 switches_per_year_VAL=OMV['switches_per_year'], switches_per_year_SEL=OMS['switches_per_year'],
                 whipsaw_VAL={k: v for k, v in OMV['whipsaw'].items() if k != 'episodes'}, whipsaw_SEL={k: v for k, v in OMS['whipsaw'].items() if k != 'episodes'})
OUT['P6'] = dict(wbar_VAL=OMV['wbar'], wbar_SEL=OMS['wbar'], TE_rot_vs_0050_VAL=OMV['TE_rot_vs_0050_ann'], TE_main_vs_0050_VAL=OMV['TE_main_vs_0050_ann'],
                 TE_reduction_VAL=1 - OMV['TE_rot_vs_0050_ann'] / OMV['TE_main_vs_0050_ann'],
                 TE_rot_vs_0050_SEL=OMS['TE_rot_vs_0050_ann'], TE_main_vs_0050_SEL=OMS['TE_main_vs_0050_ann'], TE_reduction_SEL=1 - OMS['TE_rot_vs_0050_ann'] / OMS['TE_main_vs_0050_ann'])

# ---------------- 9. 機制描述 P4 與 2023～2026 描述延伸 ----------------
inst = pd.read_csv(L.RAW / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign', 'trust'], parse_dates=['date'])
if SMOKE: inst = inst[inst.date <= CUT]
FW = inst.pivot_table(index='date', columns='code', values='foreign', aggfunc='sum').reindex(columns=P.C.columns)
TW = inst.pivot_table(index='date', columns='code', values='trust', aggfunc='sum').reindex(columns=P.C.columns)
CR = P.C.reindex(FW.index); FV = FW * 1000 * CR; TV = TW * 1000 * CR; VV = P.V.reindex(FW.index)
rows = []
for k in range(len(Wv)):
    a, b = Wv.index[k], Wv['end'].iloc[k]
    if a < pd.Timestamp('2021-10-08'): continue
    idx, v = P.pool(a); top = idx[np.argsort(-v)[:5]]
    m = (FV.index > a) & (FV.index <= b)
    if m.sum() == 0: continue
    net = FV.loc[m].iloc[:, idx].sum(); netT = FV.loc[m].iloc[:, top].sum().sum()
    tnet = TV.loc[m].iloc[:, idx].sum(); tnetT = TV.loc[m].iloc[:, top].sum().sum()
    val = VV.loc[m].iloc[:, idx].sum(); valT = VV.loc[m].iloc[:, top].sum().sum()
    rows.append(dict(start=a, FC=float(netT / net.abs().sum()), TC=float(tnetT / tnet.abs().sum()) if tnet.abs().sum() > 0 else np.nan,
                     VS=float(valT / val.sum()) if val.sum() > 0 else np.nan, d=float(Wv['d'].iloc[k]), state=st_all.get(a), top5=','.join(P.C.columns[j] for j in top)))
F4 = pd.DataFrame(rows).set_index('start'); F4['excess'] = F4.FC - F4.VS
def p4(df):
    return dict(n=int(len(df)), mean_FC_minus_VS_C=float(df[df.state == 'C'].excess.mean()) if (df.state == 'C').any() else None,
                mean_FC_minus_VS_D=float(df[df.state == 'D'].excess.mean()) if (df.state == 'D').any() else None,
                n_C=int((df.state == 'C').sum()), n_D=int((df.state == 'D').sum()), corr_excess_same_window_d=float(df.excess.corr(df.d)),
                corr_excess_next_window_d=float(df.excess.corr(df.d.shift(-1))), mean_trust_TC_C=float(df[df.state == 'C'].TC.mean()) if (df.state == 'C').any() else None,
                mean_trust_TC_D=float(df[df.state == 'D'].TC.mean()) if (df.state == 'D').any() else None)
OUT['P4'] = dict(all_2021_10_2026_09=p4(F4), SEL_part=p4(L.sel(F4, L.SEL)), VAL_part=p4(L.sel(F4, L.VAL)),
                 rows=[dict(start=str(i.date()), FC=round(r.FC, 4), VS=round(r.VS, 4), excess=round(r.excess, 4), TC=None if not np.isfinite(r.TC) else round(r.TC, 4),
                            d_pp=round(r.d * 100, 2), state=r.state, top5=r.top5) for i, r in F4.iterrows()],
                 definition='FC＝視窗內外資對前 5 大（決策日 60 日均成交值）的淨買超金額 ÷ 池內 300 檔外資淨買超金額絕對值總和；VS＝前 5 大成交值占池內成交值；TC＝投信同口徑。')
# 描述延伸：2019～2026 逐月表、狀態分段、ETF 贏家
sd_all = list(Wv.index) + [END]
r_main_all = L.win_ret(MAIN['AB'], sd_all)
tab = []
S2 = L.signal(Wv, 'S2', 6); S16 = L.signal(Wv, 'S1', 6)
for d_, r in L.sel(Wv, ('2019-01-01', '2026-10-31')).iterrows():
    tab.append(dict(start=str(d_.date()), r0050=round(r.r50 * 100, 2), rEW=round(r.rEW * 100, 2), d=round(r.d * 100, 2), main=round(float(r_main_all.get(d_, np.nan)) * 100, 2),
                    top5=round(r.top5 * 100, 2), top1=round(r.top1 * 100, 2), S2_L6_pp=round(float(S2[d_]) * 100, 2), S1_L6_pp=round(float(S16[d_]) * 100, 2), state=st_all.get(d_)))
OUT['describe_2019_2026'] = dict(monthly=tab, segments_main_state=[dict(start=str(st_all.dropna().index[a].date()), months=int(b - a + 1), state=l)
                                                                   for a, b, l in L.runs(list(L.sel(st_all.dropna(), ('2019-01-01', '2026-10-31'))))])
dval = L.sel(Wv['d'], L.VAL)
OUT['d_summary_VAL'] = dict(n=int(len(dval)), mean_pp=float(dval.mean() * 100), sd_pp=float(dval.std() * 100), t=L.tstat(dval), frac_pos=float((dval > 0).mean()), cum_log=float(dval.sum()))
OUT['d_summary_PRE'] = dict(n=int(len(dpre)), mean_pp=float(dpre.mean() * 100), sd_pp=float(dpre.std() * 100), t=L.tstat(dpre), frac_pos=float((dpre > 0).mean()), cum_log=float(dpre.sum()))
tr = pd.read_csv(L.RAW / 'etf_tr_m.csv.gz', dtype={'code': str})
if SMOKE: tr = tr[tr.ym <= '2022-12']
codes = ['0050', '0052', '006208', '00733', '0056', '00878', '00881', '00692', '00713', '00919', '00929', '00940', '00980A', '00981A', '00982A', '00984A', '00985A']
T = tr[tr.code.isin(codes)].pivot_table(index='ym', columns='code', values='tr')
H = T / T.shift(1); H = H.where(~(H < 0.5), H * (1 / H).round()); H = (H - 1) * 100; H = H.iloc[1:]; H.index = pd.PeriodIndex(H.index, freq='M')
half = (1 + H / 100).groupby([H.index.year, (H.index.month - 1) // 6]).apply(lambda g: g.prod(min_count=1)) - 1
OUT['etf_winner_rotation_halfyear_all'] = {f'{y}H{h + 1}': {c: round(float(v) * 100, 1) for c, v in row.dropna().sort_values(ascending=False).items()} for (y, h), row in half.iterrows()}
ex = pd.read_csv(L.RAW / 'ex_tsmc.csv', parse_dates=['date'])
if SMOKE: ex = ex.iloc[:5]
OUT['ex_tsmc_note'] = dict(span=[str(ex.date.min().date()), str(ex.date.max().date())], cum_r0050=float((1 + ex.r0050 / 100).prod() - 1), cum_ex_tsmc=float((1 + ex.ex_tsmc / 100).prod() - 1),
                           cum_r2330=float((1 + ex.r2330 / 100).prod() - 1), w2330_last=float(ex.w2330.iloc[-1]))

# ---------------- 10. 檢定力、有效檢定數 ----------------
x = OMV['_series']['r_rot'] - OMV['_series']['r_sta']
OUT['power'] = dict(n=int(len(x)), sd_pp=float(x.std() * 100), MDE80_pp=float(2.8 * x.std() / np.sqrt(len(x)) * 100),
                    note='MDE80＝2.8×sd÷√n（雙尾 5%、檢定力 80%）。')
OUT['effective_tests'] = dict(selection_versions=len([s for s in SPECS.values() if not s.get('derived')]), j_versions=13, exec_tiers=8, etf_legs=2, cost_and_leg_variants=4,
                              note='判定只用主版本；挑選經過 8 版，Bonferroni(8) 單尾 5% 門檻 t≈2.5。j、級距、ETF、成本是穩健性，不是額外挑選。')

# ---------------- 11. 判定（事前規則，機械套用）----------------
def verdict():
    sel = OMS; vr = OMV; vf = OMVF
    a = (sel['vs_static']['t'] >= 1.0) and (sel['vs_0050']['t'] >= 1.5)
    vs_static_val = min(vr['vs_static']['mean_pp'], vf['vs_static']['mean_pp'])
    b = (vs_static_val > 0) and (vr['vs_0050']['mean_pp'] > 0) and (vr['switches'] >= 2)
    stv = MV['stress']
    jm = OUT['jscan']['median']
    c = (min(stv['VAL_static_real']['vs_static_mean_pp'], stv['VAL_static_frozen']['vs_static_mean_pp']) > 0 and stv['VAL_static_real']['vs_0050_mean_pp'] > 0
         and jm['VAL_vs_static_mean_pp'] > 0 and jm['VAL_vs_0050_mean_pp'] > 0)
    d = min(OUT['endpoints']['VAL_static_real']['drop_best3_mean_pp'], OUT['endpoints']['VAL_static_frozen']['drop_best3_mean_pp']) >= 0
    undecid = (sel['switches'] < 2) or (vr['switches'] < 2)
    bad = (vs_static_val <= 0) or (vr['vs_0050']['mean_pp'] <= 0)
    if undecid: v = '無法判斷'
    elif bad: v = '不可用'
    elif a and b and c and d: v = '待觀察'
    else: v = '不可用（未達待觀察門檻）'
    # 機制（P1／P5）
    s6 = PRE['S1_L6|SEL_2019_2022']; s12 = PRE['S1_L12|SEL_2019_2022']; p6_ = PRE['S1_L6|PRE_2006_2018']; p12 = PRE['S1_L12|PRE_2006_2018']
    p1_sel = ((s6['beta'] > 0 and s6['t_hac'] >= 1.5 and s12['beta'] > 0) or (s12['beta'] > 0 and s12['t_hac'] >= 1.5 and s6['beta'] > 0))
    p1_struct = p1_sel and (p6_['beta'] < 0.5 * s6['beta']) and (p12['beta'] < 0.5 * s12['beta'])
    p5_fail = (vs_static_val <= 0) or (vr['switches'] < 2)
    return dict(verdict=v, a=a, b=b, c=c, d=d, undecidable=undecid, bad=bad, vs_static_val_min_pp=vs_static_val,
                P1_selection_supported=p1_sel, P1_structure_supported=p1_struct, P5_topic_not_established=p5_fail)
OUT['VERDICT'] = verdict()
OUT['finished_utc'] = pd.Timestamp.utcnow().isoformat(); OUT['runtime_s'] = time.time() - T0
if SMOKE:
    L.save_json(OUT, Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/cr_smoke.json'))
else:
    RES['OOS'] = OUT
    L.save_json(RES)
print(json.dumps(OUT['VERDICT'], ensure_ascii=False))
print('runtime', OUT['runtime_s'])

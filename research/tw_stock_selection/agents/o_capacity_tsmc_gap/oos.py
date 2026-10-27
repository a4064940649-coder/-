#!/usr/bin/env python3
"""樣本外：驗證期（2019-04-11～2026-10-12）與保留期（2024-01～2026-10-12）各跑一次，規格照 PREREG，不再修改。
輸出 results_oos.json，並把摘要併入 results.json 的 OOS。"""
import sys, json, time, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ocore as O
K, V = O.K, O.V
T0 = time.time()
RJ = O.HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R and 'PREREG_sha256' in R, 'PREREG 不存在：先執行 freeze.py'
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改過'
assert hashlib.sha256((O.HERE / 'iterations.json').read_bytes()).hexdigest() == R['PREREG']['iterations_json_sha256'], 'iterations.json 被改過'
POOL = R['PREREG']['選定股池']; WSEL = R['PREREG']['選定台積電權重 w']
assert 'OOS' not in R, 'OOS 只能跑一次'
def log(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)
out = {'PREREG_sha256': R['PREREG_sha256'], 'pool': POOL, 'w': WSEL}
PER_KEYS = ('SEL', 'VAL', 'VAL_A', 'HOLD', 'S1', 'S2', 'S3', 'ALL')

def strip(st): return {k: v for k, v in st.items() if k != 'yearly'}

def full(univ='U300', w=0.0, n=10, cost=O.NORMAL):
    lo, hi = O.UNIV[univ]
    S = O.V2R(lo, hi)
    pl = S.plan(n)
    t0, t1, sd = S.bounds()
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    nav, lg = O.simulate_w(O.tsmc_plan(pl, w), *cost, t0, t1)
    P = O.periods(sd)
    st = {k: O.segstat(nav, ew, sd, *P[k]) for k in PER_KEYS}
    return dict(S=S, pl=pl, nav=nav, ew=ew, sd=sd, st=st, P=P, turnover=lg['turnover'])

# ================= B. 股池 × w0 =================
U = {}
out['universe'] = {}
for n in (10, 30):
    base = None
    for u in ('U300', 'U51_300', 'U101_300', 'U100'):
        r = full(u, 0.0, n); U[(u, n)] = r
        if u == 'U300': base = r
        Wb = O.window_returns(base['nav'], base['sd']); Eb = O.window_returns(base['ew'], base['sd'])
        Wv = O.window_returns(r['nav'], r['sd']); Ev = O.window_returns(r['ew'], r['sd'])
        d = {'stats': {k: strip(v) for k, v in r['st'].items()}, 'yearly': r['st']['ALL']['yearly'],
             'paired_vs_U300': {k: O.paired_t(Wv, Wb, *r['P'][k][:2]) for k in PER_KEYS},
             'paired_excessEW_vs_U300': {k: O.paired_t(Wv - Ev, Wb - Eb, *r['P'][k][:2]) for k in PER_KEYS}}
        rs = full(u, 0.0, n, O.STRESS)
        d['stress'] = {k: strip(v) for k, v in rs['st'].items()}
        out['universe'][f'{u}|n{n}'] = d
        log('univ', u, n, {k: (d['stats'][k]['CAGR'], d['stats'][k]['t_0050'], d['stats'][k]['t_EW']) for k in ('SEL', 'VAL', 'HOLD', 'ALL')}, d['paired_vs_U300']['VAL'])

# ================= C. 台積電權重 =================
out['tsmc'] = {}
TS = {}
for w in (0.0, 0.10, 0.20, 0.30):
    r = full(POOL, w) if w > 0 else U[(POOL, 10)]
    TS[w] = r
    rs = full(POOL, w, cost=O.STRESS)
    Wv = O.window_returns(r['nav'], r['sd']); W0 = O.window_returns(TS[0.0]['nav'], r['sd'])
    out['tsmc'][f'w{int(w*100)}'] = {'stats': {k: strip(v) for k, v in r['st'].items()}, 'yearly': r['st']['ALL']['yearly'],
                                     'stress': {k: strip(v) for k, v in rs['st'].items()},
                                     'paired_vs_w0': {k: O.paired_t(Wv, W0, *r['P'][k][:2]) for k in PER_KEYS}}
    log('tsmc', w, {k: (r['st'][k]['CAGR'], r['st'][k]['maxDD'], r['st'][k]['t_0050'], r['st'][k]['TE_ann'], r['st'][k]['IR']) for k in ('SEL', 'VAL', 'HOLD', 'ALL')})

# 對照：0050 x%＋主策略 (1−x)%（每次換股日再平衡；0050 交易成本 0.0855%＋稅 0.1%＋滑價 0.1% 只算賣出、買進 0.1855%）
def mix_0050(nav, sd, x):
    e = K.E50.reindex(nav.index)
    rm, re_ = nav.pct_change().fillna(0).to_numpy(), e.pct_change().fillna(0).to_numpy()
    exd = set(nav.index.get_indexer(pd.DatetimeIndex(sd[:-1])) + 1)
    a, b = 1 - x, x; out_ = np.empty(len(nav))
    for i in range(len(nav)):
        if i > 0: a *= 1 + rm[i]; b *= 1 + re_[i]
        if i in exd:
            tot = a + b; tb = x * tot; tr = b - tb
            c = (0.000855 + 0.001 + 0.001) * max(tr, 0) + (0.000855 + 0.001) * max(-tr, 0)
            tot -= c; a, b = (1 - x) * tot, x * tot
        out_[i] = a + b
    return pd.Series(out_, index=nav.index)
out['mix0050'] = {}
for x in (0.2, 0.3):
    r0 = TS[0.0]; mv = mix_0050(r0['nav'], r0['sd'], x)
    out['mix0050'][f'0050_{int(x*100)}'] = {k: strip(O.segstat(mv, r0['ew'], r0['sd'], *r0['P'][k])) for k in PER_KEYS}
    out['mix0050'][f'0050_{int(x*100)}']['yearly'] = O.segstat(mv, r0['ew'], r0['sd'], *r0['P']['ALL'])['yearly']
log('mix', {k: {kk: (vv['CAGR'], vv['t_0050']) for kk, vv in v.items() if kk in ('VAL', 'ALL')} for k, v in out['mix0050'].items()})
# 0050 與 2330 本身
P0 = TS[0.0]['P']
def tsmc_cagr(ad, bd):
    ix = O.TSMC_IDX.index
    s0 = ix[ix < pd.Timestamp(ad)][-1]; s1 = ix[ix <= pd.Timestamp(bd)][-1]
    return round(float((O.TSMC_IDX[s1] / O.TSMC_IDX[s0]) ** (365.25 / (s1 - s0).days) - 1) * 100, 2)
out['bench'] = {k: {'E50_CAGR': TS[0.0]['st'][k]['E50_CAGR'], 'TSMC_CAGR': tsmc_cagr(P0[k][2], P0[k][3])} for k in PER_KEYS}

# ================= A. 名次歸因 =================
S0, pl10, pl30 = TS[0.0]['S'], TS[0.0]['pl'], U[('U300', 30)]['pl']
EV = {10: O.event_table(S0, pl10), 30: O.event_table(U[('U300', 30)]['S'], pl30)}
SP = O.SPLIT
RANGES = {'SEL': ('2006-01-01', str((SP - pd.Timedelta(days=1)).date())), 'VAL': (str(SP.date()), '2026-12-31'),
          'VAL_A': (str(SP.date()), '2023-12-31'), 'HOLD': ('2024-01-01', '2026-12-31'),
          '2006_2013': ('2006-01-01', '2013-12-31'), '2014_2020': ('2014-01-01', '2020-12-31'), '2021_2026': ('2021-01-01', '2026-12-31'),
          'ALL': ('2006-01-01', '2026-12-31')}
ind = V.SL['industry']
ELEC = ('半導體', '電子', '光電', '通信', '電腦', '資訊')
out['attrib'] = {}
for n in (10, 30):
    df = EV[n]
    df['ind'] = df['code'].map(lambda c: ind.get(c, '（無）') if isinstance(ind.get(c, None), str) else '（無）')
    df['elec'] = df['ind'].map(lambda s: any(k in s for k in ELEC))
    r = {}
    yr = df.assign(y=df['d'].dt.year).groupby(['y', 'bucket']).size().unstack(fill_value=0)
    yr.columns = [O.BNAME[c] for c in yr.columns]
    r['rank_dist_by_year'] = {int(y): {**{k: int(v) for k, v in row.items()}, 'median_rank': float(df[df['d'].dt.year == y]['rank'].median())} for y, row in yr.iterrows()}
    for lab, (lo, hi) in RANGES.items():
        x = df[(df['d'] >= pd.Timestamp(lo)) & (df['d'] <= pd.Timestamp(hi))]
        r[lab] = {'ex_pool': O.group_summary(x, 'ex_pool'), 'ex_bucket': O.group_summary(x, 'ex_bucket'),
                  'mml_ex_pool': O.mml_stats(O.mid_minus_large(x, 'ex_pool')), 'mml_ex_bucket': O.mml_stats(O.mid_minus_large(x, 'ex_bucket')),
                  'clustered_ex_pool': O.clustered_reg(x, 'ex_pool'), 'clustered_ex_bucket': O.clustered_reg(x, 'ex_bucket'),
                  'share_by_bucket': {O.BNAME[b]: round(float((x['bucket'] == b).mean()), 3) for b in range(4)},
                  'bucket_ew_minus_pool_ew_pp': {O.BNAME[b]: round(float((x[x['bucket'] == b]['ew_bucket'] - x[x['bucket'] == b]['ew_pool']).mean()) * 100, 3) for b in range(4)}}
    # 逐年 mml
    s = O.mid_minus_large(df, 'ex_pool')
    r['mml_by_year'] = {int(y): {'mean_pp': round(float(g.mean()) * 100, 2), 'n': int(len(g))} for y, g in s.groupby(s.index.year)}
    # 穩健性：去掉最大貢獻股（101～300 組對超額貢獻最大的代碼）、去掉前 5% 事件、電子／非電子
    rob = {}
    for lab in ('SEL', 'VAL', 'ALL'):
        lo, hi = RANGES[lab]; x = df[(df['d'] >= pd.Timestamp(lo)) & (df['d'] <= pd.Timestamp(hi))]
        top_mid = x[x['bucket'] >= 2].groupby('code')['ex_pool'].sum().idxmax()
        top_lg = x[x['bucket'] < 2].groupby('code')['ex_pool'].sum().idxmax()
        q = x['ex_pool'].quantile(0.95)
        rob[lab] = {'drop_top_mid_code': top_mid, 'mml_drop_top_mid': O.mml_stats(O.mid_minus_large(x[x['code'] != top_mid])),
                    'drop_top_large_code': top_lg, 'mml_drop_top_large': O.mml_stats(O.mid_minus_large(x[x['code'] != top_lg])),
                    'mml_wins_top5pct': O.mml_stats(O.mid_minus_large(x.assign(ex_pool=x['ex_pool'].clip(upper=q)))),
                    'mml_elec': O.mml_stats(O.mid_minus_large(x[x['elec']])), 'mml_nonelec': O.mml_stats(O.mid_minus_large(x[~x['elec']])),
                    'mml_ex_construction': O.mml_stats(O.mid_minus_large(x[x['ind'] != '建材營造業'])),
                    'mml_stress_cost': None}
        rob[lab]['industry_top_by_bucket'] = {O.BNAME[b]: x[x['bucket'] == b]['ind'].value_counts().head(5).to_dict() for b in range(4)}
    r['robust'] = rob
    out['attrib'][f'n{n}'] = r
    log('attrib', n, {k: r[k]['mml_ex_pool'] for k in ('SEL', 'VAL', 'HOLD', 'ALL')})
# 壓力成本的事件表（n10）
EVs = O.event_table(S0, pl10, cost=O.STRESS)
for lab in ('SEL', 'VAL', 'ALL'):
    lo, hi = RANGES[lab]; x = EVs[(EVs['d'] >= pd.Timestamp(lo)) & (EVs['d'] <= pd.Timestamp(hi))]
    out['attrib']['n10']['robust'][lab]['mml_stress_cost'] = O.mml_stats(O.mid_minus_large(x))

# 全池訊號效力（各名次組內前 1/5 減組內等權，第 1～3 個月）SEL／VAL
ex = [p[0] for p in pl10]; sig = [p[3] for p in pl10]
def oo(t, t2, idx):
    g = K.Cn[t, idx] / K.On[t, idx] * np.prod(K.GROW[t + 1:t2 + 1, idx], axis=0)
    ok = K.TRADED[t2, idx] & np.isfinite(K.On[t2, idx])
    return np.where(ok, g * K.On[t2, idx] / np.where(ok, K.Cn[t2, idx], 1.0), np.nan) - 1
eff = {}
for i, d in enumerate(sig):
    cs = S0.cross_section(d); el = np.where(cs['el'])[0]; bk = O.bucket_of(cs['rank'][el])
    for b in range(4):
        m = el[bk == b]
        if len(m) < 15: continue
        scb = cs['sc'][m]; topq = m[scb >= np.quantile(scb, 0.8)]
        for h in (1, 2, 3):
            if i + h >= len(ex): continue
            ta, tb = ex[i + h - 1], ex[i + h]
            eff.setdefault((b, h), []).append((d, np.nanmean(oo(ta, tb, topq)) - np.nanmean(oo(ta, tb, m))))
out['signal_efficacy'] = {}
for lab in ('SEL', 'VAL', 'HOLD'):
    lo, hi = RANGES[lab]
    out['signal_efficacy'][lab] = {O.BNAME[b]: {f'month{h}': O.mml_stats(pd.Series([v for d, v in eff[(b, h)] if pd.Timestamp(lo) <= d <= pd.Timestamp(hi)])) for h in (1, 2, 3)} for b in range(4)}
log('eff VAL', {k: v['month1'] for k, v in out['signal_efficacy']['VAL'].items()})

# ================= D. 2330 歸因 =================
r0 = TS[0.0]
Wd = pd.DataFrame({'S': O.window_returns(r0['nav'], r0['sd']), 'E50': O.window_returns(K.E50, r0['sd']),
                   'TSMC': O.window_returns(O.TSMC_IDX, r0['sd']), 'EW': O.window_returns(r0['ew'], r0['sd'])})
Wd['y'] = Wd['S'] - Wd['E50']; Wd['x_tsmc'] = Wd['TSMC'] - Wd['E50']; Wd['x_ew'] = Wd['EW'] - Wd['E50']
REG = {'2006_2013': ('2006-01-01', '2013-12-31'), '2014_2020': ('2014-01-01', '2020-12-31'), '2021_2026': ('2021-01-01', '2026-12-31'),
       'SEL': RANGES['SEL'], 'VAL': RANGES['VAL'], 'HOLD': RANGES['HOLD'], '2006_2019Q1': RANGES['SEL'], 'ALL': RANGES['ALL']}
out['tsmc_regression'] = {}
for lab, (lo, hi) in REG.items():
    x = Wd[(Wd.index >= pd.Timestamp(lo)) & (Wd.index <= pd.Timestamp(hi))].dropna()
    a = O.ols(x['y'], x[['x_tsmc']]); b = O.ols(x['y'], x[['x_ew']]); c = O.ols(x['y'], x[['x_tsmc', 'x_ew']])
    out['tsmc_regression'][lab] = {'on_tsmc': {k: v for k, v in a.items() if k != 'resid'}, 'on_ew': {k: v for k, v in b.items() if k != 'resid'},
                                   'on_both': {k: v for k, v in c.items() if k != 'resid'}, 'mean_y_pp': round(float(x['y'].mean()) * 100, 3),
                                   'mean_x_tsmc_pp': round(float(x['x_tsmc'].mean()) * 100, 3), 'mean_x_ew_pp': round(float(x['x_ew'].mean()) * 100, 3), 'n': int(len(x))}
# 用 VAL 期 β（單因子、雙因子）扣除後 HOLD 殘差
xv = Wd[(Wd.index >= SP)].dropna(); xh = Wd[(Wd.index >= pd.Timestamp('2024-01-01'))].dropna(); xs = Wd[Wd.index < SP].dropna()
bv1 = out['tsmc_regression']['VAL']['on_tsmc']['coef']['x_tsmc']
bv2 = out['tsmc_regression']['VAL']['on_both']['coef']
def resid_stats(x, b1=None, b2=None):
    e = x['y'] - (b1 * x['x_tsmc'] if b1 is not None else 0) - ((b2['x_tsmc'] * x['x_tsmc'] + b2['x_ew'] * x['x_ew']) if b2 is not None else 0)
    return {'mean_pp': round(float(e.mean()) * 100, 3), 't': round(K.tstat(e), 2), 'n': int(len(e))}
out['residual_excess'] = {
    'note': '殘差月超額＝(主策略−0050) − β×(2330−0050)［− β_ew×(等權−0050)］；β 用 VAL 期估計（樣本內），另列 SEL 的原始月超額作比較',
    'SEL_raw': resid_stats(xs), 'VAL_raw': resid_stats(xv), 'HOLD_raw': resid_stats(xh),
    'HOLD_minus_bVAL_tsmc': resid_stats(xh, b1=bv1), 'VAL_minus_bVAL_tsmc': resid_stats(xv, b1=bv1),
    'HOLD_minus_bVAL_both': resid_stats(xh, b2=bv2), 'SEL_minus_bSEL_tsmc': resid_stats(xs, b1=out['tsmc_regression']['SEL']['on_tsmc']['coef']['x_tsmc']),
    'beta_VAL_tsmc': bv1, 'beta_VAL_both': bv2}
out['hold_monthly'] = [{'signal': str(d.date()), 'main_minus_0050_pp': round(r_['y'] * 100, 2), 'tsmc_minus_0050_pp': round(r_['x_tsmc'] * 100, 2),
                        'beta_part_pp': round(bv1 * r_['x_tsmc'] * 100, 2), 'ew_minus_0050_pp': round(r_['x_ew'] * 100, 2),
                        'resid_pp': round((r_['y'] - bv1 * r_['x_tsmc']) * 100, 2)} for d, r_ in xh.iterrows()]
log('reg', {k: (v['on_tsmc']['coef'], v['on_tsmc']['R2']) for k, v in out['tsmc_regression'].items()}, out['residual_excess'])

# 台積電在 0050 權重（粗略）2019～2026，校對 2025-09～2026-09 實際值
out['tsmc_weight_proxy'] = O.tsmc_weight_proxy(S0, sig, 2019, 2026)
try:
    xt = pd.read_csv(K.X / 'ex_tsmc.csv', parse_dates=['date'])
    out['tsmc_weight_actual'] = {'ex_tsmc_w2330_mean_by_year': {int(y): round(float(g['w2330'].mean()), 2) for y, g in xt.groupby(xt['date'].dt.year)},
                                 'etf0050_w_2330': float(pd.read_csv(K.X / 'etf0050_w.csv', dtype={'code': str}).query("code=='2330'")['weight'].iloc[0])}
except Exception as e:
    out['tsmc_weight_actual'] = str(e)
log('wproxy', out['tsmc_weight_proxy'], out['tsmc_weight_actual'])

# ================= E. 容量 =================
cap = {}
for n in (10, 30):
    df = EV[n]
    for lab in ('SEL', 'VAL', 'HOLD'):
        lo, hi = RANGES[lab]; x = df[(df['d'] >= pd.Timestamp(lo)) & (df['d'] <= pd.Timestamp(hi))]
        c = {}
        for C in (1e6, 1e7, 5e7):
            ratio = (C / n) / x['adv20']
            c[f'{int(C/1e4)}萬'] = {'median_pct': round(float(ratio.median()) * 100, 3), 'p90_pct': round(float(ratio.quantile(0.9)) * 100, 3),
                                   'p99_pct': round(float(ratio.quantile(0.99)) * 100, 3), 'share_gt5pct': round(float((ratio > 0.05).mean()), 4),
                                   'by_bucket_share_gt5pct': {O.BNAME[b]: round(float((ratio[x['bucket'] == b] > 0.05).mean()), 4) for b in range(4)}}
        c['adv20_median_by_bucket_億'] = {O.BNAME[b]: round(float(x[x['bucket'] == b]['adv20'].median()) / 1e8, 3) for b in range(4)}
        cap[f'n{n}|{lab}'] = c
adv_year = {}
for d in sig:
    cs = S0.cross_section(d); rk = cs['rank']; v = cs['v60']
    for b, (lo, hi) in enumerate(O.BUCKETS):
        m = (rk >= lo) & (rk <= hi)
        adv_year.setdefault(d.year, {}).setdefault(O.BNAME[b], []).append(float(np.median(v[m])))
cap['pool_v60_median_by_bucket_億_by_year'] = {y: {b: round(float(np.median(x)) / 1e8, 2) for b, x in v.items()} for y, v in adv_year.items()}
# 大型基金換算（2026 年各組 ADV 中位數）：規模 A 的基金持有 1% 權重，若每天只能占 ADV 10%，需要幾個交易日建倉
a26 = cap['pool_v60_median_by_bucket_億_by_year'][2026]
cap['fund_days_to_build_1pct_at_10pctADV'] = {f'{A}億': {b: round(A * 0.01 / (0.1 * v), 1) for b, v in a26.items()} for A in (100, 500, 2800)}
cap['aum_for_n10_position_eq_5pctADV20_億'] = {lab: {b: round(10 * 0.05 * v, 2) for b, v in cap[f'n10|{lab}']['adv20_median_by_bucket_億'].items()} for lab in ('SEL', 'VAL', 'HOLD')}
out['capacity'] = cap
log('cap', cap['n10|HOLD']['5000萬'], cap['fund_days_to_build_1pct_at_10pctADV'])

# ================= 判定 =================
a10 = out['attrib']['n10']
sel_m, val_m, all_t = a10['SEL']['mml_ex_pool']['mean_pp'], a10['VAL']['mml_ex_pool']['mean_pp'], a10['ALL']['mml_ex_pool']['t']
if sel_m > 0 and val_m > 0 and all_t >= 2: cap_v = '容量溢酬成立'
elif (sel_m > 0) != (val_m > 0) or all_t < 1: cap_v = '不成立'
else: cap_v = '部分成立'
J = {'capacity': {'SEL_mean_pp': sel_m, 'VAL_mean_pp': val_m, 'ALL_t': all_t, 'verdict': cap_v}}
J['pool'] = {'chosen': POOL, 'verdict': '凍結選 U300（51～300、101～300 在挑選期配對 t 只有 −0.19、−0.28，未達 1.5），SOP 股池維持前 300' if POOL == 'U300' else 'TBD'}
w0, wc = out['tsmc']['w0']['stats'], out['tsmc'][f'w{int(WSEL*100)}']['stats']
c1 = wc['VAL']['t_0050'] > w0['VAL']['t_0050']; c2 = (w0['ALL']['CAGR'] - wc['ALL']['CAGR']) <= 2.0; c3 = wc['ALL']['maxDD'] >= w0['ALL']['maxDD']
hold_imp = wc['HOLD']['t_0050'] > w0['HOLD']['t_0050']
J['tsmc'] = {'w': WSEL, 'VAL_t_w': wc['VAL']['t_0050'], 'VAL_t_w0': w0['VAL']['t_0050'], 'ALL_CAGR_loss_pp': round(w0['ALL']['CAGR'] - wc['ALL']['CAGR'], 2),
             'ALL_maxDD_w': wc['ALL']['maxDD'], 'ALL_maxDD_w0': w0['ALL']['maxDD'], 'cond_VAL_t_higher': bool(c1), 'cond_loss_le_2pp': bool(c2), 'cond_DD_not_deeper': bool(c3),
             'HOLD_t_improves': bool(hold_imp),
             'verdict': '可用的配置選項' if (c1 and c2 and c3) else ('待觀察' if hold_imp else '不可用')}
out['judgement'] = J
log('judgement', J)
(O.HERE / 'results_oos.json').write_text(O.dumps(out), encoding='utf-8')
R['OOS'] = {'written_at': pd.Timestamp.now(tz='UTC').isoformat(), 'file': 'results_oos.json', 'judgement': J,
            'sha256_results_oos': hashlib.sha256((O.HERE / 'results_oos.json').read_bytes()).hexdigest()}
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
log('done')

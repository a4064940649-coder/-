#!/usr/bin/env python3
"""樣本外（凍結後只跑一次）：驗證期 2019-01～2023-12、保留期 2024-01～2026-10-12、跨循環；PREREG 必須已存在。輸出 oos.json、navs_oos.parquet。"""
import sys, json, time, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import hb_core as H
RES = json.loads((H.HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG_sha256' in RES and 'PREREG' in RES, 'PREREG 不存在：不能計算樣本外'
txt = json.dumps(RES['PREREG'], ensure_ascii=False, sort_keys=True)
assert hashlib.sha256(txt.encode('utf-8')).hexdigest() == RES['PREREG_sha256'], 'PREREG 被改過'
for p, s in RES['PREREG']['code_sha256'].items():
    assert H.sha(H.HERE / p) == s, f'{p} 在凍結後被改過'
assert H.sha(H.HERE / 'iterations.json') == RES['PREREG']['iterations_sha256']
SELV = RES['PREREG']['selected_version']; SPEC = RES['PREREG']['selected_spec']
K, V, R = H.K, H.V, H.R
IT = json.loads((H.HERE / 'iterations.json').read_text(encoding='utf-8'))
ALLSPEC = {k: {a: b for a, b in v['spec'].items() if a != 'family'} for k, v in IT['versions'].items()}
T0 = time.time()
st = H.setup(); S = H.make_S(); P = H.Prep(S)
DAYS = P.days; END = S.end_day
t0, t1, SD = H.bounds(DAYS, END)
XR = st['XR']
COSTS = {'gross': (0.0, 0.0), 'normal': H.NORMAL, 'stress': H.STRESS, 'flat': H.FLAT, 'cap10m': 'cap'}
PER = dict(H.PER_OOS)
PER['Y2009'] = (pd.Timestamp('2009-01-01'), pd.Timestamp('2009-12-31'))
PER['2020-04~06'] = (pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30'))
PER['S4_2024-2026'] = PER['HOLD']
E50 = H.E50
out = {'selected': SELV, 'spec': SPEC, 'prereg_sha256': RES['PREREG_sha256'], 'periods': {k: [str(a.date()), str(b.date())] for k, (a, b) in PER.items()}}
NAV = {}; LOG = {}
for c, cost in COSTS.items():
    for nm in ('BASE', SELV):
        nav, lg = H.run(P, ALLSPEC[nm], cost, DAYS, END, XR, log_detail=True)
        NAV[f'{nm}|{c}'] = nav; LOG[f'{nm}|{c}'] = lg
print('main runs', round(time.time() - T0), flush=True)
plb = LOG['BASE|normal']['plan']
ew, _ = K.simulate([(p[0], p[2]) for p in plb], 0, 0, t0, t1)
NAV['EW'] = ew
v2n = pd.read_parquet(H.AG / 'main_v2' / 'navs_gapfix.parquet')['EXFIN|n10|normal']
v2n = v2n[v2n.index <= END]
# V3 對照（重建＝v3_results R_DEL|E1）
ref = json.loads((H.AG / 'o_disposal_rejudge' / 'v3_results.json').read_text(encoding='utf-8'))['runs']['R_DEL|E1']['stats']['ALL']
cg_all = R.cagr_mdd(NAV['BASE|normal'], H.SEL_A, END)[0]
assert abs(cg_all - ref['CAGR']) < 1e-6
stats = {}
for key in [k for k in NAV if k != 'EW']:
    nm, c = key.split('|')
    stats[key] = {p: H.stats_block(NAV[key], SD, a, b, base=(NAV[f'BASE|{c}'] if nm != 'BASE' else None), ew=ew) for p, (a, b) in PER.items()}
stats['EW'] = {p: H.stats_block(ew, SD, a, b) for p, (a, b) in PER.items()}
v2sd = [d for d in SD if d >= v2n.index[0]]
stats['main_v2_EXFIN_n10_normal'] = {p: H.stats_block(v2n, v2sd, max(a, v2n.index[0]), b) for p, (a, b) in PER.items() if b > v2n.index[0]}
out['stats'] = stats
# 周轉（各期平均；每期 tov 對應第 i 個換股日）
def tov_by(lg, days):
    tv = pd.Series(lg['tov'][1:], index=days[1:])
    return {p: float(tv[(tv.index >= a) & (tv.index <= b)].mean()) for p, (a, b) in PER.items() if ((tv.index >= a) & (tv.index <= b)).any()}
out['turnover'] = {k: tov_by(LOG[k], DAYS) for k in ('BASE|normal', f'{SELV}|normal')}
# 判定（挑選期，T1～T5）
def thr(nm):
    dn = H.window_diff(NAV[f'{nm}|normal'], NAV['BASE|normal'], SD, H.SEL_A, H.SEL_B)
    ds = H.window_diff(NAV[f'{nm}|stress'], NAV['BASE|stress'], SD, H.SEL_A, H.SEL_B)
    dg = H.window_diff(NAV[f'{nm}|gross'], NAV['BASE|gross'], SD, H.SEL_A, H.SEL_B)
    be_sel = IT['versions'][nm]['breakeven_slip_mean0'] - IT['versions']['BASE']['breakeven_slip_mean0']
    o = dict(T1_normal_mean_pp=float(dn.mean() * 100), T1=bool(dn.mean() > 0), T2_stress_mean_pp=float(ds.mean() * 100), T2=bool(ds.mean() >= 0),
             T3_breakeven_gain_pct=be_sel * 100, T3=bool(be_sel >= 0.0005), T4_drop_best3_pp=H.drop_best(dn, 3)['mean_pp'], T4=bool(H.drop_best(dn, 3)['mean_pp'] >= 0),
             T5_gross_mean_pp=float(dg.mean() * 100), T5=bool(dg.mean() * 100 >= -0.05))
    o['all_pass'] = all(o[k] for k in ('T1', 'T2', 'T3', 'T4', 'T5'))
    return o
out['thresholds'] = {SELV: thr(SELV)}
# 損益兩平（跨循環）：t 對 0050 降到 2.3／月均超額＝0 的單邊滑價
def be_all(spec, target, lo=0.0, hi=0.03, a=H.SEL_A, b=H.HOLD_B):
    def f(s):
        nav, _ = H.run(P, spec, H.rates_slip(s), DAYS, END, XR)
        x = R.tvs(nav, E50.reindex(nav.index), SD, a, b)
        return x['t'] - 2.3 if target == 't23' else x['mean_pp']
    if f(lo) <= 0: return 0.0
    if f(hi) > 0: return hi
    for _ in range(16):
        m = 0.5 * (lo + hi)
        if f(m) > 0: lo = m
        else: hi = m
    return 0.5 * (lo + hi)
out['breakeven_ALL'] = {nm: {'t23': be_all(ALLSPEC[nm], 't23'), 'mean0': be_all(ALLSPEC[nm], 'mean0')} for nm in ('BASE', SELV)}
# 全額手續費（0.1425% 不打折）
fullfee = (0.001425 + 0.001, 0.001425 + 0.003 + 0.001)
out['full_fee_ALL'] = {}
for nm in ('BASE', SELV):
    nav, _ = H.run(P, ALLSPEC[nm], fullfee, DAYS, END, XR)
    out['full_fee_ALL'][nm] = H.stats_block(nav, SD, H.SEL_A, H.HOLD_B)
print('breakeven done', round(time.time() - T0), flush=True)
# 端點敏感度（對 0050，跨循環，一般成本）
def endpoint(nav):
    ra, rb = R.win_returns(nav, SD), R.win_returns(E50.reindex(nav.index), SD)
    dd = (ra - rb).dropna()
    o = dict(t=R.tstat(dd), drop_last=R.tstat(dd.iloc[:-1]))
    for k in (1, 3, 5): o[f'drop_best_{k}'] = R.tstat(dd.sort_values().iloc[:-k])
    yrs = pd.Series([d.year for d in dd.index], index=dd.index)
    by = dd.groupby(yrs).sum().idxmax(); o['drop_best_year'] = (R.tstat(dd[yrs != by]), int(by))
    o['to_2025'] = R.tstat(dd[dd.index <= pd.Timestamp('2025-12-31')])
    o['cal_t'] = V.cal_t(nav, ew, '2006-04-01', str(END.date()))[0]
    return o
out['endpoint_ALL_vs0050'] = {nm: endpoint(NAV[f'{nm}|normal']) for nm in ('BASE', SELV)}
# Bonferroni（單尾常態近似）
nv = IT['n_versions_total']
out['bonferroni'] = {}
for nm in ('BASE', SELV):
    t = stats[f'{nm}|normal']['ALL']['vs0050']['t']; p1 = H.p_one(t)
    out['bonferroni'][nm] = dict(t=t, p_one=p1, m10=min(1, p1 * 10), m_cum=10 + nv, p_bonf_mcum=min(1, p1 * (10 + nv)))
ps = stats[f'{SELV}|stress']['SEL']['paired']
out['bonferroni']['paired_selected_stress_SEL'] = dict(t=ps['t'], p_one=H.p_one(ps['t']), m=nv, p_bonf=min(1, H.p_one(ps['t']) * nv))
# AB（只用已公告營收，Monte Carlo 20 次平均淨值，一般成本）
ab = {nm: [] for nm in ('BASE', SELV)}
for seed in range(20):
    Sx = H.make_S(unavail=V.make_unavail('mc', seed)); Px = H.Prep(Sx)
    assert Px.days == DAYS
    for nm in ab:
        nav, _ = H.run(Px, ALLSPEC[nm], H.NORMAL, DAYS, END, XR)
        ab[nm].append(nav)
abn = {nm: pd.concat(v, axis=1).mean(axis=1) for nm, v in ab.items()}
out['AB'] = {nm: {p: H.stats_block(abn[nm], SD, a, b, base=(abn['BASE'] if nm != 'BASE' else None)) for p, (a, b) in PER.items()} for nm in abn}
out['AB_seed_spread_ALL_CAGR'] = {nm: [float(np.percentile([R.cagr_mdd(x, H.SEL_A, END)[0] for x in v], q)) for q in (5, 50, 95)] for nm, v in ab.items()}
print('AB done', round(time.time() - T0), flush=True)
# 逐年
def yearly(nav):
    y = nav.groupby(nav.index.year).last(); y0 = pd.concat([pd.Series([nav.iloc[0]], index=[nav.index[0].year - 1]), y])
    return {int(k): float(v * 100) for k, v in (y0.pct_change().dropna()).items()}
e50 = E50.reindex(NAV['BASE|normal'].index)
out['yearly'] = {'BASE': yearly(NAV['BASE|normal']), SELV: yearly(NAV[f'{SELV}|normal']), '0050': yearly(e50), 'EW': yearly(ew)}
# 分產業（扣成本前每窗等權貢獻，選中版本 − 原版）與去掉最大貢獻股
pls = LOG[f'{SELV}|normal']['plan']
ind = {c: i for c, i in zip(V.COLS, V.IND)}
def ind_contrib(pl, a, b):
    con, _ = V.stock_contrib(pl, t1, a=a, b=b)
    s = {}
    for c, x in con.items(): s[ind[c]] = s.get(ind[c], 0.0) + x * 100
    return s
out['industry'] = {}
for p in ('SEL', 'POST', 'ALL'):
    a, b = PER[p]
    cb, cs = ind_contrib(plb, a, b), ind_contrib(pls, a, b)
    diff = {k: cs.get(k, 0) - cb.get(k, 0) for k in set(cb) | set(cs)}
    out['industry'][p] = dict(diff_top=sorted(diff.items(), key=lambda x: -abs(x[1]))[:8], sum_diff=sum(diff.values()))
out['top_contrib'] = {nm: {p: V.top_contrib(pl, t1, PER[p][0], PER[p][1]) for p in ('SEL', 'VAL', 'HOLD', 'ALL')} for nm, pl in (('BASE', plb), (SELV, pls))}
# 去掉「選中 − 原版」差異最大貢獻股：該股在兩版的報酬都換成同期其他持股平均後的配對差（扣成本前，近似）
def contrib_by_stock(pl, a, b):
    con, _ = V.stock_contrib(pl, t1, a=a, b=b); return con
for p in ('SEL', 'POST'):
    a, b = PER[p]
    cb, cs = contrib_by_stock(plb, a, b), contrib_by_stock(pls, a, b)
    dif = {k: cs.get(k, 0) - cb.get(k, 0) for k in set(cb) | set(cs)}
    top = sorted(dif.items(), key=lambda x: -x[1])[:3]; bot = sorted(dif.items(), key=lambda x: x[1])[:3]
    out.setdefault('diff_by_stock', {})[p] = dict(sum_pp=float(sum(dif.values()) * 100), top3=[(k, float(v * 100)) for k, v in top], bottom3=[(k, float(v * 100)) for k, v in bot],
                                                  sum_ex_top1_pp=float((sum(dif.values()) - top[0][1]) * 100))
# 容量（新買名額部位 100 萬 > 60 日均成交值 1%）
V60 = H._V60()
def cap_share(pl):
    o = {}
    for p in ('SEL', 'POST', 'ALL'):
        a, b = PER[p]; new = []; ntr = []
        for i, (e, pk, _, d) in enumerate(pl):
            if not (a <= d <= b) or i == 0: continue
            prev = set(pl[i - 1][1].tolist())
            nn = [j for j in pk if j not in prev]
            ntr.append(len(nn))
            new += [bool(1e6 > 0.01 * V60[e - 1, j]) for j in nn]
        o[p] = dict(share_over=float(np.mean(new)) if new else None, new_names_per_month=float(np.mean(ntr)) if ntr else None)
    return o
out['capacity'] = {'BASE': cap_share(plb), SELV: cap_share(pls)}
# 其餘版本：2019+ 配對差（只報告）
rep = {}
for nm in ALLSPEC:
    if nm in ('BASE', SELV): continue
    rr = {}
    for c in ('normal', 'stress'):
        nav, lg = H.run(P, ALLSPEC[nm], COSTS[c], DAYS, END, XR)
        rr[c] = {p: R.paired(nav, NAV[f'BASE|{c}'], SD, *PER[p]) for p in ('SEL', 'VAL', 'HOLD', 'POST', 'ALL')}
        if c == 'normal':
            rr['ALL_CAGR'] = R.cagr_mdd(nav, H.SEL_A, END)[0]; rr['ALL_t0050'] = R.tvs(nav, E50.reindex(nav.index), SD, H.SEL_A, H.HOLD_B)['t']
            rr['turnover'] = lg['turnover']
    rep[nm] = rr
out['report_only_other_versions'] = rep
print('others done', round(time.time() - T0), flush=True)
pd.DataFrame({k: v for k, v in NAV.items()}).to_parquet(H.HERE / 'navs_oos.parquet')
(H.HERE / 'oos.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print('done', round(time.time() - T0))

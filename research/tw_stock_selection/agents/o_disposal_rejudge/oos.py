#!/usr/bin/env python3
"""樣本外（驗證期／保留期，皆已被看過 → 只報告）＋跨循環＋穩健性。PREREG 必須已存在。輸出 results.json 的 OOS、ROBUST。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import drj_core as R
K, V = R.K, R.V
HERE = R.HERE
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG_sha256' in res and 'PREREG' in res, 'PREREG 不存在：不能計算樣本外'
SELV = res['PREREG']['selected_version']
LISTV = sys.argv[1] if len(sys.argv) > 1 else 'A'
D = R.load_list(use_proxy_gapyears=(LISTV == 'AP'))
DS = R.Disp(D)
NORMAL = K.rates(0.001)
PER = {'SEL': (R.SEL_A, R.SEL_B), 'SEL_ex2021': (R.SEL_A, pd.Timestamp('2020-12-31')), 'VAL': (R.VAL_A, R.VAL_B), 'HOLD': (R.HOLD_A, R.HOLD_B),
       'POST': (R.VAL_A, R.HOLD_B), 'ALL': (R.SEL_A, R.HOLD_B)}
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)

class OnlyMask:
    """E5 但指定代碼改成 E1 處理（剔除貢獻最大股用）。"""
def make(ver, n, force_excl=()):
    S = R.make_strategy(ver if ver != 'E4' else 'E0', DS)
    if force_excl:
        base = S._mask
        fe = {K.CI[c] for c in force_excl}
        def m2(d, _b=base):
            m = _b(d)
            for j in DS.at(d):
                if j in fe: m[j] = True
            return m
        S._mask = m2
    return S

def run(ver, n, cost, force_excl=()):
    S = make(ver, n, force_excl)
    t0, t1, sd = S.bounds()
    pl = S.plan(n, 'official')
    dm = R.defer_map(pl, DS) if ver == 'E4' else None
    if cost == 'flat_stress':
        b, s = K.rates('stress'); extra = None
    else:
        b, s = NORMAL; extra = 0.005 if cost == 'stress' else None
    nav, lg = R.simulate2(pl, b, s, t0, t1, defer=dm, extra=extra, DS=DS)
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    return dict(nav=nav, sd=sd, pl=pl, lg=lg, ew=ew, S=S)

def stats(r, bench_main=None):
    nav, sd, ew = r['nav'], r['sd'], r['ew']
    e50 = R.E50.reindex(nav.index)
    o = {}
    for k, (a, b) in PER.items():
        last = min(pd.Timestamp(b), nav.index[-1])
        cg, mdd = R.cagr_mdd(nav, a, last)
        c50, m50 = R.cagr_mdd(e50, a, last)
        cew, _ = R.cagr_mdd(ew, a, last)
        o[k] = dict(CAGR=cg, maxDD=mdd, E50_CAGR=c50, EW_CAGR=cew, vs0050=R.tvs(nav, e50, sd, a, b), vsEW=R.tvs(nav, ew, sd, a, b))
    return o

def yearly(nav):
    r = nav.pct_change().dropna()
    return {int(y): round(float((1 + g).prod() - 1) * 100, 2) for y, g in r.groupby(r.index.year)}

out = {'list_version': LISTV, 'selected': SELV, 'runs': {}, 'paired': {}}
RUN = {}
for n in (10, 30):
    for cost in ('normal', 'stress', 'flat_stress'):
        for ver in ['E1', 'E0', 'E2', 'E3', 'E4', 'E5']:
            if n == 30 and cost == 'flat_stress': continue
            r = run(ver, n, cost); RUN[(ver, n, cost)] = r
            out['runs'][f'{ver}|n{n}|{cost}'] = dict(stats=stats(r), turnover=r['lg']['turnover'], n_frozen=len(r['lg']['frozen']),
                                                    defer_entered=r['lg'].get('defer_entered'), defer_never=r['lg'].get('defer_never'))
            if ver != 'E1':
                out['paired'][f'{ver}|n{n}|{cost}'] = {k: R.paired(r['nav'], RUN[('E1', n, cost)]['nav'], r['sd'], a, b) for k, (a, b) in PER.items()}
            log(ver, n, cost, {k: round(v['CAGR'], 2) for k, v in out['runs'][f'{ver}|n{n}|{cost}']['stats'].items()},
                {k: round(v['mean_pp'], 3) for k, v in out['paired'].get(f'{ver}|n{n}|{cost}', {}).items()})
# 官方主策略淨值（main_v2；2021-09 前無處置濾網）
ref = pd.read_parquet(R.MV2 / 'navs_gapfix.parquet')['EXFIN|n10|normal'].dropna()
r1 = RUN[('E1', 10, 'normal')]
out['vs_official_main_v2'] = {v: {k: R.paired(RUN[(v, 10, 'normal')]['nav'], ref.reindex(r1['nav'].index), r1['sd'], a, b) for k, (a, b) in PER.items()} for v in ('E1', 'E0', SELV)}
out['official_main_v2_stats'] = stats(dict(nav=ref.reindex(r1['nav'].index), sd=r1['sd'], ew=r1['ew']))
out['yearly'] = {v: yearly(RUN[(v, 10, 'normal')]['nav']) for v in ('E1', 'E0', SELV)}
out['yearly']['0050'] = yearly(R.E50.reindex(r1['nav'].index))
out['yearly_diff_vs_E1_pp'] = {v: {y: round(out['yearly'][v][y] - out['yearly']['E1'][y], 2) for y in out['yearly']['E1']} for v in ('E0', SELV)}
# 均值與波動拆解（對 0050）
dec = {}
for v in ('E1', 'E0', SELV):
    s = out['runs'][f'{v}|n10|normal']['stats']
    dec[v] = {k: dict(mean_pp=s[k]['vs0050']['mean_pp'], sd_pp=s[k]['vs0050']['sd_pp'], t=s[k]['vs0050']['t']) for k in ('SEL', 'POST', 'ALL', 'HOLD')}
out['t_decomposition_vs_0050'] = dec

# ---------------- 受影響名額（全期）與單股貢獻 ----------------
def affected(ver, n):
    S0 = RUN[(ver, n, 'normal')]['S'] if (ver, n, 'normal') in RUN else make(ver, n)
    pl = RUN[(ver, n, 'normal')]['pl']; p1 = {p[3]: set(p[1].tolist()) for p in RUN[('E1', n, 'normal')]['pl']}
    rows = []
    for (e, pk, _, d) in pl:
        info = DS.at(d)
        for j in pk:
            if j in info and j not in p1.get(d, set()):
                i = info[j]
                rows.append(dict(sig=str(d.date()), code=K.COLS[j], name=V.SL['name'].get(K.COLS[j], ''), industry=V.SL['industry'].get(K.COLS[j], ''),
                                 nth=i['nth'], rem=i['rem'], thr=i['thr'], kind=i['kind'], regime=i['regime']))
    return pd.DataFrame(rows)
AFF = {}
for v in ('E0', SELV):
    for n in (10, 30):
        a = affected(v, n); AFF[(v, n)] = a
        if len(a):
            a['y'] = a.sig.str[:4].astype(int)
            out.setdefault('affected', {})[f'{v}|n{n}'] = dict(total=int(len(a)), by_period={k: int(((a.sig >= str(p[0].date())) & (a.sig <= str(p[1].date()))).sum()) for k, p in PER.items()},
                                                             by_year={int(y): int(c) for y, c in a.groupby('y').size().items()},
                                                             by_industry={k: int(c) for k, c in a.industry.value_counts().items()},
                                                             nth={int(k): int(c) for k, c in a.nth.value_counts().items()})
pd.concat({f'{k[0]}|n{k[1]}': v for k, v in AFF.items()}).to_csv(HERE / f'affected_slots_{LISTV}.csv')

# 剔除貢獻最大 3 檔（挑選期，選中版本 n10 一般）：逐檔把該股改為 E1 處理，看配對差累積少多少
def sum_diff(r, a, b):
    p = R.paired(r['nav'], RUN[('E1', 10, 'normal')]['nav'], r['sd'], a, b); return p['mean_pp'] * p['n']
base_sel = sum_diff(RUN[(SELV, 10, 'normal')], *PER['SEL'])
base_all = sum_diff(RUN[(SELV, 10, 'normal')], *PER['ALL'])
codes_sel = sorted(set(AFF[(SELV, 10)][AFF[(SELV, 10)].sig <= str(R.SEL_B.date())].code))
contrib = {}
for c in codes_sel:
    rr = run(SELV, 10, 'normal', force_excl=(c,))
    contrib[c] = base_sel - sum_diff(rr, *PER['SEL'])
top3 = sorted(contrib, key=lambda c: -contrib[c])[:3]
rr3 = run(SELV, 10, 'normal', force_excl=tuple(top3))
p3 = R.paired(rr3['nav'], RUN[('E1', 10, 'normal')]['nav'], rr3['sd'], *PER['SEL'])
p3all = R.paired(rr3['nav'], RUN[('E1', 10, 'normal')]['nav'], rr3['sd'], *PER['ALL'])
pos_total = sum(v for v in contrib.values() if v > 0)
out['P5_right_tail_SEL'] = dict(sum_paired_diff_pp=base_sel, per_code_contrib_pp={c: round(v, 3) for c, v in sorted(contrib.items(), key=lambda x: -x[1])},
                                top3=[(c, V.SL['name'].get(c, '')) for c in top3], top3_share_of_total=sum(contrib[c] for c in top3) / base_sel if base_sel != 0 else None,
                                top3_share_of_positive=sum(contrib[c] for c in top3) / pos_total if pos_total > 0 else None,
                                paired_SEL_after_drop_top3=p3, paired_ALL_after_drop_top3=p3all)
# 同樣對 E0（P1 的右尾）
codes0 = sorted(set(AFF[('E0', 10)][AFF[('E0', 10)].sig <= str(R.SEL_B.date())].code))
base0 = sum_diff(RUN[('E0', 10, 'normal')], *PER['SEL'])
c0 = {}
for c in codes0:
    rr = run('E0', 10, 'normal', force_excl=(c,)); c0[c] = base0 - sum_diff(rr, *PER['SEL'])
t30 = sorted(c0, key=lambda c: -c0[c])[:3]
rr03 = run('E0', 10, 'normal', force_excl=tuple(t30))
out['P5_right_tail_E0_SEL'] = dict(sum_paired_diff_pp=base0, per_code_contrib_pp={c: round(v, 3) for c, v in sorted(c0.items(), key=lambda x: -x[1])},
                                   top3=[(c, V.SL['name'].get(c, '')) for c in t30], top3_share=sum(c0[c] for c in t30) / base0 if base0 else None,
                                   paired_SEL_after_drop_top3=R.paired(rr03['nav'], RUN[('E1', 10, 'normal')]['nav'], rr03['sd'], *PER['SEL']))
log('P5', out['P5_right_tail_SEL']['top3'], out['P5_right_tail_SEL']['top3_share_of_total'], p3['mean_pp'])

# 月份明細（選中版本、E0 對 E1，非零月）
md = {}
for v in ('E0', SELV):
    r = RUN[(v, 10, 'normal')]
    dd = R.win_returns(r['nav'], r['sd']) - R.win_returns(RUN[('E1', 10, 'normal')]['nav'], r['sd'])
    md[v] = {str(k.date()): round(float(x) * 100, 3) for k, x in dd.items() if abs(x) > 1e-7}
out['nonzero_months_pp'] = md

# ---------------- 2021-09 後的事件層（只報告）：處置組 vs 注意股未處置（安慰劑 B） ----------------
att = pd.read_csv(K.X / 'attention_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
Cn, On, G, TR = K.Cn, K.On, K.GROW, K.TRADED
def r_oo(j, a, b):
    if not (TR[a, j] and np.isfinite(On[a, j])): return np.nan
    x = Cn[a, j] / On[a, j] * np.prod(G[a + 1:b + 1, j])
    if TR[b, j] and np.isfinite(On[b, j]): x = x * On[b, j] / Cn[b, j]
    return x - 1
pl0 = RUN[('E0', 10, 'normal')]['pl']
evd, evp = [], []
for i, (e, pk, _, d) in enumerate(pl0):
    if d < R.VAL_A or i + 1 >= len(pl0): continue
    e2 = pl0[i + 1][0]; info = DS.at(d); t = e - 1
    a10 = set(att[(att.date <= d) & (att.date > K.IDX[max(t - 10, 0)])].code)
    rr = {int(j): r_oo(int(j), e, e2) for j in pk}
    dj = [j for j in rr if j in info]
    pj = [j for j in rr if j not in info and K.COLS[j] in a10 and DS.in_disp_day(j, t) is None]
    oth = [rr[j] for j in rr if j not in dj and j not in pj]
    if not oth: continue
    om = np.nanmean(oth)
    for j in dj: evd.append(dict(sig=d, x=rr[j] - om, rem=info[j]['rem'], thr=info[j]['thr'], nth=info[j]['nth']))
    for j in pj: evp.append(dict(sig=d, x=rr[j] - om))
def mt(df):
    if len(df) == 0: return None
    g = df.groupby('sig').x.mean(); return dict(mean_pp=float(g.mean() * 100), t=R.tstat(g.values), n_months=int(len(g)), n_events=int(len(df)))
evd = pd.DataFrame(evd); evp = pd.DataFrame(evp)
out['post2021_event_level_report_only'] = dict(disposal=mt(evd), attention_not_disposed_placebo=mt(evp),
                                               disposal_R2=mt(evd[evd.rem >= evd.thr]) if len(evd) else None, disposal_R01=mt(evd[evd.rem < evd.thr]) if len(evd) else None,
                                               disposal_nth1=mt(evd[evd.nth == 1]) if len(evd) else None, disposal_nth2=mt(evd[evd.nth >= 2]) if len(evd) else None)

# ---------------- 與處置股策略（20% 部位）的重複曝險 ----------------
a0 = AFF[('E0', 10)]
out['overlap_with_disposal_sleeve'] = dict(
    note='處置股策略在處置第一天收盤買、最後一天收盤賣，所有上市處置股都是候選（10 槽）。主策略若不排除處置股，入選的處置股必然也是處置策略的候選。',
    E0_affected_slots_post2021=int((a0.sig >= str(R.VAL_A.date())).sum()), E0_affected_slots_SEL=int((a0.sig <= str(R.SEL_B.date())).sum()),
    weight_if_both='主策略 80%×1/10＝8%＋處置部位 20%×1/10＝2% → 同一檔 10%，且在同一個處置窗內同方向。')
(HERE / f'oos_{LISTV}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
pd.DataFrame({f'{k[0]}|n{k[1]}|{k[2]}': v['nav'] for k, v in RUN.items()}).to_parquet(HERE / f'navs_full_{LISTV}.parquet')
log('done')

#!/usr/bin/env python3
"""S47：資金配置家族 15 格（衛星 {50,60,70,80,100}% × 每檔 {÷8,÷10,÷12}，空位 0050）＋ S70_D10_CASH（空位留現金），F1＋E0、嚴成本、三個正式開關全開。
每格 20 條路徑（含 100 萬位移 0 主跑）＋主跑成本加倍。python3 -I -B run.py → results.json、navs.csv.gz。SPEC.json／PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s47lib as L
from s47lib import s17lib, s10lib, s4lib, stlib, cslib

H = Path(__file__).resolve().parent
for n in ('SPEC', 'PREREG'):
    assert hashlib.sha256((H / f'{n}.json').read_bytes()).hexdigest() == (H / f'{n}.sha256').read_text().split()[0], f'{n}.json 與 sha256 不符'
SPEC = json.loads((H / 'SPEC.json').read_text())
for n, v in SPEC['code_sha256_at_write'].items():
    assert hashlib.sha256((H / n).read_bytes()).hexdigest() == v, f'{n} 在登記後被改過'
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
RT = dict(S=F['S']['STD'], av=F['av'])
del F
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, D24 = lastb('2019-01-01'), lastb('2024-01-01')
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'y2019_2023': (D19, D24)}
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])


def mt(nav):
    m = {k: stlib.cagr(nav, a, b) for k, (a, b) in WIN.items()}
    m['maxdd'] = stlib.maxdd(nav, D0, DE)
    m['calmar'] = m['since2019'] / abs(m['maxdd'])
    m['calmar_full'] = m['full'] / abs(m['maxdd'])
    m['lag_days'] = int(L.longest_lag(nav, bench)[0])
    return m


def yearly(nav):
    y = nav.groupby(nav.index.year).last()
    prev = pd.concat([pd.Series([nav.iloc[0]], index=[y.index[0] - 1]), y.iloc[:-1]])
    prev.index = y.index
    return {int(k): float(v) for k, v in (y / prev.values - 1).items()}


ALL = L.CELLS + [L.CASH_CELL]
res = dict(spec_sha256=(H / 'SPEC.sha256').read_text().split()[0], prereg_sha256=(H / 'PREREG.sha256').read_text().split()[0],
           flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK), src_sha256=L.SRC_SHA,
           bench_0050=dict({k: v for k, v in mt(bench).items() if k not in ('lag_days',)}, yearly=yearly(bench), seg_dd=s17lib.seg_dd(bench)),
           cells={c['id']: dict(sat_w=c['sat_w'], div=c['div'], idle=c['idle'], paths={}) for c in ALL}, acceptance=dict(P0=True))
navs = {'0050': bench}
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    for c in ALL:
        r = res['cells'][c['id']]
        for cap in L.CAPS:
            nav, lg, st = L.sim(P, sg, U, sat_w=c['sat_w'], div=c['div'], idle=c['idle'], cap=cap)
            r['paths'][f'{int(cap/1e4)}萬|+{s}'] = mt(nav)
            if s == 0 and cap == 1e6:
                x = lg[lg['why'] != 'open_end']
                sf = st['stockfrac']; first = lg['ent_date'].min()
                lag = L.longest_lag(nav, bench)
                r['main'] = dict(r['paths']['100萬|+0'], n_closed=int(len(x)), n_buy=int(st['n_buy']), win_rate=float((x['ret'] > 0).mean()), mean_ret=float(x['ret'].mean()),
                                 avg_npos=float(st['avg_npos']), stockfrac_mean=float(sf.loc[sf.index >= first].mean()),
                                 sat_idle_frac_mean=float(((1 - sf) - st['B'] / nav).loc[sf.index >= first].mean()),
                                 rb_up=int(st['rb_up']), rb_down=int(st['rb_down']), base_sold_events=len(st['base_sold']), e50_orders=int(st['e50_orders']),
                                 lag=dict(days=int(lag[0]), start=lag[1], end=lag[2]),
                                 top5_share=s17lib.top5_share(lg, nav, 1e6), yearly=yearly(nav), seg_dd=s17lib.seg_dd(nav), end_nav=float(nav.iloc[-1]))
                r['main']['since2019_costx2'] = stlib.cagr(L.sim(P, sg, U, sat_w=c['sat_w'], div=c['div'], idle=c['idle'], cap=cap, cost='strict_x2')[0], D19, DE)
                navs[c['id']] = nav
                if c['id'] == 'S50_D10':                       # A3 補驗：ndiv=10 副本＝原引擎（非現行衛星比例）
                    nav2, lg2, _ = L.sim(P, sg, U, sat_w=0.5, div=10, cap=cap, force_copy=True)
                    cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
                    res['acceptance']['A3_S50_D10'] = bool(np.array_equal(nav.to_numpy(), nav2.to_numpy()) and lg[cols].equals(lg2[cols]))
    del sg; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

for c in ALL:
    r = res['cells'][c['id']]; pp = r['paths'].values()
    g = lambda k: np.array([v[k] for v in pp])
    r['path_summary'] = dict(since2019_median=float(np.median(g('since2019'))), since2019_min=float(g('since2019').min()), since2019_max=float(g('since2019').max()),
                             maxdd_median=float(np.median(g('maxdd'))), maxdd_worst=float(g('maxdd').min()),
                             calmar_median=float(np.median(g('calmar'))), calmar_full_median=float(np.median(g('calmar_full'))),
                             since2024_median=float(np.median(g('since2024'))), y2019_2023_median=float(np.median(g('y2019_2023'))),
                             full_median=float(np.median(g('full'))), lag_days_median=float(np.median(g('lag_days'))),
                             above_0050=int((g('since2019') > res['bench_0050']['since2019']).sum()))

# ---- 驗收
dl = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cm = res['cells'][L.CUR]['main']
a0 = dict(since2019=cm['since2019'], full=cm['full'], maxdd=cm['maxdd'], n_closed=cm['n_closed'])
a0['tol_ok'] = bool(abs(a0['since2019'] - L.ACC['since2019']) < 2.5e-4 and abs(a0['full'] - L.ACC['full']) < 2.5e-4
                    and abs(a0['maxdd'] - L.ACC['maxdd']) < 2.5e-4 and a0['n_closed'] == L.ACC['n_closed'])
a0['bitwise_vs_deadline_ALL'] = bool(cm['since2019'] == dl['standard']['ALL']['since2019'] and cm['full'] == dl['standard']['ALL']['full']
                                     and cm['maxdd'] == dl['standard']['ALL']['maxdd_full'])
res['acceptance']['A0'] = a0
cp = res['cells'][L.CUR]['paths']
d2 = [abs(cp[k]['since2019'] - v['since2019']) for k, v in dl['paths']['ALL'].items()]
d2d = [abs(cp[k]['maxdd'] - v['maxdd_full']) for k, v in dl['paths']['ALL'].items()]
res['acceptance']['A2'] = dict(n=len(d2), max_abs_diff=float(max(d2)), max_abs_diff_maxdd=float(max(d2d)), ok=bool(len(d2) == 20 and max(d2) < 1e-9 and max(d2d) < 1e-9))
off = Path('/home/user/-/xq_scripts/site/navs_official.csv.gz')
o = pd.read_csv(off, parse_dates=['date']).set_index('date')['E0|W10_70']
dd_ = (navs[L.CUR].reindex(o.index) - o).abs()
res['acceptance']['official_nav'] = dict(n=int(len(o)), max_abs_diff_yuan=float(dd_.max()), ok=bool(dd_.notna().all() and dd_.max() <= 0.0051))
assert a0['tol_ok'] and a0['bitwise_vs_deadline_ALL'] and res['acceptance']['A2']['ok'] and res['acceptance']['A3_S50_D10'] and res['acceptance']['official_nav']['ok'], f"驗收失敗 {res['acceptance']}"

# ---- 判定（事前規則）
ids = [c['id'] for c in L.CELLS]
PS = {k: res['cells'][k]['path_summary'] for k in res['cells']}
med = {k: PS[k]['since2019_median'] for k in ids}
ddm = {k: PS[k]['maxdd_median'] for k in ids}
cal = {k: PS[k]['calmar_median'] for k in ids}
fam_med, fam_cal = float(np.median(list(med.values()))), float(np.median(list(cal.values())))
allp = np.array([v['since2019'] for k in ids for v in res['cells'][k]['paths'].values()])
alld = np.array([v['maxdd'] for k in ids for v in res['cells'][k]['paths'].values()])
rank = lambda d, k0, sub=None: 1 + sum(d[k] > d[k0] for k in (sub or ids))
r_ret, r_dd, r_cal = rank(med, L.CUR), rank(ddm, L.CUR), rank(cal, L.CUR)
P3 = (med[L.CUR] - fam_med) * 100
P3c = cal[L.CUR] - fam_cal
P3c_rel = P3c / fam_cal
row10 = [k for k in ids if k.endswith('_D10')]
col70 = [k for k in ids if k.startswith('S70_')]


def pair(k):
    pp, q = res['cells'][k]['paths'], cp
    d = np.array([pp[x]['since2019'] - q[x]['since2019'] for x in q]) * 100
    d1 = np.array([pp[x]['y2019_2023'] - q[x]['y2019_2023'] for x in q]) * 100
    d2_ = np.array([pp[x]['since2024'] - q[x]['since2024'] for x in q]) * 100
    ddd = np.array([pp[x]['maxdd'] - q[x]['maxdd'] for x in q]) * 100
    dc = np.array([pp[x]['calmar'] - q[x]['calmar'] for x in q])
    dl_ = np.array([pp[x]['lag_days'] - q[x]['lag_days'] for x in q])
    npos, md = int((d > 0).sum()), float(np.median(d))
    lab = '比現行好' if (npos >= 15 and md >= 2) else ('比現行差' if (npos <= 5 and md <= -2) else '差不多')
    if lab == '比現行好':
        lab = '穩定地比較好' if (np.median(d1) > 0 and np.median(d2_) > 0) else '比現行好但只靠一段'
    ncp, mc = int((dc > 0).sum()), float(np.median(dc))
    rl = '風險調整後較好' if (ncp >= 15 and mc >= 0.10) else ('風險調整後較差' if (ncp <= 5 and mc <= -0.10) else '風險調整後差不多')
    note = '報酬較高但回撤較深（風險換報酬），不算改進' if (lab in ('穩定地比較好', '比現行好但只靠一段') and np.median(ddd) <= -2) else ''
    cand = bool(lab == '穩定地比較好' and rl != '風險調整後較差' and np.median(ddd) > -2)
    return dict(n_pos=npos, median_pp=md, seg2019_2023_median_pp=float(np.median(d1)), seg2024_median_pp=float(np.median(d2_)),
                maxdd_median_pp=float(np.median(ddd)), n_dd_shallower=int((ddd > 0).sum()), calmar_n_pos=ncp, calmar_median=mc,
                lag_days_median_diff=float(np.median(dl_)), label=lab, risk_label=rl, risk_note=note, improvement_candidate=cand)


pairs = {k: pair(k) for k in list(res['cells']) if k != L.CUR}
opt_ret, opt_cal = P3 >= 3, P3c_rel >= 0.10
if not opt_ret and not opt_cal:
    verdict = '沒被挑過（不必因配置打折）'
elif opt_ret and not opt_cal:
    verdict = '現行報酬高是承擔較多風險換來的（風險偏好），不算挑過；報酬預期不因配置打折，但必須和回撤一起報'
else:
    verdict = '現行配置偏樂觀：報酬預期改用家族 300 條路徑中位數（附第 10 百分位），回撤預期改用 300 條回撤中位數（附最深 10%）'
best = lambda r: '是' if r == 1 else ('前段' if r <= 4 else '不是')
cashp = pairs[L.CASH]
res['decision'] = dict(
    family_path_medians=med, family_maxdd_medians=ddm, family_calmar_medians=cal,
    family_median=fam_med, family_min=float(min(med.values())), family_max=float(max(med.values())),
    family_300_median=float(np.median(allp)), family_300_p10=float(np.percentile(allp, 10)),
    family_300_maxdd_median=float(np.median(alld)), family_300_maxdd_deepest10=float(np.percentile(alld, 10)),
    family_calmar_median=fam_cal,
    cur_path_median=med[L.CUR], cur_calmar=cal[L.CUR], cur_rank_return=int(r_ret), cur_rank_maxdd=int(r_dd), cur_rank_calmar=int(r_cal),
    sub_rank=dict(row_D10_return=int(rank(med, L.CUR, row10)), row_D10_calmar=int(rank(cal, L.CUR, row10)),
                  col_S70_return=int(rank(med, L.CUR, col70)), col_S70_calmar=int(rank(cal, L.CUR, col70))),
    P3_pp=float(P3), P3_calmar=float(P3c), P3_calmar_rel=float(P3c_rel),
    verdict_return=('報酬口徑偏樂觀' if opt_ret else '報酬口徑沒被挑過'),
    verdict_calmar=('風險調整後偏樂觀' if opt_cal else '風險調整後沒被挑過'),
    verdict=verdict, verdict_best_return=best(r_ret), verdict_best_calmar=best(r_cal),
    pairs=pairs, improvement_candidates=[k for k, v in pairs.items() if v['improvement_candidate']],
    idle_verdict=('空位買 0050 不是較好的選擇' if cashp['label'] in ('穩定地比較好', '比現行好但只靠一段') else '空位買 0050 沒有吃虧'))
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
pd.DataFrame(navs).round(2).to_csv(H / 'navs.csv.gz', compression='gzip')
code = {n: hashlib.sha256((H / n).read_bytes()).hexdigest() for n in ('s47lib.py', 'prep.py', 'prereg_write.py', 'run.py')}
(H / 'code.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in code.items()))
print(json.dumps({k: res['decision'][k] for k in ('cur_rank_return', 'cur_rank_maxdd', 'cur_rank_calmar', 'P3_pp', 'P3_calmar_rel', 'verdict', 'idle_verdict', 'family_median')}, ensure_ascii=False))
print('完成', res['meta'])

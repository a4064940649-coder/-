#!/usr/bin/env python3
"""S53：波動調整部位家族 6 格（A_D6、A_D8、B60、B120、B60_R、B120_R）＋現行＋參考 REF_S70_D8；F1＋E0、嚴成本、三個正式開關全開。
每格 20 條路徑＋主跑成本加倍＋容量（1,000 萬 × 位移 0～3）。python3 -I -B run.py → results.json。SPEC.json／PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s53lib as L
from s53lib import s17lib, s10lib, stlib, cslib

H = Path(__file__).resolve().parent
for n in ('SPEC', 'PREREG'):
    assert hashlib.sha256((H / f'{n}.json').read_bytes()).hexdigest() == (H / f'{n}.sha256').read_text().split()[0], f'{n}.json 與 sha256 不符'
SPEC = json.loads((H / 'SPEC.json').read_text())
for n, v in SPEC['code_sha256_at_write'].items():
    assert hashlib.sha256((H / n).read_bytes()).hexdigest() == v, f'{n} 在登記後被改過'
TG = {k: v for k, v in SPEC['cells']['targets'].items() if k in ('60', '120')}
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = L.base(P)
s17lib.e50_fix_suspension(P)
SIG = L.sigma60(P)
L.slim(P); gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, D24 = lastb('2019-01-01'), lastb('2024-01-01')
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'y2019_2023': (D19, D24)}
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
sys.path.insert(0, str(L.AG / 's47_allocation_family'))
from s47lib import longest_lag                              # noqa: E402


def mt(nav):
    m = {k: stlib.cagr(nav, a, b) for k, (a, b) in WIN.items()}
    m['maxdd'] = stlib.maxdd(nav, D0, DE)
    m['calmar'] = m['since2019'] / abs(m['maxdd'])
    m['calmar_full'] = m['full'] / abs(m['maxdd'])
    m['lag_days'] = int(longest_lag(nav, bench)[0])
    return m


def yearly(nav):
    y = nav.groupby(nav.index.year).last()
    prev = pd.concat([pd.Series([nav.iloc[0]], index=[y.index[0] - 1]), y.iloc[:-1]])
    prev.index = y.index
    return {int(k): float(v) for k, v in (y / prev.values - 1).items()}


ALL = list(L.CELLS) + [L.REF_CELL]
res = dict(spec_sha256=(H / 'SPEC.sha256').read_text().split()[0], prereg_sha256=(H / 'PREREG.sha256').read_text().split()[0],
           flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK), src_sha256=L.SRC_SHA, targets=TG,
           bench_0050=dict({k: v for k, v in mt(bench).items() if k != 'lag_days'}, yearly=yearly(bench), seg_dd=s17lib.seg_dd(bench)),
           cells={c: dict(paths={}, capacity={}) for c in ALL}, acceptance=dict(P0=True))
kw = dict(SIG=SIG, targets=TG)
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    for c in ALL:
        r = res['cells'][c]
        for cap in L.CAPS:
            nav, lg, st = L.sim(P, sg, U, c, cap=cap, **kw)
            r['paths'][f'{int(cap/1e4)}萬|+{s}'] = mt(nav)
            if s == 0 and cap == 1e6:
                x = lg[lg['why'] != 'open_end']
                sf = st['stockfrac']; first = lg['ent_date'].min()
                lag = longest_lag(nav, bench)
                m = dict(r['paths']['100萬|+0'], n_closed=int(len(x)), n_buy=int(st['n_buy']), win_rate=float((x['ret'] > 0).mean()),
                         avg_npos=float(st['avg_npos']), stockfrac_mean=float(sf.loc[sf.index >= first].mean()),
                         rb_up=int(st['rb_up']), rb_down=int(st['rb_down']), rb_trim_orders=int(st['rb_trim_orders']), fee_tax_slip=float(st['fee'] + st['tax'] + st['slip']),
                         lag=dict(days=int(lag[0]), start=lag[1], end=lag[2]), top5_share=s17lib.top5_share(lg, nav, 1e6),
                         yearly=yearly(nav), seg_dd=s17lib.seg_dd(nav), end_nav=float(nav.iloc[-1]))
                ss = st['satser'].dropna()
                if len(ss):
                    m['sat_target'] = dict(n=int(len(ss)), mean=float(ss.mean()), median=float(ss.median()), p10=float(ss.quantile(.1)), p90=float(ss.quantile(.9)),
                                           frac_lo=float((ss <= L.VT_LO + 1e-9).mean()), frac_hi=float((ss >= L.VT_HI - 1e-9).mean()),
                                           by_year={int(y): float(v) for y, v in ss.groupby(ss.index.year).mean().items()})
                if L.CELLS.get(c, {}).get('kind') == 'A':
                    S2, _ = s10lib.variant_S(sg, 'E0')
                    W = L.make_wmult(S2, U, SIG, L.CELLS[c]['cap_div'])
                    mb = [W.get(int(a), {}).get(int(j), 1.0) for a, j in zip(x['sig_t'], x['j'])]
                    m['buy_mult'] = dict(mean=float(np.mean(mb)), median=float(np.median(mb)), frac_gt1=float(np.mean(np.array(mb) > 1 + 1e-9)))
                    del S2, W
                m['since2019_costx2'] = stlib.cagr(L.sim(P, sg, U, c, cap=cap, cost='strict_x2', **kw)[0], D19, DE)
                r['main'] = m
                if c == 'CUR':
                    nav0, lg0, _ = L.sim(P, sg, U, c, cap=cap, orig=True)
                    cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
                    res['acceptance']['A3_main'] = bool(np.array_equal(nav.to_numpy(), nav0.to_numpy()) and lg[cols].equals(lg0[cols]))
                    navc = nav
                    del nav0, lg0
            del nav, lg, st
        _, _, st = L.sim(P, sg, U, c, cap=1e7, **kw)
        r['capacity'][f'1000萬|+{s}'] = L.capped_rate(st)
        del st
    del sg; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

for c in ALL:
    r = res['cells'][c]; pp = list(r['paths'].values())
    g = lambda k: np.array([v[k] for v in pp])
    r['path_summary'] = dict(since2019_median=float(np.median(g('since2019'))), since2019_min=float(g('since2019').min()), since2019_max=float(g('since2019').max()),
                             maxdd_median=float(np.median(g('maxdd'))), maxdd_worst=float(g('maxdd').min()),
                             calmar_median=float(np.median(g('calmar'))), calmar_full_median=float(np.median(g('calmar_full'))),
                             full_median=float(np.median(g('full'))), since2024_median=float(np.median(g('since2024'))), y2019_2023_median=float(np.median(g('y2019_2023'))),
                             lag_days_median=float(np.median(g('lag_days'))))
    cr = [r['capacity'][f'1000萬|+{s}']['capped_rate'] for s in L.SHIFTS]
    r['capacity_summary'] = dict(median4=float(np.median(cr)), shift0=cr[0], min4=float(min(cr)), max4=float(max(cr)))

# ---- 驗收
dl = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cm = res['cells']['CUR']['main']
a0 = dict(since2019=cm['since2019'], full=cm['full'], maxdd=cm['maxdd'], n_closed=cm['n_closed'])
a0['tol_ok'] = bool(abs(a0['since2019'] - L.ACC['since2019']) < 2.5e-4 and abs(a0['full'] - L.ACC['full']) < 2.5e-4
                    and abs(a0['maxdd'] - L.ACC['maxdd']) < 2.5e-4 and a0['n_closed'] == L.ACC['n_closed'])
a0['bitwise_vs_deadline_ALL'] = bool(cm['since2019'] == dl['standard']['ALL']['since2019'] and cm['full'] == dl['standard']['ALL']['full']
                                     and cm['maxdd'] == dl['standard']['ALL']['maxdd_full'])
res['acceptance']['A0'] = a0
cp = res['cells']['CUR']['paths']
d2 = [abs(cp[k]['since2019'] - v['since2019']) for k, v in dl['paths']['ALL'].items()]
d2d = [abs(cp[k]['maxdd'] - v['maxdd_full']) for k, v in dl['paths']['ALL'].items()]
res['acceptance']['A2'] = dict(n=len(d2), max_abs_diff=float(max(d2)), max_abs_diff_maxdd=float(max(d2d)), ok=bool(len(d2) == 20 and max(d2) < 1e-9 and max(d2d) < 1e-9))
s47 = json.loads((L.AG / 's47_allocation_family' / 'results.json').read_text())['cells']['S70_D8']['paths']
rp = res['cells'][L.REF_CELL]['paths']
res['acceptance']['REF_vs_S47_S70_D8_maxdiff'] = float(max(abs(rp[k]['since2019'] - s47[k]['since2019']) for k in rp))
off = Path('/home/user/-/xq_scripts/site/navs_official.csv.gz')
o = pd.read_csv(off, parse_dates=['date']).set_index('date')['E0|W10_70']
dd_ = (navc.reindex(o.index) - o).abs()
res['acceptance']['official_nav'] = dict(n=int(len(o)), max_abs_diff_yuan=float(dd_.max()), ok=bool(dd_.notna().all() and dd_.max() <= 0.0051))
assert a0['tol_ok'] and a0['bitwise_vs_deadline_ALL'] and res['acceptance']['A2']['ok'] and res['acceptance']['A3_main'] and res['acceptance']['official_nav']['ok'], f"驗收失敗 {res['acceptance']}"


# ---- 判定（事前規則）
def pair(k):
    pp, q = res['cells'][k]['paths'], cp
    d = np.array([pp[x]['since2019'] - q[x]['since2019'] for x in q]) * 100
    d1 = np.array([pp[x]['y2019_2023'] - q[x]['y2019_2023'] for x in q]) * 100
    d2_ = np.array([pp[x]['since2024'] - q[x]['since2024'] for x in q]) * 100
    ddd = np.array([pp[x]['maxdd'] - q[x]['maxdd'] for x in q]) * 100
    dc = np.array([pp[x]['calmar'] - q[x]['calmar'] for x in q])
    dlg = np.array([pp[x]['lag_days'] - q[x]['lag_days'] for x in q])
    npos, md = int((d > 0).sum()), float(np.median(d))
    ncp, mc = int((dc > 0).sum()), float(np.median(dc))
    mdd = float(np.median(ddd))
    cl = 'Calmar 較好' if (ncp >= 15 and mc >= 0.10) else ('Calmar 較差' if (ncp <= 5 and mc <= -0.10) else 'Calmar 差不多')
    if npos >= 15 and md >= 2:
        rl = '報酬較好（穩定）' if (np.median(d1) > 0 and np.median(d2_) > 0) else '報酬較好（只靠一段）'
    elif npos <= 5 and md <= -2:
        rl = '報酬較差'
    else:
        rl = '報酬差不多'
    dn = '回撤較深' if mdd <= -2 else ('回撤較淺' if mdd >= 2 else '回撤相近')
    goal = bool(npos >= 15 and md >= 1 and mdd >= -0.5)
    cand = bool(cl == 'Calmar 較好' and rl != '報酬較差' and mdd > -1)
    capd = (res['cells'][k]['capacity_summary']['median4'] - res['cells']['CUR']['capacity_summary']['median4']) * 100
    return dict(n_pos=npos, median_pp=md, seg2019_2023_median_pp=float(np.median(d1)), seg2024_median_pp=float(np.median(d2_)),
                maxdd_median_pp=mdd, n_dd_shallower=int((ddd > 0).sum()), calmar_n_pos=ncp, calmar_median=mc, lag_days_median_diff=float(np.median(dlg)),
                capacity_diff_pp=float(capd), calmar_label=cl, return_label=rl, dd_label=dn, goal='達成' if goal else '未達成',
                followup_candidate=cand, capacity_note='容量變差' if capd >= 3 else '')


pairs = {k: pair(k) for k in L.FAMILY + [L.REF_CELL]}
cands = [k for k in L.FAMILY if pairs[k]['followup_candidate']]
goals = [k for k in L.FAMILY if pairs[k]['goal'] == '達成']
verdict = ('(a) 波動調整有值得驗證的候選' if cands else
           ('(b) 有加碼好處但 Calmar 沒明顯改善，屬風險偏好' if goals else '(c) 波動調整在本策略沒有改善風險調整報酬，不採用'))
res['decision'] = dict(pairs=pairs, followup_candidates=cands, goal_achieved=goals, verdict=verdict, ref_note='REF_S70_D8 只描述，不進判定。')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
code = {n: hashlib.sha256((H / n).read_bytes()).hexdigest() for n in ('s53lib.py', 'prep.py', 'prereg_write.py', 'run.py')}
(H / 'code.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in code.items()))
for k in ['CUR'] + L.FAMILY + [L.REF_CELL]:
    ps = res['cells'][k]['path_summary']; cs = res['cells'][k]['capacity_summary']; pr = pairs.get(k, {})
    print(k, f"{ps['since2019_median']*100:.2f} {ps['maxdd_median']*100:.2f} {ps['calmar_median']:.3f} cap {cs['median4']*100:.1f}",
          {kk: pr[kk] for kk in ('n_pos', 'median_pp', 'maxdd_median_pp', 'calmar_n_pos', 'calmar_median', 'calmar_label', 'return_label', 'goal')} if pr else '')
print(verdict, '完成', res['meta'], res['acceptance'])

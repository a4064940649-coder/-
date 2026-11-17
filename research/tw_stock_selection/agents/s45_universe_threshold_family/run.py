#!/usr/bin/env python3
"""S45：股池門檻家族 15 格（60 日均成交值 {300,500,1000,2000,3000} 萬 × 收盤 {5,10,20} 元），W10_70、F1＋E0、嚴成本、三個正式開關全開。
每格：100 萬主跑、成本加倍、20 條路徑、容量（1,000 萬無上限、1,000 萬／3,000 萬 5% 上限）。
python3 -I -B run.py → results.json、navs.csv.gz、trades.csv.gz。SPEC.json／PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s45lib as L
from s45lib import s17lib, s10lib, s4lib, stlib, cslib

H = Path(__file__).resolve().parent
for n in ('SPEC', 'PREREG'):
    assert hashlib.sha256((H / f'{n}.json').read_bytes()).hexdigest() == (H / f'{n}.sha256').read_text().split()[0], f'{n}.json 與 sha256 不符'
PRE = json.loads((H / 'PREREG.json').read_text())
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U_ref = stlib.universe(P, 'twse') & ~INNOV
U = {c['id']: L.universe(P, INNOV, c['adv'], c['px']) for c in L.CELLS}
A1 = bool(np.array_equal(U[L.CUR], U_ref))
assert A1, 'A1 失敗'
V60 = P.V60.astype(np.float32)
del U_ref, INNOV
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
mt = lambda nav: dict({k: stlib.cagr(nav, a, b) for k, (a, b) in WIN.items()}, maxdd=stlib.maxdd(nav, D0, DE))
AMT, O = P.AMT, P.O


def yearly(nav):
    y = nav.groupby(nav.index.year).last()
    prev = pd.concat([pd.Series([nav.iloc[0]], index=[y.index[0] - 1]), y.iloc[:-1]])
    prev.index = y.index
    return {int(k): float(v) for k, v in (y / prev.values - 1).items()}


def dist(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)),
                gt1pct=float((x > 0.01).mean()), gt5pct=float((x > 0.05).mean()))


res = dict(spec_sha256=(H / 'SPEC.sha256').read_text().split()[0], prereg_sha256=(H / 'PREREG.sha256').read_text().split()[0],
           flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           bench_0050=dict(mt(bench), yearly=yearly(bench)), cells={}, acceptance=dict(P0=True, A1_universe_bitwise=A1))
navs = {'0050': bench}
trades = []
for c in L.CELLS:
    cid = c['id']; u = U[cid]
    r = dict(adv=c['adv'], px=c['px'], paths={})
    for s in L.SHIFTS:
        sg = s10lib.make_signal_s10(P, u, RT, shift=s)
        for cap in L.CAPS:
            nav, lg, st = L.sim(P, sg, u, cap=cap)
            m = mt(nav)
            r['paths'][f'{int(cap/1e4)}萬|+{s}'] = m
            if s == 0 and cap == 1e6:
                x = lg[lg['why'] != 'open_end']
                r['main'] = dict(m, n_closed=int(len(x)), avg_hold_days=float(x['hold_days'].mean()), win_rate=float((x['ret'] > 0).mean()),
                                 mean_ret=float(x['ret'].mean()), avg_npos=float(st['avg_npos']), n_buy=int(st['n_buy']),
                                 top5_share=s17lib.top5_share(lg, nav, cap), yearly=yearly(nav), seg_dd=s17lib.seg_dd(nav),
                                 end_nav=float(nav.iloc[-1]))
                r['main']['since2019_costx2'] = stlib.cagr(L.sim(P, sg, u, cap=cap, cost='strict_x2')[0], D19, DE)
                navs[cid] = nav
                t = lg[['code', 'ent_date', 'exit_date', 'ret', 'pnl', 'why', 'hold_days']].copy()
                t.insert(0, 'cell', cid); trades.append(t)
                if cid == L.CUR:
                    navc = nav
        if s == 0:                                           # 容量
            cap_res = {}
            for K, cp in ((1e7, 0.0), (1e7, 0.05), (3e7, 0.05)):
                nav, lg, st = L.sim(P, sg, u, cap=K, cap_pct=cp)
                e, j = lg['ent_t'].to_numpy(), lg['j'].to_numpy()
                part = lg['shares'].to_numpy() * O[e, j] / AMT[e, j]
                part60 = lg['shares'].to_numpy() * O[e, j] / V60[e - 1, j]
                cap_res[f'{int(K/1e4)}萬|cap{cp:.2f}'] = dict(since2019=stlib.cagr(nav, D19, DE), maxdd=stlib.maxdd(nav, D0, DE), n_buy=int(st['n_buy']),
                                                             n_capped=int(st['blk_cap']), capped_per_buy=float(st['blk_cap'] / max(st['n_buy'], 1)),
                                                             participation_day=dist(part), participation_v60=dist(part60))
            r['capacity'] = cap_res
        del sg; gc.collect()
    pv = np.array([v['since2019'] for v in r['paths'].values()])
    r['path_summary'] = dict(since2019_median=float(np.median(pv)), since2019_min=float(pv.min()), since2019_max=float(pv.max()),
                             maxdd_median=float(np.median([v['maxdd'] for v in r['paths'].values()])),
                             since2024_median=float(np.median([v['since2024'] for v in r['paths'].values()])),
                             y2019_2023_median=float(np.median([v['y2019_2023'] for v in r['paths'].values()])),
                             above_0050=int((pv > res['bench_0050']['since2019']).sum()))
    res['cells'][cid] = r
    print(cid, f"主跑 {r['main']['since2019']*100:.2f}% 路徑中位數 {r['path_summary']['since2019_median']*100:.2f}%", f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---- 驗收 A0／A2
dl = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cm = res['cells'][L.CUR]['main']
a0 = dict(since2019=cm['since2019'], full=cm['full'], maxdd=cm['maxdd'], n_closed=cm['n_closed'])
a0['tol_ok'] = bool(abs(a0['since2019'] - L.ACC['since2019']) < 2.5e-4 and abs(a0['full'] - L.ACC['full']) < 2.5e-4
                    and abs(a0['maxdd'] - L.ACC['maxdd']) < 2.5e-4 and a0['n_closed'] == L.ACC['n_closed'])
a0['bitwise_vs_deadline_ALL'] = bool(cm['since2019'] == dl['standard']['ALL']['since2019'] and cm['full'] == dl['standard']['ALL']['full']
                                     and cm['maxdd'] == dl['standard']['ALL']['maxdd_full'])
res['acceptance']['A0'] = a0
d2 = [abs(res['cells'][L.CUR]['paths'][k]['since2019'] - v['since2019']) for k, v in dl['paths']['ALL'].items()]
res['acceptance']['A2'] = dict(n=len(d2), max_abs_diff=float(max(d2)), ok=bool(len(d2) == 20 and max(d2) < 1e-9))
off = Path('/home/user/-/research/tw_stock_selection/xq_scripts/site/navs_official.csv.gz')
res['acceptance']['official_nav'] = '存在（未對照：見 FINDINGS）' if off.exists() else '不存在（寫題與執行時都沒有 xq_scripts/site/navs_official.csv.gz）'
assert a0['tol_ok'] and res['acceptance']['A2']['ok'], f"驗收失敗 {res['acceptance']}"

# ---- 判定（事前規則）
ids = [c['id'] for c in L.CELLS]
med = {k: res['cells'][k]['path_summary']['since2019_median'] for k in ids}
ddm = {k: res['cells'][k]['path_summary']['maxdd_median'] for k in ids}
fam_med = float(np.median(list(med.values())))
allp = np.array([v['since2019'] for k in ids for v in res['cells'][k]['paths'].values()])
rank_ret = 1 + sum(med[k] > med[L.CUR] for k in ids)
rank_dd = 1 + sum(ddm[k] > ddm[L.CUR] for k in ids)
P3 = (med[L.CUR] - fam_med) * 100
cur_paths = res['cells'][L.CUR]['paths']
pairs = {}
for k in ids:
    if k == L.CUR:
        continue
    pp = res['cells'][k]['paths']
    d = np.array([pp[q]['since2019'] - cur_paths[q]['since2019'] for q in cur_paths]) * 100
    d1 = np.array([pp[q]['y2019_2023'] - cur_paths[q]['y2019_2023'] for q in cur_paths]) * 100
    d2_ = np.array([pp[q]['since2024'] - cur_paths[q]['since2024'] for q in cur_paths]) * 100
    ddd = np.array([pp[q]['maxdd'] - cur_paths[q]['maxdd'] for q in cur_paths]) * 100
    npos, md = int((d > 0).sum()), float(np.median(d))
    lab = '比現行好' if (npos >= 15 and md >= 2) else ('比現行差' if (npos <= 5 and md <= -2) else '差不多')
    if lab == '比現行好':
        lab = '穩定地比較好' if (np.median(d1) > 0 and np.median(d2_) > 0) else '比現行好但只靠一段'
    pairs[k] = dict(n_pos=npos, median_pp=md, seg2019_2023_median_pp=float(np.median(d1)), seg2024_median_pp=float(np.median(d2_)),
                    maxdd_median_pp=float(np.median(ddd)), n_dd_shallower=int((ddd > 0).sum()), label=lab)


def tier(k):
    g = res['cells'][k]['capacity']['1000萬|cap0.00']['participation_day']['gt5pct']
    return '1,000 萬可行' if g < 0.15 else ('1,000 萬受限' if g < 0.30 else '只適合小資金（數百萬以下）')


TORD = {'1,000 萬可行': 0, '1,000 萬受限': 1, '只適合小資金（數百萬以下）': 2}
tiers = {k: tier(k) for k in ids}
notes = {k: '報酬較高但容量較差，小資金以外不可行' for k in ids if k != L.CUR and (med[k] > med[L.CUR] or pairs[k]['label'] in ('穩定地比較好', '比現行好但只靠一段'))
         and TORD[tiers[k]] > TORD[tiers[L.CUR]]}
res['decision'] = dict(
    family_path_medians={k: med[k] for k in ids}, family_median=fam_med, family_min=float(min(med.values())), family_max=float(max(med.values())),
    family_300_median=float(np.median(allp)), family_300_p10=float(np.percentile(allp, 10)),
    cur_path_median=med[L.CUR], cur_rank_return=int(rank_ret), cur_rank_maxdd=int(rank_dd), P3_pp=float(P3),
    verdict_optimism=('現行門檻偏樂觀' if P3 >= 3 else '沒被挑過（不必因門檻打折）'),
    verdict_best=('是' if rank_ret == 1 else ('前段' if rank_ret <= 4 else '不是')),
    pairs=pairs, capacity_tier=tiers, capacity_notes=notes)
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
pd.DataFrame(navs).round(2).to_csv(H / 'navs.csv.gz', compression='gzip')
tr = pd.concat(trades, ignore_index=True)
for col in ('ent_date', 'exit_date'):
    tr[col] = pd.to_datetime(tr[col]).dt.strftime('%Y-%m-%d')
tr['ret'] = tr['ret'].round(5); tr['pnl'] = tr['pnl'].round(0)
tr.to_csv(H / 'trades.csv.gz', index=False, compression='gzip')
code = {n: hashlib.sha256((H / n).read_bytes()).hexdigest() for n in ('s45lib.py', 'prep.py', 'prereg_write.py', 'run.py')}
(H / 'code.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in code.items()))
print(json.dumps({k: res['decision'][k] for k in ('cur_rank_return', 'cur_rank_maxdd', 'P3_pp', 'verdict_optimism', 'verdict_best', 'family_median')}, ensure_ascii=False))
print('完成', res['meta'])

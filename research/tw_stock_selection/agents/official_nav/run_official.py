#!/usr/bin/env python3
"""正式淨值序列（2026-11-17 起網站與正式數字的口徑）：除權息參考價修正＋營收期限修正全開。
產品擁有者決定正式採用 DEADLINE_FIX 與 INS15_MASK（依據 agents/deadline_engine_fix/、agents/s40_revenue_deadline_pit/、agents/s41_ins15_handling/）。
共用引擎（cslib）的三個開關預設仍是 False（過去研究照樣重現）；本程式是「正式管線」，明確開啟：
  cslib.EXRIGHT_FIX＝True、cslib.DEADLINE_FIX＝True、cslib.INS15_MASK＝True。
用法：python3 -I -B run_official.py [--no-paths]
欄位與口徑同 agents/exright_engine_fix/run_engine.py 1（產生 xq_scripts/site/navs_exright.csv.gz 的那一套；全部 W10_70、嚴成本；
標準路徑＝100 萬、位移 0；20 條路徑＝60／80／100／150／200 萬 × 位移 0～3）：
  E0｜W10_70＝現行（研究版名次、A10 換股日）：DEADLINE_FIX（a10_rows→cslib.rev_avail_days）與 INS15_MASK（s10lib.make_signal_s10）都生效。
  XT｜W10_70＝XQ 門檻版（MinSurge 2.5／ExitSurge 2.0、腳本股池）、A10 換股日（＝exright_engine_fix 的 XA）：DEADLINE_FIX 生效；
             INS15_MASK 不適用（s19lib.make_signal_thr 不讀這個開關；XQ 回測用 XQ 自己的營收資料時點，引擎不另外模擬）。
  XT_B11｜W10_70＝XQ 門檻版、舊換股日 B11（＝exright_engine_fix 的 XT，只供對照）：B11 自己算換股日，兩個期限開關都不影響（只有除權息修正）。
輸出：xq_scripts/site/navs_official.csv.gz（欄位同 navs_exright.csv.gz）、out/res_official.json（指標、逐年、路徑、網站 risk 用的附帶數字、驗收）。
驗收：E0 標準路徑 2019 起＝48.76%（deadline_engine_fix ALL／S40 ALL），差 ≤ 0.02 個百分點；並和 deadline_engine_fix/out/nav_std.csv.gz 的 ALL 欄逐日比對。
暫存（面板快取）放 scratchpad/offfix/。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
SITE = Path('/home/user/-/xq_scripts/site')
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/offfix/cache')
TMP.mkdir(parents=True, exist_ok=True)
OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)
PATHS = '--no-paths' not in sys.argv
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)
assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False, '共用引擎預設必須是關閉'
cslib.EXRIGHT_FIX = True
cslib.DEADLINE_FIX = True
cslib.INS15_MASK = True

P = s19lib.load()
assert P.exright_fix
s19lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
UX, Cadj, liqX = s19lib.script_universe(P, INNOV)
del Cadj, liqX, INNOV
S, _ = s19lib.rev_std(P)
months = list(S.index)
avA = s19lib.a10_rows(P, months)                           # DEADLINE_FIX 生效
avB, _dl = s19lib.b11_rows(P, months, 11)                  # 不受期限開關影響
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入', P.exright_info, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
STRATS = ('E0', 'XA', 'XT')                                 # 輸出欄名：E0、XT（＝XA，A10）、XT_B11（＝XT，B11）
COLNAME = {'E0': 'E0|W10_70', 'XA': 'XT|W10_70', 'XT': 'XT_B11|W10_70'}


def build(sid, s):
    if sid == 'E0':
        return s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=s)
    av = avB if sid == 'XT' else avA
    return s19lib.make_signal_thr(P, U, UX, None, S, av, 2.5, 2.0, shift=s)


def sim(sid, Sg, cap):
    return s10lib.simulate_s10(P, Sg, UX if sid in ('XT', 'XA') else U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap)


def yearly(nav):
    yr = nav.groupby(nav.index.year).last(); prev = nav.iloc[0]; out = {}
    for y in yr.index:
        out[int(y)] = float(yr.loc[y] / prev - 1); prev = yr.loc[y]
    return out


def extras(Sg, nav, lg):                                     # 同 exright_engine_fix/run_engine.py 的 extras（網站 risk.json／README 第 12 節）
    b = pd.Series(P.E50, index=P.IDX)
    DM = P.IDX[(P.t0 + P.T - 1) // 2]
    x = lg[lg['why'] != 'open_end']
    w = x[x['ret'] > 0]['ret']; l = x[x['ret'] <= 0]['ret']
    top = x.sort_values('pnl', ascending=False).head(5)
    blk, _ = s2lib.block_from(lg, 5)
    nb = s10lib.simulate_s10(P, Sg, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6, block=blk)[0]
    n2 = s10lib.simulate_s10(P, Sg, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict_x2'], capital=1e6)[0]
    rr = ((nav / nav.shift(252) - 1) - (b.reindex(nav.index) / b.reindex(nav.index).shift(252) - 1)).loc['2016-03-01':].dropna()
    yb = yearly(b.loc[D0:DE]); ys = yearly(nav)
    return dict(win_rate=float(len(w) / len(x)), avg_win=float(w.mean()), avg_loss=float(l.mean()), n_closed=int(len(x)),
                avg_hold_days=float(x['hold_days'].mean()),
                top5_pnl_share=float(top['pnl'].sum() / x['pnl'].sum()), top5=[str(c) for c in top['code']],
                block5_since2019=stlib.cagr(nb, D19, DE), block5_full=stlib.cagr(nb, D0, DE),
                costx2_since2019=stlib.cagr(n2, D19, DE), costx2_full=stlib.cagr(n2, D0, DE),
                half1=stlib.cagr(nav, D0, DM), half2=stlib.cagr(nav, DM, DE),
                roll252_worst=float(rr.min()), roll252_worst_end=str(rr.idxmin().date()), roll252_frac_behind=float((rr < 0).mean()),
                yearly_0050=yb, beat_years=int(sum(ys[y] > yb[y] for y in ys if y != 2015)), n_years=int(sum(1 for y in ys if y != 2015)),
                e50_since2019=stlib.cagr(b, D19, DE), e50_full=stlib.cagr(b, D0, DE), e50_since2024=stlib.cagr(b, D24, DE), e50_maxdd=stlib.maxdd(b, D0, DE))


res = dict(flags=dict(EXRIGHT_FIX=True, DEADLINE_FIX=True, INS15_MASK=True), exright_info=P.exright_info,
           start=str(D0.date()), end=str(DE.date()), T=int(P.T), N=int(P.N), columns=COLNAME,
           ins15_applies={'E0': True, 'XA': False, 'XT': False}, deadline_applies={'E0': True, 'XA': True, 'XT': False},
           av_changed_vs_rule={str(m): str(P.IDX[avA[m]].date()) for m in cslib.deadline_exceptions() if m in avA and avA[m] < P.T}, strats={})
navs = {}
for sid in STRATS:
    R = dict(paths={}, paths_dd={})
    for s in (range(4) if PATHS else (0,)):
        Sg = build(sid, s)
        if s == 0:
            R['n_signal_months'] = len(Sg['info'])
            R['targets'] = {x['month']: x['targets'] for x in Sg['info']}
            R['sig_dates'] = {x['month']: x.get('sig_date', str(P.IDX[x['k']].date())) for x in Sg['info']}
        for cap in ((6e5, 8e5, 1e6, 1.5e6, 2e6) if PATHS else (1e6,)):
            nav, lg, st = sim(sid, Sg, cap)
            key = f'{int(cap/1e4)}萬|+{s}'
            R['paths'][key] = stlib.cagr(nav, D19, DE); R['paths_dd'][key] = stlib.maxdd(nav, D0, DE)
            if s == 0 and cap == 1e6:
                sd = nav.loc[D0:DE]; dd = sd / sd.cummax() - 1; tr = dd.idxmin(); pk = sd.loc[:tr].idxmax()
                R['std'] = dict(cagr_since2019=stlib.cagr(nav, D19, DE), cagr_full=stlib.cagr(nav, D0, DE), cagr_since2024=stlib.cagr(nav, D24, DE),
                                maxdd=stlib.maxdd(nav, D0, DE), dd_peak=str(pk.date()), dd_trough=str(tr.date()), final_nav=float(nav.iloc[-1]),
                                n_trades_closed=int((lg['why'] != 'open_end').sum()), n_trades_all=int(len(lg)), yearly=yearly(nav),
                                open_end=sorted(lg.loc[lg['why'] == 'open_end', 'code'].astype(str).tolist()))
                navs[sid] = nav
                if sid == 'E0':
                    R['extras'] = extras(Sg, nav, lg)
            del nav, lg, st
        del Sg; gc.collect()
    if PATHS:
        v = np.array(list(R['paths'].values())); d = np.array(list(R['paths_dd'].values()))
        R['paths_summary'] = dict(cagr19_median=float(np.median(v)), cagr19_min=float(v.min()), cagr19_max=float(v.max()),
                                  maxdd_median=float(np.median(d)), maxdd_worst=float(d.min()))
    res['strats'][COLNAME[sid]] = R
    print(sid, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in R['std'].items() if k not in ('yearly', 'open_end')},
          R.get('paths_summary'), f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---- 驗收
e0 = res['strats']['E0|W10_70']['std']
dl = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())['standard']['ALL']
ref = pd.read_csv(AG / 'deadline_engine_fix' / 'out' / 'nav_std.csv.gz', index_col=0, parse_dates=True)['ALL']
a = navs['E0'].reindex(ref.index)
acc = dict(target=0.4876, since2019=e0['cagr_since2019'], deadline_engine_fix_ALL=dl['since2019'],
           gap_vs_4876_pp=(e0['cagr_since2019'] - 0.4876) * 100, gap_vs_dlfix_pp=(e0['cagr_since2019'] - dl['since2019']) * 100,
           nav_vs_dlfix_ALL_max_rel=float(np.nanmax(np.abs(a.to_numpy() / ref.to_numpy() - 1))))
acc['pass'] = bool(abs(acc['gap_vs_4876_pp']) <= 0.02 and abs(acc['gap_vs_dlfix_pp']) <= 0.02)
res['acceptance'] = acc
res['peak_mb'] = rss(); res['secs'] = round(time.time() - t0)
json.dump(res, open(OUT / 'res_official.json', 'w'), ensure_ascii=False, default=str, indent=1)
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]
df = pd.DataFrame({'0050': bench / bench.iloc[0] * 1e6, **{COLNAME[k]: navs[k] for k in STRATS}}).round(2)
df.index.name = 'date'
df.to_csv(SITE / 'navs_official.csv.gz')
print('acceptance', acc)
print('done', res['secs'], 's', res['peak_mb'], 'MB')
assert acc['pass'], '驗收失敗'

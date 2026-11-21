#!/usr/bin/env python3
"""正式淨值序列（2026-11-21 起網站與正式數字的口徑）：除權息參考價修正＋營收期限修正＋減資復牌修正＋0050 官方含息序列，五個開關全開。
產品擁有者決定正式採用（依據 agents/deadline_engine_fix/、agents/s40_revenue_deadline_pit/、agents/s41_ins15_handling/、agents/r_fix_e50_reduce/FINDINGS.md）。
共用引擎（cslib）的開關預設仍是關（過去研究照樣重現）；本程式是「正式管線」，明確開啟：
  cslib.EXRIGHT_FIX＝True、cslib.DEADLINE_FIX＝True、cslib.INS15_MASK＝True（2026-11-17 起）；
  cslib.REDUCE_FIX＝True（減資／面額變更恢復買賣日用事件參考價，raw/twse_reduce/twse_reduce_events.csv）、
  cslib.E50_SOURCE＝cslib.E50_OFFICIAL_CSV（0050 含息序列改用官方日 K 自建檔 raw/e50_tr_official/etf_0050_tr.csv）（2026-11-21 起）。
改版前（2026-11-17～11-20，48.76%）的程式存成 run_official_v1_pre_reduce.py（輸出到 archive_2026-11-21_pre_v2/rerun/）；改版前的正式檔備份在 archive_2026-11-21_pre_v2/。
用法：python3 -I -B run_official.py [--no-paths]
欄位與口徑同 agents/exright_engine_fix/run_engine.py 1（產生 xq_scripts/site/navs_exright.csv.gz 的那一套；全部 W10_70、嚴成本；
標準路徑＝100 萬、位移 0；20 條路徑＝60／80／100／150／200 萬 × 位移 0～3）：
  E0｜W10_70＝現行（研究版名次、A10 換股日）：DEADLINE_FIX（a10_rows→cslib.rev_avail_days）與 INS15_MASK（s10lib.make_signal_s10）都生效。
  XT｜W10_70＝XQ 門檻版（MinSurge 2.5／ExitSurge 2.0、腳本股池）、A10 換股日（＝exright_engine_fix 的 XA）：DEADLINE_FIX 生效；
             INS15_MASK 不適用（s19lib.make_signal_thr 不讀這個開關；XQ 回測用 XQ 自己的營收資料時點，引擎不另外模擬）。
  XT_B11｜W10_70＝XQ 門檻版、舊換股日 B11（＝exright_engine_fix 的 XT，只供對照）：B11 自己算換股日，兩個期限開關都不影響。
輸出：xq_scripts/site/navs_official.csv.gz（欄位同 navs_exright.csv.gz）、out/res_official.json（指標、逐年、路徑、網站 risk 用的附帶數字、驗收）、
      out/trades_<欄>.csv（標準路徑交易明細）。
驗收（任一不符就結束碼非 0）：E0 標準路徑 2019 起 48.86%、全期 34.46%、最大回撤 −31.61%、已出場 381 筆、20 條路徑中位數 46.42%（差 ≤ 0.005 個百分點），
  並與稽核獨立複製的 agents/r_fix_e50_reduce/out/AB/navs.csv.gz 四欄逐日差 0（取代原本對 deadline_engine_fix ALL 的比對）。
暫存（面板快取，只有原始日 K 面板，與開關無關）放 scratchpad/offfix/cache。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
SITE = Path('/home/user/-/xq_scripts/site')
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/offfix/cache')
TMP.mkdir(parents=True, exist_ok=True)
OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)
AB = AG / 'r_fix_e50_reduce' / 'out' / 'AB'                 # 稽核獨立複製的 A＋B 結果（驗收對照）
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
assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None, '共用引擎預設必須是關閉'
cslib.EXRIGHT_FIX = True
cslib.DEADLINE_FIX = True
cslib.INS15_MASK = True
cslib.REDUCE_FIX = True                                    # 2026-11-21 起：減資／面額變更恢復買賣日用事件參考價
cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV                  # 2026-11-21 起：0050 官方含息序列

P = s19lib.load()
assert P.exright_fix and P.reduce_fix and P.e50_source == str(cslib.E50_OFFICIAL_CSV)
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
print('載入', P.exright_info, P.reduce_info, P.e50_source, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

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


res = dict(flags=dict(EXRIGHT_FIX=True, DEADLINE_FIX=True, INS15_MASK=True, REDUCE_FIX=True, E50_SOURCE=str(cslib.E50_OFFICIAL_CSV)),
           exright_info=P.exright_info, reduce_info=P.reduce_info, e50_missing_days=int(P.e50_missing),
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
                lg.drop(columns=[c for c in ('j',) if c in lg.columns]).to_csv(OUT / f"trades_{COLNAME[sid].split('|')[0]}.csv", index=False)
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

# ---- 驗收（對稽核獨立複製的 A＋B：agents/r_fix_e50_reduce/out/AB）
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]
df = pd.DataFrame({'0050': bench / bench.iloc[0] * 1e6, **{COLNAME[k]: navs[k] for k in STRATS}}).round(2)
df.index.name = 'date'
df.to_csv(SITE / 'navs_official.csv.gz')
e0 = res['strats']['E0|W10_70']
TARGET = dict(cagr_since2019=0.4886, cagr_full=0.3446, maxdd=-0.3161, n_trades_closed=381, paths_median=0.4642)
acc = dict(target=TARGET, since2019=e0['std']['cagr_since2019'], full=e0['std']['cagr_full'], maxdd=e0['std']['maxdd'], n_trades_closed=e0['std']['n_trades_closed'],
           paths_median=e0['paths_summary']['cagr19_median'] if PATHS else None, e50_since2019=e0['extras']['e50_since2019'])
ok = (abs(acc['since2019'] - 0.4886) <= 5e-5 and abs(acc['full'] - 0.3446) <= 5e-5 and abs(acc['maxdd'] + 0.3161) <= 5e-5 and acc['n_trades_closed'] == 381
      and (not PATHS or abs(acc['paths_median'] - 0.4642) <= 5e-5))
a = pd.read_csv(SITE / 'navs_official.csv.gz'); b = pd.read_csv(AB / 'navs.csv.gz')
same = a.shape == b.shape and list(a.columns) == list(b.columns) and list(a['date']) == list(b['date'])
num = [c for c in a.columns if c != 'date']
acc['nav_vs_audit_AB_same_shape'] = bool(same)
acc['nav_vs_audit_AB_max_abs_diff'] = float((a[num] - b[num]).abs().max().max()) if same else None
ab = json.loads((AB / 'res.json').read_text())['strats']
mine = json.loads(json.dumps(res['strats'], ensure_ascii=False, default=str))      # 轉成和存檔相同的型別再比（年份鍵是字串）
acc['targets_same_as_audit_AB'] = bool(all(mine[k]['targets'] == ab[k]['targets'] for k in ab))
acc['std_same_as_audit_AB'] = bool(all(mine[k]['std'] == ab[k]['std'] for k in ab))
acc['paths_extras_same_as_audit_AB'] = bool(all(mine[k].get('paths') == ab[k].get('paths') and mine[k].get('extras') == ab[k].get('extras') for k in ab)) if PATHS else None
acc['pass'] = bool(ok and same and acc['nav_vs_audit_AB_max_abs_diff'] == 0 and acc['targets_same_as_audit_AB'] and acc['std_same_as_audit_AB'])
res['acceptance'] = acc
res['peak_mb'] = rss(); res['secs'] = round(time.time() - t0)
json.dump(res, open(OUT / 'res_official.json', 'w'), ensure_ascii=False, default=str, indent=1)
print('acceptance', acc)
print('done', res['secs'], 's', res['peak_mb'], 'MB')
assert acc['pass'], '驗收失敗'

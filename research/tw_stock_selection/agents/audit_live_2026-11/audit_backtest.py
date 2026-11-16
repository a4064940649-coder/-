#!/usr/bin/env python3
"""稽核第 4 項：用 s20_persistent_growth 的 s20lib／s17lib 引擎（不改）重跑 E0／W10_70 標準路徑（100 萬、位移 0、嚴成本、全期連續），
核對 xq_scripts/README.md 第 12 節：2019 年起每年 48.13%、2015-07 起每年 34.08%、最大回撤 −33.4%、379 筆交易。
另把淨值和 s19_xq_threshold/navs.csv.gz 的 E0|W10_70 欄逐日比對。
快取一律放 scratchpad/audit/bt（結束刪）。輸出 audit_backtest.json（本資料夾）。python3 -I -B audit_backtest.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/audit/bt')
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                                   # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
# 快取改到稽核暫存區（不碰其他研究的暫存）
s20lib.S20 = TMP; s17lib.S17 = TMP; s10lib.S10 = TMP; s9lib.S9 = TMP; s8lib.S8 = TMP; s7lib.S7 = TMP; s6lib.S6 = TMP; s5lib.S5 = TMP
s4lib.S4 = TMP; s3lib.S3 = TMP; s2lib.S2 = TMP; stlib.S1 = TMP; cslib.TMP = TMP

t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s20lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
G = s20lib.growth_features(P)
RT = dict(S=G['S'], av=G['av'])
susp = s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
S2, ekw = s10lib.variant_S(S0, 'E0')
nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, sat_w=0.7, sizing='W10', lock=False, **ekw)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, D24 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2024-01-01')
c19, cfull, c24 = stlib.cagr(nav, D19, DE), stlib.cagr(nav, D0, DE), stlib.cagr(nav, D24, DE)
mdd = stlib.maxdd(nav, D0, DE)
s = nav.loc[D0:DE]; dd = s / s.cummax() - 1; trough = dd.idxmin(); peak = s.loc[:trough].idxmax()
n_all = int(len(lg)); n_closed = int((lg['why'] != 'open_end').sum())
tsx = stlib.trade_stats(lg)
# 和 s19 navs.csv.gz 逐日比
nv = pd.read_csv(AG / 's19_xq_threshold' / 'navs.csv.gz', parse_dates=['date']).set_index('date')
ref = nv['E0|W10_70']
j = ref.index.intersection(nav.index)
rel = (nav.reindex(j) / ref.reindex(j) - 1).abs()
# 期末持股（2026-10-22 開倉中）給名單稽核用
open_end = lg[lg['why'] == 'open_end']
oe = [dict(code=str(r['code']) if 'code' in lg.columns else None) for _, r in open_end.iterrows()]
cols = [c for c in ('code', 'ent_date', 'shares', 'ent_px') if c in lg.columns]
out = dict(
    engine='s20lib→s17lib→s10lib.simulate_s10（E0、sat_w 0.7、W10、lock False、COSTS strict、100 萬、位移 0、全期連續）；s17lib.e50_fix_suspension 已套用',
    panel=dict(T=int(P.T), N=int(P.N), first=str(P.IDX[0].date()), last=str(DE.date()), start=str(D0.date())),
    e50_suspension_fix=susp,
    cagr_since2019=c19, cagr_full=cfull, cagr_since2024=c24, maxdd=mdd, dd_peak=str(peak.date()), dd_trough=str(trough.date()),
    final_nav=float(nav.iloc[-1]), n_trades_all=n_all, n_trades_closed=n_closed, n_open_end=n_all - n_closed,
    win_rate=tsx.get('win_rate'), avg_win=tsx.get('avg_win'), avg_loss=tsx.get('avg_loss'),
    bench_cagr_since2019=stlib.cagr(bench, D19, DE), bench_cagr_full=stlib.cagr(bench, D0, DE), bench_maxdd=stlib.maxdd(bench, D0, DE),
    vs_s19_navs=dict(n_days=int(len(j)), max_abs_rel_diff=float(rel.max()), final_ref=float(ref.iloc[-1])),
    open_end_positions=open_end[cols].astype(str).to_dict('records') if cols else [],
    lg_columns=list(map(str, lg.columns)),
    peak_rss_mb=rss(), seconds=round(time.time() - t0),
)
json.dump(out, open(HERE / 'audit_backtest.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ('open_end_positions', 'e50_suspension_fix')}, ensure_ascii=False, default=str))

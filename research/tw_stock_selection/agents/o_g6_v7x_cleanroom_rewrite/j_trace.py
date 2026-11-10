# R4（解封後；J 角色）分歧逐筆追蹤：同一份 K 名單 × 同一設定，重寫引擎（xengine）與原引擎（cz_v7x.CSim，只讀 import）
# 各跑一次並記錄每一筆成交，找出第一個不同的日期、股票、股數、現金；印出該日原始資料（開收盤、PC、V60、名次、漲停、處置）。
# 用法：python3 -I -B audit_run.py J j_trace.py <variant> <j> <seed> <cell> [out.json]
import sys, json, math, pickle
import numpy as np, pandas as pd
HERE = '/home/user/-/research/tw_stock_selection/agents/o_g6_v7x_cleanroom_rewrite/'
AG = '/home/user/-/research/tw_stock_selection/agents/'
sys.path.insert(0, HERE)
import xdata, xengine
sys.path.insert(0, AG + 'o_engine_delist_consideration'); import cz_v7x as CV; import cz_engine as CE; sys.path.pop(0)
D = CV.D; X = CV.X
var, j, sd, cell = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
outp = sys.argv[5] if len(sys.argv) > 5 else None
KD = pickle.load(open(xdata.SCR.replace('g6A/', 'g6K_lists/') + 'k_all_lists.pkl', 'rb'))
pl = KD['variants'][var][(j, sd)]
merger = 'orig' if var == 'orig' else 'cons_main'
CELLS = {'N100': ('normal', 1e6), 'S100': ('stress', 1e6), 'N300': ('normal', 3e6), 'N1000': ('normal', 1e7), 'O30': ('normal', 3e5)}
cost, tier = CELLS[cell]
# ---- 原引擎 ----
VR = json.load(open(str(CV.V7X / 'results.json')))
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
ST = None
if merger == 'cons_main':
    E = pd.read_csv(AG + 'o_engine_delist_consideration/events_v2.csv', dtype={'code': str, 'successor': str})
    R2 = json.load(open(AG + 'o_engine_delist_consideration/results.json', encoding='utf-8'))
    dd = R2['PREREG']['events_v2']['summary']['d']['d_used']
    ST = CE.build_settlement(E, 'main', d=dd, TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
oplan = sorted([(int(D.TPOS[pd.Timestamp(e)]), np.array([D.CI[c] for c in pk], int), D.TPOS[pd.Timestamp(s)]) for s, e, pk in pl], key=lambda x: x[0])
cfg = dict(tier=tier, mode='GROW', model='K5T', cost=cost); cfg.update(BASE)
sm = CV.CSim(oplan, cfg, settle=ST, cmode='orig' if merger == 'orig' else 'consideration')
OT = []
def wrap(name):
    f = getattr(sm, name)
    def g(t, jj, *a, **k):
        s0 = sm.S[jj]; c0 = sm.cash + sm.cash_late
        r = f(t, jj, *a, **k)
        ds = sm.S[jj] - s0; dc = sm.cash + sm.cash_late - c0
        if abs(ds) > 1e-9: OT.append((str(D.IDX[t].date()), D.COLS[jj], float(ds), float(dc), name))
        return r
    setattr(sm, name, g)
for nm in ('fill_whole', 'fill_odd', 'fill_disp_sell'): wrap(nm)
onav = sm.run()
odates = D.IDX[oplan[0][0]:oplan[0][0] + len(onav)]
# ---- 重寫引擎 ----
import os, j_patch
if 'tiecol' in os.environ.get('G6_PATCH', ''): j_patch.patch_tiecol()
if 'oddlots' in os.environ.get('G6_PATCH', ''): j_patch.patch_oddlots()
W = j_patch.patch(xengine.World(), ','.join(x for x in os.environ.get('G6_PATCH', '').split(',') if x not in ('tiecol', 'oddlots'))); dates = W.dates
mplan = [(int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(s)))), int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(e)))), [W.ci[c] for c in pk]) for s, e, pk in pl]
end = int(np.searchsorted(dates.values, np.datetime64(odates[-1])))
mcfg = dict(tier=int(tier), merger=merger, record=True)
if cost == 'stress': mcfg['cost'] = 'stress'
ms = xengine.Sim(W, mplan, end, mcfg); mnav = ms.run()
MT = [(str(dates[t].date()), W.codes[jj], float(n), float(cash), '') for t, jj, n, cash in ms.trades]
# ---- 比對 ----
def agg(T):
    d = {}
    for dt, c, n, cash, _ in T:
        a = d.setdefault((dt, c), [0.0, 0.0]); a[0] += n; a[1] += cash
    return d
A, B = agg(MT), agg(OT)
keys = sorted(set(A) | set(B))
first = None; ndiff = 0
for k in keys:
    a = A.get(k, [0, 0]); b = B.get(k, [0, 0])
    if abs(a[0] - b[0]) > 0.5 or abs(a[1] - b[1]) > max(1.0, 1e-6 * abs(b[1])):
        ndiff += 1
        if first is None: first = (k, a, b)
ms_ser = pd.Series(mnav[mplan[0][0]:end + 1], index=dates[mplan[0][0]:end + 1])
os_ser = (onav if isinstance(onav, pd.Series) else pd.Series(np.asarray(onav), index=odates)).reindex(ms_ser.index)
rr = (ms_ser / ms_ser.iloc[0]) / (os_ser / os_ser.iloc[0]) - 1
fnav = rr[rr.abs() > 1e-6].index.min()
out = dict(variant=var, j=j, seed=sd, cell=cell, n_trades_mine=len(MT), n_trades_orig=len(OT), n_daycode_diff=ndiff, n_daycode=len(keys),
           first_trade_diff=None, first_nav_diff_date=str(fnav.date()) if pd.notna(fnav) else None,
           final_nav_mine=float(ms_ser.iloc[-1] / ms_ser.iloc[0]), final_nav_orig=float(os_ser.iloc[-1] / os_ser.iloc[0]))
if first is not None:
    (dt, c), a, b = first
    t = int(np.searchsorted(dates.values, np.datetime64(dt))); jj = W.ci[c]; oj = D.CI[c]; ot = int(D.TPOS[pd.Timestamp(dt)])
    raw = dict(O=float(W.O[t, jj]), C=float(W.C[t, jj]), PC=float(W.PC[t, jj]), V=float(W.V[t, jj]), V60_prev=float(W.V60p[t, jj]), SIG_prev=float(W.SIGp[t, jj]),
               VR_prev=float(W.VRp[t, jj]), UP=float(W.UP[t, jj]), DN=float(W.DN[t, jj]), disp_mine=bool(W.DISP[t, jj]),
               orig_O=float(D.O[ot, oj]), orig_C=float(D.C[ot, oj]), orig_PC=float(D.PC[ot, oj]), orig_V=float(D.V[ot, oj]), orig_V60p=float(D.V60p[ot, oj]))
    out['first_trade_diff'] = dict(date=dt, code=c, mine_shares=a[0], mine_cash=a[1], orig_shares=b[0], orig_cash=b[1], raw=raw,
                                   mine_day=[x for x in MT if x[0] == dt], orig_day=[x for x in OT if x[0] == dt])
print(json.dumps(out, ensure_ascii=False, indent=1, default=float)[:6000])
if outp: json.dump(out, open(outp, 'w'), ensure_ascii=False, indent=1, default=float)
if len(sys.argv) > 6 and sys.argv[6] == 'dbg':
    print('IDX type', type(D.IDX), D.IDX[:3], 'dates', dates[:3], 'onav', onav[:3], onav[-3:], len(onav), oplan[0][0], len(D.IDX))
    print('MT', MT[:12]); print('OT', OT[:12])
if os.environ.get('G6_NAVDBG'):
    d0 = pd.Timestamp(os.environ['G6_NAVDBG'])
    w = rr[(rr.index >= d0 - pd.Timedelta(days=5)) & (rr.index <= d0 + pd.Timedelta(days=10))]
    print(w.to_string())
    # 兩邊在該日持股
    t = int(np.searchsorted(dates.values, np.datetime64(d0)))
    print('trades around', [x for x in MT if d0 - pd.Timedelta(days=3) <= pd.Timestamp(x[0]) <= d0 + pd.Timedelta(days=3)][:20])
    print('orig trades around', [x for x in OT if d0 - pd.Timedelta(days=3) <= pd.Timestamp(x[0]) <= d0 + pd.Timedelta(days=3)][:20])
    yr = (ms_ser / ms_ser.iloc[0]) / (os_ser / os_ser.iloc[0])
    print(yr.resample('YE').last().to_string())
if os.environ.get('G6_CODEDBG'):
    cc, a0, a1 = os.environ['G6_CODEDBG'].split(',')
    print('MT', [x for x in MT if x[1] == cc and a0 <= x[0] <= a1]); print('OT', [x for x in OT if x[1] == cc and a0 <= x[0] <= a1])

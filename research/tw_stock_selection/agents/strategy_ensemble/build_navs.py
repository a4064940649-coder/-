#!/usr/bin/env python3
"""strategy_ensemble：組出各候選策略的日頻淨值（全部扣成本、可成交版）→ strategy_nav.parquet。
只讀其他資料夾的既有檔案；兩條沒有現成到 2026 的淨值在這裡重建（不改原檔）：
  * DISP_PX：近似處置股 10 槽（disposal_bear_longhistory 的近似規則逐字照抄 + exec_rebuild/dlib.Sleeve），延伸到 2026-10，閒置持 0050。
    先用同一程式重現原檔欄 proxy_only_incl_2022_1M_cash_0.8（閒置現金、到 2022-12-30）做核對。
  * E0050_MA：0050 + S1 開關（timing_daily 的 S1：換股日加權指數收盤 < MA200 → 場外，t+1 收盤成交，pos_t=sig_{t-2}，現金 0，0050 進出成本）。
執行：python3 -I -B build_navs.py"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; A = HERE.parent
sys.path.insert(0, str(A / 'exec_rebuild'))
import exlib as L, dlib as Dl   # noqa: E402

adj, C, V, R1 = L.adj, L.C, L.V, L.RET
idx = adj.index; cols = adj.columns; CI = L.CI; T = len(idx)
RETN, ADJN = R1.to_numpy(), adj.to_numpy()
LIM = np.where(idx < pd.Timestamp('2015-06-01'), 0.07, 0.10)
twr = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
tx = twr.reindex(idx).ffill()
lastvalid = np.array([np.max(np.where(np.isfinite(ADJN[:, j]))[0]) if np.isfinite(ADJN[:, j]).any() else -1 for j in range(len(cols))])

# ---- 近似事件（逐字照抄 disposal_bear_longhistory/run.py）----
A6 = adj / adj.shift(6) - 1
m6 = tx / tx.shift(6) - 1
att = (A6 >= 0.32) & (A6.sub(m6, axis=0) >= 0.20)
trig = att & att.shift(1, fill_value=False) & att.shift(2, fill_value=False)
TR = trig.to_numpy()
prox = []
for c in range(len(cols)):
    i = 0; col = TR[:, c]
    while i < T:
        if col[i]:
            s, e = i + 1, i + 10
            if e < T: prox.append((cols[c], s, e))
            i = e + 1
        else:
            i += 1
prox = pd.DataFrame(prox, columns=['code', 'i0', 'i1'])

def execute(ev):
    ev = ev.sort_values(['i0']).reset_index(drop=True)
    kept_exit = {}; rows = []
    for r in ev.itertuples():
        c = r.code; j = CI[c]; i0, i1 = r.i0, r.i1
        if not np.isfinite(ADJN[i0, j]): continue
        if c in kept_exit and i0 <= kept_exit[c]: continue
        if RETN[i0, j] >= LIM[i0] - 0.01: continue
        k = i1
        if lastvalid[j] >= i1:
            while k < lastvalid[j] and not (np.isfinite(RETN[k, j]) and RETN[k, j] > -(LIM[k] - 0.006)): k += 1
        kept_exit[c] = k
        rows.append((j, i0, k, L.VN[i0, j], L.VN[k, j]))
    return rows

EP = execute(prox)
def mk_events(rows, rt=0.008):
    ev = {}
    for j, i0, k, vi, vo in rows:
        vals = [v for v in (vi, vo) if np.isfinite(v) and v > 0]
        ev.setdefault(i0, []).append((j, k, rt, 0.05 * min(vals) if vals else 0.0))
    return ev
def run(rows, idle, start='2005-01-03', end='2022-12-30', capital=1e6):
    s = Dl.Sleeve(mk_events(rows), 10, idle, True)
    t0 = idx.searchsorted(pd.Timestamp(start)); t1 = idx.searchsorted(pd.Timestamp(end), 'right') - 1
    s.deposit(capital, t0 - 1); nav = [capital]; inv = []
    for t in range(t0, t1 + 1):
        s.step(t); e = s.equity(t); nav.append(e); inv.append(s.pos_value(t) / e)
    return pd.Series(nav, index=idx[t0 - 1:t1 + 1]) / capital, float(np.mean(inv))

check = {}
orig = pd.read_parquet(A / 'disposal_bear_longhistory' / 'nav.parquet')['proxy_only_incl_2022_1M_cash_0.8']
rep, _ = run(EP, 'cash')
o = orig / orig.iloc[0]; d = (rep.reindex(o.index) - o).abs().max()
check['proxy_cash_repro_vs_original_maxabs_nav_diff'] = float(d)
print('proxy repro max |diff|', d, flush=True)
END = str(idx[-1].date())
disp_px_0050, inv1 = run(EP, '0050', end=END)
disp_px_cash, inv2 = run(EP, 'cash', end=END)
check['DISP_PX_mean_invested_share'] = {'idle0050': round(inv1, 3), 'cash': round(inv2, 3)}
check['n_proxy_events_executed'] = len(EP)

# ---- 0050 + S1 開關（照 timing_daily/run.py overlay）----
nav_main = pd.read_parquet(A / 'revenue_longhistory' / 'oos_nav.parquet')
CAL = nav_main.index                                   # 共同日曆＝主策略淨值日曆（面板日，2006-04-11～2026-10-12）
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = e50.reindex(e50.index.union(CAL)).ffill().reindex(CAL)
ma200 = twr.rolling(200, min_periods=200).mean()
def asof(s): return s.reindex(s.index.union(CAL)).ffill().reindex(CAL)
s2 = asof((twr >= ma200).astype(float))
sidx = pd.Series(CAL, index=CAL)
REB = pd.DatetimeIndex(sidx.groupby([CAL.year, CAL.month]).apply(lambda s: s[s.dt.day >= 11].iloc[0] if (s.dt.day >= 11).any() else pd.NaT).dropna())
s1 = pd.Series(np.nan, index=CAL); s1[REB] = s2[REB]; s1 = s1.ffill().fillna(1.0)
r = E50.pct_change().fillna(0.0).to_numpy()
pos = s1.shift(2).to_numpy().copy(); pos[:2] = 1.0
nxt = np.r_[pos[1:], pos[-1]]
ex = (pos > 0.5) & (nxt < 0.5); en = (pos < 0.5) & (nxt > 0.5)
f = np.where(pos > 0.5, 1 + r, 1.0) * np.where(ex, 1 - 0.001855, 1.0) * np.where(en, 1 - 0.000855, 1.0)
e50ma = pd.Series(np.cumprod(f), index=CAL)
def cagr(nav, a, b):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]; p = nav[nav.index < pd.Timestamp(a)]
    s0 = p.iloc[-1] if len(p) else s.iloc[0]; s0d = p.index[-1] if len(p) else s.index[0]
    return round(float((s.iloc[-1] / s0) ** (365.25 / (s.index[-1] - s0d).days) - 1), 4)
check['E0050_MA_repro'] = {'selection_CAGR': cagr(e50ma, '2006-04-01', '2013-12-31'), 'timing_daily_reported': 0.0662,
                           'validation_CAGR': cagr(e50ma, '2014-01-01', '2020-12-31'), 'timing_daily_reported_val': 0.1088}
print(check['E0050_MA_repro'])

# ---- 匯總 ----
mf = pd.read_parquet(A / 'momentum_family' / 'long_nav.parquet')
tb = pd.read_parquet(A / 'turnover_buffer' / 'buffer_nav.parquet')
ta = pd.read_parquet(A / 'w1_rev_yoy_turnaround' / 'nav.parquet')
mv = pd.read_parquet(A / 'w1_minervini_trend_template' / 'nav.parquet')
dr = pd.read_parquet(A / 'disposal_reconcile' / 'sleeve_nav.parquet')
ex_d = pd.read_parquet(A / 'exec_rebuild' / 'disposal_nav.parquet')
su = pd.read_parquet(A / 'w3_revenue_sue' / 'nav.parquet')
SRC = {
 'N30': (nav_main['M2_n30_slip0.001'], 'revenue_longhistory/oos_nav.parquet:M2_n30_slip0.001'),
 'N10': (nav_main['M2_n10_slip0.001'], 'revenue_longhistory/oos_nav.parquet:M2_n10_slip0.001'),
 'H52': (mf['P1_52wHigh_n30_0.001'], 'momentum_family/long_nav.parquet:P1_52wHigh_n30_0.001'),
 'H52K90': (tb['A_P1only_n30_K90_0.001'], 'turnover_buffer/buffer_nav.parquet:A_P1only_n30_K90_0.001'),
 'TURN_C': (ta['C_H1_0.001'], 'w1_rev_yoy_turnaround/nav.parquet:C_H1_0.001'),
 'MINV': (mv['BASE_ALL_ME'], 'w1_minervini_trend_template/nav.parquet:BASE_ALL_ME（滑價 0.1%）'),
 'DISP_PX': (disp_px_0050, '本資料夾重建：近似處置 10 槽、100 萬、0.8% 來回、閒置 0050，2005～2026-10'),
 'E0050': (E50, 'etf_0050.csv adj'),
 'E0050_MA': (e50ma, '本資料夾重建：0050＋S1（timing_daily 主版本）'),
 'DISP_REAL': (dr['A_0050'], 'disposal_reconcile/sleeve_nav.parquet:A_0050（上市正式版 10 槽、閒置 0050；2021-10 前以 0050 代替）'),
 'SURGE': (su['S_main_0.001'], 'w3_revenue_sue/nav.parquet:S_main_0.001'),
}
out = {}
for k, (s, _) in SRC.items():
    s = s.dropna()
    rr = s.pct_change()
    rr = rr.reindex(s.index.union(CAL))
    # 對到共同日曆：先把淨值 ffill 到共同日曆（避免跨日報酬遺失）
    nv = s.reindex(s.index.union(CAL)).ffill().reindex(CAL)
    nv[CAL < s.index[0]] = np.nan
    out[k] = nv
NAV = pd.DataFrame(out)
# DISP_REAL 在 2021-10 之前：以 0050 報酬接續（閒置資金＝0050，與 run_alloc「處置部位在 2021-10 之前是閒置資金」一致）
rD = NAV['DISP_REAL'].pct_change(); r5 = NAV['E0050'].pct_change()
first = NAV['DISP_REAL'].first_valid_index()
rD[CAL <= first] = r5[CAL <= first]
NAV['DISP_REAL'] = (1 + rD.fillna(0)).cumprod()
# 也存一份「閒置現金」處置（只作參考）
NAV['DISP_PX_cash'] = disp_px_cash.reindex(disp_px_cash.index.union(CAL)).ffill().reindex(CAL)
NAV['DISP_REAL_cash_ref'] = dr['A_cash'].reindex(dr.index.union(CAL)).ffill().reindex(CAL)
NAV = NAV.loc[:CAL[-1]]
NAV.to_parquet(HERE / 'strategy_nav.parquet')
info = {'sources': {k: v[1] for k, v in SRC.items()}, 'first_valid': {k: str(NAV[k].first_valid_index().date()) for k in NAV}, 'last': str(CAL[-1].date()), 'checks': check}
(HERE / 'nav_build.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(info, ensure_ascii=False, indent=1))

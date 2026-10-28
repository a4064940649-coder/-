#!/usr/bin/env python3
"""反方審查（成本與可成交）o7：只讀研究員檔案，輸出印在 stdout 並存 scratchpad。不改研究員任何檔案。
python3 -I -B review_exec_o7.py
檢查：A 重現；B 執行延遲（T+1 → T+2/3/5 收盤）；C 轉換成本損益兩平（主腿單邊滑價加大）；
D 散戶便宜靜態（不每月再平衡／零成本）；E 2024-01 缺決策日補回；F 轉換日漲跌停（主腿 10 檔）；G 逐次轉換明細。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import cr_lib as L, cr_eval as E

OUTF = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_exec_o7.json')
R = {}
W = pd.read_parquet(L.DATA / 'W_full.parquet')
NV4 = pd.read_parquet(L.RAW / 'navs_v4.parquet'); main = NV4['AB_V4|normal'].dropna(); END = main.index[-1]
mainS = NV4['AB_V4|stress'].dropna()
Wv = W[W['end'] <= END]
e50 = L.e50_series()
spec = dict(id='V5', sig='S2', L=6, xC=1.0, h=0.0)
SIM0 = '2018-03-01'


def exec_map_k(dec, ws, ix, k):
    out = {}
    for d, w in zip(dec, ws):
        j = ix.searchsorted(pd.Timestamp(d), side='right') + (k - 1)
        if j < len(ix) and w is not None and np.isfinite(w): out[ix[j]] = float(w)
    return out


def run(per, k=1, cost='normal', mainnav=main, Wx=Wv, static_mode='monthly', wbar=None):
    S, st, w = E.version_states(Wx, spec)
    dec = [d for d in Wx.index if d >= pd.Timestamp(SIM0) and w.get(d) is not None]
    wb = float(L.sel(w.dropna().astype(float), per).mean()) if wbar is None else wbar
    rot, lg = L.rotate(mainnav, e50, exec_map_k(dec, [w[d] for d in dec], mainnav.index, k), cost)
    if static_mode == 'monthly':
        sdec = dec
    elif static_mode == 'annual':
        sdec = [d for d in dec if d.month == 1] or dec[:1]
        sdec = [dec[0]] + [d for d in sdec if d != dec[0]]
    sta, lgs = L.rotate(mainnav, e50, exec_map_k(sdec, [wb] * len(sdec), mainnav.index, k), cost)
    B = dict(S=S, st=st, w=w, wbar=wb, rot=rot, sta=sta, log=lg, logs=lgs, dec=dec)
    sd_end = min(Wx['end'][(Wx.index >= pd.Timestamp(per[0])) & (Wx.index <= pd.Timestamp(per[1]))].max(), mainnav.index[-1])
    o = E.evaluate(B, Wx, mainnav, e50, per, sd_end, label='V5')
    x = o['_series']['r_rot'] - o['_series']['r_sta']
    xs = x.sort_values(ascending=False)
    return dict(vs_static_pp=o['vs_static']['mean_pp'], t=o['vs_static']['t'], vs_0050_pp=o['vs_0050']['mean_pp'], vs_0050_t=o['vs_0050']['t'],
                vs_main_pp=o['vs_main']['mean_pp'], CAGR_rot=o['CAGR_rot'], CAGR_static=o['CAGR_static'], CAGR_main=o['CAGR_main'], CAGR_0050=o['CAGR_0050'],
                switches=o['switches'], drop3_pp=float(x.drop(xs.index[:3]).mean() * 100)), B, o


# ---- A. 重現 ----
a, BV, oV = run(L.VAL); s, BS, oS = run(L.SEL)
R['A_repro'] = dict(VAL=a, SEL=s)

# ---- B. 執行延遲 ----
R['B_exec_delay'] = {f'T+{k}': dict(VAL=run(L.VAL, k=k)[0], SEL=run(L.SEL, k=k)[0]) for k in (1, 2, 3, 5)}

# ---- C. 轉換成本加大（主腿單邊滑價 s；稅照收）；靜態也用同成本每月再平衡 ----
FEE = L.FEE
for sl in (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03):
    L.COST[f's{sl}'] = (FEE + sl, FEE + 0.003 + sl, FEE + 0.0005, FEE + 0.001 + 0.0005)
R['C_switch_slip'] = {}
for sl in (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03):
    R['C_switch_slip'][f'main_slip_{sl*100:.2f}%'] = dict(VAL=run(L.VAL, cost=f's{sl}')[0], SEL=run(L.SEL, cost=f's{sl}')[0])
# 只罰轉換、不罰靜態再平衡（靜態用 normal）：損益兩平
def run_mixed(per, sl):
    L.COST['tmp'] = (FEE + sl, FEE + 0.003 + sl, FEE + 0.0005, FEE + 0.001 + 0.0005)
    S, st, w = E.version_states(Wv, spec)
    dec = [d for d in Wv.index if d >= pd.Timestamp(SIM0) and w.get(d) is not None]
    wb = float(L.sel(w.dropna().astype(float), per).mean())
    rot, lg = L.rotate(main, e50, L.exec_map(dec, [w[d] for d in dec], main.index), 'tmp')
    sta, _ = L.rotate(main, e50, L.exec_map(dec, [wb] * len(dec), main.index), 'normal')
    sd = list(Wv.index) + [END]
    rr = L.sel(L.win_ret(rot, sd), per); rs = L.sel(L.win_ret(sta, sd), per)
    return float((rr - rs).mean() * 100)
be = {}
for sl in np.arange(0, 0.061, 0.005):
    be[f'{sl*100:.1f}%'] = run_mixed(L.VAL, sl)
R['C2_breakeven_VAL_rotation_only_penalised'] = be

# ---- D. 散戶便宜靜態：每年 1 月才再平衡；零成本靜態 ----
R['D_static_alt'] = dict(annual_rebal_VAL=run(L.VAL, static_mode='annual')[0], annual_rebal_SEL=run(L.SEL, static_mode='annual')[0])

# ---- E. 2024-01 缺決策日：用 2023-12-29 的 top5 補一個 2024-01 決策點，看狀態是否改變 ----
P = L.Panel()
d_last = P.asof('2024-01-11')
idx, v = P.pool(d_last)
t5_jan = float(np.sort(v)[::-1][:5].sum() / v.sum())
dec_dates = list(Wv.index)
prev6 = [d for d in dec_dates if d <= pd.Timestamp('2023-07-31')][-1]
S_jan = t5_jan - float(Wv.loc[prev6, 'top5'])
R['E_2024_01'] = dict(asof_used=str(d_last.date()), top5_jan=t5_jan, top5_6m_ago=float(Wv.loc[prev6, 'top5']), ref_6m_ago=str(prev6.date()),
                      S2_jan=S_jan, state_jan='C' if S_jan > 0 else 'D', state_dec=str(BV['st'].get(pd.Timestamp('2023-12-11'))),
                      state_feb=str(BV['st'].get(pd.Timestamp('2024-02-15'))),
                      panel_gap=[str(x.date()) for x in P.IDX[(P.IDX >= pd.Timestamp('2023-12-25')) & (P.IDX <= pd.Timestamp('2024-02-20'))][:3]])

# ---- F. 轉換日漲跌停：0050→主腿（買進 10 檔，於 exec 日收盤）／主腿→0050（賣出） ----
plan = pd.read_csv(L.RAW / 'plan_V4.csv', parse_dates=['signal', 'exec'], dtype={'picks': str})
C = P.C; H = pd.read_parquet(L.PANEL / 'high.parquet'); Lo = None
try:
    Lo = pd.read_parquet(L.PANEL / 'low.parquet')
except Exception:
    pass
lg = BV['log']
rows = []
for _, r in lg.iterrows():
    dt = pd.Timestamp(r['date'])
    if dt < pd.Timestamp('2019-01-01'): continue
    pr = plan[plan['exec'] <= dt].iloc[-1]
    picks = [p.strip() for p in pr['picks'].split(',')]
    direction = 'into_0050' if r['w'] > 0.5 else 'into_main'
    # 賣出主腿時持有的是 exec 日開盤新換的 picks（NAV 口徑）；買進主腿時買的也是本月 picks
    if dt not in C.index:
        rows.append(dict(date=str(dt.date()), direction=direction, note='面板無此日')); continue
    i = C.index.get_loc(dt)
    prev = C.iloc[i - 1]
    lim_up = []; lim_dn = []
    for p in picks:
        if p not in C.columns: continue
        c0, c1, h1 = prev[p], C.loc[dt, p], H.loc[dt, p]
        if not (np.isfinite(c0) and np.isfinite(c1)): continue
        ch = c1 / c0 - 1
        if ch >= 0.095 and np.isfinite(h1) and c1 >= h1 - 1e-9: lim_up.append(p)
        if ch <= -0.095: lim_dn.append(p)
    rows.append(dict(date=str(dt.date()), direction=direction, moved=float(r['moved']), cost_pp=float(r['cost']) * 100,
                     limit_up_close=lim_up, limit_down_close=lim_dn, picks=picks))
R['F_switch_days_limits'] = rows

# ---- G. 驗證期年度轉換成本與周轉 ----
lgv = lg[(lg['date'] >= pd.Timestamp('2023-01-12')) & (lg['date'] <= END)]
yrs = (END - pd.Timestamp('2023-01-11')).days / 365.25
R['G_turnover_VAL'] = dict(n_switch_events=int((lgv['moved'] > 0.01).sum()), turnover_per_year=float(lgv['moved'].sum() / yrs),
                           switch_cost_pp_total=float(lgv['cost'].sum() * 100), switch_cost_pp_per_year=float(lgv['cost'].sum() * 100 / yrs),
                           static_rebal_cost_pp_per_year=float(BV['logs'][(BV['logs']['date'] >= pd.Timestamp('2023-01-12')) & (BV['logs']['date'] <= END)]['cost'].sum() * 100 / yrs))

# ---- H. 壓力成本 ＋ T+2 ＋ 拿掉最好 3 個月 合併 ----
L.COST['s0.01'] = L.COST['s0.01']
R['H_combined_adverse'] = dict(VAL_stressNAV_slip1pct_T2=run(L.VAL, k=2, cost='s0.01', mainnav=mainS)[0],
                               SEL_stressNAV_slip1pct_T2=run(L.SEL, k=2, cost='s0.01', mainnav=mainS)[0])

OUTF.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=L._js), encoding='utf-8')
print(json.dumps(R, ensure_ascii=False, indent=1, default=L._js))

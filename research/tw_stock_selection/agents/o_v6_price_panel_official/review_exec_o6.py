#!/usr/bin/env python3
"""反方審查（成本與可成交角度）：o_v6_price_panel_official。只讀研究員檔案，不改任何東西。
python3 -I -B review_exec_o6.py
1. 賣出日開盤跌停（官方開盤）次數與鎖死跌停（O=H=L=跌停）次數；引擎無跌停賣不掉處理。
2. 賣出日處於處置期（需預收款券、分盤撮合）的次數。
3. 保留期（共同基期 2023-12-29）扣可執行拖累後對 0050。
4. 2024-01-12 換股拖累記 0 的影響：以其他換股日的平均拖累估計。
5. 2024-01 新增換股：下單金額對當日官方成交值。
6. 周轉與成本拆解。
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

H = Path('/home/user/-/research/tw_stock_selection/agents/o_v6_price_panel_official')
AG = H.parent
D = H / 'data'
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
out = {}

O = pd.read_parquet(D / 'open.parquet'); C = pd.read_parquet(D / 'close_raw.parquet')
Hh = pd.read_parquet(D / 'high.parquet'); Lw = pd.read_parquet(D / 'low.parquet')
R = pd.read_parquet(D / 'ret.parquet'); F = pd.read_parquet(D / 'flag.parquet'); V = pd.read_parquet(D / 'value.parquet')
S = pd.read_parquet(D / 'src.parquet')
IDX = C.index
PREVC = C.ffill().shift(1)

def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)

def ld_price(ref, lim):
    raw = ref * (1 - lim); t = tick(raw)
    return np.ceil(raw / t - 1e-9) * t

def lu_price(ref, lim):
    raw = ref * (1 + lim); t = tick(raw)
    return np.floor(raw / t + 1e-9) * t

def refp(d, c):
    cc = C.at[d, c]; r = R.at[d, c]
    ref = cc / (1 + r) if np.isfinite(r) else np.nan
    if (F.at[d, c] == 1) or not np.isfinite(ref): ref = PREVC.at[d, c]
    return ref

# ---------------- 1/2. 賣出日跌停與處置 ----------------
dsp5 = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
dh = pd.read_csv(AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
DS = pd.concat([dsp5[['code', 'start', 'end']], dh[['code', 'start', 'end']]]).dropna().drop_duplicates()
dsg = {c: g for c, g in DS.groupby('code')}
def in_disp(c, d):
    g = dsg.get(c)
    return g is not None and bool(((g.start <= d) & (g.end >= d)).any())

def scan(planf):
    pl = pd.read_csv(planf, parse_dates=['signal', 'exec'])
    rows = []; prev = []
    for _, r in pl.iterrows():
        e = r['exec']; picks = r['picks'].split(',') if isinstance(r['picks'], str) else []
        sells = [c for c in prev if c not in picks]; buys = [c for c in picks if c not in prev]
        lim = 0.07 if e < pd.Timestamp('2015-06-01') else 0.10
        for kind, lst in (('sell', sells), ('buy', buys)):
            for c in lst:
                if c not in C.columns or e not in IDX: continue
                o = O.at[e, c]; ref = refp(e, c)
                if not (np.isfinite(o) and np.isfinite(ref)):
                    rows.append(dict(exec=e, code=c, kind=kind, open_na=True)); continue
                ld = ld_price(ref, lim); lu = lu_price(ref, lim)
                hi, lo = Hh.at[e, c], Lw.at[e, c]
                rows.append(dict(exec=e, code=c, kind=kind, open_na=False, open_ld=bool(o <= ld + 1e-6),
                                 locked_ld=bool(o <= ld + 1e-6 and np.isfinite(hi) and hi <= ld + 1e-6),
                                 open_lu=bool(o >= lu - 1e-6), disp=in_disp(c, e), src=int(S.at[e, c]) if np.isfinite(S.at[e, c]) else -1,
                                 gap=float(o / ref - 1), val=float(V.at[e, c]) if np.isfinite(V.at[e, c]) else np.nan))
        prev = picks
    return pd.DataFrame(rows)

for name in ('plan_V6.csv', 'plan_AB_V6_s0.csv', 'plan_AB_V6_B10_s0.csv'):
    df = scan(H / name)
    sl = df[(df.kind == 'sell') & (~df.open_na)]; by = df[(df.kind == 'buy') & (~df.open_na)]
    w = sl[sl.exec >= pd.Timestamp('2024-01-01')]
    o = dict(n_sells=int(len(sl)), n_sell_open_limitdown=int(sl.open_ld.sum()), n_sell_locked_limitdown=int(sl.locked_ld.sum()),
             sell_locked_list=sl[sl.locked_ld][['exec', 'code', 'gap']].astype(str).values.tolist(),
             n_sell_in_disposal=int(sl.disp.sum()), n_buy_in_disposal=int(by.disp.sum()),
             n_sell_open_na=int(((df.kind == 'sell') & df.open_na).sum()),
             sell_gap_mean_pct=float(sl.gap.mean() * 100), sell_gap_p05_pct=float(sl.gap.quantile(0.05) * 100),
             buy_gap_mean_pct=float(by.gap.mean() * 100), buy_gap_p95_pct=float(by.gap.quantile(0.95) * 100),
             n_buy_open_lu_in_plan=int(by.open_lu.sum()),
             hold_sells=int(len(w)), hold_sell_disp=int(w.disp.sum()), hold_sell_ld=int(w.open_ld.sum()))
    out[f'trade_scan|{name}'] = o
    print(name, {k: v for k, v in o.items() if k != 'sell_locked_list'}, o['sell_locked_list'][:20], flush=True)
    if name == 'plan_AB_V6_s0.csv':
        TRD = df

# ---------------- 3. 保留期扣可執行拖累 ----------------
NV = pd.read_parquet(H / 'navs_v6.parquet')
EX = pd.read_parquet(AG / 'o_exec_capacity_tiers' / 'navs_abavg.parquet')
BEN = pd.read_parquet(AG / 'o_exec_capacity_tiers' / 'data' / 'bench.parquet')
rzi = BEN['AB_V4|normal'].pct_change()
E50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = E50.reindex(E50.index.union(NV.index)).ffill().reindex(NV.index)
TI = [f'main|{t}|{m}|A' for t in ('30萬', '100萬', '300萬', '1000萬') for m in ('FIX', 'GROW')]
def cagr(n, a, b):
    s = n.loc[a:b]; yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100
a, b = pd.Timestamp('2023-12-29'), NV.index[-1]
hold = {'0050': cagr(E50, a, b)}
for base in ('AB_V6|normal', 'AB_V6_B10|normal'):
    nav = NV[base]; hold[base] = cagr(nav, a, b)
    for col in TI:
        drag = (EX[col].pct_change() - rzi).reindex(nav.index).fillna(0.0)
        ne = (1 + nav.pct_change().fillna(0) + drag).cumprod()
        hold[f'EXEC|{base}|{col}'] = cagr(ne, a, b)
out['hold_common_base_2023-12-29_CAGR'] = hold
print('HOLD', {k: round(v, 2) for k, v in hold.items()}, flush=True)

# ---------------- 4. 2024-01-12 換股拖累 = 0 ----------------
pl = pd.read_csv(H / 'plan_AB_V6_s0.csv', parse_dates=['signal', 'exec'])
ex_days = [d for d in pl['exec'] if d in EX.index]
miss = [d for d in pl['exec'] if d not in EX.index]
dr = {}
for col in TI:
    drag = (EX[col].pct_change() - rzi)
    v = drag.reindex(ex_days).dropna()
    rec = v[v.index >= pd.Timestamp('2021-01-01')]
    dr[col] = dict(mean_exec_day_drag_pct_all=float(v.mean() * 100), mean_exec_day_drag_pct_2021on=float(rec.mean() * 100),
                   hold_total_drag_pct_per_yr=float(drag.loc['2024-01-01':].sum() * 100 / ((EX.index[-1] - pd.Timestamp('2024-01-01')).days / 365.25)))
out['exec_day_drag'] = dr; out['exec_days_missing_in_EX'] = [str(d.date()) for d in miss]
print('missing exec days in EX', out['exec_days_missing_in_EX']); print('drag', json.dumps(dr, ensure_ascii=False), flush=True)

# ---------------- 5. 2024-01 新增換股：下單金額對官方成交值 ----------------
e = pd.Timestamp('2024-01-12')
r = pl[pl['exec'] == e]
if len(r):
    pk = r.iloc[0]['picks'].split(',')
    vv = {c: float(V.at[e, c]) / 1e4 for c in pk}
    out['2024-01-12_picks_value_wan'] = vv
    for amt in (3, 10, 30, 100):
        out[f'2024-01-12_order_{amt}wan_over_dayvalue_pct'] = {c: round(amt / v * 100, 3) for c, v in vv.items()}
    print('2024-01-12 day value (萬)', {c: round(v) for c, v in vv.items()})

# 同池 2024-02～2025-09 換股日 picks 的成交值是否用估計值（OHLC4×股數）—無法分辨，只報 src
w = TRD[(TRD.exec >= pd.Timestamp('2024-01-01')) & (TRD.exec <= pd.Timestamp('2025-09-30')) & (~TRD.open_na)]
out['src_counts_trades_2024_2025-09'] = {str(k): int(v) for k, v in w.src.value_counts().items()}

# ---------------- 6. 周轉與成本 ----------------
res = json.loads((H / 'results.json').read_text())
to = res['RESULTS_RUN']['turnover']['V6|normal']
FEE = 0.001425 * 0.6
out['cost_math'] = dict(turnover_per_rebalance=to, annual_one_side=to * 12,
                        annual_cost_normal_pct=to * 12 * (2 * (FEE + 0.001) + 0.003) * 100,
                        annual_tax_pct=to * 12 * 0.003 * 100,
                        annual_cost_stress_pct=to * 12 * 2 * 0.005 * 100)
print('cost', out['cost_math'])
(H / 'review_exec_o6.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

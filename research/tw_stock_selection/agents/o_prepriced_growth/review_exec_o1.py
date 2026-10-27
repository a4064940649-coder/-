#!/usr/bin/env python3
"""反方審查（成本與可成交）：python3 -I -B review_exec_o1.py — 只讀研究員檔案，不寫回任何 json。
只用挑選期 2006-07～2013-12（同研究員），不碰 2014 年以後。
檢查：成本敏感度／損益兩平成本、最低手續費、整張可行性、成交值占比（容量）、平方根衝擊、
開盤跌停賣不掉、處置資料覆蓋、基準是否扣成本、隔日可交易的前視。
"""
import sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as O
V, K = O.V, O.K

R = O.revenue()
S = V.V2('gapfix', R['YL'], R['Y5'], exfin=True)
L = O.Leg(S)
d0 = O.leg_start(L)
days = [d for d in S.sig_days if d0 <= d <= O.SEL_END]
t1 = K.IDX.get_loc(K.IDX[K.IDX <= O.SEL_END][-1]); sd = days + [K.IDX[t1]]
FEE, TAX = K.FEE, K.TAX
plm = [p for p in S.plan(10, 'official') if p[3] in set(days)]
t0 = plm[0][0]
ew, _ = K.simulate([(p[0], p[2]) for p in plm], 0, 0, t0, t1)
ewn, _ = K.simulate([(p[0], p[2]) for p in plm], *K.rates(0.001), t0, t1)
wr_ew = O.window_returns(ew, sd); wr_ewn = O.window_returns(ewn, sd)
wr_50 = O.window_returns(K.E50.reindex(ew.index), sd)

def ex(pl, b, s, bench=None):
    nav, lg = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1)
    w = O.window_returns(nav, sd)
    return (w - (wr_ew if bench is None else bench)).to_numpy(), lg, w

out = {}
VALUE = K.V.to_numpy() if hasattr(K, 'V') else None
VN = pd.read_parquet(K.P / 'value.parquet').reindex(index=K.IDX, columns=K.COLS).to_numpy()
plans = {v: L.plan(*v.split('|')[:2], int(v.split('|')[2][1:]), days) for v in ('AVG|B1|n30', 'AVG|B1|n10')}
plans['MAIN|n10'] = plm

# ---------- 1. 成本敏感度與損益兩平 ----------
sens = {}
for v in ('AVG|B1|n30', 'AVG|B1|n10', 'MAIN|n10'):
    pl = plans[v]; r = {}
    for lab, (b, s) in {'gross': (0, 0), 'slip0.10': K.rates(0.001), 'slip0.15': K.rates(0.0015), 'slip0.20': K.rates(0.002),
                        'slip0.30': K.rates(0.003), 'minfee_10k_slip0.10': (0.002 + 0.001, 0.002 + TAX + 0.001),
                        'tax_only': (0, TAX)}.items():
        x, lg, w = ex(pl, b, s)
        r[lab] = {'mean_pct_m': x.mean() * 100, 't_EW_gross_bench': O.tstat(x), 't_EW_net_bench': O.tstat((w - wr_ewn).to_numpy()),
                  't_0050': O.tstat((w - wr_50).to_numpy()), 'turn': lg['turnover']}
    # 兩平：每邊額外滑價 s 使 t_EW = 1.5／平均超額 = 0（線性內插）
    grid = np.arange(0.0, 0.0031, 0.0005); tt = []; mm = []
    for sl in grid:
        x, _, _ = ex(pl, FEE + sl, FEE + TAX + sl); tt.append(O.tstat(x)); mm.append(x.mean() * 100)
    r['slip_grid'] = {f'{sl*100:.2f}%': {'t': t_, 'mean': m_} for sl, t_, m_ in zip(grid, tt, mm)}
    sens[v] = r
out['cost_sensitivity'] = sens

# ---------- 2. 交易明細：整張可行性、成交值占比、跳動點、開盤跌停 ----------
def limit_down_price(ref, lim):
    raw = ref * (1 - lim); tk = K.tick(raw)
    return np.ceil(raw / tk - 1e-9) * tk

def trade_detail(pl, n):
    rows = []; prev = set()
    for p in pl:
        e = p[0]; cur = set(int(j) for j in p[1]); t = e - 1
        lim = 0.07 if K.IDX[e] < pd.Timestamp('2015-06-01') else 0.10
        ref = K.ref_price(e); ld = limit_down_price(ref, lim)
        opn = K.On[e]
        for j in cur - prev:
            rows.append(('B', e, j, opn[j], S.V60[t][j], VN[e][j], False))
        for j in prev - cur:
            lockd = bool(np.isfinite(opn[j]) and np.isfinite(ld[j]) and opn[j] <= ld[j] + 1e-6)
            rows.append(('S', e, j, opn[j], S.V60[t][j], VN[e][j], lockd))
        prev = cur
    df = pd.DataFrame(rows, columns=['side', 'e', 'j', 'open', 'v60', 'v_exec', 'open_limdown'])
    df['lot_value'] = df['open'] * 1000
    df['tick_bp'] = K.tick(df['open'].to_numpy()) / df['open'] * 1e4
    # 日報酬波動（20 日）作為平方根衝擊的 sigma
    sig = []
    for _, rr in df.iterrows():
        x = K.Rn[max(rr.e - 21, 0):rr.e - 1, int(rr.j)]
        sig.append(np.nanstd(x))
    df['sigma_d'] = sig
    return df

cap_levels = [3e5, 1e6, 3e6, 1e7, 3e7]
det = {}
for v, n in (('AVG|B1|n30', 30), ('AVG|B1|n10', 10), ('MAIN|n10', 10)):
    df = trade_detail(plans[v], n)
    q = lambda s: {k: float(np.nanquantile(s, k2)) for k, k2 in (('p10', .1), ('p50', .5), ('p90', .9))}
    r = {'n_trades': len(df), 'n_sells': int((df.side == 'S').sum()),
         'sell_open_limitdown_share': float(df.loc[df.side == 'S', 'open_limdown'].mean()),
         'sell_open_limitdown_n': int(df.loc[df.side == 'S', 'open_limdown'].sum()),
         'lot_value_NT': q(df.lot_value), 'v60_NT': q(df.v60), 'v_exec_NT': q(df.v_exec), 'tick_bp': q(df.tick_bp),
         'sigma_d': q(df.sigma_d)}
    cap = {}
    for C in cap_levels:
        pos = C / n
        part = pos / df.v60; part_x = pos / df.v_exec
        lots = pos / df.lot_value
        # 整張：持有 floor(lots) 張，誤差＝|floor-lots|/lots；lots<1 → 買不到任何一張
        lt1 = float((lots < 1).mean()); err = np.where(lots >= 1, (lots - np.floor(lots)) / lots, 1.0)
        # 平方根衝擊：impact = 1.0 × sigma_d × sqrt(Q/ADV)
        imp = 1.0 * df.sigma_d * np.sqrt(part)
        fee_eff = np.maximum(20.0 / pos, FEE)
        cap[f'{int(C/1e4)}萬'] = {'pos_NT': pos, 'part_v60_p50': float(np.nanmedian(part)), 'part_v60_p90': float(np.nanquantile(part, .9)),
                                 'part_execday_p90': float(np.nanquantile(part_x, .9)),
                                 'share_lt_1lot': lt1, 'mean_weight_err_wholelot': float(np.nanmean(err)),
                                 'sqrt_impact_mean_bp': float(np.nanmean(imp) * 1e4), 'fee_eff_bp_per_side': float(fee_eff * 1e4)}
    r['capacity'] = cap
    det[v] = r
    det[v]['_df'] = df
out['trades'] = {k: {kk: vv for kk, vv in v.items() if kk != '_df'} for k, v in det.items()}

# ---------- 3. 依容量把衝擊＋最低手續費併入成本，重算 t ----------
cap_t = {}
for v, n in (('AVG|B1|n30', 30), ('AVG|B1|n10', 10), ('MAIN|n10', 10)):
    r = {}
    for C in cap_levels:
        c = det[v]['capacity'][f'{int(C/1e4)}萬']
        extra = 0.001 + c['sqrt_impact_mean_bp'] / 1e4            # 滑價 0.1%（價差）＋ 平方根衝擊
        fee = c['fee_eff_bp_per_side'] / 1e4
        x, lg, w = ex(plans[v], fee + extra, fee + TAX + extra)
        r[f'{int(C/1e4)}萬'] = {'per_side_ex_tax_bp': (fee + extra) * 1e4, 't_EW': O.tstat(x), 'mean_pct_m': x.mean() * 100,
                               't_0050': O.tstat((w - wr_50).to_numpy())}
    cap_t[v] = r
out['capacity_adjusted_t'] = cap_t

# ---------- 4. 處置資料覆蓋與前視 ----------
out['disposal_data_first_start'] = str(K.DSP['start'].min().date())
out['disposal_data_covers_selection'] = bool(K.DSP['start'].min() <= pd.Timestamp('2013-12-31'))
# 隔日有開盤成交的條件：訊號日當下無法知道；被這條件剔除的股池檔數
drop = []
for d in days:
    t = K.IDX.get_loc(d); e = t + 1
    cs = S.cross_section(d)
    v60 = S.V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    drop.append(int((base & ~(np.isfinite(K.On[e]) & K.TRADED[e])).sum()))
out['lookahead_next_open_filter_drops_per_month'] = {'mean': float(np.mean(drop)), 'max': int(np.max(drop))}
out['EW_bench_is_gross'] = True
out['t_EW_vs_net_EW_selected'] = sens['AVG|B1|n30']['slip0.10']['t_EW_net_bench']

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)): return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
print(json.dumps(clean(out), ensure_ascii=False, indent=1))

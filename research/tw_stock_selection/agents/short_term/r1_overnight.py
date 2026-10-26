#!/usr/bin/env python3
"""規則 1：隔夜 vs 盤中報酬分解（上市流動股）與隔夜動能五分位。

事前固定：
  1a 全體流動股 EW：隔夜 on_t = 今開／昨收 − 1（flag 0，除權息日剔除）、盤中 id_t = 今收／今開 − 1。描述性分解（逐年、各窗口）。
  1b 可交易版：全體流動股「收盤買、隔日開盤賣」（兩筆非當沖，成本 0.5%）；
      全體流動股「開盤買、收盤賣」（現股當沖 0.23%，收盤跌停改隔日開盤賣 0.5%）。
  1c 隔夜動能：t 日訊號＝過去 20 個交易日（含 t）隔夜報酬平均（至少 15 筆），流動股五分位；
      t 收盤買（收盤鎖漲停者剔除）、t+1 開盤賣，報酬 on_{t+1}，成本 0.5%。報告 Q1～Q5 毛報酬、Q5−Q1、Q5 淨報酬。
  1d （事前附帶）同一排序下 t+1 盤中報酬（開盤買、收盤賣，當沖 0.23%）：Q1 與 Q5 淨報酬（文獻的「拔河」現象：隔夜強者盤中弱）。
挑選：S1（2015-06～2019-04）平均淨報酬 > 0 且 t ≥ 2 才算通過；S0（2005-01～2015-05）另列；驗證 V_official（src 1,2,4）與 V_all。
"""
import sys, time
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/short_term')
import numpy as np, pandas as pd
import stlib as L

t0 = time.time()
D = L.load()
on, idr, elig, F = D['on'], D['id'], D['elig'], D['flag']
res = {'description': __doc__}

# ---------- 1a 描述性分解 ----------
both = elig & on.notna() & idr.notna()
ew_on = on.where(both).mean(axis=1); ew_id = idr.where(both).mean(axis=1)
ew_cc = ((1 + on) * (1 + idr) - 1).where(both).mean(axis=1)
desc = {}
for w in L.WINDOWS:
    m = L.day_set(D, w); d = m[m].index
    a, b, c = ew_on.reindex(d).dropna(), ew_id.reindex(d).dropna(), ew_cc.reindex(d).dropna()
    desc[w] = {'days': int(len(a)), 'mean_overnight_pct': round(a.mean() * 100, 4), 't_overnight': L.tstat(a),
               'mean_intraday_pct': round(b.mean() * 100, 4), 't_intraday': L.tstat(b),
               'mean_close_to_close_pct': round(c.mean() * 100, 4),
               'ann_overnight_compound': round(float((1 + a).prod() ** (252 / len(a)) - 1), 4),
               'ann_intraday_compound': round(float((1 + b).prod() ** (252 / len(b)) - 1), 4),
               'share_overnight_positive_days': round(float((a > 0).mean()), 3)}
yr = {}
for y in range(2005, 2027):
    d = ew_on.index[(ew_on.index.year == y)]
    a, b = ew_on.reindex(d).dropna(), ew_id.reindex(d).dropna()
    if len(a) < 20: continue
    yr[y] = {'overnight_cum_pct': round(float(((1 + a).prod() - 1) * 100), 1), 'intraday_cum_pct': round(float(((1 + b).prod() - 1) * 100), 1),
             'src': int(D['daysrc'].reindex(d).mode().iloc[0])}
res['1a_decomposition'] = {'by_window': desc, 'by_year_EW_compound': yr}

# ---------- 1b 全體可交易版 ----------
pc_lock = D['lock_up_close'].shift(1).fillna(False).astype(bool)   # 昨收鎖漲停 → 昨收買不到
mask_on_all = elig & on.notna() & ~pc_lock
r, g, c, n = L.run_rule(D, on, L.COST_RT, mask_on_all, stress=('cost_1pct', L.COST_RT_STRESS))
res['1b_all_close_buy_open_sell'] = r
gdt, cdt = L.daytrade_exit(D)
mask_id_all = elig & idr.notna() & ~D['open_limit_up'].fillna(False).astype(bool)
r, g, c, n = L.run_rule(D, gdt, cdt, mask_id_all, stress=('cost_0.38pct_pre2017tax', L.COST_DT_PRE2017))
res['1b_all_open_buy_close_sell_daytrade'] = r

# ---------- 1c/1d 隔夜動能五分位 ----------
sig = on.rolling(20, min_periods=15).mean()          # 只用到 t（t 的開盤已知）
nxt_on = on.shift(-1)                                  # 報酬 on_{t+1}
nxt_g, nxt_c = gdt.shift(-1), cdt.shift(-1)            # t+1 開盤買收盤賣
nxt_olu = D['open_limit_up'].shift(-1).fillna(False).astype(bool)
lock_t = D['lock_up_close'].fillna(False).astype(bool)
base = elig & sig.notna() & nxt_on.notna() & ~lock_t
rk = sig.where(base).rank(axis=1, method='first')
cnt = base.sum(axis=1)
q = np.ceil(rk.div(cnt, axis=0) * 5).clip(1, 5)
q = q[cnt >= 50].reindex(q.index)
quint = {}
Gq_on, Gq_id = {}, {}
for k in range(1, 6):
    mk = (q == k)
    # 報酬歸屬到 t+1（實際實現日），所以 cohort 以 t 計算後 shift 一天
    g_on, c_on, n_on = L.cohort(nxt_on, L.COST_RT, mk)
    Gq_on[k] = g_on
    mk_id = mk & ~nxt_olu
    g_id, c_id, n_id = L.cohort(nxt_g, nxt_c, mk_id)
    Gq_id[k] = (g_id, c_id, n_id)
    quint[f'Q{k}'] = (g_on, n_on)

def shift_idx(s):
    """index t → t+1（CAL 上的下一個交易日）。"""
    CAL = D['CAL']; pos = CAL.get_indexer(s.index)
    ok = (pos >= 0) & (pos + 1 < len(CAL))
    return pd.Series(s.values[ok], index=CAL[pos[ok] + 1])

out1c = {}
for w in L.WINDOWS:
    m = L.day_set(D, w); d = m[m].index
    row = {}
    for k in range(1, 6):
        go = shift_idx(Gq_on[k]).reindex(d).dropna(); gi = shift_idx(Gq_id[k][0]).reindex(d).dropna()
        row[f'Q{k}_overnight_pct'] = round(go.mean() * 100, 4); row[f'Q{k}_t_on'] = L.tstat(go)
        row[f'Q{k}_intraday_next_pct'] = round(gi.mean() * 100, 4); row[f'Q{k}_t_id'] = L.tstat(gi)
    sp = (shift_idx(Gq_on[5]) - shift_idx(Gq_on[1])).reindex(d).dropna()
    row['Q5_minus_Q1_overnight_pct'] = round(sp.mean() * 100, 4); row['t_Q5_Q1_overnight'] = L.tstat(sp)
    spi = (shift_idx(Gq_id[1][0]) - shift_idx(Gq_id[5][0])).reindex(d).dropna()
    row['Q1_minus_Q5_intraday_pct'] = round(spi.mean() * 100, 4); row['t_Q1_Q5_intraday'] = L.tstat(spi)
    row['days'] = int(len(sp))
    out1c[w] = row
res['1c_quintile_table'] = out1c

# 可交易：Q5 收盤買隔日開盤賣（0.5%）、Q1 同（對照）、Q1/Q5 隔日當沖
for k in (5, 1):
    g = shift_idx(Gq_on[k]); n = shift_idx(quint[f'Q{k}'][1])
    res[f'1c_Q{k}_close_buy_open_sell'] = L.evaluate_all(D, g, L.COST_RT, n, g - L.COST_RT, stress=('cost_1pct', L.COST_RT_STRESS))
for k in (1, 5):
    g, c, n = (shift_idx(x) for x in Gq_id[k])
    res[f'1d_Q{k}_next_day_daytrade'] = L.evaluate_all(D, g, c, n, g - c, stress=('cost_0.38pct_pre2017tax', L.COST_DT_PRE2017))

res['runtime_sec'] = round(time.time() - t0, 1)
L.save('results_r1.json', res)
print(f"done {res['runtime_sec']}s")
for k, v in res.items():
    if isinstance(v, dict) and 'S1_2015_2019' in v:
        print(k, {w: L.brief(v[w]) for w in L.WINDOWS})
print(pd.DataFrame(out1c).T.to_string())
print(pd.DataFrame(desc).T.to_string())

#!/usr/bin/env python3
"""規則 2：開盤跳空缺口（上市流動股）。

事前固定（gap_t = open_t / close_{t−1} − 1，flag 0 日才算，除權息日剔除）：
  2a 跳空下跌 ≤ −3%：開盤買、收盤賣（現股當沖 0.23%；收盤跌停賣不掉→隔日開盤賣、0.5%）
  2b 跳空下跌 ≤ −5%：同上
  2c 跳空上漲 ≥ +3%（開盤即漲停者買不到，剔除）：同上
  2d～2f 上述三者改為持有到隔日收盤（開盤買 → t+1 收盤賣，非當沖 0.5%）；t+1 報酬用含息 ret（flag 3/4/5 剔除）
  附帶（事前一併列出，全部報告）：跳空上漲 ≥ +5%（當沖、隔日）、跳空下跌 −5%～−3% 區間（當沖）
  敏感度（看完初步結果後加，只作出場假設檢查、不作挑選）：收盤跌停也假設能在收盤賣出（naive）、一半成交（half）
組合：同日事件等權；當沖型資金每天全進全出；隔日型用兩個交替資金池（各 1/2），每池每 2 日進出一次。
"""
import sys, time
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/short_term')
import numpy as np, pandas as pd
import stlib as L

t0 = time.time()
D = L.load()
on, idr, elig, F, R = D['on'], D['id'], D['elig'], D['flag'], D['ret']
gap = on        # 已是 flag 0、前一交易日有收盤、無異常值
olu = D['open_limit_up'].fillna(False).astype(bool)
gdt, cdt = L.daytrade_exit(D)
# 隔日收盤賣：(close_t/open_t)·(1+ret_{t+1}) − 1
r_next = R.shift(-1).where(F.shift(-1).isin([0, 1, 2]) & (R.shift(-1).abs() <= D['lim'].shift(-1).values[:, None] + 0.03))
g2 = (1 + idr) * (1 + r_next) - 1
res = {'description': __doc__}
rules = {
    'gap_down_3': gap <= -0.03,
    'gap_down_5': gap <= -0.05,
    'gap_up_3': (gap >= 0.03) & ~olu,
    'gap_up_5': (gap >= 0.05) & ~olu,
    'gap_down_3to5': (gap <= -0.03) & (gap > -0.05),
}
summary = {}
for name, cond in rules.items():
    mask = elig & cond
    r, g, c, n = L.run_rule(D, gdt, cdt, mask, stress=('cost_0.38pct_pre2017tax', L.COST_DT_PRE2017))
    res[f'{name}__daytrade'] = r
    summary[f'{name}__daytrade'] = {w: L.brief(r[w]) for w in L.WINDOWS}
    # 敏感度：假設收盤跌停也能在收盤賣出（樂觀；實務上鎖跌停多半賣不掉）
    r_nv, _, _, _ = L.run_rule(D, idr, L.COST_DT, mask)
    res[f'{name}__daytrade_naive_close_exit'] = r_nv
    summary[f'{name}__daytrade_naive_close_exit'] = {w: L.brief(r_nv[w]) for w in L.WINDOWS}
    # 敏感度：收盤跌停者一半在收盤賣出、一半隔日開盤賣
    cldm = D['close_limit_down'].fillna(False).astype(bool)
    g_half = gdt.where(~cldm, 0.5 * gdt + 0.5 * idr); c_half = cdt.where(~cldm, 0.5 * L.COST_RT + 0.5 * L.COST_DT)
    r_h, _, _, _ = L.run_rule(D, g_half, c_half, mask)
    res[f'{name}__daytrade_half_fill_at_limit_down'] = r_h
    summary[f'{name}__daytrade_half_fill_at_limit_down'] = {w: L.brief(r_h[w]) for w in L.WINDOWS}
    if name == 'gap_down_3to5': continue
    # 持有到隔日收盤：cohort 報酬、兩池交替的日報酬
    gross, cc, nn = L.cohort(g2, L.COST_RT, mask)
    gd1, _, _ = L.cohort(idr, 0.0, mask & g2.notna())          # 第 1 天（開→收）
    gd2, _, _ = L.cohort(r_next, 0.0, mask & g2.notna())       # 第 2 天（收→收）
    CAL = D['CAL']
    d1 = gd1.reindex(CAL) - L.COST_RT          # 進場日扣整筆成本
    d2 = gd2.reindex(CAL).shift(1)
    daily = 0.5 * d1.fillna(0) + 0.5 * d2.fillna(0)
    rr = L.evaluate_all(D, gross, L.COST_RT, nn, daily, need_next=True, stress=('cost_1pct', L.COST_RT_STRESS))
    res[f'{name}__hold_to_next_close'] = rr
    summary[f'{name}__hold_to_next_close'] = {w: L.brief(rr[w]) for w in L.WINDOWS}

# 資料檢查：極端事件、各來源下的平均
chk = {}
for name in ['gap_down_5', 'gap_up_3']:
    mask = (elig & rules[name]).fillna(False).astype(bool) & gdt.notna()
    st = mask.stack(); st = st[st]
    ix = st.index
    vals = gdt.stack().reindex(ix)
    srcs = D['src'].stack().reindex(ix)
    cld = D['close_limit_down'].stack().reindex(ix).astype(float)
    chk[name] = {'n': int(len(vals)), 'p1_pct': round(float(vals.quantile(0.01) * 100), 2), 'p99_pct': round(float(vals.quantile(0.99) * 100), 2),
                 'share_close_limit_down_exit_next_open': round(float(cld.mean()), 4),
                 'mean_trade_by_src_pct': {int(k): round(float(v * 100), 3) for k, v in vals.groupby(srcs.values).mean().items()},
                 'n_by_src': {int(k): int(v) for k, v in vals.groupby(srcs.values).size().items()}}
# 收盤跌停事件的拆解（naive 與保守出場差在哪）
for name in ['gap_down_3', 'gap_down_5']:
    mask = (elig & rules[name]).fillna(False).astype(bool) & idr.notna()
    cldm = mask & D['close_limit_down'].fillna(False).astype(bool)
    for w in L.WINDOWS:
        dm = L.day_set(D, w); dd = dm[dm].index
        a = idr.reindex(dd).where(cldm.reindex(dd)).stack().dropna(); b = gdt.reindex(dd).where(cldm.reindex(dd)).stack().dropna()
        o_ = idr.reindex(dd).where(mask.reindex(dd) & ~cldm.reindex(dd)).stack().dropna()
        chk.setdefault(name + '_limit_down_split', {})[w] = {
            'share_events_close_limit_down': round(len(a) / max(len(a) + len(o_), 1), 3),
            'limit_down_events_mean_open_to_close_pct': round(float(a.mean() * 100), 2),
            'limit_down_events_mean_open_to_next_open_pct': round(float(b.mean() * 100), 2),
            'other_events_mean_open_to_close_pct': round(float(o_.mean() * 100), 2)}
res['data_checks'] = chk
res['runtime_sec'] = round(time.time() - t0, 1)
L.save('results_r2.json', res)
print(f"done {res['runtime_sec']}s")
for k, v in summary.items():
    print('==', k)
    print(pd.DataFrame(v).T.to_string())
print(chk)

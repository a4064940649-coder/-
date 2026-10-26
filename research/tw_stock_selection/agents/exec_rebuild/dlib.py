#!/usr/bin/env python3
"""處置股固定槽位模擬（事前固定）。

* 事件：disposal_5y（上市、面板內），依公告日（同日依檔案順序）排隊；進場＝處置第一個交易日收盤，出場＝最後一個交易日收盤。
* 同一檔處置期重疊（本筆第一天 ≤ 前一筆保留事件的出場日）→ 只保留第一筆；仍持有中的代碼也不再進場。
* 再處置：同一檔在本筆第一天前 30 個交易日內有另一筆處置結束（同 agents/events/ev5 定義）→ 來回成本 1.2%，否則 0.8%（進出各半）。
* 進場濾網：進場日 ret ≥ 9% → 不買。出場：最後一天 ret ≤ −9% → 延到第一個 ret > −9% 的交易日收盤。
* 槽位：K 槽，每筆金額 = min(當下槽位資產淨值 ÷ K, 5% × min(進場日成交值, 出場日成交值), 可用閒置資金)；沒有空槽就跳過。
* 不做每日再平衡：部位買進後隨價格漂移，直到出場。
* 閒置：現金（0 報酬）或 0050（買 手續費 0.0855%、賣 0.0855%＋稅 0.1%）。
"""
import numpy as np, pandas as pd
import exlib as L

ETF_BUY, ETF_SELL = L.FEE, L.FEE + 0.001
IDX, AC, RETN, VN, CI = L.IDX, L.AC, L.RETN, L.VN, L.CI
E50 = L.E50.to_numpy()

def build_events(start='2021-10-01', ret_filter=True, delay_exit=True, drop_overlap=True, cost_first=0.008, cost_second=0.012):
    dp = L.dsp[L.dsp['code'].isin(L.COLS)].copy()
    dp['_o'] = np.arange(len(dp)); dp = dp.sort_values(['date', '_o']).reset_index(drop=True)
    prev_end, kept_exit = {}, {}
    ev, stat = {}, {'records': 0, 'dropped_overlap': 0, 'dropped_entry_ret_ge_9pct': 0, 'delayed_exit': 0, 'second': 0, 'kept': 0}
    T = len(IDX); s0 = IDX.searchsorted(pd.Timestamp(start))
    for _, r in dp.iterrows():
        c = r['code']; j = CI[c]
        i0 = IDX.searchsorted(r['start']); i1 = IDX.searchsorted(r['end'], 'right') - 1
        second = c in prev_end and prev_end[c] >= i0 - 30 and prev_end[c] < i0
        prev_end[c] = max(prev_end.get(c, -1), i1)
        if i0 >= T or i1 <= i0 or not np.isfinite(L.ADJ_RAW[i0, j]) or i0 < s0: continue
        stat['records'] += 1
        if drop_overlap and c in kept_exit and i0 <= kept_exit[c]:
            stat['dropped_overlap'] += 1; continue
        if ret_filter and not (np.isfinite(RETN[i0, j]) and RETN[i0, j] < 0.09):
            stat['dropped_entry_ret_ge_9pct'] += 1; continue
        k = i1
        if delay_exit:
            while k < T - 1 and not (np.isfinite(RETN[k, j]) and RETN[k, j] > -0.09): k += 1
            if k > i1: stat['delayed_exit'] += 1
        kept_exit[c] = k
        rt = cost_second if second else cost_first; stat['second'] += int(second); stat['kept'] += 1
        vals = [v for v in (VN[i0, j], VN[k, j]) if np.isfinite(v) and v > 0]
        cap = 0.05 * min(vals) if vals else 0.0
        ev.setdefault(i0, []).append((j, k, rt, cap))
    return ev, stat

class Sleeve:
    def __init__(self, events, K=10, idle='cash', use_cap=True):
        self.ev, self.K, self.idle, self.use_cap = events, K, idle, use_cap
        self.cash = 0.0; self.u50 = 0.0; self.pos = {}       # j -> [units, exit_day, half_cost]
        self.stat = {'entered': 0, 'skipped_no_slot': 0, 'capped': 0, 'cap_fill_ratio': [], 'skipped_held': 0}
        self.trades = []
    def idle_value(self, t): return self.cash + (self.u50 * E50[t] if self.u50 else 0.0)
    def pos_value(self, t): return sum(p[0] * AC[t, j] for j, p in self.pos.items())
    def equity(self, t): return self.idle_value(t) + self.pos_value(t)
    def deposit(self, x, t):                  # 存入閒置資金 x ≥ 0（持 0050 版本要付買進成本）
        if self.idle == 'cash': self.cash += x
        else: self.u50 += x * (1 - ETF_BUY) / E50[t]
    def withdraw_idle(self, x, t):           # 從閒置提出 x（淨額），回傳實際提出
        avail = self.cash if self.idle == 'cash' else self.u50 * E50[t] * (1 - ETF_SELL)
        x = min(x, max(avail, 0.0))
        if self.idle == 'cash': self.cash -= x
        else: self.u50 -= x / (1 - ETF_SELL) / E50[t]
        return x
    def step(self, t):
        # 出場
        for j in [j for j, p in self.pos.items() if p[1] == t]:
            units, _, hc, t_in, v_in = self.pos.pop(j)
            proceeds = units * AC[t, j] * (1 - hc); self.trades.append((t_in, t, proceeds / v_in - 1))
            self.deposit(proceeds, t)
        # 進場
        for j, k, rt, cap in self.ev.get(t, []):
            if j in self.pos: self.stat['skipped_held'] += 1; continue
            if len(self.pos) >= self.K: self.stat['skipped_no_slot'] += 1; continue
            E = self.equity(t); amt = E / self.K
            if self.use_cap and cap < amt: amt = cap; self.stat['capped'] += 1
            amt = self.withdraw_idle(amt, t)
            self.stat['cap_fill_ratio'].append(amt / (E / self.K) if E > 0 else 0)
            if amt <= 0: continue
            self.pos[j] = [amt * (1 - rt / 2) / AC[t, j], k, rt / 2, t, amt]; self.stat['entered'] += 1

def run_standalone(capital, K=10, idle='cash', use_cap=True, t_start='2021-10-01', t_end=None, **evkw):
    ev, est = build_events(start=t_start, **evkw)
    t0 = IDX.searchsorted(pd.Timestamp(t_start)); t1 = L.pos(L.dates[-1]) if t_end is None else IDX.searchsorted(pd.Timestamp(t_end), 'right') - 1
    s = Sleeve(ev, K, idle, use_cap)
    nav = np.empty(t1 - t0 + 1); held = np.empty(t1 - t0 + 1, int); inv = np.empty(t1 - t0 + 1)
    # 起始：前一日收盤放入資金
    s.deposit(capital, t0 - 1)
    nav0 = capital
    for t in range(t0, t1 + 1):
        s.step(t)
        nav[t - t0] = s.equity(t); held[t - t0] = len(s.pos); inv[t - t0] = s.pos_value(t) / nav[t - t0]
    navs = pd.Series(np.r_[nav0, nav], index=IDX[t0 - 1:t1 + 1])
    st = dict(est); st.update({k: v for k, v in s.stat.items() if k != 'cap_fill_ratio'})
    cf = np.array(s.stat['cap_fill_ratio'])
    st['mean_fill_vs_full_slot'] = round(float(cf.mean()), 3) if len(cf) else None
    st['share_trades_capacity_bound'] = round(float((cf < 0.999).mean()), 3) if len(cf) else None
    dist = pd.Series(held).value_counts().sort_index()
    st['held_count_distribution_share_days'] = {int(k): round(float(v / len(held)), 3) for k, v in dist.items()}
    st['mean_held'] = round(float(held.mean()), 2); st['mean_invested_share'] = round(float(inv.mean()), 3)
    st['share_days_all_slots_full'] = round(float((held >= K).mean()), 3)
    tr = np.array([x[2] for x in s.trades])
    if len(tr) > 2:
        st['closed_trades'] = int(len(tr)); st['trade_net_mean_pct'] = round(float(tr.mean() * 100), 2); st['trade_net_median_pct'] = round(float(np.median(tr) * 100), 2)
        st['trade_win_rate'] = round(float((tr > 0).mean()), 3); st['trade_t'] = round(float(tr.mean() / (tr.std(ddof=1) / np.sqrt(len(tr)))), 2)
    return navs, st

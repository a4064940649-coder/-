#!/usr/bin/env python3
"""上櫃處置股（history.csv.gz，2025-09-24 起）用正式版定義重算，並做 10 槽組合（仿 exec_rebuild dlib.Sleeve）。
A：上市單獨 2021-10～2026-10；B：上市單獨 2025-09-24～；C：上市＋上櫃合併 2025-09-24～。輸出 results_tpex.json。"""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rlib as L

FEE = 0.001425 * 0.6; ETF_BUY, ETF_SELL = FEE, FEE + 0.001
E50 = L.E50.to_numpy()
CFG = L.CFG_OFFICIAL
res = {}

# ---------- 上市正式版交易 ----------
L.set_market(L.TWSE)
TW = L.trades(CFG); TW['mkt'] = 'TWSE'
TW_AC = L.adj.ffill().to_numpy(); TW_V = L.VN.copy()
# ---------- 上櫃 ----------
MP = L.load_hist_market('TPEX')
recP = L.records_market(MP)
L.set_market(MP)
TP, stP = L.trades(CFG, recP, with_stat=True); TP['mkt'] = 'TPEX'
TP_ref = {'tradable_no_liq': L.trades(dict(CFG, liq='none'), recP), 'cost0.5': L.trades(dict(CFG, cost='0.5'), recP),
          'overlap_first': L.trades(dict(CFG, overlap='first'), recP), 'entry_lock': L.trades(dict(CFG, entry='lock'), recP),
          'exit_none': L.trades(dict(CFG, exit='none'), recP)}
TP_AC = MP['ADJ_FFILL']; TP_V = MP['VN']
# 上市用 history 管線同期對照（檢查兩條管線一致）
MW = L.load_hist_market('TWSE'); recW = L.records_market(MW); L.set_market(MW)
TWh = L.trades(CFG, recW)
L.set_market(L.TWSE)

W0 = pd.Timestamp('2025-09-24')
PERW = {'window_2025-09-24_2026-10': ('2025-09-24', '2026-10-31'), '2025Q4': ('2025-09-24', '2025-12-31'), '2026': ('2026-01-01', '2026-10-31')}
def per(df): return {k: L.stats(df[(df['date'] >= a) & (df['date'] <= b)]) for k, (a, b) in PERW.items()}
res['TPEX_official'] = {'counts': stP, 'by_period': per(TP), 'raw_net': per(TP.assign(net=TP['raw_net'])),
                        'vs0050_net': per(TP.assign(net=TP['ab50'] - TP['cost'])),
                        'variants': {k: per(v)['window_2025-09-24_2026-10'] for k, v in TP_ref.items()},
                        'monthly_n': {str(k): int(v) for k, v in TP.groupby(TP['date'].dt.to_period('M')).size().items()},
                        'monthly_mean_net_pct': {str(k): round(v * 100, 2) for k, v in TP.groupby(TP['date'].dt.to_period('M'))['net'].mean().items()},
                        'top5': [{'code': r.code, 'date': str(r.date.date()), 'exit': str(r.exit.date()), 'net_pct': round(r.net * 100, 1)} for r in TP.sort_values('net', ascending=False).head(5).itertuples()],
                        'bottom5': [{'code': r.code, 'date': str(r.date.date()), 'exit': str(r.exit.date()), 'net_pct': round(r.net * 100, 1)} for r in TP.sort_values('net').head(5).itertuples()]}
TWw = TW[TW['date'] >= W0]
res['TWSE_official_same_window_panel'] = per(TWw)
res['TWSE_official_same_window_history_pipeline'] = per(TWh)
BOTH = pd.concat([TWw, TP])
res['TWSE+TPEX_events_pooled'] = per(BOTH)
TP.assign(date=TP['date'].dt.strftime('%Y-%m-%d'), exit=TP['exit'].dt.strftime('%Y-%m-%d'), ann_date=TP['ann_date'].dt.strftime('%Y-%m-%d')).drop(columns=['j']).to_csv(HERE / 'official_trades_tpex.csv', index=False)


# ---------- 10 槽組合 ----------
class Sleeve:
    """仿 dlib.Sleeve：K 槽、每筆＝min(淨值/K, cap, 可用閒置)，沒空槽跳過，不再平衡；閒置現金或 0050。"""
    def __init__(self, ev, AC, K=10, idle='0050'):
        self.ev, self.AC, self.K, self.idle = ev, AC, K, idle
        self.cash = 0.0; self.u50 = 0.0; self.pos = {}; self.trades = []
        self.stat = {'entered': 0, 'skipped_no_slot': 0, 'skipped_held': 0, 'capped': 0, 'by_mkt_entered': {}, 'by_mkt_skipped': {}}
    def idle_value(self, t): return self.cash + self.u50 * E50[t]
    def pos_value(self, t): return sum(p[0] * self.AC[t, j] for j, p in self.pos.items())
    def equity(self, t): return self.idle_value(t) + self.pos_value(t)
    def deposit(self, x, t):
        if self.idle == 'cash': self.cash += x
        else: self.u50 += x * (1 - ETF_BUY) / E50[t]
    def withdraw(self, x, t):
        avail = self.cash if self.idle == 'cash' else self.u50 * E50[t] * (1 - ETF_SELL)
        x = min(x, max(avail, 0.0))
        if self.idle == 'cash': self.cash -= x
        else: self.u50 -= x / (1 - ETF_SELL) / E50[t]
        return x
    def step(self, t):
        for j in [j for j, p in self.pos.items() if p[1] == t]:
            u, _, hc, t_in, v_in, mk = self.pos.pop(j)
            pr = u * self.AC[t, j] * (1 - hc); self.trades.append((t_in, t, pr / v_in - 1, mk)); self.deposit(pr, t)
        for j, k, rt, cap, mk in self.ev.get(t, []):
            if j in self.pos: self.stat['skipped_held'] += 1; continue
            if len(self.pos) >= self.K:
                self.stat['skipped_no_slot'] += 1; self.stat['by_mkt_skipped'][mk] = self.stat['by_mkt_skipped'].get(mk, 0) + 1; continue
            E = self.equity(t); amt = E / self.K
            if cap < amt: amt = cap; self.stat['capped'] += 1
            amt = self.withdraw(amt, t)
            if amt <= 0: continue
            self.pos[j] = [amt * (1 - rt / 2) / self.AC[t, j], k, rt / 2, t, amt, mk]; self.stat['entered'] += 1
            self.stat['by_mkt_entered'][mk] = self.stat['by_mkt_entered'].get(mk, 0) + 1


def make_events(dfs, Vs, offs):
    ev = {}
    for df, V, off in zip(dfs, Vs, offs):
        for r in df.sort_values(['i0', 'ann_date', 'ord']).itertuples():
            v = V[r.i0 - 1, r.j]
            cap = 0.05 * v if np.isfinite(v) and v > 0 else 0.0
            ev.setdefault(r.i0, []).append((r.j + off, r.i1, r.cost, cap, r.mkt))
    return ev


def run_sleeve(ev, AC, start, idle, end=None, K=10, capital=1e6):
    t0 = L.IDX.searchsorted(pd.Timestamp(start)); t1 = L.T - 1 if end is None else L.IDX.searchsorted(pd.Timestamp(end), 'right') - 1
    s = Sleeve(ev, AC, K, idle); s.deposit(capital, t0 - 1)
    nav = [capital]; held = []
    for t in range(t0, t1 + 1):
        s.step(t); nav.append(s.equity(t)); held.append(len(s.pos))
    nav = pd.Series(nav, index=L.IDX[t0 - 1:t1 + 1])
    return nav, s, np.array(held)


def nav_stats(nav, s=None, held=None):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    b = L.E50.reindex(nav.index); b = b / b.iloc[0]
    r = nav.pct_change().dropna(); rb = b.pct_change().dropna()
    mr = nav.resample('ME').last().pct_change().dropna(); mb = b.resample('ME').last().pct_change().dropna()
    # 第一個月：從起點算
    ex_m = (mr - mb).dropna()
    o = {'start': str(nav.index[0].date()), 'end': str(nav.index[-1].date()), 'total_ret_pct': round((nav.iloc[-1] / nav.iloc[0] - 1) * 100, 1),
         'CAGR_pct': round(((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100, 1), 'maxDD_pct': round(float((nav / nav.cummax() - 1).min()) * 100, 1),
         'vol_pct': round(float(r.std() * math.sqrt(252)) * 100, 1),
         '0050_total_ret_pct': round((b.iloc[-1] - 1) * 100, 1), '0050_CAGR_pct': round((b.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
         '0050_maxDD_pct': round(float((b / b.cummax() - 1).min()) * 100, 1),
         't_vs_0050_daily': L.tstat(r - rb), 't_vs_0050_monthly': L.tstat(ex_m), 'n_months': int(len(ex_m))}
    ys = {}
    for y in sorted(set(nav.index.year)):
        prev = nav[nav.index.year < y]; cur = nav[nav.index.year == y]
        seg = pd.concat([prev.iloc[-1:], cur]) if len(prev) else cur
        bp = b[b.index.year < y]; bc = b[b.index.year == y]; bs = pd.concat([bp.iloc[-1:], bc]) if len(bp) else bc
        ys[int(y)] = {'ret_pct': round((seg.iloc[-1] / seg.iloc[0] - 1) * 100, 1), '0050_pct': round((bs.iloc[-1] / bs.iloc[0] - 1) * 100, 1),
                      'maxDD_pct': round(float((seg / seg.cummax() - 1).min()) * 100, 1)}
    o['yearly'] = ys
    if s is not None:
        tr = pd.DataFrame(s.trades, columns=['t_in', 't_out', 'ret', 'mkt'])
        o['entered'] = s.stat['entered']; o['skipped_no_slot'] = s.stat['skipped_no_slot']; o['capped'] = s.stat['capped']
        o['entered_by_mkt'] = s.stat['by_mkt_entered']; o['skipped_by_mkt'] = s.stat['by_mkt_skipped']
        o['mean_held'] = round(float(held.mean()), 2); o['share_days_full'] = round(float((held >= s.K).mean()), 3); o['share_days_empty'] = round(float((held == 0).mean()), 3)
        if len(tr) > 2:
            o['closed_trades'] = int(len(tr)); o['trade_net_mean_pct'] = round(tr['ret'].mean() * 100, 2); o['trade_net_median_pct'] = round(tr['ret'].median() * 100, 2)
            o['trade_t'] = L.tstat(tr['ret']); o['trade_hit'] = round(float((tr['ret'] > 0).mean()), 3)
            o['trade_by_mkt'] = {m: {'n': int(len(g)), 'mean_pct': round(g['ret'].mean() * 100, 2), 'median_pct': round(g['ret'].median() * 100, 2)} for m, g in tr.groupby('mkt')}
    return o


nT = TW_AC.shape[1]
AC_C = np.hstack([TW_AC, TP_AC]); V_C = np.hstack([TW_V, TP_V])
ev_tw = make_events([TW], [TW_V], [0])
# 合併版：同日排隊依公告日、檔案順序（跨市場）
EVROWS = pd.concat([TW.assign(off=0), TP.assign(off=nT)])
ev_c = {}
for r in EVROWS.sort_values(['i0', 'ann_date', 'ord']).itertuples():
    V = TW_V if r.mkt == 'TWSE' else TP_V
    v = V[r.i0 - 1, r.j]; cap = 0.05 * v if np.isfinite(v) and v > 0 else 0.0
    ev_c.setdefault(r.i0, []).append((r.j + r.off, r.i1, r.cost, cap, r.mkt))

port = {}
navs = {}
for idle in ('0050', 'cash'):
    nav, s, h = run_sleeve(ev_tw, TW_AC, '2021-10-01', idle); port[f'A_TWSE_2021-10_{idle}'] = nav_stats(nav, s, h); navs[f'A_{idle}'] = nav
    for a, b, nm in (('2021-10-01', '2023-12-31', 'sel'), ('2024-01-01', None, 'val')):
        seg = nav[(nav.index >= pd.Timestamp(a)) & ((nav.index <= pd.Timestamp(b)) if b else True)]
        pre = nav[nav.index < pd.Timestamp(a)].iloc[-1:]
        port[f'A_TWSE_2021-10_{idle}'][f'{nm}_window'] = nav_stats(pd.concat([pre, seg]))
    nav, s, h = run_sleeve(ev_tw, TW_AC, '2025-09-24', idle); port[f'B_TWSE_2025-09-24_{idle}'] = nav_stats(nav, s, h); navs[f'B_{idle}'] = nav
    nav, s, h = run_sleeve(ev_c, AC_C, '2025-09-24', idle); port[f'C_TWSE+TPEX_2025-09-24_{idle}'] = nav_stats(nav, s, h); navs[f'C_{idle}'] = nav
    # 敏感度：K＝20（合併後事件變多，10 槽是否太少）
    nav, s, h = run_sleeve(ev_c, AC_C, '2025-09-24', idle, K=20); port[f'C_K20_TWSE+TPEX_2025-09-24_{idle}'] = nav_stats(nav, s, h)
    nav, s, h = run_sleeve(ev_tw, TW_AC, '2025-09-24', idle, K=20); port[f'B_K20_TWSE_2025-09-24_{idle}'] = nav_stats(nav, s, h)
res['portfolio'] = port
pd.DataFrame(navs).to_parquet(HERE / 'sleeve_nav.parquet')


def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


(HERE / 'results_tpex.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
K = ('n', 'mean_pct', 'median_pct', 'hit', 't_month', 't_month_gross', 't_date', 'n_months', 'ex_top5pct_mean_pct', 'ex_top5pct_t_month', 'mean_raw_net_pct')
print('TPEX counts', stP)
for nm in ('TPEX_official', ):
    for p, x in res[nm]['by_period'].items(): print(nm, p, {k: x.get(k) for k in K})
    for p, x in res[nm]['vs0050_net'].items(): print(nm, 'vs0050', p, {k: x.get(k) for k in ('n', 'mean_pct', 'median_pct', 't_month')})
    for p, x in res[nm]['variants'].items(): print(nm, 'var', p, {k: x.get(k) for k in ('n', 'mean_pct', 'median_pct', 't_month')})
    print(res[nm]['monthly_n']); print(res[nm]['monthly_mean_net_pct']); print(res[nm]['top5']); print(res[nm]['bottom5'])
for nm in ('TWSE_official_same_window_panel', 'TWSE_official_same_window_history_pipeline', 'TWSE+TPEX_events_pooled'):
    for p, x in res[nm].items(): print(nm, p, {k: x.get(k) for k in K})
for k, v in port.items():
    print('==', k, {q: v.get(q) for q in ('total_ret_pct', 'CAGR_pct', 'maxDD_pct', 'vol_pct', '0050_total_ret_pct', '0050_CAGR_pct', '0050_maxDD_pct', 't_vs_0050_daily', 't_vs_0050_monthly',
                                       'entered', 'skipped_no_slot', 'mean_held', 'share_days_full', 'closed_trades', 'trade_net_mean_pct', 'trade_net_median_pct', 'trade_t', 'trade_by_mkt', 'skipped_by_mkt')})
    print('   yearly', v['yearly'])
    for w in ('sel_window', 'val_window'):
        if w in v: print('  ', w, {q: v[w].get(q) for q in ('total_ret_pct', 'CAGR_pct', 'maxDD_pct', '0050_CAGR_pct', 't_vs_0050_monthly')})

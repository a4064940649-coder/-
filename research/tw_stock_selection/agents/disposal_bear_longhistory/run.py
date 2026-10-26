#!/usr/bin/env python3
"""處置股（近似規則重建）2005～2021 長歷史與空頭期檢驗。所有規則事前固定，見 FINDINGS。

事件
* 2005-01～2021-09：完全沿用 bt_disposal_proxy_history.py 的近似規則（6 日還原漲幅 ≥32% 且比加權指數多 ≥20pp，連續 3 日 → 隔日起處置 10 日；
  期間不重複觸發）。進場＝處置第一天收盤、出場＝第 10 天收盤。
* 2021-10 起（含 2022 空頭）：disposal_5y 真實名單（上市、面板內）；進場＝start 收盤、出場＝end 收盤。
* 執行濾網：進場日 ret ≥ 漲跌幅−1% → 買不到；出場日 ret ≤ −(漲跌幅−0.6%)（或無成交）→ 延到第一個非跌停且有成交日收盤。
  漲跌幅：2015-06-01 前 7%，之後 10%。持有中下市（之後無資料）→ 以最後成交價計。
* 同一檔：本筆進場日 ≤ 前一筆保留事件的（延後後）出場日 → 只留第一筆。
統計：異常報酬＝個股含息報酬 − 流動性合格（60 日均值 ≥1,000 萬、股價 ≥10）上市等權（同 agents/events）；
  t_month＝同進場月平均後的 t；成本 0.8%／1.2% 來回。
市況：進場前一日加權指數收盤 < 200 日均線（加權指數自身日曆計算）＝空頭；另列六段空頭。
組合：agents/exec_rebuild/dlib.py 的 Sleeve（10 槽固定金額、5% 成交值上限、閒置現金或 0050），不改原檔。
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'agents' / 'exec_rebuild'))
import exlib as L, dlib as Dl                                    # noqa: E402

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
adj, C, V, R1, H = L.adj, L.C, L.V, L.RET, L.H
FLAG = pd.read_parquet(P / 'flag.parquet').reindex(index=adj.index, columns=adj.columns).to_numpy()
idx = adj.index; cols = adj.columns; CI = L.CI; T = len(idx)
RETN, ADJN = R1.to_numpy(), adj.to_numpy()
LIM = np.where(idx < pd.Timestamp('2015-06-01'), 0.07, 0.10)

# ---- 加權指數、200 日均線（自身日曆）----
twr = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
ma = twr.rolling(200).mean()
bear_tw = (twr < ma).where(ma.notna())
tx = twr.reindex(idx).ffill()                                     # 同 proxy 腳本
bear_day = bear_tw.reindex(idx, method='ffill')                   # 第 i 日收盤是否在均線下
BEAR_ENTRY = np.r_[np.nan, bear_day.to_numpy(dtype=float)[:-1]]   # 進場前一日

# ---- 對照池 ----
liq = (V.rolling(60, min_periods=60).mean() >= 1e7) & (C >= 10)
mkt = R1.where(liq).mean(axis=1).fillna(0)
LA = np.log1p(R1.fillna(0).clip(lower=-0.99)).cumsum().to_numpy()
LM = np.log1p(mkt.clip(lower=-0.99)).cumsum().to_numpy()
LIQN = liq.to_numpy()
lastvalid = np.array([np.max(np.where(np.isfinite(ADJN[:, j]))[0]) if np.isfinite(ADJN[:, j]).any() else -1 for j in range(len(cols))])

# ---- 近似事件（逐字沿用 proxy 規則）----
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
prox = pd.DataFrame(prox, columns=['code', 'i0', 'i1']); prox['start'] = idx[prox.i0]; prox['src'] = 'proxy'

dp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
n_real_raw = len(dp); n_real_notinpanel = int((~dp.code.isin(cols)).sum())
dp = dp[dp.code.isin(cols)].copy(); dp['_o'] = np.arange(len(dp)); dp = dp.sort_values(['date', '_o'])
real = pd.DataFrame({'code': dp.code.values, 'i0': idx.searchsorted(dp.start.values), 'i1': idx.searchsorted(dp.end.values, 'right') - 1})
real = real[(real.i0 < T) & (real.i1 > real.i0)].copy(); real['start'] = idx[real.i0]; real['src'] = 'real'

# ---- 校對（同 proxy 腳本的 ±5 日曆日比對）----
lo, hi = pd.Timestamp('2021-10-01'), pd.Timestamp('2026-09-30')
def match_flags(a, b):
    bset = {}
    for c, d in zip(b.code, b.start): bset.setdefault(c, []).append(d)
    return np.array([any(abs((d - x).days) <= 5 for x in bset.get(c, [])) for c, d in zip(a.code, a.start)])
rw = real[(real.start >= lo) & (real.start <= hi)]; pw = prox[(prox.start >= lo) & (prox.start <= hi)]
prox_true = pd.Series(False, index=prox.index); prox_true.loc[pw.index] = match_flags(pw, rw)
calib = {'real_events': int(len(rw)), 'proxy_events': int(len(pw)), 'recall': round(float(match_flags(rw, pw).mean()), 3),
         'precision': round(float(prox_true.loc[pw.index].mean()), 3)}

# ---- 執行濾網、去重疊、報酬 ----
def execute(ev, entry_filter=True):
    ev = ev.sort_values(['i0']).reset_index(drop=True)
    kept_exit = {}; rows = []; st = {'raw': len(ev), 'no_price_entry': 0, 'untradable_entry': 0, 'overlap': 0, 'delayed_exit': 0, 'delisted_in_hold': 0}
    for r in ev.itertuples():
        c = r.code; j = CI[c]; i0, i1 = r.i0, r.i1
        if not np.isfinite(ADJN[i0, j]): st['no_price_entry'] += 1; continue
        if c in kept_exit and i0 <= kept_exit[c]: st['overlap'] += 1; continue
        if entry_filter and RETN[i0, j] >= LIM[i0] - 0.01: st['untradable_entry'] += 1; continue
        k = i1; dl = False
        if lastvalid[j] < i1:
            dl = True; st['delisted_in_hold'] += 1
        else:
            while k < lastvalid[j] and not (np.isfinite(RETN[k, j]) and RETN[k, j] > -(LIM[k] - 0.006)): k += 1
            if k > i1: st['delayed_exit'] += 1
        kept_exit[c] = k
        raw = math.expm1(LA[k, j] - LA[i0, j]); m = math.expm1(LM[k] - LM[i0])
        fl = FLAG[i0 + 1:k + 1, j]; fl = fl[np.isfinite(fl)]
        rows.append({'code': c, 'j': j, 'i0': i0, 'i1': i1, 'k': k, 'date': idx[i0], 'src': r.src, 'raw': raw, 'ab': raw - m,
                     'bear': BEAR_ENTRY[i0], 'liq': bool(LIQN[i0 - 1, j]) if i0 > 0 else False, 'delisted_in_hold': dl,
                     'corp_flag': bool(np.isin(fl, [3, 4, 5]).any()), 'v_in': L.VN[i0, j], 'v_out': L.VN[k, j]})
    return pd.DataFrame(rows), st

EP, stP = execute(prox)
ER, stR = execute(real)
EP_nf, _ = execute(prox, entry_filter=False)          # 參考：假設漲停也買得到（不可實行）
EP['true_pos'] = False
key = {(c, d) for c, d, t in zip(prox.code, prox.start, prox_true) if t}
EP['true_pos'] = [(c, d) in key for c, d in zip(EP.code, EP.date)]

def tstat(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std() > 0 else None

def summ(df, cost=0.008):
    df = df.dropna(subset=['ab'])
    o = {'n': int(len(df))}
    if len(df) < 5: return o
    ab = df['ab']; net = ab - cost
    m = net.groupby(df['date'].dt.to_period('M')).mean(); d = net.groupby(df['date']).mean()
    def w(q): lo_, hi_ = ab.quantile([q, 1 - q]); return round((ab.clip(lo_, hi_).mean() - cost) * 100, 2)
    top = ab.sort_values(ascending=False)
    pos_sum = ab[ab > 0].sum()
    o.update({'mean_raw_pct': round(df['raw'].mean() * 100, 2), 'mean_ab_pct': round(ab.mean() * 100, 2),
              'net_ab_pct': round(net.mean() * 100, 2), 'net_ab_1.2_pct': round((ab.mean() - 0.012) * 100, 2),
              'median_ab_pct': round(ab.median() * 100, 2), 'median_net_ab_pct': round(net.median() * 100, 2),
              'hit_net_ab': round(float((net > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3),
              't_month': tstat(m), 't_date': tstat(d), 'n_months': int(len(m)),
              'net_winsor1_pct': w(0.01), 'net_winsor5_pct': w(0.05),
              'net_ex_top10_pct': round((top.iloc[10:].mean() - cost) * 100, 2) if len(top) > 15 else None,
              'top10_share_of_total_ab': round(float(top.iloc[:10].sum() / ab.sum()), 2) if ab.sum() > 0 else None,
              'p10_p90_ab_pct': [round(ab.quantile(.1) * 100, 1), round(ab.quantile(.9) * 100, 1)]})
    return o

def sub(df, a=None, b=None, **kw):
    x = df
    if a: x = x[x.date >= pd.Timestamp(a)]
    if b: x = x[x.date <= pd.Timestamp(b)]
    for k, v in kw.items(): x = x[x[k] == v]
    return x

res = {'design': __doc__, 'calibration_proxy_rule_2021-10_2026-09': calib, 'exec_stats_proxy_all_years': stP, 'exec_stats_real': stR,
       'real_list_rows_not_in_panel': {'raw_rows': n_real_raw, 'not_in_panel': n_real_notinpanel}}

# 資料缺漏
last = adj.apply(lambda s: s.last_valid_index())
res['survivorship'] = {
    'codes_with_data_ending_2019-05-16': int((last == pd.Timestamp('2019-05-16')).sum()),
    'note': '2019-05-17 起只有存活者：在 2019-05-16 仍交易、之後下市者（30 檔）其後資料缺漏；2019-05～2021-09 期間新上市又下市者完全不在面板（檔數未知）。',
    'stocks_with_price_by_year': {int(y): int(v) for y, v in adj.notna().groupby(idx.year).any().sum(axis=1).items() if 2016 <= y <= 2022}}

MAIN = sub(EP, '2005-01-01', '2021-09-30')
R22 = sub(ER, '2022-01-01', '2022-12-31')
BEARS = {'2008-05~2009-02': ('2008-05-01', '2009-02-28'), '2011-08~2011-12': ('2011-08-01', '2011-12-31'), '2015-06~2016-02': ('2015-06-01', '2016-02-29'),
         '2018-10~2019-01': ('2018-10-01', '2019-01-31'), '2020-02~2020-03': ('2020-02-01', '2020-03-31'), '2022-03~2022-12': ('2022-03-01', '2022-12-31')}
def in_bears(df):
    m = np.zeros(len(df), bool)
    for a, b in BEARS.values(): m |= ((df.date >= a) & (df.date <= b)).to_numpy()
    return m

G = {}
G['main_2005-01_2021-09'] = MAIN
G['7pct_2005-01_2015-05'] = sub(EP, '2005-01-01', '2015-05-31')
G['10pct_2015-06_2021-09'] = sub(EP, '2015-06-01', '2021-09-30')
G['with_delisted_2005-01_2019-05-16'] = sub(EP, '2005-01-01', '2019-05-16')
G['survivors_only_2019-05-17_2021-09'] = sub(EP, '2019-05-17', '2021-09-30')
G['bull_MA200_2005-2021-09'] = MAIN[MAIN.bear == 0]
G['bear_MA200_2005-2021-09'] = MAIN[MAIN.bear == 1]
G['bear_MA200_7pct'] = G['7pct_2005-01_2015-05'][G['7pct_2005-01_2015-05'].bear == 1]
G['bear_MA200_10pct'] = G['10pct_2015-06_2021-09'][G['10pct_2015-06_2021-09'].bear == 1]
G['liquid_main'] = MAIN[MAIN.liq]
G['no_corp_flag_main'] = MAIN[~MAIN.corp_flag]
G['REF_untradable_included_main'] = sub(EP_nf, '2005-01-01', '2021-09-30')
G['REF_untradable_included_7pct'] = sub(EP_nf, '2005-01-01', '2015-05-31')
G['REF_untradable_included_10pct'] = sub(EP_nf, '2015-06-01', '2021-09-30')
G['liquid_7pct'] = G['7pct_2005-01_2015-05'][G['7pct_2005-01_2015-05'].liq]
G['liquid_10pct'] = G['10pct_2015-06_2021-09'][G['10pct_2015-06_2021-09'].liq]
G['illiquid_main'] = MAIN[~MAIN.liq]
for k, (a, b) in BEARS.items():
    G[f'period_{k}' + ('_real' if k.startswith('2022') else '_proxy')] = sub(ER if k.startswith('2022') else EP, a, b)
G['period_2022-03~2022-12_proxy_for_reference'] = sub(EP, '2022-03-01', '2022-12-31')
POOL_ALL = pd.concat([MAIN, R22])
bear_union = POOL_ALL[(POOL_ALL.bear == 1).to_numpy() | in_bears(POOL_ALL)]
G['DECISION_bear_union_MA200_or_6periods_2005-2022'] = bear_union
G['bear_MA200_pooled_2005-2022'] = POOL_ALL[POOL_ALL.bear == 1]
G['six_periods_pooled'] = POOL_ALL[in_bears(POOL_ALL)]
G['real_2021-10_2026-10'] = sub(ER, '2021-10-01')
G['real_bear_MA200_2021-10_2026'] = sub(ER, '2021-10-01')[sub(ER, '2021-10-01').bear == 1]
G['proxy_2021-10_2026-09'] = sub(EP, '2021-10-01', '2026-09-30')
G['proxy_true_pos_2021-10_2026-09'] = G['proxy_2021-10_2026-09'][G['proxy_2021-10_2026-09'].true_pos]
G['proxy_false_pos_2021-10_2026-09'] = G['proxy_2021-10_2026-09'][~G['proxy_2021-10_2026-09'].true_pos]
G['real_2021-10_2026-09'] = sub(ER, '2021-10-01', '2026-09-30')
G['liquid_bear_union'] = bear_union[bear_union.liq]
res['groups'] = {k: summ(v) for k, v in G.items()}
res['yearly_main'] = {}
YR = pd.concat([EP[EP.date <= '2021-09-30'], ER[ER.date >= '2021-10-01']])
for y, g in YR.groupby(YR.date.dt.year):
    res['yearly_main'][int(y)] = {'src': 'proxy' if y <= 2020 else ('proxy≤09/real≥10' if y == 2021 else 'real'), 'n': int(len(g)),
                                  'net_ab_pct': round((g.ab.mean() - 0.008) * 100, 2), 'median_ab_pct': round(g.ab.median() * 100, 2),
                                  't_month': tstat((g.ab - 0.008).groupby(g.date.dt.to_period('M')).mean()), 'bear_share': round(float(g.bear.mean()), 2)}
# 右尾檢查：前 10 大事件
top = MAIN.sort_values('ab', ascending=False).head(10)
res['top10_main'] = [{'code': r.code, 'date': str(r.date.date()), 'raw_pct': round(r.raw * 100, 1), 'ab_pct': round(r.ab * 100, 1), 'corp_flag': r.corp_flag, 'liq': r.liq} for r in top.itertuples()]
# bull-bear 差異
def welch(a, b):
    a, b = a.dropna(), b.dropna()
    return {'diff_pct': round(float((a.mean() - b.mean()) * 100), 2), 't': round(float((a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))), 2)}
res['bear_minus_bull_main'] = welch(G['bear_MA200_2005-2021-09'].ab, G['bull_MA200_2005-2021-09'].ab)

# ---- 偏誤校正 ----
g = res['groups']
gap = g['real_2021-10_2026-09']['mean_ab_pct'] - g['proxy_2021-10_2026-09']['mean_ab_pct']
res['bias_calibration'] = {
    'given_real_ab_pct': 3.62, 'given_proxy_ab_pct': 2.60, 'given_precision': 0.71, 'given_recall': 0.56,
    'this_pipeline_real_ab_pct': g['real_2021-10_2026-09']['mean_ab_pct'], 'this_pipeline_proxy_ab_pct': g['proxy_2021-10_2026-09']['mean_ab_pct'],
    'this_pipeline_true_pos_ab_pct': g['proxy_true_pos_2021-10_2026-09']['mean_ab_pct'], 'this_pipeline_false_pos_ab_pct': g['proxy_false_pos_2021-10_2026-09']['mean_ab_pct'],
    'gap_real_minus_proxy_pp': round(gap, 2),
    'main_net_ab_pct_proxy': g['main_2005-01_2021-09']['net_ab_pct'],
    'main_net_ab_pct_shift_adjusted': round(g['main_2005-01_2021-09']['net_ab_pct'] + gap, 2),
    'bear_union_net_ab_pct_shift_adjusted': round(g['DECISION_bear_union_MA200_or_6periods_2005-2022']['net_ab_pct'] + gap, 2)}

# ---- 事前決策 ----
d = g['DECISION_bear_union_MA200_or_6periods_2005-2022']
fail_bear = (d['net_ab_pct'] <= 0) or (d['t_month'] is None or d['t_month'] < 1.5) or (d['median_net_ab_pct'] <= 0)
fail_regime = (g['7pct_2005-01_2015-05']['net_ab_pct'] <= 0) or (g['10pct_2015-06_2021-09']['net_ab_pct'] <= 0)
res['decisions'] = {'bear_rule_inputs': {k: d[k] for k in ('n', 'net_ab_pct', 't_month', 'median_net_ab_pct', 'median_ab_pct')},
                    'bear_switch_TAIEX<MA200_to_cash': bool(fail_bear),
                    'regime_inputs': {'7pct_net': g['7pct_2005-01_2015-05']['net_ab_pct'], '10pct_net': g['10pct_2015-06_2021-09']['net_ab_pct']},
                    'cap_disposal_sleeve_20pct': bool(fail_regime)}

# ---- 組合模擬（exec_rebuild Sleeve）----
E50 = L.E50; TW = tx
def mk_events(df, rt, switch=False):
    ev = {}
    for r in df.itertuples():
        if switch and r.bear == 1: continue
        vals = [v for v in (r.v_in, r.v_out) if np.isfinite(v) and v > 0]
        ev.setdefault(r.i0, []).append((r.j, r.k, rt, 0.05 * min(vals) if vals else 0.0))
    return ev
def run(df, capital=1e6, rt=0.008, idle='cash', switch=False, start='2005-01-03', end='2022-12-30'):
    s = Dl.Sleeve(mk_events(df, rt, switch), 10, idle, True)
    t0 = idx.searchsorted(pd.Timestamp(start)); t1 = idx.searchsorted(pd.Timestamp(end), 'right') - 1
    s.deposit(capital, t0 - 1); nav = [capital]; inv = []
    for t in range(t0, t1 + 1):
        s.step(t); e = s.equity(t); nav.append(e); inv.append(s.pos_value(t) / e)
    st = {k: v for k, v in s.stat.items() if k != 'cap_fill_ratio'}
    cf = np.array(s.stat['cap_fill_ratio']); st['share_trades_capacity_bound'] = round(float((cf < 0.999).mean()), 3) if len(cf) else None
    st['mean_invested_share'] = round(float(np.mean(inv)), 3)
    return pd.Series(nav, index=idx[t0 - 1:t1 + 1]), st
def ystats(nav):
    out = {}
    for y in sorted(set(nav.index.year)):
        prev = nav[nav.index.year < y]; cur = nav[nav.index.year == y]
        if len(prev) == 0: continue
        seg = pd.concat([prev.iloc[-1:], cur])
        out[int(y)] = {'ret_pct': round((seg.iloc[-1] / seg.iloc[0] - 1) * 100, 1), 'maxDD_pct': round(float((seg / seg.cummax() - 1).min()) * 100, 1)}
    return out
def pstats(nav, a, b):
    seg = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    pre = nav[nav.index < pd.Timestamp(a)].iloc[-1:]
    seg = pd.concat([pre, seg])
    return {'ret_pct': round((seg.iloc[-1] / seg.iloc[0] - 1) * 100, 1), 'maxDD_pct': round(float((seg / seg.cummax() - 1).min()) * 100, 1)}
def whole(nav, a, b):
    seg = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
    return {'CAGR': round(float((seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1), 4), 'maxDD': round(float((seg / seg.cummax() - 1).min()), 4)}

PORT_EV = pd.concat([EP[EP.date <= '2021-09-30'], ER[ER.date >= '2021-10-01']])
PORT_EV_PROXY = EP
variants = {'base_1M_cash_0.8': dict(), 'base_1M_0050_0.8': dict(idle='0050'), 'stress_1M_cash_1.2': dict(rt=0.012),
            'base_5M_cash_0.8': dict(capital=5e6), 'switch_MA200_1M_cash_0.8': dict(switch=True), 'switch_MA200_1M_0050_0.8': dict(switch=True, idle='0050'),
            'proxy_only_incl_2022_1M_cash_0.8': dict(_proxy=True)}
bench = {'0050': E50.loc['2004-12-31':'2022-12-30'], 'TAIEX_price': TW.loc['2004-12-31':'2022-12-30']}
port = {'bench': {k: {'yearly': ystats(v), 'bear_periods': {p: pstats(v, a, b) for p, (a, b) in BEARS.items()}, 'whole_2005_2021-09': whole(v, '2004-12-31', '2021-09-30')} for k, v in bench.items()}}
navs = {}
for name, kw in variants.items():
    kw = dict(kw); df = PORT_EV_PROXY if kw.pop('_proxy', False) else PORT_EV
    nav, st = run(df, **kw); navs[name] = nav
    port[name] = {'diag': st, 'whole_2005_2021-09': whole(nav, '2004-12-31', '2021-09-30'), 'whole_2005_2022': whole(nav, '2004-12-31', '2022-12-30'),
                  'yearly': ystats(nav), 'bear_periods': {p: pstats(nav, a, b) for p, (a, b) in BEARS.items()}}
    print(name, port[name]['whole_2005_2021-09'], {y: port[name]['yearly'][y] for y in (2008, 2011, 2015, 2018, 2022)}, flush=True)
res['portfolio'] = port
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
print(json.dumps({k: res[k] for k in ('calibration_proxy_rule_2021-10_2026-09', 'exec_stats_proxy_all_years', 'exec_stats_real', 'survivorship', 'bias_calibration', 'decisions', 'bear_minus_bull_main')}, ensure_ascii=False, indent=0))
for k, v in res['groups'].items(): print(k, v)
print(res['yearly_main']); print(res['top10_main'])
for y in (2008, 2011, 2015, 2018, 2022): print(y, {k: port[k]["yearly"].get(y) for k in port if k != "bench"}, {k: port["bench"][k]["yearly"].get(y) for k in port["bench"]})

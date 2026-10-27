#!/usr/bin/env python3
"""正2 50%＋現金 50% 再平衡：回測主程式。先核對 PREREG 雜湊，結果寫入 results.json['results']。"""
import json, hashlib, math, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; AG = H.parent
sys.path.insert(0, str(AG.parent)); from btlib import tstat  # noqa
RJ = json.loads((H / 'results.json').read_text())
assert hashlib.sha256(json.dumps(RJ['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RJ['PREREG_sha256'], 'PREREG changed'
RAW = Path('/home/user/jonatw/finlab-txd/data/raw')
rd = lambda f, c: pd.read_csv(RAW / f, parse_dates=['date']).set_index('date')[c].sort_index()
e50 = rd('etf_0050.csv', 'adj'); l2a = rd('etf_00631L.csv', 'adj'); tx = rd('taiex_twii.csv', 'close')
e50.loc['2026-09-20'] = np.nan                       # 壞列（PREREG data_cleaning）
IDX = e50.index.union(l2a.index); IDX = IDX[IDX >= '2003-07-01']
E = e50.reindex(IDX).ffill(); LA = l2a.reindex(IDX).ffill()
rE = E.pct_change(); rLA = LA.pct_change()
TX = tx.reindex(tx.index.union(IDX)).ffill(); MA200 = TX.rolling(200, min_periods=200).mean().reindex(IDX); TX = TX.reindex(IDX)
dt_days = pd.Series(IDX, index=IDX).diff().dt.days.fillna(1).to_numpy()
# 短利（% 年）
RATE_ON = {2003: 1.07, 2004: 1.03, 2005: 1.32, 2006: 1.55, 2007: 1.98, 2008: 1.85, 2009: 0.11, 2010: 0.19, 2011: 0.34, 2012: 0.43,
           2013: 0.39, 2014: 0.39, 2015: 0.35, 2016: 0.19, 2017: 0.18, 2018: 0.18, 2019: 0.18, 2020: 0.10, 2021: 0.08, 2022: 0.26,
           2023: 0.68, 2024: 0.82, 2025: 0.82, 2026: 0.82}
RATE_CBC = [('2003-06-27', 1.375), ('2004-10-01', 1.625), ('2004-12-31', 1.75), ('2005-03-25', 1.875), ('2005-07-01', 2.0), ('2005-09-16', 2.125),
            ('2005-12-23', 2.25), ('2006-03-31', 2.375), ('2006-06-30', 2.5), ('2006-09-29', 2.625), ('2006-12-29', 2.75), ('2007-03-30', 2.875),
            ('2007-06-22', 3.125), ('2007-09-21', 3.25), ('2007-12-21', 3.375), ('2008-03-28', 3.5), ('2008-06-27', 3.625), ('2008-09-26', 3.5),
            ('2008-10-09', 3.25), ('2008-10-30', 3.0), ('2008-11-10', 2.75), ('2008-12-12', 2.0), ('2009-01-08', 1.5), ('2009-02-19', 1.25),
            ('2010-06-25', 1.375), ('2010-10-01', 1.5), ('2010-12-31', 1.625), ('2011-04-01', 1.75), ('2011-07-01', 1.875), ('2015-09-25', 1.75),
            ('2015-12-18', 1.625), ('2016-03-25', 1.5), ('2016-07-01', 1.375), ('2020-03-20', 1.125), ('2022-03-18', 1.375), ('2022-06-17', 1.5),
            ('2022-09-23', 1.625), ('2022-12-16', 1.75), ('2023-03-24', 1.875), ('2024-03-22', 2.0)]
r_on = pd.Series([RATE_ON[d.year] for d in IDX], index=IDX) / 100
cbc = pd.Series([v for _, v in RATE_CBC], index=pd.DatetimeIndex([d for d, _ in RATE_CBC])); r_cbc = cbc.reindex(cbc.index.union(IDX)).ffill().reindex(IDX) / 100
def synth(rate): return 2 * rE - (0.01 + rate) * dt_days / 365.25
rLS = synth(r_on); rLS_cbc = synth(r_cbc)
L_START = l2a.index[0]                                # 2014-10-31
def hybrid(rs): return rs.where(IDX <= L_START, rLA)
L2_MAIN = hybrid(rLS); L2_CBC = hybrid(rLS_cbc)

# ---------- 合成誤差 ----------
def te_check():
    o = {}
    m = (IDX > L_START)
    for nm, rs in (('synth_overnight', rLS), ('synth_cbc', rLS_cbc)):
        d = (rLA[m] - rs[m]).dropna(); yrs = (IDX[m][-1] - IDX[m][0]).days / 365.25
        ga = (1 + rLA[m].fillna(0)).prod() ** (1 / yrs) - 1; gs = (1 + rs[m].fillna(0)).prod() ** (1 / yrs) - 1
        o[nm] = {'period': f'{IDX[m][0].date()}~{IDX[m][-1].date()}', 'daily_TE_ann': round(float(d.std() * math.sqrt(252)), 4),
                 'corr_daily': round(float(np.corrcoef(rLA[m].fillna(0), rs[m].fillna(0))[0, 1]), 4),
                 'actual_CAGR': round(float(ga), 4), 'synth_CAGR': round(float(gs), 4), 'actual_minus_synth_CAGR': round(float(ga - gs), 4)}
        yy = {}
        for y in sorted(set(IDX[m].year)):
            mm = m & (IDX.year == y)
            yy[int(y)] = round(float(((1 + rLA[mm].fillna(0)).prod() - (1 + rs[mm].fillna(0)).prod()) * 100), 2)
        o[nm]['yearly_actual_minus_synth_pp'] = yy
    # 對 etf_tr_m（2020-08 起，月）
    t = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/etf_tr_m.csv.gz'); t = t[t.code == '00631L']
    t['p'] = pd.PeriodIndex(t.ym, freq='M'); trm = t.set_index('p')['tr'].pct_change().dropna()
    trm = trm[trm.index <= pd.Period('2026-09', 'M')]
    for nm, rs in (('actual_daily_file', rLA), ('synth_overnight', rLS), ('synth_cbc', rLS_cbc)):
        lv = (1 + rs.fillna(0)).cumprod(); mr = lv.groupby(IDX.to_period('M')).last().pct_change().reindex(trm.index)
        d = (mr - trm).dropna()
        o[f'vs_etf_tr_m__{nm}'] = {'months': int(len(d)), 'mean_diff_pct_m': round(float(d.mean() * 100), 3), 'TE_ann_monthly': round(float(d.std() * math.sqrt(12)), 4),
                                    'cum_actual_trm': round(float((1 + trm.loc[d.index]).prod() - 1), 3), 'cum_this': round(float((1 + mr.loc[d.index]).prod() - 1), 3)}
    return o

# ---------- 成本 ----------
FEE = 0.001425 * 0.6
COSTS = {'gross': (0, 0), 'slip0.001': (FEE + 0.001, FEE + 0.001 + 0.001), 'exlib_stocktax': (FEE + 0.001, FEE + 0.003 + 0.001), 'stress': (0.005, 0.005)}

def first_days(months):
    s = pd.Series(IDX, index=IDX); g = s.groupby([IDX.year, IDX.month]).first()
    return set(d for (y, m), d in g.items() if m in months)

FD_JAN = first_days({1}); FD_JANJUL = first_days({1, 7})

def sim(spec, rL, cash_rate, cost, start, end=None, w0=0.5, band=0.10):
    """回傳日 NAV（收盤）與交易次數、周轉。spec: A/B/C/L2_100/0050/MA200。"""
    bc, sc = COSTS[cost]
    i0 = IDX.get_indexer([pd.Timestamp(start)], method='bfill')[0]; i1 = len(IDX) - 1 if end is None else IDX.get_indexer([pd.Timestamp(end)], method='ffill')[0]
    rl = rL.to_numpy(); re_ = rE.to_numpy(); n = i1 - i0 + 1
    nav = np.empty(n); trades = 0; tov = 0.0; blocked = 0
    risky_r = re_ if spec in ('0050', 'MA200') else rl
    wtgt = 1.0 if spec in ('L2_100', '0050', 'MA200') else w0
    x = wtgt * (1 - bc); c = 1 - wtgt                  # 初始買進
    pending = False; ma_on = True
    lockup = rl >= 0.194; lockdn = rl <= -0.194
    if spec in ('0050', 'MA200'): lockup = re_ >= 0.094; lockdn = re_ <= -0.094
    for k in range(n):
        t = i0 + k
        if k > 0:
            x *= 1 + (risky_r[t] if np.isfinite(risky_r[t]) else 0.0)
            c *= (1 + cash_rate) ** (dt_days[t] / 365.25)
            d = IDX[t]
            if spec == 'A' and d in FD_JAN: pending = True
            if spec == 'B' and d in FD_JANJUL: pending = True
            if pending:                                   # 成交（本日收盤）
                V = x + c
                if spec == 'MA200':
                    tgt = V * (1 - bc) if ma_on else 0.0
                else:
                    tgt = 0.5 * V
                dx = tgt - x
                if (dx > 0 and lockup[t]) or (dx < 0 and lockdn[t]):
                    blocked += 1
                else:
                    if spec != 'MA200':
                        for _ in range(5):
                            cst = bc * max(dx, 0) + sc * max(-dx, 0); tgt = 0.5 * (V - cst); dx = tgt - x
                    cst = bc * max(dx, 0) + sc * max(-dx, 0)
                    if spec == 'MA200':
                        if ma_on: x = (c) * (1 - bc); c = 0.0
                        else: c = x * (1 - sc); x = 0.0
                    else:
                        x = tgt; c = V - cst - tgt
                    trades += 1; tov += abs(dx) / V; pending = False
            V = x + c
            if spec == 'C' and not pending and abs(x / V - 0.5) > band: pending = True   # 收盤訊號 → 下一日成交
            if spec == 'MA200' and not pending and d in MA_CHECK:
                on = bool(TX.iloc[t] >= MA200.iloc[t]) if np.isfinite(MA200.iloc[t]) else True
                if on != (x > 0): ma_on = on; pending = True
        nav[k] = x + c
    s = pd.Series(nav, index=IDX[i0:i1 + 1])
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return s, {'trades': trades, 'trades_per_yr': round(trades / yrs, 2), 'turnover_per_yr': round(tov / yrs, 3), 'blocked_days': blocked}

_s = pd.Series(IDX, index=IDX)
MA_CHECK = set(g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([IDX.year, IDX.month]) if (g.dt.day >= 11).any())

# ---------- 統計 ----------
EWN = pd.read_parquet(AG / 'momentum_family' / 'long_nav.parquet')['EW']
ME = pd.DatetimeIndex(_s.groupby([IDX.year, IDX.month]).last().to_numpy())

def mret(nav, last=None):
    pts = ME[(ME >= nav.index[0]) & (ME <= nav.index[-1])]
    if last is not None:
        pts = pts[pts <= last]; pts = pts.append(pd.DatetimeIndex([last])).unique()
    v = nav.reindex(pts, method='ffill'); return (v / v.shift(1) - 1).dropna()

def stats(nav, a, b, comps):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    if len(s) < 20: return None
    prev = nav[nav.index < s.index[0]]
    if len(prev): s = pd.concat([prev.iloc[-1:], s])
    r = s.pct_change().dropna(); yrs = (s.index[-1] - s.index[0]).days / 365.25
    eq = s / s.iloc[0]
    o = {'start': str(s.index[0].date()), 'end': str(s.index[-1].date()), 'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), 'total': round(float(eq.iloc[-1] - 1), 4),
         'vol': round(float(r.std() * math.sqrt(252)), 4), 'maxDD_daily': round(float((eq / eq.cummax() - 1).min()), 4)}
    o['calmar'] = round(o['CAGR'] / abs(o['maxDD_daily']), 2) if o['maxDD_daily'] < 0 else None
    for nm, cn in comps.items():
        last = min(s.index[-1], cn.dropna().index[-1])
        m1 = mret(s, last); m2 = mret(cn.reindex(s.index.union(cn.index)).ffill().reindex(s.index), last)
        ex = (m1 - m2.reindex(m1.index)).dropna()
        o[f't_vs_{nm}'] = tstat(ex); o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 3); o[f'n_m_{nm}'] = int(len(ex))
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

W = RJ['PREREG']['windows']
res = {'tracking_check': te_check()}
print(json.dumps(res['tracking_check'], ensure_ascii=False)[:1500])

START = '2005-01-03'
bench = {}
bench['0050'], bi0 = sim('0050', L2_MAIN, 0.01, 'slip0.001', START)
bench['L2_100'], bi1 = sim('L2_100', L2_MAIN, 0.01, 'slip0.001', START)
bench['MA200_0050'], bi2 = sim('MA200', L2_MAIN, 0.01, 'slip0.001', START)
comps = {'0050': bench['0050'], 'EW': EWN, 'L2_100': bench['L2_100']}
runs = {}
for spec in ('A', 'B', 'C'):
    for cost in ('slip0.001', 'gross', 'exlib_stocktax', 'stress'):
        for cr in (0.01, 0.0):
            for lbl, rL in (('hybrid_on', L2_MAIN), ('hybrid_cbc', L2_CBC)):
                if (cost != 'slip0.001' or cr != 0.01) and lbl != 'hybrid_on': continue
                key = f'{spec}|{cost}|cash{int(cr*100)}|{lbl}'
                runs[key] = sim(spec, rL, cr, cost, START)
for nm, (nav, info) in {'0050': (bench['0050'], bi0), 'L2_100': (bench['L2_100'], bi1), 'MA200_0050': (bench['MA200_0050'], bi2)}.items():
    runs[f'BENCH_{nm}'] = (nav, info)
runs['BENCH_EW_gross'] = (EWN[EWN.index >= START], {})
WIN = {k: v for k, v in W.items() if not k.startswith('long')}
out = {}
for key, (nav, info) in runs.items():
    out[key] = {'info': info, 'windows': {k: stats(nav, a, b, {kk: vv for kk, vv in comps.items() if not key.endswith(kk)}) for k, (a, b) in WIN.items()}, 'yearly_pct': yearly(nav)}
res['runs_from_2005'] = out
# 驗證期重新起跑
fresh = {}
for spec in ('A', 'B', 'C'):
    nav, info = sim(spec, L2_MAIN, 0.01, 'slip0.001', '2019-05-02')
    b0, _ = sim('0050', L2_MAIN, 0.01, 'slip0.001', '2019-05-02')
    fresh[spec] = {'info': info, 'valid_price': stats(nav, '2019-05-01', '2026-10-31', {'0050': b0, 'EW': EWN})}
res['fresh_start_2019-05'] = fresh
# 長歷史（2003-07 起；2014-10 前合成）
lng = {}
b0, _ = sim('0050', L2_MAIN, 0.01, 'slip0.001', '2003-07-02'); b1, _ = sim('L2_100', L2_MAIN, 0.01, 'slip0.001', '2003-07-02')
for spec in ('A', 'B', 'C'):
    nav, info = sim(spec, L2_MAIN, 0.01, 'slip0.001', '2003-07-02')
    lng[spec] = {'info': info, 'stats': stats(nav, '2003-07-01', '2026-10-31', {'0050': b0, 'L2_100': b1})}
lng['0050'] = stats(b0, '2003-07-01', '2026-10-31', {}); lng['L2_100'] = stats(b1, '2003-07-01', '2026-10-31', {'0050': b0})
# 全合成（不用實際 00631L）2005 起
for spec in ('A',):
    nav, info = sim(spec, rLS, 0.01, 'slip0.001', START)
    lng[f'{spec}_pure_synth_2005'] = {k: stats(nav, a, b, {'0050': bench['0050']}) for k, (a, b) in WIN.items() if k in ('select_price', 'valid_price', 'full_2005')}
res['long_history'] = lng
# 空頭段（0050 高點→低點）
EP = {'GFC_2007-10-29_2008-11-20': ('2007-10-29', '2008-11-20'), 'Euro_2011-05-02_2011-12-19': ('2011-05-02', '2011-12-19'),
      'COVID_2020-01-20_2020-03-19': ('2020-01-20', '2020-03-19'), 'Bear_2022-01-04_2022-10-25': ('2022-01-04', '2022-10-25'),
      'Tariff_2025-01-22_2025-04-09': ('2025-01-22', '2025-04-09')}
ep = {}
for nm, (a, b) in EP.items():
    row = {}
    for key in ('A|slip0.001|cash1|hybrid_on', 'B|slip0.001|cash1|hybrid_on', 'C|slip0.001|cash1|hybrid_on', 'BENCH_0050', 'BENCH_L2_100', 'BENCH_MA200_0050', 'BENCH_EW_gross'):
        nav = runs[key][0]; s = nav[(nav.index >= a) & (nav.index <= b)]
        row[key.split('|')[0]] = round(float(s.iloc[-1] / s.iloc[0] - 1), 4)
    ep[nm] = row
res['bear_episodes'] = ep
RJ['results'] = res
navdf = pd.DataFrame({k: v[0] for k, v in runs.items() if k.endswith('hybrid_on') and '|slip0.001|cash1|' in k or k.startswith('BENCH')})
navdf.to_parquet(H / 'nav.parquet')
(H / 'results.json').write_text(json.dumps(RJ, ensure_ascii=False, indent=1, default=float))
# 摘要
for key in ['A|slip0.001|cash1|hybrid_on', 'B|slip0.001|cash1|hybrid_on', 'C|slip0.001|cash1|hybrid_on', 'A|stress|cash1|hybrid_on', 'A|slip0.001|cash0|hybrid_on', 'A|slip0.001|cash1|hybrid_cbc', 'BENCH_0050', 'BENCH_L2_100', 'BENCH_MA200_0050', 'BENCH_EW_gross']:
    w = out[key]['windows']
    print(key, out[key]['info'])
    for k in ('select_price', 'valid_price', 'select_2021', 'valid_2024', 'y2024_2025', 'bear_2008', 'bear_2022'):
        x = w[k]; print('   ', k, {kk: x.get(kk) for kk in ('CAGR', 'maxDD_daily', 'vol', 't_vs_0050', 't_vs_EW', 't_vs_L2_100')})
print(json.dumps(ep, indent=0)); print(json.dumps(lng, ensure_ascii=False)[:3000]); print(json.dumps(fresh)[:2000])

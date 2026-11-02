#!/usr/bin/env python3
"""反方審查（資料與統計）：o_v7_listing_exec_engine。只讀研究員檔案，不寫回任何研究員檔案；輸出印到 stdout。
檢查：
 R1 可執行表「贏」的統計支撐：B10 一般成本 j 中位數各格，逐 j 超額拆 SEL(≤2018)／2019 起，算 j 平均月超額序列 t（含 NW）、年份集中度。
 R2 官方日 vs j 分布、AB_V7 零衝擊逐年超額集中度、拿掉最佳年、Newey-West t。
 R3 上市矩陣：L1 vs 強證據（L5）在替換窗內的差異股日，是否碰到 V7 選股名單（前視／非 PIT 市場別風險）。
 R4 j 序號與 t 的單調關係（換股日相依）。
"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
H = Path('/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine')
E50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))) if len(x) > 2 else float('nan')

def nw_t(x, L=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); m = x.mean(); e = x - m
    g0 = (e @ e) / n; s = g0
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return float(m / math.sqrt(s / n))

def sample(nav, days):
    s = nav.dropna(); s = s[~s.index.duplicated(keep='last')]
    v = s.reindex(s.index.union(days)).ffill().reindex(days); v[days < s.index[0]] = np.nan
    return v

def winx(nav, sd):
    D = pd.DatetimeIndex(sd); v = sample(nav, D).to_numpy(); b = sample(E50, D).to_numpy()
    return pd.Series(v[1:] / v[:-1] - b[1:] / b[:-1], index=D[:-1]).dropna()

def cagr(nav, a, b):
    s = nav.dropna(); s = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1

R = json.load(open(H / 'results.json'))
PJ = pd.read_csv(H / 'data/plans_j7.csv', dtype={'picks': str})
JEND = json.load(open(H / 'data/jscan_end_days.json'))
NE = pd.read_parquet(H / 'navs_exec7.parquet')
grid = {}
for (st, j), g in PJ[PJ.seed == 0].groupby(['strat', 'j']):
    grid[(st, j)] = list(pd.to_datetime(g.signal)) + [pd.Timestamp(JEND[f'{st}|J{j}'])]

print('=== R1 可執行表（B10 一般成本）逐 j 超額拆期 ===')
out = {}
for tier in ('30萬', '100萬', '300萬', '1000萬'):
    for ver in ('FIX', 'GROW'):
        for model in ('K5T', 'TICK'):
            rows = []; xs_all = []
            for j in range(13):
                col = f'B10|J{j}|{model}|normal|{tier}|{ver}'
                nav = NE[col].dropna(); sd = grid[('AB_V7_B10', j)]
                x = winx(nav, sd); xs_all.append(x.reset_index(drop=True))
                a, b = nav.index[0], nav.index[-1]
                e50 = E50.reindex(nav.index).ffill()
                pre = (cagr(nav, a, '2018-12-31') - cagr(e50, a, '2018-12-31')) * 100
                post = (cagr(nav, '2019-01-01', b) - cagr(e50, '2019-01-01', b)) * 100
                full = (cagr(nav, a, b) - cagr(e50, a, b)) * 100
                x19 = x[x.index >= '2019-01-01']; x0 = x[x.index < '2019-01-01']
                x_ex0609 = x[x.index >= '2010-01-01']
                rows.append(dict(j=j, full=full, pre=pre, post=post, t=tstat(x), t_pre=tstat(x0), t_post=tstat(x19), t_from2010=tstat(x_ex0609),
                                 mean_post_pp=x19.mean() * 100))
            df = pd.DataFrame(rows)
            med = df.median()
            out[f'{tier}|{ver}|{model}'] = med.to_dict()
            print(f'{tier:>5} {ver:4} {model:4}  全期超額 {med.full:+.2f}pp t {med.t:.2f} | ≤2018 超額 {med.pre:+.2f}pp t {med.t_pre:.2f} | 2019起 超額 {med.post:+.2f}pp t {med.t_post:.2f} | 2010起 t {med.t_from2010:.2f} | 2019 起負超額 j 數 {(df.post < 0).sum()}/13')

print('\n=== R2 AB_V7 官方日零衝擊 ===')
N7 = pd.read_parquet(H / 'navs_v7.parquet')
P = pd.read_csv(H / 'plan_AB_V7_s0.csv')
sd = list(pd.to_datetime(P.signal)) + [pd.Timestamp('2026-10-12')]
for col in ('AB_V7|normal', 'AB_V7_B10|normal'):
    x = winx(N7[col], sd)
    yr = x.groupby(x.index.year).sum() * 100
    lx = np.log1p(x)
    print(col, f'n={len(x)} t={tstat(x):.2f} NW3 t={nw_t(x,3):.2f} NW6 t={nw_t(x,6):.2f} AR1={x.autocorr(1):.3f}')
    print('  逐年超額和(pp)前 5：', yr.sort_values(ascending=False).head(5).round(1).to_dict())
    top3 = yr.sort_values(ascending=False).head(3).index
    print(f'  總和 {yr.sum():.1f}pp；前 3 年占 {yr[top3].sum() / yr.sum() * 100:.0f}%')
    for drop in (1, 2, 3):
        dy = yr.sort_values(ascending=False).head(drop).index
        print(f'  拿掉最佳 {drop} 年 {list(dy)} t={tstat(x[~x.index.year.isin(dy)]):.2f}')
    print(f'  2006-2009 以外 t={tstat(x[x.index >= "2010-01-01"]):.2f}; 2010-2018 t={tstat(x[(x.index >= "2010-01-01") & (x.index < "2019-01-01")]):.2f}; 2019起 t={tstat(x[x.index >= "2019-01-01"]):.2f}')

# 官方日 vs j
NJ = pd.read_parquet(H / 'navs_jscan_v7.parquet')
for base in ('AB_V7|normal', 'AB_V7_B10|normal'):
    ts = []
    st = 'AB_V7' if base.startswith('AB_V7|') else 'AB_V7_B10'
    for j in range(13):
        x = winx(NJ[f'{base}|J{j}'], grid[(st, j)]); ts.append(tstat(x))
    ts = np.array(ts)
    print(f'{base} j t: {np.round(ts, 2).tolist()} 中位數 {np.median(ts):.2f}；與 j 的 Spearman ρ={pd.Series(ts).corr(pd.Series(range(13)), method="spearman"):.2f}')
    # 官方日月內日
    print('  官方日訊號日的「月內第幾個交易日」分布：', pd.Series(pd.to_datetime(P.signal).dt.day).describe()[['min', '50%', 'max']].to_dict())
    for j in (0, 6, 12):
        g = PJ[(PJ.strat == st) & (PJ.j == j) & (PJ.seed == 0)]
        print(f'  j={j} 訊號日日期中位數 {pd.to_datetime(g.signal).dt.day.median()}')

print('\n=== R3 上市矩陣 L1 vs 強證據（替換窗）===')
LM = pd.read_parquet(H / 'data/listed_twse.parquet'); LS = pd.read_parquet(H / 'data/listed_twse_strong.parquet')
W = (LM.index >= '2024-01-02') & (LM.index <= '2025-09-23')
cols = [c for c in LM.columns if c in LS.columns]
A = LM.loc[W, cols]; B = LS.loc[W, cols].reindex(A.index).fillna(False)
cl = pd.read_parquet(H / 'data/close_raw.parquet')
only_L1 = (A & ~B); n = int(only_L1.to_numpy().sum())
has_px = only_L1 & cl.reindex(index=A.index, columns=cols).notna()
codes = only_L1.any()[only_L1.any()].index.tolist()
codes_px = has_px.any()[has_px.any()].index.tolist()
print(f'L1 上市但強證據不上市：{n} 股日、{len(codes)} 檔；其中 V7 面板有價 {int(has_px.to_numpy().sum())} 股日、{len(codes_px)} 檔：{codes_px[:40]}')
pv = pd.read_csv(H / 'plan_V7.csv', dtype={'picks': str}); pv['signal'] = pd.to_datetime(pv.signal)
hit = []
for _, r in pv[(pv.signal >= '2024-01-02') & (pv.signal <= '2025-09-23')].iterrows():
    for c in r.picks.split(','):
        if c in codes_px and bool(has_px.loc[:r.signal, c].any()):
            hit.append((str(r.signal.date()), c))
print('V7 官方日名單碰到這些股：', hit[:30])
hitj = set()
for _, r in PJ[(pd.to_datetime(PJ.signal) >= '2024-01-02') & (pd.to_datetime(PJ.signal) <= '2025-09-23')].iterrows():
    for c in r.picks.split(','):
        if c in codes_px: hitj.add(c)
print('j 名單（全部 seed）碰到這些股：', sorted(hitj))
B2 = LS.loc[W, cols].reindex(A.index).fillna(False)
only_S = (B2 & ~A); print(f'強證據上市但 L1 不上市：{int(only_S.to_numpy().sum())} 股日')

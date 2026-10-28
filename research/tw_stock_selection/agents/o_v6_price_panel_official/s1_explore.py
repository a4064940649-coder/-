#!/usr/bin/env python3
"""步驟 1：探索（只看資料，不看任何策略報酬）→ explore.json
(a) 來源盤點；(b) 重疊期官方對官方逐筆比對；(c) 官方對 src 3（Yahoo 5 分 K 近似）的誤差分布與分組；
(d) 池內股日覆蓋率（V4 換股日前 300 股池＋曾持有）；(e) 93 檔缺值與 2024-04～06 股池漏檔。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v6common as C, srcload as SL

P = C.load_panel()
IDX4, COLS = P['close_raw'].index, P['close_raw'].columns
CSET = set(COLS)
cal = C.taiex_cal()
S = {k: SL.get(k) for k in SL.LOADERS}
for k in S: S[k] = S[k][S[k].code.isin(CSET)]
out = {}

# ---------------- (a) 來源盤點 ----------------
UNITS = {'hschi': 'FinMind TaiwanStockPrice：Trading_Volume＝股、Trading_money＝元、spread＝漲跌價差（除權息日 FinMind 記 0）；原始價、不還原',
         'hawan': '同 FinMind（119 檔）', 'henry': '同 FinMind（含 PER 等附加欄）',
         'tasi': 'kbar 逐日全市場：vol＝張（本程式 ×1000 換成股，下面驗證）；沒有成交值、沒有漲跌價差；原始價',
         'bacardi': '上市＋上櫃逐日：volume＝股、turnover＝元、change＝漲跌價差；原始價（只取 market=TWSE）',
         'history': '官方日 K：volume＝股；沒有成交值、沒有漲跌價差（面板 src 4 的 value＝volume×close）',
         'leochen': 'leochen1/stock_v data/history（證交所 STOCK_DAY 格式，994 檔上市股，2010～2025-08-19）：成交量＝股；沒有成交值'}
inv = {}
for k, d in S.items():
    inv[k] = {'range': [str(d.date.min().date()), str(d.date.max().date())], 'n_codes_in_panel': int(d.code.nunique()),
              'n_rows_in_panel_codes': int(len(d)), 'units': UNITS[k],
              'n_rows_2024_01_02_to_02_14': int(((d.date >= C.GAP_A) & (d.date <= C.GAP_B)).sum()),
              'n_rows_in_V6_window': int(((d.date >= C.W_A) & (d.date <= C.W_B)).sum())}
out['inventory'] = inv

# ---------------- (b) 官方對官方 ----------------
def cmp(a, b, a0, b0, fields=('close', 'open', 'high', 'low', 'vol', 'value')):
    x = S[a][(S[a].date >= a0) & (S[a].date <= b0)]; y = S[b][(S[b].date >= a0) & (S[b].date <= b0)]
    m = x.merge(y, on=['date', 'code'], suffixes=('_a', '_b'))
    o = {'pair': f'{a} vs {b}', 'period': [str(pd.Timestamp(a0).date()), str(pd.Timestamp(b0).date())], 'n': int(len(m))}
    for f in fields:
        fa, fb = m[f + '_a'], m[f + '_b']
        ok = fa.notna() & fb.notna()
        if ok.sum() == 0: continue
        if f in ('vol', 'value'):
            r = (fa[ok] / fb[ok])
            o[f] = {'n': int(ok.sum()), 'exact': float((np.abs(fa[ok] - fb[ok]) < 0.5).mean()), 'ratio_median': float(r.median()),
                    'within_1pct': float((np.abs(r - 1) <= 0.01).mean()), 'within_2pct': float((np.abs(r - 1) <= 0.02).mean())}
        else:
            o[f] = {'n': int(ok.sum()), 'exact': float((np.abs(fa[ok] - fb[ok]) < 1e-6).mean()),
                    'within_0p1pct': float((np.abs(fa[ok] / fb[ok] - 1) <= 0.001).mean())}
    # 不一致的例子（收盤）
    bad = m[np.abs(m.close_a - m.close_b) > 1e-6]
    o['close_mismatch_examples'] = bad[['date', 'code', 'close_a', 'close_b']].head(8).astype(str).values.tolist()
    o['close_mismatch_by_month'] = {str(k): int(v) for k, v in bad.groupby(bad.date.dt.to_period('M')).size().items()}
    return o

pairs = [('henry', 'hschi', '2020-01-01', '2022-12-31'),
         ('hschi', 'tasi', '2023-10-01', '2025-05-28'),
         ('hschi', 'bacardi', '2025-04-01', '2025-05-28'),
         ('tasi', 'bacardi', '2025-04-01', '2025-09-24'),
         ('hawan', 'hschi', '2020-01-01', '2025-05-28'),
         ('hawan', 'bacardi', '2025-04-01', '2025-12-31'),
         ('bacardi', 'history', '2025-09-25', '2026-10-27'),
         ('tasi', 'history', '2025-09-25', '2026-05-15'),
         ('leochen', 'hschi', '2020-01-01', '2025-05-28'), ('leochen', 'bacardi', '2025-04-01', '2025-08-19'), ('leochen', 'tasi', '2023-10-01', '2025-08-19')]
out['official_vs_official'] = [cmp(*p) for p in pairs]

# 面板官方（src 2，2020～2023；src 4，2025-09-24 起）對 FinMind／bacardi：確認接縫兩側同口徑
def panel_long(a0, b0, srcs):
    sl = slice(pd.Timestamp(a0), pd.Timestamp(b0))
    d = {f: P[f].loc[sl].stack() for f in ('close_raw', 'open', 'high', 'value', 'src')}
    df = pd.DataFrame(d).reset_index(); df.columns = ['date', 'code', 'close', 'open', 'high', 'value', 'src']
    return df[df.src.isin(srcs)]
def cmp_panel(name, a0, b0, srcs):
    pl = panel_long(a0, b0, srcs)
    y = S[name][(S[name].date >= a0) & (S[name].date <= b0)]
    m = pl.merge(y, on=['date', 'code'], suffixes=('_p', '_s'))
    o = {'pair': f'panel src{srcs} vs {name}', 'period': [a0, b0], 'n': int(len(m))}
    for f in ('close', 'open', 'high', 'value'):
        a, b = m[f + '_p'], m[f + '_s']; ok = a.notna() & b.notna()
        if ok.sum() == 0: continue
        o[f] = {'exact': float((np.abs(a[ok] - b[ok]) < (0.5 if f == 'value' else 1e-4)).mean()),
                'within_0p1pct': float((np.abs(a[ok] / b[ok] - 1) <= 0.001).mean()), 'n': int(ok.sum())}
    return o
out['panel_official_vs_sources'] = [cmp_panel('hschi', '2020-01-01', '2023-12-29', [2]), cmp_panel('bacardi', '2025-09-24', '2026-10-16', [4])]

# ---------------- (c) 官方對 src 3 近似 ----------------
# 官方組合：hschi（到 2025-05-28）→ bacardi → tasi（只做比對用）
def official_combo(a0, b0):
    parts = []
    taken = set()
    for k in ('hschi', 'bacardi', 'tasi'):
        d = S[k][(S[k].date >= a0) & (S[k].date <= b0)]
        key = list(zip(d.date, d.code))
        keep = np.array([kk not in taken for kk in key])
        d = d[keep].assign(srcname=k)
        taken |= set(zip(d.date, d.code))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)
OF = official_combo(pd.Timestamp('2023-10-01'), pd.Timestamp('2025-09-24'))
OFw = {f: OF.pivot(index='date', columns='code', values=f).reindex(columns=COLS) for f in ('open', 'high', 'low', 'close', 'vol', 'value')}
sl3 = slice(pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23'))
src3 = (P['src'].loc[sl3] == 3)
pc, po, ph = P['close_raw'].loc[sl3], P['open'].loc[sl3], P['high'].loc[sl3]
oc, oo, oh = OFw['close'].reindex(pc.index), OFw['open'].reindex(pc.index), OFw['high'].reindex(pc.index)
m = src3 & pc.notna() & oc.notna()
err_c = (pc / oc - 1).where(m); err_o = (po / oo - 1).where(m & po.notna() & oo.notna()); err_h = (ph / oh - 1).where(m & ph.notna() & oh.notna())
def dist(e):
    v = e.stack().dropna().to_numpy(); a = np.abs(v)
    return {'n': int(len(v)), 'abs_median_pct': float(np.median(a) * 100), 'abs_mean_pct': float(a.mean() * 100), 'p90_abs_pct': float(np.quantile(a, 0.9) * 100),
            'p99_abs_pct': float(np.quantile(a, 0.99) * 100), 'exact_share': float((a < 1e-6).mean()), 'mean_signed_pct': float(v.mean() * 100)}
# 日報酬相關（同一股，連續兩日都有值）
o_ret = (oc / oc.shift(1) - 1); p_ret = (pc / pc.shift(1) - 1)
mm = m & m.shift(1, fill_value=False)
rr = pd.DataFrame({'o': o_ret.where(mm).stack(), 'p': p_ret.where(mm).stack()}).dropna()
rr = rr[(rr.o.abs() < 0.11) & (rr.p.abs() < 0.11)]          # 去掉除權息／公司行動日（只看價格近似誤差）
# 分組：收盤漲跌停（官方）、52 週比 ≥ 0.95（V4 面板 adj）、V60 十分位
prevc = OFw['close'].ffill().shift(1).reindex(pc.index)
def tick(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
lu = np.floor(prevc * 1.1 / tick(prevc * 1.1) + 1e-9) * tick(prevc * 1.1)
ld = np.ceil(prevc * 0.9 / tick(prevc * 0.9) - 1e-9) * tick(prevc * 0.9)
lim_up = (oc >= lu - 1e-6); lim_dn = (oc <= ld + 1e-6)
near_lim = (oc / prevc - 1).abs() >= 0.09
A4 = P['adj']; nobs = P['close_raw'].notna().sum(axis=1); good = nobs >= 300
hi = A4[good].rolling(252, min_periods=200).max().reindex(A4.index).ffill()
s52 = (A4 / hi).loc[sl3]
v60 = P['value'][good].rolling(60, min_periods=50).mean().reindex(A4.index).loc[sl3]
rk = v60.rank(axis=1, ascending=False)
grp = {'all': m, 'close_limit_up': m & lim_up, 'close_limit_down': m & lim_dn, 'close_near_limit(|r|>=9%)': m & near_lim,
       'not_near_limit': m & ~near_lim, 's52_ge_0.95': m & (s52 >= 0.95), 's52_lt_0.95': m & (s52 < 0.95)}
for lo in range(0, 300, 50):
    grp[f'V60_rank_{lo + 1}_{lo + 50}'] = m & (rk > lo) & (rk <= lo + 50)
grp['V60_rank_gt300'] = m & (rk > 300)
out['approx_error_src3_vs_official'] = {
    'note': '誤差＝src 3 ÷ 官方 − 1；官方＝hschi（到 2025-05-28）→ bacardi → tasi；分組依官方收盤與 V4 面板訊號（52 週比 lenient、V60 排名）',
    'close': {g: dist(err_c.where(mk)) for g, mk in grp.items()},
    'open': {g: dist(err_o.where(mk)) for g, mk in grp.items() if g in ('all', 'close_limit_up', 's52_ge_0.95', 's52_lt_0.95', 'V60_rank_1_50', 'V60_rank_251_300')},
    'high': dist(err_h),
    'daily_ret_corr': float(rr.o.corr(rr.p)), 'daily_ret_n': int(len(rr)),
    'daily_ret_abs_diff_median_pct': float((rr.o - rr.p).abs().median() * 100),
    'by_month_close_abs_median_pct': {str(k): float(v) for k, v in (err_c.abs().stack().groupby(level=0).median().groupby(lambda d: d.to_period('M')).median() * 100).items()},
    'close_limit_up_official_but_not_src3': int((m & lim_up & ~(pc >= lu - 1e-6)).sum().sum()),
    'close_limit_up_src3_but_not_official': int((m & ~lim_up & (pc >= lu - 1e-6)).sum().sum()),
    'open_limit_up_official': int((m & (oo >= lu - 1e-6)).sum().sum()),
    'open_limit_up_src3': int((m & (po >= lu - 1e-6)).sum().sum()),
    'open_limit_up_disagree': int((m & ((oo >= lu - 1e-6) != (po >= lu - 1e-6))).sum().sum()),
}

# ---------------- (d) 池內股日覆蓋率 ----------------
plan = pd.read_csv(C.V4DIR / 'plan_V4.csv', dtype=str)
sig = pd.to_datetime(plan.signal); held = {pd.Timestamp(s): p.split(',') for s, p in zip(plan.signal, plan.picks)}
v60s = P['value'][good].rolling(60, min_periods=60).mean().reindex(A4.index)
def pool_at(d):
    t = A4.index.get_indexer([d])[0]
    while t >= 0 and not np.isfinite(v60s.iloc[t]).any(): t -= 1
    v = v60s.iloc[t]; px = P['close_raw'].iloc[t]
    ok = (v >= 1e7) & (px >= 10)
    return set(v[ok].sort_values(ascending=False).index[:300])
cal6 = cal[(cal >= C.W_A) & (cal <= C.W_B)]
sigs = [d for d in sig if pd.Timestamp('2023-12-01') <= d <= C.W_B]
popu = []
for i, d in enumerate(sigs):
    nxt = sigs[i + 1] if i + 1 < len(sigs) else C.W_B + pd.Timedelta(days=1)
    days = cal6[(cal6 >= max(d, C.W_A)) & (cal6 < nxt)]
    names = pool_at(d) | set(held.get(d, []))
    for dd in days:
        for c in names: popu.append((dd, c))
POP = pd.DataFrame(popu, columns=['date', 'code']).drop_duplicates()
have = {}
for k in ('hschi', 'bacardi', 'tasi', 'hawan', 'history', 'leochen'):
    have[k] = set(zip(S[k].date, S[k].code))
def cov(sub, srcs):
    key = list(zip(sub.date, sub.code))
    return float(np.mean([any(kk in have[s] for s in srcs) for kk in key])) if len(key) else float('nan')
segs = {'2024-01-02~02-14': (C.GAP_A, C.GAP_B), '2024-02-15~2025-05-28': (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-05-28')),
        '2025-05-29~09-23': (pd.Timestamp('2025-05-29'), C.W_B), 'ALL': (C.W_A, C.W_B)}
covd = {}
for s, (a, b) in segs.items():
    sub = POP[(POP.date >= a) & (POP.date <= b)]
    covd[s] = {'n_stock_days': int(len(sub)), 'hschi': cov(sub, ['hschi']), 'tasi': cov(sub, ['tasi']), 'bacardi': cov(sub, ['bacardi']),
               'hschi+tasi': cov(sub, ['hschi', 'tasi']), 'hschi+bacardi+tasi': cov(sub, ['hschi', 'bacardi', 'tasi']),
               'hschi+tasi+bacardi+leochen': cov(sub, ['hschi', 'tasi', 'bacardi', 'leochen']),
               'all_official(+hawan)': cov(sub, ['hschi', 'bacardi', 'tasi', 'leochen', 'hawan'])}
out['pool_coverage'] = {'population': 'V4 換股日前 300 股池（strict V60、價 ≥ 10）＋曾持有，乘上持有窗內 TAIEX 交易日；2024-01 用 2023-12-11 的池', 'by_segment': covd}
miss = POP[[not any(kk in have[s] for s in ('hschi', 'bacardi', 'tasi', 'leochen')) for kk in zip(POP.date, POP.code)]]
out['pool_coverage']['missing_by_code_top'] = miss.code.value_counts().head(20).to_dict()
out['pool_coverage']['missing_by_date_top'] = {str(k.date()): int(v) for k, v in miss.date.value_counts().head(20).items()}
out['pool_coverage']['n_missing'] = int(len(miss))
miss.to_csv(C.HERE / 'explore_missing_pool_stockdays.csv', index=False)

# ---------------- (e) 93 檔缺值與股池漏檔 ----------------
w93 = slice(pd.Timestamp('2024-02-15'), pd.Timestamp('2024-03-15'))
c93 = P['close_raw'].loc[w93]
hasv = c93.notna().sum()
was_traded_before = P['close_raw'].loc['2023-12-01':'2023-12-29'].notna().sum() > 0
mis = [c for c in COLS if was_traded_before[c] and hasv[c] < len(c93) * 0.5]
off93 = OFw['close'].loc[w93].reindex(columns=mis)
out['missing_93'] = {'n_stocks_missing_ge_half_window_V4': len(mis), 'codes': mis[:120],
                     'official_coverage_of_their_stock_days': float(off93.notna().to_numpy().mean()),
                     'in_V4_top300_pool_2023_12_11': int(len(set(mis) & pool_at(pd.Timestamp('2023-12-11'))))}
C.jdump(out, C.HERE / 'explore.json')
print('coverage', covd)
print('93', out['missing_93']['n_stocks_missing_ge_half_window_V4'], out['missing_93']['official_coverage_of_their_stock_days'], out['missing_93']['in_V4_top300_pool_2023_12_11'])
for o in out['official_vs_official']: print(o['pair'], o['n'], {f: o[f]['exact'] for f in ('close', 'open', 'vol', 'value') if f in o})
for o in out['panel_official_vs_sources']: print(o)
a = out['approx_error_src3_vs_official']
print({g: round(v['abs_median_pct'], 3) for g, v in a['close'].items()}, a['daily_ret_corr'])
print({g: round(v['abs_median_pct'], 3) for g, v in a['open'].items()})

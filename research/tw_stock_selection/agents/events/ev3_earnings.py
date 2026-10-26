#!/usr/bin/env python3
"""檢定 3：財報季效應。事前固定規則：
法定截止日 3/31（年報）、5/15（Q1）、8/14（Q2）、11/14（Q3）；D＝截止日當天或之前最後一個交易日。
  pre 窗＝D 之前 20 個交易日（D-19 … D，共 20 日報酬）；post 窗＝D 之後 20 個交易日（D+1 … D+20）；其餘為 other。
  （2013 年 IFRS 前法定截止日不同，市場效應另報 2013 起子樣本。）
A. 月營收年增率（2021 起）：
  * 訊號：當日收盤時「已公告」的最新月營收年增率；月份 m 的營收視為在 m+1 月 11 日（含）之後第一個交易日收盤可用。
  * 每日以前一日收盤資訊分五組（流動性合格上市股），等權；記錄 Q5（高年增）− 等權池、Q5 − Q1 的每日報酬。
  * 比較 pre／post／other 三種日子的平均日超額（bp）與 t 值；pre−other、post−other 差異 t（Welch）。
  * 可交易版本：只在 pre（或 post）窗持有 Q5，其餘時間持 0050；每窗進出扣來回 0.5% ＋ 窗內月營收換股周轉成本。
  * 逐窗表：每個截止日 pre／post 窗 Q5 對等權超額。
  挑選期 2021～2023、驗證期 2024～2026。
B. 整體市場（0050 含息、加權指數、流動性合格等權）：pre 窗 vs 其他日子的日均報酬、每窗 20 日報酬 vs 無條件 20 日平均；
  只在 pre 窗持有 0050（其餘現金）vs 只在非 pre 窗持有。挑選期 2005-01～2019-04、驗證期 2019-05～2026-10。
"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evlib as L

D = L.load(); idx = D['idx']; R = D['ret']; liq = D['liq']
# 截止日窗口
dl = []
for y in range(2004, 2027):
    for m, d in [(3, 31), (5, 15), (8, 14), (11, 14)]:
        t = pd.Timestamp(y, m, d)
        if t > idx[-1] or t < idx[0]: continue
        dl.append(idx.searchsorted(t, 'right') - 1)
cls = pd.Series('other', index=idx); win_id = pd.Series('', index=idx)
for i in dl:
    lab = str(idx[i].date())
    cls.iloc[max(i - 19, 0):i + 1] = 'pre'; win_id.iloc[max(i - 19, 0):i + 1] = 'pre_' + lab
    cls.iloc[i + 1:i + 21] = 'post'; win_id.iloc[i + 1:i + 21] = 'post_' + lab

def welch(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return round(float((a.mean() - b.mean()) / math.sqrt(a.var() / len(a) + b.var() / len(b))), 2)

def day_stats(x, c):
    o = {}
    for k in ('pre', 'post', 'other'):
        v = x[c == k].dropna(); o[k] = {'days': int(len(v)), 'mean_bp': round(float(v.mean() * 1e4), 2), 't': L.tstat(v)}
    o['t_pre_minus_other'] = welch(x[c == 'pre'], x[c == 'other']); o['t_post_minus_other'] = welch(x[c == 'post'], x[c == 'other'])
    return o

res = {}
# ---------- A. 月營收 ----------
rv = pd.read_csv(L.X / 'rev_5y.csv.gz', dtype={'code': str}); rv = rv[rv['code'].isin(R.columns)].dropna(subset=['yoy'])
per = pd.PeriodIndex(rv['ym'], freq='M')
avail = pd.to_datetime([f'{p.year + (p.month == 12)}-{p.month % 12 + 1:02d}-11' for p in per])
rv['avail'] = idx[np.minimum(idx.searchsorted(avail), len(idx) - 1)]
rv = rv.sort_values(['avail', 'ym']).drop_duplicates(['avail', 'code'], keep='last')
Y = rv.pivot(index='avail', columns='code', values='yoy').reindex(idx).ffill(limit=60).reindex(columns=R.columns)
sig = Y.where(liq).shift(1)                        # t 日報酬用 t-1 收盤已知資訊
rk = sig.rank(axis=1, pct=True)
ok = sig.notna() & R.notna()
nOK = ok.sum(axis=1)
q5 = (rk > 0.8) & ok; q1 = (rk <= 0.2) & ok
pool = R.where(ok).mean(axis=1); r5 = R.where(q5).mean(axis=1); r1 = R.where(q1).mean(axis=1)
A0 = pd.Timestamp('2021-02-01')
m = (idx >= A0) & (nOK >= 100)
ex5 = (r5 - pool)[m]; sp = (r5 - r1)[m]; c = cls[m]
res['A_revenue'] = {'note': '每日超額 bp；pre=截止日前 20 交易日；Q5=已知最新月營收年增率最高 1/5', 'n_stocks_median': int(nOK[m].median())}
for pname, (a, b) in {'all': ('2021-01-01', '2026-12-31'), 'sel_2021_2023': ('2021-01-01', '2023-12-31'), 'val_2024_2026': ('2024-01-01', '2026-12-31')}.items():
    mm = (ex5.index >= a) & (ex5.index <= b)
    res['A_revenue'][pname] = {'Q5_minus_pool': day_stats(ex5[mm], c[mm]), 'Q5_minus_Q1': day_stats(sp[mm], c[mm])}
# 逐窗
wt = pd.DataFrame({'ex5': ex5, 'sp': sp, 'w': win_id[m]})
pw = wt[wt['w'] != ''].groupby('w').agg(ex5=('ex5', lambda x: (1 + x).prod() - 1), sp=('sp', lambda x: (1 + x).prod() - 1), days=('ex5', 'size'))
pw = pw[pw['days'] >= 15]
res['A_revenue']['per_window_pct'] = {k: {'Q5_ex_pool': round(v['ex5'] * 100, 2), 'Q5_minus_Q1': round(v['sp'] * 100, 2)} for k, v in pw.iterrows()}
for side in ('pre', 'post'):
    s = pw[pw.index.str.startswith(side)]
    for pname, (a, b) in {'sel_2021_2023': ('2021', '2023-12-31'), 'val_2024_2026': ('2024', '2026-12-31')}.items():
        ss = s[(s.index.str[len(side) + 1:] >= a) & (s.index.str[len(side) + 1:] <= b)]
        res['A_revenue'][f'{side}_windows_{pname}'] = {'n': int(len(ss)), 'mean_Q5_ex_pool_pct': round(float(ss['ex5'].mean() * 100), 2),
            't': L.tstat(ss['ex5']), 'hit': round(float((ss['ex5'] > 0).mean()), 2), 'mean_Q5_minus_Q1_pct': round(float(ss['sp'].mean() * 100), 2)}
# 參照：同樣 20 日、其他時段（other）平均 20 日 Q5 超額（以每日平均 × 20 近似）
# 可交易版本：pre（或 post）窗持 Q5，其他持 0050
memb = q5.astype(np.int8)
chg = (memb.diff().abs().sum(axis=1) / memb.sum(axis=1).replace(0, np.nan)).fillna(0) / 2   # 單邊換手比例
for side in ('pre', 'post', 'always'):
    inwin = (cls == side) if side != 'always' else pd.Series(True, index=idx)
    first_day = inwin & ~inwin.shift(1, fill_value=False); last_day = inwin & ~inwin.shift(-1, fill_value=False)
    r = pd.Series(np.where(inwin, r5, D['e50']), index=idx)
    cost = np.where(first_day, L.COST / 2, 0) + np.where(last_day, L.COST / 2, 0) + np.where(inwin & ~first_day, chg * L.COST, 0)
    if side != 'always':   # 0050 側切換成本忽略（ETF 稅 0.1%，小）
        pass
    r = (r - cost)[m]
    for pname, (a, b) in {'sel_2021_2023': ('2021-02-01', '2023-12-31'), 'val_2024_2026': ('2024-01-01', '2026-12-31')}.items():
        mm = (r.index >= a) & (r.index <= b)
        res['A_revenue'][f'trade_Q5_in_{side}_else_0050_{pname}'] = L.perf(r[mm], D['e50'][m][mm], D['mkt_liq'][m][mm])
# ---------- B. 整體市場 ----------
tw = D['twii'].reindex(idx).pct_change()
mk = {'0050': D['e50'], 'TAIEX_price': tw, 'EW_liquid': D['mkt_liq']}
res['B_market'] = {}
SPL = {'sel_2005_2019-04': ('2005-01-01', '2019-04-30'), 'val_2019-05_2026': ('2019-05-01', '2026-12-31'), 'ifrs_2013_2026': ('2013-01-01', '2026-12-31')}
for name, x in mk.items():
    o = {}
    for pname, (a, b) in SPL.items():
        mm = (idx >= a) & (idx <= b)
        o[pname] = day_stats(x[mm], cls[mm])
        w20 = pd.DataFrame({'r': x[mm], 'w': win_id[mm]}); w20 = w20[w20['w'].str.startswith('pre')].groupby('w')['r'].apply(lambda v: (1 + v).prod() - 1)
        uncond = float(((1 + x[mm]).rolling(20).apply(np.prod, raw=True) - 1).mean())
        o[pname]['pre_window_20d_mean_pct'] = round(float(w20.mean() * 100), 2); o[pname]['uncond_20d_mean_pct'] = round(uncond * 100, 2)
        o[pname]['pre_window_hit'] = round(float((w20 > 0).mean()), 2); o[pname]['n_windows'] = int(len(w20))
        by_q = {}
        for md in ['03-31', '05-15', '08-14', '11-14']:
            sel = w20[[k for k in w20.index if (pd.Timestamp(k[4:]) + pd.Timedelta(days=0)).strftime('%m-%d') <= md and (pd.Timestamp(k[4:])).month == int(md[:2])]]
            by_q[md] = {'n': int(len(sel)), 'mean_pct': round(float(sel.mean() * 100), 2), 'hit': round(float((sel > 0).mean()), 2)}
        o[pname]['pre_window_by_deadline'] = by_q
        if name == '0050':
            for side in ('pre', 'notpre'):
                inw = (cls[mm] == 'pre') if side == 'pre' else (cls[mm] != 'pre')
                rr = x[mm].where(inw, 0.0).fillna(0); eq = (1 + rr).cumprod(); yrs = (rr.index[-1] - rr.index[0]).days / 365.25
                o[pname][f'hold_0050_only_{side}'] = {'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'share_days': round(float(inw.mean()), 3)}
            eq = (1 + x[mm].fillna(0)).cumprod(); o[pname]['buy_hold_0050_CAGR'] = round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4)
    res['B_market'][name] = o
L.save('3_earnings_season', res)
import json
for p in ('sel_2021_2023', 'val_2024_2026'):
    print(p, json.dumps(res['A_revenue'][p], ensure_ascii=False))
    for side in ('pre', 'post'): print(side, res['A_revenue'][f'{side}_windows_{p}'])
    for side in ('pre', 'post', 'always'):
        v = res['A_revenue'][f'trade_Q5_in_{side}_else_0050_{p}']; print('trade', side, v['strategy'], v['0050'], v['t_vs_0050'])
for n, o in res['B_market'].items():
    for p, v in o.items(): print(n, p, {k: v[k] for k in v if k != 'pre_window_by_deadline'})

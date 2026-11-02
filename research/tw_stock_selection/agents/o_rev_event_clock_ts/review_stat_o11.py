#!/usr/bin/env python3
"""o11 反方審查（資料與統計）：只讀研究員檔案，重算 P2 並做穩健性檢查。python3 -I -B review_stat_o11.py"""
import math, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
FS = pd.read_csv(H / 'data' / 'firstseen_hist.csv', dtype={'code': str})
ts = FS[FS.valid_any & FS.rel.notna() & FS.rev_pct.notna()].copy(); ts['rel'] = ts.rel.astype(int)
ts['grp'] = np.where(ts.rel >= 3, 'early', np.where(ts.rel <= 0, 'dead', 'mid'))
out = {}
def tstat(d):
    d = np.asarray(d, float); return d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
def nw(d, L):
    d = np.asarray(d, float); n = len(d); e = d - d.mean(); g0 = (e @ e) / n; s = g0
    for l in range(1, L + 1): s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return d.mean() / math.sqrt(s / n)
g = ts[ts.grp != 'mid']
mm = g.groupby(['anchor', 'grp']).rev_pct.mean().unstack()
out['anchors_with_valid'] = int(ts.anchor.nunique())
out['anchors_dropped'] = mm[mm.isna().any(axis=1)].index.tolist()
out['n_group_by_anchor'] = g.groupby(['anchor', 'grp']).size().unstack().fillna(0).astype(int).to_dict('index')
d = (mm['early'] - mm['dead']).dropna()
out['replicate'] = dict(mean=d.mean(), t=tstat(d), n=len(d))
out['nw_t'] = {L: nw(d.values, L) for L in (1, 2, 3)}
out['ac1_monthly_diff'] = float(pd.Series(d.values).autocorr(1))
# 稀疏月：早公告組 < 10 檔的月份拿掉
cnt = g.groupby(['anchor', 'grp']).size().unstack().fillna(0)
okm = cnt[(cnt.early >= 10) & (cnt.dead >= 10)].index
d10 = d[d.index.isin(okm)]; out['min10_each'] = dict(mean=d10.mean(), t=tstat(d10), n=len(d10))
# 股票重複：早公告者是否為固定一批股票（廠商固定效果）
e = g[g.grp == 'early']; dd = g[g.grp == 'dead']
out['n_unique_early_codes'] = int(e.code.nunique()); out['n_unique_dead_codes'] = int(dd.code.nunique())
top = e.code.value_counts().head(10); out['top_early_codes'] = top.to_dict()
# 股內去平均（只用同時有早、期限日紀錄的股票）
both = set(e.code) & set(dd.code); out['n_codes_both'] = len(both)
gb = g[g.code.isin(both)].copy()
gb['dm'] = gb.rev_pct - gb.groupby('code').rev_pct.transform('mean')
per = gb.groupby(['code', 'grp']).dm.mean().unstack().dropna()
dw = per['early'] - per['dead']
out['within_firm'] = dict(n_codes=len(dw), mean=dw.mean(), t_by_code=tstat(dw))
# 以股票分群：每檔股票的早減期限差（未去平均，等同上面）；另做 between：每檔股票「早公告比例」與平均百分位的相關
s = ts.groupby('code').agg(share_early=('grp', lambda x: (x == 'early').mean()), pct=('rev_pct', 'mean'), n=('rev_pct', 'size'))
s = s[s.n >= 5]; out['between_corr_share_early_vs_pct'] = dict(n=len(s), corr=float(s.share_early.corr(s.pct)))
# 去掉貢獻最大的 10 檔早公告股
cont = e.groupby('code').rev_pct.sum() - e.groupby('code').size() * dd.rev_pct.mean()
drop = set(cont.sort_values(ascending=False).head(10).index)
g2 = g[~g.code.isin(drop)]; mm2 = g2.groupby(['anchor', 'grp']).rev_pct.mean().unstack().dropna(); d2 = mm2.early - mm2.dead
out['drop_top10_early_contrib'] = dict(mean=d2.mean(), t=tstat(d2), n=len(d2))
# 逐年（閘門口徑）
out['by_year_strict'] = {y: dict(mean=v.mean(), t=tstat(v) if len(v) > 2 else None, n=len(v)) for y, v in d.groupby(d.index.str[:4])}
# 早公告股中「池內營收百分位」本身的機械性：rev_pct 用的是 y3（含 m-1）；檢查早/期限日組在 el 的比率
out['el_rate'] = g.groupby('grp').el.mean().to_dict()
# 星期效果：有效樣本的公告日星期分布（週末缺口排除週一首見）
ts['wd'] = pd.to_datetime(ts.a_day).dt.dayofweek
out['strict_weekday'] = ts.wd.value_counts().sort_index().to_dict()
# 前瞻檢定力重算
sd = 0.51 * math.sqrt(240) / 2.8; se = sd / math.sqrt(24)
from statistics import NormalDist
N = NormalDist(); out['power_check'] = {mu: 1 - N.cdf(1.5 - mu / se) for mu in (0, 0.3, 0.5)}
# t 分布（23 自由度）下 t≥1.5 的誤升級率近似
out['note'] = 'P2 只描述營收排名，未涉任何報酬'
print(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))

# ---- 補充：對稱剔除與股票層級 bootstrap ----
def mdiff(gg):
    m_ = gg.groupby(['anchor', 'grp']).rev_pct.mean().unstack().dropna(); v = m_.early - m_.dead
    return v.mean(), (tstat(v) if len(v) > 2 else float('nan')), len(v)
sym = {}
for k in (5, 10, 20):
    hi = set(cont.sort_values(ascending=False).head(k).index); lo = set(cont.sort_values().head(k).index)
    sym[k] = dict(drop_top=mdiff(g[~g.code.isin(hi)]), drop_bottom=mdiff(g[~g.code.isin(lo)]), drop_both=mdiff(g[~g.code.isin(hi | lo)]))
freq = set(e.code.value_counts().head(10).index)
sym['drop_10_most_frequent_early'] = mdiff(g[~g.code.isin(freq)])
rng = np.random.default_rng(11); codes = np.array(sorted(g.code.unique())); bs = []
gi = {c: x for c, x in g.groupby('code')}
for _ in range(1000):
    pick = rng.choice(codes, len(codes), replace=True)
    gg = pd.concat([gi[c].assign(code=f'{c}_{i}') for i, c in enumerate(pick)])
    bs.append(mdiff(gg)[0])
bs = np.array(bs)
sym['stock_bootstrap'] = dict(mean=bs.mean(), sd=bs.std(), t_like=d.mean() / bs.std(), p_le_0=float((bs <= 0).mean()), p_lt_005=float((bs < 0.05).mean()))
# 池內 rev_pct 母體均值 0.5 為基準：早組、期限日組各自偏離
sym['level'] = dict(early=float(e.rev_pct.mean()), dead=float(dd.rev_pct.mean()), all_pool=float(FS[FS.rev_pct.notna()].rev_pct.mean()))
print(json.dumps(sym, ensure_ascii=False, indent=1, default=float))

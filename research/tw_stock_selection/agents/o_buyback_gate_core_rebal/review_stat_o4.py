# -*- coding: utf-8 -*-
"""反方審查（資料與統計，o4）：只讀研究員檔案，不寫回。python3 -I -B review_stat_o4.py"""
import sys, json, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
PRE = res['PREREG']; O = res['OOS']
def p(*a): print(*a, flush=True)

# 0. 檔案時序與 sha
p('rlib sha now', hashlib.sha256((HERE / 'rlib.py').read_bytes()).hexdigest()[:12], 'prereg', PRE['files_sha256']['rlib.py'][:12])
N = L.NAVS
p('NaN per col', N.isna().sum().to_dict(), 'index', N.index[0].date(), N.index[-1].date(), len(N))
# 0050 是否有零報酬的填補日（停牌或缺值被 ffill）
r50 = N['E50'].pct_change()
p('E50 zero-return days', int((r50 == 0).sum()), 'of', int(r50.notna().sum()))

# 1. 重現判定
for cost in ('normal', 'stress'):
    a, b = L.PER['SEL']
    n1 = L.sim(0.3, 'NONE', a, b, cost=cost)['nav']; n0 = L.sim(0.3, 'M', a, b, cost=cost)['nav']
    p('judge', cost, round(L.paired(n1, n0)['ann_diff_pp'], 4), round(L.ann_diff_drop_year(n1, n0, 2015), 4))

# 2. 頻率「不重要」的檢定力：POST 期每條規則對 M 的年化差標準誤與 80% 檢定力最小可偵測效果
a, b = L.PER['POST']
m = L.sim(0.3, 'M', a, b)['nav']
pw = {}
for ru in L.RULES:
    if ru == 'M': continue
    d = (L.win(L.sim(0.3, ru, a, b)['nav']) - L.win(m)).dropna()
    se_ann = d.std(ddof=1) / math.sqrt(len(d)) * 12 * 100
    pw[ru] = dict(n=len(d), mean_ann_pp=round(d.mean() * 1200, 3), se_ann_pp=round(se_ann, 3), mde80_pp=round(2.8 * se_ann, 2))
p('POWER POST w30', pw)

# 3. 主策略腿與 70/30 對 0050：集中在少數年份／少數月份？
def concentr(nav, tag):
    b0 = L.NAVS['E50'].reindex(nav.index); b0 = b0 / b0.iloc[0]
    d = (L.win(nav) - L.win(b0)).dropna()
    yd = L.yearly_diff(nav, b0).sort_values(ascending=False)
    srt = d.sort_values(ascending=False)
    out = dict(t=round(L.tstat(d), 3), nwt=round(L.nw_t(d), 3), n=len(d), top_years=yd.head(4).round(1).to_dict())
    for k in (1, 3, 5, 10): out[f't_drop_best{k}m'] = round(L.tstat(d.drop(srt.index[:k])), 3)
    # 拿掉最好 1／2／3 個日曆年
    yrs = d.groupby(d.index.year).sum().sort_values(ascending=False).index.tolist()
    for k in (1, 2, 3):
        out[f't_drop_best{k}y'] = round(L.tstat(d[~d.index.year.isin(yrs[:k])]), 3)
    # 前後半
    h = len(d) // 2
    out['t_first_half'] = round(L.tstat(d.iloc[:h]), 3); out['t_second_half'] = round(L.tstat(d.iloc[h:]), 3)
    # 正超額年數
    out['pos_years'] = f'{int((yd > 0).sum())}/{len(yd)}'
    p('CONC', tag, out)
    return d
a, b = L.PER['ALL']
dA = concentr(L.sim(0.0, 'M', a, b)['nav'], 'MAIN100 ALL')
d7 = concentr(L.sim(0.3, 'M', a, b)['nav'], '70/30M ALL')
d23 = concentr(L.sim(0.3, 'M', a, b, B='A2330')['nav'], '2330_30 ALL')
for cost in ('stress',):
    concentr(L.sim(0.3, 'M', a, b, cost=cost)['nav'], '70/30M ALL stress')
for P in ('SEL', 'POST', 'HOLD'):
    concentr(L.sim(0.3, 'M', *L.PER[P])['nav'], f'70/30M {P}')
    concentr(L.sim(0.0, 'M', *L.PER[P])['nav'], f'MAIN100 {P}')

# 4. 2330 30% 的 t 3.46：拿掉 2330 的事後贏家性質——換成 0050 腿以外的「同期最大市值股」無法做，改看 2330 腿本身對 0050 的 t
b2330 = L.bench('A2330', a, b); b50 = L.bench('E50', a, b)
d2 = (L.win(b2330) - L.win(b50)).dropna()
p('2330 alone vs 0050 ALL t', round(L.tstat(d2), 2), 'CAGR2330', round(L.cagr(b2330) * 100, 2))

# 5. 前緣斜率 ALL／POST
fr = O['frontier']
for P in ('SEL', 'ALL', 'POST', 'HOLD'):
    x0 = fr[f'M|0|{P}']; x3 = fr[f'M|30|{P}']
    p('SLOPE', P, round((x0['CAGR'] - x3['CAGR']) / (x0['TE'] - x3['TE']), 3), 'IR_main', round(x0['ex_pp'] / x0['TE'], 3))

# 6. NONE 的優勢是否只是曝險：在 POST，用 2006 起的連續路徑（期初已漂到 ~90/10）
p('CONT', {k: (round(v['paired_vs_M']['ann_diff_pp'], 2), round(v['paired_vs_M']['t'], 2)) for k, v in O['continuous_path'].items() if '|30' in k})

# 7. 多重檢定：把 it6 壓力成本、it2 也算進去時的有效族數；簡單用 max-t 置換（符號翻轉）SEL 63 格
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
a, b = L.PER['SEL']
D = {}
for w in (0.2, 0.3, 0.4):
    mref = L.win(L.sim(w, 'M', a, b)['nav'])
    for ru in L.RULES:
        if ru == 'M': continue
        D[f'{int(w*100)}|{ru}'] = (L.win(L.sim(w, ru, a, b)['nav']) - mref).to_numpy()
X = np.column_stack(list(D.values()))
tobs = X.mean(0) / (X.std(0, ddof=1) / math.sqrt(len(X)))
rng = np.random.default_rng(4)
# 區塊（12 個月）符號翻轉保留各格之間與時間上的相關
nb = int(math.ceil(len(X) / 12)); mx = []
Xc = X - X.mean(0)
for _ in range(4000):
    s = np.repeat(rng.choice([-1, 1], nb), 12)[:len(X)]
    Y = Xc * s[:, None]
    mx.append(np.max(np.abs(Y.mean(0) / (Y.std(0, ddof=1) / math.sqrt(len(Y))))))
mx = np.array(mx)
k30 = list(D).index('30|NONE'); kb5 = list(D).index('30|B5')
p('MAXT family=24 SEL: NONE30 t', round(tobs[k30], 2), 'adj p', round(float((mx >= abs(tobs[k30])).mean()), 3),
  '| B5_30 t', round(tobs[kb5], 2), 'adj p', round(float((mx >= abs(tobs[kb5])).mean()), 3))

# 8. 端點：POST 期 NONE 對 M 的逐年
a, b = L.PER['POST']
p('POST yearly NONE-M w30', L.yearly_diff(L.sim(0.3, 'NONE', a, b)['nav'], L.sim(0.3, 'M', a, b)['nav']).round(2).to_dict())

# 9. 閘門：2010 年事件、事件聚集
G = pd.read_csv(HERE / 'gate_events.csv', parse_dates=['date', 'sig'])
g2 = G[G.v60 >= 2e8]
p('GATE ge2e8 n', len(g2), 'unique codes', g2.code.nunique(), 'top codes', g2.code.value_counts().head(5).to_dict(),
  'median r20', round(g2.r20_net.median() * 100, 2), 'mean', round(g2.r20_net.mean() * 100, 2), 'frac>0', round((g2.r20_net > 0).mean(), 2),
  't mean', round(L.tstat(g2.r20_net.dropna()), 2))
p('GATE all date<2011', int((G.date < '2011-01-01').sum()))
# 2015/2011 群聚（市場急跌期）
p('GATE ge2e8 by month top', g2.sig.dt.to_period('M').value_counts().head(5).to_dict())

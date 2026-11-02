#!/usr/bin/env python3
"""主策略重建（近似）與 P6／過濾版：
主策略＝每月 11 日後第一個交易日收盤算分數：上市、排除金融、60 日均值 ≥ 1,000 萬、收盤 ≥ 10、非處置，取 60 日均值前 300 名；
分數＝0.5×52 週比百分位＋0.5×營收 3 月年增率百分位（只用已公布月份）；取前 10 檔，隔日開盤等權買、下次換股隔日開盤賣。
每檔附上最近一次（A ≤ 訊號日、180 日內）季報的 SUE⊥（PIT）。
用法：python3 -I -B s2b_main_overlap.py sel|val|all   （sel：訊號日 ≤ 2019-04-30；val：2019-05 起，需 PREREG）"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pcore import DATA, HERE, load_panel, tstat, FIN_IND

mode = sys.argv[1] if len(sys.argv) > 1 else 'sel'
if mode in ('val', 'all'):
    R = json.load(open(HERE / 'results.json'))
    assert 'PREREG' in R and R['PREREG'].get('sha256'), 'PREREG 不存在，不可計算樣本外'
P = load_panel(); IDX = pd.DatetimeIndex(P['IDX']); COLS = np.array(P['COLS']); T, N = P['C'].shape
O, C, Rr, V60, MKT = P['O'], P['C'], P['R'], P['V60'], P['MKT']
G = np.where(np.isfinite(Rr), 1 + Rr, 1.0); CUM = np.cumprod(G, axis=0); CF = pd.DataFrame(C).ffill().to_numpy()
HI = pd.DataFrame(CUM).rolling(252, min_periods=240).max().to_numpy(); R52 = CUM / HI
fin = pd.read_parquet(DATA / 'fin_long.parquet'); ind = fin.sort_values(['y', 'q']).groupby('code')['industry'].last()
okc = np.array([c.isdigit() and len(c) == 4 and not c.startswith('0') and not c.startswith('91') and ind.get(c, 'ci') not in FIN_IND for c in COLS])
rv = pd.read_parquet(DATA / 'rev_long.parquet')
YOY = rv.pivot(index='ym', columns='code', values='yoy_pct').sort_index(); YOY.index = pd.PeriodIndex(YOY.index, freq='M')
YOY3 = YOY.rolling(3, min_periods=3).mean().reindex(columns=COLS)
dsp = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['start', 'end'])
E = pd.read_parquet(DATA / 'events.parquet')
E = E[E['SUEP'].notna()][['code', 'A', 'k', 'SUEP', 'SUEP_thr20', 'SUEP_thr80', 'SUE']].sort_values('A')
s = pd.Series(IDX, index=IDX)
sig_days = [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([IDX.year, IDX.month]) if (g.dt.day >= 11).any()]
sig_days = [d for d in sig_days if d >= pd.Timestamp('2016-01-11')]
rows = []
for n, d in enumerate(sig_days[:-1]):
    t = IDX.get_loc(d); t2 = IDX.get_loc(sig_days[n + 1])
    if t + 1 >= T or t2 + 1 >= T:
        continue
    pool = okc & (MKT == 1) & (V60[t] >= 1e7) & (CF[t] >= 10) & np.isfinite(R52[t])
    dd = dsp[(dsp['start'] <= d) & (dsp['end'] >= d)]['code']
    pool &= ~np.isin(COLS, dd.to_numpy())
    w = np.where(pool)[0]
    w = w[np.argsort(-V60[t, w])][:300]
    m_av = pd.Period(d - pd.Timedelta(days=10), 'M') - 1
    if m_av not in YOY3.index:
        continue
    y3 = YOY3.loc[m_av].to_numpy()[w]
    ok = np.isfinite(y3)
    w = w[ok]; y3 = y3[ok]
    sc = 0.5 * pd.Series(R52[t, w]).rank(pct=True).to_numpy() + 0.5 * pd.Series(y3).rank(pct=True).to_numpy()
    order = w[np.argsort(-sc, kind='stable')]
    e, x = t + 1, t2 + 1
    for rk, j in enumerate(order[:30]):
        if not (np.isfinite(O[e, j]) and O[e, j] > 0):
            continue
        v = C[e, j] / O[e, j] if np.isfinite(C[e, j]) else 1.0
        v *= CUM[x - 1, j] / CUM[e, j]
        gx = O[x, j] * G[x, j] / C[x, j] if (np.isfinite(O[x, j]) and np.isfinite(C[x, j]) and C[x, j] > 0) else 1.0
        rows.append(dict(sig=d, code=COLS[j], rank=rk, ret=v * gx - 1))
M = pd.DataFrame(rows)
# 附上最近 SUE⊥
M = M.sort_values('sig')
M = pd.merge_asof(M, E.rename(columns={'A': 'Adt'}).sort_values('Adt'), left_on='sig', right_on='Adt', by='code', direction='backward',
                  tolerance=pd.Timedelta(days=180), allow_exact_matches=False)
M['bot'] = M['SUEP'] <= M['SUEP_thr20']; M['top'] = M['SUEP'] >= M['SUEP_thr80']
cut = pd.Timestamp('2019-04-30')
if mode == 'sel':
    M = M[M['sig'] <= cut]
elif mode == 'val':
    M = M[M['sig'] > cut]
picks = M[M['rank'] < 10]
out = {'mode': mode, 'months': int(picks['sig'].nunique()), 'picks': int(len(picks)), 'with_suep': float(picks['SUEP'].notna().mean())}
# P6a：入選股中 SUE⊥ 最低 1/5 的下一期報酬 − 其他入選股（按月配對）
a = picks[picks['bot']].groupby('sig')['ret'].mean(); b = picks[~picks['bot'] & picks['SUEP'].notna()].groupby('sig')['ret'].mean()
d = (a - b).dropna()
out['P6a_bot_minus_other'] = {'mean': float(d.mean()), 't': float(tstat(d)), 'n_months': int(len(d)), 'share_bot': float(picks['bot'].mean())}
# P6b：右尾（前 5% 持股期報酬）中高 SUE⊥ 比例 vs 20%
q95 = picks['ret'].quantile(0.95)
tail = picks[picks['ret'] >= q95]
out['P6b_tail_top_share'] = {'share_top_in_tail': float(tail['top'].mean()), 'share_top_all': float(picks['top'].mean()), 'n_tail': int(len(tail))}
# 過濾版：剔除 SUE⊥ 最低 1/5，依名次遞補到 10 檔
base = picks.groupby('sig')['ret'].mean()
flt = M[~M['bot']].sort_values(['sig', 'rank']).groupby('sig').head(10).groupby('sig')['ret'].mean()
dd = (flt - base).dropna()
out['filter_minus_main'] = {'mean_m': float(dd.mean()), 't': float(tstat(dd)), 'n_months': int(len(dd)),
                            'main_ann_gross': float((1 + base).prod() ** (12 / len(base)) - 1), 'filter_ann_gross': float((1 + flt).prod() ** (12 / len(flt)) - 1)}
pd.DataFrame({'main': base, 'filter': flt}).to_csv(DATA / f'main_recon_{mode}.csv')
json.dump(out, open(DATA / f'main_overlap_{mode}.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

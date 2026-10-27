#!/usr/bin/env python3
"""一週短期反轉回測。規則見 results.json 的 PREREG（prereg.py 事前寫入，不改）。
引擎與成本只從 exec_rebuild/exlib.py 匯入（sim_portfolio、cost_rates、價格陣列）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402
tstat = L.tstat

P = L.P
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag']}
cols = list(L.COLS); keep = np.array([not c.startswith('91') for c in cols])
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
G = {k: v.loc[good] for k, v in D.items()}
IG = G['adj'].index; TG = len(IG); N = L.N
F = G['flag']; bad5 = F == 5
ADJ = G['adj'].mask(bad5); CR = G['close_raw'].mask(bad5); HI = G['high'].mask(bad5); V = G['value']; RT = G['ret'].mask(bad5)
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
R5 = ADJ / ADJ.shift(5) - 1
lim = np.where(IG < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
LOCKUP = (RT.to_numpy() >= lim - 0.006) & (CR.to_numpy() >= HI.to_numpy() * 0.999)
LOCKDN = RT.to_numpy() <= -(lim - 0.006)
mkt4 = (F == 4).sum(axis=1) >= 50
FL = (F.isin([3, 4, 5]) & ~((F == 4) & mkt4.to_numpy()[:, None])).astype(float)
FLW = FL.rolling(6, min_periods=1).max().to_numpy() > 0           # [d-5, d]
VOLSH = (V / CR).where(V > 0)
VOL5 = VOLSH.rolling(5, min_periods=3).mean(); VOL50 = VOLSH.shift(5).rolling(50, min_periods=40).mean()
LOWVOL = (VOL5 < VOL50).to_numpy()

# 產業
PREFIX_IND = {'11': '水泥工業', '12': '食品工業', '13': '塑膠工業', '14': '紡織纖維', '15': '電機機械', '16': '電器電纜',
              '17': '化學工業', '18': '玻璃陶瓷', '19': '造紙工業', '20': '鋼鐵工業', '21': '橡膠工業', '22': '汽車工業',
              '25': '建材營造業', '26': '航運業', '27': '觀光餐旅', '28': '金融保險業', '29': '貿易百貨業', '99': '其他業'}
_sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str}).drop_duplicates('code').set_index('code')['industry']
IND, IND_SRC = [], {'list': 0, 'prefix': 0, 'unclassified': 0}
for c in cols:
    if c in _sl.index and isinstance(_sl[c], str): IND.append(_sl[c]); IND_SRC['list'] += 1
    elif c[:2] in PREFIX_IND: IND.append(PREFIX_IND[c[:2]]); IND_SRC['prefix'] += 1
    else: IND.append('未分類'); IND_SRC['unclassified'] += 1
IND = np.array(IND)

# 週曆
wk = IG.to_period('W-SUN')
last_of_week = pd.Series(np.arange(TG), index=IG).groupby(wk).max().to_numpy()
SIG = [i for i in last_of_week if IG[i] >= pd.Timestamp('2005-01-01') and i + 1 < TG]
OPEN_OK = np.isfinite(G['open'].to_numpy()) & np.isfinite(ADJ.to_numpy())
LPOS = {d: k for k, d in enumerate(L.IDX)}
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - 0.006
GAP_UP = (L.AO_raw / L.AC_prev - 1) >= thrL   # L.IDX 索引（L.AC_prev 已 ffill，跨過 2018-01-13）

ADJn, CRn, V60n, R5n = ADJ.to_numpy(), CR.to_numpy(), V60.to_numpy(), R5.to_numpy()
BASE = (V60n >= 1e7) & (CRn >= 10) & np.isfinite(ADJn) & keep[None, :]
POOL = BASE & np.isfinite(R5n) & ~LOCKUP & ~LOCKDN & ~FLW

# 殘差反轉分數（週報酬＝相鄰訊號日收盤）
SA = ADJ.ffill(limit=5).to_numpy()[SIG]; Wr = SA[1:] / SA[:-1] - 1; Wr = np.vstack([np.full((1, N), np.nan), Wr])
Wr[~np.isfinite(ADJn[SIG])] = np.nan
base_prev = np.vstack([np.zeros((1, N), bool), BASE[SIG][:-1]])
Mk = np.array([np.nanmean(np.clip(Wr[k][base_prev[k] & np.isfinite(Wr[k])], -0.5, 1.0)) if (base_prev[k] & np.isfinite(Wr[k])).sum() > 30 else np.nan for k in range(len(SIG))])
RES = np.full((len(SIG), N), np.nan)
for k in range(51, len(SIG)):
    Rw = Wr[k - 51:k + 1]; M = Mk[k - 51:k + 1][:, None] * np.ones((1, N))
    ok = np.isfinite(Rw) & np.isfinite(M); cnt = ok.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        Rz = np.where(ok, Rw, 0); Mz = np.where(ok, M, 0)
        mx = Mz.sum(0) / cnt; my = Rz.sum(0) / cnt
        sxx = np.where(ok, (M - mx) ** 2, 0).sum(0); sxy = np.where(ok, (M - mx) * (Rw - my), 0).sum(0)
        b = sxy / sxx; a = my - b * mx
        E = np.where(ok, Rw - a - b * M, np.nan)
        sd = np.sqrt(np.nansum(E ** 2, 0) / (cnt - 2))
        sc = E[-1] / sd
    sc[(cnt < 40) | ~np.isfinite(sc)] = np.nan
    RES[k] = sc

def score_for(var, k, i):
    if var in ('main', 'c_low_volume_losers', 'd_hold2w', 'd_hold2w_phase2'): return R5n[i].copy()
    if var == 'a_industry_adj':
        s = R5n[i].copy(); p = POOL[i]
        g = pd.Series(IND[p]); sz = g.map(g.value_counts()); g = g.where(sz >= 5, '小產業合併').to_numpy()
        m = pd.Series(s[p]).groupby(g).transform('mean').to_numpy()
        out = np.full(N, np.nan); out[p] = s[p] - m; return out
    if var == 'b_residual': return RES[k].copy()

def build(var):
    events, pools = {}, {}; held = set(); info = {'weeks': 0, 'blocked': 0, 'n_names': [], 'pool': [], 'fallback_weeks': 0}
    for k, i in enumerate(SIG):
        if var == 'd_hold2w' and k % 2 == 1: continue
        if var == 'd_hold2w_phase2' and k % 2 == 0: continue
        ex = i + 1
        sc = score_for(var, k, i)
        ok = POOL[i] & OPEN_OK[ex] & np.isfinite(sc)
        npool = int((POOL[i] & OPEN_OK[ex]).sum())
        if ok.sum() < 100: continue
        n = int(round(0.1 * ok.sum()))
        exL = LPOS[IG[ex]]
        order = np.where(ok, sc, np.inf); cand = np.argsort(order, kind='stable')[:int(ok.sum())]
        pick = []
        for j in cand:
            if GAP_UP[exL, j] and j not in held: info['blocked'] += 1; continue
            pick.append(j)
            if len(pick) == n: break
        if var == 'c_low_volume_losers':
            lv = [j for j in pick if LOWVOL[i, j]]
            if len(lv) >= 5: pick = lv
            else: info['fallback_weeks'] += 1
        events[exL] = ('open', np.array(pick)); pools[exL] = ('open', np.where(POOL[i] & OPEN_OK[ex])[0])
        held = set(pick); info['weeks'] += 1; info['n_names'].append(len(pick)); info['pool'].append(npool)
    info['n_names'] = round(float(np.mean(info['n_names'])), 1); info['pool'] = round(float(np.mean(info['pool'])), 1)
    return events, pools, info

t0 = LPOS[IG[SIG[0] + 1]]; t1 = len(L.IDX) - 1
E50 = L.E50.iloc[t0:t1 + 1]
WIN = {'select_2005-01_2019-04': [('2005-01-01', '2019-04-30')], 'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
       'y2024_2025': [('2024-01-01', '2025-12-31')],
       'valid_ex_src3': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
       'full_2005-01_2026-10': [('2005-01-01', '2026-10-31')]}

def stats(nav, segs, b50, ew):
    r = nav.pct_change(); m = np.zeros(len(r), bool)
    for a, b in segs: m |= (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    m &= r.notna().to_numpy()
    rr = r[m]; span = 0.0
    for a, b in segs:
        x = rr[(rr.index >= pd.Timestamp(a)) & (rr.index <= pd.Timestamp(b))]
        if len(x): span += (x.index[-1] - r.index[max(r.index.get_loc(x.index[0]) - 1, 0)]).days / 365.25
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(span, 2)}
    mon = lambda s: (1 + s).groupby(s.index.to_period('M')).prod() - 1
    sm = mon(rr)
    for nm, bn in (('0050', b50), ('EW', ew)):
        bd = bn.pct_change().reindex(rr.index).fillna(0); bm = mon(bd); ex = sm - bm
        eb = (1 + bd).cumprod()
        o[f'{nm}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
        o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 3); o[f't_vs_{nm}_m'] = tstat(ex); o[f'hit_vs_{nm}_m'] = round(float((ex > 0).mean()), 3)
    o['n_months'] = int(len(sm))
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

res = json.load(open(HERE / 'results.json'))
out = {'meta': {'bad_days_dropped': BAD, 'n_signal_weeks': len(SIG), 'first_signal': str(IG[SIG[0]].date()), 'last_signal': str(IG[SIG[-1]].date()),
                'industry_source_counts': IND_SRC, 'market_flag4_days_ignored': [str(x.date()) for x in IG[mkt4.to_numpy()]]}}
navs = {}
ew_done = None
for var in ['main', 'a_industry_adj', 'b_residual', 'c_low_volume_losers', 'd_hold2w', 'd_hold2w_phase2']:
    ev, pools, info = build(var)
    if var == 'main':
        ew, _ = L.sim_portfolio(pools, 0.0, 0.0, t0, t1); navs['EW_pool'] = ew; ew_done = ew
    block = {'info': info}
    for cname, cr in (('gross', (0.0, 0.0)), ('normal', L.cost_rates(0.001)), ('stress', L.cost_rates('stress'))):
        nav, tov = L.sim_portfolio(ev, cr[0], cr[1], t0, t1); navs[f'{var}_{cname}'] = nav
        block[cname] = {'turnover_oneway_per_rebal': round(tov, 3), 'windows': {w: stats(nav, s, E50, ew_done) for w, s in WIN.items()}}
        if cname == 'normal': block['yearly_normal_pct'] = yearly(nav)
    out[var] = block
    w = block['normal']['windows']
    print(var, info, 'tov', block['normal']['turnover_oneway_per_rebal'], flush=True)
    for cn in ('gross', 'normal', 'stress'):
        for wn in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'valid_ex_src3', 'y2024_2025'):
            x = block[cn]['windows'][wn]
            print(f'  {cn:6s} {wn:24s} CAGR {x["CAGR"]:.4f} DD {x["maxDD"]:.3f} | EW {x["EW_CAGR"]:.4f} t_EW {x["t_vs_EW_m"]} | 0050 {x["0050_CAGR"]:.4f} t_50 {x["t_vs_0050_m"]}', flush=True)
out['yearly_EW_pool_pct'] = yearly(navs['EW_pool']); out['yearly_0050_pct'] = yearly(E50)
pd.DataFrame(navs).to_parquet(HERE / 'navs.parquet')
res['RESULTS'] = out
json.dump(res, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('done')

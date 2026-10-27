#!/usr/bin/env python3
"""w12 外資總買超布林擇時：依 results.json 的 PREREG 執行（不改 PREREG）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # 只讀：FEE、cost_rates、面板路徑
from btlib import tstat  # noqa: E402

RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
PRE = RES['PREREG']
OUT = {'PREREG': PRE}
X = L.X; P = L.P

# ---------------- 面板與日曆 ----------------
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet')
good = adj.notna().sum(axis=1) >= 300
OUT['dropped_panel_days_lt300'] = [str(d.date()) for d in adj.index[~good]]
adj, C, V = adj.loc[good], C.loc[good], V.loc[good]
cols = [c for c in adj.columns if len(c) == 4 and c.isdigit() and not c.startswith('0') and not c.startswith('91')]
adj, C, V = adj[cols], C[cols], V[cols]
PIDX = adj.index
RN = adj.pct_change(fill_method=None)
V60 = V.rolling(60, min_periods=50).mean()

e = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
e = e.drop(index=[d for d in [pd.Timestamp('2026-07-10'), pd.Timestamp('2026-09-20')] if d in e.index])
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date').sort_index()
START, END = pd.Timestamp('2021-10-08'), pd.Timestamp('2026-10-22')
gapd = e.index[(e.index >= '2024-01-02') & (e.index <= '2024-02-14')]
CAL = PIDX.union(gapd); CAL = CAL[(CAL >= START) & (CAL <= END)]
NC = len(CAL)
TXc = tx['close'].reindex(CAL); TXo = tx['open'].reindex(CAL)
OUT['calendar_check'] = {'n_days': NC, 'taiex_missing': [str(d.date()) for d in CAL[TXc.isna() | TXo.isna()]],
                         '0050_missing_filled_by_taiex': [str(d.date()) for d in CAL[e.reindex(CAL).isna()]]}
TXc = TXc.ffill(); TXo = TXo.fillna(TXc)
E = e.reindex(CAL)
for i in range(1, NC):
    if not np.isfinite(E.iloc[i]): E.iloc[i] = E.iloc[i - 1] * TXc.iloc[i] / TXc.iloc[i - 1]
E = E.to_numpy(); TXcn = TXc.to_numpy(); TXon = TXo.to_numpy()
O50 = np.r_[np.nan, E[:-1] * TXon[1:] / TXcn[:-1]]          # 0050 開盤近似
r50 = np.r_[np.nan, E[1:] / E[:-1] - 1]

# ---------------- 外資總買超 F ----------------
ins = pd.read_csv(X / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign'], parse_dates=['date'])
ih = pd.read_csv(X / 'inst_hist.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign'], parse_dates=['date'])
ih['code'] = ih['code'].str.zfill(4); ih = ih[ih['date'] > ins['date'].max()]
ins = pd.concat([ins, ih]); ins = ins[ins['code'].isin(cols)]
FOR = ins.pivot_table(index='date', columns='code', values='foreign', aggfunc='sum').reindex(columns=cols)
idays = FOR.index[(FOR.index >= START) & (FOR.index <= END)]
FOR = FOR.reindex(idays)
Cp = C.reindex(idays)
amt = FOR * 1000.0 * Cp
nstk = (FOR.notna() & Cp.notna()).sum(axis=1)
Fd = amt.sum(axis=1, min_count=1).where(nstk >= 300)
F1 = pd.Series(np.nan, index=CAL); F1.loc[Fd.index.intersection(CAL)] = Fd.reindex(Fd.index.intersection(CAL))
OUT['F_check'] = {
    'inst_days_in_cal': int(len(idays.intersection(CAL))), 'cal_days_without_F': [str(d.date()) for d in CAL[F1.isna()]][:60],
    'n_cal_days_without_F': int(F1.isna().sum()), 'inst_days_not_in_cal': [str(d.date()) for d in idays.difference(CAL)],
    'stocks_per_day_median': int(nstk.median()), 'stocks_per_day_min': int(nstk[nstk > 0].min()),
    'F_NT$bn_mean': round(float(F1.mean() / 1e9), 2), 'F_NT$bn_sd': round(float(F1.std() / 1e9), 2),
    'F_top5_NT$bn': {str(d.date()): round(float(v / 1e9), 1) for d, v in F1.nlargest(5).items()},
    'F_bottom5_NT$bn': {str(d.date()): round(float(v / 1e9), 1) for d, v in F1.nsmallest(5).items()},
    'corr_F_same_day_0050': round(float(pd.Series(r50, index=CAL).corr(F1)), 3),
}
Fv = F1.dropna()
F5v = Fv.rolling(5, min_periods=5).sum()

def band_state(flow, N, k):
    s = Fv if flow == 'F1' else F5v.dropna()
    mid = s.rolling(N, min_periods=N).mean(); sd = s.rolling(N, min_periods=N).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    sig = pd.Series(np.nan, index=s.index)
    sig[s > up] = 1.0; sig[s < lo] = 0.0
    first = mid.first_valid_index()
    sig = sig.reindex(CAL)
    tgt = sig.ffill().fillna(0.0).to_numpy()          # 收盤後目標部位（初始現金）
    sigarr = sig.to_numpy()
    return tgt, first, sigarr

# ---------------- 部位與報酬 ----------------
LOCKTH = 0.10 - L.LTOL
def positions(tgt, mode):
    """pos[d] = d 日收盤時持有（1）或現金（0）。訊號 t → t+1 成交；開盤漲停買不到 → 順延。"""
    pos = np.zeros(NC)
    for d in range(1, NC):
        want = tgt[d - 1]; cur = pos[d - 1]
        if want > cur and mode == 'open' and O50[d] / E[d - 1] - 1 >= LOCKTH: want = cur
        pos[d] = want
    return pos

def daily_ret(pos, mode, buy, sell, cash_rate=0.0):
    cd = (1 + cash_rate) ** (1 / 252) - 1
    r = np.zeros(NC)
    for d in range(1, NC):
        a, b = pos[d - 1], pos[d]
        if mode == 'open':
            g = (1 + a * (O50[d] / E[d - 1] - 1)) * (1 + b * (E[d] / O50[d] - 1))
            if a == 0 and b == 0: g = 1 + cd
            elif a == 1 and b == 0: g *= 1 + 0  # 開盤後持現金，當日利息略
        else:
            g = 1 + a * (E[d] / E[d - 1] - 1) if a == 1 else 1 + cd
        if b > a: g *= (1 - buy)
        if b < a: g *= (1 - sell)
        r[d] = g - 1
    return r

COSTS = {'zero': (0.0, 0.0), 'task_0.1': (0.001, 0.001), 'etf_base': (L.FEE, L.FEE + 0.001),
         'exlib_slip0.1': (L.FEE + 0.001, L.FEE + 0.002), 'stress': L.cost_rates('stress')}
OUT['cost_check_roundtrip_pct'] = {k: round((1 - (1 - b) * (1 - s)) * 100, 3) for k, (b, s) in COSTS.items()}

# ---------------- 基準：同池等權、上市等權 ----------------
RNc = RN.reindex(CAL).to_numpy(); V60c = V60.shift(1).reindex(CAL).to_numpy(); Cprev = C.shift(1).reindex(CAL).to_numpy()
inpanel = CAL.isin(PIDX)
poolEW = np.full(NC, np.nan); listEW = np.full(NC, np.nan); mon = CAL.to_period('M'); u = None; um = None
for t in range(1, NC):
    if not inpanel[t]: continue
    if um != mon[t]:
        v, px = V60c[t], Cprev[t]; base = (v >= 1e7) & (px >= 10)
        order = np.argsort(-np.where(base, v, -np.inf))[:300]; u = np.zeros(len(cols), bool); u[order] = True; u &= base; um = mon[t]
    row = RNc[t]; poolEW[t] = np.nanmean(row[u]); listEW[t] = np.nanmean(row)
poolEW[(CAL >= '2024-01-01') & (CAL <= '2024-02-29')] = np.nan   # 缺價月不比
listEW_f = np.nan_to_num(listEW)

def win(a, b, first):
    a = max(pd.Timestamp(a), first)
    return (CAL > a) & (CAL <= pd.Timestamp(b)), a

def cagr(x, idx):
    eq = np.cumprod(1 + x); yrs = (idx[-1] - idx[0]).days / 365.25 + 1 / 252
    return float(eq[-1] ** (1 / yrs) - 1), float((eq / np.maximum.accumulate(eq) - 1).min())

def monthly(x, idx):
    return pd.Series(x, index=idx).groupby(idx.to_period('M')).apply(lambda g: float(np.prod(1 + g.to_numpy()) - 1) if g.notna().all() else np.nan)

PER = {'IS_2021_2023': ('2021-10-08', '2023-12-31'), 'OOS_2024_2026': ('2024-01-01', '2026-10-22'),
       'sub_2024_2025': ('2024-01-01', '2025-12-31'), 'full': ('2021-10-08', '2026-10-22')}

def evaluate(r, pos, first, a, b):
    m, a0 = win(a, b, first)
    idx = CAL[m]; x = r[m]; p = pos[m]
    if m.sum() < 20: return None
    c, dd = cagr(x, idx); c50, dd50 = cagr(r50[m], idx)
    pe = poolEW[m]; ok = np.isfinite(pe)
    cpe, ddpe = cagr(np.nan_to_num(pe), idx)
    ms = monthly(x, idx); m50 = monthly(r50[m], idx); mpe = monthly(pe, idx); mexp = pd.Series(p, index=idx).groupby(idx.to_period('M')).mean()
    ex50 = ms - m50; expe = (ms - mpe).dropna(); exm = ms - mexp * m50
    pfull = np.r_[pos[np.argmax(m) - 1], p]
    o = {'start': str(idx[0].date()), 'end': str(idx[-1].date()), 'years': round((idx[-1] - idx[0]).days / 365.25, 2),
         'CAGR': round(c, 4), 'vol': round(float(np.std(x) * math.sqrt(252)), 4), 'maxDD': round(dd, 4),
         'exposure': round(float(p.mean()), 3), 'switches': int((np.diff(pfull) != 0).sum()), 'buys': int((np.diff(pfull) > 0).sum()),
         '0050_CAGR': round(c50, 4), '0050_maxDD': round(dd50, 4), 'poolEW_CAGR(缺月以0計)': round(cpe, 4), 'poolEW_maxDD': round(ddpe, 4),
         'n_months': int(ms.notna().sum()),
         'vs0050_m_mean_pct': round(float(ex50.mean() * 100), 3), 't_vs0050_m': tstat(ex50),
         'vsEW_m_mean_pct': round(float(expe.mean() * 100), 3), 't_vsEW_m': tstat(expe), 'n_months_EW': int(len(expe)),
         't_vs0050_daily': tstat(x - r50[m]), 't_vsEW_daily': tstat((x - pe)[ok]),
         'expo_matched_vs0050_m_mean_pct': round(float(exm.mean() * 100), 3), 't_expo_matched_vs0050_m': tstat(exm),
         'yearly_pct': {int(y): round(float(np.prod(1 + g) - 1) * 100, 1) for y, g in pd.Series(x, index=idx).groupby(idx.year)},
         '0050_yearly_pct': {int(y): round(float(np.prod(1 + g) - 1) * 100, 1) for y, g in pd.Series(r50[m], index=idx).groupby(idx.year)}}
    o['C1_vsEW'] = bool(o['vsEW_m_mean_pct'] > 0 and np.isfinite(o['t_vsEW_m']) and o['t_vsEW_m'] >= 2)
    o['C2_not_lose_0050'] = bool(o['CAGR'] >= o['0050_CAGR'])
    o['pass'] = o['C1_vsEW'] and o['C2_not_lose_0050']
    return o

def spells(r, pos, first):
    out = []; d = 1
    while d < NC:
        if pos[d] == 1 and pos[d - 1] == 0:
            st = d
            while d + 1 < NC and pos[d + 1] == 1: d += 1
            ex = d + 1 if d + 1 < NC else None    # 出場成交日（開盤賣）
            seg = slice(st, (ex + 1) if ex is not None else NC)
            net = float(np.prod(1 + r[seg]) - 1)
            le = float(np.prod(1 + listEW_f[st:(ex if ex else NC)]) - 1)    # 收盤近似：st-1 收→ex-1 收
            g50 = float(E[(ex - 1) if ex else NC - 1] / E[st - 1] - 1)
            out.append({'entry': str(CAL[st].date()), 'exit': str(CAL[ex].date()) if ex else 'open', 'days': int((ex or NC) - st),
                        'net_pct': round(net * 100, 2), 'listedEW_pct': round(le * 100, 2), 'AR_vs_listedEW_pct': round((net - le) * 100, 2),
                        '0050_close_to_close_pct': round(g50 * 100, 2)})
        d += 1
    return out

def spell_stats(sp, a, b):
    s = [x for x in sp if a <= x['entry'] <= b and x['exit'] != 'open']
    if not s: return {'n': 0}
    df = pd.DataFrame(s); df['m'] = pd.to_datetime(df.entry).dt.to_period('M')
    g = df.groupby('m')[['net_pct', 'AR_vs_listedEW_pct']].sum()
    return {'n': int(len(df)), 'n_months': int(len(g)), 'net_mean_pct': round(float(df.net_pct.mean()), 2), 'hit': round(float((df.net_pct > 0).mean()), 3),
            't_net_month': tstat(g.net_pct), 'AR_mean_pct': round(float(df.AR_vs_listedEW_pct.mean()), 2), 't_AR_month': tstat(g.AR_vs_listedEW_pct),
            'mean_days': round(float(df.days.mean()), 1)}

def random_shift(pos, mode, buy, sell, first, a, b):
    m, _ = win(a, b, first); ii = np.where(m)[0]; p0 = pos[ii]
    act = None; vals = []
    for sh in range(len(ii)):
        p = pos.copy(); p[ii] = np.roll(p0, sh); p[ii[0] - 1] = p[ii[-1]] if sh else pos[ii[0] - 1]
        r = daily_ret(p, mode, buy, sell)
        c, _ = cagr(r[m], CAL[m]); vals.append(c)
        if sh == 0: act = c
    vals = np.array(vals)
    return {'actual_CAGR': round(act, 4), 'shift_median_CAGR': round(float(np.median(vals)), 4), 'shift_p95_CAGR': round(float(np.percentile(vals, 95)), 4),
            'pctile_of_actual': round(float((vals < act).mean()), 3), 'n_shifts': int(len(vals))}

# ---------------- 訊號資訊（扣成本前） ----------------
def info(tgt, first):
    """t 日收盤後目標=1 vs 0 時，t+1 開盤→t+2 開盤（≈下一交易日）0050 報酬，描述用"""
    nxt = np.r_[O50[2:] / O50[1:-1] - 1, np.nan, np.nan][:NC]
    res = {}
    for per, (a, b) in PER.items():
        m, _ = win(a, b, first); m = m & np.isfinite(nxt)
        x1 = nxt[m & (tgt == 1)]; x0 = nxt[m & (tgt == 0)]
        res[per] = {'next_day_0050_when_long_pct': round(float(np.mean(x1) * 100), 3) if len(x1) else None, 'n_long': int(len(x1)),
                    'next_day_0050_when_cash_pct': round(float(np.mean(x0) * 100), 3) if len(x0) else None, 'n_cash': int(len(x0))}
    return res

# ---------------- 執行 ----------------
VAR = [(f, n, k) for f in ('F1', 'F5') for n in (60, 120, 252) for k in (1.0, 1.5, 2.0)]
OUT['variants'] = {}
for f, n, k in VAR:
    tgt, first, sig = band_state(f, n, k)
    name = f'{f}_N{n}_k{k}'
    V_ = {'first_band_day': str(first.date()), 'n_up_signals': int(np.nansum(sig == 1)), 'n_down_signals': int(np.nansum(sig == 0))}
    modes = ('open', 'close') if (f, n, k) == ('F1', 120, 1.5) else ('open',)
    for mode in modes:
        pos = positions(tgt, mode)
        for cn, (bu, se) in COSTS.items():
            for cr in ((0.0, 0.01) if (cn == 'exlib_slip0.1' and mode == 'open') else (0.0,)):
                r = daily_ret(pos, mode, bu, se, cr)
                V_[f'{mode}|{cn}|cash{cr}'] = {per: evaluate(r, pos, first, a, b) for per, (a, b) in PER.items()}
        r = daily_ret(pos, mode, *COSTS['exlib_slip0.1'])
        sp = spells(r, pos, first)
        V_[f'{mode}|spells_exlib_slip0.1'] = {per: spell_stats(sp, a, b) for per, (a, b) in PER.items()}
        if (f, n, k) == ('F1', 120, 1.5) and mode == 'open':
            V_['spell_list'] = sp
            V_['random_shift_exlib'] = {per: random_shift(pos, mode, *COSTS['exlib_slip0.1'], first, a, b) for per, (a, b) in PER.items() if per != 'full'}
            V_['signal_info_gross'] = info(tgt, first)
            pd.DataFrame({'F_NTbn': F1.to_numpy() / 1e9, 'target': tgt, 'pos_open': pos, 'r_net': r, 'r_0050': r50, 'poolEW': poolEW}, index=CAL).to_csv(HERE / 'main_daily.csv.gz')
    OUT['variants'][name] = V_
    m = V_['open|exlib_slip0.1|cash0.0']
    print(name, first.date(), {p: (m[p]['CAGR'], m[p]['0050_CAGR'], m[p]['t_vsEW_m'], m[p]['t_vs0050_m'], m[p]['exposure'], m[p]['switches']) for p in ('IS_2021_2023', 'OOS_2024_2026')})

# ---------------- 判定 ----------------
mv = OUT['variants']['F1_N120_k1.5']['open|exlib_slip0.1|cash0.0']
isp, oop = mv['IS_2021_2023'], mv['OOS_2024_2026']
if min(isp['n_months'], oop['n_months']) < 12: verdict = '無法測試'
elif isp['pass'] and oop['pass']: verdict = '可用'
elif isp['pass'] or oop['pass']: verdict = '待觀察'
else: verdict = '不可用'
OUT['decision'] = {'IS': {k: isp[k] for k in ('C1_vsEW', 'C2_not_lose_0050', 'pass')}, 'OOS': {k: oop[k] for k in ('C1_vsEW', 'C2_not_lose_0050', 'pass')},
                   'verdict': verdict, 'price_split_2005_2019': 'N/A（外資資料 2021-10 起）'}
# 全格摘要
OUT['grid_summary'] = {nm: {p: {k: v[f'open|exlib_slip0.1|cash0.0'][p][k] for k in ('CAGR', '0050_CAGR', 'maxDD', 'exposure', 'switches', 't_vsEW_m', 't_vs0050_m', 'pass')}
                            for p in PER} for nm, v in OUT['variants'].items()}
(HERE / 'results.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('verdict', verdict)

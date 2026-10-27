#!/usr/bin/env python3
"""disposal_v2 主程式：依 results.json 的 PREREG 執行 a～e。只寫入本資料夾。
用法：python3 -I -B run.py
資料陣列沿用 disposal_reconcile/rlib.py（只匯入、不寫 bytecode、不改檔）。"""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'disposal_reconcile'))
import rlib as L  # noqa: E402

RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in RES, '先跑 prereg.py'

IDX = L.IDX; T = len(IDX)
FEE = 0.001425 * 0.6; ETF_BUY, ETF_SELL = FEE, FEE + 0.001
E50 = L.E50.to_numpy()
CHANGE = pd.Timestamp('2026-08-10')
P_SEL = (pd.Timestamp('2021-10-01'), pd.Timestamp('2023-12-31'))
P_VAL = (pd.Timestamp('2024-01-01'), pd.Timestamp('2026-10-22'))
P_L12 = (pd.Timestamp('2025-10-23'), pd.Timestamp('2026-10-22'))
P_ALL = (pd.Timestamp('2021-10-01'), pd.Timestamp('2026-10-22'))
W_TP = (pd.Timestamp('2025-09-24'), pd.Timestamp('2026-10-22'))

# ---------------- 加權指數：空頭開關、交易日曆 ----------------
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
tx_bear = (tx < tx.rolling(200, min_periods=200).mean())
TXI = tx.index


def bear_before(d):
    k = TXI.searchsorted(d) - 1          # 最後一個 < d 的交易日
    return bool(tx_bear.iloc[k]) if k >= 0 else False


def tdays(s, e):
    return int(TXI.searchsorted(e, 'right') - TXI.searchsorted(s))


# ---------------- disposal_today.py 的採用規則（全部記錄、日曆日） ----------------
dp = L.DP_ALL.copy()
dp = dp.sort_values(['code', 'start'], kind='stable')
take = {}
for code, g in dp.groupby('code', sort=False):
    last_end = None
    for r in g.itertuples():
        if last_end is not None and r.start <= last_end + pd.Timedelta(days=3):
            take[r.ordr] = False
        else:
            take[r.ordr] = True; last_end = r.end
TAKE = take


def build(rec, cfg, mkt):
    """cfg: chain(tool/recon), bear, liq, cost('0.8/1.2' 或數字)。回傳逐筆 DataFrame 與計數。"""
    st = {'records': 0, 'not_taken_chain': 0, 'invalid_or_incomplete': 0, 'gap': 0, 'bear': 0, 'untradable_ret9': 0,
          'illiquid': 0, 'held_busy': 0, 'kept': 0, 'delayed': 0}
    rows = []; busy = {}
    for r in rec.sort_values(['s_i', 'ann_i', 'ordr']).itertuples():
        st['records'] += 1
        if not TAKE[r.ordr]: st['not_taken_chain'] += 1; continue
        if r.s_i >= T or r.e_i <= r.s_i or r.end > IDX[-1] or r.start < P_ALL[0]: st['invalid_or_incomplete'] += 1; continue
        if r.start_in_gap or r.end_in_gap or (r.start < L.GAP[0] and r.end > L.GAP[1]): st['gap'] += 1; continue
        j = L.CI[r.code]; i0 = r.s_i
        if busy.get(r.code, -1) >= i0: st['held_busy'] += 1; continue
        if cfg['bear'] and bear_before(IDX[i0]): st['bear'] += 1; continue
        if not L.tradable({'entry': 'ret9'}, i0, j): st['untradable_ret9'] += 1; continue
        if cfg['liq'] and not L.LIQL[i0, j]: st['illiquid'] += 1; continue
        i1 = L.delay_exit({'exit': 'ret9'}, r.e_i, j)
        if not np.isfinite(L.ADJN[i1, j]): st['invalid_or_incomplete'] += 1; continue
        busy[r.code] = i1; st['kept'] += 1; st['delayed'] += int(i1 > r.e_i)
        raw = math.expm1(L.LA[i1, j] - L.LA[i0, j]); ml = math.expm1(L.LML[i1] - L.LML[i0]); m5 = math.expm1(L.LE[i1] - L.LE[i0])
        c = (0.012 if r.second else 0.008) if cfg['cost'] == '0.8/1.2' else float(cfg['cost'])
        v0 = L.VN[i0 - 1, j]
        run6 = L.ADJN[i0 - 1, j] / L.ADJN[i0 - 7, j] - 1 if i0 >= 7 else np.nan
        rows.append(dict(mkt=mkt, code=r.code, j=j, i0=i0, plan_e=r.e_i, i1=i1, ann_i=r.ann_i, ord=r.ordr, ann_date=r.date,
                         date=IDX[i0], exit=IDX[i1], hold=i1 - i0, plan_n=r.e_i - i0, second=bool(r.second),
                         len_td=tdays(r.start, r.end), raw=raw, gross=raw - ml, ab50=raw - m5, cost=c, net=raw - ml - c,
                         raw_net=raw - c, net50=raw - m5 - c, cap=0.05 * v0 if np.isfinite(v0) else 0.0, run6=run6,
                         delayed=i1 > r.e_i))
    df = pd.DataFrame(rows).sort_values(['date', 'code']).reset_index(drop=True)
    return df, st


# ---------------- 統計 ----------------
def tst(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std() > 0 else None


def S(df, col='net'):
    o = {'n': int(len(df))}
    if len(df) < 3:
        if len(df): o['mean_pct'] = round(float(df[col].mean() * 100), 2)
        return o
    x = df[col]; m = x.groupby(df['date'].dt.to_period('M')).mean()
    xs = x.sort_values(); k5 = max(1, int(math.ceil(len(xs) * 0.05)))
    o.update({'mean_pct': round(float(x.mean() * 100), 2), 'median_pct': round(float(x.median() * 100), 2),
              'hit': round(float((x > 0).mean()), 3), 't_month': tst(m), 'n_months': int(len(m)),
              'ex_top5pct_mean_pct': round(float(xs.iloc[:-k5].mean() * 100), 2)})
    return o


def sub(df, p): return df[(df['date'] >= p[0]) & (df['date'] <= p[1])]


def per(df, col='net'):
    return {'sel': S(sub(df, P_SEL), col), 'val': S(sub(df, P_VAL), col), 'all': S(sub(df, P_ALL), col),
            'last12': S(sub(df, P_L12), col), 'pre_change': S(df[df['date'] < CHANGE], col), 'post_change': S(df[df['date'] >= CHANGE], col)}


def cluster_t(df, flag, col='net'):
    """col ~ a + b·flag，月份群聚 CR1 標準誤；回傳 b、t。"""
    d = df.dropna(subset=[col]); y = d[col].to_numpy(); X = np.c_[np.ones(len(d)), d[flag].astype(float).to_numpy()]
    if X[:, 1].sum() < 2 or (1 - X[:, 1]).sum() < 2: return None
    XtX = np.linalg.inv(X.T @ X); b = XtX @ X.T @ y; e = y - X @ b
    g = d['date'].dt.to_period('M').to_numpy(); G = pd.unique(g); meat = np.zeros((2, 2))
    for k in G:
        s = (X[g == k] * e[g == k, None]).sum(0); meat += np.outer(s, s)
    n, kk, nG = len(y), 2, len(G)
    V = XtX @ meat @ XtX * (nG / (nG - 1)) * ((n - 1) / (n - kk))
    return {'diff_pct': round(float(b[1] * 100), 2), 't_cluster_month': round(float(b[1] / math.sqrt(V[1, 1])), 2), 'n_clusters': int(nG)}


# ---------------- 建事件：上市、上櫃 ----------------
L.set_market(L.TWSE)
CFG_OFF = dict(bear=True, liq=True, cost='0.8/1.2')
TW, stTW = build(L.REC, CFG_OFF, 'TWSE')
VAR = {'V_TOOL_noliq': dict(CFG_OFF, liq=False), 'V_NOBEAR': dict(CFG_OFF, bear=False)}
TWV = {k: build(L.REC, c, 'TWSE') for k, c in VAR.items()}
TW_C = {c: build(L.REC, dict(CFG_OFF, cost=c), 'TWSE')[0] for c in ('0.008', '0.012', '0.02')}
TW_AC = L.adj.ffill().to_numpy()
# reconcile 正式版（對照）
L.set_market(L.TWSE)
REC_OFF = L.trades(L.CFG_OFFICIAL); REC_OFF['raw_net'] = REC_OFF['raw'] - REC_OFF['cost']

MP = L.load_hist_market('TPEX'); recP = L.records_market(MP)
L.set_market(MP)
TP, stTP = build(recP, CFG_OFF, 'TPEX')
TP_C = {c: build(recP, dict(CFG_OFF, cost=c), 'TPEX')[0] for c in ('0.008', '0.012', '0.02')}
TPV = {k: build(recP, c, 'TPEX') for k, c in VAR.items()}
TP_AC = MP['ADJ_FFILL']
L.set_market(L.TWSE)

ALL = pd.concat([TW, TP], ignore_index=True)

# 分組
q = TW['run6'].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
def lab_run6(x):
    if not np.isfinite(x): return 'na'
    return f'Q{int(np.searchsorted(q, x, side="right")) + 1}'
def lab_len(n):
    return '5' if n == 5 else '6-7' if n <= 7 else '8-9' if n <= 9 else '10' if n == 10 else '11-12' if n <= 12 else '>=13'
for d in (TW, TP, ALL, *TW_C.values(), *TP_C.values()):
    d['g_second'] = np.where(d['second'], 'second', 'first')
    d['g_match'] = np.where(d['date'] >= CHANGE, '2min', np.where(d['second'], '20min', '5min'))
    d['g_len'] = d['len_td'].map(lab_len)
    d['g_regime'] = np.where(d['date'] >= CHANGE, 'post_2026-08-10', 'pre')
    d['g_run6'] = d['run6'].map(lab_run6)

res = {'counts': {'TWSE': stTW, 'TPEX': stTP, **{f'TWSE_{k}': v[1] for k, v in TWV.items()}, **{f'TPEX_{k}': v[1] for k, v in TPV.items()}},
       'run6_quintile_cuts_pct': [round(float(x * 100), 1) for x in q]}

# ---------------- a. 逐筆與分組 ----------------
A = {'TWSE_official': {m: per(TW, m) for m in ('net', 'raw_net', 'net50')},
     'TPEX_official': {m: per(TP, m) for m in ('net', 'raw_net', 'net50')},
     'POOLED_official': {m: per(ALL, m) for m in ('net', 'raw_net')},
     'variants_TWSE': {k: per(v[0]) for k, v in TWV.items()},
     'variants_TPEX': {k: per(v[0]) for k, v in TPV.items()},
     'reconcile_official_TWSE(merge_known, no bear)': per(REC_OFF.assign(net=REC_OFF['net']))}
A['yearly_TWSE'] = {int(y): S(g) for y, g in TW.groupby(TW['date'].dt.year)}
A['yearly_TWSE_raw_net'] = {int(y): S(g, 'raw_net') for y, g in TW.groupby(TW['date'].dt.year)}
GR = {}
for gname in ('g_second', 'g_match', 'g_len', 'g_regime', 'g_run6'):
    GR[gname] = {}
    for src, df in (('TWSE', TW), ('POOLED', ALL)):
        GR[gname][src] = {str(k): {'all': S(g), 'sel': S(sub(g, P_SEL)), 'val': S(sub(g, P_VAL)), 'last12': S(sub(g, P_L12)),
                                   'raw_net_all': S(g, 'raw_net')} for k, g in df.groupby(gname)}
# 市場：同窗格
GR['g_market_same_window'] = {m: {'net': S(sub(d, W_TP)), 'raw_net': S(sub(d, W_TP), 'raw_net'), 'net50': S(sub(d, W_TP), 'net50')} for m, d in (('TWSE', TW), ('TPEX', TP))}
# 再細分：制度變更前後 × 市場（2026 年內比較）
GR['regime_x_market_2026'] = {f'{m}|{k}': S(g) for (m, k), g in sub(ALL, (pd.Timestamp('2026-01-01'), P_VAL[1])).groupby(['mkt', 'g_regime'])}
GR['g_second_cluster_diff_TWSE'] = {'all': cluster_t(TW.assign(f=TW['second']), 'f'), 'val': cluster_t(sub(TW, P_VAL).assign(f=lambda d: d['second']), 'f')}
A['groups'] = GR
res['a'] = A

# ---------------- b. 逐日報酬曲線 ----------------
def curves(df):
    out = []
    for r in df.itertuples():
        M = L.TWSE if r.mkt == 'TWSE' else MP
        N = r.plan_n; j = r.j; i0 = r.i0
        la = M['LA'][i0:i0 + N + 1, j]; lm = M['LML'][i0:i0 + N + 1]
        ab = np.expm1(la - la[0]) - np.expm1(lm - lm[0]); rw = np.expm1(la - la[0])
        out.append({'date': r.date, 'mkt': r.mkt, 'N': N, 'ab': ab, 'raw': rw})
    return out


CV = curves(ALL)
def curve_summary(items):
    if not items: return {'n': 0}
    Ns = pd.Series([c['N'] for c in items]); N = int(Ns.mode().iloc[0])
    it = [c for c in items if c['N'] == N]
    mat = np.vstack([c['ab'] for c in it]); h = N // 2
    dates = pd.Series([c['date'] for c in it])
    early = mat[:, h]; late = mat[:, N] - mat[:, h]; exit_m2 = mat[:, N - 2] if N >= 3 else np.full(len(it), np.nan)
    mm = lambda x: pd.Series(x).groupby(dates.dt.to_period('M').to_numpy()).mean()
    o = {'N_used(交易日差)': N, 'n': int(len(it)), 'mean_cum_ab_by_day_pct': [round(float(v * 100), 2) for v in mat.mean(0)],
         'median_cum_ab_by_day_pct': [round(float(v * 100), 2) for v in np.median(mat, 0)],
         'mean_cum_raw_by_day_pct': [round(float(v * 100), 2) for v in np.vstack([c['raw'] for c in it]).mean(0)],
         'early_first_half_pct': round(float(early.mean() * 100), 2), 'early_t_month': tst(mm(early)),
         'late_second_half_pct': round(float(late.mean() * 100), 2), 'late_t_month': tst(mm(late)),
         'late_minus_early_pct': round(float((late - early).mean() * 100), 2), 'late_minus_early_t_month': tst(mm(late - early)),
         'exit_2days_early_pct': round(float(np.nanmean(exit_m2) * 100), 2), 'hold_to_end_pct': round(float(mat[:, N].mean() * 100), 2),
         'last_day_only_pct': round(float((mat[:, N] - mat[:, N - 1]).mean() * 100), 2),
         'last_day_only_t_month': tst(mm(mat[:, N] - mat[:, N - 1]))}
    le, ll = o['early_first_half_pct'], o['late_second_half_pct']
    o['verdict_late_concentration'] = ('成立' if ll > le and ll > 0 and (o['late_t_month'] or 0) >= 2 else
                                       '部分成立' if ll > le and ll > 0 else '不成立')
    return o


B = {}
sel = lambda f: [c for c in CV if f(c)]
B['all_10day_pre_change'] = curve_summary(sel(lambda c: c['N'] == 9 and c['date'] < CHANGE))
B['TWSE_10day_sel'] = curve_summary(sel(lambda c: c['N'] == 9 and c['mkt'] == 'TWSE' and c['date'] <= P_SEL[1]))
B['TWSE_10day_val_pre_change'] = curve_summary(sel(lambda c: c['N'] == 9 and c['mkt'] == 'TWSE' and P_VAL[0] <= c['date'] < CHANGE))
B['last12_10day'] = curve_summary(sel(lambda c: c['N'] == 9 and c['date'] >= P_L12[0]))
B['last12_10day_TWSE'] = curve_summary(sel(lambda c: c['N'] == 9 and c['date'] >= P_L12[0] and c['mkt'] == 'TWSE'))
B['last12_10day_TPEX'] = curve_summary(sel(lambda c: c['N'] == 9 and c['date'] >= P_L12[0] and c['mkt'] == 'TPEX'))
B['post_change_5day'] = curve_summary(sel(lambda c: c['N'] == 4 and c['date'] >= CHANGE))
B['n_events_by_N_last12'] = pd.Series([c['N'] for c in CV if c['date'] >= P_L12[0]]).value_counts().sort_index().to_dict()
res['b'] = {k: ({int(a): int(b) for a, b in v.items()} if k.startswith('n_events') else v) for k, v in B.items()}

# ---------------- c. G2 排除（上市） ----------------
rv = pd.read_csv(L.X / 'rev_5y.csv.gz', dtype={'code': str}); rv['p'] = pd.PeriodIndex(rv['ym'], freq='M')
REV = rv.pivot_table(index='p', columns='code', values='revenue', aggfunc='last')
lg = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}, parse_dates=['month'])
lg['p'] = lg['month'].dt.to_period('M')
LREV = lg.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last')
REVF = REV.combine_first(LREV)          # rev_5y 優先，缺月用長期營收（同為千元）
good = (L.adj.notna().sum(axis=1) >= 300)
AG_ = L.adj[good]; HI = AG_.rolling(252, min_periods=200).max(); S52 = (AG_ / HI)
GI = AG_.index


def g2_info(r):
    d = IDX[r.i0 - 1]; k = GI.searchsorted(d, 'right') - 1; dd = GI[k]
    s52 = S52.iat[k, S52.columns.get_loc(r.code)] if r.code in S52.columns else np.nan
    m = pd.Period(d, 'M') - (1 if d.day >= 11 else 2)
    ms = [m - 2, m - 1, m]; mp = [x - 12 for x in ms]
    if r.code in REVF.columns:
        a = REVF[r.code].reindex(ms); b = REVF[r.code].reindex(mp)
        y3 = a.sum() / b.sum() - 1 if a.notna().all() and b.notna().all() and b.sum() > 0 else np.nan
    else:
        y3 = np.nan
    return s52, y3


g = [g2_info(r) for r in TW.itertuples()]
for d in (TW, *TW_C.values()):
    d['s52'] = [x[0] for x in g]; d['yoy3'] = [x[1] for x in g]
    d['G2'] = (d['yoy3'] < 0) & (d['s52'] < 0.8)
C = {'missing': {'s52_na': int(TW['s52'].isna().sum()), 'yoy3_na': int(TW['yoy3'].isna().sum())}}
P_SEL_G2 = (pd.Timestamp('2021-01-01'), pd.Timestamp('2023-12-31'))
for pn, p in (('sel', P_SEL_G2), ('val', P_VAL), ('all', P_ALL), ('last12', P_L12)):
    d = sub(TW, p)
    C[pn] = {'all': S(d), 'G2': S(d[d['G2']]), 'nonG2': S(d[~d['G2']]), 'diff_cluster': cluster_t(d.assign(f=d['G2']), 'f'),
             'G2_raw_net': S(d[d['G2']], 'raw_net'), 'nonG2_raw_net': S(d[~d['G2']], 'raw_net'),
             'share_G2': round(float(d['G2'].mean()), 3)}
def okp(c): return (c['G2'].get('mean_pct') is not None and c['nonG2'].get('mean_pct') is not None
                    and c['G2']['mean_pct'] < c['nonG2']['mean_pct'] and c['G2']['mean_pct'] < 0)
vt = (C['val']['diff_cluster'] or {}).get('t_cluster_month')
C['decision'] = ('通過' if okp(C['sel']) and okp(C['val']) and vt is not None and vt <= -2 else
                 '待觀察（不採用）' if okp(C['sel']) else '不採用')
C['G2_by_year'] = {int(y): {'G2': S(gg[gg['G2']]), 'nonG2': S(gg[~gg['G2']])} for y, gg in TW.groupby(TW['date'].dt.year)}
C['G2_events'] = [{'code': r.code, 'date': str(r.date.date()), 'yoy3_pct': round(r.yoy3 * 100, 1), 's52': round(r.s52, 3), 'net_pct': round(r.net * 100, 2)}
                  for r in TW[TW['G2']].itertuples()]
res['c'] = C


# ---------------- 10 槽 ----------------
class Sleeve:
    def __init__(self, ev, K=10, idle='0050'):
        self.ev = {}
        for e in ev: self.ev.setdefault(e['i0'], []).append(e)
        for k in self.ev: self.ev[k].sort(key=lambda e: (e['ann_i'], e['ord']))
        self.K, self.idle = K, idle; self.cash = 0.0; self.u50 = 0.0; self.pos = {}; self.trades = []
        self.stat = {'entered': 0, 'skipped_no_slot': 0, 'capped': 0}
    def ac(self, m, t, j): return (TW_AC if m == 'TWSE' else TP_AC)[t, j]
    def equity(self, t): return self.cash + self.u50 * E50[t] + sum(p['u'] * self.ac(k[0], t, k[1]) for k, p in self.pos.items())
    def deposit(self, x, t):
        if self.idle == 'cash': self.cash += x
        else: self.u50 += x * (1 - ETF_BUY) / E50[t]
    def withdraw_idle(self, x, t):
        avail = self.cash if self.idle == 'cash' else self.u50 * E50[t] * (1 - ETF_SELL)
        x = min(x, max(avail, 0.0))
        if self.idle == 'cash': self.cash -= x
        else: self.u50 -= x / (1 - ETF_SELL) / E50[t]
        return x
    def step(self, t):
        for k in [k for k, p in self.pos.items() if p['i1'] == t]:
            p = self.pos.pop(k); pr = p['u'] * self.ac(k[0], t, k[1]) * (1 - p['hc'])
            self.trades.append({'date': IDX[p['t0']], 'net_trade': pr / p['amt'] - 1, 'mkt': k[0], 'code': p['code']}); self.deposit(pr, t)
        for e in self.ev.get(t, []):
            key = (e['mkt'], e['j'])
            if key in self.pos: continue
            if len(self.pos) >= self.K: self.stat['skipped_no_slot'] += 1; continue
            E = self.equity(t); amt = E / self.K
            if e['cap'] < amt: amt = e['cap']; self.stat['capped'] += 1
            amt = self.withdraw_idle(amt, t)
            if amt <= 0: continue
            self.pos[key] = {'u': amt * (1 - e['cost'] / 2) / self.ac(e['mkt'], t, e['j']), 'i1': e['i1'], 'hc': e['cost'] / 2, 't0': t, 'amt': amt, 'code': e['code']}
            self.stat['entered'] += 1


def run_sleeve(df, idle='0050', start='2021-09-30', end=None, K=10):
    ev = df.to_dict('records'); s = Sleeve(ev, K, idle)
    t0 = IDX.get_loc(pd.Timestamp(start)) if pd.Timestamp(start) in IDX else IDX.searchsorted(pd.Timestamp(start))
    t1 = T - 1 if end is None else IDX.searchsorted(pd.Timestamp(end), 'right') - 1
    s.deposit(1.0, t0); nav = []; held = []
    for t in range(t0 + 1, t1 + 1):
        s.step(t); nav.append(s.equity(t)); held.append(len(s.pos))
    nav = pd.Series([1.0] + nav, index=IDX[t0:t1 + 1])
    tr = pd.DataFrame(s.trades)
    return nav, s, tr, np.array(held)


REB = []
_s = pd.Series(IDX[IDX >= pd.Timestamp('2021-03-01')], index=IDX[IDX >= pd.Timestamp('2021-03-01')])
for _, gg in _s.groupby([_s.index.year, _s.index.month]):
    x = gg[gg.dt.day >= 11]
    if len(x): REB.append(x.iloc[0])
REB = pd.DatetimeIndex(REB)
E50S = pd.Series(E50, index=IDX)


def nstats(nav, a=None, b=None):
    nav = nav.dropna()
    if a is not None: nav = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    if len(nav) < 30: return None
    r = nav.pct_change().dropna(); yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    eq = nav / nav.iloc[0]
    o = {'start': str(nav.index[0].date()), 'end': str(nav.index[-1].date()), 'CAGR_pct': round(float((eq.iloc[-1] ** (1 / yrs) - 1) * 100), 1),
         'total_pct': round(float((eq.iloc[-1] - 1) * 100), 1), 'maxDD_pct': round(float((eq / eq.cummax() - 1).min() * 100), 1),
         'vol_pct': round(float(r.std() * math.sqrt(252) * 100), 1)}
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}
    b5 = E50S.reindex(nav.index); e5 = b5 / b5.iloc[0]
    o['0050_CAGR_pct'] = round(float((e5.iloc[-1] ** (1 / yrs) - 1) * 100), 1); o['0050_maxDD_pct'] = round(float((e5 / e5.cummax() - 1).min() * 100), 1)
    o['0050_yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in b5.pct_change().dropna().groupby(b5.index[1:].year)}
    rb = [d for d in REB if nav.index[0] <= d <= nav.index[-1]]
    rb = pd.DatetimeIndex(([nav.index[0]] if rb and rb[0] != nav.index[0] else []) + rb + ([nav.index[-1]] if rb and rb[-1] != nav.index[-1] else []))
    pr = nav.reindex(rb).pct_change().dropna(); pb = b5.reindex(rb).pct_change().dropna()
    ex = pr - pb
    o['n_months'] = int(len(ex)); o['excess_vs_0050_m_pct'] = round(float(ex.mean() * 100), 2); o['t_vs_0050_m'] = tst(ex)
    return o


SLV = {}
NAVS = {}
for idle in ('0050', 'cash'):
    nav, s, tr, held = run_sleeve(TW, idle)
    NAVS[f'TWSE_K10_{idle}'] = nav
    SLV[f'TWSE_K10_idle_{idle}'] = {'full': nstats(nav), 'sel': nstats(nav, '2021-09-30', '2023-12-31'), 'val': nstats(nav, '2023-12-29', '2026-10-22'),
                                    'last12': nstats(nav, '2025-10-22', '2026-10-22'), 'stat': s.stat, 'mean_held': round(float(held.mean()), 2),
                                    'share_days_full': round(float((held >= 10).mean()), 3),
                                    'trades': S(tr.rename(columns={'net_trade': 'net'})) if len(tr) else None}
for c, d in TW_C.items():
    nav, s, tr, held = run_sleeve(d, '0050'); NAVS[f'TWSE_K10_0050_cost{c}'] = nav
    SLV[f'TWSE_K10_idle_0050_cost{c}'] = {'full': nstats(nav), 'val': nstats(nav, '2023-12-29', '2026-10-22'), 'last12': nstats(nav, '2025-10-22', '2026-10-22')}
for k, (d, _) in TWV.items():
    nav, s, tr, held = run_sleeve(d, '0050'); NAVS[f'TWSE_K10_0050_{k}'] = nav
    SLV[f'TWSE_K10_idle_0050_{k}'] = {'full': nstats(nav), 'val': nstats(nav, '2023-12-29', '2026-10-22')}
# G2 排除版（不論是否通過都報告）
nav, s, tr, held = run_sleeve(TW[~TW['G2']], '0050'); NAVS['TWSE_K10_0050_exG2'] = nav
SLV['TWSE_K10_idle_0050_exG2'] = {'full': nstats(nav), 'sel': nstats(nav, '2021-09-30', '2023-12-31'), 'val': nstats(nav, '2023-12-29', '2026-10-22')}
# 上市＋上櫃（2025-09-24 起）
both = pd.concat([sub(TW, W_TP), TP], ignore_index=True)
for idle in ('0050', 'cash'):
    nav, s, tr, held = run_sleeve(both, idle, start='2025-09-24'); NAVS[f'BOTH_K10_{idle}'] = nav
    SLV[f'TWSE+TPEX_K10_idle_{idle}_2025-09-24'] = {'full': nstats(nav), 'stat': s.stat, 'mean_held': round(float(held.mean()), 2),
                                                    'share_days_full': round(float((held >= 10).mean()), 3),
                                                    'trades': S(tr.rename(columns={'net_trade': 'net'})) if len(tr) else None}
    nav, s, tr, held = run_sleeve(sub(TW, W_TP), idle, start='2025-09-24')
    SLV[f'TWSE_only_K10_idle_{idle}_2025-09-24'] = {'full': nstats(nav)}
res['sleeve'] = SLV

# ---------------- d. 與主策略 n10 合併 ----------------
MN = pd.read_parquet(AG / 'exec_rebuild' / 'main_nav.parquet')['M2_n10_slip0.001'].dropna()
MB, MS = FEE + 0.001, FEE + 0.003 + 0.001


def combo(wd, disp_df, idle='0050'):
    ev = disp_df.to_dict('records'); s = Sleeve(ev, 10, idle)
    t0 = IDX.get_loc(MN.index[0]); t1 = IDX.get_loc(MN.index[-1]); mn = MN.reindex(IDX[t0:t1 + 1]).ffill().to_numpy(); mn = mn / mn[0]
    mval = (1 - wd) * mn[0]; s.deposit(wd, t0); out = [mval + s.equity(t0)]; rebs = set(REB)
    for t in range(t0 + 1, t1 + 1):
        mval *= mn[t - t0] / mn[t - t0 - 1]; s.step(t)
        E = mval + s.equity(t)
        if IDX[t] in rebs and wd > 0:
            Td = wd * E; dcur = s.equity(t)
            if Td > dcur:
                need = Td - dcur; mval -= need / (1 - MS); s.deposit(need, t)
            else:
                x = s.withdraw_idle(dcur - Td, t); mval += x * (1 - MB)
        out.append(mval + s.equity(t))
    return pd.Series(out, index=IDX[t0:t1 + 1])


D = {}
for w in (0.0, 0.1, 0.2, 0.3):
    for lab, df in (('official', TW), ('cost0.012', TW_C['0.012']), ('cost0.02', TW_C['0.02'])):
        if lab != 'official' and w not in (0.2,): continue
        nav = combo(w, df); NAVS[f'combo_n10_{int(100-w*100)}_{int(w*100)}_{lab}'] = nav
        D[f'main{int(100 - w * 100)}/disp{int(w * 100)}|{lab}'] = {'full_2021-03': nstats(nav), 'from_2021-10': nstats(nav, '2021-09-30', '2026-10-12'),
                                                                  'sel_2021-03_2023': nstats(nav, '2021-03-11', '2023-12-31'),
                                                                  'val_2024_2026-10': nstats(nav, '2023-12-29', '2026-10-12'),
                                                                  'val_ex2026': nstats(nav, '2023-12-29', '2025-12-31')}
nav = combo(0.2, TW, 'cash'); D['main80/disp20|official_idle_cash'] = {'full_2021-03': nstats(nav), 'val_2024_2026-10': nstats(nav, '2023-12-29', '2026-10-12')}
res['d'] = D

# ---------------- e. 成本敏感度（逐筆） ----------------
res['e'] = {'TWSE': {c: per(d) for c, d in TW_C.items()}, 'TPEX': {c: per(d) for c, d in TP_C.items()},
            'sleeve_TWSE_0050': {c: SLV[f'TWSE_K10_idle_0050_cost{c}'] for c in TW_C}}

# ---------------- 右尾、資料檢查 ----------------
res['tails'] = {'TWSE_top10': [{'code': r.code, 'date': str(r.date.date()), 'exit': str(r.exit.date()), 'net_pct': round(r.net * 100, 1)} for r in TW.nlargest(10, 'net').itertuples()],
                'TWSE_bottom5': [{'code': r.code, 'date': str(r.date.date()), 'exit': str(r.exit.date()), 'net_pct': round(r.net * 100, 1)} for r in TW.nsmallest(5, 'net').itertuples()],
                'TPEX_top5': [{'code': r.code, 'date': str(r.date.date()), 'net_pct': round(r.net * 100, 1)} for r in TP.nlargest(5, 'net').itertuples()],
                'corp_flag_3_4_5_in_hold_TWSE': int(sum(bool(np.isin(L.FN[r.i0 + 1:r.i1 + 1, r.j], [3, 4, 5]).any()) for r in TW.itertuples())),
                'src3_share_TWSE': round(float(np.mean([L.SN[r.i0, r.j] == 3 or L.SN[r.i1, r.j] == 3 for r in TW.itertuples()])), 3)}
res['monthly_TWSE'] = {str(k): {'n': int(len(g)), 'mean_net_pct': round(float(g['net'].mean() * 100), 2)} for k, g in TW.groupby(TW['date'].dt.to_period('M'))}
res['monthly_TPEX'] = {str(k): {'n': int(len(g)), 'mean_net_pct': round(float(g['net'].mean() * 100), 2)} for k, g in TP.groupby(TP['date'].dt.to_period('M'))}

# ---------------- 判定（依 PREREG） ----------------
tw = A['TWSE_official']['net']; t12 = per(TW_C['0.012'])
cond_ok = (tw['sel'].get('t_month') or -9) >= 2 and (tw['val'].get('t_month') or -9) >= 2 and tw['last12'].get('mean_pct', -1) > 0 and (t12['val'].get('t_month') or -9) >= 2
cond_watch = (tw['val'].get('t_month') or -9) >= 2 and tw['last12'].get('mean_pct', -1) > 0
verdict = '可用' if cond_ok else '待觀察' if cond_watch else '不可用'
post = A['POOLED_official']['net']['post_change'].get('mean_pct', -1)
l12c = res['b']['last12_10day']['verdict_late_concentration']
cap = 30 if verdict == '可用' else (10 if (post <= 0 or l12c == '不成立') else 20) if verdict == '待觀察' else 0
res['DECISION'] = {'verdict': verdict, 'cap_pct': cap, 'inputs': {'sel_t': tw['sel'].get('t_month'), 'val_t': tw['val'].get('t_month'), 'last12_mean': tw['last12'].get('mean_pct'),
                   'val_t_cost1.2': t12['val'].get('t_month'), 'post_change_pooled_mean': post, 'last12_late_concentration': l12c}, 'G2': C['decision']}

# ---------------- 輸出 ----------------
def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Period)): return str(o)
    return str(o)


RES.update({'RESULTS': res})
(HERE / 'results.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
pd.DataFrame(NAVS).to_parquet(HERE / 'navs.parquet')
out = ALL.drop(columns=['j']).copy()
out = out.merge(TW[['code', 'i0', 's52', 'yoy3', 'G2']], on=['code', 'i0'], how='left') if 'G2' in TW else out
out.to_csv(HERE / 'trades_official.csv.gz', index=False)
print(json.dumps(res['DECISION'], ensure_ascii=False))

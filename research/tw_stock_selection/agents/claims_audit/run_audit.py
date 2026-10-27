#!/usr/bin/env python3
"""claims_audit 主程式：網路宣稱績效逐步拆解（瀑布）。規則與步驟見 results.json PREREG（prereg.py 事前寫入）。
資料、成本、漲停判定、60 日均值修補從 revenue_longhistory/lhlib.py 匯入（只讀取）。執行：python3 -I -B run_audit.py"""
import json, math, sys, time, re
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'revenue_longhistory'))
import lhlib as LH  # noqa: E402
T0 = time.time()
IDX, COLS, N = LH.IDX, LH.COLS, LH.N
tstat = LH.tstat
GOOD = LH.GOOD.to_numpy()
GI = np.where(GOOD)[0]
NEXT = np.full(len(IDX), -1)
for a, b in zip(GI[:-1], GI[1:]): NEXT[a] = b
ADJ_RAW, OPEN_RAW = LH.ADJ_RAW, LH.OPEN_RAW
NOT91 = np.array([not c.startswith('91') for c in COLS])
SURV = np.isfinite(ADJ_RAW[-5:]).any(axis=0)

# ---- 價格（含 0050 欄）----
E50 = LH.E50.ffill()
J50 = N
AC2 = np.hstack([LH.AC, E50.to_numpy()[:, None]])
AO2 = np.hstack([LH.AO, E50.shift(1).bfill().to_numpy()[:, None]])
TX = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index().reindex(IDX, method='ffill')

# ---- 漲停 ----
LIM = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
RT = LH.RET.to_numpy()
HI = LH.H.to_numpy(); CR = LH.C.to_numpy()
LOCKUP_CLOSE = (RT >= LIM - 0.006) & (CR >= HI * 0.999)
LOCKDN_CLOSE = RT <= -(LIM - 0.006)
GAPUP = LH.GAP_UP_OPEN_ERA

# ---- 訊號用滾動值（剔除稀疏日）----
_ag = LH.adj.loc[LH.GOOD]
MA = {n: _ag.rolling(n).mean().reindex(IDX).to_numpy() for n in (5, 10, 20, 60)}
ADJF = LH.adj.to_numpy()
V60 = LH.v60_fix.to_numpy()
LIQ = (V60 >= 1e7) & (CR >= 10) & np.isfinite(ADJF) & NOT91[None, :]

# ---- 營收 ----
YOY = LH.yoy_long().reindex(columns=COLS)
d0 = pd.read_csv(LH.LONG, dtype={'code': str, 'market': str, 'name': str})
d0['p'] = pd.PeriodIndex(pd.to_datetime(d0['month']), freq='M')
names_all = d0.groupby('code')['name'].apply(lambda s: '|'.join(sorted(set(s.dropna()))))
d0 = d0.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
d0 = d0[d0['code'].isin(COLS)]
REV = d0.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
REV = REV.reindex(pd.period_range(REV.index.min(), REV.index.max(), freq='M')).reindex(columns=COLS)
REV = REV.where(REV > 0)
cnt_prior = REV.notna().cumsum().shift(1)
hm_all = REV.cummax().ffill().shift(1)
RECORD = (REV.notna() & (cnt_prior >= 36) & (REV > hm_all))
RECBASE = REV.notna() & (cnt_prior >= 36)
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str}).drop_duplicates('code').set_index('code')
KY = np.array(['KY' in (str(sl['name'].get(c, '')) + '|' + str(names_all.get(c, ''))) for c in COLS])

# ---- 月頻日曆 ----
s = pd.Series(np.arange(len(IDX)), index=IDX)[GOOD]
MD = [int(g[g.index.day >= 11].iloc[0]) for _, g in s.groupby([s.index.year, s.index.month]) if (g.index.day >= 11).any()]
MD = [i for i in MD if IDX[i] >= pd.Timestamp('2006-02-01') and NEXT[i] >= 0]
def tm(i): return pd.Period(IDX[i], 'M') - 1
IFRS_T = (pd.Period('2013-01', 'M'), pd.Period('2014-03', 'M'))
def is_ifrs(i): return IFRS_T[0] <= tm(i) <= IFRS_T[1]
def ym(df, i, lag=0):
    p = tm(i) - lag
    return df.loc[p].to_numpy(dtype=float) if p in df.index else np.full(N, np.nan)

# ---- 週頻日曆（同 w3_weekly_reversal）----
gidx = IDX[GOOD]
wk = gidx.to_period('W-SUN')
WD = [int(GI[x]) for x in pd.Series(np.arange(len(gidx)), index=gidx).groupby(wk).max().to_numpy()]
WD = [i for i in WD if IDX[i] >= pd.Timestamp('2005-01-01') and NEXT[i] >= 0]
_adjG = LH.adj.loc[LH.GOOD]
R5 = (_adjG / _adjG.shift(5) - 1).reindex(IDX).to_numpy()
FLG = pd.read_parquet(LH.P / 'flag.parquet').reindex(index=IDX, columns=COLS)
mkt4 = ((FLG == 4).sum(axis=1) >= 50).to_numpy()
FL = (FLG.isin([3, 4, 5]).to_numpy() & ~((FLG == 4).to_numpy() & mkt4[:, None])).astype(float)
FLW = pd.DataFrame(FL[GOOD]).rolling(6, min_periods=1).max().to_numpy() > 0
FLWfull = np.zeros((len(IDX), N), bool); FLWfull[GOOD] = FLW

# ---- 規則：回傳 (pool 遮罩, 候選排序 or 遮罩, 是否排名型) ----
def rule_R1(ver):
    def f(i):
        y0 = ym(YOY, i); ma60 = MA[60][i]
        pool = LIQ[i] & np.isfinite(y0) & np.isfinite(ma60)
        sig = pool & (y0 > 20)
        if ver in ('b', 'c'): sig &= ADJF[i] > ma60
        if ver == 'c':
            y1 = ym(YOY, i, 1); y2 = ym(YOY, i, 2)
            sig &= np.isfinite(y1) & np.isfinite(y2) & (y2 < y1) & (y1 < y0)
        return pool, sig, None
    return f
def rule_R2(i):
    pool = LIQ[i] & ym(RECBASE.astype(float), i).astype(bool) if tm(i) in RECBASE.index else np.zeros(N, bool)
    sig = pool & (ym(RECORD.astype(float), i) > 0)
    return pool, sig, None
def rule_R3(i):
    y0 = ym(YOY, i)
    pool = LIQ[i] & ~KY & np.isfinite(y0) & np.isfinite(MA[60][i])
    return pool, None, ('R3', y0)
def score_R3(pool, aux, i):
    y0 = aux[1]
    pct = pd.Series(np.where(pool, y0, np.nan)).rank(pct=True).to_numpy()
    bull = (MA[5][i] > MA[10][i]) & (MA[10][i] > MA[20][i]) & (MA[20][i] > MA[60][i])
    sc = 50 * bull + 50 * pct
    sc[~pool | (sc < 60) | ~np.isfinite(sc)] = np.nan
    order = np.argsort(-np.nan_to_num(sc, nan=-1e9), kind='stable')
    return [j for j in order if np.isfinite(sc[j])], 5
def rule_R5(i):
    pool = LIQ[i] & np.isfinite(R5[i]) & ~LOCKUP_CLOSE[i] & ~LOCKDN_CLOSE[i] & ~FLWfull[i]
    return pool, None, ('R5', None)
def score_R5(pool, aux, i):
    sc = np.where(pool, R5[i], np.inf)
    n = int(round(0.1 * pool.sum()))
    order = np.argsort(sc, kind='stable')[:int(pool.sum())]
    return list(order), n

RULES = {
 'R1a': dict(f=rule_R1('a'), cal=MD, monthly=True, claim=[('2006-10-01', '2018-09-30')], oos={'post_publication': [('2018-10-01', '2026-10-31')]}),
 'R1b': dict(f=rule_R1('b'), cal=MD, monthly=True, claim=[('2006-10-01', '2018-09-30')], oos={'post_publication': [('2018-10-01', '2026-10-31')]}),
 'R1c': dict(f=rule_R1('c'), cal=MD, monthly=True, claim=[('2006-10-01', '2018-09-30')], oos={'post_publication': [('2018-10-01', '2026-10-31')]}),
 'R2': dict(f=rule_R2, cal=[i for i in MD if IDX[i] >= pd.Timestamp('2008-02-01')], monthly=True, claim=[('2008-02-01', '2018-09-30')], oos={'post_publication': [('2018-10-01', '2026-10-31')]}),
 'R3': dict(f=rule_R3, cal=[i for i in MD if IDX[i] >= pd.Timestamp('2006-03-01')], monthly=True, claim=[('2012-06-01', '2022-05-31')],
            oos={'pre+post': [('2006-03-01', '2012-05-31'), ('2022-06-01', '2026-10-31')], 'pre_2006-03_2012-05': [('2006-03-01', '2012-05-31')], 'post_2022-06_2026-10': [('2022-06-01', '2026-10-31')]}),
 'R5': dict(f=rule_R5, cal=WD, monthly=False, claim=[('2005-01-01', '2019-04-30')], oos={'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')]}),
}
SCORERS = {'R3': score_R3, 'R5': score_R5}

def tranche(pick):
    k = len(pick)
    if k >= 5: return np.array(pick, dtype=int), np.full(k, 1.0 / k)
    return np.array(list(pick) + [J50], dtype=int), np.r_[np.full(k, 0.2), (5 - k) * 0.2]

def build(rule, mode, surv_only, limitup):
    R = RULES[rule]; ev, evw, recs = {}, {}, []; held = set(); nblk = 0
    for i in R['cal']:
        ex = i if mode == 'close' else NEXT[i]
        if R['monthly'] and is_ifrs(i):
            ev[ex] = (np.array([J50]), np.array([1.0]), mode); evw[ex] = ev[ex]; recs.append((i, ex, [], True)); held = set(); continue
        pool, sig, aux = R['f'](i)
        pool = pool.copy()
        if surv_only: pool &= SURV
        trad = np.isfinite(ADJ_RAW[ex]) & (np.isfinite(OPEN_RAW[ex]) if mode == 'open' else True)
        pool &= trad
        if pool.sum() < (100 if not R['monthly'] else 30):
            continue
        blk = (GAPUP[ex] if mode == 'open' else LOCKUP_CLOSE[ex]) if limitup else np.zeros(N, bool)
        if aux is None:
            cand = np.where(pool & sig)[0]
            pick = [j for j in cand if not (blk[j] and j not in held)]
            nblk += len(cand) - len(pick)
        else:
            order, n = SCORERS[aux[0]](pool, aux, i)
            pick = []
            for j in order:
                if blk[j] and j not in held: nblk += 1; continue
                pick.append(j)
                if len(pick) == n: break
        ev[ex] = (*tranche(pick), mode)
        pp = np.where(pool)[0]; evw[ex] = (pp, np.full(len(pp), 1.0 / len(pp)), mode)
        recs.append((i, ex, pick, False)); held = set(pick)
    return ev, evw, recs, nblk

def sim_w(events, buy, sell, t0, t1):
    NN = N + 1; u = np.zeros(NN); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            pk, w, mode = events[t]
            Pz = AC2[t] if mode == 'close' else AO2[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash; Vn = Vt; tgt = np.zeros(NN)
            for _ in range(6):
                tgt[:] = 0; tgt[pk] = w * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(NN); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * AC2[t]) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')

def in_w(dt, segs): return any(pd.Timestamp(a) <= dt <= pd.Timestamp(b) for a, b in segs)

def evaluate(recs, nav, ew, segs, buy, sell, mode):
    """期間報酬：訊號日 i_k 收盤 → i_{k+1} 收盤（close 模式即成交價；open 模式同中心慣例）。"""
    sig_i = [r[0] for r in recs]; out = {}
    rows = []
    for k in range(len(recs) - 1):
        i0, ex0, pick, blocked = recs[k]; i1, ex1 = recs[k + 1][0], recs[k + 1][1]
        if blocked or not in_w(IDX[i0], segs): continue
        a, b = IDX[i0], IDX[i1]
        rows.append({'d': a, 'days': (b - a).days, 's': nav[b] / nav[a] - 1, 'e': ew[b] / ew[a] - 1,
                     'm50': E50.iloc[i1] / E50.iloc[i0] - 1, 'tx': TX.iloc[i1] / TX.iloc[i0] - 1,
                     'npick': len(pick)})
        P0 = AC2[ex0] if mode == 'close' else AO2[ex0]; P1 = AC2[ex1] if mode == 'close' else AO2[ex1]
        for j in pick:
            r = P1[j] / P0[j] - 1
            if np.isfinite(r): rows[-1].setdefault('tr', []).append((1 + r) * (1 - sell) / (1 + buy) - 1)
    df = pd.DataFrame(rows)
    if df.empty: return None
    yrs = df.days.sum() / 365.25
    cg = lambda x: float((1 + x).prod() ** (1 / yrs) - 1)
    mon = df.groupby(df.d.dt.to_period('M')).agg({'s': lambda x: (1 + x).prod() - 1, 'e': lambda x: (1 + x).prod() - 1, 'm50': lambda x: (1 + x).prod() - 1, 'tx': lambda x: (1 + x).prod() - 1})
    tr = np.array([x for L_ in df.get('tr', pd.Series(dtype=object)).dropna() for x in L_])
    win = tr[tr > 0]; los = tr[tr <= 0]
    out = {'years': round(yrs, 2), 'n_periods': int(len(df)), 'n_months': int(len(mon)), 'CAGR': round(cg(df.s), 4), 'EW_CAGR': round(cg(df.e), 4),
           '0050_CAGR': round(cg(df.m50), 4), 'TAIEX_price_CAGR': round(cg(df.tx), 4),
           'excess_vs_EW_CAGR': round(cg(df.s) - cg(df.e), 4), 'excess_vs_0050_CAGR': round(cg(df.s) - cg(df.m50), 4),
           't_vs_EW_m': tstat(mon.s - mon.e), 't_vs_0050_m': tstat(mon.s - mon.m50),
           'hit_vs_EW_m': round(float((mon.s > mon.e).mean()), 3), 'hit_vs_TAIEX_m': round(float((mon.s > mon.tx).mean()), 3),
           'n_trades': int(len(tr)), 'trade_win_rate': round(float((tr > 0).mean()), 3) if len(tr) else None,
           'trade_mean_pct': round(float(tr.mean() * 100), 2) if len(tr) else None, 'trade_median_pct': round(float(np.median(tr) * 100), 2) if len(tr) else None,
           'win_loss_ratio': round(float(win.mean() / -los.mean()), 2) if len(win) and len(los) and los.mean() < 0 else None,
           'avg_picks': round(float(df.npick.mean()), 1)}
    ys = df.groupby(df.d.dt.year).s.apply(lambda x: (1 + x).prod() - 1)
    out['yearly_pct'] = {int(k): round(float(v) * 100, 1) for k, v in ys.items()}
    lg = np.log1p(ys); top2 = lg.sort_values(ascending=False).iloc[:2]
    out['best2_years'] = {int(k): round(float(np.expm1(v)) * 100, 1) for k, v in top2.items()}
    rest = lg.drop(top2.index)
    out['CAGR_ex_best2_years'] = round(float(np.exp(rest.sum() / max(yrs - 2, 0.5)) - 1), 4) if len(rest) else None
    return out

STEPS = [('a', 'close', True, False, False), ('b', 'open', True, False, False), ('c', 'open', False, False, False),
         ('d', 'open', False, True, False), ('e', 'open', False, True, True)]
ONE = [('only_open', 'open', True, False, False), ('only_delisted', 'close', False, False, False),
       ('only_cost', 'close', True, True, False), ('only_limitup', 'close', True, False, True)]
BUY, SELL = LH.cost_rates(0.001); SB, SS = LH.cost_rates('stress')

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
res['meta'] = {'n_survivors': int(SURV.sum()), 'n_codes': int(N), 'n_KY': int(KY.sum()), 'cost_buy': BUY, 'cost_sell': SELL,
               'monthly_signals': len(MD), 'weekly_signals': len(WD), 'first_month': str(IDX[MD[0]].date()), 'first_week': str(IDX[WD[0]].date())}
OUT = {}
for rule, R in RULES.items():
    t0 = min(R['cal']); t1 = len(IDX) - 1
    o = {'steps': {}, 'one_at_a_time_from_a': {}}
    cache = {}
    def run(mode, surv, cost, lim, b=BUY, s_=SELL):
        key = (mode, surv, lim)
        if key not in cache: cache[key] = build(rule, mode, surv, lim)
        ev, evw, recs, nblk = cache[key]
        bb, ss = (b, s_) if cost else (0.0, 0.0)
        nav, tov = sim_w(ev, bb, ss, t0, t1)
        ewk = (mode, surv)
        if ('ew',) + ewk not in cache: cache[('ew',) + ewk] = sim_w(evw, 0.0, 0.0, t0, t1)[0]
        return recs, nav, cache[('ew',) + ewk], tov, nblk, bb, ss
    for name, mode, surv, cost, lim in STEPS:
        recs, nav, ew, tov, nblk, bb, ss = run(mode, surv, cost, lim)
        o['steps'][name] = {'claim_window': evaluate(recs, nav, ew, R['claim'], bb, ss, mode), 'turnover': round(tov, 3), 'n_blocked': int(nblk)}
        if name == 'e':
            o['steps']['f'] = {k: evaluate(recs, nav, ew, sg, bb, ss, mode) for k, sg in R['oos'].items()}
            o['steps']['e']['full_2005_2026'] = evaluate(recs, nav, ew, [('2005-01-01', '2026-10-31')], bb, ss, mode)
            o['steps']['e']['center_select_2005-01_2019-04'] = evaluate(recs, nav, ew, [('2005-01-01', '2019-04-30')], bb, ss, mode)
            o['steps']['e']['center_valid_2019-05_2026-10'] = evaluate(recs, nav, ew, [('2019-05-01', '2026-10-31')], bb, ss, mode)
            recs2, nav2, ew2, _, _, b2, s2 = run(mode, surv, True, lim, SB, SS)
            o['steps']['e_stress'] = {'claim_window': evaluate(recs2, nav2, ew2, R['claim'], b2, s2, mode), **{k: evaluate(recs2, nav2, ew2, sg, b2, s2, mode) for k, sg in R['oos'].items()}}
        if name == 'c':  # 也報 (c) 在樣本外，方便拆「期間」本身
            o['steps']['c']['oos'] = {k: evaluate(recs, nav, ew, sg, bb, ss, mode) for k, sg in R['oos'].items()}
    for name, mode, surv, cost, lim in ONE:
        recs, nav, ew, tov, nblk, bb, ss = run(mode, surv, cost, lim)
        o['one_at_a_time_from_a'][name] = evaluate(recs, nav, ew, R['claim'], bb, ss, mode)
    OUT[rule] = o
    st = o['steps']
    def fm(x): return f"{x['CAGR']*100:6.1f}% EW {x['EW_CAGR']*100:5.1f}% ex {x['excess_vs_EW_CAGR']*100:5.1f} t {x['t_vs_EW_m']:5.2f} win {x['trade_win_rate']}"
    print(rule, round(time.time() - T0), flush=True)
    for k in 'abcde': print('  ', k, fm(st[k]['claim_window']), 'tov', st[k]['turnover'], 'blk', st[k]['n_blocked'])
    for k, v in st['f'].items(): print('   f', k, fm(v), '0050', v['0050_CAGR'])
    for k, v in o['one_at_a_time_from_a'].items(): print('   1x', k, fm(v))

# ---- 附加證據：市場本身的存活者偏誤（同池等權 = 共同股池）----
def ew_market(surv, mode='open', segs=[('2005-01-01', '2019-04-30')]):
    ev = {}; recs = []
    for i in MD:
        ex = NEXT[i] if mode == 'open' else i
        pool = LIQ[i].copy()
        if surv: pool &= SURV
        pool &= np.isfinite(ADJ_RAW[ex]) & np.isfinite(OPEN_RAW[ex])
        pp = np.where(pool)[0]; ev[ex] = (pp, np.full(len(pp), 1 / len(pp)), mode); recs.append((i, ex, [], False))
    nav, _ = sim_w(ev, 0, 0, min(MD), len(IDX) - 1)
    return {k: evaluate(recs, nav, nav, sg, 0, 0, mode)['CAGR'] for k, sg in {'2006-02_2019-04': segs, '2006-02_2012-12': [('2006-02-01', '2012-12-31')], '2014-01_2019-04': [('2014-01-01', '2019-04-30')]}.items()}
EXTRA = {'EW_market_survivors_only': ew_market(True), 'EW_market_all_incl_delisted': ew_market(False)}
print('EXTRA', EXTRA)
res['RESULTS'] = OUT; res['EXTRA'] = EXTRA
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print('done', round(time.time() - T0))

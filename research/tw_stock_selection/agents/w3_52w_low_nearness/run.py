#!/usr/bin/env python3
"""接近 52 週低點（Chen-Yu NL）＋ 52 週低點反彈。規則見 results.json 的 PREREG（prereg.py 事前寫入，不改）。
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
cols = list(L.COLS); N = L.N; keep = np.array([not c.startswith('91') for c in cols])
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
G = {k: v.loc[good] for k, v in D.items()}
IG = G['adj'].index; TG = len(IG)
F = G['flag']; bad5 = F == 5
ADJ = G['adj'].mask(bad5); CR = G['close_raw'].mask(bad5); OP = G['open'].mask(bad5); V = G['value']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
LOW250 = ADJ.rolling(250, min_periods=240).min()
NL = ADJ / LOW250
A = ADJ.ffill(limit=10)
MOM = A.shift(21) / A.shift(252) - 1
MA20 = ADJ.rolling(20, min_periods=20).mean()
mkt4 = (F == 4).sum(axis=1) >= 50
FL = (F.isin([3, 4, 5]) & ~((F == 4) & mkt4.to_numpy()[:, None])).astype(float)
FLW = FL.rolling(21, min_periods=1).max().to_numpy() > 0           # [d-20, d]
ADJn, CRn, V60n, NLn, MOMn, LOWn, MAn = (x.to_numpy() for x in (ADJ, CR, V60, NL, MOM, LOW250, MA20))
BASE = (V60n >= 1e7) & (CRn >= 10) & np.isfinite(ADJn) & keep[None, :] & ~FLW
POOL = BASE & np.isfinite(NLn)
OPEN_OK = np.isfinite(OP.to_numpy()) & np.isfinite(ADJn)
LPOS = {d: k for k, d in enumerate(L.IDX)}
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - 0.006
GAP_UP = (L.AO_raw / L.AC_prev - 1) >= thrL

# 月曆：每月最後交易日
me = pd.Series(np.arange(TG), index=IG).groupby(IG.to_period('M')).max().to_numpy()
SIG = [i for i in me if IG[i] >= pd.Timestamp('2004-12-01') and i + 1 < TG]

def legs_for(i):
    ex = i + 1; ok = POOL[i] & OPEN_OK[ex]; n = int(ok.sum())
    if n < 100: return None
    order = np.argsort(np.where(ok, NLn[i], np.inf), kind='stable')[:n]
    cuts = [int(math.floor(n * k / 10)) for k in range(11)]
    dec = [order[cuts[k]:cuts[k + 1]] for k in range(10)]
    return ex, ok, order, dec

def pick_fill(cand, size, held, exL, info):
    pick = []
    for j in cand:
        if GAP_UP[exL, j] and j not in held: info['blocked'] += 1; continue
        pick.append(j)
        if len(pick) == size: break
    return pick

def build_main():
    ev = {f'D{k+1}': {} for k in range(10)}; ev['D1_leg'] = {}; ev['D10_leg'] = {}; ev['EW'] = {}
    ev['aD1'] = {}; ev['aD10'] = {}
    for q in range(5):
        ev[f'mq{q+1}_D1'] = {}; ev[f'mq{q+1}_D10'] = {}; ev[f'mom_Q{q+1}'] = {}
    held = {'D1_leg': set(), 'D10_leg': set(), 'aD1': set(), 'aD10': set()}
    info = {'months': 0, 'blocked': 0, 'pool': [], 'nD1': [], 'nlD1': [], 'nlD10': [], 'pool_a': [], 'n_aD1': []}
    ia = {'blocked': 0}
    for i in SIG:
        r = legs_for(i)
        if r is None: continue
        ex, ok, order, dec = r; exL = LPOS[IG[ex]]; blk = GAP_UP[exL]
        info['months'] += 1; info['pool'].append(int(ok.sum())); info['nD1'].append(len(dec[0]))
        info['nlD1'].append(float(np.median(NLn[i, dec[0]]))); info['nlD10'].append(float(np.median(NLn[i, dec[9]])))
        for k in range(10):
            ev[f'D{k+1}'][exL] = ('open', np.array([j for j in dec[k] if not blk[j]]))
        p1 = pick_fill(order, len(dec[0]), held['D1_leg'], exL, info); ev['D1_leg'][exL] = ('open', np.array(p1)); held['D1_leg'] = set(p1)
        p10 = pick_fill(order[::-1], len(dec[9]), held['D10_leg'], exL, ia); ev['D10_leg'][exL] = ('open', np.array(p10)); held['D10_leg'] = set(p10)
        ev['EW'][exL] = ('open', np.where(ok)[0])
        # 變體 a：動能五分組內 NL 排序
        ok2 = ok & np.isfinite(MOMn[i]); n2 = int(ok2.sum())
        if n2 >= 100:
            mo = np.argsort(np.where(ok2, MOMn[i], np.inf), kind='stable')[:n2]
            mc = [int(math.floor(n2 * k / 5)) for k in range(6)]
            u1, u10 = [], []
            for q in range(5):
                mem = mo[mc[q]:mc[q + 1]]
                ev[f'mom_Q{q+1}'][exL] = ('open', np.array([j for j in mem if not blk[j]]))
                srt = mem[np.argsort(NLn[i, mem], kind='stable')]; m = len(srt); sz = int(round(0.1 * m))
                c1 = [int(math.floor(m * k / 10)) for k in range(11)]
                ev[f'mq{q+1}_D1'][exL] = ('open', np.array([j for j in srt[c1[0]:c1[1]] if not blk[j]]))
                ev[f'mq{q+1}_D10'][exL] = ('open', np.array([j for j in srt[c1[9]:c1[10]] if not blk[j]]))
                u1 += pick_fill(srt, sz, held['aD1'], exL, ia); u10 += pick_fill(srt[::-1], sz, held['aD10'], exL, ia)
            ev['aD1'][exL] = ('open', np.array(u1)); ev['aD10'][exL] = ('open', np.array(u10))
            held['aD1'] = set(u1); held['aD10'] = set(u10); info['pool_a'].append(n2); info['n_aD1'].append(len(u1))
    for k in ('pool', 'nD1', 'nlD1', 'nlD10', 'pool_a', 'n_aD1'):
        info[k] = round(float(np.mean(info[k])), 3)
    info['blocked_other_legs'] = ia['blocked']
    return ev, info

t0 = LPOS[IG[SIG[0] + 1]]; t1 = len(L.IDX) - 1
E50 = L.E50.iloc[t0:t1 + 1]
WIN = {'select_2005-01_2019-04': [('2005-01-01', '2019-04-30')], 'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
       'y2024_2025': [('2024-01-01', '2025-12-31')],
       'valid_survivor_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
       'valid_ex_src3': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
       'full_2005-01_2026-10': [('2005-01-01', '2026-10-31')]}
mon = lambda s: (1 + s).groupby(s.index.to_period('M')).prod() - 1

def segmask(idx, segs):
    m = np.zeros(len(idx), bool)
    for a, b in segs: m |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    return m

def stats(nav, segs, b50, ew):
    r = nav.pct_change(); m = segmask(r.index, segs) & r.notna().to_numpy()
    rr = r[m]; span = 0.0
    for a, b in segs:
        x = rr[(rr.index >= pd.Timestamp(a)) & (rr.index <= pd.Timestamp(b))]
        if len(x): span += (x.index[-1] - r.index[max(r.index.get_loc(x.index[0]) - 1, 0)]).days / 365.25
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(span, 2)}
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

def spread(na, nb, segs):
    ra, rb = mon(na.pct_change().dropna()), mon(nb.pct_change().dropna())
    d = (ra - rb).dropna(); d = d[segmask(d.index.to_timestamp(), segs)]
    return {'mean_pct_m': round(float(d.mean() * 100), 3), 't': tstat(d), 'n': int(len(d))}

# ================= 事件型（變體 b） =================
Fn = F.to_numpy()
UNIV = BASE & np.isin(Fn, [0, 1, 2])
SIGB = UNIV & (ADJn <= 1.05 * LOWn) & np.vstack([np.zeros((1, N), bool), ADJn[:-1] <= MAn[:-1]]) & (ADJn > MAn)
fac = ADJ / CR
AOg = (OP * fac).to_numpy()
ACf = ADJ.ffill().to_numpy()
limg = np.where(IG < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
ACp = np.vstack([np.full((1, N), np.nan), ACf[:-1]])
GAPg = (AOg / ACp - 1) >= limg - 0.006
PX = pd.DataFrame(AOg).bfill().to_numpy(); PX = np.where(np.isfinite(PX), PX, ACf)
UNIVd = pd.DataFrame(UNIV, index=IG)
oo = pd.DataFrame(AOg, index=IG).shift(-1) / pd.DataFrame(AOg, index=IG) - 1
oo = oo.where(UNIVd.shift(1, fill_value=False)).clip(-0.5, 1.0)
LEW = np.concatenate([[0.0], np.cumsum(np.log1p(oo.mean(axis=1).fillna(0).to_numpy()))])
ew_cc = G['ret'].mask(bad5).where(UNIVd.shift(1, fill_value=False).to_numpy()).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50g = L.e50.reindex(IG.union(L.e50.index)).ffill().reindex(IG); r50g = e50g.pct_change().fillna(0); E50g = e50g.to_numpy()
SRC3 = (IG >= pd.Timestamp('2024-02-15')) & (IG <= pd.Timestamp('2025-09-23'))
BUY, SELL = L.cost_rates(0.001); BUY_S, SELL_S = L.cost_rates('stress')

def build_events(time_only=False):
    ev = []; nb = nno = 0
    for j in range(N):
        free = -1
        for t in np.where(SIGB[:, j])[0]:
            if t < free: continue
            e = t + 1
            if e >= TG: continue
            if not np.isfinite(AOg[e, j]): nno += 1; continue
            if GAPg[e, j]: nb += 1; continue
            x = e + 20; why = 'time'
            if not time_only:
                p0 = AOg[e, j]; stop = LOWn[t, j]
                for s in range(e, min(e + 19, TG - 1) + 1):
                    a = ADJn[s, j]
                    if not np.isfinite(a): continue
                    if a < stop: x = s + 1; why = 'stop'; break
                    if a >= 1.15 * p0: x = s + 1; why = 'tp'; break
            done = x < TG
            ev.append((j, t, e, x, done, why)); free = x
    return ev, nb, nno

def event_table(ev):
    rows = []
    for j, t, e, x, done, why in ev:
        if not done: continue
        raw = PX[x, j] / AOg[e, j] - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1; net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        ew = math.expm1(LEW[x] - LEW[e]); b50 = E50g[x - 1] / E50g[t] - 1
        rows.append((cols[j], IG[t], IG[e], x - e, why, raw, net, net_s, ew, b50, bool(SRC3[t:x + 1].any())))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'hold', 'why', 'raw', 'net', 'net_s', 'ew', 'b50', 'src3'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']; df['ab_gross'] = df['raw'] - df['ew']
    return df

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 1),
            'exit_mix': {k: round(float(v), 3) for k, v in df['why'].value_counts(normalize=True).items()},
            'mean_raw_pct': round(df['raw'].mean() * 100, 3), 'mean_net_pct': round(df['net'].mean() * 100, 3), 'mean_EW_pct': round(df['ew'].mean() * 100, 3),
            'mean_ab_gross_pct': round(df['ab_gross'].mean() * 100, 3), 'mean_ab_net_pct': round(df['ab'].mean() * 100, 3),
            'median_ab_net_pct': round(df['ab'].median() * 100, 3), 't_month_ab_net': tstat(gm['ab'].mean()), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()), 'mean_ab_net_stress_pct': round(df['ab_s'].mean() * 100, 3),
            't_month_ab_net_stress': tstat(gm['ab_s'].mean()), 'mean_net_vs_0050_pct': round(df['ab50'].mean() * 100, 3),
            't_month_vs_0050': tstat(gm['ab50'].mean()), 'hit_ab_net': round(float((df['ab'] > 0).mean()), 3)}

def portfolio(ev, buy, sell):
    R = np.zeros((TG, N), np.float32); W = np.zeros((TG, N), np.float32); ent = np.zeros(TG); exi = np.zeros(TG)
    for j, t, e, x, done, _ in ev:
        xe = x if done else TG - 1
        W[e:xe + 1, j] += 1; R[e, j] = ACf[e, j] / AOg[e, j] - 1
        if xe > e: R[e + 1:xe + 1, j] = np.nan_to_num(ACf[e + 1:xe + 1, j] / ACf[e:xe, j] - 1)
        if done: R[x, j] = PX[x, j] / ACf[x - 1, j] - 1; exi[x] += 1
        ent[e] += 1
    W = np.minimum(W, 1); n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(n > 0, (W * R).sum(1) / np.where(n > 0, n, 1), 0.0)
        c = np.where(n > 0, (ent * buy + exi * sell) / np.where(n > 0, n, 1), 0.0)
    return pd.Series(g - c, index=IG), pd.Series(n, index=IG)

def pstats(r, npos, segs):
    m = segmask(r.index, segs); rr = r[m]; bb = r50g[m]; ee = ew_cc[m]
    yrs = sum(((rr[(rr.index >= pd.Timestamp(a)) & (rr.index <= pd.Timestamp(b))].index[-1] - rr[(rr.index >= pd.Timestamp(a)) & (rr.index <= pd.Timestamp(b))].index[0]).days / 365.25) for a, b in segs)
    def st(x):
        eq = (1 + x).cumprod(); return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}; o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
    o['share_days_invested'] = round(float((npos[m] > 0).mean()), 3); o['avg_positions_when_invested'] = round(float(npos[m][npos[m] > 0].mean()), 1)
    ms, mb, mE = mon(rr), mon(bb), mon(ee)
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - mE).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - mE)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    return o

def main():
    res = json.load(open(HERE / 'results.json'))
    # 存活者偏誤診斷：各期間股池中「之後下市」（最後有值日 < 2026-10-01）的檔數比例
    last_valid = D['adj'].apply(lambda s: s.last_valid_index())
    dead = (last_valid < pd.Timestamp('2026-10-01')).to_numpy()
    surv = {}
    for nm, (a, b) in {'2005-2018': ('2005-01-01', '2018-12-31'), '2019-2023': ('2019-01-01', '2023-12-31'), '2024-2026': ('2024-01-01', '2026-10-31')}.items():
        ss = [i for i in SIG if pd.Timestamp(a) <= IG[i] <= pd.Timestamp(b)]
        pooled = POOL[ss].any(0)
        delist_in = ((last_valid >= pd.Timestamp(a)) & (last_valid <= pd.Timestamp(b))).to_numpy()
        surv[nm] = {'distinct_in_pool': int(pooled.sum()), 'delisted_within_period': int((pooled & delist_in & dead).sum())}
    out = {'meta': {'bad_days_dropped': BAD, 'n_signal_months': len(SIG), 'first_signal': str(IG[SIG[0]].date()), 'last_signal': str(IG[SIG[-1]].date()),
                    'survivorship_check': surv, 'market_flag4_days_ignored': [str(x.date()) for x in IG[mkt4.to_numpy()]]}}
    ev, info = build_main(); out['meta']['main_info'] = info; print(info, flush=True)
    navs = {}
    for k in list(ev):
        if k.startswith(('D', 'EW', 'mq', 'mom')):
            navs[f'{k}_gross'], _ = L.sim_portfolio(ev[k], 0.0, 0.0, t0, t1)
    ew = navs['EW_gross']
    sel = WIN['select_2005-01_2019-04']
    sp_sel = spread(navs['D1_gross'], navs['D10_gross'], sel)
    leg = 'D1_leg' if sp_sel['mean_pct_m'] > 0 else 'D10_leg'
    out['direction'] = {'select_D1_minus_D10_gross': sp_sel, 'tradable_leg': leg}
    print('direction', out['direction'], flush=True)
    # 十分組表
    dec = {}
    for w, s in WIN.items():
        dec[w] = {f'D{k+1}': stats(navs[f'D{k+1}_gross'], s, E50, ew)['CAGR'] for k in range(10)}
        dec[w]['EW'] = stats(ew, s, E50, ew)['CAGR']
        dec[w]['D1_minus_D10'] = spread(navs['D1_gross'], navs['D10_gross'], s)
    out['decile_gross_CAGR'] = dec
    # 主要腿與變體 a
    for var in ['D1_leg', 'D10_leg', 'aD1', 'aD10']:
        block = {}
        for cname, cr in (('gross', (0.0, 0.0)), ('normal', L.cost_rates(0.001)), ('stress', L.cost_rates('stress'))):
            nav, tov = L.sim_portfolio(ev[var], cr[0], cr[1], t0, t1); navs[f'{var}_{cname}'] = nav
            block[cname] = {'turnover_oneway_per_rebal': round(tov, 3), 'windows': {w: stats(nav, s, E50, ew) for w, s in WIN.items()}}
            if cname == 'normal': block['yearly_normal_pct'] = yearly(nav)
        out[var] = block
        for cn in ('gross', 'normal', 'stress'):
            for wn in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'valid_ex_src3', 'y2024_2025'):
                x = block[cn]['windows'][wn]
                print(f'{var:7s} {cn:6s} {wn:24s} CAGR {x["CAGR"]:.4f} DD {x["maxDD"]:.3f} | EW {x["EW_CAGR"]:.4f} t_EW {x["t_vs_EW_m"]} | 0050 {x["0050_CAGR"]:.4f} t_50 {x["t_vs_0050_m"]}', flush=True)
    # 變體 a 附表：動能組內價差、與動能價差的相關
    va = {}
    for w, s in WIN.items():
        va[w] = {f'momQ{q+1}_D1_minus_D10': spread(navs[f'mq{q+1}_D1_gross'], navs[f'mq{q+1}_D10_gross'], s) for q in range(5)}
        va[w]['aD1_minus_aD10_gross'] = spread(navs['aD1_gross'], navs['aD10_gross'], s)
        va[w]['WML_momQ5_minus_Q1'] = spread(navs['mom_Q5_gross'], navs['mom_Q1_gross'], s)
        nlS = mon(navs['D1_gross'].pct_change().dropna()) - mon(navs['D10_gross'].pct_change().dropna())
        wml = mon(navs['mom_Q5_gross'].pct_change().dropna()) - mon(navs['mom_Q1_gross'].pct_change().dropna())
        z = pd.concat([nlS, wml], axis=1).dropna(); z = z[segmask(z.index.to_timestamp(), s)]
        va[w]['corr_NLspread_vs_WML'] = round(float(z.corr().iloc[0, 1]), 3)
        comb = 0.5 * z.iloc[:, 0] + 0.5 * z.iloc[:, 1]
        va[w]['half_NL_half_WML_gross'] = {'mean_pct_m': round(float(comb.mean() * 100), 3), 't': tstat(comb)}
    out['variant_a_table'] = va
    out['yearly_EW_pool_pct'] = yearly(ew); out['yearly_0050_pct'] = yearly(E50)
    out['yearly_D1_gross_pct'] = yearly(navs['D1_gross']); out['yearly_D10_gross_pct'] = yearly(navs['D10_gross'])
    # 判定（主）
    def passes(w):
        x = out[leg]['normal']['windows'][w]
        ok_ew = (x['t_vs_EW_m'] == x['t_vs_EW_m']) and x['t_vs_EW_m'] >= 2; ok50 = x['CAGR'] >= x['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok50), 'pass': bool(ok_ew and ok50)}
    vd = {'leg': leg, 'select': passes('select_2005-01_2019-04'), 'valid': passes('valid_2019-05_2026-10')}
    k = vd['select']['pass'] + vd['valid']['pass']; vd['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    def vpass(v, w):
        x = out[v]['normal']['windows'][w]; return bool(x['t_vs_EW_m'] >= 2 and x['CAGR'] >= x['0050_CAGR'])
    vd['variant_a'] = {v: {w: vpass(v, w) for w in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')} for v in ('aD1', 'aD10')}
    # 變體 b
    EWIN = {'select_2005-01_2019-04': [('2005-01-01', '2019-04-30')], 'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
            'y2024_2025': [('2024-01-01', '2025-12-31')], 'valid_survivor_2019-05_2023-12': [('2019-05-01', '2023-12-31')]}
    vb = {}
    for nm, to in (('b_bounce', False), ('b_time_only_diag', True)):
        evb, nb, nno = build_events(to); df = event_table(evb)
        o = {'raw_signal_days': int(SIGB.sum()), 'events_taken': len(evb), 'blocked_gap_up': nb, 'no_open': nno}
        for w, s in EWIN.items():
            a, b = s[0]; o[f'event_{w}'] = ev_stats(df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))])
        o['event_valid_ex_src3'] = ev_stats(df[(df['sig'] >= pd.Timestamp('2019-05-01')) & ~df['src3']])
        r, npos = portfolio(evb, BUY, SELL); rs, _ = portfolio(evb, BUY_S, SELL_S); rg, _ = portfolio(evb, 0.0, 0.0)
        for w, s in EWIN.items():
            o[f'port_{w}'] = pstats(r, npos, s); o[f'port_{w}']['CAGR_stress'] = pstats(rs, npos, s)['CAGR']; o[f'port_{w}']['CAGR_gross'] = pstats(rg, npos, s)['CAGR']
        o['yearly_events'] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2)} for y, g in df.groupby(df['sig'].dt.year)}
        o['top_raw'] = df.nlargest(5, 'raw')[['code', 'sig', 'raw', 'why']].astype(str).to_dict('records')
        vb[nm] = o
        if nm == 'b_bounce': df.to_csv(HERE / 'events_b.csv.gz', index=False)
        for w in EWIN:
            e = o[f'event_{w}']; p = o[f'port_{w}']
            print(nm, w, 'n', e.get('n'), 'ab', e.get('mean_ab_net_pct'), 't', e.get('t_month_ab_net'), 'vs50', e.get('mean_net_vs_0050_pct'), '| port', p['CAGR'], '50', p['0050_CAGR'], 'EW', p['EW_CAGR'], 't_EW', p['t_vs_EW_m'], flush=True)
    out['variant_b'] = vb
    def bpass(w):
        e = vb['b_bounce'][f'event_{w}']; p = vb['b_bounce'][f'port_{w}']
        return bool(e.get('mean_ab_net_pct', -1) > 0 and (e.get('t_month_ab_net') or -9) >= 2 and p['CAGR'] >= p['0050_CAGR'])
    vd['variant_b'] = {w: bpass(w) for w in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    out['verdict'] = vd
    print(json.dumps(vd, ensure_ascii=False), flush=True)
    pd.DataFrame({k: v for k, v in navs.items() if k.startswith(('D1', 'D10', 'EW', 'aD'))}).to_parquet(HERE / 'navs.parquet')
    res['RESULTS'] = out
    json.dump(res, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x))
    print('done')

if __name__ == '__main__':
    main()

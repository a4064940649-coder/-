#!/usr/bin/env python3
"""VR 成交量比率（附 PSY）回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
事件／組合統計架構沿用 w4_vol_leads_price_vma_cross/run.py（複製，不 import）；成本與月頻引擎用 exec_rebuild/exlib。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成本模型與 sim_portfolio（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
names = ['adj', 'close_raw', 'open', 'value', 'flag', 'src', 'ret']
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in names}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if c.isdigit() and len(c) == 4 and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
F = D['flag']; bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); V = D['value']
RET = D['ret'].mask(bad5)
fac = ADJ / CR
AO_raw = (OP * fac)
ACf = ADJ.ffill()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
UNIV = (V60 >= 1e7) & (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()

# ---------------- 指標 ----------------
VOL = (V / CR).where(ADJ.notna())
up = RET > 1e-9; dn = RET < -1e-9; fl = RET.notna() & ~up & ~dn
okv = VOL.notna() & RET.notna()
Vz = VOL.where(okv)
su = Vz.where(up, 0).where(okv).rolling(26, min_periods=26).sum()
sd = Vz.where(dn, 0).where(okv).rolling(26, min_periods=26).sum()
sf = Vz.where(fl, 0).where(okv).rolling(26, min_periods=26).sum()
num = su + 0.5 * sf; den = sd + 0.5 * sf
with np.errstate(divide='ignore', invalid='ignore'):
    VR = (num / den * 100)
VR = VR.where(~((den == 0) & num.notna()), np.inf)
PSY = up.astype(float).where(RET.notna()).rolling(12, min_periods=12).mean() * 100
MA60 = ADJ.rolling(60, min_periods=60).mean()
dA = ACf.diff()
gain = dA.clip(lower=0); loss = (-dA).clip(lower=0)
ag = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(); al = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
RSI = 100 - 100 / (1 + ag / al)
RSI = RSI.where(ADJ.notna())

lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
AC_prev = ACf.shift(1).to_numpy()
GAP_UP = (AO_raw.to_numpy() / AC_prev - 1) >= lim - 0.006
AOn = AO_raw.to_numpy(); ACn = ACf.to_numpy()
PX = AO_raw.bfill().to_numpy()
PX = np.where(np.isfinite(PX), PX, ACn)

oo = (AO_raw.shift(-1) / AO_raw - 1)
oo = oo.where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0)
ew_oo = oo.mean(axis=1).fillna(0).to_numpy()
LEW = np.concatenate([[0.0], np.cumsum(np.log1p(ew_oo))])
ew_cc = RET.where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0)
E50n = e50.to_numpy()

BUY, SELL = L.cost_rates(0.001)
BUY_S, SELL_S = L.cost_rates('stress')

VRn = VR.to_numpy(); PSYn = PSY.to_numpy()

def vr_signal(Lv, filt):
    S = UNIV & (VR.shift(1) < Lv) & (VR >= Lv)
    if filt == 'ma60': S = S & (ADJ > MA60)
    return S.fillna(False).to_numpy()

def build_events(S, EXIT, H):
    ev = []; nblock = 0; nnoopen = 0
    for j in range(N):
        ts = np.where(S[:, j])[0]; free_from = -1
        for t in ts:
            if t < free_from: continue
            e = t + 1
            if e >= T: continue
            if not np.isfinite(AOn[e, j]): nnoopen += 1; continue
            if GAP_UP[e, j]: nblock += 1; continue
            hi = min(e + H, T)
            hit = np.where(EXIT[e:hi, j])[0]
            if len(hit): x = e + hit[0] + 1; why = 'signal'
            else: x = e + H; why = 'time'
            done = x < T
            ev.append((j, t, e, x, done, why)); free_from = x
    return ev, nblock, nnoopen

def event_table(ev):
    rows = []
    for j, t, e, x, done, why in ev:
        if not done: continue
        p0 = AOn[e, j]; p1 = PX[x, j]; raw = p1 / p0 - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1
        net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        ew = math.expm1(LEW[x] - LEW[e]); b50 = E50n[x - 1] / E50n[t] - 1
        rows.append((cols[j], IDX[t], IDX[e], x - e, why, raw, net, net_s, ew, b50))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'hold', 'why', 'raw', 'net', 'net_s', 'ew', 'b50'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']; df['ab_gross'] = df['raw'] - df['ew']
    return df

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31')}

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 1),
            'share_exit_signal': round(float((df['why'] == 'signal').mean()), 3),
            'mean_raw_pct': round(df['raw'].mean() * 100, 3), 'mean_net_pct': round(df['net'].mean() * 100, 3),
            'mean_EW_pct': round(df['ew'].mean() * 100, 3),
            'mean_ab_gross_pct': round(df['ab_gross'].mean() * 100, 3), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': round(df['ab'].mean() * 100, 3), 'median_ab_net_pct': round(df['ab'].median() * 100, 3),
            't_month_ab_net': tstat(gm['ab'].mean()),
            'mean_ab_net_stress_pct': round(df['ab_s'].mean() * 100, 3), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_net_vs_0050_pct': round(df['ab50'].mean() * 100, 3), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3)}

def portfolio(ev, buy, sell):
    R = np.zeros((T, N), np.float32); W = np.zeros((T, N), np.float32); ent = np.zeros(T); exi = np.zeros(T)
    for j, t, e, x, done, why in ev:
        xe = x if done else T - 1
        W[e:xe + 1, j] += 1
        R[e, j] = ACn[e, j] / AOn[e, j] - 1
        if xe > e: R[e + 1:xe + 1, j] = np.nan_to_num(ACn[e + 1:xe + 1, j] / ACn[e:xe, j] - 1)
        if done:
            R[x, j] = PX[x, j] / ACn[x - 1, j] - 1 if x > e else PX[x, j] / AOn[e, j] - 1
            exi[x] += 1
        ent[e] += 1
    W = np.minimum(W, 1); n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(n > 0, (W * R).sum(1) / np.where(n > 0, n, 1), 0.0)
        c = np.where(n > 0, (ent * buy + exi * sell) / np.where(n > 0, n, 1), 0.0)
    return pd.Series(g - c, index=IDX), pd.Series(n, index=IDX)

def pstats(r, npos, a, b, bench=None, ew=None):
    bench = r50 if bench is None else bench; ew = ew_cc if ew is None else ew
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    rr = r[m]; bb = bench.reindex(rr.index).fillna(0); ee = ew.reindex(rr.index).fillna(0)
    yrs = (rr.index[-1] - rr.index[0]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod(); return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}
    o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
    o['vol'] = round(float(rr.std() * math.sqrt(252)), 4)
    if npos is not None:
        o['share_days_invested'] = round(float((npos[m] > 0).mean()), 3)
        o['avg_positions_when_invested'] = round(float(npos[m][npos[m] > 0].mean()), 1) if (npos[m] > 0).any() else 0
    mo = lambda x: (1 + x).groupby(x.index.to_period('M')).prod() - 1
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    return o

def run_event_spec(S, EXIT, H, key, save=False):
    ev, nb, nno = build_events(S, EXIT, H)
    df = event_table(ev)
    o = {'raw_signal_days': int(S.sum()), 'events_taken': len(ev), 'blocked_gap_up': nb, 'no_open': nno}
    for w, (a, b) in WIN.items():
        o[f'event_{w}'] = ev_stats(df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))])
    r, npos = portfolio(ev, BUY, SELL); rs, _ = portfolio(ev, BUY_S, SELL_S); rg, _ = portfolio(ev, 0.0, 0.0)
    for w, (a, b) in WIN.items():
        o[f'port_{w}'] = pstats(r, npos, a, b)
        o[f'port_{w}']['CAGR_stress'] = pstats(rs, npos, a, b)['CAGR']; o[f'port_{w}']['CAGR_gross'] = pstats(rg, npos, a, b)['CAGR']
    o['yearly_events'] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2)} for y, g in df.groupby(df['sig'].dt.year)}
    if save: df.to_csv(Path(L.P).parent / f'w4_vr_{key}_events.csv.gz', index=False)
    es, ev_ = o['event_select_2005-01_2019-04'], o['event_valid_2019-05_2026-10']
    ps, pv = o['port_select_2005-01_2019-04'], o['port_valid_2019-05_2026-10']
    print(key, 'n', len(df), '| S ab', es.get('mean_ab_net_pct'), 'g', es.get('mean_ab_gross_pct'), 't', es.get('t_month_ab_net'), 'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'],
          '| V ab', ev_.get('mean_ab_net_pct'), 'g', ev_.get('mean_ab_gross_pct'), 't', ev_.get('t_month_ab_net'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], flush=True)
    return o, df, S

# ---------------- 橫斷面（月底五分組） ----------------
LCI = np.array([L.CI[c] for c in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_L = (L.AO_raw / L.AC_prev - 1) >= thrL
def cross_section():
    s = pd.Series(IDX, index=IDX)
    ME = [d for d in s.groupby([IDX.year, IDX.month]).last() if d >= pd.Timestamp('2005-01-01')]
    VRme = VR.reindex(ME); Ume = UNIV.reindex(ME)
    ev = {q: {} for q in range(5)}; pools = {}; held = {q: set() for q in range(5)}; nblk = 0; nn = []
    for d in ME[:-1]:
        ex = L.pos(d) + 1
        if ex >= len(L.IDX): continue
        trad = np.isfinite(L.OPEN_RAW[ex, LCI]) & np.isfinite(L.ADJ_RAW[ex, LCI])
        v = VRme.loc[d].to_numpy(); ok = Ume.loc[d].to_numpy() & ~np.isnan(v) & trad
        if ok.sum() < 50: continue
        vv = pd.Series(v[ok], index=np.where(ok)[0]).replace(np.inf, 1e12)
        q = pd.qcut(vv.rank(method='first'), 5, labels=False)
        pools[ex] = ('open', LCI[np.where(ok)[0]]); nn.append(int(ok.sum()))
        for k in range(5):
            mem = q.index[q.to_numpy() == k]
            pick = [LCI[j] for j in mem if not (GAP_L[ex, LCI[j]] and LCI[j] not in held[k])]
            nblk += len(mem) - len(pick)
            ev[k][ex] = ('open', np.array(pick)); held[k] = set(pick)
    t0 = L.pos(ME[0]); t1 = L.pos(ME[-1])
    def nav(events, cost):
        b, s_ = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        return L.sim_portfolio(events, b, s_, t0, t1)
    out = {'n_pool_avg': round(float(np.mean(nn)), 1), 'blocked_gap_up': nblk}
    ewn, ewto = nav(pools, 0.001); ew_g, _ = nav(pools, 'gross')
    ewr = ewn.pct_change()
    out['EW_turnover'] = round(ewto, 3)
    navs = {}
    for k in range(5):
        n1, to = nav(ev[k], 0.001); ns, _ = nav(ev[k], 'stress'); ng, _ = nav(ev[k], 'gross')
        navs[k] = n1
        o = {'turnover_month': round(to, 3)}
        for w, (a, b) in WIN.items():
            o[w] = pstats(n1.pct_change().fillna(0), None, a, b, bench=L.r50.reindex(n1.index).fillna(0), ew=ewr.fillna(0))
            o[w]['CAGR_stress'] = pstats(ns.pct_change().fillna(0), None, a, b)['CAGR']
            o[w]['CAGR_gross'] = pstats(ng.pct_change().fillna(0), None, a, b, ew=ew_g.pct_change().fillna(0))['CAGR']
            o[w]['t_vs_EW_m_gross'] = pstats(ng.pct_change().fillna(0), None, a, b, ew=ew_g.pct_change().fillna(0))['t_vs_EW_m']
        out[f'Q{k+1}'] = o
        print('XS Q%d' % (k + 1), 'to', o['turnover_month'], ' '.join(f"{w[:5]} CAGR {o[w]['CAGR']} 50 {o[w]['0050_CAGR']} EW {o[w]['EW_CAGR']} tEW {o[w]['t_vs_EW_m']} t50 {o[w]['t_vs_0050_m']} DD {o[w]['maxDD']}" for w in WIN), flush=True)
    # Q1−Q5 多空（毛、只看方向）
    ls = navs[0].pct_change() - navs[4].pct_change()
    mo = (1 + ls.fillna(0)).groupby(ls.index.to_period('M')).prod() - 1
    out['Q1_minus_Q5_monthly'] = {w: {'mean_pct': round(float(mo[(mo.index >= pd.Period(a, 'M')) & (mo.index <= pd.Period(b, 'M'))].mean() * 100), 3),
                                      't': tstat(mo[(mo.index >= pd.Period(a, 'M')) & (mo.index <= pd.Period(b, 'M'))])} for w, (a, b) in WIN.items()}
    return out

def main():
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'VR_quantiles_in_univ': {str(q): round(float(np.nanquantile(np.where(np.isfinite(VRn) & UNIV.to_numpy(), VRn, np.nan), q)), 1) for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
                    'share_VR_gt350_in_univ': round(float(((VR > 350) & UNIV).sum().sum() / UNIV.sum().sum()), 4),
                    'share_VR_gt450_in_univ': round(float(((VR > 450) & UNIV).sum().sum() / UNIV.sum().sum()), 4),
                    'share_VR_lt60_in_univ': round(float(((VR < 60) & UNIV).sum().sum() / UNIV.sum().sum()), 4),
                    'share_VR_lt40_in_univ': round(float(((VR < 40) & UNIV).sum().sum() / UNIV.sum().sum()), 4),
                    'share_PSY_lt25_in_univ': round(float(((PSY < 25) & UNIV).sum().sum() / UNIV.sum().sum()), 4),
                    'share_PSY_gt75_in_univ': round(float(((PSY > 75) & UNIV).sum().sum() / UNIV.sum().sum()), 4)}}
    print(json.dumps(out['meta'], ensure_ascii=False))
    res = {}
    EXV = {X: (VR > X).fillna(False).to_numpy() for X in (350, 450)}
    prim_S = None
    for Lv in (60, 40):
        for filt in ('none', 'ma60'):
            S = vr_signal(Lv, filt)
            for X in (350, 450):
                for H in (20, 60):
                    key = f'VR_L{Lv}_{filt}_X{X}_H{H}'
                    o, df, _ = run_event_spec(S, EXV[X], H, key, save=(key == 'VR_L60_none_X350_H20'))
                    res[key] = o
                    if key == 'VR_L60_none_X350_H20': prim_S, prim_df = S, df
    # PSY
    SP = (UNIV & (PSY.shift(1) >= 25) & (PSY < 25)).fillna(False).to_numpy()
    EXP = (PSY > 75).fillna(False).to_numpy()
    for H in (60, 20):
        o, df, _ = run_event_spec(SP, EXP, H, f'PSY_H{H}')
        res[f'PSY_H{H}'] = o
        if H == 60: psy_df = df
    out['results'] = res
    # RSI 重疊
    Rn = RSI.to_numpy(); lowR = np.nan_to_num(Rn, nan=100) < 30
    lowR10 = pd.DataFrame(lowR).rolling(11, min_periods=1).max().to_numpy() > 0
    ts, js = np.where(prim_S)
    SR = (UNIV & (RSI.shift(1) < 30) & (RSI >= 30)).fillna(False).to_numpy()
    near = pd.DataFrame(prim_S.astype(float)).rolling(21, center=True, min_periods=1).max().to_numpy() > 0
    rt, rj = np.where(SR)
    pts, pjs = np.where(SP)
    out['rsi_overlap'] = {'VR_primary_signals': int(len(ts)), 'share_VR_with_RSI14_lt30_in_t-10..t': round(float(lowR10[ts, js].mean()), 3),
                          'share_VR_with_RSI14_lt30_on_t': round(float(lowR[ts, js].mean()), 3),
                          'RSI_rebound30_signals': int(len(rt)), 'share_RSI_rebound_with_VR_signal_within_pm10': round(float(near[rt, rj].mean()), 3),
                          'PSY_signals': int(len(pts)), 'share_PSY_with_RSI14_lt30_on_t': round(float(lowR[pts, pjs].mean()), 3),
                          'share_VR_signal_with_PSY_lt25_in_t-10..t': round(float((pd.DataFrame((np.nan_to_num(PSYn, nan=100) < 25).astype(float)).rolling(11, min_periods=1).max().to_numpy() > 0)[ts, js].mean()), 3)}
    print(out['rsi_overlap'], flush=True)
    out['cross_section'] = cross_section()
    def passes_ev(o, w):
        e = o[f'event_{w}']; p = o[f'port_{w}']
        ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
        ok_50 = p['CAGR'] >= p['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    def passes_xs(o):
        okk = (o['t_vs_EW_m'] or -9) >= 2 and o['excess_vs_EW_pct_m'] > 0
        return {'beat_EW_t2': bool(okk), 'not_lose_0050': bool(o['CAGR'] >= o['0050_CAGR']), 'pass': bool(okk and o['CAGR'] >= o['0050_CAGR'])}
    W2 = ['select_2005-01_2019-04', 'valid_2019-05_2026-10']
    vp = {w: passes_ev(res['VR_L60_none_X350_H20'], w) for w in W2}
    xp = {w: passes_xs(out['cross_section']['Q1'][w]) for w in W2}
    lab = lambda d: (lambda k: '可用' if k == 2 else ('待觀察' if k == 1 else '不可用'))(sum(v['pass'] for v in d.values()))
    vs = {'VR_event_primary': vp, 'VR_event_verdict': lab(vp), 'XS_Q1': xp, 'XS_verdict': lab(xp)}
    order = ['不可用', '待觀察', '可用']
    vs['verdict'] = max(vs['VR_event_verdict'], vs['XS_verdict'], key=order.index)
    vs['variants_pass'] = {k: {w: passes_ev(o, w)['pass'] for w in W2} for k, o in res.items()}
    vs['xs_all_pass'] = {f'Q{k}': {w: passes_xs(out['cross_section'][f'Q{k}'][w])['pass'] for w in W2} for k in range(1, 6)}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()

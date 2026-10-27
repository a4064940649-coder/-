#!/usr/bin/env python3
"""均線多頭排列＋均線同步上揚：月換股（M2）與日頻進出版。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成交引擎與成本（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'value', 'flag', 'ret']}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
F = D['flag']; bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); V = D['value']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
UNIV = (V60 >= 1e7) & (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()

MA = {n: ADJ.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60, 120)}
UPm = {n: MA[n] > MA[n].shift(5) for n in MA}
S_SHORT = (MA[5] > MA[10]) & (MA[10] > MA[20]) & (MA[20] > MA[60]) & UPm[5] & UPm[10] & UPm[20] & UPm[60] & (ADJ > MA[20])
S_LONG = (MA[20] > MA[60]) & (MA[60] > MA[120]) & UPm[20] & UPm[60] & UPm[120] & (ADJ > MA[20])
X_SHORT = (MA[5] < MA[10]) | (ADJ < MA[20])
X_LONG = (MA[20] < MA[60]) | (ADJ < MA[20])
AF = ADJ.ffill(limit=10)
RET60 = ADJ / AF.shift(60) - 1
MOM = AF.shift(21) / AF.shift(252) - 1
S52 = ADJ / ADJ.rolling(252, min_periods=240).max()

# ---------- 對應到 exlib 的完整日曆 ----------
LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[c] for c in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP_L = (L.AO_raw / L.AC_prev - 1) >= thrL

_s = pd.Series(IDX, index=IDX)
ME = list(_s.groupby([IDX.year, IDX.month]).last())
ME = [d for d in ME if pd.Timestamp('2004-12-01') <= d <= pd.Timestamp('2026-09-30')]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in ME}

def monthly_events(S, mode):
    events, pools, info = {}, {}, []
    held = set(); nblk = 0
    for d in ME:
        e = NEXT[d]; ex = LPOS[e]
        u = UNIV.loc[d].to_numpy()
        trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
        pool = u & trad
        sig = pool & S.loc[d].fillna(False).to_numpy()
        blocked = GAP_UP_L[ex, LCOL]
        idx = np.where(sig)[0]
        if mode == 'TOP30':
            r = RET60.loc[d].to_numpy()[idx]
            idx = idx[np.argsort(-np.nan_to_num(r, nan=-9e9), kind='stable')]
        pick = []
        for j in idx:
            if blocked[j] and LCOL[j] not in held: nblk += 1; continue
            pick.append(LCOL[j])
            if mode == 'TOP30' and len(pick) == 30: break
        events[ex] = ('open', np.array(pick, dtype=int)); pools[ex] = ('open', LCOL[pool])
        held = set(pick)
        info.append((d, int(pool.sum()), int(sig.sum()), len(pick)))
    return events, pools, nblk, pd.DataFrame(info, columns=['d', 'pool', 'signal', 'held']).set_index('d')

def sim_portfolio(events, buy, sell, t0, t1):
    """exlib.sim_portfolio 原樣邏輯；唯一差異：選股為空時全數賣出持有現金（exlib 版本會除以 0）。"""
    u = np.zeros(L.N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]
            Pz = L.AO[t] if mode == 'open' else L.AC[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash
            if len(pk) == 0:
                cash = Vt - sell * h.sum(); tov.append(h.sum() / Vt if Vt > 0 else 0); u = np.zeros(L.N)
            else:
                tgt = np.zeros(L.N); w = 1.0 / len(pk); Vn = Vt
                for _ in range(6):
                    tgt[:] = 0; tgt[pk] = w * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                    Vn = Vt - cost
                tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
                u = np.zeros(L.N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
    return pd.Series(nav, index=L.IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31')}

def pstats(r, bench, ew, a, b, npos=None):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    rr = r[m].fillna(0); bb = bench.reindex(rr.index).fillna(0); ee = ew.reindex(rr.index).fillna(0)
    yrs = (rr.index[-1] - r.index[max(np.argmax(m) - 1, 0)]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod(); return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}
    o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
    o['vol'] = round(float(rr.std() * math.sqrt(252)), 4)
    mo = lambda x: (1 + x).groupby(x.index.to_period('M')).prod() - 1
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['n_months'] = int(len(ms))
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['hit_vs_EW_m'] = round(float((ms > me).mean()), 3)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    if npos is not None:
        nn = npos[m]; o['share_days_invested'] = round(float((nn > 0).mean()), 3); o['avg_positions_when_invested'] = round(float(nn[nn > 0].mean()), 1)
    return o

# ---------- 日頻版本（事件） ----------
AOn = (OP * (ADJ / CR)).to_numpy(); ACn = ADJ.ffill().to_numpy(); ADJn = ADJ.to_numpy()
lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
GAP_UP = (AOn / np.vstack([np.full((1, N), np.nan), ACn[:-1]]) - 1) >= lim - 0.006
PX = pd.DataFrame(AOn).bfill().to_numpy(); PX = np.where(np.isfinite(PX), PX, ACn)
oo = (pd.DataFrame(AOn, index=IDX, columns=cols).shift(-1) / pd.DataFrame(AOn, index=IDX, columns=cols) - 1)
oo = oo.where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0)
LEW = np.concatenate([[0.0], np.cumsum(np.log1p(oo.mean(axis=1).fillna(0).to_numpy()))])
ew_cc = D['ret'].where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
E50n = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50_clean = E50n.pct_change().fillna(0); E50a = E50n.to_numpy()
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))
BUY, SELL = L.cost_rates(0.001); BUY_S, SELL_S = L.cost_rates('stress')

def daily_events(S, X):
    Sn = (S & UNIV).to_numpy(); Sprev = np.vstack([np.zeros((1, N), bool), S.fillna(False).to_numpy()[:-1]])
    first = Sn & ~Sprev; Xn = X.fillna(False).to_numpy()
    ev = []; nblock = nno = 0
    for j in range(N):
        free = -1
        for t in np.where(first[:, j])[0]:
            if t < free: continue
            e = t + 1
            if e >= T: continue
            if not np.isfinite(AOn[e, j]): nno += 1; continue
            if GAP_UP[e, j]: nblock += 1; continue
            seg = Xn[e:min(e + 250, T), j]; hit = np.where(seg)[0]
            x = e + hit[0] + 1 if len(hit) else e + 250
            ev.append((j, t, e, x, x < T)); free = x
    return ev, nblock, nno

def event_table(ev):
    rows = []
    for j, t, e, x, done in ev:
        if not done: continue
        raw = PX[x, j] / AOn[e, j] - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1; net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        rows.append((cols[j], IDX[t], IDX[e], x - e, raw, net, net_s, math.expm1(LEW[x] - LEW[e]), E50a[x - 1] / E50a[t] - 1, bool(SRC3[t:x + 1].any())))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'hold', 'raw', 'net', 'net_s', 'ew', 'b50', 'src3'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']; df['ab_gross'] = df['raw'] - df['ew']
    return df

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 1),
            'mean_raw_pct': round(df['raw'].mean() * 100, 3), 'mean_net_pct': round(df['net'].mean() * 100, 3), 'mean_EW_pct': round(df['ew'].mean() * 100, 3),
            'mean_ab_gross_pct': round(df['ab_gross'].mean() * 100, 3), 'mean_ab_net_pct': round(df['ab'].mean() * 100, 3),
            'median_ab_net_pct': round(df['ab'].median() * 100, 3), 't_month_ab_net': tstat(gm['ab'].mean()), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_stress_pct': round(df['ab_s'].mean() * 100, 3), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_net_vs_0050_pct': round(df['ab50'].mean() * 100, 3), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3)}

def ev_portfolio(ev, buy, sell):
    R = np.zeros((T, N), np.float32); W = np.zeros((T, N), np.float32); ent = np.zeros(T); exi = np.zeros(T)
    for j, t, e, x, done in ev:
        xe = x if done else T - 1
        W[e:xe + 1, j] += 1; R[e, j] = ACn[e, j] / AOn[e, j] - 1
        if xe > e: R[e + 1:xe + 1, j] = np.nan_to_num(ACn[e + 1:xe + 1, j] / ACn[e:xe, j] - 1)
        if done: R[x, j] = PX[x, j] / ACn[x - 1, j] - 1; exi[x] += 1
        ent[e] += 1
    W = np.minimum(W, 1); n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(n > 0, (W * R).sum(1) / np.where(n > 0, n, 1), 0.0)
        c = np.where(n > 0, (ent * buy + exi * sell) / np.where(n > 0, n, 1), 0.0)
    return pd.Series(g - c, index=IDX), pd.Series(n, index=IDX)

# ---------- 因子重疊 ----------
def factor_overlap(S):
    rows = []
    for k, d in enumerate(ME[:-1]):
        ex0 = LPOS[NEXT[d]]; ex1 = LPOS[NEXT[ME[k + 1]]]
        u = UNIV.loc[d].to_numpy() & np.isfinite(L.OPEN_RAW[ex0, LCOL])
        fr = L.AO[ex1, LCOL] / L.AO[ex0, LCOL] - 1
        s = S.loc[d].fillna(False).to_numpy().astype(float); mo = MOM.loc[d].to_numpy(); hi = S52.loc[d].to_numpy()
        ok = u & np.isfinite(fr) & np.isfinite(mo) & np.isfinite(hi)
        if ok.sum() < 50 or s[ok].sum() < 3 or s[ok].sum() == ok.sum(): continue
        df = pd.DataFrame({'s': s[ok], 'mo': mo[ok], 'hi': hi[ok], 'y': np.clip(fr[ok], -0.5, 1.0)})
        rm, rh = df['mo'].rank(pct=True), df['hi'].rank(pct=True)
        c_mo = df['s'].corr(rm); c_hi = df['s'].corr(rh)
        Xm = np.column_stack([np.ones(len(df)), df['s'], rm - 0.5, rh - 0.5])
        b = np.linalg.lstsq(Xm, df['y'].to_numpy(), rcond=None)[0]
        Xu = np.column_stack([np.ones(len(df)), df['s']]); bu = np.linalg.lstsq(Xu, df['y'].to_numpy(), rcond=None)[0]
        rows.append((d, c_mo, c_hi, bu[1], b[1], b[2], b[3], df['s'].mean(), rm[df['s'] == 1].mean(), rh[df['s'] == 1].mean()))
    return pd.DataFrame(rows, columns=['d', 'corr_mom', 'corr_52w', 'spread_raw', 'resid_sig', 'b_mom', 'b_52w', 'share', 'mom_pct_of_sig', 'hi_pct_of_sig']).set_index('d')

def fo_stats(fo, a, b):
    x = fo[(fo.index >= pd.Timestamp(a)) & (fo.index <= pd.Timestamp(b))]
    return {'n_months': int(len(x)), 'share_of_pool_signal': round(float(x['share'].mean()), 3),
            'corr_signal_vs_mom12_1_rank': round(float(x['corr_mom'].mean()), 3), 'corr_signal_vs_52w_high_rank': round(float(x['corr_52w'].mean()), 3),
            'signal_stocks_avg_mom_pctile': round(float(x['mom_pct_of_sig'].mean()), 3), 'signal_stocks_avg_52w_pctile': round(float(x['hi_pct_of_sig'].mean()), 3),
            'raw_spread_sig_minus_rest_pct_m': round(float(x['spread_raw'].mean() * 100), 3), 't_raw_spread': tstat(x['spread_raw']),
            'resid_sig_coef_pct_m': round(float(x['resid_sig'].mean() * 100), 3), 't_resid_sig': tstat(x['resid_sig']),
            'b_mom_pct_m': round(float(x['b_mom'].mean() * 100), 3), 't_b_mom': tstat(x['b_mom']),
            'b_52w_pct_m': round(float(x['b_52w'].mean() * 100), 3), 't_b_52w': tstat(x['b_52w'])}

def main():
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'first_rebalance_signal': str(ME[0].date()), 'last_rebalance_signal': str(ME[-1].date())}}
    t0 = LPOS[NEXT[ME[0]]]; t1 = len(L.IDX) - 1
    res = {}; navs = {}
    ew_nav = None
    for sn, S in (('SHORT', S_SHORT), ('LONG', S_LONG)):
        for mode in ('ALL', 'TOP30'):
            key = f'{sn}_{mode}'
            ev, pools, nb, info = monthly_events(S, mode)
            if ew_nav is None:
                ew_nav, _ = sim_portfolio(pools, 0.0, 0.0, t0, t1); ew_net, _ = sim_portfolio(pools, BUY, SELL, t0, t1)
                navs['EW_gross'] = ew_nav; navs['EW_net'] = ew_net
            nav, tov = sim_portfolio(ev, BUY, SELL, t0, t1)
            nav_s, _ = sim_portfolio(ev, BUY_S, SELL_S, t0, t1); nav_g, _ = sim_portfolio(ev, 0.0, 0.0, t0, t1)
            navs[key] = nav; navs[key + '_gross'] = nav_g
            r = nav.pct_change(); b50 = L.E50.reindex(nav.index).pct_change(); ew = ew_nav.pct_change()
            npos = pd.Series(info['held'].to_numpy(), index=[L.IDX[LPOS[NEXT[d]]] for d in info.index]).reindex(nav.index).ffill()
            o = {'blocked_gap_up': nb, 'turnover_value_per_rebalance': round(tov, 3),
                 'months_empty': int((info['held'] == 0).sum()),
                 'avg_signal_count': round(float(info['signal'].mean()), 1), 'avg_pool': round(float(info['pool'].mean()), 1),
                 'months_signal_gt30': int((info['signal'] > 30).sum())}
            for w, (a, b) in WIN.items():
                o[w] = pstats(r, b50, ew, a, b, npos)
                o[w]['CAGR_stress'] = pstats(nav_s.pct_change(), b50, ew, a, b)['CAGR']
                o[w]['CAGR_gross'] = pstats(nav_g.pct_change(), b50, ew, a, b)['CAGR']
                o[w]['t_gross_vs_EW_m'] = pstats(nav_g.pct_change(), b50, ew, a, b)['t_vs_EW_m']
                o[w]['EW_net_CAGR'] = pstats(navs['EW_net'].pct_change(), b50, ew, a, b)['CAGR']
                sub = info[(info.index >= pd.Timestamp(a)) & (info.index <= pd.Timestamp(b))]
                o[w]['avg_signal_count'] = round(float(sub['signal'].mean()), 1); o[w]['avg_held'] = round(float(sub['held'].mean()), 1)
            res[key] = o
            ps, pv = o['select_2005-01_2019-04'], o['valid_2019-05_2026-10']
            print(key, '| S', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 't50', ps['t_vs_0050_m'],
                  '| V', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'], 't50', pv['t_vs_0050_m'], flush=True)
            if mode == 'ALL':
                info.to_csv(HERE / f'monthly_counts_{sn}.csv')
    out['monthly'] = res
    # 日頻
    dres = {}
    for sn, S, X in (('SHORT', S_SHORT, X_SHORT), ('LONG', S_LONG, X_LONG)):
        ev, nb, nno = daily_events(S, X); df = event_table(ev)
        o = {'events_taken': len(ev), 'blocked_gap_up': nb, 'no_open': nno}
        r, npos = ev_portfolio(ev, BUY, SELL); rs, _ = ev_portfolio(ev, BUY_S, SELL_S); rg, _ = ev_portfolio(ev, 0.0, 0.0)
        for w, (a, b) in WIN.items():
            sub = df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))]
            o[f'event_{w}'] = ev_stats(sub)
            o[f'port_{w}'] = pstats(r, r50_clean, ew_cc, a, b, npos)
            o[f'port_{w}']['CAGR_stress'] = pstats(rs, r50_clean, ew_cc, a, b)['CAGR']; o[f'port_{w}']['CAGR_gross'] = pstats(rg, r50_clean, ew_cc, a, b)['CAGR']
        o['event_valid_ex_src3'] = ev_stats(df[(df['sig'] >= pd.Timestamp('2019-05-01')) & ~df['src3']])
        o['yearly_events'] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2)} for y, g in df.groupby(df['sig'].dt.year)}
        dres[f'DAILY_{sn}'] = o
        es, ev_ = o['event_select_2005-01_2019-04'], o['event_valid_2019-05_2026-10']
        print('DAILY', sn, 'n', len(df), '| S ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'CAGR', o['port_select_2005-01_2019-04']['CAGR'],
              '| V ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'), 'CAGR', o['port_valid_2019-05_2026-10']['CAGR'], flush=True)
        if sn == 'SHORT': df.to_csv(Path(L.P).parent / 'w1_bull_alignment_daily_short_events.csv.gz', index=False)
    out['daily'] = dres
    # 因子重疊
    fres = {}
    for sn, S in (('SHORT', S_SHORT), ('LONG', S_LONG)):
        fo = factor_overlap(S)
        fres[sn] = {w: fo_stats(fo, a, b) for w, (a, b) in WIN.items()}
        print('FO', sn, json.dumps(fres[sn], ensure_ascii=False), flush=True)
    out['factor_overlap'] = fres
    # 判定
    def passes(o):
        ok_ew = (o['excess_vs_EW_pct_m'] > 0) and ((o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9) >= 2)
        ok_50 = o['CAGR'] >= o['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['SHORT_ALL']
    vs = {'select': passes(pr['select_2005-01_2019-04']), 'valid': passes(pr['valid_2019-05_2026-10'])}
    k = vs['select']['pass'] + vs['valid']['pass']
    vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: {w: passes(o[w])['pass'] for w in ['select_2005-01_2019-04', 'valid_2019-05_2026-10']} for kk, o in res.items()}
    vs['daily_variants_pass'] = {kk: {w: passes(o[f'port_{w}'])['pass'] for w in ['select_2005-01_2019-04', 'valid_2019-05_2026-10']} for kk, o in dres.items()}
    out['verdict'] = vs
    pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()

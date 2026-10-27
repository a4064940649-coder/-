#!/usr/bin/env python3
"""Minervini 趨勢模板（8 條件）月換股（M2）＋日頻 MA50 出場變體＋與 52 週高點／12-1 動能的比較。
規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成交引擎價格陣列與成本（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'value', 'flag']}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if len(c) == 4 and c.isdigit() and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
F = D['flag']; bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); V = D['value']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
HIST = ADJ.notna().cumsum()
UNIV = (V60 >= 1e7) & (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna() & (HIST >= 260)

AF = ADJ.ffill(limit=10)
MA = {n: AF.rolling(n, min_periods=n).mean().where(ADJ.notna()) for n in (50, 150, 200)}
LO52 = AF.rolling(252, min_periods=240).min(); HI52 = AF.rolling(252, min_periods=240).max()
RS_SCORE = 0.4 * (AF / AF.shift(63) - 1) + 0.2 * (AF / AF.shift(126) - 1) + 0.2 * (AF / AF.shift(189) - 1) + 0.2 * (AF / AF.shift(252) - 1)
RS_SCORE = RS_SCORE.where(ADJ.notna())
RS_PCT = RS_SCORE.where(UNIV).rank(axis=1, pct=True) * 100
MOM = AF.shift(21) / AF.shift(252) - 1
S52 = ADJ / HI52
c = ADJ
COND = {
    'C1': (c > MA[150]) & (c > MA[200]),
    'C2': MA[150] > MA[200],
    'C3': MA[200] > MA[200].shift(21),
    'C3s': MA[200] > MA[200].shift(105),
    'C4': (MA[50] > MA[150]) & (MA[50] > MA[200]),
    'C5': c > MA[50],
    'C6': c >= 1.30 * LO52,
    'C6l': c >= 1.25 * LO52,
    'C7': c >= 0.75 * HI52,
    'C7s': c >= 0.85 * HI52,
    'C8': RS_PCT >= 70,
}
def tmpl(c3='C3', c6='C6', c7='C7'):
    keys = ['C1', 'C2', c3, 'C4', 'C5', c6, c7, 'C8']
    s = COND[keys[0]]
    for k in keys[1:]: s = s & COND[k]
    return s & UNIV
SIG = {'BASE': tmpl(), 'STRICT': tmpl(c3='C3s', c7='C7s'), 'LOOSE': tmpl(c6='C6l')}
EXIT50 = (ADJ < MA[50])

# ---------- 對應到 exlib 的完整日曆 ----------
LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[x] for x in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP_L = (L.AO_raw / L.AC_prev - 1) >= thrL
EXIT_L = np.zeros((len(L.IDX), L.N), bool)
_ex = EXIT50.reindex(L.IDX).fillna(False).to_numpy().astype(bool)
EXIT_L[:, LCOL] = _ex

_s = pd.Series(IDX, index=IDX)
ME = list(_s.groupby([IDX.year, IDX.month]).last())
ME = [d for d in ME if pd.Timestamp('2004-12-01') <= d <= pd.Timestamp('2026-09-30')]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in ME}

def pool_at(d):
    ex = LPOS[NEXT[d]]
    u = UNIV.loc[d].to_numpy()
    trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
    return ex, u & trad

def monthly_events(S, mode):
    events, pools, info = {}, {}, []
    held = set(); nblk = 0
    for d in ME:
        ex, pool = pool_at(d)
        sig = pool & S.loc[d].fillna(False).to_numpy()
        blocked = GAP_UP_L[ex, LCOL]
        idx = np.where(sig)[0]
        if mode == 'TOP20':
            r = RS_SCORE.loc[d].to_numpy()[idx]
            idx = idx[np.argsort(-np.nan_to_num(r, nan=-9e9), kind='stable')]
        pick = []
        for j in idx:
            if blocked[j] and LCOL[j] not in held: nblk += 1; continue
            pick.append(LCOL[j])
            if mode == 'TOP20' and len(pick) == 20: break
        events[ex] = ('open', np.array(pick, dtype=int)); pools[ex] = ('open', LCOL[pool])
        held = set(pick)
        info.append((d, int(pool.sum()), int(sig.sum()), len(pick)))
    return events, pools, nblk, pd.DataFrame(info, columns=['d', 'pool', 'signal', 'held']).set_index('d')

def topk_events(score, counts):
    """每月在同池取 score 前 counts[d] 檔（開盤漲停未持有者遞補）。"""
    events = {}; held = set()
    for d in ME:
        ex, pool = pool_at(d); k = counts.get(d, 0)
        s = score.loc[d].to_numpy(); ok = pool & np.isfinite(s)
        idx = np.where(ok)[0]; idx = idx[np.argsort(-s[idx], kind='stable')]
        blocked = GAP_UP_L[ex, LCOL]; pick = []
        for j in idx:
            if len(pick) >= k: break
            if blocked[j] and LCOL[j] not in held: continue
            pick.append(LCOL[j])
        events[ex] = ('open', np.array(pick, dtype=int)); held = set(pick)
    return events

def sim(events, buy, sell, t0, t1, ma50_exit=False):
    """exlib.sim_portfolio 同邏輯（d+1 開盤調回等權、漂移後市值計周轉）；加：空選股 → 全賣持現金；
    ma50_exit：非換股日開盤賣出前一日收盤 < MA50 的持股，資金留現金。"""
    u = np.zeros(L.N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []; npos = np.zeros(t1 - t0 + 1); nexit = 0
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]
            Pz = L.AO[t]
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
        elif ma50_exit and t > t0:
            sel = (u > 0) & EXIT_L[t - 1]
            if sel.any():
                val = (u[sel] * np.nan_to_num(L.AO[t, sel])).sum()
                cash += val * (1 - sell); u[sel] = 0; nexit += int(sel.sum())
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
        npos[t - t0] = int((u > 0).sum())
    sim.nexit = nexit
    return pd.Series(nav, index=L.IDX[t0:t1 + 1]), (float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')), pd.Series(npos, index=L.IDX[t0:t1 + 1])

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'), 'valid_survivor_2019-05_2023-12': ('2019-05-01', '2023-12-31'),
       'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31')}
MAINW = list(WIN)[:3]

def mo(x): return (1 + x).groupby(x.index.to_period('M')).prod() - 1

def pstats(r, bench, ew, a, b, npos=None):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    rr = r[m].fillna(0); bb = bench.reindex(rr.index).fillna(0); ee = ew.reindex(rr.index).fillna(0)
    yrs = (rr.index[-1] - r.index[max(np.argmax(m) - 1, 0)]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod(); return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}
    o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
    o['vol'] = round(float(rr.std() * math.sqrt(252)), 4)
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['n_months'] = int(len(ms))
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['hit_vs_EW_m'] = round(float((ms > me).mean()), 3)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    if npos is not None:
        nn = npos[m]; o['share_days_invested'] = round(float((nn > 0).mean()), 3)
        o['avg_positions_when_invested'] = round(float(nn[nn > 0].mean()), 1) if (nn > 0).any() else 0
    return o

def paired(ra, rb, a, b):
    m = (ra.index >= pd.Timestamp(a)) & (ra.index <= pd.Timestamp(b))
    x = mo(ra[m].fillna(0)) - mo(rb.reindex(ra.index)[m].fillna(0))
    return {'diff_pct_m': round(float(x.mean() * 100), 3), 't': tstat(x), 'n_months': int(len(x)), 'hit': round(float((x > 0).mean()), 3)}

# ---------- 因子重疊／Fama–MacBeth ----------
def factor_overlap(S):
    rows = []
    for k, d in enumerate(ME[:-1]):
        ex0, u = pool_at(d); ex1 = LPOS[NEXT[ME[k + 1]]]
        fr = L.AO[ex1, LCOL] / L.AO[ex0, LCOL] - 1
        s = S.loc[d].fillna(False).to_numpy().astype(float); mo_ = MOM.loc[d].to_numpy(); hi = S52.loc[d].to_numpy()
        ok = u & np.isfinite(fr) & np.isfinite(mo_) & np.isfinite(hi)
        if ok.sum() < 50 or s[ok].sum() < 3 or s[ok].sum() == ok.sum(): continue
        df = pd.DataFrame({'s': s[ok], 'mo': mo_[ok], 'hi': hi[ok], 'y': np.clip(fr[ok], -0.5, 1.0)})
        rm, rh = df['mo'].rank(pct=True), df['hi'].rank(pct=True)
        Xm = np.column_stack([np.ones(len(df)), df['s'], rm - 0.5, rh - 0.5])
        b = np.linalg.lstsq(Xm, df['y'].to_numpy(), rcond=None)[0]
        Xu = np.column_stack([np.ones(len(df)), df['s']]); bu = np.linalg.lstsq(Xu, df['y'].to_numpy(), rcond=None)[0]
        rows.append((d, df['s'].corr(rm), df['s'].corr(rh), bu[1], b[1], b[2], b[3], df['s'].mean(), rm[df['s'] == 1].mean(), rh[df['s'] == 1].mean()))
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
    BUY, SELL = L.cost_rates(0.001); BUY_S, SELL_S = L.cost_rates('stress')
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'first_rebalance_signal': str(ME[0].date()), 'last_rebalance_signal': str(ME[-1].date()), 'last_day': str(L.IDX[-1].date())}}
    # 條件通過率（月底、股池內）
    cr = {}
    for k, s in COND.items():
        v = [float(s.loc[d][UNIV.loc[d]].mean()) for d in ME if UNIV.loc[d].sum() > 0]
        cr[k] = round(float(np.nanmean(v)), 3)
    out['meta']['condition_pass_rate_in_universe_monthend'] = cr
    t0 = LPOS[NEXT[ME[0]]]; t1 = len(L.IDX) - 1
    res = {}; navs = {}; infos = {}
    _, pools, _, _ = monthly_events(SIG['BASE'], 'ALL')
    ew_nav, _, _ = sim(pools, 0.0, 0.0, t0, t1); ew_net, _, _ = sim(pools, BUY, SELL, t0, t1)
    navs['EW_gross'] = ew_nav; navs['EW_net'] = ew_net
    b50 = L.E50.reindex(ew_nav.index).pct_change(); ew = ew_nav.pct_change()
    for sn in ('BASE', 'STRICT', 'LOOSE'):
        for mode in ('ALL', 'TOP20'):
            ev, _, nb, info = monthly_events(SIG[sn], mode)
            infos[f'{sn}_{mode}'] = (ev, info)
            for ex_mode in ('ME', 'MA50'):
                key = f'{sn}_{mode}_{ex_mode}'; m50 = ex_mode == 'MA50'
                nav, tov, npos = sim(ev, BUY, SELL, t0, t1, m50); nexit = sim.nexit
                nav_s, _, _ = sim(ev, BUY_S, SELL_S, t0, t1, m50); nav_g, _, _ = sim(ev, 0.0, 0.0, t0, t1, m50)
                navs[key] = nav; navs[key + '_gross'] = nav_g
                r = nav.pct_change()
                o = {'blocked_gap_up': nb, 'turnover_value_per_rebalance': round(tov, 3), 'ma50_exits': nexit,
                     'months_empty': int((info['held'] == 0).sum()), 'n_months_total': int(len(info)),
                     'avg_signal_count': round(float(info['signal'].mean()), 1), 'avg_pool': round(float(info['pool'].mean()), 1),
                     'months_signal_gt20': int((info['signal'] > 20).sum())}
                for w, (a, b) in WIN.items():
                    o[w] = pstats(r, b50, ew, a, b, npos)
                    ps_ = pstats(nav_s.pct_change(), b50, ew, a, b); pg = pstats(nav_g.pct_change(), b50, ew, a, b)
                    o[w]['CAGR_stress'] = ps_['CAGR']; o[w]['t_stress_vs_EW_m'] = ps_['t_vs_EW_m']
                    o[w]['CAGR_gross'] = pg['CAGR']; o[w]['t_gross_vs_EW_m'] = pg['t_vs_EW_m']
                    o[w]['EW_net_CAGR'] = pstats(ew_net.pct_change(), b50, ew, a, b)['CAGR']
                    sub = info[(info.index >= pd.Timestamp(a)) & (info.index <= pd.Timestamp(b))]
                    o[w]['avg_signal_count'] = round(float(sub['signal'].mean()), 1); o[w]['avg_held_at_rebalance'] = round(float(sub['held'].mean()), 1)
                    o[w]['months_empty'] = int((sub['held'] == 0).sum())
                res[key] = o
                ps, pv, p2 = (o[w] for w in MAINW)
                print(key, '| S', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 't50', ps['t_vs_0050_m'],
                      '| V', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'], 't50', pv['t_vs_0050_m'],
                      '| 24-25', p2['CAGR'], p2['0050_CAGR'], p2['t_vs_EW_m'], p2['t_vs_0050_m'], 'avgsig', o['avg_signal_count'], flush=True)
            if mode == 'ALL': info.to_csv(HERE / f'monthly_counts_{sn}.csv')
    out['monthly'] = res
    # ---------- 對 52 週高點／12-1／RS-only 的增量 ----------
    va = {}
    ev_b, info_b = infos['BASE_ALL']
    counts = {d: int(n) for d, n in info_b['held'].items()}
    r_prim = navs['BASE_ALL_ME'].pct_change()
    for cname, score in (('52wHigh', S52), ('mom12_1', MOM), ('RS_only', RS_SCORE)):
        evc = topk_events(score.where(UNIV), counts)
        navc, tovc, _ = sim(evc, BUY, SELL, t0, t1); navs[f'matched_{cname}'] = navc
        rc = navc.pct_change()
        va[f'matched_{cname}'] = {'turnover': round(tovc, 3), **{w: {'comparator': {k: pstats(rc, b50, ew, a, b)[k] for k in ('CAGR', 'maxDD', 't_vs_EW_m', 't_vs_0050_m')},
                                                                    'primary_minus_comparator': paired(r_prim, rc, a, b)} for w, (a, b) in WIN.items()}}
    r_t20 = navs['BASE_TOP20_ME'].pct_change()
    c20 = {d: 20 for d in ME}
    for cname, score in (('52wHigh', S52), ('mom12_1', MOM)):
        evc = topk_events(score.where(UNIV), c20)
        navc, tovc, _ = sim(evc, BUY, SELL, t0, t1); navs[f'top20_{cname}'] = navc
        rc = navc.pct_change()
        va[f'top20_{cname}'] = {'turnover': round(tovc, 3), **{w: {'comparator': {k: pstats(rc, b50, ew, a, b)[k] for k in ('CAGR', 'maxDD', 't_vs_EW_m', 't_vs_0050_m')},
                                                                  'TOP20_minus_comparator': paired(r_t20, rc, a, b)} for w, (a, b) in WIN.items()}}
    out['value_add'] = va
    for k, v in va.items():
        print('VA', k, {w: (v[w]['comparator']['CAGR'], v[w].get('primary_minus_comparator', v[w].get('TOP20_minus_comparator'))) for w in MAINW}, flush=True)
    fo = factor_overlap(SIG['BASE'])
    out['factor_overlap_BASE'] = {w: fo_stats(fo, a, b) for w, (a, b) in WIN.items()}
    print('FO', json.dumps({w: out['factor_overlap_BASE'][w] for w in MAINW}, ensure_ascii=False), flush=True)
    # 判定
    def passes(o):
        t = o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9
        ok_ew = (o['excess_vs_EW_pct_m'] > 0) and (t >= 2)
        ok_50 = o['CAGR'] >= o['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['BASE_ALL_ME']
    vs = {'select': passes(pr['select_2005-01_2019-04']), 'valid': passes(pr['valid_2019-05_2026-10'])}
    k = vs['select']['pass'] + vs['valid']['pass']
    vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: {w: passes(o[w])['pass'] for w in MAINW[:2]} for kk, o in res.items()}
    out['verdict'] = vs
    pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()

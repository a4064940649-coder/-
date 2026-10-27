#!/usr/bin/env python3
"""Amihud ILLIQ 月換股（M2）＋變體 (a)(b)(c)(d)＋Lou-Shu Fama–MacBeth。規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成交引擎價格陣列與成本（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'value', 'flag', 'ret', 'src']}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if len(c) == 4 and c.isdigit() and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; N = len(cols)
F = D['flag']; ADJ = D['adj']; CR = D['close_raw']; V = D['value']; RET = D['ret']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
BASEU = (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()

valid = V.notna() & (V > 0) & RET.notna() & ~F.isin([3, 4, 5])
X = (RET.abs() / (V / 1e6)).where(valid)
VM = (V / 1e6).where(valid)
ILLIQ250 = X.rolling(250, min_periods=200).mean()
MV250 = VM.rolling(250, min_periods=200).mean()
VC = 1.0 / MV250
ILLIQ21 = X.rolling(21, min_periods=15).mean()

LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[x] for x in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP_L = (L.AO_raw / L.AC_prev - 1) >= thrL
FLAG345_L = np.zeros((len(L.IDX), L.N), bool)
FLAG345_L[:, LCOL] = F.isin([3, 4, 5]).reindex(L.IDX).fillna(False).to_numpy()

_s = pd.Series(IDX, index=IDX)
ME = list(_s.groupby([IDX.year, IDX.month]).last())
ME = [d for d in ME if pd.Timestamp('2004-12-01') <= d <= pd.Timestamp('2026-09-30')]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in ME}

def pool_at(d, thr):
    ex = LPOS[NEXT[d]]
    u = (BASEU.loc[d] & (V60.loc[d] >= thr)).to_numpy()
    trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
    return ex, u & trad

def resid_score(d, pool):
    il = ILLIQ250.loc[d].to_numpy(); mv = MV250.loc[d].to_numpy()
    ok = pool & np.isfinite(il) & np.isfinite(mv) & (il > 0)
    out = np.full(N, np.nan)
    if ok.sum() < 20: return out
    y = np.log(il[ok]); x = np.log(mv[ok]); Xm = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(Xm, y, rcond=None)[0]; out[ok] = y - Xm @ b
    return out

def score_at(name, d, pool):
    if name == 'A250': return ILLIQ250.loc[d].to_numpy()
    if name == 'A21': return ILLIQ21.loc[d].to_numpy()
    if name == 'VC': return VC.loc[d].to_numpy()
    if name == 'RES': return resid_score(d, pool)
    raise KeyError(name)

def quint(s, ok, nq=5):
    """回傳 ok 內各股組別 0..nq-1（0 = 最流動），與 ok 外 -1。"""
    g = np.full(N, -1); idx = np.where(ok)[0]
    r = pd.Series(s[idx]).rank(method='first')
    g[idx] = pd.qcut(r, nq, labels=False).to_numpy()
    return g

def build(score_name, thr=1e7, terc=False, group=4):
    """group=4 → Q5（遞補）；group in 0..3 → 診斷組（不遞補，剔除開盤漲停未持有）。"""
    events, pools, info = {}, {}, []
    held = set(); nblk = 0; ncand = 0
    for d in ME:
        ex, pool = pool_at(d, thr)
        s = score_at(score_name, d, pool)
        ok = pool & np.isfinite(s)
        if terc: ok &= np.isfinite(MV250.loc[d].to_numpy())
        if ok.sum() < 50: continue
        blocked = GAP_UP_L[ex, LCOL]
        if terc:
            mv = MV250.loc[d].to_numpy(); tg = quint(mv, ok, 3)
            target = []; order_all = []
            for k in range(3):
                okk = ok & (tg == k); gq = quint(s, okk, 5)
                mem = np.where(gq == group)[0]; target.append(len(mem))
                order_all.append(np.where(okk)[0][np.argsort(-s[okk], kind='stable')])
            pick = []
            for k in range(3):
                got = 0
                for j in order_all[k]:
                    if got >= target[k]: break
                    if blocked[j] and LCOL[j] not in held: nblk += 1; continue
                    pick.append(LCOL[j]); got += 1
            ncand += sum(target)
        else:
            gq = quint(s, ok, 5); nk = int((gq == group).sum()); ncand += nk
            if group == 4:
                idx = np.where(ok)[0]; idx = idx[np.argsort(-s[idx], kind='stable')]
                pick = []
                for j in idx:
                    if len(pick) >= nk: break
                    if blocked[j] and LCOL[j] not in held: nblk += 1; continue
                    pick.append(LCOL[j])
            else:
                mem = np.where(gq == group)[0]
                pick = [LCOL[j] for j in mem if not (blocked[j] and LCOL[j] not in held)]
                nblk += nk - len(pick)
        events[ex] = ('open', np.array(pick, dtype=int)); pools[ex] = ('open', LCOL[ok])
        held = set(pick)
        pv = V60.loc[d].to_numpy()[[np.where(LCOL == c)[0][0] for c in pick]] if pick else np.array([np.nan])
        info.append((d, int(ok.sum()), len(pick), float(np.nanmedian(pv)), float(np.nanpercentile(pv, 10)),
                     float(np.nanmedian(V60.loc[d].to_numpy()[ok]))))
    inf = pd.DataFrame(info, columns=['d', 'pool', 'held', 'v60_med', 'v60_p10', 'pool_v60_med']).set_index('d')
    return events, pools, inf, nblk, ncand

def sim(events, buy, sell, t0, t1):
    """exlib.sim_portfolio 同邏輯；另記錄持有期 flag 3/4/5 股數日。"""
    u = np.zeros(L.N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []; fl = 0; pdays = 0
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
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
        hm = u > 0; pdays += int(hm.sum()); fl += int((hm & FLAG345_L[t]).sum())
    sim.flag_share = fl / pdays if pdays else float('nan')
    return pd.Series(nav, index=L.IDX[t0:t1 + 1]), (float(np.mean(tov[1:])) if len(tov) > 1 else float('nan'))

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'), 'valid_survivor_2019-05_2023-12': ('2019-05-01', '2023-12-31'),
       'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31')}
MAINW = list(WIN)[:3]

def mo(x): return (1 + x).groupby(x.index.to_period('M')).prod() - 1

def pstats(r, bench, ew, a, b):
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
    return o

def fama_macbeth(thr=1e7):
    rows = []
    for k, d in enumerate(ME[:-1]):
        ex0, pool = pool_at(d, thr); ex1 = LPOS[NEXT[ME[k + 1]]]
        fr = L.AO[ex1, LCOL] / L.AO[ex0, LCOL] - 1
        il = ILLIQ250.loc[d].to_numpy(); vc = VC.loc[d].to_numpy(); rs = resid_score(d, pool)
        ok = pool & np.isfinite(fr) & np.isfinite(il) & np.isfinite(vc) & np.isfinite(rs)
        if ok.sum() < 50: continue
        y = np.clip(fr[ok], -0.5, 1.0)
        ri = pd.Series(il[ok]).rank(pct=True).to_numpy() - 0.5
        rv = pd.Series(vc[ok]).rank(pct=True).to_numpy() - 0.5
        rr = pd.Series(rs[ok]).rank(pct=True).to_numpy() - 0.5
        b1 = np.linalg.lstsq(np.column_stack([np.ones(len(y)), ri]), y, rcond=None)[0]
        b2 = np.linalg.lstsq(np.column_stack([np.ones(len(y)), rv, rr]), y, rcond=None)[0]
        bv = np.linalg.lstsq(np.column_stack([np.ones(len(y)), rv]), y, rcond=None)[0]
        br = np.linalg.lstsq(np.column_stack([np.ones(len(y)), rr]), y, rcond=None)[0]
        rows.append((d, b1[1], b2[1], b2[2], bv[1], br[1], np.corrcoef(ri, rv)[0, 1], np.corrcoef(ri, rr)[0, 1], int(ok.sum())))
    return pd.DataFrame(rows, columns=['d', 'b_illiq', 'b_vc', 'b_res', 'b_vc_only', 'b_res_only', 'corr_illiq_vc', 'corr_illiq_res', 'n']).set_index('d')

def fm_stats(fm, a, b):
    x = fm[(fm.index >= pd.Timestamp(a)) & (fm.index <= pd.Timestamp(b))]
    o = {'n_months': int(len(x)), 'avg_n': round(float(x['n'].mean()), 1),
         'corr_rank_illiq_vc': round(float(x['corr_illiq_vc'].mean()), 3), 'corr_rank_illiq_res': round(float(x['corr_illiq_res'].mean()), 3)}
    for c in ('b_illiq', 'b_vc', 'b_res', 'b_vc_only', 'b_res_only'):
        o[c + '_pct_m'] = round(float(x[c].mean() * 100), 3); o['t_' + c] = tstat(x[c])
    return o

def main():
    BUY, SELL = L.cost_rates(0.001); BUY_S, SELL_S = L.cost_rates('stress')
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'first_rebalance_signal': str(ME[0].date()), 'last_rebalance_signal': str(ME[-1].date()), 'last_day': str(L.IDX[-1].date())}}
    t0 = LPOS[NEXT[ME[0]]]; t1 = len(L.IDX) - 1
    src3 = (L.IDX >= pd.Timestamp('2024-02-15')) & (L.IDX <= pd.Timestamp('2025-09-23'))
    specs = {'A250_Q5': ('A250', 1e7, False, 4), 'A21_Q5': ('A21', 1e7, False, 4), 'VC_Q5': ('VC', 1e7, False, 4),
             'RES_Q5': ('RES', 1e7, False, 4), 'A250_TERC_Q5': ('A250', 1e7, True, 4), 'A250_Q5_T30M': ('A250', 3e7, False, 4)}
    for g in range(4): specs[f'A250_Q{g+1}_diag'] = ('A250', 1e7, False, g)
    ew = {}
    for thr in (1e7, 3e7):
        _, pools, _, _, _ = build('A250', thr)
        g_, _ = sim(pools, 0.0, 0.0, t0, t1); n_, _ = sim(pools, BUY, SELL, t0, t1)
        ew[thr] = (g_, n_)
    b50 = L.E50.reindex(ew[1e7][0].index).pct_change()
    res = {}; navs = {'EW_gross': ew[1e7][0], 'EW_net': ew[1e7][1], 'EW30M_gross': ew[3e7][0]}
    for key, (sn, thr, terc, grp) in specs.items():
        ev, _, info, nb, nc = build(sn, thr, terc, grp)
        nav, tov = sim(ev, BUY, SELL, t0, t1); fshare = sim.flag_share
        nav_s, _ = sim(ev, BUY_S, SELL_S, t0, t1); nav_g, _ = sim(ev, 0.0, 0.0, t0, t1)
        navs[key] = nav; navs[key + '_gross'] = nav_g
        ewr = ew[thr][0].pct_change(); ewn = ew[thr][1].pct_change()
        r = nav.pct_change()
        o = {'blocked_gap_up': nb, 'q_members_total': nc, 'blocked_share': round(nb / nc, 4) if nc else None,
             'turnover_value_per_rebalance': round(tov, 3), 'flag345_share_of_position_days': round(fshare, 5),
             'avg_pool': round(float(info['pool'].mean()), 1), 'avg_held': round(float(info['held'].mean()), 1)}
        for w, (a, b) in WIN.items():
            o[w] = pstats(r, b50, ewr, a, b)
            ps_ = pstats(nav_s.pct_change(), b50, ewr, a, b); pg = pstats(nav_g.pct_change(), b50, ewr, a, b)
            o[w]['CAGR_stress'] = ps_['CAGR']; o[w]['t_stress_vs_EW_m'] = ps_['t_vs_EW_m']; o[w]['t_stress_vs_0050_m'] = ps_['t_vs_0050_m']
            o[w]['CAGR_gross'] = pg['CAGR']; o[w]['t_gross_vs_EW_m'] = pg['t_vs_EW_m']
            o[w]['EW_net_CAGR'] = pstats(ewn, b50, ewr, a, b)['CAGR']
            sub = info[(info.index >= pd.Timestamp(a)) & (info.index <= pd.Timestamp(b))]
            o[w]['avg_pool'] = round(float(sub['pool'].mean()), 1); o[w]['avg_held'] = round(float(sub['held'].mean()), 1)
            o[w]['holding_v60_median_NTD_m'] = round(float(sub['v60_med'].median()) / 1e6, 1)
            o[w]['holding_v60_p10_NTD_m'] = round(float(sub['v60_p10'].median()) / 1e6, 1)
            o[w]['pool_v60_median_NTD_m'] = round(float(sub['pool_v60_med'].median()) / 1e6, 1)
            o[w]['capacity_AUM_NTD_m_10pct_p10'] = round(float((sub['held'] * sub['v60_p10'] * 0.1).median()) / 1e6, 1)
        res[key] = o
        ps, pv, p2 = (o[w] for w in MAINW)
        print(key, 'tov', o['turnover_value_per_rebalance'], 'blk', o['blocked_share'], 'held', o['avg_held'],
              '| S', ps['CAGR'], 'g', ps['CAGR_gross'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 't50', ps['t_vs_0050_m'],
              '| V', pv['CAGR'], 'g', pv['CAGR_gross'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'], 't50', pv['t_vs_0050_m'],
              '| 24-25', p2['CAGR'], p2['0050_CAGR'], p2['EW_CAGR'], p2['t_vs_EW_m'], p2['t_vs_0050_m'], flush=True)
        if key == 'A250_Q5': info.to_csv(HERE / 'monthly_counts_A250_Q5.csv')
    out['portfolios'] = res
    # Q5-Q1 毛報酬 spread
    sp = {}
    q5 = navs['A250_Q5_gross'].pct_change(); q1 = navs['A250_Q1_diag_gross'].pct_change()
    for w, (a, b) in WIN.items():
        m = (q5.index >= pd.Timestamp(a)) & (q5.index <= pd.Timestamp(b))
        x = mo(q5[m].fillna(0)) - mo(q1[m].fillna(0))
        sp[w] = {'Q5_minus_Q1_gross_pct_m': round(float(x.mean() * 100), 3), 't': tstat(x), 'n_months': int(len(x))}
    out['Q5_minus_Q1_gross'] = sp
    fm = fama_macbeth()
    out['fama_macbeth_lou_shu'] = {w: fm_stats(fm, a, b) for w, (a, b) in WIN.items()}
    for w in MAINW: print('FM', w, json.dumps(out['fama_macbeth_lou_shu'][w], ensure_ascii=False), flush=True)
    print('SPREAD', json.dumps(sp), flush=True)
    def passes(o):
        t = o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9
        ok_ew = (o['excess_vs_EW_pct_m'] > 0) and (t >= 2)
        ok_50 = o['CAGR'] >= o['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['A250_Q5']
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

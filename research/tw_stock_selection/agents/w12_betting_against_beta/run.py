#!/usr/bin/env python3
"""低 Beta／BAB（Frazzini–Pedersen 2014）只做多版月換股（M2）。規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
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
IDX = D['adj'].index; N = len(cols)
F = D['flag']; ADJ = D['adj']; CR = D['close_raw']; V = D['value']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
BASEU = (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()

# ---------- 報酬 ----------
simple = ADJ / ADJ.shift(1) - 1
bad = simple.isna() | F.isin([3, 4, 5]) | (simple.abs() > 0.25)
RS = simple.where(~bad)                       # 清理後簡單報酬
RL = np.log1p(RS)                             # 清理後對數報酬
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
MKT = {'TAIEX': np.log(tx.reindex(IDX) / tx.reindex(IDX).shift(1)), '0050': np.log(e50.reindex(IDX) / e50.reindex(IDX).shift(1))}

def r3(x):
    return x.rolling(3, min_periods=3).sum()

def roll_corr(X, y, w, minp):
    """X: DataFrame, y: Series；只用兩者同時有值的配對。"""
    Y = pd.DataFrame(np.repeat(y.to_numpy()[:, None], X.shape[1], 1), index=X.index, columns=X.columns)
    ok = X.notna() & Y.notna()
    Xz = X.where(ok, 0.0); Yz = Y.where(ok, 0.0)
    R = lambda Z: Z.rolling(w, min_periods=1).sum()
    n = R(ok.astype(float)); sx = R(Xz); sy = R(Yz); sxx = R(Xz * Xz); syy = R(Yz * Yz); sxy = R(Xz * Yz)
    cov = sxy - sx * sy / n; vx = sxx - sx ** 2 / n; vy = syy - sy ** 2 / n
    with np.errstate(invalid='ignore', divide='ignore'):
        c = cov / np.sqrt(vx * vy)
    return c.where(n >= minp)

_s = pd.Series(IDX, index=IDX)
ME = list(_s.groupby([IDX.year, IDX.month]).last())
ME = [d for d in ME if pd.Timestamp('2008-12-01') <= d <= pd.Timestamp('2026-09-30')]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in ME}
MEI = pd.DatetimeIndex(ME)

print('computing betas...', flush=True)
SIGI = RL.rolling(252, min_periods=120).std().reindex(MEI)
TVOL60 = RS.rolling(60, min_periods=45).std().reindex(MEI)
BETA, RHO, SIGM = {}, {}, {}
RL3 = r3(RL)
for mk, m in MKT.items():
    rho = roll_corr(RL3, r3(m), 1260, 750).reindex(MEI)
    sm = m.rolling(252, min_periods=120).std().reindex(MEI)
    bts = rho.mul(SIGI).div(sm, axis=0)
    BETA[mk] = 0.6 * bts + 0.4; RHO[mk] = rho; SIGM[mk] = sm
print('done', flush=True)

PREFIX_IND = {'11': '水泥工業', '12': '食品工業', '13': '塑膠工業', '14': '紡織纖維', '15': '電機機械', '16': '電器電纜',
              '17': '化學工業', '18': '玻璃陶瓷', '19': '造紙工業', '20': '鋼鐵工業', '21': '橡膠工業', '22': '汽車工業',
              '25': '建材營造業', '26': '航運業', '27': '觀光餐旅', '28': '金融保險業', '29': '貿易百貨業', '99': '其他業'}
_sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str}).drop_duplicates('code').set_index('code')['industry']
IND = np.array([(_sl[c] if (c in _sl.index and isinstance(_sl[c], str)) else PREFIX_IND.get(c[:2], '未分類')) for c in cols])

LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[x] for x in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP_L = (L.AO_raw / L.AC_prev - 1) >= thrL

POOL, EXI = {}, {}
for d in ME:
    ex = LPOS[NEXT[d]]
    u = (BASEU.loc[d] & (V60.loc[d] >= 1e7)).to_numpy()
    trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
    POOL[d] = u & trad; EXI[d] = ex

def quint(s, ok, nq=5):
    g = np.full(N, -1); idx = np.where(ok)[0]
    r = pd.Series(s[idx]).rank(method='first')
    g[idx] = pd.qcut(r, nq, labels=False).to_numpy()
    return g

def build(spec):
    """events[ex] = ('open', 面板欄位 idx, 權重)。另回傳每期事前 β（加權）與診斷資訊。"""
    mk = spec.get('mkt', 'TAIEX'); kind = spec['kind']
    events, pools, info = {}, {}, []
    held = set(); nblk = 0; ncand = 0; nskip = 0
    for d in ME:
        ex = EXI[d]
        b = BETA[mk].loc[d].to_numpy()
        ok = POOL[d] & np.isfinite(b)
        if kind == 'tvol':
            tv = TVOL60.loc[d].to_numpy(); ok = ok & np.isfinite(tv)
        if ok.sum() < 50:
            nskip += 1; continue
        blocked = GAP_UP_L[ex, LCOL]
        idx = np.where(ok)[0]
        if kind in ('q', 'tvol'):
            s = b if kind == 'q' else tv
            gq = quint(s, ok, 5); grp = spec['grp']
            mem = np.where(gq == grp)[0]; ncand += len(mem)
            if spec.get('sub', False):
                order = idx[np.argsort(s[idx], kind='stable')]
                pick = []
                for j in order:
                    if len(pick) >= len(mem): break
                    if blocked[j] and LCOL[j] not in held: nblk += 1; continue
                    pick.append(j)
            else:
                pick = [j for j in mem if not (blocked[j] and LCOL[j] not in held)]; nblk += len(mem) - len(pick)
            pick = np.array(pick, dtype=int); w = np.full(len(pick), 1.0 / max(len(pick), 1))
        elif kind == 'lowN':
            order = idx[np.argsort(b[idx], kind='stable')]; pick = []; ncand += spec['n']
            for j in order:
                if len(pick) >= spec['n']: break
                if blocked[j] and LCOL[j] not in held: nblk += 1; continue
                pick.append(j)
            pick = np.array(pick, dtype=int); w = np.full(len(pick), 1.0 / len(pick))
        elif kind == 'rw':
            z = pd.Series(b[idx]).rank(method='average').to_numpy(); zb = z.mean()
            raw = np.maximum(zb - z, 0) if spec['leg'] == 'L' else np.maximum(z - zb, 0)
            keep = raw > 0; cand = idx[keep]; raw = raw[keep]; ncand += len(cand)
            okb = np.array([not (blocked[j] and LCOL[j] not in held) for j in cand], bool)
            nblk += int((~okb).sum())
            pick = cand[okb]; w = raw[okb] / raw[okb].sum()
        events[ex] = ('open', LCOL[pick], w); pools[ex] = ('open', LCOL[idx], np.full(len(idx), 1.0 / len(idx)))
        held = set(LCOL[pick])
        info.append((d, int(ok.sum()), len(pick), float(np.sum(w * b[pick])), float(np.nanmedian(V60.loc[d].to_numpy()[pick])) / 1e6,
                     float(np.nanmedian(CR.loc[d].to_numpy()[pick])), float(np.sum(w * SIGI.loc[d].to_numpy()[pick])) * math.sqrt(252),
                     float(np.sum(w * RHO[mk].loc[d].to_numpy()[pick])), [cols[j] for j in pick]))
    inf = pd.DataFrame(info, columns=['d', 'pool', 'held', 'beta_exante', 'v60_med_m', 'price_med', 'sigma_ann_w', 'rho_w', 'members']).set_index('d')
    return events, pools, inf, nblk, ncand, nskip

def sim(events, buy, sell, t0, t1):
    """exlib.sim_portfolio 同邏輯，支援目標權重。"""
    u = np.zeros(L.N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk, wt = events[t]
            Pz = L.AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash
            if len(pk) == 0:
                cash = Vt - sell * h.sum(); tov.append(h.sum() / Vt if Vt > 0 else 0); u = np.zeros(L.N)
            else:
                tgt = np.zeros(L.N); Vn = Vt
                for _ in range(6):
                    tgt[:] = 0; tgt[pk] = wt * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                    Vn = Vt - cost
                tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
                u = np.zeros(L.N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
    return pd.Series(nav, index=L.IDX[t0:t1 + 1]), (float(np.mean(tov[1:])) if len(tov) > 1 else float('nan'))

PRE = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['PREREG']
WIN = {k: tuple(v) for k, v in PRE['windows'].items()}
MAINW = list(WIN)[:3]
RF_D = 0.01 / 252
RF_M = 1.01 ** (1 / 12) - 1

def mo(x): return (1 + x).groupby(x.index.to_period('M')).prod() - 1

def capm(ms, mb):
    y = ms - RF_M; x = mb - RF_M
    X = np.column_stack([np.ones(len(x)), x.to_numpy()]); yy = y.to_numpy()
    b, *_ = np.linalg.lstsq(X, yy, rcond=None); e = yy - X @ b
    s2 = (e @ e) / (len(yy) - 2); cov = s2 * np.linalg.inv(X.T @ X)
    r2 = 1 - (e @ e) / ((yy - yy.mean()) @ (yy - yy.mean()))
    return {'alpha_pct_m': round(float(b[0] * 100), 3), 'alpha_ann_pct': round(float(((1 + b[0]) ** 12 - 1) * 100), 2),
            't_alpha': round(float(b[0] / math.sqrt(cov[0, 0])), 2), 'beta_real': round(float(b[1]), 3), 'R2': round(float(r2), 3)}

def pstats(r, bench, ew, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    rr = r[m].fillna(0); bb = bench.reindex(rr.index).fillna(0); ee = ew.reindex(rr.index).fillna(0)
    yrs = (rr.index[-1] - r.index[max(np.argmax(m) - 1, 0)]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod(); return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}
    o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
    o['vol'] = round(float(rr.std() * math.sqrt(252)), 4); o['0050_vol'] = round(float(bb.std() * math.sqrt(252)), 4)
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['n_months'] = int(len(ms))
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['hit_vs_EW_m'] = round(float((ms > me).mean()), 3)
    o['sharpe'] = round(float((ms - RF_M).mean() / (ms - RF_M).std() * math.sqrt(12)), 3)
    o['0050_sharpe'] = round(float((mb - RF_M).mean() / (mb - RF_M).std() * math.sqrt(12)), 3)
    o['capm_vs_0050'] = capm(ms, mb)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    return o

def lev_from(nav, info):
    """r_lev = rf + (r − rf)/β̄，β̄ 自換股執行日起整月固定。"""
    r = nav.pct_change()
    bser = pd.Series(np.nan, index=r.index)
    for d, row in info.iterrows():
        bser.loc[L.IDX[LPOS[NEXT[d]]]] = row['beta_exante']
    bser = bser.ffill()
    return RF_D + (r - RF_D) / bser, bser

def series_stats(x, a, b):
    m = (x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b)); xx = x[m].fillna(0); ms = mo(xx)
    eq = (1 + xx + RF_D).cumprod(); yrs = len(xx) / 252
    return {'mean_pct_m': round(float(ms.mean() * 100), 3), 't': tstat(ms), 'vol_ann': round(float(ms.std() * math.sqrt(12)), 4),
            'sharpe': round(float(ms.mean() / ms.std() * math.sqrt(12)), 3), 'cash_plus_BAB_CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4),
            'maxDD_cash_plus_BAB': round(float((eq / eq.cummax() - 1).min()), 4),
            'yearly_pct': {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in xx.groupby(xx.index.year)}, 'n_months': int(len(ms))}

def main():
    COST = {'base': L.cost_rates(0.001), 'c05': L.cost_rates(0.0), 'stress': L.cost_rates('stress')}
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'costs_buy_sell': COST, 'rf_annual': 0.01,
                    'first_rebalance_signal': str(ME[0].date()), 'last_rebalance_signal': str(ME[-1].date()), 'last_day': str(L.IDX[-1].date())}}
    t0 = LPOS[NEXT[ME[0]]]; t1 = len(L.IDX) - 1
    specs = {
        'Q1_EW_TAIEX': {'kind': 'q', 'grp': 0, 'sub': True},
        'LOW30_EW_TAIEX': {'kind': 'lowN', 'n': 30},
        'RW_LOW_TAIEX': {'kind': 'rw', 'leg': 'L'},
        'RW_HIGH_TAIEX': {'kind': 'rw', 'leg': 'H'},
        'Q2_EW_TAIEX': {'kind': 'q', 'grp': 1}, 'Q3_EW_TAIEX': {'kind': 'q', 'grp': 2},
        'Q4_EW_TAIEX': {'kind': 'q', 'grp': 3}, 'Q5_EW_TAIEX': {'kind': 'q', 'grp': 4},
        'Q1_EW_0050': {'kind': 'q', 'grp': 0, 'sub': True, 'mkt': '0050'},
        'Q5_EW_0050': {'kind': 'q', 'grp': 4, 'mkt': '0050'},
        'RW_LOW_0050': {'kind': 'rw', 'leg': 'L', 'mkt': '0050'},
        'RW_HIGH_0050': {'kind': 'rw', 'leg': 'H', 'mkt': '0050'},
        'TVOL60_Q1': {'kind': 'tvol', 'grp': 0, 'sub': True},
    }
    b50 = L.E50.reindex(L.IDX[t0:t1 + 1]).pct_change()
    ewcache = {}; res = {}; navs = {}; infos = {}
    for key, spec in specs.items():
        ev, pools, info, nb, nc, nskip = build(spec)
        infos[key] = info
        pk = (spec.get('mkt', 'TAIEX'), spec['kind'] == 'tvol')
        if pk not in ewcache:
            g_, _ = sim(pools, 0.0, 0.0, t0, t1); ewcache[pk] = g_; navs[f'EW_{pk[0]}{"_tv" if pk[1] else ""}'] = g_
        ewr = ewcache[pk].pct_change()
        nav, tov = sim(ev, *COST['base'], t0, t1)
        nav_s, _ = sim(ev, *COST['stress'], t0, t1); nav_c, _ = sim(ev, *COST['c05'], t0, t1); nav_g, _ = sim(ev, 0.0, 0.0, t0, t1)
        navs[key] = nav; navs[key + '_gross'] = nav_g; navs[key + '_stress'] = nav_s
        r = nav.pct_change()
        o = {'blocked_gap_up': nb, 'cand_total': nc, 'blocked_share': round(nb / nc, 4) if nc else None, 'months_skipped_lt50': nskip,
             'turnover_value_per_rebalance': round(tov, 3)}
        for w, (a, b) in WIN.items():
            o[w] = pstats(r, b50, ewr, a, b)
            ps_ = pstats(nav_s.pct_change(), b50, ewr, a, b); pg = pstats(nav_g.pct_change(), b50, ewr, a, b); pc = pstats(nav_c.pct_change(), b50, ewr, a, b)
            o[w]['CAGR_stress'] = ps_['CAGR']; o[w]['t_stress_vs_EW_m'] = ps_['t_vs_EW_m']; o[w]['t_stress_vs_0050_m'] = ps_['t_vs_0050_m']
            o[w]['CAGR_c05'] = pc['CAGR']; o[w]['t_c05_vs_EW_m'] = pc['t_vs_EW_m']
            o[w]['CAGR_gross'] = pg['CAGR']; o[w]['t_gross_vs_EW_m'] = pg['t_vs_EW_m']; o[w]['excess_gross_vs_EW_pct_m'] = pg['excess_vs_EW_pct_m']
            o[w]['capm_gross_vs_0050'] = pg['capm_vs_0050']
            sub = info[(info.index >= pd.Timestamp(a) - pd.Timedelta(days=31)) & (info.index <= pd.Timestamp(b))]
            o[w]['avg_pool'] = round(float(sub['pool'].mean()), 1); o[w]['avg_held'] = round(float(sub['held'].mean()), 1)
            o[w]['beta_exante_avg'] = round(float(sub['beta_exante'].mean()), 3)
            o[w]['sigma_ann_avg'] = round(float(sub['sigma_ann_w'].mean()), 3); o[w]['rho_avg'] = round(float(sub['rho_w'].mean()), 3)
            o[w]['holding_v60_median_NTD_m'] = round(float(sub['v60_med_m'].median()), 1)
            o[w]['holding_price_median'] = round(float(sub['price_med'].median()), 1)
        res[key] = o
        ps, pv, p2 = (o[w] for w in MAINW)
        print(f"{key:15s} tov {o['turnover_value_per_rebalance']} blk {o['blocked_share']} skip {nskip}",
              '| S', ps['CAGR'], 'g', ps['CAGR_gross'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 't50', ps['t_vs_0050_m'], 'DD', ps['maxDD'], 'bx', ps['beta_exante_avg'], 'br', ps['capm_vs_0050']['beta_real'], 'a', ps['capm_vs_0050']['alpha_pct_m'], ps['capm_vs_0050']['t_alpha'],
              '| V', pv['CAGR'], 'g', pv['CAGR_gross'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'], 't50', pv['t_vs_0050_m'], 'DD', pv['maxDD'], 'br', pv['capm_vs_0050']['beta_real'], 'a', pv['capm_vs_0050']['alpha_pct_m'], pv['capm_vs_0050']['t_alpha'],
              '| 24-25', p2['CAGR'], p2['0050_CAGR'], p2['EW_CAGR'], p2['t_vs_EW_m'], p2['t_vs_0050_m'], flush=True)
    out['portfolios'] = res
    infos['Q1_EW_TAIEX'].drop(columns='members').to_csv(HERE / 'monthly_counts_Q1_EW_TAIEX.csv')

    # 槓桿到 β=1（理論）
    lev = {}
    for key in ['Q1_EW_TAIEX', 'RW_LOW_TAIEX', 'LOW30_EW_TAIEX', 'Q1_EW_0050']:
        for tag, suf in (('net', ''), ('gross', '_gross')):
            rl, bser = lev_from(navs[key + suf], infos[key])
            nv = (1 + rl.fillna(0)).cumprod(); navs[f'LEV_{key}{suf}'] = nv
            ewr = ewcache[('0050' if key.endswith('0050') else 'TAIEX', False)].pct_change()
            lev[f'LEV_{key}_{tag}'] = {w: pstats(rl, b50, ewr, a, b) for w, (a, b) in WIN.items()}
            for w, (a, b) in WIN.items():
                bb = bser[(bser.index >= pd.Timestamp(a)) & (bser.index <= pd.Timestamp(b))]
                lev[f'LEV_{key}_{tag}'][w]['leverage_avg'] = round(float((1 / bb).mean()), 3)
            q = lev[f'LEV_{key}_{tag}']
            print('LEV', key, tag, {w: (q[w]['CAGR'], q[w]['0050_CAGR'], q[w]['maxDD'], q[w]['t_vs_0050_m'], q[w]['capm_vs_0050']['beta_real'], q[w]['leverage_avg']) for w in MAINW}, flush=True)
    out['levered_to_beta1'] = lev

    # BAB 理論多空
    bab = {}
    for name, lk, hk in (('BAB_RW_TAIEX', 'RW_LOW_TAIEX', 'RW_HIGH_TAIEX'), ('BAB_Q_TAIEX', 'Q1_EW_TAIEX', 'Q5_EW_TAIEX'),
                         ('BAB_RW_0050', 'RW_LOW_0050', 'RW_HIGH_0050'), ('BAB_Q_0050', 'Q1_EW_0050', 'Q5_EW_0050')):
        for tag, suf in (('gross', '_gross'), ('net_legs', '')):
            rl, bl = lev_from(navs[lk + suf], infos[lk]); rh, bh = lev_from(navs[hk + suf], infos[hk])
            x = (rl - RF_D) - (rh - RF_D)     # = (rL−rf)/βL − (rH−rf)/βH
            bab[f'{name}_{tag}'] = {w: series_stats(x, a, b) for w, (a, b) in WIN.items()}
            for w, (a, b) in WIN.items():
                m = (bl.index >= pd.Timestamp(a)) & (bl.index <= pd.Timestamp(b))
                bab[f'{name}_{tag}'][w]['beta_L_avg'] = round(float(bl[m].mean()), 3); bab[f'{name}_{tag}'][w]['beta_H_avg'] = round(float(bh[m].mean()), 3)
                ms = mo(x[m].fillna(0)); mb = mo(b50[m].fillna(0))
                bab[f'{name}_{tag}'][w]['realized_beta_vs_0050'] = round(float(np.polyfit(mb.to_numpy() - RF_M, ms.to_numpy(), 1)[0]), 3)
            q = bab[f'{name}_{tag}']
            print('BAB', name, tag, {w: (q[w]['mean_pct_m'], q[w]['t'], q[w]['sharpe'], q[w]['realized_beta_vs_0050']) for w in MAINW}, flush=True)
    out['BAB_long_short_theory'] = bab

    # 重疊率與相關
    rows = []
    for d in ME:
        b = BETA['TAIEX'].loc[d].to_numpy(); tv = TVOL60.loc[d].to_numpy()
        ok = POOL[d] & np.isfinite(b) & np.isfinite(tv)
        if ok.sum() < 50: continue
        gb = quint(b, ok); gt = quint(tv, ok)
        q1b = set(np.where(gb == 0)[0]); q1t = set(np.where(gt == 0)[0]); q5b = set(np.where(gb == 4)[0]); q5t = set(np.where(gt == 4)[0])
        b0 = BETA['0050'].loc[d].to_numpy(); ok0 = ok & np.isfinite(b0); g0 = quint(b0, ok0)
        s = pd.DataFrame({'b': b[ok], 'tv': tv[ok], 'sig': SIGI.loc[d].to_numpy()[ok], 'rho': RHO['TAIEX'].loc[d].to_numpy()[ok]}).rank()
        rows.append({'d': d, 'n': int(ok.sum()), 'overlap_Q1beta_Q1tvol': len(q1b & q1t) / len(q1b), 'overlap_Q5beta_Q5tvol': len(q5b & q5t) / len(q5b),
                     'overlap_Q1_taiex_vs_0050beta': len(q1b & set(np.where(g0 == 0)[0])) / len(q1b),
                     'sp_beta_tvol60': s['b'].corr(s['tv']), 'sp_beta_sigma1y': s['b'].corr(s['sig']), 'sp_beta_rho': s['b'].corr(s['rho']),
                     'sp_rho_tvol60': s['rho'].corr(s['tv']),
                     'beta_med': float(np.median(b[ok])), 'beta_q1_max': float(np.max(b[list(q1b)])), 'rho_med': float(np.median(RHO['TAIEX'].loc[d].to_numpy()[ok]))})
    ov = pd.DataFrame(rows).set_index('d'); ov.to_csv(HERE / 'overlap_monthly.csv')
    out['overlap'] = {w: {k: round(float(ov.loc[(ov.index >= pd.Timestamp(a) - pd.Timedelta(days=31)) & (ov.index <= pd.Timestamp(b)), k].mean()), 3) for k in ov.columns}
                      for w, (a, b) in WIN.items()}
    print('OVERLAP', json.dumps({w: out['overlap'][w] for w in MAINW}, ensure_ascii=False), flush=True)

    # Q1 產業組成
    cnt = {}
    sel = infos['Q1_EW_TAIEX']
    for w, (a, b) in list(WIN.items())[:2]:
        sub = sel[(sel.index >= pd.Timestamp(a) - pd.Timedelta(days=31)) & (sel.index <= pd.Timestamp(b))]
        allm = [IND[cols.index(c)] for mem in sub['members'] for c in mem]
        vc = pd.Series(allm).value_counts(normalize=True).head(8)
        cnt[w] = {k: round(float(v), 3) for k, v in vc.items()}
        topn = pd.Series([c for mem in sub['members'] for c in mem]).value_counts().head(10)
        cnt[w + '_most_held'] = {k: int(v) for k, v in topn.items()}
    out['Q1_industry_mix'] = cnt
    print('IND', json.dumps(cnt, ensure_ascii=False), flush=True)

    def passes(o):
        t = o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9
        ok_ew = (o['excess_vs_EW_pct_m'] > 0) and (t >= 2)
        ok_50 = o['CAGR'] >= o['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    SW, VW = MAINW[0], MAINW[1]
    vs = {'primary': 'Q1_EW_TAIEX', 'select': passes(res['Q1_EW_TAIEX'][SW]), 'valid': passes(res['Q1_EW_TAIEX'][VW])}
    k = vs['select']['pass'] + vs['valid']['pass']
    vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: {w: passes(o[w])['pass'] for w in MAINW[:2]} for kk, o in res.items()}
    vs['levered_pass'] = {kk: {w: passes(o[w])['pass'] for w in MAINW[:2]} for kk, o in lev.items()}
    out['verdict'] = vs
    pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()

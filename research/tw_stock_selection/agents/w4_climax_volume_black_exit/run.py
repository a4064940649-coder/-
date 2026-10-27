#!/usr/bin/env python3
"""高檔爆量長黑出場（疊加在主策略 M2 上）。規則見 results.json 的 PREREG（prereg.py，事前寫入）。
主策略選股與換股引擎沿用 revenue_longhistory/lhlib.py（import，只讀取）；成本用 lhlib.cost_rates（= exec_rebuild/exlib）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'revenue_longhistory'))
import lhlib as L  # noqa: E402
from btlib import tstat  # noqa: E402

IDX, COLS, N = L.IDX, L.COLS, L.N
pos = L.pos
AO, AC, OPEN_RAW, ADJ_RAW = L.AO, L.AC, L.OPEN_RAW, L.ADJ_RAW
GU = L.GAP_UP_OPEN_ERA
GD = L.GAPRET <= -L.LIM_ERA + L.LTOL                      # 開盤跌停
E50n = L.E50.to_numpy()
FLAG = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS)
SRC = pd.read_parquet(L.P / 'src.parquet').reindex(index=IDX, columns=COLS)

# ---------------- 觸發指標（剔除有值 < 300 檔的日子後再滾動） ----------------
G = L.GOOD
F = FLAG.loc[G]; bad5 = F == 5
A = L.adj.loc[G].mask(bad5); Cr = L.C.loc[G].mask(bad5); Op = L.O.loc[G].mask(bad5); Vv = L.V.loc[G]; Rg = L.RET.loc[G].mask(bad5)
VOL = (Vv / Cr).where((Vv > 0) & (Cr > 0))
MIN60 = A.shift(1).rolling(60, min_periods=48).min()
C1 = (A / MIN60 - 1) >= 0.30
VAVG = {'V20': (VOL.shift(1).rolling(20, min_periods=16).mean(), 2.5),
        'V240': (VOL.shift(1).rolling(240, min_periods=192).mean(), 2.5),
        'V5': (VOL.shift(1).rolling(5, min_periods=4).mean(), 2.0)}
C2 = {k: VOL >= m * avg for k, (avg, m) in VAVG.items()}
fok = F.isin([0, 1, 2])
AOg = Op * A / Cr; prevC = A.ffill().shift(1)
C3 = {'BLK': (Cr < Op) & (Rg <= -0.015), 'OHLC': (AOg > prevC) & (Cr < Op)}
SPECS = [f'{v}x{b}' for v in ('V20', 'V240', 'V5') for b in ('BLK', 'OHLC')]
PRIMARY = 'V20xBLK'
TRIG = {}
for s in SPECS:
    v, b = s.split('x')
    m = (C1 & C2[v] & C3[b] & fok & A.notna()).reindex(IDX, fill_value=False).fillna(False)
    TRIG[s] = m.to_numpy(bool)
IS_SRC3 = (SRC == 3).to_numpy()


# ---------------- 主策略排名清單（逐字照 lhlib.Setup.selection 的合格條件與排序） ----------------
def rank_lists(S):
    el = S.t300 & S.s52.notna() & ~S.inD & S.yoy3.notna()
    sc = 0.5 * S.s52.where(el).rank(axis=1, pct=True) + 0.5 * S.yoy3.where(el).rank(axis=1, pct=True)
    out = {}
    for d in S.dates[:-1]:
        i = pos(d); ex = i + 1
        okb = el.loc[d].to_numpy()
        trad = np.isfinite(OPEN_RAW[ex]) & np.isfinite(ADJ_RAW[ex])
        ok = okb & trad & np.isfinite(sc.loc[d].to_numpy())
        if ok.sum() < 50: continue
        order = sc.loc[d].to_numpy().copy(); order[~ok] = np.nan
        cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
        out[ex] = (cand, np.where(ok)[0])
    return out


def sim(RL, n, trig, buy, sell, t0, t1, idle='cash', slip=0.001):
    """回傳 dict(nav, sold, cost, events, picks)。trig=None → 主策略不加出場（應與 lhlib.sim_portfolio 一致）。"""
    if slip == 'stress': b50 = s50 = 0.005
    else: b50 = L.FEE + slip; s50 = L.FEE + 0.001 + slip
    reb = set(RL)
    u = np.zeros(N); cash = 1.0; e50u = 0.0; pending = {}
    nav = np.empty(t1 - t0 + 1); sold = np.zeros(t1 - t0 + 1); cst = np.zeros(t1 - t0 + 1)
    events, picks, tov = [], {}, []
    last_pool = None
    for t in range(t0, t1 + 1):
        k = t - t0
        if t in reb:
            cand, pool = RL[t]
            if e50u > 0:                                   # 0050 於訊號日（t−1）收盤賣出
                v = e50u * E50n[t - 1]; cash += v * (1 - s50); cst[k] += v * s50; e50u = 0.0
            held = set(np.where(u > 0)[0])
            pk = []
            for j in cand:
                if j in pending: continue
                if GU[t, j] and j not in held: continue
                pk.append(j)
                if len(pk) == n: break
            for j, (tt, _) in pending.items():           # 換股日一併賣出（換股引擎會賣掉不在目標的持股）
                events.append({'trig': tt, 'sell': t, 'j': j, 'px': AO[t, j], 'pool': pool, 'at_reb': True})
            pending = {}
            pk = np.array(pk)
            Pz = AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash
            tgt = np.zeros(N); w = 1.0 / len(pk); Vn = Vt
            for _ in range(6):
                tgt[:] = 0; tgt[pk] = w * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            sold[k] += np.clip(h - tgt, 0, None).sum(); cst[k] += cost
            u = np.zeros(N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
            picks[t] = set(pk.tolist()); last_pool = pool
        elif pending:
            proceeds = 0.0
            for j in list(pending):
                if np.isfinite(OPEN_RAW[t, j]) and not GD[t, j]:
                    v = u[j] * AO[t, j]; proceeds += v * (1 - sell); cst[k] += v * sell; sold[k] += v
                    events.append({'trig': pending[j][0], 'sell': t, 'j': j, 'px': AO[t, j], 'pool': last_pool, 'at_reb': False})
                    u[j] = 0.0; del pending[j]
            cash += proceeds
            if idle == '0050' and cash > 0 and (t + 1) not in reb and t < t1:
                e50u += cash * (1 - b50) / E50n[t]; cst[k] += cash * b50; cash = 0.0
        nav[k] = np.nansum(u * AC[t]) + cash + e50u * E50n[t]
        if trig is not None:
            for j in np.where((u > 0) & trig[t])[0]:
                if j not in pending: pending[j] = (t, None)
    out = {'nav': pd.Series(nav, index=IDX[t0:t1 + 1]), 'sold': pd.Series(sold, index=IDX[t0:t1 + 1]),
           'cost': pd.Series(cst, index=IDX[t0:t1 + 1]), 'events': events, 'picks': picks,
           'tov': float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')}
    return out


# ---------------- 統計 ----------------
def month_rets(nav, DATES):
    nd = nav.reindex(DATES); return (nd.shift(-1) / nd - 1).iloc[:-1]

def in_segs(ix, segs):
    m = np.zeros(len(ix), bool)
    for a, b in segs: m |= (ix >= pd.Timestamp(a)) & (ix <= pd.Timestamp(b))
    return m

def years_of(nav, segs):
    m = in_segs(nav.index, segs); return m.sum() / 252.0

def port_stats(res, base, segs, DATES, ew):
    o = L.daily_stats(res['nav'], segs, DATES, L.E50, ew)
    if o is None: return None
    m = in_segs(res['nav'].index, segs); yrs = m.sum() / 252.0; avgnav = res['nav'][m].mean()
    o['turnover_per_year'] = round(float((res['sold'][m] / res['nav'].shift(1).fillna(1)[m]).sum() / yrs), 3)
    o['cost_per_year_pct'] = round(float((res['cost'][m] / res['nav'].shift(1).fillna(1)[m]).sum() / yrs * 100), 3)
    if base is not None:
        pr = month_rets(res['nav'], DATES); pb = month_rets(base['nav'], DATES)
        d = (pr - pb).dropna(); d = d[in_segs(d.index, segs)]
        ob = L.daily_stats(base['nav'], segs, DATES)
        o['vs_base'] = {'diff_pct_m': round(float(d.mean() * 100), 3), 't_diff_m': tstat(d), 'n_months': int(len(d)),
                        'hit_m': round(float((d > 0).mean()), 3),
                        'CAGR_diff': round(o['CAGR'] - ob['CAGR'], 4), 'maxDD_diff': round(o['maxDD'] - ob['maxDD'], 4),
                        'base_CAGR': ob['CAGR'], 'base_maxDD': ob['maxDD']}
    return o

def event_table(res, RL, buy, sell):
    reb = sorted(RL); rows = []
    for e in res['events']:
        x = e['sell']; j = e['j']
        nxt = [r for r in reb if r > x]
        if not nxt: continue
        R = nxt[0]
        rs = AO[R, j] / e['px'] - 1
        pool = e['pool']
        rp = np.nanmean(AO[R, pool] / AO[x, pool] - 1) if pool is not None else np.nan
        rep = j in res['picks'].get(R, set())
        rows.append({'trig': IDX[e['trig']], 'sell': IDX[x], 'code': COLS[j], 'at_reb': e['at_reb'], 'days': R - x,
                     'r_stock': rs, 'r_pool': rp, 'abn': rs - rp, 'repicked': rep,
                     'net_benefit_cash': -rs - (buy + sell) * rep, 'src3': bool(IS_SRC3[e['trig'], j])})
    return pd.DataFrame(rows)

def event_stats(ev, a, b):
    if ev is None or len(ev) == 0: return {'n': 0}
    e = ev[(ev['trig'] >= pd.Timestamp(a)) & (ev['trig'] <= pd.Timestamp(b))].dropna(subset=['r_stock', 'r_pool'])
    if len(e) == 0: return {'n': 0}
    g = e.groupby(e['trig'].dt.to_period('M'))
    o = {'n': int(len(e)), 'n_months': int(g.ngroups), 'mean_days_to_refill': round(float(e['days'].mean()), 1),
         'share_at_rebalance': round(float(e['at_reb'].mean()), 3), 'repicked_share': round(float(e['repicked'].mean()), 3)}
    for c in ('r_stock', 'abn', 'net_benefit_cash'):
        o[f'{c}_mean_pct'] = round(float(e[c].mean() * 100), 3); o[f'{c}_median_pct'] = round(float(e[c].median() * 100), 3)
        o[f't_{c}_month'] = tstat(g[c].mean())
    o['abn_hit_neg'] = round(float((e['abn'] < 0).mean()), 3)
    return o


def main():
    res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    assert 'PREREG' in res
    S = L.Setup(L.yoy_long(), start='2005-01-01')
    RL = rank_lists(S)
    t0 = pos(S.dates[0]); t1 = pos(S.dates[-1])
    a0 = str(S.dates[0].date()); END = str(IDX[t1].date())
    W = {'select': [(a0, '2019-04-30')], 'valid': [('2019-05-01', END)], 'y2024_2025': [('2024-01-01', '2025-12-31')],
         'rev2021_select': [('2021-03-01', '2023-12-31')], 'rev2021_valid': [('2024-01-01', END)],
         'src3': [('2024-02-15', '2025-09-23')], 'valid_official_only': [('2019-05-01', '2024-02-14'), ('2025-09-24', END)],
         'sub_select_2006_2012': [(a0, '2012-12-31')], 'sub_select_2013_2019-04': [('2013-01-01', '2019-04-30')],
         'all': [(a0, END)]}
    res['meta'] = {'first_signal': a0, 'last_signal': str(S.dates[-1].date()), 'end': END, 'n_rebalances': len(RL),
                   'trigger_counts_all_stocks_all_days': {s: int(TRIG[s].sum()) for s in SPECS}}
    ew, _ = L.sim_portfolio({k: ('open', v[1]) for k, v in RL.items()}, 0.0, 0.0, t0, t1)

    # 一致性檢查：不加出場 = lhlib 原引擎
    b, s = L.cost_rates(0.001)
    ev30, _ = S.selection(30)
    ref, _ = L.sim_portfolio(ev30, b, s, t0, t1)
    chk = sim(RL, 30, None, b, s, t0, t1)
    assert np.allclose(ref.to_numpy(), chk['nav'].to_numpy(), rtol=1e-10), 'baseline mismatch'
    res['meta']['baseline_equals_lhlib'] = True

    out = {}; navs = {'EW_pool': ew}; evs = {}
    KEYW = list(W)
    def run(tag, n, spec, slip, idle='cash', trig=None, RLx=RL, tt0=t0, tt1=t1, Wx=W, ewx=ew, DATES=S.DATES, base=None):
        if slip == 'gross': bb = ss = 0.0
        else: bb, ss = L.cost_rates(slip)
        tr = trig if trig is not None else (None if spec == 'BASE' else TRIG[spec])
        r = sim(RLx, n, tr, bb, ss, tt0, tt1, idle=idle, slip=0.0 if slip == 'gross' else slip)
        o = {'windows': {k: port_stats(r, base, sg, DATES, ewx) for k, sg in Wx.items()}}
        if spec != 'BASE':
            et = event_table(r, RLx, bb, ss); evs[tag] = et
            o['events'] = {k: event_stats(et, sg[0][0], sg[-1][1]) if len(sg) == 1 else None for k, sg in Wx.items()}
            o['events'] = {k: v for k, v in o['events'].items() if v is not None}
            if len(et):
                o['events_by_year'] = {int(y): int(c) for y, c in et.groupby(et['trig'].dt.year).size().items()}
        out[tag] = o; navs[tag] = r['nav']
        w = o['windows']
        print(tag, {k: (w[k]['CAGR'], w[k]['maxDD'], w[k].get('t_vs_EW_m'), (w[k].get('vs_base') or {}).get('t_diff_m')) for k in ('select', 'valid', 'y2024_2025') if w.get(k)}, flush=True)
        return r

    for n in (30, 10):
        slips = (0.001, 0.003, 'stress', 'gross') if n == 30 else (0.001,)
        for slip in slips:
            base = run(f'n{n}_BASE_slip{slip}', n, 'BASE', slip)
            for spec in SPECS:
                if n == 30 and slip in (0.003, 'stress', 'gross') and spec != PRIMARY: continue
                run(f'n{n}_{spec}_slip{slip}_cash', n, spec, slip, base=base)
                if n == 30 and slip == 0.001:
                    run(f'n{n}_{spec}_slip{slip}_0050', n, spec, slip, idle='0050', base=base)
    # 診斷：src 3 期間不觸發
    base = sim(RL, 30, None, b, s, t0, t1)
    tr = TRIG[PRIMARY] & ~IS_SRC3
    run(f'DIAG_n30_{PRIMARY}_noSRC3_slip0.001_cash', 30, PRIMARY, 0.001, trig=tr, base=base)
    # 診斷：2021 起 rev_5y 實盤版主策略
    S5 = L.Setup(L.yoy_rev5y(), start='2021-01-01'); RL5 = rank_lists(S5)
    a5 = pos(S5.dates[0]); b5 = pos(S5.dates[-1])
    ew5, _ = L.sim_portfolio({k: ('open', v[1]) for k, v in RL5.items()}, 0.0, 0.0, a5, b5)
    W5 = {k: W[k] for k in ('rev2021_select', 'rev2021_valid', 'y2024_2025', 'src3')}
    W5['rev5y_all'] = [('2021-03-01', END)]
    base5 = sim(RL5, 30, None, b, s, a5, b5)
    run('DIAG_rev5y_n30_BASE_slip0.001', 30, 'BASE', 0.001, RLx=RL5, tt0=a5, tt1=b5, Wx=W5, ewx=ew5, DATES=S5.DATES)
    for spec in SPECS:
        run(f'DIAG_rev5y_n30_{spec}_slip0.001_cash', 30, spec, 0.001, RLx=RL5, tt0=a5, tt1=b5, Wx=W5, ewx=ew5, DATES=S5.DATES, base=base5)

    res['results'] = out
    # 判定（PRIMARY = n30、V20xBLK、滑價 0.1%、現金）
    P = out[f'n30_{PRIMARY}_slip0.001_cash']
    dec = {}
    for k in ('select', 'valid'):
        w = P['windows'][k]; ev = P['events'][k]
        A_ok = (w['t_vs_EW_m'] >= 2) and (w['CAGR'] >= w['0050_CAGR'])
        B_ok = w['vs_base']['t_diff_m'] >= 2
        dec[k] = {'A_vs_EW_t': w['t_vs_EW_m'], 'CAGR': w['CAGR'], '0050_CAGR': w['0050_CAGR'], 'A_pass': bool(A_ok),
                  'B_t_diff_vs_base': w['vs_base']['t_diff_m'], 'B_pass': bool(B_ok), 'pass': bool(A_ok and B_ok),
                  'n_events': ev['n'], 'n_event_months': ev.get('n_months', 0),
                  'testable': bool(ev['n'] >= 30 and ev.get('n_months', 0) >= 12)}
    if not all(dec[k]['testable'] for k in dec): v = '無法測試'
    else:
        npass = sum(dec[k]['pass'] for k in dec)
        v = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
    dec['verdict'] = v
    res['DECISION'] = dec
    print('DECISION', json.dumps(dec, ensure_ascii=False), flush=True)
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    sp = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
    pd.DataFrame(navs).to_parquet(sp / 'w4_climax_navs.parquet')
    pd.concat({k: v for k, v in evs.items() if len(v)}, names=['tag']).to_csv(sp / 'w4_climax_events.csv.gz')

if __name__ == '__main__':
    main()

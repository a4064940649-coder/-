#!/usr/bin/env python3
"""寶塔線（n 日收盤）翻紅買、第 2 根轉黑賣（莊文宗）。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
事件／組合統計架構從 w4_chu_jiahong_strong_bar_ma5_exit/run.py 複製改寫（不 import），成本用 exec_rebuild/exlib.cost_rates。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild')); sys.path.insert(0, str(HERE.parents[1]))
import exlib as L  # noqa: E402  成本模型（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
names = ['adj', 'close_raw', 'open', 'value', 'flag']
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in names}
D['ret'] = pd.read_parquet(P / 'ret.parquet')
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if c.isdigit() and len(c) == 4 and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
F = D['flag']; bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); V = D['value']
fac = ADJ / CR
AO_raw = OP * fac
ACf = ADJ.ffill()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
FOK = F.isin([0, 1, 2]) & ADJ.notna()
UNIV = (V60 >= 1e7) & (CR >= 10) & FOK
UN = UNIV.fillna(False).to_numpy()

lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
AC_prev = ACf.shift(1).to_numpy()
AOn = AO_raw.to_numpy(); ACn = ACf.to_numpy(); ADJn = ADJ.to_numpy()
GAP_UP = (AOn / AC_prev - 1) >= lim - 0.006
PX = AO_raw.bfill().to_numpy()
PX = np.where(np.isfinite(PX), PX, ACn)
VALO = np.where(np.isfinite(AOn), AOn, AC_prev)   # 開盤估值（停牌用前一日收盤）

# ---------- 寶塔線狀態（每檔只在有收盤的日子上計算） ----------
def tower(n):
    """回傳 dict j -> (pos 全域索引, state 陣列(1/-1/0 未定), dn 新黑棒, ma20)。"""
    out = {}
    for j in range(N):
        a = ADJn[:, j]; pos = np.where(np.isfinite(a))[0]
        if len(pos) < n + 2: continue
        s = pd.Series(a[pos])
        hh = s.shift(1).rolling(n, min_periods=n).max().to_numpy(); ll = s.shift(1).rolling(n, min_periods=n).min().to_numpy()
        sv = s.to_numpy()
        up = sv > hh; dn = sv < ll
        st = np.where(up, 1.0, np.where(dn, -1.0, np.nan))
        st = pd.Series(st).ffill().fillna(0).to_numpy()
        ma = s.rolling(20, min_periods=20).mean().to_numpy()
        out[j] = (pos, st, dn, ma, sv)
    return out

def next_true(b):
    """nx[k] = 最小 k' >= k 使 b[k'] 為真；無則 len(b)。"""
    K = len(b); nx = np.full(K + 1, K, np.int64)
    for k in range(K - 1, -1, -1):
        nx[k] = k if b[k] else nx[k + 1]
    return nx

CAP = 250

def build_events(TW, use_ma, mode):
    """mode: ZWZ / STRICT / SIMPLE。回傳事件 (j, t, e, legs=[(frac, x, done, xp)], capped)。"""
    ev = []; nblock = 0; nnoopen = 0; nsig = 0; ncap = 0
    for j, (pos, st, dn, ma, sv) in TW.items():
        K = len(pos)
        flip = np.zeros(K, bool); flip[1:] = (st[1:] == 1) & (st[:-1] == -1)
        sig = flip & UN[pos, j]
        if use_ma: sig &= np.isfinite(ma) & (sv >= ma)
        ks = np.where(sig)[0]
        if len(ks) == 0: continue
        nsig += len(ks)
        nb = next_true(st == -1); nnb = next_true(dn)
        free_from = -1
        for k in ks:
            t = pos[k]
            if t < free_from: continue
            e = t + 1
            if e >= T: continue
            if not (np.isfinite(AOn[e, j]) and np.isfinite(ADJn[e, j])): nnoopen += 1; continue
            if GAP_UP[e, j]: nblock += 1; continue
            assert k + 1 < K and pos[k + 1] == e
            def gx(kk):
                """compressed index kk 的黑棒 → 全域出場日（kk>=K 表示尚未發生）；套用 250 日上限。"""
                if kk >= K: x = T  # 尚未出場
                else: x = pos[kk] + 1
                capd = x - e > CAP
                if capd: x = e + CAP
                return x, capd
            s1 = nb[k + 1]
            if mode == 'SIMPLE':
                x, c = gx(s1); legs = [(1.0, x)]; capped = c
            else:
                x1, c1 = gx(s1)
                if s1 >= K or c1: legs = [(1.0, x1)]; capped = c1
                else:
                    u = nb[s1 + 1] if mode == 'ZWZ' else nnb[s1 + 1]
                    x2, c2 = gx(u); legs = [(0.5, x1), (0.5, x2)]; capped = c2
            ncap += capped
            L2 = []
            for fr, x in legs:
                done = x < T
                L2.append((fr, x, done, PX[x, j] if done else np.nan))
            ev.append((j, t, e, L2))
            free_from = max(x for _, x in legs)
    return ev, dict(raw_signals=nsig, blocked_gap_up=nblock, no_open=nnoopen, capped=ncap)

def ew_series(mask):
    m = mask.shift(1, fill_value=False)
    oo = (AO_raw.shift(-1) / AO_raw - 1).where(m).clip(-0.5, 1.0).mean(axis=1).fillna(0).to_numpy()
    cc = D['ret'].where(m).clip(-0.5, 1.0).mean(axis=1).fillna(0)
    return np.concatenate([[0.0], np.cumsum(np.log1p(oo))]), oo, cc
LEW, EW_OO, ew_cc = ew_series(UNIV)
LALL, _, all_cc = ew_series(FOK)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0)
E50n = e50.to_numpy()
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))

BUY, SELL = L.cost_rates(0.001)
BUY_S, SELL_S = L.cost_rates('stress')

def event_table(ev):
    rows = []
    for j, t, e, legs in ev:
        if not all(l[2] for l in legs): continue
        p0 = AOn[e, j]
        g = sum(fr * xp for fr, x, _, xp in legs) / p0
        raw = g - 1
        net = g * (1 - SELL) / (1 + BUY) - 1
        net_s = g * (1 - SELL_S) / (1 + BUY_S) - 1
        ew = sum(fr * math.expm1(LEW[x] - LEW[e]) for fr, x, _, _ in legs)
        al = sum(fr * math.expm1(LALL[x] - LALL[e]) for fr, x, _, _ in legs)
        b50 = sum(fr * (E50n[x - 1] / E50n[t] - 1) for fr, x, _, _ in legs)
        xl = max(x for _, x, _, _ in legs)
        hold = sum(fr * (x - e) for fr, x, _, _ in legs)
        rows.append((cols[j], IDX[t], IDX[e], hold, xl - e, len(legs), raw, net, net_s, ew, al, b50, bool(SRC3[t:xl + 1].any()),
                     AOn[e, j] / ACn[t, j] - 1))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'hold', 'hold_last', 'nlegs', 'raw', 'net', 'net_s', 'ew', 'allew', 'b50', 'src3', 'gap'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']
    df['ab_gross'] = df['raw'] - df['ew']; df['ab_all'] = df['net'] - df['allew']
    return df

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31')}

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days_weighted': round(float(df['hold'].mean()), 2),
            'avg_hold_last_leg': round(float(df['hold_last'].mean()), 2), 'share_two_legs': round(float((df['nlegs'] == 2).mean()), 3),
            'mean_overnight_gap_pct': round(df['gap'].mean() * 100, 3),
            'mean_raw_pct': round(df['raw'].mean() * 100, 3), 'mean_net_pct': round(df['net'].mean() * 100, 3),
            'mean_EW_pct': round(df['ew'].mean() * 100, 3),
            'mean_ab_gross_pct': round(df['ab_gross'].mean() * 100, 3), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': round(df['ab'].mean() * 100, 3), 'median_ab_net_pct': round(df['ab'].median() * 100, 3),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': round(df['ab_s'].mean() * 100, 3), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_ab_net_vs_allEW_pct': round(df['ab_all'].mean() * 100, 3), 't_month_vs_allEW': tstat(gm['ab_all'].mean()),
            'mean_net_vs_0050_pct': round(df['ab50'].mean() * 100, 3), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3)}

def ev_portfolio(ev, buy, sell):
    G = np.zeros(T); W = np.zeros(T); C = np.zeros(T)
    for j, t, e, legs in ev:
        C[e] += buy
        for fr, x, done, xp in legs:
            xe = x if done else T - 1
            r = np.zeros(xe - e + 1)
            r[0] = ACn[e, j] / AOn[e, j] - 1
            if xe > e: r[1:] = np.nan_to_num(ACn[e + 1:xe + 1, j] / ACn[e:xe, j] - 1)
            if done:
                r[-1] = xp / ACn[x - 1, j] - 1
                C[x] += sell * fr
            G[e:xe + 1] += fr * r; W[e:xe + 1] += fr
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(W > 0, G / np.where(W > 0, W, 1), 0.0); c = np.where(W > 0, C / np.where(W > 0, W, 1), 0.0)
    return pd.Series(g - c, index=IDX), pd.Series(W, index=IDX)

def red_hold2(TW, use_ma, buy, sell):
    RED = np.zeros((T, N), bool)
    for j, (pos, st, dn, ma, sv) in TW.items():
        ok = st == 1
        if use_ma: ok = ok & np.isfinite(ma) & (sv >= ma)
        RED[pos[ok], j] = True
    RED &= UN
    w = np.zeros(N); cash = 1.0
    ret = np.zeros(T); npos = np.zeros(T); tov = np.zeros(T)
    for d in range(1, T - 1):
        # 在 d 開盤：依 d−1 收盤訊號調整（w 已是 d 開盤的漂移後權重，總和＋cash＝1）
        tgt_set = RED[d - 1]
        tradable = np.isfinite(AOn[d])
        held = w > 0
        frozen = held & ~tradable
        S = tgt_set & tradable & (held | ~GAP_UP[d])
        tgt = np.zeros(N); tgt[frozen] = w[frozen]
        rest = 1 - tgt.sum()
        if S.sum() > 0: tgt[S] = rest / S.sum()
        cost = buy * np.clip(tgt - w, 0, None).sum() + sell * np.clip(w - tgt, 0, None).sum()
        tov[d] = np.clip(w - tgt, 0, None).sum()
        npos[d] = (tgt > 0).sum()
        cash = 1 - tgt.sum()
        # d 開盤 → d+1 開盤
        gr = np.nan_to_num(VALO[d + 1] / VALO[d], nan=1.0, posinf=1.0, neginf=1.0)
        gr = np.clip(gr, 0.5, 2.0)
        wv = tgt * gr; tot = wv.sum() + cash
        ret[d] = (1 - cost) * tot - 1
        w = wv / tot; cash = cash / tot
    return pd.Series(ret, index=IDX), pd.Series(npos, index=IDX), pd.Series(tov, index=IDX)

def pstats(r, npos, a, b, ew):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    rr = r[m]; bb = r50[m]; ee = ew[m]
    yrs = (rr.index[-1] - rr.index[0]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod()
        return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}
    o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
    o['vol'] = round(float(rr.std() * math.sqrt(252)), 4)
    o['share_days_invested'] = round(float((npos[m] > 0).mean()), 3)
    o['avg_positions_when_invested'] = round(float(npos[m][npos[m] > 0].mean()), 1)
    mo = lambda x: (1 + x).groupby(x.index.to_period('M')).prod() - 1
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    return o

SPECS = [('PRIMARY', 3, True, 'ZWZ'), ('NOMA', 3, False, 'ZWZ'), ('SIMPLE', 3, True, 'SIMPLE'), ('SIMPLE_NOMA', 3, False, 'SIMPLE'),
         ('STRICT', 3, True, 'STRICT'), ('N4', 4, True, 'ZWZ'), ('N5', 5, True, 'ZWZ')]
RSPECS = [('RED_HOLD', 3, False), ('RED_HOLD_MA', 3, True), ('RED_HOLD_N4', 4, False), ('RED_HOLD_N5', 5, False)]

def main():
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'univ_stock_days': int(UN.sum())}}
    TWS = {n: tower(n) for n in (3, 4, 5)}
    # 狀態統計
    for n, TW in TWS.items():
        red = sum(int(((st == 1) & UN[pos, j]).sum()) for j, (pos, st, dn, ma, sv) in TW.items())
        out['meta'][f'n{n}_red_share_in_univ'] = round(red / UN.sum(), 3)
    res = {}; yearly_ev = {}
    EWS = pd.Series(EW_OO, index=IDX)
    for key, n, ma, mode in SPECS:
        ev, info = build_events(TWS[n], ma, mode)
        df = event_table(ev)
        o = dict(info); o['events_taken'] = len(ev)
        for w, (a, b) in WIN.items():
            o[f'event_{w}'] = ev_stats(df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))])
        o['event_valid_ex_src3'] = ev_stats(df[(df['sig'] >= pd.Timestamp('2019-05-01')) & ~df['src3']])
        o['event_select_sub_2005-2011'] = ev_stats(df[(df['sig'] >= '2005-01-01') & (df['sig'] <= '2011-12-31')])
        o['event_select_sub_2012-2019-04'] = ev_stats(df[(df['sig'] >= '2012-01-01') & (df['sig'] <= '2019-04-30')])
        r, npos = ev_portfolio(ev, BUY, SELL)
        rs, _ = ev_portfolio(ev, BUY_S, SELL_S)
        rg, _ = ev_portfolio(ev, 0.0, 0.0)
        for w, (a, b) in WIN.items():
            o[f'port_{w}'] = pstats(r, npos, a, b, ew_cc)
            o[f'port_{w}']['CAGR_stress'] = pstats(rs, npos, a, b, ew_cc)['CAGR']
            pg = pstats(rg, npos, a, b, ew_cc)
            o[f'port_{w}']['CAGR_gross'] = pg['CAGR']; o[f'port_{w}']['t_vs_EW_m_gross'] = pg['t_vs_EW_m']
        yearly_ev[key] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'ab_gross_pct': round(g['ab_gross'].mean() * 100, 2)}
                          for y, g in df.groupby(df['sig'].dt.year)}
        res[key] = o
        es, ev_ = o['event_select_2005-01_2019-04'], o['event_valid_2019-05_2026-10']
        ps, pv = o['port_select_2005-01_2019-04'], o['port_valid_2019-05_2026-10']
        print(key, 'n', len(df), info, '| S gross', es.get('mean_ab_gross_pct'), 'net', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'),
              'hold', es.get('avg_hold_days_weighted'), 'CAGR', ps['CAGR'], 'g', ps['CAGR_gross'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'],
              '| V gross', ev_.get('mean_ab_gross_pct'), 'net', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'),
              'CAGR', pv['CAGR'], 'g', pv['CAGR_gross'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], flush=True)
        df.to_csv(Path(L.P).parent / f'w4_tower_{key}_events.csv.gz', index=False)
    rres = {}
    for key, n, ma in RSPECS:
        r, npos, tov = red_hold2(TWS[n], ma, BUY, SELL)
        rs, _, _ = red_hold2(TWS[n], ma, BUY_S, SELL_S)
        rg, _, _ = red_hold2(TWS[n], ma, 0.0, 0.0)
        o = {}
        for w, (a, b) in WIN.items():
            m = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
            o[w] = pstats(r, npos, a, b, EWS)
            o[w]['CAGR_stress'] = pstats(rs, npos, a, b, EWS)['CAGR']
            pg = pstats(rg, npos, a, b, EWS)
            o[w]['CAGR_gross'] = pg['CAGR']; o[w]['t_vs_EW_m_gross'] = pg['t_vs_EW_m']; o[w]['t_vs_0050_m_gross'] = pg['t_vs_0050_m']
            o[w]['avg_daily_oneway_turnover'] = round(float(tov[m].mean()), 4)
        rres[key] = o
        print(key, {w: (o[w]['CAGR'], o[w]['CAGR_gross'], o[w]['EW_CAGR'], o[w]['0050_CAGR'], o[w]['t_vs_EW_m'], o[w]['t_vs_EW_m_gross'], o[w]['avg_daily_oneway_turnover']) for w in WIN}, flush=True)
    out['results'] = res; out['results_red_hold'] = rres; out['yearly_events'] = yearly_ev
    def passes(o, w):
        e = o[f'event_{w}']; p = o[f'port_{w}']
        ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
        ok_50 = p['CAGR'] >= p['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['PRIMARY']
    vs = {'select': passes(pr, 'select_2005-01_2019-04'), 'valid': passes(pr, 'valid_2019-05_2026-10')}
    k = vs['select']['pass'] + vs['valid']['pass']
    thin = any(pr[f'event_{w}'].get('n', 0) < 30 or pr[f'event_{w}'].get('n_months', 0) < 24 for w in ['select_2005-01_2019-04', 'valid_2019-05_2026-10'])
    vs['verdict'] = '無法測試' if thin else ('可用' if k == 2 else ('待觀察' if k == 1 else '不可用'))
    vs['variants_pass'] = {kk: {w: passes(o, w) for w in WIN} for kk, o in res.items()}
    vs['red_hold_pass'] = {kk: {w: {'beat_EW_t2': bool(o[w]['t_vs_EW_m'] >= 2 and o[w]['CAGR'] > o[w]['EW_CAGR']), 'not_lose_0050': bool(o[w]['CAGR'] >= o[w]['0050_CAGR'])} for w in WIN} for kk, o in rres.items()}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8'))
    d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps({k: v for k, v in vs.items() if k in ('select', 'valid', 'verdict')}, ensure_ascii=False))

if __name__ == '__main__':
    main()

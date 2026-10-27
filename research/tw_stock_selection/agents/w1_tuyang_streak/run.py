#!/usr/bin/env python3
"""土洋合作（外資＋投信同步連買 3 日）事件回測＋CMoney 月換股版。規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成本模型、0050、M2 價格陣列（只讀取）
from btlib import tstat  # noqa: E402

P = L.P; X = L.X
DATA = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
BUY, SELL = L.cost_rates(0.001)
BUY_S, SELL_S = L.cost_rates('stress')
r3 = lambda x: round(float(x), 3)
is_common = lambda c: len(c) == 4 and c.isdigit() and not c.startswith('0') and not c.startswith('91')

# ---------------- 三大法人（inst 日曆） ----------------
INS = pd.read_csv(X / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign', 'trust'], parse_dates=['date'])
INS = INS[INS['code'].map(is_common)]
FOR_ALL = INS.pivot_table(index='date', columns='code', values='foreign', aggfunc='sum')
TR_ALL = INS.pivot_table(index='date', columns='code', values='trust', aggfunc='sum')

# ---------------- 價格資料 ----------------
def load_panel():
    names = ['adj', 'close_raw', 'open', 'value', 'flag', 'src']
    D = {k: pd.read_parquet(P / f'{k}.parquet').loc['2021-01-01':] for k in names}
    good = D['adj'].notna().sum(axis=1) >= 300
    bad = [str(x.date()) for x in D['adj'].index[~good]]
    cols = [c for c in D['adj'].columns if is_common(c)]
    for k in D: D[k] = D[k].loc[good, cols]
    bad5 = D['flag'] == 5
    ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5)
    V = D['value']; VOL = V / D['close_raw']
    FL = D['flag'].fillna(0).isin([0, 1, 2])
    return dict(ADJ=ADJ, CR=CR, OP=OP, V=V, VOL=VOL, FLOK=FL, bad=bad, name='TWSE_panel')

def load_history():
    """上市＋上櫃官方日 K（2025-09-24 起）＋ exdiv 參考價還原。"""
    h = pd.read_csv(DATA / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    sl = pd.read_csv(DATA / 'stock_list.csv', dtype={'code': str})
    h = h[h['code'].map(is_common) & h['code'].isin(sl['code'])]
    CR = h.pivot_table(index='date', columns='code', values='close', aggfunc='last')
    OP = h.pivot_table(index='date', columns='code', values='open', aggfunc='last').reindex_like(CR)
    VOL = h.pivot_table(index='date', columns='code', values='volume', aggfunc='last').reindex_like(CR)
    CR = CR.where(CR > 0); OP = OP.where(OP > 0)
    VOL = VOL.where(CR.notna())
    V = VOL * CR
    ex = pd.read_csv(X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[(ex['date'] >= CR.index[0]) & ex['code'].isin(CR.columns) & (ex['prev_close'] > 0) & (ex['ref_price'] > 0)]
    prev = CR.ffill().shift(1)
    ret = CR / prev - 1
    for _, r in ex.iterrows():
        if r['date'] in ret.index:
            ret.loc[r['date'], r['code']] = CR.loc[r['date'], r['code']] / r['ref_price'] - 1
    ret = ret.where(CR.notna()).clip(-0.5, 1.0)
    ADJ = (1 + ret.fillna(0)).cumprod().where(CR.notna())
    ADJ = ADJ * (CR.bfill().iloc[0] / ADJ.bfill().iloc[0])                 # 只是尺度
    FL = CR.notna()
    mk = sl.set_index('code')['market'].reindex(CR.columns)
    return dict(ADJ=ADJ, CR=CR, OP=OP, V=V, VOL=VOL, FLOK=FL, bad=[], name='TWSE+TPEX_history', market=mk)

class Universe:
    def __init__(self, D):
        self.D = D
        ADJ, CR, OP, V, VOL = D['ADJ'], D['CR'], D['OP'], D['V'], D['VOL']
        self.cols = list(ADJ.columns); self.N = len(self.cols)
        self.PIDX = ADJ.index; self.TP = len(self.PIDX)
        fac = ADJ / CR
        self.AO_raw = OP * fac
        ACf = ADJ.ffill()
        v0 = V.fillna(0).where(V.notna().cumsum() > 0)
        V60 = v0.rolling(60, min_periods=60).mean()
        self.LIQ = (V60 >= 1e7) & (CR >= 10) & D['FLOK'] & ADJ.notna()
        AC_prev = ACf.shift(1).to_numpy()
        self.GAP_UP = (self.AO_raw.to_numpy() / AC_prev - 1) >= 0.10 - 0.006
        self.AOn = self.AO_raw.to_numpy(); self.ACn = ACf.to_numpy()
        PX = self.AO_raw.bfill().to_numpy(); self.PX = np.where(np.isfinite(PX), PX, self.ACn)
        oo = (self.AO_raw.shift(-1) / self.AO_raw - 1).where(self.LIQ.shift(1, fill_value=False)).clip(-0.5, 1.0)
        e = oo.mean(axis=1).fillna(0).to_numpy()
        self.LEW = np.concatenate([[0.0], np.cumsum(np.log1p(e))])          # LEW[d] = log 等權指數於 d 日開盤
        rcc = (ADJ / ACf.shift(1) - 1)
        self.ew_cc = rcc.where(self.LIQ.shift(1, fill_value=False)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
        e50 = L.e50.reindex(self.PIDX.union(L.e50.index)).ffill().reindex(self.PIDX)
        self.r50 = e50.pct_change().fillna(0); self.E50n = e50.to_numpy()
        self.SRC3 = (self.PIDX >= pd.Timestamp('2024-02-15')) & (self.PIDX <= pd.Timestamp('2025-09-23'))
        # 價量條件（面板日曆）
        VOLz = VOL.fillna(0).where(VOL.notna().cumsum() > 0)
        MIN60 = ADJ.rolling(60, min_periods=40).min()
        self.EXT = ADJ / MIN60 - 1
        self.VOL3 = VOLz.rolling(3, min_periods=3).mean()
        self.VOLP20 = VOLz.shift(3).rolling(20, min_periods=15).mean()
        # inst 日曆
        i0, i1 = self.PIDX[0], self.PIDX[-1]
        FO = FOR_ALL.loc[i0:i1].reindex(columns=self.cols).fillna(0.0)
        TR = TR_ALL.loc[i0:i1].reindex(columns=self.cols).fillna(0.0)
        self.IIDX = FO.index; self.TI = len(self.IIDX)
        miss = self.PIDX[(self.PIDX >= self.IIDX[0]) & (self.PIDX <= self.IIDX[-1])].difference(self.IIDX)
        assert miss.empty, miss
        ppos = pd.Series(np.arange(self.TP), index=self.PIDX)
        self.PP = ppos.reindex(self.IIDX).fillna(-1).astype(int).to_numpy()
        self.NEXTP = np.searchsorted(self.PIDX.values, self.IIDX.values)
        self.FO, self.TR = FO, TR
        ri = lambda df: df.reindex(self.IIDX)
        cr = ri(CR); vl = ri(VOL) / 1000.0
        has = cr.notna() & ri(V).notna()
        has3 = has & has.shift(1, fill_value=False) & has.shift(2, fill_value=False)
        both = (FO > 0) & (TR > 0)
        c1 = both & both.shift(1, fill_value=False) & both.shift(2, fill_value=False)
        A = (FO + TR) * cr
        c2 = (A.shift(2) < A.shift(1)) & (A.shift(1) < A)
        c3 = (FO + TR).rolling(3).sum() >= 0.10 * vl.rolling(3).sum()
        c4 = ri(self.EXT) < 0.30
        c5 = ri(self.VOL3) > ri(self.VOLP20)
        pool = ri(self.LIQ).fillna(False).astype(bool)
        base = (c1 & has3 & pool)
        self.SIG = {'TY': (base & c2 & c3 & c4 & c5).to_numpy(), 'TY_BASE': base.to_numpy(), 'TY_NOINC': (base & c3 & c4 & c5).to_numpy()}
        self.PRI = A.rolling(3).sum().to_numpy()
        self.cond_counts = {k: int(v.loc['2021-10-12':].to_numpy().sum()) for k, v in
                            {'pool_c1_has3': base, '+c2': base & c2, '+c2c3': base & c2 & c3, '+c2c3c4': base & c2 & c3 & c4,
                             '+all': base & c2 & c3 & c4 & c5}.items()}

    def build_events(self, S, H, start='2021-10-12'):
        ev = []; nblock = 0; nnoopen = 0; nsig = 0
        s0 = int(np.searchsorted(self.IIDX.values, np.datetime64(start)))
        for j in range(self.N):
            ts = np.where(S[:, j])[0]; ts = ts[ts >= s0]
            free_from = -1
            for t in ts:
                nsig += 1
                if t < free_from: continue
                e = t + 1
                if e >= self.TI: continue
                ep = self.PP[e]
                if ep < 0 or not np.isfinite(self.AOn[ep, j]): nnoopen += 1; continue
                if self.GAP_UP[ep, j]: nblock += 1; continue
                x = e + H; why = 'hold'
                if x < self.TI: xp = int(self.NEXTP[x])
                else: xp = int(self.PP[self.TI - 1] + (x - (self.TI - 1)))
                if xp >= self.TP: xp = self.TP - 1; why = 'censored'
                ev.append(dict(j=j, t=t, tp=int(self.PP[t]), e=e, ep=int(ep), x=x, xp=xp, why=why,
                               pri=float(self.PRI[t, j]) if np.isfinite(self.PRI[t, j]) else 0.0))
                free_from = x
        return ev, dict(raw_signals=nsig, events=len(ev), blocked_gap_up=nblock, no_open=nnoopen)

    def exit_px(self, v):
        return self.ACn[v['xp'], v['j']] if v['why'] == 'censored' else self.PX[v['xp'], v['j']]

    def event_table(self, ev):
        rows = []
        for v in ev:
            j, tp, ep, xp = v['j'], v['tp'], v['ep'], v['xp']
            p0 = self.AOn[ep, j]; p1 = self.exit_px(v)
            raw = p1 / p0 - 1
            net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1
            net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
            ew = math.expm1(self.LEW[xp] - self.LEW[ep])
            b50 = self.E50n[max(xp - 1, tp)] / self.E50n[tp] - 1
            rows.append((self.cols[j], self.IIDX[v['t']], self.PIDX[ep], self.PIDX[xp], xp - ep, v['why'], raw, net, net_s, ew, b50,
                         bool(self.SRC3[tp:xp + 1].any()), v['pri']))
        df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'exit', 'hold', 'why', 'raw', 'net', 'net_s', 'ew', 'b50', 'src3', 'amt3'])
        df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']
        df['ab_gross'] = df['raw'] - df['ew']
        return df

    def portfolio(self, ev, buy, sell, slots=20):
        ent = {}
        for k, v in enumerate(ev): ent.setdefault(v['ep'], []).append(k)
        exits = {}
        for k, v in enumerate(ev): exits.setdefault(v['xp'], []).append(k)
        held = {}; cash = 1.0; nav = np.empty(self.TP); npos = np.zeros(self.TP); skipped = 0
        AOn, ACn = self.AOn, self.ACn
        for d in range(self.TP):
            for k in [k for k in exits.get(d, []) if k in held and ev[k]['why'] != 'censored']:
                u = held.pop(k); cash += u * self.PX[d, ev[k]['j']] * (1 - sell)
            if d in ent:
                pv = cash + sum(u * (AOn[d, ev[k]['j']] if np.isfinite(AOn[d, ev[k]['j']]) else ACn[d - 1, ev[k]['j']]) for k, u in held.items())
                for k in sorted(ent[d], key=lambda k: -ev[k]['pri']):
                    if len(held) >= slots or cash <= 1e-9: skipped += 1; continue
                    amt = min(pv / slots, cash); cash -= amt
                    held[k] = amt / (1 + buy) / AOn[d, ev[k]['j']]
            nav[d] = cash + sum(u * ACn[d, ev[k]['j']] for k, u in held.items())
            npos[d] = len(held)
        return pd.Series(nav, index=self.PIDX), pd.Series(npos, index=self.PIDX), skipped

    def pstats(self, nav, npos, a, b):
        r = nav.pct_change().fillna(0)
        m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
        rr = r[m]; bb = self.r50[m]; ee = self.ew_cc[m]
        yrs = (rr.index[-1] - rr.index[0]).days / 365.25
        def st(x):
            eq = (1 + x).cumprod()
            return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
        o = {}
        o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
        o['vol'] = round(float(rr.std() * math.sqrt(252)), 4)
        o['avg_positions'] = round(float(npos[m].mean()), 1)
        mo = lambda x: (1 + x).groupby(x.index.to_period('M')).prod() - 1
        ms, mb, me = mo(rr), mo(bb), mo(ee)
        o['n_months'] = int(len(ms))
        o['excess_vs_0050_pct_m'] = r3((ms - mb).mean() * 100); o['t_vs_0050_m'] = tstat(ms - mb)
        o['excess_vs_EW_pct_m'] = r3((ms - me).mean() * 100); o['t_vs_EW_m'] = tstat(ms - me)
        yr = lambda x: {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in x.groupby(x.index.year)}
        o['yearly_pct'] = yr(rr); o['yearly_0050_pct'] = yr(bb); o['yearly_EW_pct'] = yr(ee)
        return o

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'n_stocks': int(df['code'].nunique()), 'avg_hold_days': round(float(df['hold'].mean()), 1),
            'censored': int((df['why'] == 'censored').sum()),
            'mean_raw_pct': r3(df['raw'].mean() * 100), 'mean_net_pct': r3(df['net'].mean() * 100), 'mean_EW_pct': r3(df['ew'].mean() * 100),
            'mean_ab_gross_pct': r3(df['ab_gross'].mean() * 100), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': r3(df['ab'].mean() * 100), 'median_ab_net_pct': r3(df['ab'].median() * 100),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': r3(df['ab_s'].mean() * 100), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_net_vs_0050_pct': r3(df['ab50'].mean() * 100), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': r3((df['ab'] > 0).mean()), 'hit_raw': r3((df['raw'] > 0).mean()),
            'ab_net_trim1pct_mean_pct': r3(df['ab'].clip(df['ab'].quantile(0.01), df['ab'].quantile(0.99)).mean() * 100)}

WIN = {'select': ('2021-10-12', '2023-12-31'), 'valid': ('2024-01-01', '2026-10-31'), 'valid_2024_2025': ('2024-01-01', '2025-12-31')}
PWIN = {'select': ('2021-10-13', '2023-12-29'), 'valid': ('2024-01-01', '2026-10-22'), 'valid_2024_2025': ('2024-01-01', '2025-12-31')}

def run_ty(U, specs, win, pwin, tag):
    res = {}; yearly_ev = {}
    for key, (sk, H) in specs.items():
        ev, diag = U.build_events(U.SIG[sk], H)
        df = U.event_table(ev)
        o = {'diag': diag}
        for w, (a, b) in win.items():
            o[f'event_{w}'] = ev_stats(df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))])
        if 'valid' in win:
            o['event_valid_ex_src3'] = ev_stats(df[(df['sig'] >= '2024-01-01') & ~df['src3']])
            o['event_valid_2026'] = ev_stats(df[df['sig'] >= '2026-01-01'])
            # 金額大的優先：依 3 日累計金額在同月內排序的上半 vs 下半
            dd = df.copy(); dd['m'] = dd['entry'].dt.to_period('M')
            dd['top'] = dd.groupby('m')['amt3'].rank(pct=True) > 0.5
            for w, (a, b) in [('select', win['select']), ('valid', win['valid'])]:
                s = dd[(dd['sig'] >= pd.Timestamp(a)) & (dd['sig'] <= pd.Timestamp(b))]
                o[f'amount_split_{w}'] = {'top_half': ev_stats(s[s['top']]), 'bottom_half': ev_stats(s[~s['top']])}
        nav, npos, skp = U.portfolio(ev, BUY, SELL); nav_s, _, _ = U.portfolio(ev, BUY_S, SELL_S); nav_g, _, _ = U.portfolio(ev, 0.0, 0.0)
        o['port_skipped_full'] = skp
        for w, (a, b) in pwin.items():
            o[f'port_{w}'] = U.pstats(nav, npos, a, b)
            o[f'port_{w}']['CAGR_stress'] = U.pstats(nav_s, npos, a, b)['CAGR']; o[f'port_{w}']['CAGR_gross'] = U.pstats(nav_g, npos, a, b)['CAGR']
        yearly_ev[key] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2),
                                   'EW_pct': round(g['ew'].mean() * 100, 2), 't_month_ab_net': tstat(g.groupby(g['entry'].dt.to_period('M'))['ab'].mean())}
                          for y, g in df.groupby(df['sig'].dt.year)}
        res[key] = o
        df.drop(columns=[]).to_csv(HERE / f'events_{tag}_{key}.csv.gz', index=False)
        line = [key, diag]
        for w in win:
            e = o[f'event_{w}']; p = o[f'port_{w}']
            line += [f'| {w} n', e.get('n'), 'ab', e.get('mean_ab_net_pct'), 't', e.get('t_month_ab_net'), 'CAGR', p['CAGR'], '50', p['0050_CAGR'], 'EW', p['EW_CAGR']]
        print(*line, flush=True)
    return res, yearly_ev

# ---------------- CMoney 月換股版（exlib 引擎） ----------------
def sim(events, buy, sell, t0, t1):
    """同 exlib.sim_portfolio，但允許空手（入選 0 檔 → 全數賣出持現金）。"""
    N = L.N; u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]
            Pz = L.AO[t] if mode == 'open' else L.AC[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash
            tgt = np.zeros(N); Vn = Vt
            for _ in range(6):
                tgt[:] = 0
                if len(pk): tgt[pk] = Vn / len(pk)
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(N)
            if len(pk): u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
            else: cash = Vn
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
    return pd.Series(nav, index=L.IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')

def run_cm(U):
    F20 = U.FO.rolling(20, min_periods=20).sum()
    TRd = U.TR
    LIQ = U.LIQ
    cols = U.cols; cidx = np.array([L.CI[c] for c in cols])
    last_inst = U.IIDX[-1]
    sig_dates = [d for d in L.DATES if pd.Timestamp('2021-11-01') <= d <= last_inst and d in F20.index and F20.loc[d].notna().any()]
    end_date = [d for d in L.DATES if d > sig_dates[-1]][0]
    events, pools = {}, {}; held = set(); nblk = 0; npick = []; info = []
    for d in sig_dates:
        i = L.pos(d); ex = i + 1
        yoy = L.YOY.reindex([pd.Period(d, 'M') - 1]).iloc[0].reindex(cols)
        pool = LIQ.loc[d].to_numpy() if d in LIQ.index else np.zeros(len(cols), bool)
        sel = pool & (F20.loc[d].to_numpy() > 1000) & (TRd.loc[d].to_numpy() > 200) & (yoy.to_numpy() > 0)
        trad = np.isfinite(L.OPEN_RAW[ex, cidx]) & np.isfinite(L.ADJ_RAW[ex, cidx])
        sel &= trad; pool_ok = pool & trad
        pk = []
        for jj in np.where(sel)[0]:
            g = cidx[jj]
            if L.GAP_UP_OPEN[ex, g] and g not in held: nblk += 1; continue
            pk.append(g)
        events[ex] = ('open', np.array(pk, dtype=int)); pools[ex] = ('open', cidx[pool_ok])
        held = set(pk); npick.append(len(pk)); info.append((str(d.date()), len(pk), int(pool_ok.sum())))
    t0 = min(events); t1 = L.pos(end_date)
    ew, _ = sim(pools, 0.0, 0.0, t0, t1)
    out = {'n_rebalances': len(sig_dates), 'first': str(sig_dates[0].date()), 'last_signal': str(sig_dates[-1].date()), 'end': str(end_date.date()),
           'avg_names': round(float(np.mean(npick)), 1), 'min_names': int(np.min(npick)), 'max_names': int(np.max(npick)),
           'months_with_0_names': int(sum(1 for x in npick if x == 0)), 'gap_up_skipped': nblk, 'per_rebalance': info}
    W = {'select': [('2021-11-01', '2023-12-31')], 'valid': [('2024-01-01', '2026-10-31')], 'valid_2024_2025': [('2024-01-01', '2025-12-31')]}
    for lab, (b, s) in {'slip0.001': (BUY, SELL), 'stress': (BUY_S, SELL_S), 'gross': (0.0, 0.0)}.items():
        nav, tov = sim(events, b, s, t0, t1)
        out[lab] = {'turnover_value_per_rebalance': round(tov, 3), 'windows': L.all_windows(nav, L.E50_NAV, ew, W)}
        print('CM', lab, {k: (v['CAGR'], v.get('0050_CAGR'), v.get('EW_CAGR'), v.get('t_vs_0050_m'), v.get('t_vs_EW_m')) for k, v in out[lab]['windows'].items() if v}, flush=True)
    return out

def main():
    out = {'meta': {'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'inst_range': [str(FOR_ALL.index[0].date()), str(FOR_ALL.index[-1].date())]}}
    U = Universe(load_panel())
    out['meta'].update({'bad_days_dropped': U.D['bad'], 'n_stocks_panel': U.N, 'avg_pool_TWSE': round(float(U.LIQ.loc['2021-10':].sum(1).mean()), 1),
                        'condition_funnel_stock_days_TWSE': U.cond_counts})
    print(out['meta'], flush=True)
    specs = {'TY_H20': ('TY', 20), 'TY_H5': ('TY', 5), 'TY_BASE_H20': ('TY_BASE', 20), 'TY_BASE_H5': ('TY_BASE', 5),
             'TY_NOINC_H20': ('TY_NOINC', 20), 'TY_NOINC_H5': ('TY_NOINC', 5)}
    res, yev = run_ty(U, specs, WIN, PWIN, 'twse')
    out['results'] = res; out['yearly_events'] = yev
    out['CM'] = run_cm(U)
    # 補充：上市＋上櫃（history，2025-09-24 起）
    U2 = Universe(load_history())
    w2 = {'supp_2025-12_2026-10': ('2025-09-24', '2026-10-31')}
    p2 = {'supp_2025-12_2026-10': ('2025-12-22', '2026-10-22')}
    res2, yev2 = run_ty(U2, {'TY_H20': ('TY', 20), 'TY_H5': ('TY', 5), 'TY_BASE_H20': ('TY_BASE', 20), 'TY_BASE_H5': ('TY_BASE', 5)}, w2, p2, 'twse_tpex')
    mk = U2.D['market']
    for key in res2:
        df = pd.read_csv(HERE / f'events_twse_tpex_{key}.csv.gz', dtype={'code': str}, parse_dates=['sig', 'entry', 'exit'])
        df['mkt'] = df['code'].map(mk)
        res2[key]['by_market'] = {m: ev_stats(g) for m, g in df.groupby('mkt')}
    # 同窗上市面板對照
    cmp = {}
    for key in ['TY_H20', 'TY_H5']:
        df = pd.read_csv(HERE / f'events_twse_{key}.csv.gz', dtype={'code': str}, parse_dates=['sig', 'entry', 'exit'])
        cmp[key] = ev_stats(df[df['sig'] >= '2025-09-24'])
    out['supplement_TWSE_TPEX'] = {'avg_pool': round(float(U2.LIQ.loc['2026-01':].sum(1).mean()), 1), 'funnel': U2.cond_counts,
                                   'results': res2, 'panel_TWSE_same_window_events': cmp}
    # 判定
    def passes(o, w):
        e = o[f'event_{w}']; p = o[f'port_{w}']
        if e.get('n', 0) < 30 or e.get('n_months', 0) < 12: return {'insufficient': True, 'pass': False}
        ok_ew = (e['mean_ab_net_pct'] > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
        ok_50 = p['CAGR'] >= p['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    def grade(vs):
        if vs['select'].get('insufficient') and vs['valid'].get('insufficient'): return '無法測試'
        k = vs['select']['pass'] + vs['valid']['pass']
        return '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    V = {}
    for key in res:
        vs = {'select': passes(res[key], 'select'), 'valid': passes(res[key], 'valid')}; vs['verdict'] = grade(vs); V[key] = vs
    cw = out['CM']['slip0.001']['windows']
    def cpass(w):
        o = cw[w]
        if o is None or o['n_months'] < 12: return {'insufficient': True, 'pass': False}
        ok_ew = o['excess_vs_EW_pct_m'] > 0 and (o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9) >= 2
        ok_50 = o['CAGR'] >= o['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    vs = {'select': cpass('select'), 'valid': cpass('valid')}; vs['verdict'] = grade(vs); V['CM'] = vs
    V['overall'] = V['TY_H20']['verdict']
    out['verdict'] = V
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8'))
    d = {'PREREG': d['PREREG'], **out}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(V, ensure_ascii=False))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""外資連續買超＋拉回 事件回測。規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成本模型、0050（只讀取）
from btlib import tstat  # noqa: E402

P = L.P; X = L.X
BUY, SELL = L.cost_rates(0.001)
BUY_S, SELL_S = L.cost_rates('stress')
r3 = lambda x: round(float(x), 3)
is_common = lambda c: len(c) == 4 and c.isdigit() and not c.startswith('0') and not c.startswith('91')

INS = pd.read_csv(X / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign'], parse_dates=['date'])
INS = INS[INS['code'].map(is_common)]
FOR_ALL = INS.pivot_table(index='date', columns='code', values='foreign', aggfunc='sum')

SRC3_A, SRC3_B = pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23')

class U:
    def __init__(self):
        names = ['adj', 'close_raw', 'open', 'value', 'flag']
        D = {k: pd.read_parquet(P / f'{k}.parquet').loc['2021-01-01':] for k in names}
        good = D['adj'].notna().sum(axis=1) >= 300
        self.bad = [str(x.date()) for x in D['adj'].index[~good]]
        cols = [c for c in D['adj'].columns if is_common(c)]
        for k in D: D[k] = D[k].loc[good, cols]
        PIDX = D['adj'].index
        CAL = PIDX.union(FOR_ALL.index); self.CAL = CAL; self.T = len(CAL)
        self.ispanel = CAL.isin(PIDX)
        self.cols = cols; self.N = len(cols)
        b5 = D['flag'] == 5
        ADJ = D['adj'].mask(b5).reindex(CAL); CR = D['close_raw'].mask(b5).reindex(CAL); OP = D['open'].mask(b5).reindex(CAL)
        V = D['value'].reindex(CAL); FLOK = D['flag'].fillna(0).isin([0, 1, 2]).reindex(CAL, fill_value=False)
        VOL = V / CR                                                    # 股
        gaprow = pd.Series(~self.ispanel, index=CAL)
        # 60 日均成交值：上市後停牌記 0，缺口列 NaN → 不跨缺口
        v0 = V.fillna(0).where(V.notna().cumsum() > 0); v0[gaprow.to_numpy()] = np.nan
        V60 = v0.rolling(60, min_periods=60).mean()
        self.LIQ = ((V60 >= 1e7) & (CR >= 10) & FLOK & ADJ.notna()).fillna(False).astype(bool)
        self.MA5 = ADJ.rolling(5, min_periods=5).mean(); self.MA20 = ADJ.rolling(20, min_periods=20).mean()
        self.ADJ, self.CR, self.VOL = ADJ, CR, VOL
        fac = ADJ / CR; AO = OP * fac
        ACf = ADJ.ffill()
        self.AOn = AO.to_numpy(); self.ACn = ACf.to_numpy(); self.ADJn = ADJ.to_numpy()
        self.GAP_UP = (self.AOn / ACf.shift(1).to_numpy() - 1) >= 0.10 - 0.006
        self.BELOW20 = (ADJ < self.MA20).to_numpy()                      # NaN → False
        fin = np.isfinite(self.ADJn); self.LASTV = np.where(fin.any(0), self.T - 1 - np.argmax(fin[::-1], axis=0), 0)
        # 下一個可用開盤（含當日）
        idx = np.where(np.isfinite(self.AOn), np.arange(self.T)[:, None], self.T)
        self.NEXTOPEN = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
        # 等權（面板日曆，開盤對開盤）
        pm = np.where(self.ispanel)[0]; self.pm = pm
        AOp = AO.iloc[pm]; LIQp = self.LIQ.iloc[pm]
        oo = (AOp.shift(-1) / AOp - 1).where(LIQp).clip(-0.5, 1.0)       # d 開盤→下一面板日開盤，d 前一日… 見下
        # 股池以「前一面板日」符合：d→d+1 的開盤報酬，用 d-1 的 LIQ
        oo = (AOp.shift(-1) / AOp - 1).where(LIQp.shift(1, fill_value=False)).clip(-0.5, 1.0)
        e = oo.mean(axis=1).fillna(0).to_numpy()
        LEWp = np.concatenate([[0.0], np.cumsum(np.log1p(e))])[:len(pm)]
        self.LEW = np.full(self.T, np.nan); self.LEW[pm] = LEWp; self.LEW = pd.Series(self.LEW).ffill().to_numpy()
        ADJp = ADJ.iloc[pm]; rcc = ADJp / ADJp.ffill().shift(1) - 1
        ew_cc = rcc.where(LIQp.shift(1, fill_value=False)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
        self.ew_cc = ew_cc
        e50 = L.e50.reindex(CAL.union(L.e50.index)).ffill().reindex(CAL)
        self.E50n = e50.to_numpy(); self.r50 = e50.iloc[pm].pct_change().fillna(0)
        self.SRC3 = (CAL >= SRC3_A) & (CAL <= SRC3_B)
        self.FO = FOR_ALL.reindex(index=CAL, columns=cols).fillna(0.0)
        self.FO[CAL < FOR_ALL.index[0]] = np.nan

    def signal(self, n=5, thr=0.03, pullback=True):
        FO = self.FO
        buy = (FO > 0)
        c1 = buy.rolling(n, min_periods=n).sum() >= n
        has = (self.VOL.notna() & self.CR.notna())
        hasn = has.rolling(n, min_periods=n).sum() >= n
        part = (FO * 1000).rolling(n, min_periods=n).sum() / self.VOL.rolling(n, min_periods=n).sum()
        c2 = part >= thr
        S = c1 & hasn & c2 & self.LIQ
        fun = {'pool_streak': int((c1 & hasn & self.LIQ).to_numpy().sum()), '+participation': int(S.to_numpy().sum())}
        if pullback:
            S = S & (self.ADJ < self.MA5) & (self.ADJ >= self.MA20)
            fun['+pullback'] = int(S.to_numpy().sum())
        return S.fillna(False).to_numpy(), part.to_numpy(), fun

    def build_events(self, S, PART, H, start='2021-10-14', last_sig='2026-10-07'):
        ev = []; nblock = 0; nnoopen = 0; nsig = 0; nstop = 0
        s0 = int(np.searchsorted(self.CAL.values, np.datetime64(start))); s1 = int(np.searchsorted(self.CAL.values, np.datetime64(last_sig), 'right'))
        T = self.T
        for j in range(self.N):
            ts = np.where(S[:, j])[0]; ts = ts[(ts >= s0) & (ts < s1)]
            free_from = -1
            for t in ts:
                nsig += 1
                if t < free_from: continue
                e = t + 1
                if e >= T: continue
                if not np.isfinite(self.AOn[e, j]): nnoopen += 1; continue
                if self.GAP_UP[e, j]: nblock += 1; continue
                tgt = e + H; why = 'time'
                # MA20 停損：d ∈ [e, tgt-2]，收盤 < MA20 → d+1
                for d in range(e, min(tgt - 1, T)):
                    if self.BELOW20[d, j]: tgt = d + 1; why = 'stop'; break
                if tgt < T and self.NEXTOPEN[tgt, j] < T:
                    xp = int(self.NEXTOPEN[tgt, j]); xprice = self.AOn[xp, j]
                else:
                    li = int(self.LASTV[j])                                  # 之後沒有開盤：下市/長停 → 最後收盤；面板結束 → censored
                    xp = max(li, e); xprice = self.ACn[xp, j]; why = 'censored' if li >= T - 1 else 'delisted'
                nstop += why == 'stop'
                ev.append(dict(j=j, t=t, e=e, x=xp, why=why, p0=self.AOn[e, j], p1=xprice,
                               pri=float(PART[t, j]) if np.isfinite(PART[t, j]) else 0.0))
                free_from = xp
        return ev, dict(raw_signals=nsig, events=len(ev), blocked_gap_up=nblock, no_open=nnoopen, stop_exits=nstop)

    def event_table(self, ev):
        rows = []
        for v in ev:
            j, t, e, x = v['j'], v['t'], v['e'], v['x']
            raw = v['p1'] / v['p0'] - 1
            net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1
            net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
            ew = math.expm1(self.LEW[x] - self.LEW[e])
            b50 = self.E50n[max(x - 1, t)] / self.E50n[t] - 1
            rows.append((self.cols[j], self.CAL[t], self.CAL[e], self.CAL[x], x - e, v['why'], raw, net, net_s, ew, b50,
                         bool(self.SRC3[t:x + 1].any()), v['pri']))
        df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'exit', 'hold', 'why', 'raw', 'net', 'net_s', 'ew', 'b50', 'src3', 'part'])
        df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']; df['ab_gross'] = df['raw'] - df['ew']
        return df

    def portfolio(self, ev, buy, sell, slots=20):
        ent, exits = {}, {}
        for k, v in enumerate(ev):
            ent.setdefault(v['e'], []).append(k); exits.setdefault(v['x'], []).append(k)
        held = {}; cash = 1.0; nav = np.empty(self.T); npos = np.zeros(self.T); skipped = 0
        AOn, ACn = self.AOn, self.ACn
        for d in range(self.T):
            for k in [k for k in exits.get(d, []) if k in held and ev[k]['why'] != 'censored']:
                u = held.pop(k); cash += u * ev[k]['p1'] * (1 - sell)
            if d in ent:
                pv = cash + sum(u * (AOn[d, ev[k]['j']] if np.isfinite(AOn[d, ev[k]['j']]) else ACn[d - 1, ev[k]['j']]) for k, u in held.items())
                for k in sorted(ent[d], key=lambda k: -ev[k]['pri']):
                    if len(held) >= slots or cash <= 1e-9: skipped += 1; continue
                    amt = min(pv / slots, cash); cash -= amt
                    held[k] = amt / (1 + buy) / AOn[d, ev[k]['j']]
            nav[d] = cash + sum(u * ACn[d, ev[k]['j']] for k, u in held.items())
            npos[d] = len(held)
        pm = self.pm
        return pd.Series(nav[pm], index=self.CAL[pm]), pd.Series(npos[pm], index=self.CAL[pm]), skipped

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
        o['vol'] = round(float(rr.std() * math.sqrt(252)), 4); o['avg_positions'] = round(float(npos[m].mean()), 1)
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
            'stop_share': r3((df['why'] == 'stop').mean()), 'censored': int((df['why'] == 'censored').sum()),
            'mean_raw_pct': r3(df['raw'].mean() * 100), 'mean_net_pct': r3(df['net'].mean() * 100), 'mean_EW_pct': r3(df['ew'].mean() * 100),
            'mean_ab_gross_pct': r3(df['ab_gross'].mean() * 100), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': r3(df['ab'].mean() * 100), 'median_ab_net_pct': r3(df['ab'].median() * 100),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': r3(df['ab_s'].mean() * 100), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_net_vs_0050_pct': r3(df['ab50'].mean() * 100), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': r3((df['ab'] > 0).mean()), 'hit_raw': r3((df['raw'] > 0).mean()),
            'max_raw_pct': r3(df['raw'].max() * 100), 'min_raw_pct': r3(df['raw'].min() * 100),
            'ab_net_trim1pct_mean_pct': r3(df['ab'].clip(df['ab'].quantile(0.01), df['ab'].quantile(0.99)).mean() * 100)}

WIN = {'select': ('2021-10-14', '2023-12-31'), 'valid': ('2024-01-01', '2026-10-07'), 'valid_2024_2025': ('2024-01-01', '2025-12-31')}
PWIN = {'select': ('2021-10-15', '2023-12-29'), 'valid': ('2024-01-01', '2026-10-22'), 'valid_2024_2025': ('2024-01-01', '2025-12-31')}
SPECS = {'BASE': dict(n=5, thr=0.03, pb=True, H=10), 'S3': dict(n=3, thr=0.03, pb=True, H=10), 'S7': dict(n=7, thr=0.03, pb=True, H=10),
         'P2': dict(n=5, thr=0.02, pb=True, H=10), 'P5': dict(n=5, thr=0.05, pb=True, H=10), 'NOPB': dict(n=5, thr=0.03, pb=False, H=10),
         'H5': dict(n=5, thr=0.03, pb=True, H=5), 'H20': dict(n=5, thr=0.03, pb=True, H=20)}

def main():
    u = U()
    out = {'meta': {'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S], 'bad_days_dropped': u.bad, 'n_stocks': u.N,
                    'inst_range': [str(FOR_ALL.index[0].date()), str(FOR_ALL.index[-1].date())],
                    'avg_pool': round(float(u.LIQ.loc['2021-10':].sum(axis=1)[u.ispanel[u.CAL >= '2021-10-01']].mean()), 1),
                    'first_pool_day_after_gap': str(u.LIQ.loc['2024-01-01':].index[u.LIQ.loc['2024-01-01':].sum(axis=1).to_numpy() > 0][0].date())}}
    print(out['meta'], flush=True)
    res = {}; yev = {}
    for key, s in SPECS.items():
        S, PART, fun = u.signal(s['n'], s['thr'], s['pb'])
        ev, diag = u.build_events(S, PART, s['H'])
        diag['funnel_stock_days'] = fun
        df = u.event_table(ev)
        o = {'diag': diag}
        for w, (a, b) in WIN.items():
            o[f'event_{w}'] = ev_stats(df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))])
        o['event_valid_ex_src3'] = ev_stats(df[(df['sig'] >= '2024-01-01') & ~df['src3']])
        o['event_valid_2026'] = ev_stats(df[df['sig'] >= '2026-01-01'])
        nav, npos, skp = u.portfolio(ev, BUY, SELL); nav_s, _, _ = u.portfolio(ev, BUY_S, SELL_S); nav_g, _, _ = u.portfolio(ev, 0.0, 0.0)
        o['port_skipped_full'] = skp
        for w, (a, b) in PWIN.items():
            o[f'port_{w}'] = u.pstats(nav, npos, a, b)
            o[f'port_{w}']['CAGR_stress'] = u.pstats(nav_s, npos, a, b)['CAGR']; o[f'port_{w}']['CAGR_gross'] = u.pstats(nav_g, npos, a, b)['CAGR']
        yev[key] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2),
                             'EW_pct': round(g['ew'].mean() * 100, 2), 't_month_ab_net': tstat(g.groupby(g['entry'].dt.to_period('M'))['ab'].mean())}
                    for y, g in df.groupby(df['sig'].dt.year)}
        res[key] = o
        df.to_csv(HERE / f'events_{key}.csv.gz', index=False)
        if key == 'BASE': nav.to_frame('nav').to_parquet(HERE / 'nav_BASE.parquet')
        line = [key, diag]
        for w in WIN:
            e = o[f'event_{w}']; p = o[f'port_{w}']
            line += [f'| {w} n', e.get('n'), 'ab', e.get('mean_ab_net_pct'), 't', e.get('t_month_ab_net'), 'CAGR', p['CAGR'], '50', p['0050_CAGR'], 'EW', p['EW_CAGR'], 'tEW', p['t_vs_EW_m'], 't50', p['t_vs_0050_m']]
        print(*line, flush=True)
    out['results'] = res; out['yearly_events'] = yev
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
    V['overall'] = V['BASE']['verdict']
    out['verdict'] = V
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8'))
    d = {'PREREG': d['PREREG'], **out}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(V, ensure_ascii=False))

if __name__ == '__main__':
    main()

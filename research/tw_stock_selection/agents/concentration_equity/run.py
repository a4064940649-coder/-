#!/usr/bin/env python3
"""concentration_equity：A 持股集中度劑量反應、B 淨值曲線濾網。規則見 results.json 的 PREREG（prereg.py 寫入，本程式只讀不改）。
執行：python3 -I -B prereg.py && python3 -I -B run.py"""
import json, hashlib, math, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
RES = json.loads((H / 'results.json').read_text(encoding='utf-8'))
PR = RES['PREREG']
assert hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256'], 'PREREG changed'
sys.path.insert(0, str(A / 'revenue_longhistory'))
import lhlib as L  # noqa: E402
from btlib import tstat  # noqa: E402

S = L.Setup(L.yoy_long(), start='2005-01-01')
DATES = S.DATES; t0 = L.pos(S.dates[0]); t1 = L.pos(S.dates[-1])
NIDX = L.IDX[t0:t1 + 1]
first = S.dates[0]
WIN = {'P1': [(first, '2013-12-31')], 'P2': [('2014-01-01', '2020-12-31')], 'P3': [('2021-03-01', '2026-10-31')],
       'OOS_2006_2020': [(first, '2020-12-31')], 'all': [(first, '2026-10-31')]}
BEAR = {'2008': ('2008-01-01', '2008-12-31'), '2011': ('2011-01-01', '2011-12-31'), '2015': ('2015-01-01', '2015-12-31'),
        '2018': ('2018-01-01', '2018-12-31'), '2020-03': ('2020-02-01', '2020-04-30'), '2022': ('2022-01-01', '2022-12-31')}
BM, SM = L.cost_rates(0.001); BS, SS = L.cost_rates('stress')

# ---------------- 選股（lhlib.Setup.selection 的改寫：可指定場外訊號日） ----------------
el = S.t300 & S.s52.notna() & ~S.inD & S.yoy3.notna()
r52 = S.s52.where(el).rank(axis=1, pct=True); rrv = S.yoy3.where(el).rank(axis=1, pct=True)
SC = (0.5 * r52 + 0.5 * rrv).to_numpy(); EL = el.to_numpy()
DPOS = {d: k for k, d in enumerate(S.dates)}

def select(n, off=frozenset()):
    """回傳 events {ex: picks(依分數排序)}、pools。off＝場外的訊號日集合（該期持現金，held 清空）。"""
    events, pools = {}, {}; held = set()
    for d in S.dates[:-1]:
        k = DPOS[d]; i = L.pos(d); ex = i + 1
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        ok = EL[k] & trad & np.isfinite(SC[k])
        if ok.sum() < 50: continue
        pools[ex] = np.where(ok)[0]
        if d in off:
            events[ex] = np.array([], dtype=int); held = set(); continue
        order = SC[k].copy(); order[~ok] = np.nan
        cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
        pick = []
        for j in cand:
            if L.GAP_UP_OPEN_ERA[ex, j] and j not in held: continue
            pick.append(j)
            if len(pick) == n: break
        events[ex] = np.array(pick, dtype=int); held = set(pick)
    return events, pools

def weights(pk, scheme):
    if len(pk) == 0: return np.array([])
    if scheme == 'EW': return np.full(len(pk), 1.0 / len(pk))
    w = np.arange(len(pk), 0, -1, dtype=float); return w / w.sum()       # 第 k 名 ∝ N+1−k

def sim(events, scheme, buy, sell):
    """lhlib.sim_portfolio 延伸：權重可不等、空清單＝全部現金。回傳 nav、平均換手、持股紀錄。"""
    AO, AC = L.AO, L.AC; N = L.N
    u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []; book = []
    for t in range(t0, t1 + 1):
        if t in events:
            pk = events[t]; w = weights(pk, scheme)
            Pz = AO[t]; h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0); Vt = h.sum() + cash
            tgt = np.zeros(N); Vn = Vt
            for _ in range(6):
                tgt[:] = 0
                if len(pk): tgt[pk] = w * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(N)
            if len(pk): u[pk] = tgt[pk] / Pz[pk]
            cash = Vn - tgt.sum(); book.append((t, pk, w))
        nav[t - t0] = np.nansum(u * AC[t]) + cash
    return pd.Series(nav, index=NIDX), float(np.mean(tov[1:])), book

def contributions(book):
    """每個持有期（換股開盤到下次換股開盤）每檔的毛額貢獻 = 權重 × 該期報酬。"""
    rows = []
    for (ta, pk, w), (tb, _, _) in zip(book[:-1], book[1:]):
        if len(pk) == 0: continue
        r = L.AO[tb, pk] / L.AO[ta, pk] - 1
        for j, wi, ri in zip(pk, w, r): rows.append((L.IDX[ta], L.COLS[j], wi, ri, wi * ri))
    return pd.DataFrame(rows, columns=['ex', 'code', 'w', 'r', 'contrib'])

def bear(nav):
    o = {}
    for k, (a, b) in BEAR.items():
        x = nav[(nav.index >= pd.Timestamp(a) - pd.Timedelta(days=10)) & (nav.index <= pd.Timestamp(b))]
        pre = x[x.index < pd.Timestamp(a)]; base = pre.iloc[-1] if len(pre) else x.iloc[0]
        y = pd.concat([pd.Series([base]), x[x.index >= pd.Timestamp(a)]]).to_numpy()
        o[k] = {'ret': round(float(y[-1] / y[0] - 1), 4), 'maxDD': round(float((y / np.maximum.accumulate(y) - 1).min()), 4)}
    return o

def in_seg(ix, segs):
    m = np.zeros(len(ix), bool)
    for a, b in segs: m |= (ix >= pd.Timestamp(a)) & (ix <= pd.Timestamp(b))
    return m

def stats(nav, ew):
    out = {}
    for w, segs in WIN.items():
        o = L.daily_stats(nav, segs, DATES, L.E50, ew)
        o['Calmar'] = round(o['CAGR'] / abs(o['maxDD']), 3) if o['maxDD'] < 0 else None
        if w != 'all': o.pop('0050_yearly_pct', None); o.pop('EW_yearly_pct', None)
        if w not in ('all',): o.pop('yearly_pct', None)
        out[w] = o
    return out

def risk(nav, book):
    cb = contributions(book); o = {}
    nd = nav.reindex(DATES); pr = (nd.shift(-1) / nd - 1).iloc[:-1].dropna()
    cm = nav.resample('ME').last().pct_change().dropna()
    for w, segs in WIN.items():
        c = cb[in_seg(pd.DatetimeIndex(cb['ex']), segs)]; p = pr[in_seg(pr.index, segs)]; m = cm[in_seg(cm.index, segs)]
        wc = c.loc[c['contrib'].idxmin()]
        o[w] = {'worst_rebal_period_ret': round(float(p.min()), 4), 'worst_rebal_period_date': str(p.idxmin().date()),
                'worst_calendar_month_ret': round(float(m.min()), 4), 'worst_calendar_month': str(m.idxmin().date())[:7],
                'worst_holding_contrib': round(float(wc['contrib']), 4), 'worst_holding': f"{wc['code']} {str(wc['ex'].date())} w={wc['w']:.3f} r={wc['r']:.3f}",
                'p1_holding_contrib': round(float(c['contrib'].quantile(0.01)), 4), 'n_holding_contrib_le_-5pct': int((c['contrib'] <= -0.05).sum()),
                'best_holding_contrib': round(float(c['contrib'].max()), 4), 'n_holdings': int(len(c))}
    return o

def sp(x, y):
    return round(float(pd.Series(x).rank().corr(pd.Series(y).rank())), 3)

res = {'dates': {'first_signal': str(S.dates[0].date()), 'last_signal': str(S.dates[-1].date()), 'n_signal_dates': len(S.dates) - 1}}
# ---------------- 等權同池、一致性檢查 ----------------
ev30, pools = select(30)
ew = L.sim_portfolio({t: ('open', p) for t, p in pools.items()}, 0.0, 0.0, t0, t1)[0]
ev_l, _ = S.selection(30)
assert all(np.array_equal(ev_l[t][1], ev30[t]) for t in ev_l) and set(ev_l) == set(ev30), 'selection mismatch'
nav_l, _ = L.sim_portfolio(ev_l, BM, SM, t0, t1); nav_m, _, _ = sim(ev30, 'EW', BM, SM)
oos = pd.read_parquet(A / 'revenue_longhistory' / 'oos_nav.parquet')
res['consistency'] = {'max_rel_diff_vs_lhlib_sim': float(np.abs(nav_m / nav_l - 1).max()),
                      'max_rel_diff_vs_oos_nav_n30': float(np.abs(nav_m / oos['M2_n30_slip0.001'].reindex(NIDX) - 1).max())}
print('consistency', res['consistency'], flush=True)
res['EW_pool'] = {w: {k: v for k, v in L.daily_stats(ew, s, DATES, L.E50).items() if k in ('CAGR', 'maxDD', 't_vs_0050_m')} for w, s in WIN.items()}
res['0050'] = {w: {k: v for k, v in L.daily_stats(L.E50.iloc[t0:t1 + 1], s, DATES).items() if k in ('CAGR', 'maxDD', 'yearly_pct')} for w, s in WIN.items()}
res['0050_bear'] = bear(L.E50.iloc[t0:t1 + 1])

# ================= A：集中度 =================
VAR = [('EW', n) for n in (3, 5, 10, 15, 20, 30, 50)] + [('SW', 10), ('SW', 30)]
A_ = {}; NAVS = {}; EVS = {}
for sch, n in VAR:
    ev = ev30 if n == 30 else select(n)[0]; EVS[n] = ev
    nav, tov, book = sim(ev, sch, BM, SM); navs, _, _ = sim(ev, sch, BS, SS); navg, _, _ = sim(ev, sch, 0.0, 0.0)
    key = f'{sch}{n}'; NAVS[key] = nav
    st = stats(nav, ew); sst = stats(navs, ew); gst = stats(navg, ew)
    A_[key] = {'turnover': round(tov, 3), 'mean_names': round(float(np.mean([len(v) for v in ev.values()])), 1),
               'windows': st, 'stress_CAGR': {w: sst[w]['CAGR'] for w in WIN}, 'stress_t_vs_0050_m': {w: sst[w]['t_vs_0050_m'] for w in WIN},
               'gross_CAGR': {w: gst[w]['CAGR'] for w in WIN}, 'risk': risk(nav, book), 'bear': bear(nav)}
    print(key, ' | '.join(f"{w} {st[w]['CAGR']*100:5.1f}% {st[w]['maxDD']*100:6.1f}% C{st[w]['Calmar']} t0050 {st[w]['t_vs_0050_m']} tEW {st[w]['t_vs_EW_m']}" for w in ('P1', 'P2', 'P3')),
          'stress', A_[key]['stress_CAGR']['P1'], A_[key]['stress_CAGR']['P2'], A_[key]['stress_CAGR']['P3'], flush=True)
res['A'] = A_
EWN = [3, 5, 10, 15, 20, 30, 50]
dose = {}
for w in ('P1', 'P2', 'P3', 'OOS_2006_2020'):
    g = {m: [A_[f'EW{n}']['windows'][w][m] for n in EWN] for m in ('CAGR', 'maxDD', 'Calmar', 't_vs_0050_m', 't_vs_EW_m', 'vol')}
    dose[w] = {f'spearman_N_vs_{m}': sp(EWN, v) for m, v in g.items()}
    dose[w]['series'] = {m: dict(zip(EWN, v)) for m, v in g.items()}
res['A_dose_response'] = dose
# 配對差 t
pair = {}
def per_ret(nav):
    nd = nav.reindex(DATES); return (nd.shift(-1) / nd - 1).iloc[:-1]
for n in (3, 5, 10, 'SW10'):
    k = f'EW{n}' if isinstance(n, int) else n
    d = per_ret(NAVS[k]) - per_ret(NAVS['EW30'])
    pair[f'{k}_minus_EW30'] = {w: {'mean_pct_m': round(float(d[in_seg(d.index, s)].mean() * 100), 2), 't': tstat(d[in_seg(d.index, s)].dropna())} for w, s in WIN.items()}
d = per_ret(NAVS['SW30']) - per_ret(NAVS['EW30'])
pair['SW30_minus_EW30'] = {w: {'mean_pct_m': round(float(d[in_seg(d.index, s)].mean() * 100), 2), 't': tstat(d[in_seg(d.index, s)].dropna())} for w, s in WIN.items()}
res['A_pair_vs_EW30'] = pair
cond = {}
for n in (3, 5, 10):
    cond[f'EW{n}'] = {w: bool(A_[f'EW{n}']['windows'][w]['CAGR'] > A_['EW30']['windows'][w]['CAGR'] and
                          A_[f'EW{n}']['windows'][w]['t_vs_0050_m'] > A_['EW30']['windows'][w]['t_vs_0050_m']) for w in ('P1', 'P2', 'P3')}
cond['SW10_info'] = {w: bool(A_['SW10']['windows'][w]['CAGR'] > A_['EW30']['windows'][w]['CAGR'] and
                             A_['SW10']['windows'][w]['t_vs_0050_m'] > A_['EW30']['windows'][w]['t_vs_0050_m']) for w in ('P1', 'P2', 'P3')}
res['A_verdict'] = {'conditions': cond, 'concentration_favourable': all(all(cond[f'EW{n}'].values()) for n in (3, 5, 10)),
                    'N10_alone_info': all(cond['EW10'].values())}
print('A verdict', res['A_verdict'], flush=True)

# ================= B：淨值曲線濾網 =================
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
txma = tx.rolling(200, min_periods=200).mean()
def asof(s, ix): return s.reindex(s.index.union(ix)).ffill().reindex(ix)
TXIN = asof((tx >= txma) | txma.isna(), NIDX).astype(bool)

def states(nav0):
    """回傳每日狀態（True＝場內），訊號只用當日收盤以前的影子淨值。"""
    v = nav0.to_numpy(); ma = nav0.rolling(200, min_periods=200).mean().to_numpy(); hi60 = nav0.rolling(60, min_periods=1).max().to_numpy()
    e1 = ~(np.isfinite(ma) & (v < ma))
    e2 = np.ones(len(v), bool); inn = True; peak = v[0]
    for i in range(len(v)):
        if inn:
            peak = max(peak, v[i])
            if v[i] / peak - 1 < -0.20: inn = False
        else:
            if v[i] >= hi60[i] - 1e-12 or (np.isfinite(ma[i]) and v[i] >= ma[i]): inn = True; peak = v[i]
        e2[i] = inn
    me = nav0.groupby([nav0.index.year, nav0.index.month]).tail(1)
    mes = (me >= me.rolling(10, min_periods=10).mean()) | me.rolling(10, min_periods=10).mean().isna()
    e3 = asof(mes, nav0.index).astype(bool).to_numpy()
    return {'E1': pd.Series(e1, nav0.index), 'E2': pd.Series(e2, nav0.index), 'E3': pd.Series(e3, nav0.index), 'S1': TXIN}

def daily_events(base_ev, st):
    """敏感度：狀態改變隔日開盤執行；回場買最近一次換股日名單。"""
    ev = {}; last = None; s = st.to_numpy()
    for t in range(t0 + 1, t1 + 1):
        k = t - t0
        if t in base_ev: last = base_ev[t]
        if t in base_ev: ev[t] = base_ev[t] if s[k - 1] else np.array([], dtype=int)
        elif k >= 2 and s[k - 1] != s[k - 2] and last is not None:
            ev[t] = last if s[k - 1] else np.array([], dtype=int)
    return ev

def switch_info(ev, nav0):
    ts = sorted(ev); st = [len(ev[t]) > 0 for t in ts]
    eps = []; cur = None
    for t, s in zip(ts, st):
        if not s and cur is None: cur = t
        if s and cur is not None: eps.append((cur, t)); cur = None
    if cur is not None: eps.append((cur, t1 + 1))
    v = nav0.to_numpy()
    E = [[str(L.IDX[a].date()), str(L.IDX[min(b, t1)].date()), round(float(v[min(b, t1 + 1) - 1 - t0] / v[a - 1 - t0] - 1), 4)] for a, b in eps]
    out = {}
    for w, segs in WIN.items():
        m = in_seg(pd.DatetimeIndex([L.IDX[t] for t in ts]), segs); ss = np.array(st)[m]
        prev = np.r_[True, np.array(st)][:len(st)][m]
        e_in = [e for e in E if in_seg(pd.DatetimeIndex([e[0]]), segs)[0]]
        out[w] = {'switches': int((ss != prev).sum()), 'pct_rebal_out': round(float((~ss).mean()), 3) if len(ss) else None,
                  'episodes': len(e_in), 'missed_up_sum': round(sum(e[2] for e in e_in if e[2] > 0), 4), 'avoided_down_sum': round(sum(e[2] for e in e_in if e[2] <= 0), 4),
                  'n_episodes_shadow_up': sum(1 for e in e_in if e[2] > 0)}
    return out, E

B = {}
for n in (30, 10):
    ev0 = EVS[n]; nav0 = NAVS[f'EW{n}']; STT = states(nav0)
    B[f'n{n}'] = {'none': {'windows': A_[f'EW{n}']['windows'], 'bear': A_[f'EW{n}']['bear']}}
    for name in ('E1', 'E2', 'E3', 'S1'):
        st = STT[name]
        off = frozenset(d for d in S.dates[:-1] if not st.loc[d])
        ev, _ = select(n, off)
        nav, tov, _ = sim(ev, 'EW', BM, SM); navs, _, _ = sim(ev, 'EW', BS, SS)
        si, E = switch_info(ev, nav0); w_ = stats(nav, ew)
        for w in WIN: w_[w].update(si[w])
        B[f'n{n}'][name] = {'windows': w_, 'bear': bear(nav), 'episodes': E, 'stress_CAGR': {w: stats(navs, ew)[w]['CAGR'] for w in WIN}}
        NAVS[f'n{n}_{name}'] = nav
        print(f'n{n} {name}', ' | '.join(f"{w} {w_[w]['CAGR']*100:5.1f}% {w_[w]['maxDD']*100:6.1f}% C{w_[w]['Calmar']} t{w_[w]['t_vs_0050_m']} sw{w_[w]['switches']}" for w in ('P1', 'P2', 'P3')), flush=True)
    for name, base in (('E1d', 'E1'), ('E2d', 'E2'), ('E3m', 'E3')):
        ev = daily_events(ev0, STT[base]); nav, _, _ = sim(ev, 'EW', BM, SM)
        si, E = switch_info(ev, nav0); w_ = stats(nav, ew)
        for w in WIN: w_[w].update(si[w])
        B[f'n{n}'][name] = {'sensitivity_only': True, 'windows': w_, 'bear': bear(nav)}
        print(f'n{n} {name}(sens)', ' | '.join(f"{w} {w_[w]['CAGR']*100:5.1f}% {w_[w]['maxDD']*100:6.1f}% C{w_[w]['Calmar']} sw{w_[w]['switches']}" for w in ('P1', 'P2', 'P3')), flush=True)
    # 挑選與判定
    base = B[f'n{n}']['none']['windows']
    cal = {k: B[f'n{n}'][k]['windows']['P1']['Calmar'] for k in ('E1', 'E2', 'E3')}
    pick = max(cal, key=cal.get)
    def judge(k):
        p = B[f'n{n}'][k]['windows']
        ok = {w: bool(p[w]['Calmar'] > base[w]['Calmar'] and p[w]['CAGR'] >= base[w]['CAGR'] - 0.03) for w in ('P2', 'P3')}
        c = sum(ok.values()); return ok, ('可用' if c == 2 else '待觀察' if c == 1 else '不可用')
    ok, vd = judge(pick)
    B[f'n{n}']['selection'] = {'P1_calmar': cal, 'none_P1_calmar': base['P1']['Calmar'], 'picked': pick, 'pass': ok, 'verdict': vd,
                               'S1_same_engine_same_rule': dict(zip(('pass', 'verdict'), judge('S1'))),
                               'all_same_rule_info_only': {k: judge(k)[1] for k in ('E1', 'E2', 'E3', 'E1d', 'E2d', 'E3m')}}
    print(f'n{n} picked {pick} {ok} {vd}; S1 {judge("S1")}', flush=True)
res['B'] = B
td = json.loads((A / 'timing_daily' / 'results.json').read_text(encoding='utf-8'))
res['B_timing_daily_S1_overlay_ref'] = {k: td['results']['main_cash0'][k] for k in ('n30|none', 'n30|S1', 'n10|none', 'n10|S1')}
res['B_timing_daily_S1_engine_ref'] = {k: td['POSTHOC_diagnostics'][k] for k in ('engine_S1_dplus1_open|n30', 'engine_S1_dplus1_open|n10')}
# 逐年
yr = {k: L.daily_stats(v, WIN['all'], DATES)['yearly_pct'] for k, v in NAVS.items()}
yr['0050'] = res['0050']['all']['yearly_pct']; res['yearly_pct'] = yr
# 濾網狀態與主策略相關：E 濾網場外期間 vs 大盤場外期間重疊
ov = {}
for n in (30, 10):
    STT = states(NAVS[f'EW{n}']); sd = pd.DatetimeIndex(S.dates[:-1])
    for k in ('E1', 'E2', 'E3'):
        a = ~STT[k].reindex(sd).to_numpy(); b = ~TXIN.reindex(sd).to_numpy()
        ov[f'n{n}_{k}'] = {'rebal_out_frac': round(float(a.mean()), 3), 'S1_out_frac': round(float(b.mean()), 3), 'both_out_frac': round(float((a & b).mean()), 3)}
res['B_overlap_with_S1'] = ov
RES['results'] = res
(H / 'results.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(NAVS).to_parquet(H / 'navs.parquet')
print('done')

#!/usr/bin/env python3
"""當沖比率籌碼濾網 A（五分位／>50%）、B（當沖比暴增事件）、C（主策略剔除 >50%）。
規則見 results.json 的 PREREG（prereg.py 事前寫入，不改）。價格、成本、開盤漲停、主策略選股取自 exec_rebuild/exlib.py。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402
tstat = L.tstat
IDX, COLS, N = L.IDX, L.COLS, L.N
AO, AC = L.AO, L.AC
GAP_UP = L.GAP_UP_OPEN
DR = np.array([c.startswith('91') for c in COLS])
SEL = ('2021-01-01', '2023-12-31'); VAL = ('2024-01-01', '2026-10-31'); V2425 = ('2024-01-01', '2025-12-31'); SRC3 = ('2024-02-15', '2025-09-23')
PERIODS = {'select_2021_2023': SEL, 'valid_2024_2026': VAL, 'valid_2024_2025': V2425, 'src3_2024-02-15_2025-09-23': SRC3, 'full_2021_2026': ('2021-01-01', '2026-10-31')}
GAPDAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 7      # 事前規則 7；敏感度（事後發現 7 會誤刪農曆年月份）用 20
OUTF = 'results.json' if GAPDAYS == 7 else f'results_gap{GAPDAYS}.json'
RES = json.load(open(HERE / 'results.json'))
log = open(HERE / ('run_log.txt' if GAPDAYS == 7 else f'run_log_gap{GAPDAYS}.txt'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

# ---------------- 當沖資料 ----------------
dt = pd.read_csv('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/scr_extra/daytrade_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
nbad = int(((dt.dt_ratio > 100) | (dt.dt_ratio < 0)).sum())
dt.loc[(dt.dt_ratio > 100) | (dt.dt_ratio < 0), 'dt_ratio'] = np.nan
DTW = dt.pivot_table(index='date', columns='code', values='dt_ratio', aggfunc='last').sort_index()
DTCAL = DTW.index
DTW = DTW.reindex(columns=COLS)
DT20 = DTW.rolling(20, min_periods=15).mean()              # 含 d
DT20P = DT20.shift(1)                                       # t−20..t−1
panel_days = IDX[(IDX >= DTCAL[0]) & (IDX <= DTCAL[-1])]
miss = [str(d.date()) for d in panel_days if d not in DTCAL]
say('dt bad values set NaN', nbad, 'panel days missing in dt calendar', len(miss), miss[:5])
def on_panel(M):  # dt 日曆 → 面板日曆（面板日不在 dt 日曆者為 NaN）
    return M.reindex(IDX).to_numpy()
DT20n, DT20Pn, DTn = on_panel(DT20), on_panel(DT20P), on_panel(DTW)

# ---------------- 股池 ----------------
V = pd.read_parquet(L.P / 'value.parquet')
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean().to_numpy()
CN = L.CN; ADJR = L.ADJ_RAW; OPR = L.OPEN_RAW
BASE = (V60 >= 1e7) & (CN >= 10) & np.isfinite(ADJR) & ~DR[None, :]

# 注意股
att = pd.read_csv(L.X / 'attention_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
ATT0 = att.date.min()
ATT = np.zeros((len(IDX), N), bool)
pos_of = {d: k for k, d in enumerate(IDX)}
for r in att.itertuples():
    if r.code in L.CI and r.date in pos_of: ATT[pos_of[r.date], L.CI[r.code]] = True
ATT20 = pd.DataFrame(ATT).rolling(20, min_periods=1).max().to_numpy().astype(bool)

# ---------------- 通用：日頻 NAV（容許空組合 = 現金） ----------------
def sim(events, buy, sell, t0, t1):
    u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            pk = events[t]; Pz = AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0); Vt = h.sum() + cash
            tgt = np.zeros(N)
            if len(pk) == 0:
                cost = sell * h.sum(); u = np.zeros(N); cash = Vt - cost; tov.append(1.0 if h.sum() > 0 else 0.0)
            else:
                w = 1.0 / len(pk); Vn = Vt
                for _ in range(6):
                    tgt[:] = 0; tgt[pk] = w * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                    Vn = Vt - cost
                tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
                u = np.zeros(N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * AC[t]) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')

def stats(nav, a, b, wdates, bench=None, ew=None):
    """日頻年化／波動／最大回撤／逐年；月窗（wdates 收盤 → 下一個 wdates 收盤，依起點歸屬）t。"""
    r = nav.pct_change()
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
    if m.sum() < 20: return None
    first = np.argmax(m); last = len(m) - 1 - np.argmax(m[::-1])
    span = (r.index[last] - r.index[max(first - 1, 0)]).days / 365.25
    rr = r[m]; eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'days': int(len(rr)), 'years': round(span, 2),
         'yearly_pct': {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}}
    def win(x):
        xd = x.reindex(wdates); p = (xd.shift(-1) / xd - 1).iloc[:-1]
        return p[(p.index >= pd.Timestamp(a)) & (p.index <= pd.Timestamp(b))]
    pr = win(nav); o['n_months'] = int(pr.notna().sum())
    for nm, bn in (('0050', bench), ('EW', ew)):
        if bn is None: continue
        ex = pr - win(bn)
        o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm}_m'] = tstat(ex)
        br = bn.pct_change().reindex(rr.index).fillna(0); eb = (1 + br).cumprod()
        o[f'{nm}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4)
    return o

def diff_t(n1, n2, a, b, wdates):
    d1 = n1.reindex(wdates); d2 = n2.reindex(wdates)
    p = (d1.shift(-1) / d1 - d2.shift(-1) / d2).iloc[:-1]
    p = p[(p.index >= pd.Timestamp(a)) & (p.index <= pd.Timestamp(b))]
    return {'mean_diff_pct_m': round(float(p.mean() * 100), 2), 't': tstat(p), 'n_months': int(p.notna().sum()), 'hit': round(float((p > 0).mean()), 3)}

# ================= A：月底五分位 =================
s = pd.Series(IDX, index=IDX)
me = [g.iloc[-1] for _, g in s.groupby([IDX.year, IDX.month])]
SIG = []
for d in me:
    i = pos_of[d]
    if d < pd.Timestamp('2020-12-01') or d > DTCAL[-1] or i + 1 >= len(IDX): continue
    if IDX[i + 1].month == d.month: continue                       # 月份未結束（面板最後一月）
    if (IDX[i + 1] - d).days > GAPDAYS: say('A: drop signal (gap)', d.date(), '->', IDX[i + 1].date()); continue
    SIG.append(d)
SIG = pd.DatetimeIndex(SIG)
say('A signals', SIG[0].date(), SIG[-1].date(), len(SIG))
GROUPS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'G50', 'NOT50', 'POOL']
EV = {g: {} for g in GROUPS}; held = {g: set() for g in GROUPS}
diag = {'pool': [], 'G50_n': [], 'blocked': {g: 0 for g in GROUPS}, 'G50_att20_share': [], 'Q_mean_dt20': {g: [] for g in GROUPS}, 'Q_med_v60_mn': {g: [] for g in GROUPS}}
for d in SIG:
    i = pos_of[d]; ex = i + 1
    pool = BASE[i] & np.isfinite(DT20n[i]) & np.isfinite(OPR[ex]) & np.isfinite(ADJR[ex])
    if pool.sum() < 100: say('A skip small pool', d.date(), pool.sum()); continue
    j = np.where(pool)[0]; x = DT20n[i, j]
    pr = pd.Series(x).rank(pct=True, method='first').to_numpy()
    q = np.minimum((pr * 5 - 1e-9).astype(int), 4)
    sets = {f'Q{k+1}': j[q == k] for k in range(5)}
    sets['G50'] = j[x > 50]; sets['NOT50'] = j[x <= 50]; sets['POOL'] = j
    diag['pool'].append(len(j)); diag['G50_n'].append(len(sets['G50']))
    if len(sets['G50']): diag['G50_att20_share'].append(float(ATT20[i, sets['G50']].mean()) if d >= ATT0 + pd.Timedelta(days=30) else np.nan)
    for g, jj in sets.items():
        diag['Q_mean_dt20'][g].append(float(np.mean(DT20n[i, jj])) if len(jj) else np.nan)
        diag['Q_med_v60_mn'][g].append(float(np.median(V60[i, jj]) / 1e6) if len(jj) else np.nan)
        if g == 'POOL':
            EV[g][ex] = jj; continue
        blk = GAP_UP[ex, jj] & ~np.isin(jj, list(held[g]))
        diag['blocked'][g] += int(blk.sum())
        pk = jj[~blk]; EV[g][ex] = pk; held[g] = set(pk.tolist())
t0 = pos_of[SIG[0]]; t1 = len(IDX) - 1
navA = {}
ew, _ = sim(EV['POOL'], 0.0, 0.0, t0, t1); navA['EW_pool'] = ew
E50 = L.E50_NAV.iloc[t0:t1 + 1]
resA = {'diag': {'pool_mean': round(float(np.mean(diag['pool'])), 1), 'G50_names_mean': round(float(np.mean(diag['G50_n'])), 1),
                 'G50_names_min_max': [int(min(diag['G50_n'])), int(max(diag['G50_n']))], 'months_G50_empty': int(sum(1 for z in diag['G50_n'] if z == 0)),
                 'blocked_gap_up_names': diag['blocked'], 'G50_share_on_attention_list_prior20d': round(float(np.nanmean(diag['G50_att20_share'])), 3),
                 'mean_dt20_by_group': {g: round(float(np.nanmean(v)), 1) for g, v in diag['Q_mean_dt20'].items()},
                 'median_v60_mn_by_group': {g: round(float(np.nanmean(v)), 1) for g, v in diag['Q_med_v60_mn'].items()},
                 'signals': len(SIG), 'first_signal': str(SIG[0].date()), 'last_signal': str(SIG[-1].date())}}
say('A diag', resA['diag'])
for g in GROUPS[:-1]:
    for cname, slip in (('gross', None), ('net', 0.001), ('stress', 'stress')):
        b, sl = (0.0, 0.0) if slip is None else L.cost_rates(slip)
        nav, tov = sim(EV[g], b, sl, t0, t1); navA[f'{g}_{cname}'] = nav
        resA[f'{g}_{cname}'] = {'turnover': round(tov, 3), **{k: stats(nav, a, bb, SIG, E50, ew) for k, (a, bb) in PERIODS.items()}}
    w = resA[f'{g}_net']
    say('A', g, 'net', {k: (v['CAGR'], v['maxDD'], v.get('t_vs_0050_m'), v.get('t_vs_EW_m')) for k, v in w.items() if isinstance(v, dict) and 'CAGR' in v}, 'tov', w['turnover'])
resA['EW_pool'] = {k: stats(ew, a, bb, SIG, E50) for k, (a, bb) in PERIODS.items()}
resA['0050'] = {k: stats(E50, a, bb, SIG) for k, (a, bb) in PERIODS.items()}
spreads = {}
for nm, (x1, x2) in {'Q1_minus_Q5_net': ('Q1_net', 'Q5_net'), 'Q1_minus_Q5_gross': ('Q1_gross', 'Q5_gross'),
                     'G50_minus_NOT50_net': ('G50_net', 'NOT50_net'), 'G50_minus_NOT50_gross': ('G50_gross', 'NOT50_gross'),
                     'NOT50_minus_EW_net': ('NOT50_net', 'EW_pool')}.items():
    spreads[nm] = {k: diff_t(navA[x1], navA[x2], a, bb, SIG) for k, (a, bb) in PERIODS.items()}
    say('A spread', nm, {k: (v['mean_diff_pct_m'], v['t']) for k, v in spreads[nm].items()})
resA['spreads'] = spreads
RES['A_screen'] = resA

# ================= B：當沖比暴增事件 =================
T = len(IDX)
dcal = np.array([(IDX[k + 1] - IDX[k]).days for k in range(T - 1)] + [1])
trig = (DT20Pn < 20) & (DTn > 60) & BASE
rows = []; nblk = 0; ngap = 0
for t in np.where(trig.any(1))[0]:
    if IDX[t] < pd.Timestamp('2021-01-01') or t + 20 >= T: continue
    jj = np.where(trig[t])[0]
    e = t + 1
    poolt = BASE[t] & np.isfinite(OPR[e])
    for h in (5, 20):
        x = t + h
        if dcal[t:x].max() > 20:   # 只剔除跨 2024-01～02 面板缺口（農曆年 ≤ 12 日不剔除）
            ngap += len(jj) if h == 20 else 0; continue
        pj = np.where(poolt)[0]
        ewr = np.nanmean(AO[x, pj] / AO[e, pj] - 1)
        r50 = L.E50.iloc[x - 1] / L.E50.iloc[t] - 1
        for j in jj:
            if not np.isfinite(OPR[e, j]): continue
            if GAP_UP[e, j]:
                if h == 5: nblk += 1
                continue
            rows.append(dict(date=IDX[t], entry=IDX[e], code=COLS[j], h=h, raw=AO[x, j] / AO[e, j] - 1, ew=ewr, r50=r50,
                             dt=DTn[t, j], dt20p=DT20Pn[t, j], att=bool(ATT[t, j]), att_known=IDX[t] >= ATT0))
EVB = pd.DataFrame(rows)
say('B events', len(EVB), 'blocked gap-up', nblk, 'dropped gap-window(h20)', ngap)
resB = {'n_trigger_stock_days': int(trig[(IDX >= pd.Timestamp('2021-01-01'))].sum()), 'blocked_gap_up_open': nblk}
def ev_stats(df, h):
    d = df[df.h == h]
    out = {}
    for k, (a, b) in PERIODS.items():
        x = d[(d.entry >= pd.Timestamp(a)) & (d.entry <= pd.Timestamp(b))]
        if len(x) < 5: out[k] = {'n': int(len(x))}; continue
        o = {'n': int(len(x)), 'n_months': int(x.entry.dt.to_period('M').nunique())}
        for cn, slip in (('gross', None), ('net', 0.001), ('stress', 'stress')):
            bb, ss = (0.0, 0.0) if slip is None else L.cost_rates(slip)
            net = (1 + x.raw) * (1 - ss) / (1 + bb) - 1
            ab = net - x.ew; a50 = net - x.r50
            gm = x.entry.dt.to_period('M')
            o[cn] = {'mean_pct': round(float(net.mean() * 100), 2), 'median_pct': round(float(net.median() * 100), 2), 'win': round(float((net > 0).mean()), 3),
                     't_month': tstat(net.groupby(gm).mean()),
                     'abn_vs_EW_mean_pct': round(float(ab.mean() * 100), 2), 'abn_vs_EW_t_month': tstat(ab.groupby(gm).mean()),
                     'abn_vs_EW_win': round(float((ab > 0).mean()), 3),
                     'abn_vs_0050_mean_pct': round(float(a50.mean() * 100), 2), 'abn_vs_0050_t_month': tstat(a50.groupby(gm).mean())}
        o['EW_window_mean_pct'] = round(float(x.ew.mean() * 100), 2)
        out[k] = o
    return out
for ver, sub in {'B_all': EVB, 'B_ex_attn': EVB[EVB.att_known & ~EVB.att], 'B_attn': EVB[EVB.att_known & EVB.att]}.items():
    resB[ver] = {f't+{h}': ev_stats(sub, h) for h in (5, 20)}
    for h in (5, 20):
        say('B', ver, h, {k: (v['n'], v.get('net', {}).get('mean_pct'), v.get('net', {}).get('abn_vs_EW_mean_pct'), v.get('net', {}).get('abn_vs_EW_t_month')) for k, v in resB[ver][f't+{h}'].items()})
yy = EVB[EVB.h == 20].copy(); yy['ab'] = (1 + yy.raw) * (1 - L.cost_rates(0.001)[1]) / (1 + L.cost_rates(0.001)[0]) - 1 - yy.ew
resB['B_all_t+20_yearly'] = {int(y): {'n': int(len(g)), 'abn_net_mean_pct': round(float(g.ab.mean() * 100), 2)} for y, g in yy.groupby(yy.entry.dt.year)}
resB['att_share_among_events_known'] = round(float(EVB[(EVB.h == 5) & EVB.att_known].att.mean()), 3)
RES['B_event'] = resB
if GAPDAYS == 7: EVB.to_parquet(HERE / 'events_B.parquet')

# ================= C：主策略剔除 dt20 > 50 =================
DATES = L.DATES; dates = list(DATES)
DT20_at = DT20.reindex(DT20.index.union(DATES)).ffill().reindex(DATES).to_numpy()
el = L.t300_d & L.yoy3.notna() & L.s52_d.notna() & ~L.inD_d
sc = 0.5 * L.s52_d.where(el).rank(axis=1, pct=True) + 0.5 * L.yoy3.where(el).rank(axis=1, pct=True)
def selC(mode):
    events, pools = {}, {}; held = set(); nexcl = []
    for k, d in enumerate(dates[:-1]):
        i = L.pos(d); ex = i + 1
        okb = el.loc[d].to_numpy(); trad = np.isfinite(OPR[ex]) & np.isfinite(ADJR[ex]); blocked = GAP_UP[ex]
        ok = okb & trad
        if ok.sum() < 50: continue
        hi = np.nan_to_num(DT20_at[k], nan=0) > 50
        ok2 = ok & ~hi if mode == 'refill' else ok
        order = sc.loc[d].to_numpy().copy(); order[~ok2] = np.nan
        cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok2.sum())]
        pick = []
        for j in cand:
            if blocked[j] and j not in held: continue
            pick.append(j)
            if len(pick) == 30: break
        if mode == 'norefill':
            base_pick = pick; pick = [j for j in pick if not hi[j]]; nexcl.append(len(base_pick) - len(pick))
        elif mode == 'base':
            nexcl.append(int(sum(hi[j] for j in pick)))
        events[ex] = ('open', np.array(pick, int)); pools[ex] = ('open', np.where(ok)[0])
        held = set(pick)
    return events, pools, nexcl
t0c = L.pos(dates[0]); t1c = L.pos(dates[-1])
WC = {'select_2021-03_2023': ('2021-03-01', '2023-12-31'), 'valid_2024_2026': VAL, 'valid_2024_2025': V2425, 'src3': SRC3, 'full_2021-03_2026-10': ('2021-03-01', '2026-10-31')}
resC = {}; navC = {}
for mode in ('base', 'refill', 'norefill'):
    ev, pools, nex = selC(mode)
    if mode == 'base':
        ewc, _ = L.sim_portfolio(pools, 0.0, 0.0, t0c, t1c)
        resC['base_pick_with_dt20_gt50_per_rebalance'] = {'mean': round(float(np.mean(nex)), 2), 'max': int(max(nex)), 'share_months_any': round(float(np.mean([z > 0 for z in nex])), 3)}
    if mode == 'norefill': resC['norefill_names_removed_mean'] = round(float(np.mean(nex)), 2)
    for cname, slip in (('net', 0.001), ('stress', 'stress')):
        b, sl = L.cost_rates(slip)
        nav, tov = L.sim_portfolio(ev, b, sl, t0c, t1c); navC[f'{mode}_{cname}'] = nav
        resC[f'{mode}_{cname}'] = {'turnover': round(tov, 3), **{k: stats(nav, a, bb, DATES, L.E50_NAV, ewc) for k, (a, bb) in WC.items()}}
    say('C', mode, {k: (v['CAGR'], v['maxDD'], v['t_vs_0050_m'], v['t_vs_EW_m']) for k, v in resC[f'{mode}_net'].items() if isinstance(v, dict) and 'CAGR' in v})
for mode in ('refill', 'norefill'):
    for cname in ('net', 'stress'):
        resC[f'{mode}_minus_base_{cname}'] = {k: diff_t(navC[f'{mode}_{cname}'], navC[f'base_{cname}'], a, bb, DATES) for k, (a, bb) in WC.items()}
        say('C diff', mode, cname, {k: (v['mean_diff_pct_m'], v['t']) for k, v in resC[f'{mode}_minus_base_{cname}'].items()})
RES['C_filter'] = resC

# ================= 判定 =================
def passes(o): return o is not None and o.get('t_vs_EW_m', -9) >= 2 and o['CAGR'] >= o['0050_CAGR']
q1 = resA['Q1_net']; ps, pv = passes(q1['select_2021_2023']), passes(q1['valid_2024_2026'])
verdict = '可用' if ps and pv else ('待觀察' if ps or pv else '不可用')
g = spreads['G50_minus_NOT50_net']
b20 = resB['B_all']['t+20']
def bt(k): return b20[k].get('net', {}).get('abn_vs_EW_t_month', float('nan'))
cR = resC['refill_minus_base_net']
RES['VERDICT'] = {'overall_A_Q1': verdict, 'A_select_pass': ps, 'A_valid_pass': pv,
                  'A_G50_avoid_evidence': bool(g['select_2021_2023']['t'] <= -2 and g['valid_2024_2026']['t'] <= -2),
                  'B_long_usable': bool(bt('select_2021_2023') >= 2 and bt('valid_2024_2026') >= 2),
                  'B_avoid_evidence': bool(bt('select_2021_2023') <= -2 and bt('valid_2024_2026') <= -2),
                  'C_filter_usable': bool(cR['select_2021-03_2023']['t'] >= 2 and cR['valid_2024_2026']['t'] >= 2),
                  'C_harmless': bool(cR['select_2021-03_2023']['mean_diff_pct_m'] >= 0 and cR['valid_2024_2026']['mean_diff_pct_m'] >= 0)}
say('VERDICT', RES['VERDICT'])
(HERE / OUTF).write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame({**navA, **{f'C_{k}': v for k, v in navC.items()}}).to_parquet(HERE / ('navs.parquet' if GAPDAYS == 7 else f'navs_gap{GAPDAYS}.parquet'))

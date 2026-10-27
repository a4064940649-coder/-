#!/usr/bin/env python3
"""二八輪動（0050 vs 中小籃 N 日動能、雙負空手）回測。先核對 PREREG 雜湊；結果寫入 results.json['results']。"""
import json, hashlib, math, sys
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent; AG = H.parent
RJ = json.loads((H / 'results.json').read_text())
assert hashlib.sha256(json.dumps(RJ['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RJ['PREREG_sha256'], 'PREREG changed'
sys.path.insert(0, str(AG / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
tstat = L.tstat

# ---------------- 日曆 ----------------
IDXF = L.IDX
cnt = L.adj.notna().sum(axis=1)
SP = set(M.SPARSE) | set(cnt.index[cnt < 300])
CALP = np.array([i for i, d in enumerate(IDXF) if d not in SP and d >= pd.Timestamp('2004-01-01')])
CAL = IDXF[CALP]; T = len(CAL); CI = {d: k for k, d in enumerate(CAL)}
print('sparse removed:', sorted(str(d.date()) for d in SP))
N = L.N; COLS = L.COLS
AO = L.AO[CALP]; AC = L.AC[CALP]; GAP = M.GAP_UP[CALP]
OPEN_OK = np.isfinite(L.OPEN_RAW[CALP]) & np.isfinite(L.ADJ_RAW[CALP])
V60 = M.V60.reindex(CAL).to_numpy(); CC = L.C.reindex(CAL).to_numpy()
NOT_DR = np.array([not c.startswith('91') for c in COLS])

# ---------------- 0050 收盤、開盤近似 ----------------
RAW = Path('/home/user/jonatw/finlab-txd/data/raw')
e50 = pd.read_csv(RAW / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
e50.loc[pd.Timestamp('2026-09-20')] = np.nan
tx = pd.read_csv(RAW / 'taiex_twii.csv', parse_dates=['date']).set_index('date').sort_index()
U = e50.index.union(tx.index).union(CAL)
E50U = e50.reindex(U).ffill()
E50C = E50U.reindex(CAL).to_numpy().copy()
E50T = E50U.reindex(tx.index)
O50_tx = E50T.shift(1) * tx['open'] / tx['close'].shift(1)
O50 = O50_tx.reindex(CAL).to_numpy().copy()
fb = ~np.isfinite(O50); O50[fb] = np.concatenate([[np.nan], E50C[:-1]])[fb]
lim = np.where(CAL < pd.Timestamp('2015-06-01'), 0.07, 0.10) - L.LTOL
G50 = (O50 / np.concatenate([[np.nan], E50C[:-1]]) - 1) >= lim
print('0050 open proxy fallback days:', int(fb.sum()), ' 0050 gap-up-limit days:', int(np.nansum(G50)))

AOx = np.hstack([AO, O50[:, None]]); ACx = np.hstack([AC, E50C[:, None]]); GAPx = np.hstack([GAP, G50[:, None]])
OKx = np.hstack([OPEN_OK, np.isfinite(O50)[:, None]])
J50 = N

# ---------------- 每月重組名單 ----------------
s = pd.Series(np.arange(T), index=CAL)
MEND = s.groupby([CAL.year, CAL.month]).last().to_numpy()  # 月底位置
REC_S, REC_P, REC_300 = {}, {}, {}
for k in MEND:
    if k + 1 >= T: continue
    v = V60[k]; base = (v >= 1e7) & (CC[k] >= 10) & NOT_DR & OPEN_OK[k + 1]
    if base.sum() < 300: continue
    vv = np.where(base, v, -np.inf); order = np.argsort(-vv); nb = int(base.sum())
    REC_S[k + 1] = (order[50:min(300, nb)], None)
    top = order[:50]; REC_P[k + 1] = (top, v[top] / v[top].sum())
    REC_300[k + 1] = (order[:min(300, nb)], None)
print('first recon exec:', CAL[min(REC_S)].date(), 'n recon:', len(REC_S), 'mean basket size:', np.mean([len(x[0]) for x in REC_S.values()]))

# ---------------- 成本 ----------------
FEE = L.FEE
COSTS = {
    'exlib_main': (L.cost_rates(0.001), (FEE + 0.001, FEE + 0.001 + 0.001)),
    'rule_stated': ((0.0025, 0.0025), (0.0005, 0.0005)),
    'stress': (L.cost_rates('stress'), (0.0025, 0.0025)),
    'gross': ((0.0, 0.0), (0.0, 0.0)),
}
def rate_vecs(cost):
    (sb, ss), (lb, ls) = COSTS[cost]
    b = np.full(N + 1, sb); s_ = np.full(N + 1, ss); b[J50] = lb; s_[J50] = ls
    return b, s_

# ---------------- 模擬器 ----------------
def sim(states, t0, t1, cost='gross', rec=REC_S, cash_rate=0.0, proxy_until=None):
    """states[t] ∈ {'L','S','C','B'}：t 開盤要持有的目標。'L'＝0050（proxy_until 前改用 REC_P 加權組合）、'S'＝中小籃、'B'＝各半。
    回傳收盤 NAV（含 t0 前一日 = 1）、每日實際狀態、交易次數。"""
    b, s_ = rate_vecs(cost)
    u = np.zeros(N + 1); cash = 1.0; cur = 'C'; mem = None; pmem = None; pending = False
    nav = np.empty(t1 - t0 + 2); nav[0] = 1.0; st = []; ntr = 0; nblk = 0
    for t in range(t0, t1 + 1):
        if t in rec: mem = rec[t][0]
        if t in REC_P: pmem = REC_P[t]
        tgt_state = states[t]
        use_proxy = proxy_until is not None and CAL[t] < proxy_until
        recon = (t in rec and tgt_state in ('S', 'B')) or (use_proxy and t in REC_P and tgt_state == 'L')
        if (tgt_state != cur or recon or pending) and mem is not None:
            P = np.nan_to_num(AOx[t]); h = u * P; Vt = h.sum() + cash
            w = np.zeros(N + 1); pending = False
            def put_S(wt):
                nonlocal nblk
                can = OKx[t, mem] & ~((h[mem] <= 0) & GAPx[t, mem])
                nblk += int((~can).sum())
                m = mem[can]
                if len(m): w[m] += wt / len(m)
            def put_L(wt):
                nonlocal pending, nblk
                if use_proxy:
                    m, ww = pmem
                    can = OKx[t, m] & ~((h[m] <= 0) & GAPx[t, m]); nblk += int((~can).sum())
                    if can.any(): w[m[can]] += wt * ww[can] / ww[can].sum()
                elif h[J50] <= 0 and (GAPx[t, J50] or not OKx[t, J50]):
                    pending = True           # 0050 開盤漲停買不到 → 順延
                else:
                    w[J50] += wt
            if tgt_state == 'S': put_S(1.0)
            elif tgt_state == 'L': put_L(1.0)
            elif tgt_state == 'B': put_S(0.5); put_L(0.5)
            Vn = Vt
            for _ in range(8):
                tg = w * Vn
                c = (b * np.clip(tg - h, 0, None)).sum() + (s_ * np.clip(h - tg, 0, None)).sum()
                Vn = Vt - c
            tg = w * Vn
            with np.errstate(invalid='ignore', divide='ignore'):
                u = np.where(w > 0, tg / np.where(P > 0, P, np.nan), 0.0)
            u = np.nan_to_num(u); cash = Vn - tg.sum()
            if tgt_state != cur: ntr += 1
            cur = 'C' if (pending and tgt_state == 'L') else tgt_state
        cash *= (1 + cash_rate) ** ((CAL[t] - CAL[t - 1]).days / 365.25)
        nav[t - t0 + 1] = np.nansum(u * np.nan_to_num(ACx[t])) + cash
        st.append(cur)
    idx = CAL[t0 - 1:t1 + 1]
    return pd.Series(nav, index=idx), pd.Series(st, index=CAL[t0:t1 + 1]), ntr, nblk

# ---------------- 訊號 ----------------
T0 = CI[CAL[CAL >= pd.Timestamp('2005-01-01')][0]]; T1 = T - 1
SI_START = min(REC_S)
SI, _, _, _ = sim(['S'] * T, SI_START, T1, 'gross')
SIC = SI.reindex(CAL).to_numpy()                      # 中小籃指數（gross，收盤）
PXI, _, _, _ = sim(['L'] * T, min(REC_P), T1, 'gross', proxy_until=pd.Timestamp('2100-01-01'))
PXC = PXI.reindex(CAL).to_numpy()
# LP：2009 前 proxy、之後接 0050（在 2009-01 第一個交易日收盤鏈接）
kj = CI[CAL[CAL >= pd.Timestamp('2009-01-01')][0]]
LPC = np.where(np.arange(T) < kj, PXC / PXC[kj] * E50C[kj], E50C)

wk = pd.Series(np.arange(T), index=CAL).groupby(CAL.to_period('W-SUN')).last().to_numpy()
WEEKLY = np.zeros(T, bool); WEEKLY[wk] = True

def make_states(n, freq='W', filt=True, XL=E50C, XS=SIC):
    states = ['C'] * T; cur = 'C'
    for k in range(T - 1):
        if (freq == 'D' or WEEKLY[k]) and k - n >= 0:
            rl = XL[k] / XL[k - n] - 1; rs = XS[k] / XS[k - n] - 1
            if np.isfinite(rl) and np.isfinite(rs):
                if filt and rl <= 0 and rs <= 0: cur = 'C'
                else: cur = 'L' if rl >= rs else 'S'
        states[k + 1] = cur
    return states

# ---------------- 統計 ----------------
MONTH_END_DAYS = CAL[MEND]
def mret(nav):
    me = nav.reindex(MONTH_END_DAYS.intersection(nav.index))
    if nav.index[0] not in me.index: me = pd.concat([nav.iloc[:1], me])
    return (me / me.shift(1) - 1).dropna()

SRC3 = (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23'))
def wstats(nav, a, b, refs, ex_src3=False):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    r = nav.pct_change().dropna(); m = (r.index >= a) & (r.index <= b)
    if ex_src3: m &= ~((r.index >= SRC3[0]) & (r.index <= SRC3[1]))
    rr = r[m]
    if len(rr) < 20: return None
    if ex_src3:
        yrs = len(rr) / 245.0
    else:
        i0 = nav.index.get_loc(rr.index[0]); yrs = (rr.index[-1] - nav.index[i0 - 1]).days / 365.25
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(yrs, 2)}
    mr = mret(nav); mm = (mr.index >= a) & (mr.index <= b)
    if ex_src3: mm &= ~((mr.index >= SRC3[0]) & (mr.index <= pd.Timestamp('2025-09-30')))
    mr = mr[mm]; o['n_months'] = int(len(mr))
    for nm, rn in refs.items():
        br = mret(rn).reindex(mr.index); ex = mr - br
        o[f'exc_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm}'] = tstat(ex)
        rb = rn.pct_change().reindex(rr.index).fillna(0); eb = (1 + rb).cumprod()
        o[f'{nm}_CAGR'] = round(float(eb.iloc[-1] ** (1 / yrs) - 1), 4)
        o[f'{nm}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o

def yearly(nav):
    r = nav.pct_change().dropna()
    return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

W = RJ['PREREG']['windows']
WIN = {k: W[k] for k in ('select_price', 'valid_price', 'y2024_2025', 'select_2021', 'valid_2024', 'full')}

def share(st, a, b):
    x = st[(st.index >= pd.Timestamp(a)) & (st.index <= pd.Timestamp(b))]
    return {k: round(float((x == k).mean()), 3) for k in ('L', 'S', 'C')}

# ---------------- 基準 ----------------
BM = {}
for cost in COSTS:
    BM[('0050', cost)] = sim(['L'] * T, T0, T1, cost)[0]
    BM[('EW_5050', cost)] = sim(['B'] * T, T0, T1, cost)[0]
    BM[('S_basket', cost)] = sim(['S'] * T, T0, T1, cost)[0]
EW300 = sim(['S'] * T, T0, T1, 'gross', rec=REC_300)[0]

def refs_for(cost):
    return {'0050': BM[('0050', 'gross')], 'EW_5050': BM[('EW_5050', cost)], 'S_basket': BM[('S_basket', cost)], 'EW300_ref': EW300}

out = {'meta': {}, 'benchmarks': {}, 'grid': {}, 'extra': {}}
out['meta'] = {'calendar_start': str(CAL[T0].date()), 'calendar_end': str(CAL[T1].date()), 'sparse_removed': sorted(str(d.date()) for d in SP),
               'n_recon': len(REC_S), 'basket_size_mean': round(float(np.mean([len(x[0]) for x in REC_S.values()])), 1),
               'o50_fallback_days': int(fb.sum()), 'o50_gapup_days': int(np.nansum(G50)),
               'note_0050_ref': '對 0050 的 t 一律對 0050 含息買進持有（gross 指數）；0050_CAGR 也是 gross 指數。'}

for nm in ('0050', 'EW_5050', 'S_basket'):
    for cost in ('exlib_main', 'gross'):
        nav = BM[(nm, cost)]
        out['benchmarks'][f'{nm}|{cost}'] = {'windows': {k: wstats(nav, *v, {'0050': BM[('0050', 'gross')]}) for k, v in WIN.items()}, 'yearly': yearly(nav)}
out['benchmarks']['EW300_ref|gross'] = {'windows': {k: wstats(EW300, *v, {'0050': BM[('0050', 'gross')]}) for k, v in WIN.items()}, 'yearly': yearly(EW300)}

navs = {}
for n in (10, 20, 60):
    for freq in ('W', 'D'):
        for filt in (True, False):
            key = f'N{n}_{freq}_{"filt" if filt else "nofilt"}'
            states = make_states(n, freq, filt)
            res = {}
            for cost in COSTS:
                nav, st, ntr, nb = sim(states, T0, T1, cost)
                navs[(key, cost)] = nav
                res[cost] = {'windows': {k: wstats(nav, *v, refs_for(cost)) for k, v in WIN.items()},
                             'valid_ex_src3': wstats(nav, *W['valid_price'], refs_for(cost), ex_src3=True),
                             'yearly': yearly(nav)}
                if cost == 'exlib_main':
                    res['n_switches'] = ntr; res['n_blocked_buys'] = nb
                    res['time_share'] = {k: share(st, *v) for k, v in WIN.items() if k in ('select_price', 'valid_price', 'full')}
                    res['switches_per_year'] = round(ntr / ((CAL[T1] - CAL[T0]).days / 365.25), 1)
            out['grid'][key] = res
            m = res['exlib_main']['windows']
            print(key, 'sel', m['select_price']['CAGR'], m['select_price']['t_vs_EW_5050'], 'val', m['valid_price']['CAGR'], m['valid_price']['t_vs_EW_5050'], 'sw/yr', res['switches_per_year'])

# 敏感度：LP（2009 前 proxy）、現金 1%
stLP = make_states(20, 'W', True, XL=LPC)
navLP = sim(stLP, T0, T1, 'exlib_main', proxy_until=pd.Timestamp('2009-01-01'))[0]
out['extra']['LP_proxy_pre2009_N20_W_filt'] = {'windows': {k: wstats(navLP, *v, refs_for('exlib_main')) for k, v in WIN.items()}, 'yearly': yearly(navLP)}
stM = make_states(20, 'W', True)
navC1 = sim(stM, T0, T1, 'exlib_main', cash_rate=0.01)[0]
out['extra']['cash1pct_N20_W_filt'] = {'windows': {k: wstats(navC1, *v, refs_for('exlib_main')) for k, v in WIN.items()}}
# proxy vs 0050
rp = pd.Series(PXC, index=CAL).pct_change(); re = pd.Series(E50C, index=CAL).pct_change()
pr = {}
for nm, (a, b) in {'2005_2008': ('2005-01-01', '2008-12-31'), 'full': ('2005-01-01', '2026-10-31')}.items():
    mk = (CAL >= a) & (CAL <= b)
    x, y = rp[mk], re[mk]
    yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    pr[nm] = {'corr_daily': round(float(x.corr(y)), 3), 'proxy_CAGR': round(float((1 + x.fillna(0)).prod() ** (1 / yrs) - 1), 4), '0050_CAGR': round(float((1 + y.fillna(0)).prod() ** (1 / yrs) - 1), 4)}
out['extra']['proxy_vs_0050'] = pr
# 訊號狀態統計（主規格）：每段的條件報酬（下週持有標的 vs 另一標的）
out['extra']['main_state_share_by_year'] = {}
st_main = sim(stM, T0, T1, 'exlib_main')[1]
for y, g in st_main.groupby(st_main.index.year):
    out['extra']['main_state_share_by_year'][int(y)] = {k: round(float((g == k).mean()), 2) for k in ('L', 'S', 'C')}

RJ['results'] = out
(H / 'results.json').write_text(json.dumps(RJ, ensure_ascii=False, indent=1, default=float))
pd.DataFrame({k[0] + '|' + k[1]: v for k, v in navs.items() if k[1] == 'exlib_main'} | {f'BM_{k[0]}|{k[1]}': v for k, v in BM.items()} | {'EW300_ref': EW300, 'SI_gross': SI.reindex(CAL[T0 - 1:])}).to_parquet(H / 'navs.parquet')
print('done')

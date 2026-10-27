#!/usr/bin/env python3
"""複製員：A5（跳過上市 [2,4) 年、本期前未持有的初次上市股）對原版 v2／V3。
python3 -I -B r_oos.py v2|v3
引擎：main_v2/v2_core（正式版 v2）與 o_infra_v3_rebuild/o3_core（V3）只讀 import；A5 規則、分類、統計都是自己寫的。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
ENG = sys.argv[1]
if ENG == 'v3':
    sys.path.insert(0, str(AG / 'o_infra_v3_rebuild')); import o3_core as O; V, K = O.V, O.K
else:
    sys.path.insert(0, str(AG / 'main_v2')); import v2_core as V; K = V.K
sys.path.pop(0)
import numpy as np, pandas as pd

L = pd.read_csv(HERE / 'my_listings.csv', dtype={'code': str}, parse_dates=['first', 'last']).set_index('code')
FIRST = np.array([L.loc[c, 'first'] if c in L.index else pd.NaT for c in K.COLS], dtype='datetime64[ns]')
KIND = np.array([L.loc[c, 'kind'] if c in L.index else 'old' for c in K.COLS])

def age_years(d):
    return (np.datetime64(d) - FIRST).astype('timedelta64[D]').astype(float) / 365.25

def mk_skip(kind, lo=2.0, hi=4.0):
    def f(d):
        a = age_years(d)
        return (KIND == kind) & (a >= lo) & (a < hi)
    return f

def plan(S, n, skipf=None):
    """與 V2.plan 相同（official 同分），另加：skipf(d) 為 True 且本期前未持有者跳過、由下一名遞補。"""
    out = []; held = set(); nblk = 0; skipped = []
    for d in S.sig_days:
        cs = S.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        e = cs['e']; gu = K.gap_up_open(e)
        sk = skipf(d) if skipf is not None else np.zeros(K.N, bool)
        cand = S.order(cs, 'official')[:int(el.sum())]
        pick = []; sk_here = []
        for j in cand:
            if gu[j] and j not in held: nblk += 1; continue
            if sk[j] and j not in held: sk_here.append(j); continue
            pick.append(j)
            if len(pick) == n: break
        out.append((e, np.array(pick, int), np.where(el)[0], d))
        skipped.append(sk_here)
        held = set(pick)
    return out, skipped

if ENG == 'v3':
    O.apply_patch()
    E = O.load_events(); H = O.load_disposal_hist()
    RES3 = json.loads((AG / 'o_infra_v3_rebuild/results.json').read_text(encoding='utf-8'))
    EXTRA = {pd.Timestamp(k): v for k, v in RES3['PREREG_params']['D_hi_extra'].items()}
    O.set_disposal('all_plus', H, EXTRA)
    YL, Y5, _ = V.load_revenue(); YR = O.load_rdel()
    S = O.V3('gapfix', YR, Y5, exfin=True)
    XR = O.exit_rules(E, 'main')
    sim = lambda pl, b, s, t0, t1: O.simulate(pl, b, s, t0, t1, xrules=XR)
    REFNAV = pd.read_parquet(AG / 'o_infra_v3_rebuild/navs_P1.parquet')['P1|R1|X1|Dhi|EXFIN|n10|normal'].dropna()
else:
    YL, Y5, _ = V.load_revenue()
    S = V.V2('gapfix', YL, Y5, exfin=True)
    sim = lambda pl, b, s, t0, t1: K.simulate(pl, b, s, t0, t1)
    REFNAV = pd.read_parquet(AG / 'main_v2/navs_gapfix.parquet')['EXFIN|n10|normal'].dropna()
t0, t1, sd = S.bounds()
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
PER = {'SEL': ('2006-04-01', '2019-04-30', '2006-04-01'), 'VAL': ('2019-05-01', '2026-10-12', '2019-05-01'),
       'VAL1': ('2019-05-01', '2023-12-31', '2019-05-01'), 'HOLD': ('2024-01-01', '2026-10-12', '2024-01-01'),
       'ALL': ('2006-04-01', '2026-10-12', '2006-04-01')}

pl0, _ = plan(S, 10)
plA, skA = plan(S, 10, mk_skip('ipo'))
plP, skP = plan(S, 10, mk_skip('transfer'))
ew, _ = sim([(p[0], p[2]) for p in pl0], 0, 0, t0, t1)
navs = {}
for c, (b, s) in COST.items():
    navs[('orig', c)] = sim([(p[0], p[1]) for p in pl0], b, s, t0, t1)[0]
    navs[('A5', c)] = sim([(p[0], p[1]) for p in plA], b, s, t0, t1)[0]
    navs[('PLAC', c)] = sim([(p[0], p[1]) for p in plP], b, s, t0, t1)[0]
ref_dev = float((navs[('orig', 'normal')].reindex(REFNAV.index) / REFNAV - 1).abs().max())

def win_ret(nav):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index]); nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]
def sel(x, a, b): return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()
def pair(navA, navB, a, b):
    x = sel(win_ret(navA) - win_ret(navB), a, b)
    return {'diff_pp': round(float(x.mean() * 100), 3), 't': round(K.tstat(x), 2), 'n': int(len(x))}
def seg(nav, k):
    a, b, ad = PER[k]
    r = V.seg_stats(nav, ew, a, b, sd, ad=ad)
    return {x: round(float(r[x]), 2) for x in ('CAGR', 'maxDD', 't_0050', 'E50_CAGR', 'EW_CAGR')}
res = {'engine': ENG, 'orig_vs_reference_nav_maxabsdev': ref_dev}
for c in COST:
    for k in PER:
        res[f'{c}|{k}'] = {'orig': seg(navs[('orig', c)], k), 'A5': seg(navs[('A5', c)], k),
                           'pair_A5': pair(navs[('A5', c)], navs[('orig', c)], *PER[k][:2]),
                           'pair_placebo': pair(navs[('PLAC', c)], navs[('orig', c)], *PER[k][:2]),
                           'PLAC_CAGR': seg(navs[('PLAC', c)], k)['CAGR']}
# 受影響名額
def affected(plx, sk, a, b):
    n_sk = 0; n_diff = 0
    for (p, q, s) in zip(pl0, plx, sk):
        if not (pd.Timestamp(a) <= p[3] <= pd.Timestamp(b)): continue
        n_sk += len(s); n_diff += len(set(p[1]) - set(q[1]))
    return {'skipped_events': n_sk, 'slots_diff': n_diff}
for k in ('SEL', 'VAL', 'ALL'):
    res[f'affected|{k}'] = affected(plA, skA, *PER[k][:2])
    res[f'affected_placebo|{k}'] = affected(plP, skP, *PER[k][:2])
# 0050 t 拆解（訊號窗月超額）
E50 = K.E50
for k in ('ALL', 'VAL'):
    for nm in ('orig', 'A5'):
        x = sel(win_ret(navs[(nm, 'normal')]) - win_ret(E50.reindex(navs[(nm, 'normal')].index)), *PER[k][:2])
        res[f'decomp|{k}|{nm}'] = {'mean_pp': round(float(x.mean() * 100), 3), 'sd_pp': round(float(x.std(ddof=1) * 100), 2), 't': round(K.tstat(x), 2)}
# 單股對配對差的貢獻（毛報酬、期間報酬＝成交日開盤→下一成交日開盤）
ex0 = [p[0] for p in pl0]
def per_ret(t, t2, pk):
    r = K.Cn[t, pk] / K.On[t, pk]
    if t2 > t: r = r * np.prod(K.GROW[t + 1:t2 + 1, pk], axis=0)
    tr = K.TRADED[t2, pk] & np.isfinite(K.On[t2, pk])
    r = np.where(tr, r * K.On[t2, pk] / np.where(tr, K.Cn[t2, pk], 1.0), r)
    return r - 1
def contrib(plx, a, b):
    con = {}; dif = []; sk_r = []; rp_r = []
    for i in range(len(pl0) - 1):
        p, q = pl0[i], plx[i]; d = p[3]
        if not (pd.Timestamp(a) <= d <= pd.Timestamp(b)): continue
        t, t2 = p[0], ex0[i + 1]
        if t2 > t1: continue
        s0, s1 = set(p[1].tolist()), set(q[1].tolist())
        out_, in_ = sorted(s0 - s1), sorted(s1 - s0)
        r_out = per_ret(t, t2, np.array(out_, int)) if out_ else np.array([])
        r_in = per_ret(t, t2, np.array(in_, int)) if in_ else np.array([])
        for j, r in zip(in_, r_in): con[K.COLS[j]] = con.get(K.COLS[j], []) + [(d, r / 10)]
        for j, r in zip(out_, r_out): con[K.COLS[j]] = con.get(K.COLS[j], []) + [(d, -r / 10)]
        sk_r += list(r_out); rp_r += list(r_in)
        for j, r in zip(out_, r_out): AFF.append((str(d.date()), K.COLS[j], 'skipped', float(r), a))
        for j, r in zip(in_, r_in): AFF.append((str(d.date()), K.COLS[j], 'added', float(r), a))
        GROSS.setdefault(a, []).append((r_in.sum() - r_out.sum()) / 10)
    return con, sk_r, rp_r
AFF = []; GROSS = {}
for k in ('SEL', 'VAL'):
    con, sk_r, rp_r = contrib(plA, *PER[k][:2])
    tot = {c: sum(v for _, v in x) for c, x in con.items()}
    top3 = sorted(tot, key=lambda c: -tot[c])[:3]
    # 剔除 top3：從配對差（訊號窗、淨值版）扣掉其貢獻
    x = sel(win_ret(navs[('A5', 'normal')]) - win_ret(navs[('orig', 'normal')]), *PER[k][:2])
    adj = x.copy()
    for c in top3:
        for d, v in con[c]:
            if d in adj.index: adj[d] -= v
    res[f'top3|{k}'] = {'top3': top3, 'top3_pp': [round(tot[c] * 100, 2) for c in top3], 'diff_ex_top3_pp': round(float(adj.mean() * 100), 3), 't': round(K.tstat(adj), 2),
                        'skipped_next_ret_mean_pct': round(float(np.mean(sk_r) * 100), 2), 'repl_next_ret_mean_pct': round(float(np.mean(rp_r) * 100), 2),
                        'n_skipped': len(sk_r), 'n_repl': len(rp_r)}
for a_, g in GROSS.items():
    res[f'gross_pair_pp|{a_}'] = {'mean_pp': round(float(np.mean(g) * 100), 3), 't': round(K.tstat(np.array(g)), 2), 'n': len(g)}
pd.DataFrame(AFF, columns=['d', 'code', 'side', 'r', 'per']).to_csv(HERE / f'r_affected_{ENG}.csv', index=False)
print(json.dumps(res, ensure_ascii=False, indent=0, default=str))
(HERE / f'r_oos_{ENG}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame({f'{a}|{c}': v for (a, c), v in navs.items()}).to_parquet(HERE / f'r_navs_{ENG}.parquet')

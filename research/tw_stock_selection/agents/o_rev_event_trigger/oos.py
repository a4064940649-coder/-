#!/usr/bin/env python3
"""樣本外（一次跑完、不改規格）：PREREG 必須已存在，且全部腳本、引擎庫、資料檔的 sha256 與 PREREG 相同。
全期 2006-04～2026-10；驗證期 2019-01～2023-12、保留期 2024-01～2026-10 各算一次。"""
import sys, json, hashlib, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

RES_F = HERE / 'results.json'
res = json.loads(RES_F.read_text(encoding='utf-8'))
assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：先跑 prereg.py'
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
for f, h in res['PREREG']['sha256'].items():
    assert sha(f) == h, f'{f} 與 PREREG 登記的 sha256 不符'
assert 'RESULTS' not in res, '樣本外已跑過，不得重跑'
import fx_core as F
L, K, V = F.L, F.K, F.V
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
st = F.setup()
PR = res['PREREG']
SELV = PR['selected_version']
Q = json.loads((F.DATA / 'q_night.json').read_text(encoding='utf-8'))['q']
SEEDS = range(40)
KEYS = ('ALL', 'SEL', 'VAL', 'HOLD', 'POST', 'S1', 'S2', 'S3', 'S4')
NV4 = pd.read_parquet(F.V4DIR / 'navs_v4.parquet')
OFF_DAYS, OFF_END = F.official_days()
W_OFF = F.windows([d for _, d in OFF_DAYS], OFF_END)
CAND = {'J+1': (1, 'day'), 'J0': (0, 'day'), 'J0N': (0, 'night')}
SLIPS = [round(0.0010 + 0.0002 * i, 4) for i in range(16)]

def seed_stats(navs, W):
    ts = [F.block(n, W, ('ALL',))['ALL']['t_0050'] for n in navs]
    d1 = [F.endpoint(n, W)['drop_best1'] for n in navs]
    ts, d1 = np.array(ts), np.array(d1)
    return {'t_median': float(np.median(ts)), 't_min': float(ts.min()), 't_max': float(ts.max()), 't_p10': float(np.percentile(ts, 10)),
            'n_below_2.3': int((ts < 2.3).sum()), 'n_below_2.0': int((ts < 2.0).sum()),
            'drop_best1_median': float(np.median(d1)), 'drop_best1_min': float(d1.min()), 'drop_best1_n_below_2.3': int((d1 < 2.3).sum()), 'n': len(ts)}

OUT = {}; NAVS = {}; PLANS = {}; WIN = {}
# ---------------- 三個候選：全期 ----------------
for name, (j, mode) in CAND.items():
    days, end = F.version_days(j)
    W = F.windows([d for _, d in days], end); WIN[name] = W
    S, pl, t0, t1, sd = F.build(days, end, None)
    nn, lg = F.sim(pl, t0, t1, 'normal'); ew0 = F.ew_nav(pl, t0, t1)
    costs = ('normal', 'stress') + (tuple(('slip', s) for s in SLIPS) if name == SELV else ())
    R = F.ab_navs(days, end, mode=mode, seeds=SEEDS, q=Q, costs=costs)
    ab = R['mean']['normal']; NAVS[name] = {'nonAB': nn, 'AB': ab, 'AB_stress': R['mean']['stress'], 'EW': R['ew']}
    if name == SELV: NAVS[name]['slip'] = {s: R['mean'][('slip', s)] for s in SLIPS}
    PLANS[name] = R['plans'][0]
    o = {'n_windows': len(days), 'first_signal': str(days[0][1].date()), 'eval_end': str(end.date()), 'turnover_nonAB': lg['turnover'],
         'nonAB': F.block(nn, W, KEYS, ew0), 'AB': F.block(ab, W, KEYS, R['ew']), 'AB_stress': F.block(R['mean']['stress'], W, KEYS),
         'AB_endpoint': F.endpoint(ab, W), 'seeds': seed_stats(R['seeds']['normal'], W), 'mean_unavail_in_top300': R['nold'],
         'lookahead_nonAB_minus_AB': {k: F.paired_cal(nn, ab, k) for k in ('ALL', 'SEL', 'VAL', 'HOLD')},
         'yearly_AB': F.yearly(ab)}
    o['AB']['ALL']['p_one'] = F.p_one(o['AB']['ALL']['t_0050']); o['AB']['ALL']['bonferroni_m10'] = min(1, 10 * o['AB']['ALL']['p_one'])
    OUT[name] = o
    log(name, 'AB ALL', round(o['AB']['ALL']['CAGR'], 3), round(o['AB']['ALL']['t_0050'], 3))

# ---------------- 官方 AB_V4（o_v4 淨值）----------------
offn = NV4['AB_V4|normal']; offs = NV4['AB_V4|stress']; offe = NV4['AB_V4|EW']
OUT['OFF_AB_V4'] = {'AB': F.block(offn, W_OFF, KEYS, offe), 'AB_stress': F.block(offs, W_OFF, KEYS), 'AB_endpoint': F.endpoint(offn, W_OFF), 'yearly_AB': F.yearly(offn)}
NAVS['OFF'] = {'AB': offn, 'AB_stress': offs, 'nonAB': NV4['V4|normal']}
OUT['OFF_AB_V4']['lookahead_nonAB_minus_AB'] = {k: F.paired_cal(NV4['V4|normal'], offn, k) for k in ('ALL', 'SEL', 'VAL', 'HOLD')}

# ---------------- 配對 ----------------
PA = {}
for a, b in (('J0', 'J+1'), ('J0N', 'J+1'), ('J0N', 'J0'), ('J+1', 'OFF'), ('J0', 'OFF'), ('J0N', 'OFF')):
    for c in ('AB', 'AB_stress'):
        PA[f'{a}-{b}|{c}'] = {k: F.paired_cal(NAVS[a][c], NAVS[b][c], k) for k in ('ALL', 'SEL', 'VAL', 'HOLD', 'POST')}
    Wa = WIN[a]; Wb = WIN[b] if b in WIN else W_OFF
    PA[f'{a}-{b}|AB|window'] = {k: F.paired_win(NAVS[a]['AB'], Wa, NAVS[b]['AB'], Wb, k) for k in ('ALL', 'SEL', 'VAL', 'HOLD', 'POST')}
OUT['paired'] = PA

# ---------------- j ≥ 0 掃描（AB，10 seeds）----------------
JS = {}
for j in range(0, 13):
    days, end = F.version_days(j); W = F.windows([d for _, d in days], end)
    R = F.ab_navs(days, end, mode='day', seeds=range(10), q=Q, costs=('normal',), keep_seed=False)
    b = F.block(R['mean']['normal'], W, ('ALL', 'SEL', 'POST'))
    JS[j] = {'t_ALL': b['ALL']['t_0050'], 't_ALL_cal': b['ALL']['t_0050_cal'], 'CAGR_ALL': b['ALL']['CAGR'], 't_SEL': b['SEL']['t_0050'], 't_POST': b['POST']['t_0050'],
             'n_windows': len(days)}
    log('scan', j, round(JS[j]['t_ALL'], 3))
tv = np.array([v['t_ALL'] for v in JS.values()]); tc = np.array([v['t_ALL_cal'] for v in JS.values()])
OUT['jscan'] = {'by_j': JS, 'j_ge0_t_median': float(np.median(tv)), 'j_ge0_t_p10': float(np.percentile(tv, 10)), 'j_ge0_t_max': float(tv.max()),
                'j_ge0_tcal_median': float(np.median(tc)), 'j_ge0_tcal_p10': float(np.percentile(tc, 10)),
                'J0N_t_ALL': OUT['J0N']['AB']['ALL']['t_0050'], 'official_AB_V4_t': OUT['OFF_AB_V4']['AB']['ALL']['t_0050']}
early = [JS[j]['t_POST'] for j in range(0, 5)]; late = [JS[j]['t_POST'] for j in range(8, 13)]
OUT['jscan']['post_t_J0_4_mean'] = float(np.mean(early)); OUT['jscan']['post_t_J8_12_mean'] = float(np.mean(late))

# ---------------- 選中版本：成本損益兩平、分產業、最大貢獻股、下單次數 ----------------
sv = SELV; W = WIN[sv]
pts = [(s, F.block(NAVS[sv]['slip'][s], W, ('ALL',))['ALL']['t_0050']) for s in SLIPS]
def cross(th):
    for (s1, a), (s2, b) in zip(pts, pts[1:]):
        if a >= th > b: return s1 + (s2 - s1) * (a - th) / (a - b)
    return None if pts[0][1] < th else float('inf')
OUT['breakeven_selected'] = {'grid': pts, 'slip_at_t2.3': cross(2.3), 'slip_at_t2.0': cross(2.0)}
pl = PLANS[sv]; t_end = K.IDX.get_loc(F.version_days(CAND[sv][0])[1])
con, per = V.stock_contrib(pl, t_end)
ind = {}
for c, v in con.items():
    i = V.SL['industry'].get(c) if c in V.SL.index else None
    if not isinstance(i, str): i = '（已下市／無產業別）'
    ind[i] = ind.get(i, 0) + v
tot = sum(con.values())
OUT['robust_selected'] = {'industry_share_top10': sorted(((k, round(v / tot, 3)) for k, v in ind.items()), key=lambda x: -x[1])[:10],
                          'top_contrib_ALL': V.top_contrib(pl, t_end, '2006-04-01', '2026-10-31'),
                          'top_contrib_HOLD': V.top_contrib(pl, t_end, '2024-01-01', '2026-10-31')}
offpl = []
P4 = pd.read_csv(F.V4DIR / 'plan_V4.csv', dtype=str)
for sg, ex, pk in zip(P4.signal, P4.exec, P4.picks):
    offpl.append((K.IDX.get_loc(pd.Timestamp(ex)), np.array([K.CI[c] for c in pk.split(',')]), None, pd.Timestamp(sg)))
def orders(plan, cap):
    rows = []; prev = set()
    for e, pk, _, d in plan:
        t = K.IDX.get_loc(d); cur = set(int(j) for j in pk)
        buys, sells = cur - prev, prev - cur
        alloc = cap / 10.0
        def cnt(js):
            n = 0; odd_only = 0; need_odd = 0
            for j in js:
                p = K.Cn[t, j]
                if not np.isfinite(p) or p <= 0: continue
                sh = int(alloc // p); lots, odd = divmod(sh, 1000)
                n += int(lots > 0) + int(odd > 0); need_odd += int(odd > 0); odd_only += int(lots == 0)
            return n, need_odd, odd_only
        nb, ob, oob = cnt(buys); ns, os_, oos_ = cnt(sells)
        rows.append(dict(d=d, n_names=len(buys) + len(sells), n_orders=nb + ns, need_odd=ob + os_, odd_only=oob + oos_,
                         pre_intraday_odd=d < pd.Timestamp('2020-10-26')))
        prev = cur
    Dd = pd.DataFrame(rows).iloc[1:]
    def summ(x):
        return {'orders_per_month': float(x.n_orders.mean()), 'names_traded_per_month': float(x.n_names.mean()),
                'share_names_need_odd_lot': float(x.need_odd.sum() / max(1, x.n_names.sum())), 'share_names_odd_only': float(x.odd_only.sum() / max(1, x.n_names.sum())), 'n_months': int(len(x))}
    return {'ALL': summ(Dd), 'pre_2020-10-26(odd lots only after-hours at close)': summ(Dd[Dd.pre_intraday_odd]), 'post_2020-10-26': summ(Dd[~Dd.pre_intraday_odd])}
OUT['orders'] = {f'{sv}|{c}': orders(pl, c) for c in (3e5, 1e7)}
OUT['orders'].update({f'OFF|{c}': orders(offpl, c) for c in (3e5, 1e7)})
# 1,000 萬：新買名額 > 1% 的 60 日均成交值
def cap_share(plan, cap=1e7):
    n = big = 0; prev = set(); S52, V60 = V.rolling_signals('gapfix')
    for e, pk, _, d in plan:
        t = K.IDX.get_loc(d)
        for j in set(int(x) for x in pk) - prev:
            v = V60[t, j]
            if np.isfinite(v) and v > 0: n += 1; big += (cap / 10) > 0.01 * v
        prev = set(int(x) for x in pk)
    return {'n_new_slots': n, 'share_gt_1pct_adv60': big / n if n else None}
OUT['capacity_1e7'] = {sv: cap_share(pl), 'OFF': cap_share(offpl)}

# ---------------- 判定 ----------------
tS = OUT[sv]['AB']['ALL']['t_0050']; tO = OUT['OFF_AB_V4']['AB']['ALL']['t_0050']
cS = OUT[sv]['AB']['ALL']['CAGR']; e50 = OUT[sv]['AB']['ALL']['E50_CAGR']
OUT['decision'] = {'selected': sv, 't_selected_AB_ALL': tS, 't_official_AB_V4': tO, 'delta_t': tS - tO, 'downgrade_expectation': bool(tS <= tO - 0.1),
                   'verdict_definition': '可用（只改定義）', 'CAGR_selected_AB_ALL': cS, 'E50_CAGR': e50,
                   'live_expectation_0050_plus_half_excess': e50 + 0.5 * (cS - e50),
                   'live_expectation_official': OUT['OFF_AB_V4']['AB']['ALL']['E50_CAGR'] + 0.5 * (OUT['OFF_AB_V4']['AB']['ALL']['CAGR'] - OUT['OFF_AB_V4']['AB']['ALL']['E50_CAGR'])}
res['RESULTS'] = F.jsonable(OUT)
RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
NVd = {}
for k, v in NAVS.items():
    for c in ('nonAB', 'AB', 'AB_stress', 'EW'):
        if c in v: NVd[f'{k}|{c}'] = v[c]
pd.DataFrame(NVd).to_parquet(HERE / 'navs_oos.parquet')
for name in CAND:
    pd.DataFrame([(str(p[3].date()), str(K.IDX[p[0]].date()), ','.join(K.COLS[j] for j in p[1])) for p in PLANS[name]],
                 columns=['signal', 'exec', 'picks']).to_csv(HERE / f'plan_{name.replace("+", "p")}_seed0.csv', index=False)
log('done', OUT['decision'])

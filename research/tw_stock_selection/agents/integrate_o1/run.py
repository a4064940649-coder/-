#!/usr/bin/env python3
"""integrate_o1 回測：照 results.json 的 PREREG 跑全部組合、統計、SEL 機械選版、判定，寫回 results.json 的 RESULTS。
只讀 import o_capacity_tsmc_gap/ocore.py（→ main_v2/v2_core.py → independent_audit/ia_core.py）。"""
import sys, json, time, hashlib, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'o_capacity_tsmc_gap'))
import ocore as O
sys.path.pop(0)
import numpy as np, pandas as pd
K = O.K
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)

RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改過'
assert 'RESULTS' not in R, 'RESULTS 已存在：只跑一次'
J = O.J2330
KEYS = ('SEL', 'VAL', 'VAL_A', 'HOLD', 'S1', 'S2', 'S3', 'ALL')

S = O.V2R(1, 300)
pl = S.plan(10)
t0, t1, sd = S.bounds()
P = O.periods(sd)
ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
log('plan', len(pl))

# ---------------- 組合的持股計畫 ----------------
plans = {'MAIN': O.tsmc_plan(pl, 0.0), 'NEW': O.tsmc_plan(pl, 0.30),
         'MIX80_20': O.tsmc_plan(pl, 0.30 * 0.2), 'MIX50_50': O.tsmc_plan(pl, 0.30 * 0.5)}
# SLOT10
slot = []; held = set(); n_in = n_forced = n_blk = 0
for p in pl:
    pk = np.asarray(p[1], int); e = p[0]
    if J in pk:
        idx = pk; n_in += 1
    elif K.gap_up_open(e)[J] and J not in held:
        idx = pk; n_blk += 1
    else:
        idx = np.concatenate([pk[:9], [J]]); n_forced += 1
    slot.append((e, idx, np.full(len(idx), 1.0 / len(idx))))
    held = set(idx.tolist())
plans['SLOT10'] = slot
slot_info = {'months': len(pl), '2330_already_top10': n_in, '2330_forced_in': n_forced, '2330_blocked_limitup': n_blk}
# TREND30
trend = []; on = []
for p in pl:
    cs = S.cross_section(p[3]); s52 = cs['s52']; el = cs['el']
    v = s52[J]
    pct = float(np.mean(s52[el] <= v)) if np.isfinite(v) and el.sum() > 0 else np.nan
    w = 0.30 if (np.isfinite(pct) and pct >= 0.5) else 0.0
    on.append((p[3], w > 0, pct))
    trend += O.tsmc_plan([p], w)
plans['TREND30'] = trend
ons = pd.Series({d: o for d, o, _ in on})
trend_info = {'share_on_all': round(float(ons.mean()), 3),
              'share_on_SEL': round(float(ons[ons.index < O.SPLIT].mean()), 3),
              'share_on_VAL': round(float(ons[ons.index >= O.SPLIT].mean()), 3),
              'share_on_HOLD': round(float(ons[ons.index >= pd.Timestamp('2024-01-01')].mean()), 3),
              'off_months': [str(d.date()) for d, o, _ in on if not o]}
log('slot', slot_info, 'trend on', trend_info['share_on_all'])

NAV, NAVS = {}, {}
for k, pln in plans.items():
    NAV[k], _ = O.simulate_w(pln, *O.NORMAL, t0, t1)
    NAVS[k], _ = O.simulate_w(pln, *O.STRESS, t0, t1)
    log('sim', k, round(float(NAV[k].iloc[-1]), 2))

# 對照：0050 30%＋MAIN 70%（照 o_capacity oos.py mix_0050 寫法）
def mix_0050(nav, sd, x):
    e = K.E50.reindex(nav.index)
    rm, re_ = nav.pct_change().fillna(0).to_numpy(), e.pct_change().fillna(0).to_numpy()
    exd = set(nav.index.get_indexer(pd.DatetimeIndex(sd[:-1])) + 1)
    a, b = 1 - x, x; out_ = np.empty(len(nav))
    for i in range(len(nav)):
        if i > 0: a *= 1 + rm[i]; b *= 1 + re_[i]
        if i in exd:
            tot = a + b; tb = x * tot; tr = b - tb
            c = (0.000855 + 0.001 + 0.001) * max(tr, 0) + (0.000855 + 0.001) * max(-tr, 0)
            tot -= c; a, b = (1 - x) * tot, x * tot
        out_[i] = a + b
    return pd.Series(out_, index=nav.index)
NAV['C0050_30'] = mix_0050(NAV['MAIN'], sd, 0.30)
NAVS['C0050_30'] = mix_0050(NAVS['MAIN'], sd, 0.30)

# ---------------- 驗證：重現 o_capacity ----------------
OO = json.loads((HERE.parent / 'o_capacity_tsmc_gap' / 'results_oos.json').read_text(encoding='utf-8'))
st0 = O.segstat(NAV['MAIN'], ew, sd, *P['ALL']); st3 = O.segstat(NAV['NEW'], ew, sd, *P['ALL'])
val = {'MAIN_ALL_CAGR': st0['CAGR'], 'o_cap_w0': OO['tsmc']['w0']['stats']['ALL']['CAGR'],
       'NEW_ALL_CAGR': st3['CAGR'], 'o_cap_w30': OO['tsmc']['w30']['stats']['ALL']['CAGR']}
assert abs(val['MAIN_ALL_CAGR'] - val['o_cap_w0']) <= 0.05 and abs(val['NEW_ALL_CAGR'] - val['o_cap_w30']) <= 0.05, val
log('validate ok', val)

# ---------------- 統計 ----------------
B = K.E50
W = {k: O.window_returns(v, sd) for k, v in NAV.items()}
WB = O.window_returns(B, sd); WT = O.window_returns(O.TSMC_IDX, sd); WE = O.window_returns(ew, sd)
def win(x, a, b): return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
def corr(x, y, a, b):
    z = pd.concat([win(x, a, b), win(y, a, b)], axis=1).dropna()
    return round(float(z.iloc[:, 0].corr(z.iloc[:, 1])), 3)
def strip(st): return {k: v for k, v in st.items() if k != 'yearly'}
def yearly(nav):
    n = nav.dropna(); g = n.groupby(n.index.year).last()
    prev = g.shift(1); prev.iloc[0] = n.iloc[0]
    return (g / prev - 1)
E50n = B.reindex(NAV['MAIN'].index)
Y50 = yearly(E50n)

out = {'validation': val, 'slot10_info': slot_info, 'trend30_info': trend_info, 'combos': {}}
for k in NAV:
    d = {'stats': {}, 'stress': {}, 'paired_vs_MAIN': {}, 'corr_vs_MAIN': {}, 'corr_excess0050_vs_MAIN': {}}
    for p in KEYS:
        a, b, ad, bd = P[p]
        d['stats'][p] = strip(O.segstat(NAV[k], ew, sd, a, b, ad, bd))
        d['stress'][p] = strip(O.segstat(NAVS[k], ew, sd, a, b, ad, bd))
        d['paired_vs_MAIN'][p] = O.paired_t(W[k], W['MAIN'], a, b)
        d['corr_vs_MAIN'][p] = corr(W[k], W['MAIN'], a, b)
        d['corr_excess0050_vs_MAIN'][p] = corr(W[k] - WB, W['MAIN'] - WB, a, b)
    y = yearly(NAV[k])
    d['yearly_pct'] = {int(i): round(float(v) * 100, 1) for i, v in y.items()}
    d['years_beating_0050'] = int((y > Y50).sum()); d['n_years'] = int(len(y))
    out['combos'][k] = d
    log(k, {p: (d['stats'][p]['CAGR'], d['stats'][p]['maxDD'], d['stats'][p]['t_0050'], d['paired_vs_MAIN'][p]['t'], d['corr_vs_MAIN'][p]) for p in ('SEL', 'VAL', 'HOLD', 'ALL')})
out['yearly_0050_pct'] = {int(i): round(float(v) * 100, 1) for i, v in Y50.items()}
out['legs_corr'] = {p: {'MAIN_vs_2330': corr(W['MAIN'], WT, *P[p][:2]), 'MAIN_vs_0050': corr(W['MAIN'], WB, *P[p][:2]),
                        'MAINex_vs_2330ex(0050)': corr(W['MAIN'] - WB, WT - WB, *P[p][:2]),
                        '2330_vs_0050': corr(WT, WB, *P[p][:2]),
                        'mean_2330_minus_MAIN_pp': round(float(win(WT - W['MAIN'], *P[p][:2]).mean()) * 100, 3),
                        't_2330_minus_MAIN': round(K.tstat(win(WT - W['MAIN'], *P[p][:2])), 3)} for p in KEYS}
log('legs', {p: out['legs_corr'][p] for p in ('SEL', 'VAL', 'HOLD', 'ALL')})

# ---------------- 選版（只用 SEL）與判定 ----------------
C = out['combos']; m = C['MAIN']['stats']
cands = ['NEW', 'MIX80_20', 'MIX50_50', 'SLOT10', 'TREND30']
ok = [c for c in cands if m['SEL']['CAGR'] - C[c]['stats']['SEL']['CAGR'] <= 1.0]
pick = max(ok, key=lambda c: C[c]['stats']['SEL']['t_0050']) if ok else 'MAIN'
sel_tab = {c: {'SEL_t_0050': C[c]['stats']['SEL']['t_0050'], 'SEL_CAGR_loss_pp': round(m['SEL']['CAGR'] - C[c]['stats']['SEL']['CAGR'], 2), 'eligible': c in ok} for c in cands}
judg = {}
for c in cands + ['C0050_30']:
    s = C[c]['stats']
    c1 = s['VAL']['t_0050'] > m['VAL']['t_0050']
    c2 = (m['ALL']['CAGR'] - s['ALL']['CAGR']) <= 2.0
    c3 = s['ALL']['maxDD'] >= m['ALL']['maxDD']
    c4 = C[c]['paired_vs_MAIN']['VAL']['t'] > -1
    hold = s['HOLD']['t_0050'] > m['HOLD']['t_0050']
    v = '可用的配置選項' if (c1 and c2 and c3 and c4) else ('待觀察' if hold else '不可用')
    judg[c] = {'VAL_t_higher': bool(c1), 'ALL_loss_le_2pp': bool(c2), 'ALL_DD_not_deeper': bool(c3), 'VAL_paired_t_gt_-1': bool(c4),
               'HOLD_t_improves': bool(hold), 'verdict': v}
out['selection'] = {'table': sel_tab, 'picked': pick}
out['judgement'] = judg
log('pick', pick, sel_tab); log('judg', judg)

pd.DataFrame(NAV).to_parquet(HERE / 'navs.parquet')
R['RESULTS'] = out
R['RESULTS_written_at'] = pd.Timestamp.now(tz='UTC').isoformat()
RJ.write_text(O.dumps(R), encoding='utf-8')
log('done')

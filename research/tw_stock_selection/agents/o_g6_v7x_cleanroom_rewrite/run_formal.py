# R3 正式執行（一次）：python3 -I -B audit_run.py A run_formal.py <K 名單 pickle>
# 斷言 PREREG 已存在且程式 sha 相符，才計算任何正式口徑統計量
import numpy as np, pandas as pd, json, os, sys, pickle, hashlib, datetime, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = len(sys.argv) > 2 and sys.argv[2] == 'smoke'
if not SMOKE:
    R = json.load(open(os.path.join(HERE, 'results.json')))
    assert 'PREREG' in R and 'RESULTS' not in R, 'PREREG 必須存在且尚未正式執行'
    PF = json.load(open(os.path.join(HERE, 'POSTFREEZE_FIXES.json')))['fixed_sha256'] if os.path.exists(os.path.join(HERE, 'POSTFREEZE_FIXES.json')) else {}
    for f, h in R['PREREG']['code_sha256'].items():
        h = PF.get(f, h)
        assert hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() == h, f'程式已變動: {f}'
else:
    R = {}
import xdata, xengine, xlists, xstats, xrun, xpaper
KP = sys.argv[1]
KD = pickle.load(open(KP, 'rb'))
L = xlists.Lists(); W = xengine.World(); dates = W.dates; E = W.X['E']; T = len(dates)
CUT = pd.Timestamp('2026-10-12')
def end_for_j(j):
    D = pd.Timestamp('2026-10-12'); tv = L.jdays.__self__.tx.values if hasattr(L, 'tx') else None
    L.jdays(0); tv = L.tx.values
    k = int(np.searchsorted(tv, np.datetime64(D), 'left')) + j
    if k >= len(tv): return T - 1
    return min(int(np.searchsorted(dates.values, tv[k], 'left')), T - 1)
ENDS = {j: end_for_j(j) for j in range(13)}
JR, SR = (range(1), range(1)) if SMOKE else (range(13), range(20))
if SMOKE:
    for v in KD['variants']: KD['variants'][v] = {k: pl[:6] for k, pl in KD['variants'][v].items() if k == (0, 0)}
    for v in KD.get('paper', {}): KD['paper'][v] = KD['paper'][v][:6]
    ENDS = {j: None for j in range(13)}
def conv(plan):
    out = []
    for s, e, pk in plan:
        si = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(s)), 'left')); ei = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(e)), 'left'))
        out.append((si, ei, [W.ci[c] for c in pk if c in W.ci]))
    return out
CELLS = {'N100': dict(tier=1_000_000), 'S100': dict(tier=1_000_000, cost='stress'), 'N300': dict(tier=3_000_000), 'N1000': dict(tier=10_000_000),
         'W30': dict(tier=300_000, whole_pre2020=True), 'O30': dict(tier=300_000)}
t0 = time.time(); log = lambda *a: print(f'[{time.time()-t0:7.0f}s]', *a, flush=True)
jobs = []; sigs = {}
# Track E：K 的原名單
for var, plans in KD['variants'].items():
    merger = 'orig' if var == 'orig' else 'cons_main'
    for (j, sd), pl in plans.items():
        p = conv(pl); sigs[(var, j)] = [x[0] for x in p]
        if SMOKE: ENDS[j] = p[-1][0]
        for cn, cfg in CELLS.items():
            jobs.append(((f'E|{var}|{merger}|{cn}', j, sd), p, ENDS[j], dict(cfg, merger=merger)))
        if var == 'orig':
            jobs.append(((f'E|{var}|cons_main|N100', j, sd), p, ENDS[j], dict(CELLS['N100'], merger='cons_main')))
# Track L：自建 AB_V7_B10
myplans = {}
for j in JR:
    days = [x for x in L.jdays(j) if dates[x[1]] < CUT]
    if SMOKE: days = days[:6]
    for sd in SR:
        p = L.plan(days, seed=sd)
        if SMOKE: ENDS[j] = p[-1][0]
        myplans[(j, sd)] = p; sigs[('mine', j)] = [x[0] for x in p]
        for cn, cfg in CELLS.items():
            jobs.append(((f'L|mine|cons_main|{cn}', j, sd), p, ENDS[j], dict(cfg, merger='cons_main')))
        jobs.append(((f'L|mine|orig|N100', j, sd), p, ENDS[j], dict(CELLS['N100'], merger='orig')))
log('lists built', len(jobs))
pickle.dump({k: [(dates[s].strftime('%Y-%m-%d'), dates[e].strftime('%Y-%m-%d'), [W.codes[c] for c in pk]) for s, e, pk in v] for k, v in myplans.items()}, open(xdata.SCR + 'my_plans_AB_V7_B10.pkl', 'wb'))
# 歸因階梯（P4）：REAL_opt × cons_main × 100 萬一般
LADDER = [('S0_close_frictionless', dict(timing='close', lots=False, lim=False, tax=False, fee=False, imp=False, caps=False, cash98=False, disp=False)),
          ('S1_open', dict(lots=False, lim=False, tax=False, fee=False, imp=False, caps=False, cash98=False, disp=False)),
          ('S2_lim', dict(lots=False, tax=False, fee=False, imp=False, caps=False, cash98=False, disp=False)),
          ('S3_tax', dict(lots=False, fee=False, imp=False, caps=False, cash98=False, disp=False)),
          ('S4_fee', dict(lots=False, imp=False, caps=False, cash98=False, disp=False)),
          ('S5_imp_caps', dict(lots=False, cash98=False, disp=False)),
          ('S6_lots_odd', dict(cash98=False, disp=False)),
          ('S7_full', dict())]
LV = 'REAL_opt' if 'REAL_opt' in KD['variants'] else list(KD['variants'])[0]
for name, cfg in LADDER:
    for (j, sd), pl in KD['variants'][LV].items():
        jobs.append(((f'A|{LV}|{name}', j, sd), conv(pl), ENDS[j], dict(CELLS['N100'], merger='cons_main', **cfg)))
# 右尾（P5）
for (j, sd), pl in KD['variants'][LV].items():
    jobs.append(((f'P5|{LV}|N100', j, sd), conv(pl), ENDS[j], dict(CELLS['N100'], merger='cons_main', record=True)))
log('jobs', len(jobs))
res = xrun.run_grid(jobs, procs=4)
log('sims done')
pickle.dump(res, open(xdata.SCR + 'formal_res_raw.pkl', 'wb'))
OUT = {'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'cells': {}}
cells = sorted(set(k[0] for k in res))
navstore = {}
for c in cells:
    if c.startswith('P5'): continue
    var = c.split('|')[1]
    js = sorted(set(k[1] for k in res if k[0] == c)); per_j = {}
    for j in js:
        ks = sorted(k for k in res if k[0] == c and k[1] == j)
        nav = xstats.seed_avg([res[k][0] for k in ks]); navstore[(c, j)] = nav
        sg = sigs[(var if not c.startswith('L|') else 'mine', j)]
        per_j[j] = xstats.metrics(nav, sg, ENDS[j], E, dates)
    OUT['cells'][c] = {'sum': xstats.jsummary(per_j), 'per_j': {j: {k: v for k, v in m.items() if k not in ('win_ex', 'win_start')} for j, m in per_j.items()},
                       'win': {j: {'ex': per_j[j]['win_ex'], 'start': per_j[j]['win_start']} for j in per_j}}
pickle.dump(navstore, open(xdata.SCR + 'formal_navs.pkl', 'wb'))
log('stats done')
# 右尾
def paper_slot_ret(e0, e1, j):
    if not (W.TR[e0, j] and np.isfinite(W.O[e0, j])): return np.nan
    v = W.C[e0, j] / W.O[e0, j]
    for t in range(e0 + 1, e1): v *= W.G[t, j]
    if W.TR[e1, j]: v *= W.G[e1, j] * W.O[e1, j] / W.C[e1, j]
    return v - 1
rows = []
for k in [k for k in res if k[0].startswith('P5')]:
    j, sd = k[1], k[2]; pl = conv(KD['variants'][LV][(j, sd)]); ex = res[k][1]
    pnl, eqo = ex['pnl'], ex['eq_open']
    for w, (s, e, pk) in enumerate(pl):
        e1 = pl[w + 1][1] if w + 1 < len(pl) else ENDS[j]
        for c in pk:
            rp = paper_slot_ret(e, e1, c)
            if not np.isfinite(rp): continue
            rows.append((j, sd, w, dates[e].year, rp / len(pk), pnl.get((w, c), 0.0) / eqo.get(w, np.nan)))
df = pd.DataFrame(rows, columns=['j', 'seed', 'w', 'year', 'paper', 'exec'])
q = df.paper.quantile(0.95); top = df[df.paper >= q]
def cap(x): return float(x.exec.sum() / x.paper.sum()) if len(x) else None
OUT['P5'] = {'n_slots': len(df), 'top5_threshold': float(q), 'capture_all_top5': cap(top),
             'capture_top5_2006_2012': cap(top[top.year <= 2012]), 'capture_top5_2013_2018': cap(top[(top.year >= 2013) & (top.year <= 2018)]),
             'capture_top5_2019on': cap(top[top.year >= 2019]), 'capture_all_slots': cap(df),
             'top5_share_of_paper_sum': float(top.paper.sum() / df.paper.sum())}
log('P5 done')
# 紙上：K 的紙上名單
PAP = {}
base = xstats.last_le(dates, '2006-03-31'); pend = xstats.last_le(dates, '2026-10-12')
if SMOKE: pend = conv(list(KD['paper'].values())[0])[-1][0]
for var, pl in KD.get('paper', {}).items():
    p = conv(pl); sg = [x[0] for x in p]
    for cost in ['normal', 'stress']:
        for merger in (['freeze', 'cons_main'] if var == 'C0' else ['cons_main']):
            nav = xpaper.run_paper(W, p, pend, cost=cost, merger=merger, start_base=base)
            m = xstats.metrics(nav, [base] + sg[1:] if False else sg, pend, E, dates)
            m['ann_from_base'] = xstats.ann(nav, base, pend, dates); m['ann0050_from_base'] = xstats.ann(E, base, pend, dates)
            PAP[f'{var}|{cost}|{merger}'] = {k: v for k, v in m.items() if k not in ('win_ex', 'win_start')}
            PAP[f'{var}|{cost}|{merger}']['win_ex'] = m['win_ex']
# 兩平點：單邊滑價 x（取代 0.1%）
def paper_bf(var, x):
    p = conv(KD['paper'][var]); sg = [z[0] for z in p]
    import xpaper as XP
    bf = 0.001425 * 0.6 + x; sf = bf + 0.003
    nav = run_paper_rates(W, p, pend, bf, sf, base)
    m = xstats.metrics(nav, sg, pend, E, dates)
    return xstats.ann(nav, base, pend, dates) - xstats.ann(E, base, pend, dates), m['t']
def run_paper_rates(W, p, end, bf, sf, base):
    import types
    src = xpaper.run_paper
    # 以參數化費率重跑：複製邏輯（cost='custom'）
    return xpaper.run_paper(W, p, end, cost=('custom', bf, sf), merger='cons_main', start_base=base)
BE = {}
for var in [v for v in KD.get('paper', {}) if v != 'C0']:
    grid = np.round(np.arange(0.0, 0.0101, 0.0005), 5); vals = []
    for x in grid:
        ex_, t_ = paper_bf(var, x); vals.append((float(x), float(ex_), float(t_)))
    def cross(idx, lvl):
        for a, b in zip(vals[:-1], vals[1:]):
            if (a[idx] - lvl) * (b[idx] - lvl) <= 0 and a[idx] != b[idx]:
                return a[0] + (lvl - a[idx]) * (b[0] - a[0]) / (b[idx] - a[idx])
        return None
    BE[var] = {'grid': vals, 'slip_t_below_1': cross(2, 1.0), 'slip_excess_zero': cross(1, 0.0)}
OUT['paper'] = PAP; OUT['breakeven'] = BE
log('paper done')
if SMOKE:
    json.dump({'smoke_keys': list(OUT['cells'])[:50], 'P5_keys': list(OUT['P5'])}, open(xdata.SCR + 'smoke_out.json', 'w'), default=float)
else:
    R['RESULTS'] = OUT
    json.dump(R, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=float)
log('written')

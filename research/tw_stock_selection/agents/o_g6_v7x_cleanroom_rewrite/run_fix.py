# 凍結後修正重跑（R4 規則「重寫錯：修正後只重跑受影響的格，記為凍結後修正」）
# 用法：python3 -I -B audit_run.py A run_fix.py <K 名單 pickle> <tag> [cells]
# 斷言：RESULTS（正式一次）已存在、POSTFREEZE_FIXES 的 sha 與程式一致；結果寫 results.json 的 RESULTS_<tag>，不覆蓋 RESULTS。
import numpy as np, pandas as pd, json, os, sys, pickle, hashlib, datetime, time
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results.json')))
assert 'PREREG' in R and 'RESULTS' in R, '正式執行必須先完成'
PF = json.load(open(os.path.join(HERE, 'POSTFREEZE_FIXES.json')))['fixed_sha256']
for f, h in R['PREREG']['code_sha256'].items():
    if f in ('run_formal.py',): continue
    h = PF.get(f, h)
    assert hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() == h, f'程式已變動: {f}'
import xdata, xengine, xlists, xstats, xrun
PATCH = os.environ.get('G6_PATCH', '')
if PATCH:  # J 診斷用：原引擎口徑（規格缺口的另一讀法）
    import j_patch
    if 'tiecol' in PATCH: j_patch.patch_tiecol()
    _wi = xengine.World.__init__
    def _wi2(self, *a, **k):
        _wi(self, *a, **k); j_patch.patch(self, ','.join(x for x in PATCH.split(',') if x != 'tiecol'))
    xengine.World.__init__ = _wi2
assert xdata.load().get('FIX_F1', False), 'F1 未生效'
KP, TAG = sys.argv[1], sys.argv[2]
WANT = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] != 'all' else ['N100', 'S100', 'N300', 'N1000', 'W30', 'O30']
VARS = sys.argv[4].split(',') if len(sys.argv) > 4 else None
TRACKL = os.environ.get('G6_TRACKL', '1') == '1'
KD = pickle.load(open(KP, 'rb'))
L = xlists.Lists(); W = xengine.World(); dates = W.dates; E = W.X['E']; T = len(dates)
CUT = pd.Timestamp('2026-10-12')
def end_for_j(j):
    L.jdays(0); tv = L.tx.values
    k = int(np.searchsorted(tv, np.datetime64(pd.Timestamp('2026-10-12')), 'left')) + j
    if k >= len(tv): return T - 1
    return min(int(np.searchsorted(dates.values, tv[k], 'left')), T - 1)
ENDS = {j: end_for_j(j) for j in range(13)}
def conv(plan):
    return [(int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(s)), 'left')), int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(e)), 'left')),
             [W.ci[c] for c in pk if c in W.ci]) for s, e, pk in plan]
CELLS = {'N100': dict(tier=1_000_000), 'S100': dict(tier=1_000_000, cost='stress'), 'N300': dict(tier=3_000_000), 'N1000': dict(tier=10_000_000),
         'W30': dict(tier=300_000, whole_pre2020=True), 'O30': dict(tier=300_000)}
t0 = time.time(); log = lambda *a: print(f'[{time.time()-t0:7.0f}s]', *a, flush=True)
jobs = []; sigs = {}
for var, plans in KD['variants'].items():
    if VARS and var not in VARS: continue
    merger = 'orig' if var == 'orig' else 'cons_main'
    for (j, sd), pl in plans.items():
        p = conv(pl); sigs[(var, j)] = [x[0] for x in p]
        for cn in WANT:
            if var == 'orig' and cn != 'N100': continue
            jobs.append(((f'E|{var}|{merger}|{cn}', j, sd), p, ENDS[j], dict(CELLS[cn], merger=merger)))
myplans = {}
for j in (range(13) if TRACKL else []):
    days = [x for x in L.jdays(j) if dates[x[1]] < CUT]
    for sd in range(20):
        p = L.plan(days, seed=sd); myplans[(j, sd)] = p; sigs[('mine', j)] = [x[0] for x in p]
        for cn in WANT:
            jobs.append(((f'L|mine|cons_main|{cn}', j, sd), p, ENDS[j], dict(CELLS[cn], merger='cons_main')))
pickle.dump({k: [(dates[s].strftime('%Y-%m-%d'), dates[e].strftime('%Y-%m-%d'), [W.codes[c] for c in pk]) for s, e, pk in v] for k, v in myplans.items()},
            open(xdata.SCR + f'my_plans_AB_V7_B10_{TAG}.pkl', 'wb'))
log('jobs', len(jobs))
res = xrun.run_grid(jobs, procs=4)
log('sims done')
OUT = {'patch': PATCH, 'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'fixes': [f['id'] for f in json.load(open(os.path.join(HERE, 'POSTFREEZE_FIXES.json')))['fixes'] if 'id' in f], 'cells': {}}
navstore = {}
for c in sorted(set(k[0] for k in res)):
    var = c.split('|')[1]; per_j = {}
    for j in sorted(set(k[1] for k in res if k[0] == c)):
        ks = sorted(k for k in res if k[0] == c and k[1] == j)
        nav = xstats.seed_avg([res[k][0] for k in ks]); navstore[(c, j)] = nav
        per_j[j] = xstats.metrics(nav, sigs[(var if not c.startswith('L|') else 'mine', j)], ENDS[j], E, dates)
    OUT['cells'][c] = {'sum': xstats.jsummary(per_j), 'per_j': {j: {k: v for k, v in m.items() if k not in ('win_ex', 'win_start')} for j, m in per_j.items()},
                       'win': {j: {'ex': per_j[j]['win_ex'], 'start': per_j[j]['win_start']} for j in per_j}}
pickle.dump(navstore, open(xdata.SCR + f'navs_{TAG}.pkl', 'wb'))
json.dump(OUT, open(os.path.join(HERE, 'logs', f'results_{TAG}.json'), 'w'), ensure_ascii=False, default=float)  # 側檔（避免同時寫入競爭）
import fcntl
_lk = open(os.path.join(HERE, 'logs', '.results.lock'), 'w'); fcntl.flock(_lk, fcntl.LOCK_EX)
R = json.load(open(os.path.join(HERE, 'results.json')))
R[f'RESULTS_{TAG}'] = OUT
json.dump(R, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=float)
log('written')

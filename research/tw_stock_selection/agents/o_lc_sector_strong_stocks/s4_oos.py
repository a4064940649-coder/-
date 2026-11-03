#!/usr/bin/env python3
"""樣本外與全期（PREREG 凍結後才可執行）。先斷言 PREREG 與程式／資料雜湊，才讀 2017-01 以後。
階段：python3 -I -B s4_oos.py paper|exec|placebo|analysis
  paper   ：全部格點＋參照（B、IND_ONLY）在 FULL（2006-04 起連續；2014 前為非 PIT 股數，只作描述）13 個 j，一般／零成本；關鍵規格加壓力成本；
            另跑官方日（G3 族群格點用）。存 out/paper.pkl。
  exec    ：關鍵規格 × 13 j × 30 萬／100 萬／300 萬／1,000 萬 × FIX／GROW × 一般／壓力（K5T），另加官方日 100 萬 GROW 一般。存 out/exec.pkl。
  placebo ：選定版 Q7 (a)(b)(c) 各 1,000 次 × 13 j（PIT 段 2014 起，驗證期取 2017 起切片），一般與零成本。存 out/placebo_<kind>.npz。
  analysis：彙總、Q1～Q7、Brinson、右尾、G1～G8、規則書，寫 results.json。"""
import sys, json, math, pickle, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lc_lib as X

RES = HERE / 'results.json'
PRE = json.loads((HERE / 'data' / 'PREREG.json').read_text(encoding='utf-8'))
_r = json.loads(RES.read_text(encoding='utf-8'))
assert 'PREREG' in _r and X.canon(_r['PREREG']) == _r['PREREG_sha256'] == PRE['_sha256_body'], 'PREREG 缺或被改'
AMF = HERE / 'data' / 'PREREG_AMEND.json'
CODE_SHA = dict(PRE['body']['code_sha256'])
if AMF.exists():
    AM = json.loads(AMF.read_text(encoding='utf-8'))
    assert AM['base_prereg_sha256'] == PRE['_sha256_body'], 'AMEND 不是對這份 PREREG'
    CODE_SHA = AM['code_sha256']
for rel, h in CODE_SHA.items(): assert X.sha(HERE / rel) == h, f'{rel} 雜湊不符'
for p, h in PRE['body']['data_sha256'].items(): assert X.sha(p) == h, f'{p} 資料雜湊不符'
BODY = PRE['body']
X.set_mode(BODY['mode'])
import lc_run as Rn, lc_exec as E, lc_placebo as PL
OUTD = HERE / 'out'; OUTD.mkdir(exist_ok=True)
SCR = X.SCR
SEL_SPEC = BODY['selected_spec']; DEF_SPEC = BODY['default_spec']
OPT = {k: SEL_SPEC.get(k, 1 if k == 'buf' else 'prop') for k in ('buf', 'overflow')}

def g(sp): return {k: sp[k] for k in ('kind', 'n', 'wt', 'cap', 'carrier') if k in sp}
def withopt(sp): return dict(sp, **OPT)
KEY = {
    'SEL': withopt(g(SEL_SPEC)), 'DEF': dict(g(DEF_SPEC), buf=1, overflow='prop'),
    'SEL_PURE': withopt(dict(g(SEL_SPEC), carrier='PURE')), 'SEL_SATv': withopt(dict(g(SEL_SPEC), carrier='SAT')),
    'CTRL_SEL': withopt(dict(g(SEL_SPEC), kind='CTRL')), 'CTRL_SEL_PURE': withopt(dict(g(SEL_SPEC), kind='CTRL', carrier='PURE')),
    'CTRL_SEL_SAT': withopt(dict(g(SEL_SPEC), kind='CTRL', carrier='SAT')),
    'CTRL_DEF': dict(g(DEF_SPEC), kind='CTRL', buf=1, overflow='prop'),
    'IND_ONLY_PURE': dict(kind='IND_ONLY', carrier='PURE'), 'IND_ONLY_SAT': dict(kind='IND_ONLY', carrier='SAT'), 'B': dict(kind='B', carrier='PURE'),
}
GRIDS = [withopt(gg) for gg in Rn.GRID]
stage = sys.argv[1]

def runp(sp, j, per, cost):
    days = X.days_for(j, per) if j != 'off' else X.sig_days('off', X.PERIODS[per][0], X.PERIODS[per][1], X.PERIODS[per][2])
    pl, info = Rn.plans(sp, days)
    R, TO = X.vrun(days, pl, cost)
    return dict(days=days, r=R[0], to=TO[0], b=X.bench_win(days)), pl, info

if stage == 'paper':
    out = {}; plans_store = {}
    specs = {Rn.name(s): s for s in GRIDS}; specs.update({k: v for k, v in KEY.items()})
    for nm, sp in specs.items():
        t0 = time.time()
        for j in Rn.JS:
            pl = info = None
            for cost in ('normal', 'zero') + (('stress',) if nm in KEY else ()):
                days = X.days_for(j, 'FULL')
                if pl is None: pl, info = Rn.plans(sp, days)
                R, TO = X.vrun(days, pl, cost)
                out[(nm, j, cost)] = dict(days=days, r=R[0], to=TO[0])
            if nm in KEY: plans_store[(nm, j)] = (pl, info)
        o, _, _ = runp(sp, 'off', 'FULL', 'normal'); out[(nm, 'off', 'normal')] = o
        print(nm, round(time.time() - t0, 1), flush=True)
    (SCR / 'paper.pkl').write_bytes(pickle.dumps(out)); (SCR / 'plans_key.pkl').write_bytes(pickle.dumps(plans_store))
    print('paper done')

if stage == 'paper_comp':
    # 切換前版本（題 1 事前預設 COMP、k=5、本題產業表）：預設格與參照，供「切換前後兩版都要報」
    X.set_mode('COMP'); out = {}
    for nm in ('DEF', 'CTRL_DEF', 'IND_ONLY_PURE', 'IND_ONLY_SAT', 'B'):
        sp = KEY[nm]
        for j in Rn.JS:
            days = X.days_for(j, 'FULL'); pl, info = Rn.plans(sp, days)
            for cost in ('normal', 'zero'):
                R, TO = X.vrun(days, pl, cost); out[(nm, j, cost)] = dict(days=days, r=R[0], to=TO[0])
        print('comp', nm, flush=True)
    (SCR / 'paper_comp.pkl').write_bytes(pickle.dumps(out)); print('paper_comp done')

if stage == 'exec':
    from multiprocessing import Pool
    PS = pickle.loads((SCR / 'plans_key.pkl').read_bytes())
    ekeys = ['SEL', 'DEF', 'SEL_PURE', 'CTRL_SEL', 'CTRL_SEL_PURE', 'CTRL_DEF', 'IND_ONLY_PURE', 'IND_ONLY_SAT', 'B']
    OFFP = {}
    for nm in ekeys:
        days = X.sig_days('off', '2006-04-01', '2026-10-31'); pl, _ = Rn.plans(KEY[nm], days); OFFP[nm] = (days, pl)
    def job(a):
        nm, j, tn, mode, cost = a
        if j == 'off': days, pl = OFFP[nm]
        else: days = X.days_for(j, 'FULL'); pl = PS[(nm, j)][0]
        r, acc = E.run_exec(days, pl, E.TIERS[tn], mode, 0.005 if cost == 'stress' else 0.0)
        return a, dict(r=r, cost_pp=E.cost_pp(acc, E.TIERS[tn], len(r)), odd_share=acc['odd_val'] / max(acc['odd_val'] + acc['lot_val'], 1),
                       pre_odd=acc['pre_odd_val'] / max(acc['odd_val'] + acc['lot_val'], 1), cap_hit=acc['cap_hit'], disp=acc['disp_sells'])
    jobs = [(nm, j, tn, mode, cost) for nm in ekeys for j in Rn.JS for tn in E.TIERS for mode in ('GROW', 'FIX') for cost in ('normal', 'stress')]
    jobs += [(nm, 'off', '100萬', 'GROW', 'normal') for nm in ekeys]
    with Pool(4) as pool: res = pool.map(job, jobs, chunksize=8)
    (SCR / 'exec.pkl').write_bytes(pickle.dumps(dict(res)))
    print('exec done', len(res))

if stage == 'placebo':
    kind = sys.argv[2]
    from multiprocessing import Pool
    cal = BODY['placebo_rho']
    def job(j):
        days = X.days_for(j, 'PIT'); nets, gross = [], []
        for c in range(4):
            W = PL.sim_weights(KEY['SEL'], days, kind, 250, 20261103 + 1000 * c + j, cal['rho_sec'], cal['rho_stk'])
            Rn_, _ = X.vrun(days, W, 'normal'); Rz, _ = X.vrun(days, W, 'zero'); del W
            nets.append(Rn_.astype(np.float32)); gross.append(Rz.astype(np.float32))
        return j, np.vstack(nets), np.vstack(gross), np.array([str(d.date()) for d in days])
    with Pool(4) as pool: res = pool.map(job, Rn.JS)
    np.savez_compressed(SCR / f'placebo_{kind}.npz', **{f'net{j}': a for j, a, b, d in res}, **{f'gross{j}': b for j, a, b, d in res}, **{f'days{j}': d for j, a, b, d in res})
    print('placebo done', kind)

if stage == 'analysis':
    exec(open(HERE / 's5_analysis.py', encoding='utf-8').read())

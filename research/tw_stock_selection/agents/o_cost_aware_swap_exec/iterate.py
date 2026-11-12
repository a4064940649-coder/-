#!/usr/bin/env python3
"""挑選期（≤ 2014-12-31，模擬以 t1 截斷）：8 個事前登記變體＋參照臂 A20，在 4 格（100 萬一般／100 萬成本全關／100 萬壓力／30 萬一般）
× 3 放回 × 13 j × 20 seeds 計算主指標與選擇條件，依 PREREG 的選擇規則選出唯一候選。
階段 1：BASE、CA80、CA90、MT、NR、DC5C、DC5E、A20；階段 2：CB1（較佳 CA＋MT＋NR）、CB2（CB1＋較佳 DC）。
輸出：iterations.json、data/sel_navs.parquet（20 seeds 平均淨值，各格各變體各 j）、data/sel_track.parquet（100 萬一般的逐日成本追蹤，13 j × 3 放回平均）。
python3 -I -B iterate.py"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; import run_lib as RL; sys.path.pop(0)
RES = HERE / 'results.json'
R0 = json.loads(RES.read_text(encoding='utf-8'))
assert CL.canon(R0['PREREG']) == R0['PREREG_sha256'] and CL.canon(R0['PREREG_POWER']) == R0['PREREG_POWER_sha256']
assert 'PREREG_FREEZE' not in R0
SEL_END = '2014-12-31'
OPTS = {'BASE': {}, 'CA80': {'ca': 0.8}, 'CA90': {'ca': 0.9}, 'MT': {'mt': 1.0}, 'NR': {'nr': True}, 'DC5C': {'dc': 5, 'dcmode': 'C'},
        'DC5E': {'dc': 5, 'dcmode': 'E'}, 'A20': {'a20': True}}
CELLS = {'N100': ('100萬', 'normal', None), 'OFF100': ('100萬', 'normal', CL.OFFC), 'S100': ('100萬', 'stress', None), 'N30': ('30萬', 'normal', None)}
NCOMP = {'CA80': 1, 'CA90': 1, 'MT': 1, 'NR': 1, 'DC5C': 1, 'DC5E': 1, 'CB1': 3, 'CB2': 4}
MDE = R0['PREREG_POWER']['MDE_primary_pp']['selection']
def run(tags, log):
    t1 = RL.t_end_idx(SEL_END)
    out = {}
    for c, (tier, cost, sw) in CELLS.items():
        for tg in tags:
            jobs = RL.jobs_for(f'{c}|{tg}', tier, cost, OPTS[tg], sw=sw, t1=t1, track=(c == 'N100'))
            out.update(RL.run_batch(jobs, nproc=2, log=f'{log}:{c}|{tg}', ckpt=f'sel_{c}_{tg}_{CL.canon(OPTS[tg])[:10]}.pkl'))
    return out
def metrics(res, tg):
    m = {}
    for c in CELLS:
        p = RL.paired(res, f'{c}|{tg}', f'{c}|BASE', '2006-01-01', SEL_END)
        pia = RL.paired(res, f'{c}|{tg}', f'{c}|BASE', '2006-01-01', SEL_END, ia=True)
        m[c] = dict(worst=p['worst'], by_variant={v: {k: p[v][k] for k in ('med', 'mean', 'max', 'min', 'p10', 'med_j')} for v in RL.VARS},
                    ia_worst_med=pia['worst']['med'])
    st = {}
    for v in RL.VARS:
        for j in range(13):
            for k, x in res[(f'N100|{tg}', v, j)]['st'].items(): st[k] = st.get(k, 0) + x
    m['stats_N100'] = st
    tr = pd.concat([res[(f'N100|{tg}', v, j)]['tr'] for v in RL.VARS for j in range(13)], axis=1, keys=range(39))
    yrs = {}
    for k in ('tax', 'fee', 'tick', 'imp', 'odd', 'disp', 'turn', 'ord'):
        cols = [tr[i][k] for i in range(39)]
        y = [float(cc.sum() / ((cc.index[-1] - cc.index[0]).days / 365.25)) for cc in cols] if k != 'ord' else [float(cc.mean()) for cc in cols]
        yrs[k] = float(np.median(y)) * (100 if k not in ('turn', 'ord') else 1)
    yrs['cost_total_pp'] = sum(yrs[k] for k in ('tax', 'fee', 'tick', 'imp', 'odd', 'disp'))
    m['track_N100'] = yrs
    # 選擇條件
    a = m['N100']['worst']['med']; b = m['OFF100']['worst']['med']; c_ = m['S100']['worst']['med']; d = m['N30']['worst']['med']
    m['cond'] = dict(a=bool(a >= 0.5 and a >= MDE / 2), b=bool(b >= -0.3), c=bool(c_ > 0), d=bool(d > 0))
    m['eligible'] = all(m['cond'].values())
    return m
if __name__ == '__main__':
    t0 = time.time()
    res = run(list(OPTS), 'stage1')
    IT = dict(written_utc=None, MDE_selection_pp=MDE, versions={})
    for tg in OPTS:
        if tg == 'BASE': continue
        IT['versions'][tg] = dict(stage=1, opt=OPTS[tg], counts_in_N=(tg != 'A20'), m=metrics(res, tg))
        print(tg, {c: round(IT['versions'][tg]['m'][c]['worst']['med'], 3) for c in CELLS}, IT['versions'][tg]['m']['cond'], flush=True)
    IT['versions']['BASE_track'] = dict(stage=1, track_N100=metrics(res, 'BASE')['track_N100'], stats_N100=metrics(res, 'BASE')['stats_N100'])
    ca = max(['CA80', 'CA90'], key=lambda k: IT['versions'][k]['m']['N100']['worst']['med'])
    dc = max(['DC5C', 'DC5E'], key=lambda k: IT['versions'][k]['m']['N100']['worst']['med'])
    OPTS['CB1'] = dict(OPTS[ca], mt=1.0, nr=True); OPTS['CB2'] = dict(OPTS['CB1'], dc=OPTS[dc]['dc'], dcmode=OPTS[dc]['dcmode'])
    IT['combo_choice'] = dict(CA=ca, DC=dc, CB1=OPTS['CB1'], CB2=OPTS['CB2'])
    res2 = run(['CB1', 'CB2'], 'stage2')
    res.update(res2)
    for tg in ('CB1', 'CB2'):
        IT['versions'][tg] = dict(stage=2, opt=OPTS[tg], counts_in_N=True, m=metrics(res, tg))
        print(tg, {c: round(IT['versions'][tg]['m'][c]['worst']['med'], 3) for c in CELLS}, IT['versions'][tg]['m']['cond'], flush=True)
    # 選擇
    el = [k for k in NCOMP if IT['versions'][k]['m']['eligible']]
    if el:
        best = max(el, key=lambda k: IT['versions'][k]['m']['N100']['worst']['med'])
        bv = IT['versions'][best]['m']['N100']['worst']['med']
        close = [k for k in el if bv - IT['versions'][k]['m']['N100']['worst']['med'] <= 0.2]
        chosen = min(close, key=lambda k: (NCOMP[k], -IT['versions'][k]['m']['N100']['worst']['med']))
        verdict = 'candidate'
    else:
        chosen = max(NCOMP, key=lambda k: IT['versions'][k]['m']['N100']['worst']['med']); verdict = 'none_eligible→不可用（OOS 只作描述）'
    IT['selection'] = dict(eligible=el, chosen=chosen, verdict=verdict, rule=R0['PREREG']['selection_rule'])
    IT['written_utc'] = pd.Timestamp.now('UTC').isoformat(); IT['runtime_s'] = round(time.time() - t0, 1)
    (HERE / 'iterations.json').write_text(CL.dumps(IT), encoding='utf-8')
    NV = pd.DataFrame({f'{k[0]}|{k[1]}|{k[2]}': v['nav'] for k, v in res.items()}).astype('float32')
    NV.to_parquet(HERE / 'data/sel_navs.parquet', compression='zstd')
    print('chosen', chosen, verdict, el, round(time.time() - t0))

#!/usr/bin/env python3
"""凍結後：6 因子（時點 T、稅 X、手續費 F、跳動點 K、衝擊 I、參與上限 C）全部 64 個組合 × REAL_{opt,p90,carry} × 13 j（seed 0）× {100 萬, 30 萬} 一般成本，
正式 V7X（LogSim 開關）。輸出每個 (組合, 變體, j, 格) 的各期年化 → data/shapley_coalitions.csv.gz（複製員可由此表自行算 Shapley）。
另存「成本全關、時點開」與「全開」的 seed 0 淨值（P7 用）→ scratchpad。
python3 -I -B run_shapley.py"""
import sys, json, hashlib, time, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import tfc_lib as TL; sys.path.pop(0)
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in R and 'PREREG_FREEZE' in R, '凍結前不得執行'
assert hashlib.sha256(json.dumps(R['PREREG_FREEZE'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_FREEZE_sha256']
FAC = ['T', 'X', 'F', 'K', 'I', 'C']
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']; TIER = ['100萬', '30萬']
_G = {}
def init(close):
    if close: TL.setup_close_timing()
    E = TL.engine(); _G['E'] = E; _G['close'] = close
    P = TL.load_plans(); P = P[P.seed == 0]; _G['G'] = {k: g for k, g in P.groupby(['variant', 'j'])}
def per_stats(nav, j, S):
    e = S.end_day('AB_V7_B10', j); nav = nav.loc[:e].dropna()
    st = S.stats(nav, 'AB_V7_B10', j)
    o = dict(CAGR=st['CAGR'], CAGR_Y19on=st['CAGR_Y19on'], CAGR_SEL=st['CAGR_SEL'], CAGR_POSTODD=st['CAGR_POSTODD'])
    for k, (a, b) in {'S0612': ('2006-01-01', '2012-12-31'), 'S1318': ('2013-01-01', '2018-12-31')}.items():
        x = nav[(nav.index >= a) & (nav.index <= b)]; prev = nav[nav.index < x.index[0]]
        s0 = prev.index[-1] if len(prev) else x.index[0]; o[k] = S.cagr(nav, s0, x.index[-1])
    return o
def job(a):
    coal, v, j, tier = a
    E = _G['E']; D = E['D']; S = E['S']
    sw = dict(tax=int('X' in coal), fee=int('F' in coal), tick=int('K' in coal), imp=int('I' in coal), caps=int('C' in coal), timing='open' if 'T' in coal else 'close')
    assert (sw['timing'] == 'close') == _G['close']
    pl = TL.to_plan(_G['G'][(v, j)], D, close=_G['close'])
    nav = E['LogSim'](pl, TL.cfg_for(E, tier, 'normal'), settle=E['ST'], cmode='consideration', sw=sw).run()
    keep = nav if (tier == '100萬' and coal in ('T', 'TXFKIC', '')) else None
    return coal, v, j, tier, per_stats(nav, j, S), keep
if __name__ == '__main__':
    t0 = time.time()
    coals = [''.join(f for f, b in zip(FAC, bits) if b) for bits in itertools.product([0, 1], repeat=6)]
    rows = []; NAVS = {}
    for close in (False, True):
        jobs = [(c, v, j, t) for c in coals if (('T' not in c) == close) for v in VARS for j in range(13) for t in TIER]
        with Pool(4, initializer=init, initargs=(close,)) as P:
            for n, (c, v, j, t, st, keep) in enumerate(P.imap_unordered(job, jobs, chunksize=2)):
                rows.append(dict(coalition=c or 'none', variant=v, j=j, tier=t, **st))
                if keep is not None: NAVS[f'{c or "none"}|{v}|{j}'] = keep
                if n % 200 == 0: print(close, n, len(jobs), round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows).sort_values(['tier', 'variant', 'coalition', 'j']).reset_index(drop=True)
    f = HERE / 'data/shapley_coalitions.csv.gz'; df.to_csv(f, index=False, compression='gzip', float_format='%.8g')
    pd.DataFrame(NAVS).to_parquet(TL.SCR / 'navs_shapley_keep.parquet')
    (HERE / 'run_shapley_meta.json').write_text(json.dumps(dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), n=len(df),
                                                                file=str(f.relative_to(HERE)), sha256=hashlib.sha256(f.read_bytes()).hexdigest()), ensure_ascii=False, indent=1))
    print('done', len(df), round(time.time() - t0))

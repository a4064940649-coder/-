#!/usr/bin/env python3
"""登記重算（V7X 可執行口徑，只跑一次）：AB_V7_B10 × 13 j × 20 seeds × K5T × {一般, 壓力} × {100萬, 300萬, 1000萬} × GROW，
引擎 orig（＝現行正式 A16，須重現 navs_oos_R4.parquet）、cons_main、cons_lo、cons_hi、cons_c50。
斷言 PREREG 存在、雜湊相符、RESULTS_V7X 尚不存在。seeds 日報酬平均成一條淨值（同 x_run.run_grid）。
python3 -I -B s10_v7x.py   淨值存 scratchpad/o17_delist/navs_v7x.parquet"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cz_lib as Z
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R and hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'RESULTS_V7X' not in R, '只跑一次'
for f, h in R['PREREG']['file_sha256'].items():
    if (HERE / f).exists(): assert Z.sha(HERE / f) == h, f
import cz_v7x as CV
import cz_engine as CE
D, X = CV.D, CV.X
for k in ('x_exec.py', 'x_data.py', 'x_stats.py', 'navs_oos_R4.parquet'):
    p = CV.V7X / ('out/' + k if k.endswith('.parquet') else k)
    assert Z.sha(p) == R['PREREG']['file_sha256'][k], k
sys.path.insert(0, str(CV.V7X)); import x_stats as S; sys.path.pop(0)
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
E = pd.read_csv(HERE / 'events_v2.csv', dtype={'code': str, 'successor': str})
d = R['PREREG']['events_v2']['summary']['d']['d_used']
kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
ST = {'cons_main': CE.build_settlement(E, 'main', d=d, **kw), 'cons_lo': CE.build_settlement(E, 'lo', d=d, **kw),
      'cons_hi': CE.build_settlement(E, 'hi', d=d, **kw), 'cons_c50': CE.build_settlement(E, 'main', d=d, cash_0050=True, **kw)}
ENG = ['orig'] + list(ST)
TIERS = {'100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
CELLS = [(c, t) for c in ('normal', 'stress') for t in TIERS]


def job(args):
    key, pl = args
    out = {}; info = {}
    for c, t in CELLS:
        cfg = dict(tier=TIERS[t], mode='GROW', model='K5T', cost=c); cfg.update(BASE)
        for eg in ENG:
            sm = CV.CSim(pl, cfg, settle=ST.get(eg), cmode='orig' if eg == 'orig' else 'consideration')
            out[(c, t, eg)] = sm.run()
            if c == 'normal' and t == '100萬' and eg != 'orig':
                info[eg] = dict(n_settled=sm.acc['n_settled'], n_fallback=sm.acc['n_fallback'],
                                ev=[(e['code'], e['type'], e['date'], e['value_lastclose'], e['parts']) for e in sm.log])
    return key, out, info


if __name__ == '__main__':
    t0 = time.time()
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    jobs = sorted(P.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    navs = {}; infos = {}
    with Pool(4) as pool:
        for n, (key, out, info) in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            for ck, nv in out.items(): navs.setdefault((key[0],) + ck, []).append((key[1], nv))
            infos[f'J{key[0]}|s{key[1]}'] = info
            if n % 20 == 0: print(n, round(time.time() - t0), 's', flush=True)
    cols = {}
    for (j, c, t, eg), L in navs.items():
        L = [nv for s, nv in sorted(L, key=lambda x: x[0])]
        cols[f'B10|J{j}|K5T|{c}|{t}|GROW|{eg}'] = S.avg_nav(L)
    df = pd.DataFrame(cols).sort_index(axis=1)
    df.to_parquet(Z.WORK / 'navs_v7x.parquet')
    # 重現：orig 對 navs_oos_R4
    R4 = pd.read_parquet(CV.V7X / 'out' / 'navs_oos_R4.parquet')
    rep = {}
    for j in range(13):
        for c, t in CELLS:
            a = df[f'B10|J{j}|K5T|{c}|{t}|GROW|orig']; b = R4[f'B10|J{j}|K5T|{c}|{t}|GROW'].reindex(a.index)
            rep[f'J{j}|{c}|{t}'] = float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy())))
    # 統計
    TAB = {}
    for c, t in CELLS:
        for eg in ENG:
            per = [S.stats(df[f'B10|J{j}|K5T|{c}|{t}|GROW|{eg}'], 'AB_V7_B10', j) for j in range(13)]
            TAB[f'{c}|{t}|{eg}'] = S.jsumm(per)
    FIELDS = ['CAGR_med', 'exc_med', 't_0050_med', 'exc_p10', 'CAGR_Y19on_med', 'exc_Y19on_med', 't_Y19on_med', 'exc_Y19on_p10']
    rows = []
    for c, t in CELLS:
        for eg in ST:
            for f in FIELDS:
                dv = TAB[f'{c}|{t}|{eg}'][f] - TAB[f'{c}|{t}|orig'][f]
                lim = 0.05 if f.startswith('t_') else 0.3
                rows.append(dict(cost=c, tier=t, engine=eg, field=f, orig=TAB[f'{c}|{t}|orig'][f], cons=TAB[f'{c}|{t}|{eg}'][f], delta=dv, limit=lim, exceed=bool(abs(dv) > lim)))
    worst = max(rows, key=lambda r: abs(r['delta']) / r['limit'])
    # 年度差（描述）：100 萬一般，cons_main − orig，j 中位數
    yd = {}
    for eg in ST:
        Y = pd.DataFrame({j: pd.Series(S.yearly_exc(df[f'B10|J{j}|K5T|normal|100萬|GROW|{eg}'], 'AB_V7_B10', j)) - pd.Series(S.yearly_exc(df[f'B10|J{j}|K5T|normal|100萬|GROW|orig'], 'AB_V7_B10', j)) for j in range(13)})
        yd[eg] = {int(y): float(v) for y, v in Y.median(axis=1).items()}
    R = json.loads(RJ.read_text(encoding='utf-8'))
    assert 'RESULTS_V7X' not in R
    R['RESULTS_V7X'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), reproduce_R4_maxabs=rep,
                            reproduce_R4_max=max(rep.values()), table=TAB, judgement=dict(rows=rows, annotate=any(r['exceed'] for r in rows), worst=worst),
                            yearly_delta_med_100m_normal=yd, settle_info=infos,
                            navs_file='scratchpad/o17_delist/navs_v7x.parquet', navs_sha256=Z.sha(Z.WORK / 'navs_v7x.parquet'))
    RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('reproduce max', max(rep.values()), 'annotate', R['RESULTS_V7X']['judgement']['annotate'], worst, round(time.time() - t0), 's')

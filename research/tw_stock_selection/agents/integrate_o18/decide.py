#!/usr/bin/env python3
"""integrate_o18 判定（只跑一次；規則全部寫在 PREREG.decision，這裡只是機械套用）。讀 RESULTS_PAPER、RESULTS_V7X，寫 DECISION。
python3 -I -B decide.py"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8')); PR = R['PREREG']
assert hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'RESULTS_PAPER' in R and 'RESULTS_V7X' in R and 'DECISION' not in R
SP, TX = R['RESULTS_PAPER']['stats'], R['RESULTS_V7X']['table']
ok_pipe = max(R['RESULTS_PAPER']['checks'].values()) < 1e-9 and max(R['RESULTS_V7X']['reproduce_T1_maxabs'].values()) < 1e-9
def P(k, c, eg, ref='vs_C0|freeze'): return SP[f'{k}|{c}|{eg}|{ref}']
def pnum(k, c, eg):
    s = P(k, c, eg); return dict(ann=s['ALL']['ann'], t=s['ALL']['t_vs_0050'], seg1=s['SEG1']['gex'], seg2=s['SEG2']['gex'], seg3=s['SEG3']['gex'],
                                 hold=s['HOLD']['gex'], mdd_win=s['ALL']['mdd'], mdd_daily=s['ALL']['mdd_daily'], t_drop_best3=s['ALL']['t_drop_best3'],
                                 diff_vs_C0_pp=s['ALL']['diff_vs_ref_pp'], t_diff_vs_C0=s['ALL']['t_diff_vs_ref'], corr_C0=s['ALL']['corr_ref'], corr_0050=s['ALL']['corr_0050'])
def vnum(v, eg, c, t):
    T = TX[f'{v}|{eg}|{c}|{t}']; return dict(exc_med=T['exc_med'], exc_p10=T['exc_p10'], t_med=T['t_0050_med'], exc_Y19on_med=T['exc_Y19on_med'],
                                             t_Y19on_med=T['t_Y19on_med'], exc_S1318_med=T['exc_S1318_med'], exc_S0612_med=T['exc_S0612_med'], CAGR_med=T['CAGR_med'], maxDD_med=T['maxDD_med'])
NEW = {'paper_normal': pnum('ALLPIT_REAL_opt', 'normal', 'v3_main'), 'paper_stress': pnum('ALLPIT_REAL_opt', 'stress', 'v3_main'),
       'v7x': {f'{c}|{t}': vnum('REAL_opt', 'v3_main', c, t) for c in ('normal', 'stress') for t in ('100萬', '300萬', '1000萬')}}
T1 = {'paper_normal': pnum('ALLPIT_REAL_opt', 'normal', 'cons_main'), 'paper_stress': pnum('ALLPIT_REAL_opt', 'stress', 'cons_main'),
      'v7x': {f'{c}|{t}': vnum('REAL_opt', 'cons_main', c, t) for c in ('normal', 'stress') for t in ('100萬', '300萬', '1000萬')}}
# D1：v3 對 cons_main（題 1 新正式數字）的變動
dl = []
for f in ('ann', 'seg1', 'seg2', 'seg3'): dl.append((f'paper_normal.{f}', NEW['paper_normal'][f] - T1['paper_normal'][f], 0.3))
dl.append(('paper_normal.t', NEW['paper_normal']['t'] - T1['paper_normal']['t'], 0.05))
dl.append(('paper_stress.ann', NEW['paper_stress']['ann'] - T1['paper_stress']['ann'], 0.3))
for k, f in (('normal|100萬', 'exc_med'), ('normal|100萬', 'exc_p10'), ('normal|100萬', 'exc_Y19on_med'), ('stress|100萬', 'exc_med'), ('normal|300萬', 'exc_med'), ('normal|1000萬', 'exc_med')):
    dl.append((f'v7x.{k}.{f}', NEW['v7x'][k][f] - T1['v7x'][k][f], 0.3))
dl.append(('v7x.normal|100萬.t_med', NEW['v7x']['normal|100萬']['t_med'] - T1['v7x']['normal|100萬']['t_med'], 0.05))
trig = [(n, d) for n, d, th in dl if abs(d) > th]
D1 = dict(deltas={n: d for n, d, _ in dl}, triggered=trig,
          verdict=('取代：正式數字改用 v3_main 版（ALLPIT_REAL_opt × v3_main；V7X REAL_opt × v3_main）' if trig else '不取代：題 1 新正式數字（cons_main）維持；旁加註 v3 對價差'))
# D2：V7X 對價界限
lo, hi = TX['REAL_opt|v3_lo|normal|100萬']['exc_med'], TX['REAL_opt|v3_hi|normal|100萬']['exc_med']
D2 = dict(lo=lo, hi=hi, main=TX['REAL_opt|v3_main|normal|100萬']['exc_med'], both_pos=bool(lo > 0 and hi > 0),
          verdict=('加註：對價界限兩端 > 0' if lo > 0 and hi > 0 else '加註：對價界限含 ≤ 0，寫出界限與決定界限的事件'))
# D3：G4 第二段
s2n, s2s = NEW['paper_normal']['seg2'], NEW['paper_stress']['seg2']
seg2_iv = {f2: P(f'ALLPIT_REAL_{f2}', 'normal', 'v3_main')['SEG2']['gex'] for f2 in ('opt', 'p90', 'carry')}
D3 = dict(seg2_normal=s2n, seg2_stress=s2s, seg2_interval=seg2_iv,
          verdict=('G4 第二段：可能翻號（整合主口徑一般或壓力成本 ≤ 0）' if (s2n <= 0 or s2s <= 0) else 'G4 第二段：整合主口徑 > 0，但覆蓋 < 80%，維持「存活者偏誤未排除」'))
# D6：敏感度
def dd(k, c='normal'): return P(k, c, 'v3_main', 'vs_ALLPIT_REAL_opt|v3_main')['ALL']['ann_diff_vs_ref']
D6 = dict(DSPR=dd('ALLPIT_REAL_opt_DSPR'), KYX=dd('ALLPIT_REAL_opt_KYX'))
D6['verdict'] = {k: ('加註（> 0.3pp）' if abs(v) > 0.3 else '可忽略') for k, v in D6.items() if k != 'verdict'}
IVP = {f: pnum(f'ALLPIT_REAL_{f}', 'normal', 'v3_main') for f in ('opt', 'p90', 'carry')}
IVV = {f: vnum(f'REAL_{f}', 'v3_main', 'normal', '100萬') for f in ('opt', 'p90', 'carry')}
G = PR['decision']['G_table_rule']
R = json.loads(RJ.read_text(encoding='utf-8')); assert 'DECISION' not in R
R['DECISION'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), pipeline_ok=ok_pipe, NEW=NEW, T1=T1, D1=D1, D2=D2, D3=D3, D6=D6, interval_paper_v3=IVP, interval_v7x_v3=IVV,
                     grade='不可投資（G1、G2、G3、G4、G5、G6、G7、G8 皆未過）', G_rule=G)
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(R['DECISION'], ensure_ascii=False, indent=1)[:6000])

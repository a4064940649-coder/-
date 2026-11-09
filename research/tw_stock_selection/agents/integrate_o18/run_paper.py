#!/usr/bin/env python3
"""integrate_o18 正式執行：紙上 ia 口徑（只跑一次）。斷言 PREREG 存在、雜湊相符、RESULTS_PAPER 尚不存在、輸入 sha 相符。
變體與引擎見 PREREG.paper；統計＝題 1 o18lib.win()／stats_pair()（訊號日收盤→下一訊號日收盤窗，對 0050 含息）。
python3 -I -B run_paper.py（在 scratchpad 執行）"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
RJ = HERE / 'results.json'
T0 = time.time()
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
R = json.loads(RJ.read_text(encoding='utf-8'))
PR = R['PREREG']
assert hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改'
assert 'RESULTS_PAPER' not in R, '只跑一次'
for f, h in PR['file_sha256'].items(): assert sha(f if f.startswith('/') else HERE / f) == h, f'{f} 雜湊不符'
sys.path.insert(0, str(HERE)); import ilib as I; sys.path.pop(0)
O = I.O
VARS = PR['paper']['variants']; ENG = PR['paper']['engines']
B = {k: I.build(v['kind'], excl=tuple(v.get('excl', [])), dsp_mode=v.get('dsp_mode', 'union')) for k, v in VARS.items()}
LSH = {k: I.lists_sha(b) for k, b in B.items()}
SM = json.loads((HERE / 'smoke.json').read_text(encoding='utf-8'))
for k in B: assert LSH[k] == SM['lists_sha'][k], ('名單與煙霧測試不同', k)
SD0 = B['C0']['sd']; assert all(B[k]['sd'] == SD0 for k in B)
log('名單', {k: len(b['plan']) for k, b in B.items()})
NAV, LOGS = {}, {}
for k, egs in ENG.items():
    for eg, costs in egs.items():
        for c in costs:
            nav, lg = I.sim(B[k], c, eg); NAV[(k, c, eg)] = nav
            if c == 'normal':
                LOGS[f'{k}|{eg}'] = dict(n_frozen=len(lg.get('frozen', [])), frozen_max_share=max([x[2] for x in lg.get('frozen', [])] or [0.0]),
                                         settled=[(s['code'], s['type'], s['date'], round(s['value_lastclose'], 6)) for s in lg.get('settled', [])])
    log('模擬', k)
OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')
CHK = {}
for c in ('normal', 'stress'):
    ref = OFF[f'EXFIN|n10|{c}'].dropna(); nav = NAV[('C0', c, 'freeze')]
    CHK[f'C0_{c}'] = float(np.nanmax(np.abs(nav.reindex(ref.index).to_numpy() - ref.to_numpy())))
assert max(CHK.values()) < 1e-12, CHK
WIN = {key: O.win(nav, SD0) for key, nav in NAV.items()}
R1 = json.loads((I.T1 / 'results.json').read_text(encoding='utf-8'))['RESULTS_PAPER']['stats']
ST = {}
for (k, c, eg), W in WIN.items():
    refs = [('C0', c, 'freeze')]                                                   # 主策略 C0（現行紙上正式）
    if k != 'C0': refs.append(('ALLPIT_REAL_opt', c, 'cons_main'))                 # 題 1 新正式數字
    if (k, c, 'cons_main') in WIN and eg != 'cons_main': refs.append((k, c, 'cons_main'))   # 同名單、引擎 v3 對 v2
    if k in ('ALLPIT_REAL_opt_KYX', 'ALLPIT_REAL_opt_DSPR'): refs.append(('ALLPIT_REAL_opt', c, 'v3_main'))   # 敏感度對整合主口徑
    for rk in dict.fromkeys(r for r in refs if r in WIN and r != (k, c, eg)):
        s = O.stats_pair(W, WIN[rk]); nav = NAV[(k, c, eg)]
        s['ALL']['mdd_daily'] = float((nav / nav.cummax() - 1).min() * 100)
        ST[f'{k}|{c}|{eg}|vs_{rk[0]}|{rk[2]}'] = s
# 重現題 1 正式數字（ALLPIT_REAL_*、cons_main）
for k in ('ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'):
    for c in ('normal', 'stress'):
        if (k, c, 'cons_main') not in WIN: continue
        a = ST[f'{k}|{c}|cons_main|vs_C0|freeze']; b = R1[f'{k}|{c}|cons_main|vs_C0']
        CHK[f'T1repro|{k}|{c}'] = max(abs(a[p][f] - b[p][f]) for p in ('ALL', 'SEG1', 'SEG2', 'SEG3', 'HOLD') for f in ('ann', 'gex', 't_vs_0050', 'diff_vs_ref_pp'))
log('統計', CHK)
R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS_PAPER' not in R
R['RESULTS_PAPER'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - T0, 1), checks=CHK, lists_sha=LSH, logs=LOGS, stats=ST)
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame({f'{k}|{c}|{eg}': v for (k, c, eg), v in NAV.items()}).to_parquet(I.SCR / 'navs_paper_i18.parquet')
for key, s in ST.items():
    if '|vs_C0|' in key:
        a = s['ALL']; print(key, 'ann %.2f t %.3f seg %.2f/%.2f/%.2f hold %.2f dC0 %.2f' % (a['ann'], a['t_vs_0050'], s['SEG1']['gex'], s['SEG2']['gex'], s['SEG3']['gex'], s['HOLD']['gex'], a['ann_diff_vs_ref']))
log('寫入 RESULTS_PAPER')

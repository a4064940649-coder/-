# K：以原統計模組（x_stats，只讀 import）彙整原引擎重跑結果 → sealed_targets.json（含 o18 正式表對照）；密封檔 sha 公開
import sys, json, pickle, hashlib
import numpy as np, pandas as pd
AG = '/home/user/-/research/tw_stock_selection/agents/'
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
sys.path.insert(0, AG + 'o_exec_engine_rewrite_v7x'); import x_stats as S
KP = SCR + 'g6_sealed/k_orig_engine.pkl'
KO = pickle.load(open(KP, 'rb'))
OFF = json.load(open(AG + 'o_delist_rev_backfill_allpit/results.json'))['RESULTS_V7X']['table']
CM = {'N100': 'normal|100萬', 'S100': 'stress|100萬', 'N300': 'normal|300萬', 'N1000': 'normal|1000萬', 'O30': 'normal|30萬'}
out = {'sealed_file': KP, 'sealed_sha256': hashlib.sha256(open(KP, 'rb').read()).hexdigest(), 'cells': {}}
for v in ['orig', 'REAL_opt', 'REAL_p90', 'REAL_carry']:
    eg = 'orig' if v == 'orig' else 'cons_main'
    for cn in CM:
        if (v, 0, cn) not in KO['avg']: continue
        per = [S.stats(KO['avg'][(v, j, cn)], 'AB_V7_B10', j) for j in range(13)]
        T = S.jsumm(per)
        rec = {k: T[k] for k in ('exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 'CAGR_Y19on_med', 'E50_CAGR_Y19on_med', 'CAGR_med')}
        ok = OFF.get(f'{v}|{eg}|{CM[cn]}')
        if ok: rec['o18_official_exc_med'] = ok['exc_med']; rec['diff_vs_o18'] = T['exc_med'] - ok['exc_med']
        out['cells'][f'{v}|{eg}|{cn}'] = rec
json.dump(out, open(SCR + 'g6_sealed/sealed_targets.json', 'w'), ensure_ascii=False, indent=1, default=float)
pub = {'sealed_targets_sha256': hashlib.sha256(open(SCR + 'g6_sealed/sealed_targets.json', 'rb').read()).hexdigest(), 'k_orig_engine_sha256': out['sealed_sha256'],
       'max_abs_diff_vs_o18_official_exc_pp': max(abs(r['diff_vs_o18']) for r in out['cells'].values() if 'diff_vs_o18' in r)}
json.dump(pub, open('/home/user/-/research/tw_stock_selection/agents/o_g6_v7x_cleanroom_rewrite/sealed_targets_public.json', 'w'), indent=1)
print(pub)
for k, r in out['cells'].items(): print(k, {kk: round(vv, 3) for kk, vv in r.items()})

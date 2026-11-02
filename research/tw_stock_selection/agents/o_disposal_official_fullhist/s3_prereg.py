#!/usr/bin/env python3
"""第 1 步（後半）凍結：在看 2006～2021-09 任何報酬之前，寫 PREREG（選定版本、執行參數、全部腳本與輸入 sha256、時間戳）。
已有 PREREG 就拒絕覆寫。"""
import sys, json, datetime
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as K

res = json.loads(K.RES_F.read_text(encoding='utf-8'))
assert 'PLAN' in res and 'PREREG' not in res, 'PREREG 已存在或 PLAN 缺'
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
sel = it['selected']; cfg = it['versions'][sel]['cfg']
files = ['core.py', 's0_gate.py', 's1_plan.py', 's2_explore.py', 's2b_iterate.py', 's3_prereg.py', 's4_oos.py', 'frozen_spec.txt',
         'data_gate.json', 'explore.json', 'iterations.json', 'universe_flags.csv',
         str(K.LIST_F), str(K.PROXY_F), str(K.D.CACHE), str(K.TAIEX_F), str(K.E50_F),
         str(K.AG / 'o_v7_listing_exec_engine' / 'data' / 'model_spec7.json'),
         str(K.AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet'),
         str(K.AG / 'o_main_spa_breaks' / 'data' / 'family_excess.parquet'), str(K.AG / 'o_main_spa_breaks' / 'data' / 'family_members.csv'),
         str(K.AG / 'o_main_spa_breaks' / 'spa_lib.py'), str(K.AG / 'o_main_spa_breaks' / 'fam_common.py')]
PREREG = {
    'time_utc': datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(timespec='seconds') + 'Z',
    'statement': '至此只看過：資料閘門（無報酬）、挑選期 2021-10～2023 探索與 6 版迭代、乾跑（全部期間換成挑選期）。2006～2021-09、2004～2006-03、2024～2026 的本題報酬都還沒算。',
    'selected_version': sel, 'n_versions_tried': it['n_versions'], 'selection_criterion': it['criterion'],
    'exec': {'cap': cfg['cap'], 'min_fill': cfg['min_fill'], 'defer': cfg['defer'], 'first_only': bool(cfg['build'].get('first_only', False))},
    'frozen_spec_file': 'frozen_spec.txt',
    'prior_prediction_by_researcher': 'P1 最可能不成立（近似名單長歷史 t −0.89）；本題為結案檢定',
    'hashes': {f: K.sha(f if f.startswith('/') else HERE / f) for f in files},
}
res['PREREG'] = PREREG
res['PREREG_sha256'] = K.canon_sha(PREREG)
K.RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('PREREG', PREREG['time_utc'], res['PREREG_sha256'])

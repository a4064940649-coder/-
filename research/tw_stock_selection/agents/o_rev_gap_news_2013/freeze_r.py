#!/usr/bin/env python3
"""步驟 4 凍結：data_freeze_r.json（代理表輸入、解析器輸出、來源 sha256、全部腳本與引擎庫）。在 gate.py 之前執行。"""
import json, hashlib, glob, subprocess
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def concat(files):
    h = hashlib.sha256()
    for f in files: h.update(Path(f).read_bytes())
    return h.hexdigest()
v4 = json.loads((AG / 'o_v4_revenue_patch_official' / 'data_freeze_v4.json').read_text(encoding='utf-8'))
fz = {'timestamp_utc': pd.Timestamp.utcnow().isoformat(), 'note': '凍結後才計算正式閘門（gate.py 會斷言本檔與 PREREG 存在且未被改）。資料一律用 V4 凍結版。',
      'data_freeze_v4_sha256': sha(AG / 'o_v4_revenue_patch_official' / 'data_freeze_v4.json'), 'files': {}, 'raw_sources': {}}
files = {}
for k, v in v4['v3_base'].items(): files[f'v4.{k}'] = v['path']
for k, v in v4['panel'].items(): files[f'v4.{k}'] = v['path']
files['v4.R_FIX'] = str(AG / 'o_v4_revenue_patch_official' / v4['R_FIX']['file'])
if v4.get('Y5_FIX'): files['v4.Y5_FIX'] = str(AG / 'o_v4_revenue_patch_official' / v4['Y5_FIX']['file'])
for f in ('raw_quarterly.parquet', 'plan_V4.csv', 'plan_V4_B10.csv', 'navs_v4.parquet', 'v4lib.py'):
    files[f'v4.{f}'] = str(AG / 'o_v4_revenue_patch_official' / f)
for rel in ('o_infra_v3_rebuild/o3_core.py', 'o_disposal_rejudge/drj_core.py', 'main_v2/v2_core.py', 'independent_audit/ia_core.py',
            'o_infra_v3_rebuild/merger_consideration.csv', 'o_infra_v3_rebuild/delist_official.csv', 'rev_announce/p_delay.json'):
    files[rel] = str(AG / rel)
files['tw_revenue_long'] = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
files['rev_5y'] = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz'
files['taiex_calendar'] = '/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv'
for f in sorted(HERE.glob('*.py')): files[f'topic.{f.name}'] = str(f)
for f in ('gap_cells_2013_2021.csv', 'sources_log.json', 'iterations.json', 'explore.json'):
    files[f'topic.{f}'] = str(HERE / f)
for f in sorted((HERE / 'data').glob('mentions_P*')): files[f'topic.data/{f.name}'] = str(f)
for k, p in files.items(): fz['files'][k] = {'path': p, 'sha256': sha(p)}
CT = AG / 'o_v4_revenue_patch_official' / 'raw/ctbc/AI_Investment/OpenIEStanza_EventEmbedding/input'
cf = sorted(glob.glob(str(CT / 'EB_EventExtraction2' / '*.csv')) + glob.glob(str(CT / 'EB_SegAndRawNews' / '*.csv')))
bf = sorted(glob.glob(str(HERE / 'raw/bofrankbo_LLM-Predict-Stock/history_data/tw/stock_news/2311/2311_news/*')))
ef = [str(HERE / 'raw/roccqqck_stock_eland/5pa_article.csv')]
pf = sorted(f for f in glob.glob(str(HERE / 'raw/ji08kk44_BDSE26_TEAM3/ptt/ptt_csv(date)/*.csv')) if Path(f).stem <= '2021-12-31')
for tag, fl, repo in (('CTBC', cf, AG / 'o_v4_revenue_patch_official/raw/ctbc'), ('BOF', bf, HERE / 'raw/bofrankbo_LLM-Predict-Stock'),
                      ('ELD', ef, HERE / 'raw/roccqqck_stock_eland'), ('PTT', pf, HERE / 'raw/ji08kk44_BDSE26_TEAM3')):
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    fz['raw_sources'][tag] = {'repo_head': head, 'n_files': len(fl), 'files_sha256_concat': concat(fl), 'file_list_sha256': hashlib.sha256('\n'.join(Path(f).name for f in fl).encode()).hexdigest()}
(HERE / 'data_freeze_r.json').write_text(json.dumps(fz, ensure_ascii=False, indent=1), encoding='utf-8')
print(len(fz['files']), {k: v['n_files'] for k, v in fz['raw_sources'].items()})

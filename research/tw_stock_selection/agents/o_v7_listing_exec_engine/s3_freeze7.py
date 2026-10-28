#!/usr/bin/env python3
"""閘門通過後凍結 V7 資料：data_freeze_v7.json（每檔 sha256）＋ data/raw_manifest.tsv（raw/ 每個檔案的 sha256）。"""
import sys, json, datetime, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7common as C
res = C.load_res()
assert res.get('GATE', {}).get('PASS'), '閘門沒過：不得凍結'
lines = []
for p in sorted(C.RAW.rglob('*')):
    if p.is_file() and '/.git/' not in str(p):
        lines.append(f'{p.relative_to(C.HERE)}\t{C.sha(p)}')
(C.DATA / 'raw_manifest.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
def ent(p): return {'path': str(p), 'sha256': C.sha(p)}
FIELDS = C.FIELDS + ['low']
fz = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
      'note': 'V7＝V6 面板在替換窗 2024-01-02～2025-09-23 改用全市場上市身分（listing_master）、颱風停市日事件順延、拿掉 2024-01 前視；營收、處置、下市、併購沿用 V4 凍結版。在任何 2024 年以後的策略報酬之前凍結。',
      'panel_version': C.jload(C.DATA / 'panel_v7_meta.json')['version'],
      'v7_panel': {k: ent(C.DATA / f'{k}.parquet') for k in FIELDS},
      'v7_meta': ent(C.DATA / 'panel_v7_meta.json'),
      'listing': {'listing_master.csv': ent(C.HERE / 'listing_master.csv'), 'listed_twse.parquet': ent(C.DATA / 'listed_twse.parquet'),
                  'listed_twse_strong.parquet': ent(C.DATA / 'listed_twse_strong.parquet'), 'listing_manual.json': ent(C.HERE / 'listing_manual.json'),
                  'listing_review.json': ent(C.DATA / 'listing_review.json')},
      'parsed_sources': {k: ent(C.DATA / f'src_{k}.parquet') for k in ('hschi', 'tasi', 'bacardi', 'bacardi_tpex', 'leochen', 'history', 'tasi_pbr', 'isin', 'tasi_rev')},
      'gate_population': ent(C.DATA / 'gate_population.parquet'),
      'model_spec7': ent(C.DATA / 'model_spec7.json'),
      'raw_manifest': ent(C.DATA / 'raw_manifest.tsv'), 'n_raw_files': len(lines),
      'raw_commits': {d.name: (d / 'COMMIT.txt').read_text().strip() if (d / 'COMMIT.txt').exists() else None for d in C.RAW.iterdir() if d.is_dir()},
      'tasi_commit': subprocess.run(['git', '-C', str(C.RAW / 'tasi'), 'log', '-1', '--format=%H %cd'], capture_output=True, text=True).stdout.strip(),
      'leochen_commit': '0f319a125cb2e06b9a92e3688bb76baf5c759fb3 Tue Aug 19 17:04:10 2025 +0800（o_v6 raw 副本）',
      'external': {'data_freeze_v6': ent(C.V6DIR / 'data_freeze_v6.json'), 'v6_panel': {k: ent(C.V6DATA / f'{k}.parquet') for k in FIELDS},
                   'navs_v6': ent(C.V6DIR / 'navs_v6.parquet'), 'navs_jscan_v6': ent(C.V6DIR / 'navs_jscan_v6.parquet'), 'plan_V6': ent(C.V6DIR / 'plan_V6.csv'),
                   'data_freeze_v4': ent(C.V4DIR / 'data_freeze_v4.json'), 'navs_v4': ent(C.V4DIR / 'navs_v4.parquet'), 'plan_V4': ent(C.V4DIR / 'plan_V4.csv'),
                   'v4_panel_scratchpad': {k: ent(C.PANEL4 / f'{k}.parquet') for k in C.FIELDS}, 'etf_0050': ent(C.E50F), 'taiex': ent(C.TAIEX),
                   'delist_official': ent(C.AG / 'o_infra_v3_rebuild' / 'delist_official.csv'), 'merger_consideration': ent(C.AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv')},
      'gate': {k: res['GATE'][k] for k in ('checks', 'PASS')}}
C.jdump(fz, C.HERE / 'data_freeze_v7.json')
print('frozen', fz['timestamp'], fz['n_raw_files'], 'raw files')

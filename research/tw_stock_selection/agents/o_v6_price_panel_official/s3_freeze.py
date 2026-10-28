#!/usr/bin/env python3
"""閘門通過後凍結 V6 資料：data_freeze_v6.json（每檔 sha256）＋ data/raw_manifest.tsv（raw/ 每個檔案的 sha256）。"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6common as C
res = C.load_res()
assert res.get('GATE', {}).get('PASS'), '閘門沒過：不得凍結'
lines = []
for p in sorted(C.RAW.rglob('*')):
    if p.is_file() and '/.git/' not in str(p):
        lines.append(f'{p.relative_to(C.HERE)}\t{C.sha(p)}')
(C.DATA / 'raw_manifest.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
def ent(p): return {'path': str(p), 'sha256': C.sha(p)}
fz = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
      'note': '在任何 2024 年以後的策略報酬之前凍結。V6＝V4 面板（scratchpad/panel，見 data_freeze_v4）在 2024-01-02～2025-09-23 換成官方日 K；營收、處置、下市、併購沿用 V4 凍結版。',
      'panel_version': json.loads((C.DATA / 'panel_v6_meta.json').read_text(encoding='utf-8'))['version'],
      'v6_panel': {k: ent(C.DATA / f'{k}.parquet') for k in C.FIELDS + ['low']},
      'v6_meta': ent(C.DATA / 'panel_v6_meta.json'),
      'parsed_sources': {k: ent(C.DATA / f'src_{k}.parquet') for k in ('hschi', 'tasi', 'bacardi', 'leochen', 'henry', 'history', 'hawan')},
      'raw_manifest': ent(C.DATA / 'raw_manifest.tsv'), 'n_raw_files': len(lines),
      'raw_commits': {d.name: (d / 'COMMIT.txt').read_text().strip() if (d / 'COMMIT.txt').exists() else None for d in C.RAW.iterdir() if d.is_dir()},
      'tasi_commit': __import__('subprocess').run(['git', '-C', str(C.RAW / 'tasi'), 'log', '-1', '--format=%H %cd'], capture_output=True, text=True).stdout.strip(),
      'leochen_commit': __import__('subprocess').run(['git', '-C', str(C.RAW / 'leochen1_stock_v'), 'log', '-1', '--format=%H %cd'], capture_output=True, text=True).stdout.strip(),
      'external': {'exdiv_5y': ent(C.XTR / 'exdiv_5y.csv.gz'), 'corp_actions': ent(C.XTR / 'corp_actions.csv'), 'shares_hist': ent(C.XTR / 'shares_hist.csv'),
                   'ex_tsmc': ent(C.XTR / 'ex_tsmc.csv'), 'etf_0050': ent(C.E50F), 'taiex': ent(C.TAIEX),
                   'data_freeze_v4': ent(C.V4DIR / 'data_freeze_v4.json'), 'navs_v4': ent(C.V4DIR / 'navs_v4.parquet'), 'plan_V4': ent(C.V4DIR / 'plan_V4.csv'),
                   'v4_panel_scratchpad': {k: ent(C.PANEL4 / f'{k}.parquet') for k in C.FIELDS}},
      'gate': {k: res['GATE'][k] for k in ('checks', 'PASS')}}
C.jdump(fz, C.HERE / 'data_freeze_v6.json')
print('frozen', fz['timestamp'], fz['n_raw_files'], 'raw files')

#!/usr/bin/env python3
"""第 1 步：把閘門需要的子集從 scratchpad 的 clone 複製（或精簡轉存）到 raw/，manifest.json 記來源、commit、sha256。
不可信資料：pickle 只用白名單載入器（safepkl.py）。"""
import sys, os, json, hashlib, glob, gzip, io, shutil, datetime
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import safepkl
SP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/opm2')
RAW = HERE / 'raw'; RAW.mkdir(exist_ok=True)
def sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()
man = {'created_utc': datetime.datetime.utcnow().isoformat() + 'Z', 'note': '原始資料子集；不可信。精簡轉存者同時記原檔 sha256。', 'sources': {}}
# A. FinBrain-tw/twstock_database：融資增減（margin_buy）、融券增減（margin_sell），2007-04-23～2021-07-09，寬表
fb = {}
for nm in ['margin_buy', 'margin_sell']:
    p = SP / 'peek' / f'fb_{nm}.pkl'
    df = safepkl.load(p)
    df.columns = df.columns.astype(str)
    out = RAW / f'finbrain_{nm}.parquet'; df.to_parquet(out)
    fb[nm] = {'orig_path': f'history/{nm}.pkl', 'orig_sha256': sha(p), 'raw_file': out.name, 'raw_sha256': sha(out), 'shape': list(df.shape), 'first': str(df.index.min().date()), 'last': str(df.index.max().date())}
man['sources']['FinBrain-tw/twstock_database'] = {'url': 'https://github.com/FinBrain-tw/twstock_database', 'commit': 'a41851b3200d7873505d7e23562953c8a7af588e', 'commit_time': '2021-07-16T13:51:37+08:00', 'files': fb,
    'semantics': 'margin_buy＝融資餘額日增減（張），margin_sell＝融券餘額日增減（張）；經與 TEJ 逐檔融資餘額差分核對（2330 2007-04-23 −1565＝−1565）'}
# B. Yuan-Hsi/SALL_Agency：TEJ 逐檔 CSV（stock_0406，存活股），抽 2005 起融資融券欄位
p = SP / 'yh_margin.parquet'; out = RAW / 'yh_tej_margin_2005_2024.parquet'; shutil.copy(p, out)
man['sources']['Yuan-Hsi/SALL_Agency'] = {'url': 'https://github.com/Yuan-Hsi/SALL_Agency', 'commit': '715cc770c2592ced0ddd605c1820db42a892ff96', 'commit_time': '2024-07-22T13:24:49+08:00',
    'orig_path': 'RL/File_Repository/stock_0406/<code>.csv（只取 4 碼非 00 開頭）', 'raw_file': out.name, 'raw_sha256': sha(out), 'extract': 's1_extract_yh.py',
    'license_note': 'TEJ 付費資料庫匯出檔被轉貼在公開 repo，授權不明；只當交叉核對與存活股錨點用'}
# D. joeycchen72/taiwan-market-history：官方 MI_MARGN、MI_QFIIS 月形成日快照（原檔照抄）＋ MI_INDEX 日檔精簡（date,code,close,value）
J = SP / 'clones' / 'joeycchen72_taiwan-market-history'
jd = RAW / 'joeyc'; jd.mkdir(exist_ok=True)
files = {}
for kind in ['margin-twse', 'qfii-twse']:
    for f in sorted(glob.glob(str(J / 'data' / 'b' / kind / '*' / '*.csv.gz'))):
        dst = jd / kind / Path(f).name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, dst)
        files[f'{kind}/{Path(f).name}'] = sha(dst)
rows = []; idx_sha = {}
for f in sorted(glob.glob(str(J / 'data' / 'twse' / '*' / '*.csv.gz'))):
    d = pd.read_csv(f, dtype=str, usecols=['code', 'close', 'value'])
    d = d[d.code.str.fullmatch(r'\d{4}')]
    d['date'] = pd.to_datetime(Path(f).name[:8]); rows.append(d); idx_sha[Path(f).name] = sha(f)
mi = pd.concat(rows, ignore_index=True)
mi['close'] = pd.to_numeric(mi.close, errors='coerce'); mi['value'] = pd.to_numeric(mi.value, errors='coerce')
out = RAW / 'joeyc_mi_index_4digit_2014_2026.parquet'; mi.to_parquet(out, index=False)
man['sources']['joeycchen72/taiwan-market-history'] = {'url': 'https://github.com/joeycchen72/taiwan-market-history', 'commit': '0a270d50033d45ccfdb9c5bffac1a75d10e9cdc9', 'commit_time': '2026-11-04T23:04:28+00:00',
    'endpoints': {'margin-twse': 'twse rwd MI_MARGN?date=YYYYMMDD&selectType=ALL', 'qfii-twse': 'twse rwd MI_QFIIS?date=YYYYMMDD&selectType=ALLBUT0999', 'twse': 'MI_INDEX 每日收盤行情'},
    'sampling': '每月形成日（11 日當天或之後第一個交易日），2014-01～2026-10；GitHub Actions 2026-10/11 回溯抓取',
    'raw_files_sha256': files, 'mi_index_compact': {'raw_file': out.name, 'raw_sha256': sha(out), 'n_files': len(idx_sha), 'orig_sha256_per_file': idx_sha}}
json.dump(man, open(RAW / 'manifest.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: (v.get('raw_file') or list(v.get('files', {}).keys())) for k, v in man['sources'].items()}, ensure_ascii=False))

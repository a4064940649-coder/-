#!/usr/bin/env python3
"""下載 voidful/tw_stocker 的 Yahoo 5 分 K（2024-02～2026-04），只取曾經入選（V4 或 AB 任一 seed）的股票。
下載內容視為不可信資料：只用 pandas 讀 CSV，不執行任何內容。逐檔驗證 sha256 與 manifest 相同，
去掉重複時間戳（value_fix 已知 bug），只保留 09:00～09:25 的 6 根 K 與每日彙總（總量、根數、13:20 收盤），
存成 raw/tw_stocker_head.parquet、raw/tw_stocker_daily.parquet，原檔刪除（1.8GB 不放進研究資料夾）；
每檔原檔 sha256 與驗證結果記在 raw/tw_stocker_download_log.json。
用法：python3 -I -B fetch_5m.py
"""
import json, subprocess, tempfile, hashlib, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
HERE = Path(__file__).resolve().parent
MAN = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/data_ext/tw_stocker_manifest.json')
m = json.load(open(MAN))
F = {x['ticker']: x for x in m['files'] if x['frequency'] == 'intraday'}
p = pd.concat([pd.read_csv(HERE / 'data' / 'plans_ab.csv', dtype=str), pd.read_csv(HERE / 'data' / 'plan_v4.csv', dtype=str)])
codes = sorted({c for x in p.picks for c in x.split(',')} & set(F))
HM = ['09:00', '09:05', '09:10', '09:15', '09:20', '09:25']

def one(code):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / 'x.csv'
        r = subprocess.run(['curl', '-sS', '-f', '-m', '600', '-o', str(tmp), f'https://raw.githubusercontent.com/voidful/tw_stocker/main/data/{code}.csv'], capture_output=True)
        if r.returncode != 0 or not tmp.exists(): return code, {'status': 'fail', 'err': r.stderr.decode()[:200]}, None, None
        h = hashlib.sha256(tmp.read_bytes()).hexdigest()
        d = pd.read_csv(tmp)
    info = {'status': 'ok', 'sha256': h, 'manifest_sha256': F[code]['sha256'], 'sha_match': h == F[code]['sha256'], 'rows_raw': int(len(d))}
    d['ts'] = pd.to_datetime(d['Datetime'], utc=True).dt.tz_convert('Asia/Taipei')
    nd = int(d.duplicated('ts').sum()); d = d.drop_duplicates('ts', keep='first').sort_values('ts')
    info['dup_ts_dropped'] = nd
    d['date'] = d['ts'].dt.strftime('%Y-%m-%d'); d['hm'] = d['ts'].dt.strftime('%H:%M')
    g = d.groupby('date')
    day = pd.DataFrame({'vol_day': g['Volume'].sum(), 'nbar': g.size(), 'close_last': g['Close'].last(), 'hm_last': g['hm'].last()}).reset_index()
    day['code'] = code
    hd = d[d.hm.isin(HM)][['date', 'hm', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
    hd['code'] = code
    return code, info, hd, day

with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(one, codes))
log = {c: i for c, i, _, _ in res}
H = pd.concat([h for _, _, h, _ in res if h is not None], ignore_index=True)
D = pd.concat([d for _, _, _, d in res if d is not None], ignore_index=True)
H.to_parquet(HERE / 'raw' / 'tw_stocker_head.parquet', index=False)
D.to_parquet(HERE / 'raw' / 'tw_stocker_daily.parquet', index=False)
summ = {'n_codes_requested': len(codes), 'n_ok': sum(1 for i in log.values() if i['status'] == 'ok'),
        'n_sha_match': sum(1 for i in log.values() if i.get('sha_match')), 'source': 'https://raw.githubusercontent.com/voidful/tw_stocker/main/data/<code>.csv',
        'manifest': str(MAN), 'manifest_sha256': hashlib.sha256(MAN.read_bytes()).hexdigest()}
json.dump({'summary': summ, 'files': log}, open(HERE / 'raw' / 'tw_stocker_download_log.json', 'w'), ensure_ascii=False, indent=1)
print(summ)

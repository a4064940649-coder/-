#!/usr/bin/env python3
"""下載 voidful/tw_stocker 的 Yahoo 5 分 K（2024-02-15～2026-04-01），範圍：2024-02-15～2026-04-01 間曾在前 300 流動性池
（vrank ≤ 300，上市、非 0 開頭 4 碼）的全部股票。下載內容視為不可信：只用 pandas 讀 CSV，不執行任何內容。
逐檔驗證 sha256 與 manifest（raw/tw_stocker_manifest.json）；去掉重複時間戳（已知 bug，保留第一筆）並記錄筆數；
只保留 09:00～09:55 的 12 根 K 棒（raw/tw5m_head.parquet），以及每日彙總（raw/tw5m_daily.parquet：總量、根數、首末時間、
13:20 收盤、全日 VWAP、09:05～09:25 VWAP 等）。原檔不保留（約 2GB）。下載紀錄：raw/tw5m_download_log.json。
用法：python3 -I -B fetch_5m.py
"""
import json, subprocess, tempfile, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
MAN = RAW / 'tw_stocker_manifest.json'
m = json.load(open(MAN))
F = {x['ticker']: x for x in m['files'] if x['frequency'] == 'intraday'}
FA = pd.read_parquet(HERE / 'data' / 'feat_all_2023.parquet', columns=['date', 'code', 'vrank'])
w = FA[(FA.date >= '2024-02-15') & (FA.date <= '2026-04-01') & (FA.vrank <= 300)]
codes_pool = sorted(c for c in w.code.unique() if len(c) == 4 and not c.startswith('0'))
codes = [c for c in codes_pool if c in F]
HM = [f'09:{m:02d}' for m in range(0, 60, 5)]

def one(code):
    for attempt in range(3):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / 'x.csv'
            r = subprocess.run(['curl', '-sS', '-f', '-m', '900', '-o', str(tmp), f'https://raw.githubusercontent.com/voidful/tw_stocker/main/data/{code}.csv'], capture_output=True)
            if r.returncode != 0 or not tmp.exists():
                err = r.stderr.decode()[:200]; continue
            h = hashlib.sha256(tmp.read_bytes()).hexdigest()
            d = pd.read_csv(tmp)
            break
    else:
        return code, {'status': 'fail', 'err': err}, None, None
    info = {'status': 'ok', 'sha256': h, 'manifest_sha256': F[code]['sha256'], 'sha_match': h == F[code]['sha256'], 'rows_raw': int(len(d))}
    d['ts'] = pd.to_datetime(d['Datetime'], utc=True).dt.tz_convert('Asia/Taipei')
    nd = int(d.duplicated('ts').sum())
    nd_conf = int(d[d.duplicated('ts', keep=False)].drop_duplicates(['ts', 'Open', 'High', 'Low', 'Close', 'Volume']).duplicated('ts').sum())
    d = d.drop_duplicates('ts', keep='first').sort_values('ts')
    info['dup_ts_dropped'] = nd; info['dup_ts_conflicting'] = nd_conf
    d['date'] = d['ts'].dt.strftime('%Y-%m-%d'); d['hm'] = d['ts'].dt.strftime('%H:%M')
    d['tp'] = (d['High'] + d['Low'] + d['Close']) / 3
    d['tpv'] = d['tp'] * d['Volume']
    g = d.groupby('date')
    win = d[d.hm.isin(['09:05', '09:10', '09:15', '09:20', '09:25'])].groupby('date')
    day = pd.DataFrame({'vol_day': g['Volume'].sum(), 'nbar': g.size(), 'hm_first': g['hm'].first(), 'hm_last': g['hm'].last(),
                        'close_last': g['Close'].last(), 'open_first': g['Open'].first(), 'high_day': g['High'].max(), 'low_day': g['Low'].min(),
                        'vwap_day': g['tpv'].sum() / g['Volume'].sum().replace(0, np.nan),
                        'vol_0905_0925': win['Volume'].sum(), 'vwap_0905_0925': win['tpv'].sum() / win['Volume'].sum().replace(0, np.nan)}).reset_index()
    day['code'] = code
    hd = d[d.hm.isin(HM)][['date', 'hm', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
    hd['code'] = code
    return code, info, hd, day

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, codes))
    log = {c: i for c, i, _, _ in res}
    H = pd.concat([h for _, _, h, _ in res if h is not None], ignore_index=True)
    D = pd.concat([d for _, _, _, d in res if d is not None], ignore_index=True)
    H.to_parquet(RAW / 'tw5m_head.parquet', index=False)
    D.to_parquet(RAW / 'tw5m_daily.parquet', index=False)
    summ = {'n_codes_pool': len(codes_pool), 'n_codes_not_in_manifest': sorted(set(codes_pool) - set(F)), 'n_codes_requested': len(codes),
            'n_ok': sum(1 for i in log.values() if i['status'] == 'ok'), 'n_sha_match': sum(1 for i in log.values() if i.get('sha_match')),
            'dup_ts_dropped_total': sum(i.get('dup_ts_dropped', 0) for i in log.values()),
            'dup_ts_conflicting_total': sum(i.get('dup_ts_conflicting', 0) for i in log.values()),
            'source': 'https://raw.githubusercontent.com/voidful/tw_stocker/main/data/<code>.csv', 'manifest_sha256': hashlib.sha256(MAN.read_bytes()).hexdigest()}
    json.dump({'summary': summ, 'files': log}, open(RAW / 'tw5m_download_log.json', 'w'), ensure_ascii=False, indent=1)
    print(summ)

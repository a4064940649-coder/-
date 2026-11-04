#!/usr/bin/env python3
"""第 0 步盤中結構校準資料：下載 voidful/tw_stocker（GitHub 公開 repo）Yahoo 5 分 K 原檔（不可信資料），
只取 2024-02-15～2026-03-31 期間曾被處置的上市普通股（d0 名單），逐檔：
  驗 sha256 與 manifest、依時間戳去重（value_fix 已知重複列 bug）、彙總成「每日一列」特徵，原檔立即刪除。
特徵：bar 數、去重後總量、第一根非零量 bar 時間與量、最後一根 bar 時間／量／收盤、倒數第二根收盤、13:20 bar 收盤。
不讀任何事件報酬。輸出 raw/yahoo5m_daily_features.parquet 與 raw/yahoo5m_download_log.json。
用法：python3 -I -B s0b_fetch5m.py"""
import json, subprocess, tempfile, hashlib, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

HERE = Path(__file__).resolve().parent
MAN = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/data_ext/tw_stocker_manifest.json')
m = json.load(open(MAN))
F = {x['ticker']: x for x in m['files'] if x['frequency'] == 'intraday'}
d = pd.read_csv(HERE / 'raw' / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['start', 'end'])
d['name'] = d['name'].fillna('')
c = d[(d.code.str.len() == 4) & (~d.code.str.startswith('0')) & (~d.name.str.contains('DR'))]
c = c[(c.end >= '2024-02-15') & (c.start <= '2026-03-31')]
codes = sorted(set(c.code) & set(F))


def one(code):
    with tempfile.TemporaryDirectory(dir='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad') as td:
        tmp = Path(td) / 'x.csv'
        r = subprocess.run(['curl', '-sS', '-f', '-m', '600', '-o', str(tmp),
                            f'https://raw.githubusercontent.com/voidful/tw_stocker/main/data/{code}.csv'], capture_output=True)
        if r.returncode != 0 or not tmp.exists():
            return code, {'status': 'fail', 'err': r.stderr.decode()[:200]}, None
        h = hashlib.sha256(tmp.read_bytes()).hexdigest()
        x = pd.read_csv(tmp, usecols=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    info = {'status': 'ok', 'sha256': h, 'manifest_sha256': F[code]['sha256'], 'sha_match': h == F[code]['sha256'], 'rows_raw': int(len(x))}
    x['ts'] = pd.to_datetime(x['Datetime'], utc=True).dt.tz_convert('Asia/Taipei')
    info['dup_ts_dropped'] = int(x.duplicated('ts').sum())
    x = x.drop_duplicates('ts', keep='first').sort_values('ts')
    x['date'] = x['ts'].dt.strftime('%Y-%m-%d'); x['hm'] = x['ts'].dt.strftime('%H:%M')
    rows = []
    for dt, g in x.groupby('date', sort=True):
        g = g.reset_index(drop=True)
        nz = g[g.Volume > 0]
        rows.append(dict(code=code, date=dt, nbar=len(g), vol=float(g.Volume.sum()),
                         hm_first=g.hm.iloc[0], open_first=float(g.Open.iloc[0]),
                         hm_first_nz=(nz.hm.iloc[0] if len(nz) else None), vol_first_nz=(float(nz.Volume.iloc[0]) if len(nz) else 0.0),
                         n_nz=int(len(nz)),
                         hm_last=g.hm.iloc[-1], vol_last=float(g.Volume.iloc[-1]), close_last=float(g.Close.iloc[-1]),
                         hm_last_nz=(nz.hm.iloc[-1] if len(nz) else None), vol_last_nz=(float(nz.Volume.iloc[-1]) if len(nz) else 0.0),
                         close_last_nz=(float(nz.Close.iloc[-1]) if len(nz) else None),
                         close_prev_nz=(float(nz.Close.iloc[-2]) if len(nz) >= 2 else None),
                         vol_prev_nz=(float(nz.Volume.iloc[-2]) if len(nz) >= 2 else None),
                         vol_max_bar=float(g.Volume.max())))
    return code, info, pd.DataFrame(rows)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, codes))
    log = {c: i for c, i, _ in res}
    D = pd.concat([f for _, _, f in res if f is not None], ignore_index=True)
    D.to_parquet(HERE / 'raw' / 'yahoo5m_daily_features.parquet', index=False)
    summ = {'n_codes_requested': len(codes), 'n_ok': sum(1 for i in log.values() if i['status'] == 'ok'),
            'n_sha_match': sum(1 for i in log.values() if i.get('sha_match')),
            'source': 'https://raw.githubusercontent.com/voidful/tw_stocker/main/data/<code>.csv',
            'manifest': str(MAN), 'manifest_sha256': hashlib.sha256(MAN.read_bytes()).hexdigest(),
            'note': '原檔下載後即刪除（不放研究資料夾）；只保留每日特徵。'}
    json.dump({'summary': summ, 'files': log}, open(HERE / 'raw' / 'yahoo5m_download_log.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))

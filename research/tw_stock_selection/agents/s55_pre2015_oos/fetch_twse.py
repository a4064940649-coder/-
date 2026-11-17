#!/usr/bin/env python3
"""S55 資料補抓（只抓公開資料；被擋就停，不換方法繞）：
(1) 0050 每月日成交（證交所 STOCK_DAY）2007-01～2015-06 → raw/pre2015/twse_0050_stock_day/YYYY-MM.json（原始回應）＋ e50_ohlc_pre2015.csv
(2) 除權除息計算結果表（證交所 TWT49U）2007-01～2013-12 → raw/pre2015/twt49u_raw/YYYY-MM.json（原始回應）
    ＋ raw/pre2015/twt49u/twse-exright/YYYY/YYYY-MM.csv.gz（與 raw/twt49u 同格式：date,code,prev_close,ref_price,value,type）
每次請求間隔 3.5 秒；任何一次非 200、非 JSON 或 stat 不是 OK（且不是「查無資料」）→ 立即停止並記錄。
用法：python3 -I -B fetch_twse.py"""
import json, time, gzip, io, subprocess, sys, hashlib
from pathlib import Path
import pandas as pd

RAW = Path('/home/user/-/research/tw_stock_selection/raw/pre2015')
LOG = RAW / 'fetch_log.json'
months = lambda a, b: [str(p) for p in pd.period_range(a, b, freq='M')]
log = json.loads(LOG.read_text()) if LOG.exists() else []


def get(url, out):
    if out.exists():
        return json.loads(out.read_text())
    time.sleep(3.5)
    r = subprocess.run(['curl', '-sS', '-m', '40', '-w', '\n%{http_code}', url], capture_output=True)
    body, _, code = r.stdout.rpartition(b'\n')
    code = code.decode().strip()
    rec = dict(url=url, http=code, bytes=len(body), t=time.strftime('%Y-%m-%dT%H:%M:%S'), curl_err=r.stderr.decode()[:200])
    try:
        d = json.loads(body.decode('utf-8'))
    except Exception:
        d = None
    rec['stat'] = d.get('stat') if isinstance(d, dict) else None
    log.append(rec); LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1))
    ok = code == '200' and isinstance(d, dict) and (d.get('stat') == 'OK' or '沒有符合' in str(d.get('stat')) or '查無' in str(d.get('stat')))
    if not ok:
        print('被擋或異常，停止：', rec, flush=True)
        sys.exit(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, ensure_ascii=False))
    return d


def roc(s):
    s = s.strip().replace('年', '/').replace('月', '/').replace('日', '')
    y, m, d = s.split('/')
    return pd.Timestamp(int(y) + 1911, int(m), int(d))


num = lambda x: pd.to_numeric(str(x).replace(',', '').strip(), errors='coerce')

# (1) 0050
rows = []
for ym in months('2007-01', '2015-06'):
    d = get(f'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={ym.replace("-", "")}01&stockNo=0050',
            RAW / 'twse_0050_stock_day' / f'{ym}.json')
    for r in d.get('data', []) or []:
        rows.append(dict(date=roc(r[0]).date(), open=num(r[3]), high=num(r[4]), low=num(r[5]), close=num(r[6]), volume=num(r[1]), amount=num(r[2])))
    print('0050', ym, len(d.get('data', []) or []), flush=True)
e = pd.DataFrame(rows).drop_duplicates('date').sort_values('date')
e.to_csv(RAW / 'e50_ohlc_pre2015.csv', index=False)

# (2) TWT49U
for ym in months('2007-01', '2013-12'):
    y, m = ym.split('-')
    last = (pd.Period(ym, 'M').to_timestamp(how='end')).strftime('%Y%m%d')
    d = get(f'https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate={y}{m}01&endDate={last}&response=json', RAW / 'twt49u_raw' / f'{ym}.json')
    f = d.get('fields') or []
    out = []
    for r in d.get('data', []) or []:
        z = dict(zip(f, r))
        out.append(dict(date=roc(z['資料日期']).strftime('%Y-%m-%d'), code=str(z['股票代號']).strip(), prev_close=num(z['除權息前收盤價']),
                        ref_price=num(z['除權息參考價']), value=num(z['權值+息值']), type=str(z['權/息']).strip()))
    df = pd.DataFrame(out, columns=['date', 'code', 'prev_close', 'ref_price', 'value', 'type'])
    p = RAW / 'twt49u' / 'twse-exright' / y / f'{ym}.csv.gz'
    p.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as g:
        g.write(df.to_csv(index=False).encode())
    p.write_bytes(buf.getvalue())
    print('TWT49U', ym, len(df), flush=True)

man = {str(p.relative_to(RAW)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.rglob('*')) if p.is_file() and p.name not in ('SHA256SUMS.json', 'fetch_log.json')}
(RAW / 'SHA256SUMS.json').write_text(json.dumps(man, ensure_ascii=False, indent=1))
print('完成', len(man), '檔')

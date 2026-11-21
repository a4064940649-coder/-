#!/usr/bin/env python3
"""官方公開資料「第二來源」抓取程式（月營收彙總、除權除息計算結果表、端點探查）。

只做正常、低頻率的公開資料請求（寫死在程式裡，不能用參數放寬）：
  - 每個網址每次執行最多請求 1 次；只有「連線失敗／逾時／5xx」才重試，最多 2 次，間隔 20 秒。
  - 同一天（台北日期）對同一個網站總請求數 ≤ 10（含重試；記在 <快照根>/_requests.csv，跨次執行累計）。
  - 任兩次請求至少間隔 3 秒（跨次執行也算）。
  - 不跟隨轉址、不偽裝瀏覽器、不換代理、不碰驗證頁：遇到 3xx／403／429／回來的是網頁而不是 JSON，
    記一列「blocked」，當天不再請求那個網站。
  - User-Agent 固定 "research-client/1.0"，沒有任何個人資料。代理與憑證用環境設定（HTTPS_PROXY、/root/.ccr/ca-bundle.crt）。

下載內容是不可信資料：原始回應原封不動（gzip，mtime 固定 0）存進 <快照根>/<類別>/，本程式放在別處；
解析只用 json／csv，逐欄驗證格式，不執行、不 import 任何下載內容。請一律用 python3 -I -B 執行。

用法：
  fetch_official.py revenue [--snap-root DIR] [--only twse,tpex]
      抓上市＋上櫃月營收彙總，存快照，轉成和 GitHub 倉庫 data/mops/revenue_hist/<YYYY-MM>_<twse|tpex>.csv 同格式的 csv
      （寫到 <快照根>/derived/revenue_hist/）。內容和上一份快照相同就不重存，只在 index.csv 記一列「未變」。
  fetch_official.py exright --start YYYY-MM-DD --end YYYY-MM-DD [--snap-root DIR]
      抓證交所 TWT49U（上市除權除息計算結果表），存快照，轉成封存格式 date,code,prev_close,ref_price,value,type
      （寫到 <快照根>/derived/twt49u/<起>_<迄>.csv）。
  fetch_official.py probe [--snap-root DIR]
      探查其他官方端點（各 1 次），結果寫 <快照根>/probe/probe_<時間>.json；回應只留前 2000 位元組當樣本。
  fetch_official.py convert-revenue <快照.json.gz> <twse|tpex> <輸出資料夾>      # 離線：不連網
  fetch_official.py convert-exright <快照.json.gz> <輸出 csv>                    # 離線：不連網
  fetch_official.py budget [--snap-root DIR]                                     # 今天各網站已用幾次請求
結束碼：0＝要抓的都抓到；3＝部分來源失敗或被擋（已記錄，其他來源照常完成）；2＝用法錯誤。
"""
import argparse, csv, datetime as dt, gzip, hashlib, io, json, os, re, ssl, sys, time
import urllib.error, urllib.request
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
SNAP_DEFAULT = HERE.parent.parent / 'raw' / 'official_snapshots'
UA = 'research-client/1.0'
MAX_PER_HOST_PER_DAY = 10
MIN_GAP_SEC = 3.0
RETRY_MAX = 2
RETRY_WAIT_SEC = 20.0
TIMEOUT = 60
MAX_BYTES = 40 * 1024 * 1024          # 單一回應上限；超過就不存（版控單檔 < 45MB）
TPE = dt.timezone(dt.timedelta(hours=8))
URL_REV = {'twse': 'https://openapi.twse.com.tw/v1/opendata/t187ap05_L',
           'tpex': 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O'}
URL_TWT49U = 'https://www.twse.com.tw/rwd/zh/exRight/TWT49U?response=json&startDate={s}&endDate={e}'
REV_COLS = ['stock_id', 'name', 'period', 'market', '產業別', '當月營收', '上月營收', '去年當月營收', '上月比較增減(%)',
            '去年同月增減(%)', '當月累計營收', '去年累計營收', '前期比較增減(%)']
REV_SRC = {'產業別': '產業別', '當月營收': '營業收入-當月營收', '上月營收': '營業收入-上月營收', '去年當月營收': '營業收入-去年當月營收',
           '上月比較增減(%)': '營業收入-上月比較增減(%)', '去年同月增減(%)': '營業收入-去年同月增減(%)',
           '當月累計營收': '累計營業收入-當月累計營收', '去年累計營收': '累計營業收入-去年累計營收',
           '前期比較增減(%)': '累計營業收入-前期比較增減(%)'}
CODE_RE = re.compile(r'^[0-9A-Z]{4,7}$')
REQ_HDR = ['utc', 'taipei_date', 'host', 'url', 'attempt', 'http_status', 'bytes', 'sha256', 'result', 'note']
IDX_HDR = ['fetched_utc', 'tag', 'url', 'http_status', 'bytes', 'sha256', 'data_period', 'n_rows', 'report_dates',
           'stored_file', 'same_as', 'note']


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read_csv(p):
    if not Path(p).exists():
        return []
    with open(p, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def append_csv(p, hdr, row):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=hdr, lineterminator='\n')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in hdr})


def gz_write(p, b):
    """gzip 固定 mtime＝0、不寫檔名：同內容永遠同位元組。"""
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9) as g:
            g.write(b)


def gz_read(p):
    p = str(p)
    if p.endswith('.gz'):
        with gzip.open(p, 'rb') as g:
            return g.read()
    return open(p, 'rb').read()


# ---------------------------------------------------------------- 低頻率請求
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):          # 不跟隨轉址：3xx 會變成 HTTPError 回到呼叫端
        return None


def _opener():
    ca = '/root/.ccr/ca-bundle.crt' if Path('/root/.ccr/ca-bundle.crt').exists() else None
    ctx = ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()
    return urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=ctx))   # 代理沿用環境變數


class Polite:
    def __init__(self, snap_root):
        self.root = Path(snap_root); self.ledger = self.root / '_requests.csv'
        self.done_urls = set()

    def today_rows(self):
        d = utcnow().astimezone(TPE).strftime('%Y-%m-%d')
        return [r for r in read_csv(self.ledger) if r['taipei_date'] == d]

    def budget(self):
        out = {}
        for r in self.today_rows():
            o = out.setdefault(r['host'], dict(used=0, blocked=False))
            o['used'] += 1; o['blocked'] |= r['result'] == 'blocked'
        return out

    def _wait_gap(self):
        rows = read_csv(self.ledger)
        if rows:
            last = dt.datetime.strptime(rows[-1]['utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc)
            gap = (utcnow() - last).total_seconds()
            if 0 <= gap < MIN_GAP_SEC + 1:
                time.sleep(MIN_GAP_SEC + 1 - gap)

    def get(self, url):
        """回傳 dict(ok, status, body, sha256, fetched_utc, result, note)。result：ok／blocked／error／skipped。"""
        host = urlsplit(url).hostname
        if url in self.done_urls:
            return dict(ok=False, status='', body=b'', result='skipped', note='這次執行已請求過這個網址')
        self.done_urls.add(url)
        last = None
        for attempt in range(1, RETRY_MAX + 2):
            b = self.budget().get(host, dict(used=0, blocked=False))
            if b['blocked']:
                return last or dict(ok=False, status='', body=b'', result='skipped', note=f'{host} 今天已被擋過，停止請求')
            if b['used'] >= MAX_PER_HOST_PER_DAY:
                return last or dict(ok=False, status='', body=b'', result='skipped', note=f'{host} 今天已請求 {b["used"]} 次，達上限 {MAX_PER_HOST_PER_DAY}')
            self._wait_gap()
            t = utcnow(); status = ''; body = b''; note = ''; result = 'error'
            try:
                req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
                with _opener().open(req, timeout=TIMEOUT) as r:
                    status = r.status; body = r.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    body = b''; note = f'回應超過 {MAX_BYTES} 位元組，不存'
                elif looks_like_webpage(body):
                    result = 'blocked'; note = '回來的是網頁不是 JSON（可能是安全性／驗證頁）'
                else:
                    result = 'ok'
            except urllib.error.HTTPError as e:
                status = e.code
                loc = (e.headers.get('Location') or '')[:200] if e.headers else ''
                if 300 <= e.code < 400 or e.code in (401, 403, 429):
                    result = 'blocked'; note = f'HTTP {e.code}' + (f' 轉址到 {loc}' if loc else '')
                else:
                    note = f'HTTP {e.code} {e.reason}'
            except Exception as e:                                    # 逾時、連線中斷、代理拒絕
                note = f'{type(e).__name__}: {e}'[:300]
                if 'Tunnel connection failed: 403' in note or 'Tunnel connection failed: 407' in note:
                    result = 'blocked'; note = '環境代理不允許這個主機：' + note
            last = dict(ok=result == 'ok', status=status, body=body if result == 'ok' else b'', sha256=sha(body) if body else '',
                        fetched_utc=t.strftime('%Y-%m-%dT%H:%M:%SZ'), result=result, note=note, attempts=attempt,
                        head=body[:2000] if body else b'')
            append_csv(self.ledger, REQ_HDR, dict(utc=t.strftime('%Y-%m-%dT%H:%M:%SZ'), taipei_date=t.astimezone(TPE).strftime('%Y-%m-%d'),
                                                  host=host, url=url, attempt=attempt, http_status=status, bytes=len(body),
                                                  sha256=last['sha256'], result=result, note=note))
            retryable = result == 'error' and (status == '' or (isinstance(status, int) and status >= 500))
            if not retryable or attempt > RETRY_MAX:
                return last
            time.sleep(RETRY_WAIT_SEC)
        return last


def looks_like_webpage(body):
    h = body[:400].lstrip().lower()
    return h.startswith(b'<!doctype') or h.startswith(b'<html') or h.startswith(b'<?xml') or h.startswith(b'<head') or h.startswith(b'<script')


# ---------------------------------------------------------------- 快照
def store_snapshot(root, kind, tag, url, res, data_period='', n_rows='', report_dates='', note=''):
    """內容（sha256）和同 tag 上一份有存檔的快照相同 → 不重存，只記一列「未變」。回傳 (檔案路徑, 是否新存)。"""
    d = Path(root) / kind; idx = d / 'index.csv'
    prev = [r for r in read_csv(idx) if r['tag'] == tag and r['http_status'] == '200' and r['sha256']]
    row = dict(fetched_utc=res.get('fetched_utc', ''), tag=tag, url=url, http_status=res.get('status', ''), bytes=len(res.get('body', b'')),
               sha256=res.get('sha256', ''), data_period=data_period, n_rows=n_rows, report_dates=report_dates, note=note or res.get('note', ''))
    if not res.get('ok'):
        row['note'] = f"{res.get('result')}：{res.get('note', '')}"
        append_csv(idx, IDX_HDR, row); return None, False
    if prev and prev[-1]['sha256'] == res['sha256']:
        same = prev[-1]['stored_file'] or prev[-1]['same_as']
        row['same_as'] = same; row['note'] = ('未變；' + row['note']).rstrip('；')
        append_csv(idx, IDX_HDR, row); return d / same, False
    name = f"{res['fetched_utc'].replace('-', '').replace(':', '')}_{tag}.json.gz"
    gz_write(d / name, res['body']); row['stored_file'] = name
    append_csv(idx, IDX_HDR, row); return d / name, True


# ---------------------------------------------------------------- 解析：月營收
def roc_ym(s):
    """民國年月 '11510' → '2026-10'。"""
    s = str(s).strip()
    if not re.fullmatch(r'\d{4,5}', s):
        raise ValueError(f'資料年月格式不對：{s!r}')
    y, m = int(s[:-2]) + 1911, int(s[-2:])
    if not (1 <= m <= 12 and 2000 <= y <= 2100):
        raise ValueError(f'資料年月不合理：{s!r}')
    return f'{y}-{m:02d}'


def clean_num(s, pct=False):
    """只接受數字；其他（空白、'-'、文字）回空字串。百分比固定兩位小數（和倉庫彙總表相同寫法）。"""
    s = str(s if s is not None else '').replace(',', '').strip()
    if not re.fullmatch(r'-?\d+(\.\d+)?', s):
        return ''
    if pct:
        return f'{float(s):.2f}'
    return s


def parse_revenue(body):
    """回傳 (rows, info)。rows＝同倉庫欄位的 dict（還沒填 market）；info＝資料年月分布、出表日期分布、被丟掉的列。"""
    data = json.loads(body.decode('utf-8-sig'))
    if not isinstance(data, list):
        raise ValueError('回應不是清單')
    rows = []; periods = {}; reports = {}; bad = []; keys = set()
    for x in data:
        if not isinstance(x, dict):
            bad.append('非物件'); continue
        keys |= set(x)
        code = str(x.get('公司代號', '')).strip()
        if not CODE_RE.match(code):
            bad.append(f'代號 {code[:12]!r}'); continue
        try:
            per = roc_ym(x.get('資料年月', ''))
        except ValueError as e:
            bad.append(f'{code} {e}'); continue
        periods[per] = periods.get(per, 0) + 1
        rd = str(x.get('出表日期', '')).strip()
        reports[rd] = reports.get(rd, 0) + 1
        r = dict(stock_id=code, name=str(x.get('公司名稱', '')).strip().replace('\n', ' ').replace(',', '，')[:40], period=per)
        for dst, src in REV_SRC.items():
            v = x.get(src, '')
            r[dst] = str(v).strip().replace(',', '，')[:40] if dst == '產業別' else clean_num(v, pct='%' in dst)
        rows.append(r)
    return rows, dict(periods=periods, report_dates=reports, n_bad=len(bad), bad=bad[:20], keys=sorted(keys))


def write_revenue_csv(rows, market, out_dir):
    """每個資料年月寫一個 <YYYY-MM>_<market>.csv（依代號排序；同代號留最後一筆）。回傳 {月份: 路徑}。"""
    out = {}
    for per in sorted({r['period'] for r in rows}):
        sel = {}
        for r in rows:
            if r['period'] == per:
                sel[r['stock_id']] = r
        p = Path(out_dir) / f'{per}_{market}.csv'; p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh, lineterminator='\n'); w.writerow(REV_COLS)
            for c in sorted(sel):
                r = dict(sel[c], market=market); w.writerow([r[k] for k in REV_COLS])
        out[per] = p
    return out


# ---------------------------------------------------------------- 解析：TWT49U
def roc_date(s):
    """'115年11月02日' → '2026-11-02'。"""
    m = re.fullmatch(r'\s*(\d{2,3})年(\d{1,2})月(\d{1,2})日\s*', str(s))
    if not m:
        raise ValueError(f'日期格式不對：{s!r}')
    return dt.date(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3))).isoformat()


def trim_num(s):
    """'1,234.50' → '1234.5'（封存是去掉千分位與尾端 0 的寫法）；只做字串處理。"""
    s = str(s if s is not None else '').replace(',', '').strip()
    if not re.fullmatch(r'-?\d+(\.\d+)?', s):
        return ''
    neg = s.startswith('-'); s = s.lstrip('-')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    s = s.lstrip('0') or '0'
    if s.startswith('.'):
        s = '0' + s
    return ('-' if neg and s != '0' else '') + s


def parse_twt49u(body):
    """回傳 (rows, info)；rows＝[date,code,prev_close,ref_price,value,type] 依日期、代號排序。"""
    j = json.loads(body.decode('utf-8-sig'))
    if not isinstance(j, dict):
        raise ValueError('回應不是物件')
    stat = str(j.get('stat', ''))
    fields = j.get('fields') or []; data = j.get('data') or []
    info = dict(stat=stat, fields=fields, n_raw=len(data), bad=[])
    if stat != 'OK' or not fields:
        info['empty'] = True
        return [], info
    need = {'date': '資料日期', 'code': '股票代號', 'prev_close': '除權息前收盤價', 'ref_price': '除權息參考價', 'value': '權值+息值', 'type': '權/息'}
    ix = {}
    for k, name in need.items():
        if name not in fields:
            raise ValueError(f'TWT49U 少了欄位 {name}；實際欄位 {fields}')
        ix[k] = fields.index(name)
    rows = []
    for x in data:
        try:
            code = str(x[ix['code']]).strip()
            if not CODE_RE.match(code):
                raise ValueError(f'代號 {code[:12]!r}')
            typ = str(x[ix['type']]).strip()
            if typ not in ('息', '權', '權息'):
                raise ValueError(f'{code} 種類 {typ[:6]!r}')
            rows.append([roc_date(x[ix['date']]), code, trim_num(x[ix['prev_close']]), trim_num(x[ix['ref_price']]), trim_num(x[ix['value']]), typ])
        except (ValueError, IndexError, TypeError) as e:
            info['bad'].append(str(e)[:80])
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, info


def write_twt49u_csv(rows, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n'); w.writerow(['date', 'code', 'prev_close', 'ref_price', 'value', 'type']); w.writerows(rows)
    return p


# ---------------------------------------------------------------- 子指令
def do_revenue(root, only=('twse', 'tpex'), pol=None):
    """回傳 {market: dict(status, result, period, n_rows, csv, snapshot, new, note)}。任何一個失敗不影響另一個。"""
    pol = pol or Polite(root); out = {}
    for mk in only:
        url = URL_REV[mk]; res = pol.get(url)
        o = dict(url=url, status=res.get('status', ''), result=res.get('result'), note=res.get('note', ''), fetched_utc=res.get('fetched_utc', ''))
        if res.get('ok'):
            try:
                rows, info = parse_revenue(res['body'])
                per = max(info['periods'], key=info['periods'].get) if info['periods'] else ''
                snap, new = store_snapshot(root, 'revenue', mk, url, res, data_period='|'.join(f'{k}:{v}' for k, v in sorted(info['periods'].items())),
                                           n_rows=len(rows), report_dates='|'.join(f'{k}:{v}' for k, v in sorted(info['report_dates'].items())),
                                           note=f"丟掉 {info['n_bad']} 列" if info['n_bad'] else '')
                files = write_revenue_csv(rows, mk, Path(root) / 'derived' / 'revenue_hist')
                o.update(period=per, periods=info['periods'], report_dates=info['report_dates'], n_rows=len(rows), n_bad=info['n_bad'],
                         csv={k: str(v) for k, v in files.items()}, snapshot=str(snap), new_snapshot=new, sha256=res['sha256'], bytes=len(res['body']))
            except (ValueError, KeyError, UnicodeDecodeError) as e:
                o.update(result='parse_error', note=f'解析失敗：{e}'[:300])
                store_snapshot(root, 'revenue', mk, url, res, note=o['note'])
        else:
            store_snapshot(root, 'revenue', mk, url, res)
        out[mk] = o
    return out


def do_exright(root, start, end, pol=None):
    pol = pol or Polite(root)
    url = URL_TWT49U.format(s=start.replace('-', ''), e=end.replace('-', '')); res = pol.get(url)
    o = dict(url=url, status=res.get('status', ''), result=res.get('result'), note=res.get('note', ''), fetched_utc=res.get('fetched_utc', ''), start=start, end=end)
    tag = f'twt49u_{start}_{end}'
    if res.get('ok'):
        try:
            rows, info = parse_twt49u(res['body'])
            snap, new = store_snapshot(root, 'exright', tag, url, res, data_period=f'{start}~{end}', n_rows=len(rows),
                                       note=(f"stat={info['stat']}" if info.get('empty') else '') + (f" 丟掉 {len(info['bad'])} 列" if info['bad'] else ''))
            p = write_twt49u_csv(rows, Path(root) / 'derived' / 'twt49u' / f'{start}_{end}.csv')
            o.update(n_rows=len(rows), stat=info['stat'], bad=info['bad'][:10], csv=str(p), snapshot=str(snap), new_snapshot=new,
                     sha256=res['sha256'], bytes=len(res['body']), first=rows[0][0] if rows else '', last=rows[-1][0] if rows else '')
        except (ValueError, KeyError, UnicodeDecodeError) as e:
            o.update(result='parse_error', note=f'解析失敗：{e}'[:300])
            store_snapshot(root, 'exright', tag, url, res, note=o['note'])
    else:
        store_snapshot(root, 'exright', tag, url, res)
    return o


PROBES = [
    ('上市每日收盤行情', 'https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL'),
    ('上市財報（綜合損益表，一般業）', 'https://openapi.twse.com.tw/v1/opendata/t187ap06_L_ci'),
    ('上市每日重大訊息', 'https://openapi.twse.com.tw/v1/opendata/t187ap04_L'),
    ('上櫃每日收盤行情', 'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes'),
    ('上櫃除權息計算結果', 'https://www.tpex.org.tw/openapi/v1/tpex_exright_daily'),
    ('上櫃財報（綜合損益表，一般業）', 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap06_O_ci'),
    ('上市減資恢復買賣參考價', 'https://www.twse.com.tw/rwd/zh/reducation/TWTAUU?response=json&startDate={s}&endDate={e}'),
]


def do_probe(root, start, end):
    pol = Polite(root); out = []
    for name, url in PROBES:
        url = url.format(s=start.replace('-', ''), e=end.replace('-', ''))
        res = pol.get(url)
        o = dict(name=name, url=url, status=res.get('status', ''), result=res.get('result'), note=res.get('note', ''),
                 fetched_utc=res.get('fetched_utc', ''), bytes=len(res.get('body', b'')), sha256=res.get('sha256', ''))
        if res.get('ok'):
            try:
                j = json.loads(res['body'].decode('utf-8-sig'))
                if isinstance(j, list):
                    o.update(shape='list', n=len(j), fields=list(j[0].keys()) if j and isinstance(j[0], dict) else [], sample=j[:2])
                elif isinstance(j, dict):
                    o.update(shape='dict', keys=list(j.keys())[:20], stat=str(j.get('stat', ''))[:60], fields=j.get('fields'),
                             n=len(j.get('data') or []), sample=(j.get('data') or [])[:2], title=str(j.get('title', ''))[:80])
            except (ValueError, UnicodeDecodeError) as e:
                o.update(result='parse_error', note=f'不是 JSON：{e}'[:200])
        out.append(o)
        print(f"{name}：{o['result']} HTTP {o['status']} {o.get('n', '')} 筆 {o['note']}")
    p = Path(root) / 'probe' / f"probe_{utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"; p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'已寫 {p}')
    return out


def main():
    ap = argparse.ArgumentParser(description='官方公開資料第二來源抓取（低頻率）')
    sp = ap.add_subparsers(dest='cmd', required=True)
    a = sp.add_parser('revenue'); a.add_argument('--snap-root', default=str(SNAP_DEFAULT)); a.add_argument('--only', default='twse,tpex')
    a = sp.add_parser('exright'); a.add_argument('--snap-root', default=str(SNAP_DEFAULT)); a.add_argument('--start', required=True); a.add_argument('--end', required=True)
    a = sp.add_parser('probe'); a.add_argument('--snap-root', default=str(SNAP_DEFAULT)); a.add_argument('--start', default=''); a.add_argument('--end', default='')
    a = sp.add_parser('convert-revenue'); a.add_argument('snapshot'); a.add_argument('market', choices=['twse', 'tpex']); a.add_argument('out_dir')
    a = sp.add_parser('convert-exright'); a.add_argument('snapshot'); a.add_argument('out_csv')
    a = sp.add_parser('budget'); a.add_argument('--snap-root', default=str(SNAP_DEFAULT))
    g = ap.parse_args()
    dre = re.compile(r'\d{4}-\d{2}-\d{2}')
    if g.cmd == 'revenue':
        only = [x for x in g.only.split(',') if x in URL_REV]
        r = do_revenue(g.snap_root, only)
        for mk, o in r.items():
            if o['result'] == 'ok':
                print(f"{mk}：HTTP {o['status']}，資料年月 {o['periods']}，{o['n_rows']} 家，出表日期 {o['report_dates']}，"
                      f"{'新快照' if o['new_snapshot'] else '內容未變（不重存）'} {o['snapshot']}；csv {list(o['csv'].values())}")
            else:
                print(f"{mk}：沒抓到（{o['result']}，HTTP {o['status']}）{o['note']}")
        sys.exit(0 if all(o['result'] == 'ok' for o in r.values()) else 3)
    if g.cmd == 'exright':
        if not (dre.fullmatch(g.start) and dre.fullmatch(g.end) and g.start <= g.end):
            sys.exit('--start／--end 要寫 YYYY-MM-DD，且起 ≤ 迄')
        o = do_exright(g.snap_root, g.start, g.end)
        if o['result'] == 'ok':
            print(f"TWT49U {g.start}～{g.end}：HTTP {o['status']}，{o['n_rows']} 筆（{o['first']}～{o['last']}），"
                  f"{'新快照' if o['new_snapshot'] else '內容未變（不重存）'} {o['snapshot']}；csv {o['csv']}")
        else:
            print(f"TWT49U：沒抓到（{o['result']}，HTTP {o['status']}）{o['note']}")
        sys.exit(0 if o['result'] == 'ok' else 3)
    if g.cmd == 'probe':
        today = utcnow().astimezone(TPE).date()
        s = g.start or (today - dt.timedelta(days=90)).isoformat(); e = g.end or today.isoformat()
        r = do_probe(g.snap_root, s, e)
        sys.exit(0 if all(o['result'] == 'ok' for o in r) else 3)
    if g.cmd == 'convert-revenue':
        rows, info = parse_revenue(gz_read(g.snapshot)); files = write_revenue_csv(rows, g.market, g.out_dir)
        print(json.dumps(dict(periods=info['periods'], report_dates=info['report_dates'], n_bad=info['n_bad'], files={k: str(v) for k, v in files.items()}), ensure_ascii=False))
    if g.cmd == 'convert-exright':
        rows, info = parse_twt49u(gz_read(g.snapshot)); write_twt49u_csv(rows, g.out_csv)
        print(json.dumps(dict(stat=info['stat'], n=len(rows), bad=info['bad'][:10], out=g.out_csv), ensure_ascii=False))
    if g.cmd == 'budget':
        b = Polite(g.snap_root).budget()
        print(json.dumps(b, ensure_ascii=False) if b else '今天還沒有請求')


if __name__ == '__main__':
    main()

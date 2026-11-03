#!/usr/bin/env python3
"""官方月營收彙總表抓取器（2026-11-03 BF-5 新增；不在 PREREG 登記；只抓公開資料）。

用途：上游 tw-stock-screener 的月營收（rev_5y、rev_hist）在法定期限日後可能延遲。這支程式向官方公開來源抓
「上市／上櫃公司每月營業收入彙總表」，原樣存檔，供 fwd_data.Source(official=...) 在上游缺值時補值。

來源（依序；OpenAPI 失敗的市場才改抓公開資訊觀測站的靜態彙總頁）
  1. 證交所 OpenAPI  https://openapi.twse.com.tw/v1/opendata/t187ap05_L        上市公司每月營業收入彙總表（JSON）
  2. 櫃買中心 OpenAPI https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O     上櫃公司每月營業收入彙總表（JSON）
  3. 公開資訊觀測站  https://mopsov.twse.com.tw/nas/t21/{sii,otc}/t21sc03_{民國年}_{月}_{0,1}.html（HTML，cp950；
     只在該市場 OpenAPI 失敗時抓；月份＝台北時間上個月；_0 國內公司、_1 外國公司）
     ※ MOPS 解析器在 2026-11-03 寫成時，本容器的網路政策擋下所有官方主機，未能用真實回應實測。

輸出：<out-root>/<台北時間 YYYYMMDDTHHMMSS+0800>/（每次執行一個新目錄，不覆寫舊目錄）
  - 每個請求的原始回應位元組原樣存檔（失敗時只記錄錯誤，不存檔）
  - FETCH.json：每個請求的 URL、HTTP 狀態、開始／完成時間（台北）、位元組數、sha256、錯誤訊息
  - SHA256SUMS：原始檔的 sha256（sha256sum -c 可驗）
時點規則：「完成抓取的時間」就是這份資料的可得時點（ann_eff）。不得把它當成歷史公告日往回套；
  fwd_data 合併時只把它用在「抓取完成時間 < 該資料的使用期限」的截斷資料夾。
安全：下載內容是不可信資料。本程式放在 raw/ 之外，請一律以 python3 -I -B 執行並以參數傳路徑；
  解析只用 json 與 html.parser，欄位逐一驗證（代號格式、年月、數字），不執行、不 import 任何下載內容。
用法：
  python3 -I -B rev_official_fetch.py [--out-root DIR] [--sources twse,tpex] [--no-mops] [--timeout 60]
  python3 -I -B rev_official_fetch.py --parse <抓取目錄>        # 只解析、印摘要（驗 sha256）
回傳碼：0＝至少一個來源取得可解析的資料；1＝全部失敗（目錄與 FETCH.json 仍保留，作為失敗紀錄）。
"""
import argparse, datetime, hashlib, json, os, re, ssl, sys, urllib.request, urllib.error
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / 'raw/official_rev'
TZ8 = datetime.timezone(datetime.timedelta(hours=8))
UA = 'Mozilla/5.0 (research; o_forward_infra_live rev_official_fetch)'
OPENAPI = {
    'twse': ('twse_t187ap05_L.json', 'https://openapi.twse.com.tw/v1/opendata/t187ap05_L'),
    'tpex': ('tpex_mopsfin_t187ap05_O.json', 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O'),
}
MOPS_MKT = {'twse': 'sii', 'tpex': 'otc'}
MAX_BYTES = 30 * 1024 * 1024          # 單一回應上限（正常約 0.3～1MB）；超過就中止，避免塞爆磁碟
CODE_RE = re.compile(r'^[0-9]{4}[0-9A-Z]{0,2}$')


def now8():
    return datetime.datetime.now(TZ8)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def ssl_ctx():
    ca = os.environ.get('SSL_CERT_FILE') or ('/root/.ccr/ca-bundle.crt' if Path('/root/.ccr/ca-bundle.crt').exists() else None)
    return ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()


def get(url, timeout):
    """回傳 (status, headers, body, err)。HTTP 錯誤也保留回應本文（原樣存檔用）。"""
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': '*/*'})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx()) as r:
            body = r.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES: return r.status, dict(r.headers), b'', f'回應超過 {MAX_BYTES} bytes，中止'
            return r.status, dict(r.headers), body, None
    except urllib.error.HTTPError as e:
        try: body = e.read(MAX_BYTES)
        except Exception: body = b''
        return e.code, dict(e.headers or {}), body, f'HTTPError {e.code} {e.reason}'
    except Exception as e:                                   # 代理 403（CONNECT 被拒）、逾時、DNS 等
        return None, {}, b'', f'{type(e).__name__}: {e}'


# ---------------------------------------------------------------- 解析（不可信資料：逐欄驗證）
def _num(s):
    if s is None: return None
    s = str(s).strip().replace(',', '')
    if s in ('', '-', '--', 'N/A', 'NA', '不適用'): return None
    try:
        x = float(s)
    except ValueError:
        return None
    return x if x == x and abs(x) < 1e15 else None


def _ym(s):
    """資料年月：民國 '11509'、'115/09'、'115-9' 或西元 '2026-09'、'202609' → 'YYYY-MM'。"""
    s = re.sub(r'\D', ' ', str(s or '')).split()
    if len(s) == 1 and len(s[0]) in (5, 6):
        t = s[0]; y, m = (int(t[:-2]), int(t[-2:]))
    elif len(s) >= 2:
        y, m = int(s[0]), int(s[1])
    else:
        return None
    if y < 1911: y += 1911
    return f'{y:04d}-{m:02d}' if 2000 <= y <= 2100 and 1 <= m <= 12 else None


def _pick(keys, *need, avoid=()):
    for k in keys:
        if all(n in k for n in need) and not any(a in k for a in avoid): return k
    return None


def parse_openapi(body):
    """證交所／櫃買 OpenAPI JSON（list of dict，中文欄名）→ [dict(code, ym, revenue, yoy, mom)]。"""
    data = json.loads(body.decode('utf-8-sig'))
    if not isinstance(data, list): raise ValueError('不是 JSON 陣列')
    out = []
    for r in data:
        if not isinstance(r, dict): continue
        ks = [str(k) for k in r.keys()]
        kc = _pick(ks, '公司代號') or _pick(ks, 'Code')
        ky = _pick(ks, '資料年月') or _pick(ks, 'YearMonth')
        kr = _pick(ks, '當月營收', avoid=('累計',))
        kyoy = _pick(ks, '去年同月增減')
        kmom = _pick(ks, '上月比較增減')
        if not (kc and ky and kr): continue
        code, ym = str(r[kc]).strip(), _ym(r[ky])
        if not CODE_RE.match(code) or ym is None: continue
        out.append(dict(code=code, ym=ym, revenue=_num(r[kr]), yoy=_num(r.get(kyoy)) if kyoy else None,
                        mom=_num(r.get(kmom)) if kmom else None))
    return out


class _Cells(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows, self.cur, self.cell, self.intd = [], None, [], False

    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.cur = []
        elif tag in ('td', 'th') and self.cur is not None: self.intd, self.cell = True, []

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cur is not None and self.intd:
            self.cur.append(''.join(self.cell).strip()); self.intd = False
        elif tag == 'tr' and self.cur is not None:
            self.rows.append(self.cur); self.cur = None

    def handle_data(self, d):
        if self.intd: self.cell.append(d)


def parse_mops_html(body, ym):
    """t21sc03 彙總頁：每列 公司代號、公司名稱、當月營收、上月營收、去年當月營收、上月比較增減(%)、去年同月增減(%)、…（千元）。"""
    p = _Cells(); p.feed(body.decode('cp950', errors='replace'))
    out = []
    for c in p.rows:
        if len(c) >= 7 and CODE_RE.match(c[0].strip()):
            out.append(dict(code=c[0].strip(), ym=ym, revenue=_num(c[2]), yoy=_num(c[6]), mom=_num(c[5])))
    return out


def parse_dir(d, verify=True):
    """讀一個抓取目錄 → (rows, meta)。rows 每列加 fetched_ts（該請求完成時間，台北 ISO）、source、file。
    verify：原始檔 sha256 必須和 FETCH.json 相同，不同就整個目錄不用（防竄改）。"""
    d = Path(d)
    meta = json.loads((d / 'FETCH.json').read_text(encoding='utf-8'))
    rows, bad = [], []
    for q in meta.get('requests', []):
        f = q.get('file')
        if not f or q.get('status') != 200: continue
        p = d / f
        if not p.is_file() or p.parent.resolve() != d.resolve(): bad.append(f'{f}: 檔案不存在或路徑不合法'); continue
        b = p.read_bytes()
        if verify and sha_bytes(b) != q.get('sha256'): bad.append(f'{f}: sha256 與 FETCH.json 不符'); continue
        try:
            rr = parse_openapi(b) if q['kind'] == 'openapi' else parse_mops_html(b, q['ym'])
        except Exception as e:
            bad.append(f'{f}: 解析失敗 {type(e).__name__}: {e}'); continue
        for r in rr: r.update(fetched_ts=q['done_taipei'], source=q['source'], file=f, fetch_dir=d.name)
        rows += rr
    if bad and verify and any('sha256' in x for x in bad): return [], dict(meta, parse_errors=bad, rejected=True)
    return rows, dict(meta, parse_errors=bad)


# ---------------------------------------------------------------- 抓取
def fetch(out_root, sources, use_mops, timeout):
    t0 = now8()
    d = Path(out_root) / t0.strftime('%Y%m%dT%H%M%S%z')
    k = 0
    while d.exists():                                   # 同一秒重跑：另開新目錄，不覆寫
        k += 1; d = Path(out_root) / (t0.strftime('%Y%m%dT%H%M%S%z') + f'_{k}')
    d.mkdir(parents=True)
    reqs = []

    def one(source, kind, fname, url, ym=None):
        st = now8()
        status, hdr, body, err = get(url, timeout)
        done = now8()
        q = dict(source=source, kind=kind, url=url, ym=ym, start_taipei=st.isoformat(), done_taipei=done.isoformat(), status=status,
                 error=err, headers={k_: v for k_, v in hdr.items() if k_.lower() in ('content-type', 'date', 'last-modified', 'etag', 'content-length')})
        if body:
            (d / fname).write_bytes(body)
            q.update(file=fname, bytes=len(body), sha256=sha_bytes(body))
            if status == 200:
                try:
                    rr = parse_openapi(body) if kind == 'openapi' else parse_mops_html(body, ym)
                    yms = sorted({r['ym'] for r in rr})
                    q.update(n_rows=len(rr), yms=yms)
                    if not rr: q['error'] = (q['error'] or '') + '可解析的列數為 0'
                except Exception as e:
                    q.update(n_rows=0, error=f'解析失敗 {type(e).__name__}: {e}')
        reqs.append(q)
        return q

    ok_mkt = {}
    for s in sources:
        fname, url = OPENAPI[s]
        q = one(s, 'openapi', fname, url)
        ok_mkt[s] = q['status'] == 200 and q.get('n_rows', 0) > 0
    if use_mops:
        p = (t0.year * 12 + t0.month - 2)
        y, m = divmod(p, 12); m += 1
        roc = y - 1911
        for s in sources:
            if ok_mkt[s]: continue
            for kk in (0, 1):
                url = f'https://mopsov.twse.com.tw/nas/t21/{MOPS_MKT[s]}/t21sc03_{roc}_{m}_{kk}.html'
                q = one(f'mops_{MOPS_MKT[s]}_{kk}', 'mops_html', f'mops_{MOPS_MKT[s]}_t21sc03_{roc}_{m}_{kk}.html', url, ym=f'{y:04d}-{m:02d}')
                ok_mkt[s] = ok_mkt[s] or (q['status'] == 200 and q.get('n_rows', 0) > 0)
    meta = dict(fetch_dir=d.name, started_taipei=t0.isoformat(), finished_taipei=now8().isoformat(), tool='rev_official_fetch.py',
                tool_sha256=sha_bytes(Path(__file__).read_bytes()), rule='每列的可得時點＝該請求 done_taipei；不得往回套成歷史公告日',
                ok_markets=ok_mkt, requests=reqs)
    (d / 'FETCH.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    sums = ''.join(f"{q['sha256']}  {q['file']}\n" for q in reqs if q.get('file'))
    (d / 'SHA256SUMS').write_text(sums, encoding='utf-8')
    return d, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--sources', default='twse,tpex')
    ap.add_argument('--no-mops', action='store_true', help='OpenAPI 失敗時不改抓公開資訊觀測站')
    ap.add_argument('--timeout', type=int, default=60)
    ap.add_argument('--parse', help='只解析既有抓取目錄並印摘要')
    a = ap.parse_args()
    if a.parse:
        rows, meta = parse_dir(a.parse)
        by = {}
        for r in rows: by.setdefault((r['source'], r['ym']), 0); by[(r['source'], r['ym'])] += 1
        print(json.dumps(dict(dir=a.parse, n_rows=len(rows), by_source_ym={f'{k[0]} {k[1]}': v for k, v in sorted(by.items())},
                              parse_errors=meta.get('parse_errors'), rejected=meta.get('rejected', False)), ensure_ascii=False, indent=1))
        return 0 if rows else 1
    srcs = [s.strip() for s in a.sources.split(',') if s.strip()]
    for s in srcs:
        if s not in OPENAPI: raise SystemExit(f'未知來源 {s}（可用：{",".join(OPENAPI)}）')
    d, meta = fetch(a.out_root, srcs, not a.no_mops, a.timeout)
    good = [q for q in meta['requests'] if q['status'] == 200 and q.get('n_rows', 0) > 0]
    for q in meta['requests']:
        print(f"  {q['source']:<12} status={q['status']} rows={q.get('n_rows', '-')} yms={q.get('yms', '-')} 完成 {q['done_taipei']}"
              + (f"  錯誤：{q['error']}" if q.get('error') else ''))
    print(f'寫入 {d}（FETCH.json、SHA256SUMS）')
    if not good:
        print('【官方月營收抓取失敗】所有來源都失敗（常見原因：網路政策擋下官方主機、代理回 403、官方網站維護）。'
              '目錄與 FETCH.json 已保留作為失敗紀錄；請到公開資訊觀測站人工核對月營收。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

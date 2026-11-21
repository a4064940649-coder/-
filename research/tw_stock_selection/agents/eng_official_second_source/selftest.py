#!/usr/bin/env python3
"""離線自我測試（不連網）。用法：python3 -I -B selftest.py [<快照根>]；全部通過印 OK、回 0。"""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_official as F, ss_lib as L          # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else F.SNAP_DEFAULT
n = 0


def ok(cond, msg):
    global n
    if not cond:
        print('失敗：' + msg); sys.exit(1)
    n += 1


ok(F.roc_ym('11510') == '2026-10' and F.roc_ym('9901') == '2010-01', '民國年月')
ok(F.roc_date('115年11月02日') == '2026-11-02', '民國日期')
ok(F.trim_num('1,234.50') == '1234.5' and F.trim_num('5.000000') == '5' and F.trim_num('0.138000') == '0.138' and F.trim_num('--') == '', '數字去尾 0')
ok(F.clean_num('17.828005954078908', pct=True) == '17.82' and F.clean_num('-19.178', pct=True) == '-19.17' and F.clean_num('abc') == '', '百分比無條件捨去')
ok(F.looks_like_webpage(b'  <!DOCTYPE html><html>') and not F.looks_like_webpage(b'[{"a":1}]'), '網頁偵測')
for bad in (b'[{"\xe5\x85\xac\xe5\x8f\xb8\xe4\xbb\xa3\xe8\x99\x9f":"=cmd()","\xe8\xb3\x87\xe6\x96\x99\xe5\xb9\xb4\xe6\x9c\x88":"11510"}]',):
    rows, info = F.parse_revenue(bad); ok(rows == [] and info['n_bad'] == 1, '怪代號要丟掉')
# TWTAUU：用 2026-11-21 探查留下的欄位與 2 筆樣本
body = json.dumps(dict(stat='OK', fields=['恢復買賣日期', '股票代號', '名稱', '停止買賣前收盤價格', '恢復買賣參考價', '漲停價格', '跌停價格', '開盤競價基準', '除權參考價', '減資原因', '詳細資料'],
                       data=[['115/08/03', '1459', '聯發', '11.85', '12.46', '13.70', '11.25', '12.45', '--', '退還股款', '1459  ,20260722'],
                             ['115/08/24', '6176', '瑞儀', '81.30', '105.06', '115.50', '94.60', '105.00', '--', '退還股款', '6176  ,20260812']]), ensure_ascii=False).encode()
rows, info = F.parse_twtauu(body)
ok(rows == [['2026-08-03', '1459', '11.85', '12.46', '退還股款'], ['2026-08-24', '6176', '81.3', '105.06', '退還股款']], 'TWTAUU 解析')
# 快照去重與請求上限（暫存資料夾，不連網）
with tempfile.TemporaryDirectory() as td:
    r = dict(ok=True, status=200, body=b'[1]', sha256=F.sha(b'[1]'), fetched_utc='2026-01-01T00:00:00Z')
    p1, new1 = F.store_snapshot(td, 'revenue', 'twse', 'u', r)
    p2, new2 = F.store_snapshot(td, 'revenue', 'twse', 'u', dict(r, fetched_utc='2026-01-02T00:00:00Z'))
    p3, new3 = F.store_snapshot(td, 'revenue', 'twse', 'u', dict(r, body=b'[2]', sha256=F.sha(b'[2]'), fetched_utc='2026-01-03T00:00:00Z'))
    ok(new1 and not new2 and new3 and p1 == p2 and len(list((Path(td) / 'revenue').glob('*.gz'))) == 2, '內容相同不重存')
    ok(F.gz_read(p1) == b'[1]' and len(F.read_csv(Path(td) / 'revenue' / 'index.csv')) == 3, '索引三列')
    pol = F.Polite(td); now = F.utcnow()
    for i in range(4):
        F.append_csv(pol.ledger, F.REQ_HDR, dict(utc=now.strftime('%Y-%m-%dT%H:%M:%SZ'), taipei_date=now.astimezone(F.TPE).strftime('%Y-%m-%d'), host='www.twse.com.tw', url='x', attempt=1, http_status=200, result='ok'))
    g = pol.get('https://www.twse.com.tw/rwd/zh/exRight/TWT49U?response=json')
    ok(g['result'] == 'skipped' and '達上限 4' in g['note'], '證交所主站一天 4 次上限（沒有真的送出請求）')
    F.append_csv(pol.ledger, F.REQ_HDR, dict(utc=now.strftime('%Y-%m-%dT%H:%M:%SZ'), taipei_date=now.astimezone(F.TPE).strftime('%Y-%m-%d'), host='openapi.twse.com.tw', url='x', attempt=1, http_status=307, result='blocked'))
    g = pol.get('https://openapi.twse.com.tw/v1/opendata/t187ap05_L')
    ok(g['result'] == 'skipped' and '被擋' in g['note'], '被擋後當天不再請求')
# 真實快照（有的話）：sha256 對得上索引、能解析
for kind in ('revenue', 'exright'):
    for r in F.read_csv(root / kind / 'index.csv'):
        if r['stored_file']:
            b = F.gz_read(root / kind / r['stored_file']); ok(F.sha(b) == r['sha256'], f"{r['stored_file']} sha256")
            rows, info = (F.parse_revenue if kind == 'revenue' else F.parse_twt49u)(b); ok(len(rows) == int(r['n_rows']), f"{r['stored_file']} 列數")
print(f'OK（{n} 項）')

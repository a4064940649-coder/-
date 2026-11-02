#!/usr/bin/env python3
"""解析投信投顧公會（SITCA）月度「基金績效評比」（邱顯比、李存修教授製作）——來源：GitHub 公開封存 z-Wind/stockTools
extraData/TW_Analysis/基金績效評比/YYYYMM.zip（raw/zwind_stockTools，視為不可信，只讀不執行）。
每月每檔基金取：分類標籤、一個月報酬（若有）、自今年以來報酬（YTD）、成立日（第二部份）、公佈時間。
輸出：data/fund_panel_long.parquet（月×基金長表）、data/fund_parse_meta.json。
用法：python3 -I -B s2a_fund_parse.py [YYYYMM ...]（不給參數＝全部）
"""
import io, re, sys, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd, xlrd

HERE = Path(__file__).resolve().parent
ZDIR = HERE / 'raw' / 'zwind_stockTools' / 'extraData' / 'TW_Analysis' / '基金績效評比'
OUT = HERE / 'data'; OUT.mkdir(exist_ok=True)

def num(x):
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace(',', '')
    try: return float(s)
    except ValueError: return np.nan

def clean(x):
    return re.sub(r'\s+', '', str(x)).replace('﹒', '.')

LABEL_RE = re.compile(r'^([一二三四五六七八九十]+[、.]|\d+[.]|[a-z][.]|[ivx]+[.]|\([0-9a-z]+\)[.])')

def parse_book(book):
    rows = []
    pubtime = None
    for sh in book.sheets():
        labels = []
        mode = None; col = {}
        for r in range(sh.nrows):
            v = sh.row_values(r)
            cv = [clean(c) for c in v]
            ne = [(i, c) for i, c in enumerate(cv) if c]
            if not ne:
                continue
            joined = ''.join(c for _, c in ne)
            if pubtime is None and '公佈時間' in joined:
                m = re.search(r'(\d{4})/(\d{1,2})/(\d{1,2})', joined)
                if m: pubtime = f'{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}'
            if ne[0][1] == '基金名稱':
                hdr = {c: i for i, c in ne}
                if '自今年以來' in hdr or '一個月' in hdr:
                    mode = 'p1'; col = {'m1': hdr.get('一個月'), 'ytd': hdr.get('自今年以來'), 'm3': hdr.get('三個月'), 'y1': hdr.get('一年')}
                elif '自成立日' in hdr or '基金' in hdr:
                    mode = 'p2'
                    # 成立日欄：第二列標題「成立日」所在欄
                    col = {'_p2hdr': r}
                else:
                    mode = None
                continue
            if mode == 'p2' and '_p2hdr' in col and r == col['_p2hdr'] + 1:
                h2 = {c: i for i, c in ne}
                col = {'inc': h2.get('成立日')}
                continue
            if ne[0][1] in ('報酬率',) or joined.startswith('報酬率排名'):
                continue
            if len(ne) <= 3 and LABEL_RE.match(ne[0][1]) and not any(np.isfinite(num(c)) for _, c in ne[1:2]):
                # 分類標籤列（目錄頁也會符合，但目錄頁之後才有資料區塊，標籤會被新的頁首覆蓋）
                lab = ne[0][1]
                lvl = 0 if re.match(r'^[一二三四五六七八九十]+', lab) else (1 if re.match(r'^\d', lab) else (2 if re.match(r'^[a-z][.]', lab) else (3 if re.match(r'^[ivx]+[.]', lab) else 4)))
                labels = [l for l in labels if l[0] < lvl] + [(lvl, lab)]
                mode = None if mode in ('p1', 'p2') else mode
                continue
            if '(第一部份)' in joined or '(第二部份)' in joined or '報酬率單位' in joined:
                continue
            if mode == 'p1' and ne[0][0] == 0:
                name = cv[0]
                vals = {k: (num(v[i]) if i is not None and i < len(v) else np.nan) for k, i in col.items()}
                if all(not np.isfinite(x) for x in vals.values()) and not any(clean(v[i]) == '-' for i in col.values() if i is not None and i < len(v)):
                    continue
                rows.append(dict(sheet=sh.name, cat='|'.join(l for _, l in labels), name=name, **vals, part=1))
            elif mode == 'p2' and ne[0][0] == 0 and col.get('inc') is not None:
                inc = str(v[col['inc']]).strip() if col['inc'] < len(v) else ''
                if isinstance(v[col['inc']], float) and v[col['inc']] > 1000:
                    try: inc = xlrd.xldate_as_datetime(v[col['inc']], book.datemode).strftime('%Y-%m-%d')
                    except Exception: pass
                rows.append(dict(sheet=sh.name, cat='|'.join(l for _, l in labels), name=cv[0], inc=inc, part=2))
    return rows, pubtime

def parse_zip(p):
    z = zipfile.ZipFile(p)
    out = []; pub = None; files = []
    for nm in z.namelist():
        if not nm.lower().endswith(('.xls', '.xlsx')): continue
        b = z.read(nm); files.append(nm)
        bk = xlrd.open_workbook(file_contents=b, encoding_override='cp950')
        rows, pt = parse_book(bk); out += rows; pub = pub or pt
    return out, pub, files

if __name__ == '__main__':
    zs = sorted(ZDIR.glob('*.zip'))
    if len(sys.argv) > 1: zs = [ZDIR / f'{a}.zip' for a in sys.argv[1:]]
    allr = []; meta = {}
    for p in zs:
        ym = p.stem
        try:
            rows, pub, files = parse_zip(p)
        except Exception as e:
            meta[ym] = {'error': repr(e)[:200]}; print(ym, 'ERR', e); continue
        for r in rows: r['ym'] = ym
        allr += rows
        n1 = sum(r['part'] == 1 for r in rows)
        meta[ym] = {'pubtime': pub, 'n_part1_rows': n1, 'files': files, 'has_m1': any(np.isfinite(r.get('m1', np.nan)) for r in rows if r['part'] == 1)}
        print(ym, pub, n1, meta[ym]['has_m1'], flush=True)
    df = pd.DataFrame(allr)
    tag = 'sample' if len(sys.argv) > 1 else 'all'
    df.to_parquet(OUT / f'fund_panel_long_{tag}.parquet')
    (OUT / f'fund_parse_meta_{tag}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1))

"""Parse SITCA monthly 基金績效評比 xls (z-Wind/stockTools archive) -> long table of domestic active equity funds.
Usage: python3 -I -B r3_fund_parse.py <dir with YYYYMM/*.xls> <out parquet>
"""
import sys, glob, os, re
import xlrd
import pandas as pd
from multiprocessing import Pool

CATS = {'科技類': 'tech', '中小型': 'smallmid', '價值型': 'value', '一般股票型': 'general', '上櫃股票型': 'otc'}
DOM_PARENT = re.compile(r'(國內股市|上市股票型|投資國內)')
PFX = re.compile(r'^\s*[\(（]?\s*([0-9]+|[a-zA-Z]|[ivx]+)\s*[\)）\.﹒、．]\s*')


def clean(x):
    return str(x).replace('　', ' ').strip()


def num(x):
    s = clean(x)
    if s in ('', '-', '--', 'N/A'):
        return None
    try:
        return float(s.replace(',', ''))
    except ValueError:
        return None


def parse_file(path):
    ym = os.path.basename(os.path.dirname(path))
    wb = xlrd.open_workbook(path, on_demand=True)
    s = wb.sheet_by_index(0)
    out = []
    labels = []  # consecutive single-cell label rows
    last_single = -10
    cat = None; part = None; cmap = None
    for r in range(s.nrows):
        row = [clean(c) for c in s.row_values(r)]
        nz = [c for c in row if c]
        if not nz:
            continue
        if len(nz) == 1 and len(nz[0]) <= 30 and num(nz[0]) is None:
            t = nz[0]
            if '第一部' in t:
                part = 1; cmap = None; continue
            if '第二部' in t:
                part = 2; cmap = None; continue
            if r == last_single + 1:
                labels.append(t)
            else:
                labels = [t]
            last_single = r
            child = PFX.sub('', labels[-1]).strip()
            parent = labels[-2] if len(labels) >= 2 else ''
            if child in CATS and (DOM_PARENT.search(parent) or (child == '上櫃股票型')):
                cat = CATS[child]
            else:
                cat = None
            part = None; cmap = None
            continue
        if any('第一部' in c for c in row):
            part = 1; cmap = None; continue
        if any('第二部' in c for c in row):
            part = 2; cmap = None; continue
        if cat is None or part is None:
            continue
        if row[0] == '基金名稱':
            nxt = [clean(c) for c in s.row_values(r + 1)] if r + 1 < s.nrows else [''] * len(row)
            cmap = {}
            for j, h in enumerate(row):
                h2 = nxt[j] if j < len(nxt) else ''
                if h == '一個月':
                    cmap['m1'] = j if h2.startswith('報酬') else j
                elif h == '自今年以來':
                    cmap['ytd'] = j
                elif h == '基金' and h2 == '成立日':
                    cmap['incep'] = j
                elif h == '基金統編':
                    cmap['code'] = j
            # 2006 style: '一個月' header col holds the return
            continue
        if cmap is None or not row[0] or row[0] == '基金名稱':
            continue
        if row[0].startswith('平均') or row[0].startswith('註') or row[0].startswith('※'):
            continue
        rec = {'ym': ym, 'cat': cat, 'part': part, 'name': row[0]}
        if part == 1:
            if 'm1' in cmap: rec['m1'] = num(row[cmap['m1']])
            if 'ytd' in cmap: rec['ytd'] = num(row[cmap['ytd']])
            if 'm1' not in cmap and 'ytd' not in cmap:
                continue
        else:
            if 'incep' in cmap: rec['incep'] = row[cmap['incep']]
            if 'code' in cmap: rec['code'] = row[cmap['code']]
        out.append(rec)
    wb.release_resources()
    return out


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(src, '*', '*.xls')))
    files = [f for f in files if '200608' <= os.path.basename(os.path.dirname(f)) <= '202609']
    with Pool(8) as p:
        res = p.map(parse_file, files)
    df = pd.DataFrame([x for r in res for x in r])
    df.to_parquet(dst)
    print(df.shape, df.ym.nunique())
    print(df.groupby(['ym', 'cat', 'part']).size().unstack([1, 2]).iloc[::24])

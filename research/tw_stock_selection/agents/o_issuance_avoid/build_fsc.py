# -*- coding: utf-8 -*-
"""Parse FSC 申報案件彙總表 (raw/fsc_cases, untrusted) into one normalized CSV.

Source: KiwiLee0925/Convertible-Bond-Issuance-Timing-and-Stock-Price-Behavior-in-Taiwan
zip '可轉債案件組合包 (extract.me).zip' (copies of 金管會證期局 yearly filing summaries).
Old format (ROC 89~104): numeric company type/result, ROC dates 'yy/mm/dd', unit 仟元.
New format (106~114): text types, dates 'yyymmdd', unit 元.
Output: fsc_cases_all.csv
"""
import os, re, sys, glob
import openpyxl
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw', 'fsc_cases')

FILES = {}
for y in range(89, 105):
    p = os.path.join(RAW, 'conv', f'case-{y}.xlsx')
    if y == 103:
        p = os.path.join(RAW, 'case-103.xlsx')
    FILES[y] = p
for y in (98, 99, 102):
    FILES[y] = os.path.join(RAW, f'recovered_{y}.csv')
FILES[106] = os.path.join(RAW, 'conv', 'case10612.xlsx')
FILES[107] = os.path.join(RAW, 'conv', 'case10712.xlsx')
FILES[108] = os.path.join(RAW, 'conv', 'case10812.xlsx')
FILES[109] = os.path.join(RAW, 'conv', 'case10912.xlsx')
FILES[110] = os.path.join(RAW, 'conv', 'case11012(xls).xlsx')
FILES[111] = os.path.join(RAW, 'case11112.xlsx')
FILES[112] = os.path.join(RAW, '112729申報案件彙總表 .xlsx')
FILES[113] = os.path.join(RAW, '11311473申報案件彙總表.xlsx')
FILES[114] = os.path.join(RAW, '1147281申報案件彙總表.xlsx')

OLD_TYPE = {'1': '上市', '2': '上櫃', '3': '未上市櫃', '4': '補辦公發', '5': '興櫃', '6': '外國企業'}
OLD_RES = {'1': '生效', '2': '退件', '3': '核減', '4': '自行撤回', '6': '撤銷'}


def roc_date(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s or s in ('None', 'nan'):
        return None
    s = s.split(',')[0].split('\n')[0].strip()
    m = re.match(r'^(\d{2,3})/(\d{1,2})/(\d{1,2})$', s)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        s2 = s.replace('.0', '')
        m = re.match(r'^(\d{2,3})(\d{2})(\d{2})$', s2)
        if not m:
            return None
        y, mo, d = map(int, m.groups())
    try:
        return pd.Timestamp(year=y + 1911, month=mo, day=d)
    except Exception:
        return None


def num(s):
    if s is None:
        return None
    s = str(s).replace(',', '').strip()
    try:
        return float(s)
    except Exception:
        return None


def clean_code(s):
    if s is None:
        return None
    s = str(s).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


rows = []
for y, p in sorted(FILES.items()):
    if not os.path.exists(p):
        print('missing', y, p)
        continue
    rec = os.path.join(RAW, f'recovered_{y}.csv')
    if y in (98, 99, 102) and os.path.exists(rec):
        # truncated xls: rows recovered by recover_biff.py (partial year)
        d = pd.read_csv(rec, dtype={'value': str})
        w = d.pivot_table(index='row', columns='col', values='value', aggfunc='first')
        w = w.reindex(columns=range(int(w.columns.max()) + 1))
        allr = [tuple(None if (isinstance(v, float) and pd.isna(v)) else v for v in row) for row in w.itertuples(index=False)]
        p = rec
    else:
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        allr = list(ws.iter_rows(values_only=True))
    hdr_i = None
    for i, r in enumerate(allr[:40]):
        if r and r[0] is not None and str(r[0]).strip() == '證券代號':
            hdr_i = i
            break
    if hdr_i is None:
        print('no header', y, p, len(allr))
        continue
    hdr = [re.sub(r'[\s\u3000\xa0]', '', str(c)) if c is not None else '' for c in allr[hdr_i]]
    new = '案件類別' in hdr
    unit = 1.0 if new else 1000.0
    idx = {h: j for j, h in enumerate(hdr)}
    for h, j in list(idx.items()):
        if h.startswith('公司名稱'):
            idx['公司名稱'] = j
    item_col = idx.get('案件類別', idx.get('申報事項'))
    amt_col = [j for j, h in enumerate(hdr) if h.startswith('金額')][0]
    eff_col = idx.get('生效日期', idx.get('申報生效日期'))
    n0 = 0
    for r in allr[hdr_i + 1:]:
        if r is None or r[0] is None:
            continue
        code = clean_code(r[0])
        if not code or not re.match(r'^[0-9A-Z]{4,6}$', code):
            continue
        g = lambda j: r[j] if j is not None and j < len(r) else None
        ctype = str(g(idx['公司型態'])).strip().replace('.0', '')
        res = str(g(idx['結案類型'])).strip().replace('.0', '')
        if not new:
            ctype = OLD_TYPE.get(ctype, ctype)
            res = OLD_RES.get(res, res)
        rows.append(dict(
            roc_year=y, code=code, ctype=ctype, result=res,
            name=str(g(idx['公司名稱'])).strip(), underwriter=str(g(idx.get('承銷商')) or '').strip(),
            item=str(g(item_col)).strip(), amount=(num(g(amt_col)) or 0) * unit,
            ccy=str(g(idx.get('幣別')) or ''),
            price=num(g(idx['發行價格'])),
            recv=roc_date(g(idx['收文日期'])), eff=roc_date(g(eff_col)),
            note1=str(g(idx.get('備註1')) or '') if not new else '',
            note2=str(g(idx.get('備註2')) or '') if not new else '',
            src=os.path.basename(p)))
        n0 += 1
    print(y, os.path.basename(p), 'rows', n0)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, 'fsc_cases_all.csv'), index=False)
print(len(df))
print(df.groupby('roc_year').agg(n=('code', 'size'), rmin=('recv', 'min'), rmax=('recv', 'max')).to_string())
print(df['item'].value_counts().head(60).to_string())

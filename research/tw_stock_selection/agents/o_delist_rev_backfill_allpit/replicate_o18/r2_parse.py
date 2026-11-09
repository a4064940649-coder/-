#!/usr/bin/env python3
"""G6 複製員 r2：從 raw/ 兩個原始檔自寫解析器（不讀研究員 *.py）。
  A：raw/taiwanan_monthly_lastver.csv.gz → 長表 (code, ym_row, rev, prev, ly, yoy)；值單位千元。
  N：raw/news_rev_segments.csv.gz → (code, ym, rev_k, seg) 依凍結規格 P5 的文字規則自寫。
python3 -I -B r2_parse.py"""
import csv, io, re, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TOP = HERE.parent
AG = TOP.parent
RAW = TOP / 'raw'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o18')

# ============================== A
d = pd.read_csv(RAW / 'taiwanan_monthly_lastver.csv.gz', dtype=str, keep_default_na=False)
rows = []; nbad = 0
for code, ln, line in zip(d.code, d.line_no, d.line):
    if ln in ('0', '1'): continue
    r = next(csv.reader(io.StringIO(line)))
    m = re.fullmatch(r'(\d{4})/(\d{2})', r[0].strip())
    if not m or len(r) < 6: nbad += 1; continue
    def num(x):
        x = x.strip().replace(',', '')
        try: return float(x)
        except ValueError: return np.nan
    rows.append(dict(code=code, ym=f'{m.group(1)}-{m.group(2)}', rev=num(r[1]), prev=num(r[2]), ly=num(r[3]), yoy=num(r[5])))
A = pd.DataFrame(rows)
ndup = int(A.duplicated(['code', 'ym']).sum())
A = A.drop_duplicates(['code', 'ym'], keep='last')
A.to_csv(SCR / 'r2_A.csv', index=False)
print('A rows', len(A), 'bad', nbad, 'dup', ndup, 'codes', A.code.nunique())

# ============================== 公司代碼與名稱
tw = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype=str)
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str)
r5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype=str, usecols=['code'])
cols = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/close_raw.parquet').columns
KNOWN = {c for c in set(tw.code) | set(sl.code) | set(r5.code) | set(cols) | set(A.code) if re.fullmatch(r'\d{4}', str(c))}
ALIAS = {}
def addn(c, n):
    n = str(n).strip()
    if not n or n == 'nan' or not re.fullmatch(r'\d{4}', c): return
    for v in {n, re.sub(r'[-－]?KY$', '', n), re.sub(r'^F-', '', n), n.replace('*', '')}:
        v = v.strip()
        if len(v) >= 2: ALIAS.setdefault(v, set()).add(c)
for c, n in zip(tw.code, tw.name): addn(c, n)
for c, n in zip(sl.code, sl.name): addn(c, n)
for f in ('grs_568e6685_tools_twse_list.csv', 'grs_893c8772_grs_stock_no.csv', 'goristock_c3c718ab_grs_stock_no.csv'):
    p = AG / 'o_pit_shares_industry_2006_2013/raw/industry_snap' / f
    if p.exists():
        for r in csv.reader(open(p, encoding='utf-8', errors='replace')):
            r = [x.strip().strip('"') for x in r]
            if len(r) >= 2 and r[0].isdigit() and len(r[0]) == 4: addn(r[0], r[1])
for c, n in zip(d.code[d.line_no == '0'], d.line[d.line_no == '0']): addn(c, n.split(',')[0])
BYFIRST = {}
for a in ALIAS: BYFIRST.setdefault(a[0], []).append(a)
for k in BYFIRST: BYFIRST[k].sort(key=len, reverse=True)
def is_han(ch): return '一' <= ch <= '鿿'

def company(seg):
    """回傳 (code or None, reason)。"""
    codes = set()
    for m in re.finditer(r'[（(]\s*(\d{4})\s*[)）]', seg):
        tail = seg[m.end(1):m.end(1) + 3]
        if re.match(r'\s*年', tail): continue
        codes.add(m.group(1))
    if codes:
        if any(c not in KNOWN for c in codes): return None, 'unknown_code'
        if len(codes) > 1: return None, 'multi'
        return next(iter(codes)), 'code'
    hits = []  # (start, end, alias)
    i = 0; L = len(seg)
    for i in range(L):
        for a in BYFIRST.get(seg[i], ()):
            if seg.startswith(a, i):
                if len(a) == 2 and i > 0 and is_han(seg[i - 1]): continue
                hits.append((i, i + len(a), a)); break
    # 最長匹配：被更長命中完全包含的命中丟掉
    keep = [h for h in hits if not any((g[0] <= h[0] and g[1] >= h[1] and (g[1] - g[0]) > (h[1] - h[0])) for g in hits)]
    cs = set()
    for _, _, a in keep: cs |= ALIAS[a]
    if not cs: return None, 'none'
    if len(cs) > 1: return None, 'multi'
    return next(iter(cs)), 'name'

THRESH = ('站上', '突破', '大關', '衝上', '上看', '挑戰', '逾', '超過', '站穩', '以上', '關卡', '水準', '破')
SEGW = ('事業', '部門', '業務', '產品', '子公司', '轉投資', '封測', '封裝', '材料', '代工', '品牌', '通路', '本業', '電子')
CUM = ('前', '累', '至', '~', '～', '-', '－', '—', '到')
MON = re.compile(r'(?:(\d{1,2})\s*月份?|元月份?|上月份?)')
NUM = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*(億|萬)\s*元')
YOY = re.compile(r'年(增|減)\s*(?:率)?\s*(\d+\.\d+)\s*[%％]')

def months(seg, adate):
    out = []
    for m in MON.finditer(seg):
        tok = m.group(0)
        if tok.startswith('上月'):
            pre = seg[max(0, m.start() - 1):m.start()]
            if pre in ('較', '比', '與', '於'): continue
            mm = (pd.Period(adate, 'M') - 1)
            mon = mm.month
        elif tok.startswith('元月'):
            mon = 1
        else:
            mon = int(m.group(1))
            if not 1 <= mon <= 12: continue
            # 「12 月 31 日」之類的日期、或數字前面還接數字（如 2019 被誤切）
            if m.start() > 0 and seg[m.start() - 1].isdigit(): continue
        after = seg[m.end():m.end() + 6]
        if '營收' not in after: continue
        before8 = seg[max(0, m.start() - 8):m.start()]
        if any(x in before8 for x in CUM): continue
        # 年份
        ad = pd.Timestamp(adate)
        cands = []
        for y in (ad.year, ad.year - 1):
            me = pd.Period(f'{y}-{mon:02d}', 'M').to_timestamp(how='end').normalize()
            if 0 <= (ad - me).days <= 75: cands.append(y)
        if len(cands) != 1: continue
        y = cands[0]
        pre = seg[max(0, m.start() - 8):m.start()]
        ym_ = re.search(r'(\d{4})\s*年\s*$', pre)
        if ym_ and int(ym_.group(1)) != y: continue
        if re.search(r'去年\s*$', pre) and y != ad.year - 1: continue
        if re.search(r'今年\s*$', pre) and y != ad.year: continue
        out.append((m, f'{y}-{mon:02d}'))
    return out

VAR = sys.argv[1] if len(sys.argv) > 1 else 'base'
def level(seg, m):
    """月份 token m 之後 14 字內第一個「X億元／X萬元」，套排除規則。回傳 (rev_k or None, reason, yoy)。
    VAR：base＝門檻字／事業別只看月份到數字之間；wide＝門檻字（站上、突破、大關…）與事業別字眼在整段出現就丟。"""
    if VAR == 'wide':
        if any(t in seg for t in ('站上', '突破', '大關', '衝上', '上看', '挑戰', '站穩', '關卡', '失守', '跌破')): return None, 'thresh_seg', None
        if re.search(r'(事業|部門|封測及材料|封裝測試及材料|子公司|轉投資)[^，。]{0,6}營收', seg): return None, 'segment_seg', None
    w = seg[m.end():]
    n = NUM.search(w)
    if not n or n.start() > 14: return None, 'no_num', None
    span = w[:n.start()]
    ctx = seg[max(0, m.start() - 6):m.start()] + span
    if '約' in ctx or '估' in ctx: return None, 'approx', None
    if any(t in span for t in THRESH) or any(t in w[n.end():n.end() + 3] for t in ('大關', '以上', '關卡')): return None, 'thresh', None
    if '、' in span or '分別' in span or re.search(r'\d\s*[、及和與]\s*$', seg[max(0, m.start() - 3):m.start()]): return None, 'enum', None
    if re.search(r'[，,]', span):
        last = re.split(r'[，,]', span)[-1]
        if not re.fullmatch(r'\s*(?:合併)?營收(?:為|達)\s*(?:新台幣|新臺幣)?\s*', last): return None, 'comma', None
    gap = span.split('營收')[0] if '營收' in span else span
    if any(x in gap for x in SEGW) or any(x in seg[max(0, m.start() - 6):m.start()] for x in ('轉投資', '子公司', '事業', '部門')): return None, 'segment', None
    x = n.group(1).replace(',', ''); unit = n.group(2)
    dec = len(x.split('.')[1]) if '.' in x else 0
    if unit == '億':
        if dec < 2: return None, 'dec', None
        v = float(x) * 1e5; u = 1e5 * 10 ** (-dec)
    else:
        v = float(x) * 10.0; u = 10.0 * 10 ** (-dec)
    level.unit = u
    if v < 1e5: return None, 'lt1e8', None
    yy = None
    t = YOY.search(w[n.end():n.end() + 40])
    if t: yy = float(t.group(2)) * (1 if t.group(1) == '增' else -1)
    return v, 'ok', yy

NW = pd.read_csv(RAW / 'news_rev_segments.csv.gz', dtype=str, keep_default_na=False)
out = []; reasons = {}
for src, art, date, si, seg in zip(NW.src, NW.art, NW.date, NW.seg_i, NW.seg):
    seg = seg.replace('　', ' ')
    if '營收' not in seg or '月' not in seg: continue
    c, why = company(seg)
    if c is None: reasons[why] = reasons.get(why, 0) + 1; continue
    ad = pd.Timestamp(date)
    for m, ym in months(seg, ad):
        v, why2, yy = level(seg, m)
        reasons[why2] = reasons.get(why2, 0) + 1
        if v is None: continue
        out.append(dict(src=src, art=art, date=date, seg_i=si, code=c, ym=ym, rev_k=v, unit_k=level.unit, yoy=yy, how=why, seg=seg))
NV = pd.DataFrame(out).drop_duplicates(['art', 'seg_i', 'code', 'ym', 'rev_k'])
NV.to_csv(SCR / ('r2_N.csv.gz' if VAR == 'base' else f'r2_N_{VAR}.csv.gz'), index=False)
print('N vals', len(NV), 'codes', NV.code.nunique(), reasons)
json.dump(dict(A_rows=len(A), A_dup=ndup, A_bad=nbad, N_vals=len(NV), N_reasons=reasons), open(HERE / ('r2_parse.json' if VAR == 'base' else f'r2_parse_{VAR}.json'), 'w'), ensure_ascii=False, indent=1)

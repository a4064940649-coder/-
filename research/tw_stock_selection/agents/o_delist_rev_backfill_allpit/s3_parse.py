#!/usr/bin/env python3
r"""步驟 2(b)(c) 新聞解析器（本題自寫；只讀 raw/news_rev_segments.csv.gz 文字）。版本由參數決定（全部記入 iterations.json）。
抽「<公司> <N>月（合併）營收 <X> 億／萬元」與緊接的「年增／年減 Y%」。輸出 data/news_vals_<ver>.parquet。
P1 規則（寫死）：
  公司：段內出現明確代碼「(dddd」→ 只用代碼（多於 1 個不同代碼 → 丟棄該段）；否則用名稱別名（最長匹配），段內 > 1 家 → 丟棄；
        段內 0 家時看前一段（同樣規則）。
  月份：「<N>月」或「元月」緊接（≤ 6 字內）營收字樣；前面是「前／1-／1～／至／累計」→ 累計，丟棄。「上月」＝文章月 − 1。
        年份：取使「月底 → 文章日」介於 0～75 天的年份；否則丟棄。
  水準：月份片語之後 ≤ 14 字內第一個「<數字>(億|萬)元」；前 4 字內有「約／逾／近／超過／上看／估／預／挑戰／達到?目標」→ 標 approx 並丟棄；
        數字後緊接「、<數字>」（多值列舉）→ 丟棄。
  年增：水準之後到段尾或下一個「前<N>月／累計／第.季／Q\d／今年／全年／上半年」之前，第一個
        「年增(率|幅)?(為|達|約)?<Y>%」「年減 → 負」「較去年同期(成長|增加|增|提升|上升) <Y>%」「(衰退|減少|下滑|減) → 負」「年成長／年衰退」。
  口徑標記：營收前 4 字內「合併」→ consol；「個體／母公司」→ standalone；「封測／事業／EMS／自有品牌…營收」→ segment（丟棄）。
P2：代碼必須是已知 4 碼代碼且後面不接「年」；不用前一段補公司；月份前 8 字（去空白）含「前／累／至／~／-」→ 累計；月份前有明寫年份且與推定年份不同 → 丟棄；
  水準前後的門檻字（站上／跳過／突破／大關／站穩／以上／逾／超過／衝破／守穩／之上）→ 丟棄；只保留億元 ≥ 2 位小數或萬元整數（精度 ≤ 0.01 億）。
P3：段內有明確代碼但都不是已知代碼 → 丟棄（不退回別名）；月份前 6 字內有事業別字樣（事業／產品／部門／封測／EMS…）→ 丟棄；
  水準前的間隔含逗號時，逗號後只能是「（合併）營收（則）?為／達」；水準 < 1.00 億元 → 丟棄（捨入誤差會超過 0.5%）。
P4：年增只收 ≥ 1 位小數者（整數年增捨入誤差 0.5pp > 0.1pp 閘門）；2 字別名前一字是中文字（可能是較長名稱的一部分，例：世界先進→先進）→ 不算。
P5：「較／比／與／於上月…營收」（指被報導月份的前一月，不是文章月 − 1）→ 不當成月份片語。
  （之後版本說明見 iterations.json。）python3 -I -B s3_parse.py P1"""
import re, sys, json, hashlib, gzip, csv
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
VER = sys.argv[1] if len(sys.argv) > 1 else 'P1'
# ---------------- 名稱 ----------------
names = {}
def add(c, n):
    if isinstance(c, str) and re.fullmatch(r'\d{4}', c) and isinstance(n, str):
        n = re.sub(r'\s', '', n).replace('*', '')
        if len(n) >= 2 and not n.isdigit(): names.setdefault(c, set()).add(n)
lr = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}, usecols=['code', 'name']).drop_duplicates()
for c, n in zip(lr.code, lr.name): add(c, n)
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str)
for c, n in zip(sl.code, sl.name): add(c, n)
tw = pd.read_csv(HERE / 'raw/taiwanan_monthly_lastver.csv.gz', dtype=str)
for c, n in zip(tw[tw.line_no == '0'].code, tw[tw.line_no == '0'].line): add(c, n.split(',')[0])
SUF = ['股份有限公司', '-KY', 'KY', '科技', '電子', '實業', '工業', '企業', '光電', '國際', '半導體', '機電', '能源', '通訊', '化學', '精密', '控股']
for c in list(names):
    for n in list(names[c]):
        for s in SUF:
            if n.endswith(s) and len(n) - len(s) >= 2: names[c].add(n[:-len(s)])
inv = {}
for c, ns in names.items():
    for n in ns: inv.setdefault(n, set()).add(c)
ALIAS = {n: next(iter(cs)) for n, cs in inv.items() if len(cs) == 1}
AMBIG = {n for n, cs in inv.items() if len(cs) > 1}
# 常見非公司詞（避免 2 字別名誤中）：只排除與一般詞相同的別名
STOP = {'營收', '台灣', '中華', '大眾', '國際', '全球', '世界', '統一', '亞洲', '東南', '新光', '台新', '國泰', '永豐', '大同', '聯合', '光寶', '宏全'} - set()
ALIAS = {n: c for n, c in ALIAS.items() if n not in {'台灣', '全球', '世界', '亞洲', '東南', '國際', '中華', '大眾', '聯合', '營收'}}
AL_RE = re.compile('|'.join(sorted(map(re.escape, ALIAS), key=len, reverse=True)))
CODE_RE = re.compile(r'[\(（]\s*(\d{4})(?:\s*-\s*(?:TW|TWO|tw))?\s*[\)）](?!\s*年)')
def companies(s):
    cs = set(CODE_RE.findall(s))
    if VER >= 'P2':
        c0 = cs; cs = {c for c in cs if c in names}
        if VER >= 'P3' and c0 and not cs: return set(), 'badcode'
    if cs: return cs, 'code'
    if VER >= 'P4':
        out = set()
        for m in AL_RE.finditer(s):
            n = m.group(0)
            if len(n) <= 2 and m.start() > 0 and re.match(r'[\u4e00-\u9fff]', s[m.start() - 1]): continue
            out.add(ALIAS[n])
        return out, 'alias'
    return {ALIAS[m] for m in AL_RE.findall(s)}, 'alias'
CN = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12}
MON_RE = re.compile(r'(?P<pre>.{0,10})(?P<m>(?:1[0-2]|0?[1-9])\s*月|元月|上月|(?:十一|十二|十|[一二三四五六七八九])月)(?P<mid>[^，,。；\d]{0,6}?)營收')
NUM = r'(?P<v>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<u>億|萬)元'
LEV_RE = re.compile(r'(?P<gap>[^\d]{0,14}?)' + NUM + r'(?P<after>.{0,3})')
CUM_PRE = re.compile(r'(前|1\s*[-－~～至到]\s*|累計|今年|全年|上半年|一至|1至)$')
APPROX = re.compile(r'(約|逾|近|超過|上看|估|預|挑戰|目標|將達|可望|有望|不到|突破)')
YOY_RE = re.compile(r'(?:年(?P<s1>增|減|成長|衰退)(?:率|幅|幅度)?(?:為|達|約|了)?\s*(?P<y1>\d+(?:\.\d+)?)\s*[%％])|'
                    r'(?:較去年同期(?:的?[\d\.,]+\s*億元)?(?P<s2>成長|增加|增|提升|上升|大增|衰退|減少|下滑|減|下降|縮減)(?:了)?\s*(?P<y2>\d+(?:\.\d+)?)\s*[%％])')
STOP_RE = re.compile(r'(前\s*\d+\s*月|累計|第.季|Q\d|今年|全年|上半年|前.季|1\s*[-－~～至]\s*\d+\s*月)')
SEGW = re.compile(r'(封測|事業|EMS|品牌|代工|本業|晶圓|部門|子公司|母公司|個體|自結)')
def parse_month(tok, art):
    tok = re.sub(r"\s", "", tok)
    if tok == '上月': m = (art.month - 2) % 12 + 1
    elif tok == '元月': m = 1
    else:
        t = tok[:-1]; m = int(t) if t.isdigit() else CN[t]
    for y in (art.year, art.year - 1):
        me = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
        if 0 <= (art - me).days <= 75: return pd.Period(f'{y}-{m:02d}', 'M')
    return None
rows = []; nseg = 0; drop = {}
def dr(k): drop[k] = drop.get(k, 0) + 1
RAWF = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'raw/news_rev_segments.csv.gz'
with gzip.open(RAWF, 'rt', encoding='utf-8') as g:
    for r in csv.DictReader(g):
        nseg += 1; s = r['seg']
        try: art = pd.Timestamp(r['date'])
        except Exception: dr('bad_date'); continue
        cs, how = companies(s)
        if not cs and VER < 'P2':
            cs, how = companies(r.get('prev', '')); how = 'prev_' + how
        if len(cs) != 1: dr('n_company_' + ('0' if not cs else '>1')); continue
        code = next(iter(cs))
        for mm in MON_RE.finditer(s):
            pre = mm.group('pre')
            if VER < 'P2':
                if CUM_PRE.search(pre[-3:]): dr('cumulative'); continue
            else:
                p8 = re.sub(r'\s', '', pre)[-8:]
                if re.search(r'[前累至~～\-－到]|\d\s*[、,]$', p8[-4:]) or re.search(r'(累計|累積|合計)', p8): dr('cumulative'); continue
            if VER >= 'P5' and re.sub(r'\s', '', mm.group('m')) == '上月' and re.search(r'(較|比|與|於|于|自|從)\s*$', pre): dr('relative_prev'); continue
            ym = parse_month(mm.group('m'), art)
            if ym is None: dr('month_window'); continue
            if VER >= 'P2':
                yy = re.search(r'((?:19|20)\d{2}|\d{3})\s*年\s*$', pre)
                if yy:
                    y0 = int(yy.group(1)); y0 = y0 + 1911 if y0 < 1900 else y0
                    if y0 != ym.year: dr('explicit_year'); continue
            mid = mm.group('mid'); basis = 'consol' if '合併' in mid else ('segment' if SEGW.search(mid) else 'plain')
            if basis == 'segment': dr('segment'); continue
            if VER >= 'P3' and SEGW.search(re.sub(r'\s', '', pre)[-6:]): dr('segment_pre'); continue
            tail = s[mm.end():]
            lv = LEV_RE.match(tail)
            if not lv: dr('no_level'); continue
            if APPROX.search(lv.group('gap')[-4:] + lv.group('gap')): dr('approx'); continue
            if re.match(r'\s*[、和及與]\s*\d', lv.group('after')): dr('list'); continue
            if STOP_RE.search(lv.group('gap')): dr('stop_in_gap'); continue
            if VER >= 'P2':
                if re.search(r'(站上|跳過|突破|大關|站穩|以上|逾|超過|衝破|守穩|之上|守住|跌破|低於|高於|不到|僅剩約)', lv.group('gap') + lv.group('after')): dr('threshold'); continue
                if lv.group('u') == '億' and len(lv.group('v').split('.')[-1] if '.' in lv.group('v') else '') < 2: dr('precision'); continue
                if lv.group('u') == '萬' and '.' in lv.group('v'): dr('precision'); continue
            if VER >= 'P3':
                gp = lv.group('gap')
                if re.search(r'[，,]', gp) and not re.fullmatch(r'[^，,]*[，,]\s*(合併)?營收(則)?(為|達)?\s*', gp): dr('comma_gap'); continue
            vstr = lv.group('v').replace(',', ''); v = float(vstr)
            rev_k = v * (1e5 if lv.group('u') == '億' else 10.0)
            if VER >= 'P3' and rev_k < 1e5: dr('small'); continue
            ndec = len(vstr.split('.')[1]) if '.' in vstr else 0
            rest = tail[lv.end():]
            st = STOP_RE.search(rest); nxt = MON_RE.search(rest)
            cut = min([x.start() for x in (st, nxt) if x] + [len(rest)])
            y = YOY_RE.search(rest[:cut]); yoy = np.nan
            if y:
                sg, val = (y.group('s1'), y.group('y1')) if y.group('y1') else (y.group('s2'), y.group('y2'))
                yoy = float(val) * (-1 if sg in ('減', '衰退', '減少', '下滑', '下降', '縮減') else 1)
                if VER >= 'P4' and '.' not in val: yoy = np.nan
            rows.append(dict(src=r['src'], art=r['art'], date=r['date'], code=code, how=how, ym=str(ym), rev_k=rev_k, ndec=ndec, unit=lv.group('u'),
                             yoy=yoy, basis=basis, seg=s[:300]))
df = pd.DataFrame(rows)
(HERE / 'data').mkdir(exist_ok=True)
df.to_parquet(HERE / f'data/news_vals_{VER}.parquet')
meta = dict(ver=VER, n_segments=nseg, n_vals=len(df), n_code_ym=int(df[['code', 'ym']].drop_duplicates().shape[0]), drops=drop,
            by_src=df.src.value_counts().to_dict(), frac_yoy=float(df.yoy.notna().mean()), basis=df.basis.value_counts().to_dict(),
            raw_sha=hashlib.sha256(RAWF.read_bytes()).hexdigest(),
            out_sha=hashlib.sha256((HERE / f'data/news_vals_{VER}.parquet').read_bytes()).hexdigest(), n_alias=len(ALIAS))
(HERE / f'data/news_vals_{VER}_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False))

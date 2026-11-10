#!/usr/bin/env python3
"""S3a：本地公開語料交叉搜尋（鉅亨 CTBC 鏡像 2013～2021、PTT Stock 推文逐日檔 2019～2021、ELD Yahoo/PTT 2016～2018 含時間、bofrankbo 逐檔新聞）。
在 PREREG 之後執行。所有語料視為不可信文字，只做字串比對；不修改來源檔（只記錄路徑＋sha）。
查詢詞＝處置相關關鍵字（不含任何目標數字或特定日期）。
python3 -I -B s5_local_search.py → data/local_mentions.csv、search_log_local.csv、notes/local_source_manifest.json"""
import re, json, glob, csv, hashlib, time, sys
WB, WA, SUF = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]) if len(sys.argv) > 3 else (80, 80, "")
from pathlib import Path
import numpy as np, pandas as pd
csv.field_size_limit(10 ** 9)
HERE = Path(__file__).resolve().parent; AG = HERE.parent
(HERE / 'notes').mkdir(exist_ok=True)
CT = AG / 'o_v4_revenue_patch_official/raw/ctbc/AI_Investment/OpenIEStanza_EventEmbedding/input/EB_SegAndRawNews'
PT = AG / 'o_rev_gap_news_2013/raw/ji08kk44_BDSE26_TEAM3/ptt/ptt_csv(date)'
EL = AG / 'o_rev_gap_news_2013/raw/roccqqck_stock_eland/5pa_article.csv'
BO = AG / 'o_rev_gap_news_2013/raw/bofrankbo_LLM-Predict-Stock/history_data/tw/stock_news'
TS = AG / 'o_disposal_rejudge/raw/tsungmin2001_tw-disposition'
KW = re.compile(r'處置股|列處置|列為處置|打入處置|進入處置|遭處置|被處置|處置期|處置措施|處置交易|處置名單|處置條件|再度處置|再次處置|第[一二]次處置|分盤交易|分盤撮合|人工管制|關禁閉|關緊閉|5分鐘撮合|五分鐘撮合|20分鐘撮合|二十分鐘撮合|預收款券')
# 名稱字典（上市＋上櫃；現行 stock_list、tsungmin 公司基本資料、rejudge 歷史名稱）
NM = {}
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str})
for c, n, mk in zip(sl.code, sl.name.fillna("").astype(str), sl.market): NM.setdefault(n.strip(), (c, mk))
for f, mk in [(TS / 'twse_companies.json', 'TWSE'), (TS / 'tpex_companies.json', 'TPEX')]:
    for r in json.loads(f.read_text(encoding='utf-8')):
        n = str(r.get('公司簡稱', '')).strip(); c = str(r.get('公司代號', '')).strip()
        if n and c: NM.setdefault(n, (c, mk))
rj = pd.read_csv(HERE / 'raw/lineage/disposal_hist_rejudge.csv', dtype={'code': str})
for c, n in zip(rj.code, rj.name.fillna('')):
    n = re.sub(r'\s+', '', n)
    if len(c) == 4 and n: NM.setdefault(n, (c, 'TWSE'))
names = sorted([n for n in NM if len(n) >= 2 and not re.fullmatch(r'[\dA-Za-z\-]+', n)], key=len, reverse=True)
NRE = re.compile('|'.join(re.escape(n) for n in names))
CRE = re.compile(r'(?<![\d.])([1-9]\d{3})(?![\d.%])')
CODES = set(sl.code) | set(rj.code[rj.code.str.len() == 4])
def extract(text):
    out = []
    for m in KW.finditer(text):
        a, b = max(0, m.start() - WB), min(len(text), m.end() + WA); w = text[a:b]
        found = {}
        for cm in CRE.finditer(w):
            c = cm.group(1)
            if c in CODES: found.setdefault(c, ('code', w))
        for nm in NRE.finditer(w):
            c, mk = NM[nm.group(0)]
            found.setdefault(c, ('name:' + nm.group(0), w))
        for c, (how, w_) in found.items(): out.append((c, how, m.group(0), w_.replace('\n', ' ')))
    return out
rows = []; log = []; man = {}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# 鉅亨
for f in sorted(CT.glob('*_cnyesnews.csv')):
    ym = re.match(r'(\d{4})(\d{1,2})_', f.name); y, m = int(ym.group(1)), int(ym.group(2))
    if not (2013 <= y <= 2021): continue
    man[str(f)] = sha(f)
    d = pd.read_csv(f, engine='python', usecols=['date', 'all'])
    nhit = 0
    for i, (dt, txt) in enumerate(zip(d["date"], d["all"].fillna("").astype(str))):
        if not KW.search(txt): continue
        nhit += 1
        for c, how, kw, w in extract(txt): rows.append(dict(src='CNYES', date=dt, art=f'{f.name}#{i}', code=c, how=how, kw=kw, snip=w[:200]))
    log.append(dict(src='CNYES', file=f.name, n_articles=len(d), n_kw_articles=nhit, query='KW 關鍵字表（處置相關，無數字）'))
# PTT 推文逐日檔（2019～2021）
for f in sorted(PT.glob('20*.csv')):
    dt = f.stem
    if not ('2019-01-01' <= dt <= '2021-12-31'): continue
    man[str(f)] = sha(f)
    txt = f.read_text(encoding='utf-8', errors='replace')
    lines = txt.split('\n'); nhit = 0
    for i, ln in enumerate(lines):
        if not KW.search(ln): continue
        nhit += 1
        for c, how, kw, w in extract(ln): rows.append(dict(src='PTT', date=dt, art=f'{f.name}#{i}', code=c, how=how, kw=kw, snip=w[:200]))
    log.append(dict(src='PTT', file=f.name, n_articles=len(lines), n_kw_articles=nhit, query='KW'))
# ELD（Yahoo 股市新聞＋PTT，2016～2018，post_time 含時間）
man[str(EL)] = sha(EL)
e = pd.read_csv(EL)
nhit = 0
for i, r in e.iterrows():
    txt = f"{r['title'] if isinstance(r['title'], str) else ''} {r['content'] if isinstance(r['content'], str) else ''}"
    if not KW.search(str(txt)): continue
    nhit += 1
    for c, how, kw, w in extract(str(txt)): rows.append(dict(src='ELD', date=str(r['post_time']), art=f"ELD#{r['id']}", code=c, how=how, kw=kw, snip=w[:200]))
log.append(dict(src='ELD', file=EL.name, n_articles=len(e), n_kw_articles=nhit, query='KW'))
# bofrankbo 逐檔新聞（檔名＝日期）
nb = 0; nbh = 0
for f in sorted(BO.glob('*/*_news/*')):
    if not f.is_file(): continue
    dt = f.name
    if not re.fullmatch(r'\d{8}', dt): continue
    nb += 1
    txt = f.read_text(encoding='utf-8', errors='replace')
    if not KW.search(txt): continue
    nbh += 1; man[str(f)] = sha(f)
    for c, how, kw, w in extract(txt): rows.append(dict(src='BOF', date=f'{dt[:4]}-{dt[4:6]}-{dt[6:]}', art=f'BOF#{f.parent.parent.name}/{dt}', code=c, how=how, kw=kw, snip=w[:200]))
log.append(dict(src='BOF', file=str(BO), n_articles=nb, n_kw_articles=nbh, query='KW'))
M = pd.DataFrame(rows)
M['date'] = pd.to_datetime(M['date'].astype(str).str[:10], errors='coerce')
M = M.dropna(subset=['date']).drop_duplicates(['src', 'art', 'code'])
M.to_csv(HERE / f'data/local_mentions{SUF}.csv', index=False)
pd.DataFrame(log).to_csv(HERE / f'search_log_local{SUF}.csv', index=False)
(HERE / f'notes/local_source_manifest{SUF}.json').write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), keyword_regex=KW.pattern, n_files=len(man), files=man), ensure_ascii=False, indent=0), encoding='utf-8')
print(len(M), M.src.value_counts().to_dict()); print(M.groupby([M.src, M.date.dt.year]).size().unstack(0).fillna(0).astype(int))

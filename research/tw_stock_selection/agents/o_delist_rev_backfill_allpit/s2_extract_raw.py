#!/usr/bin/env python3
"""步驟 2(b) 原始子集抽取（只讀來源文字，不執行來源任何程式）。來源都是已在本機的 GitHub 公開 clone（唯讀）：
  CTBC：aRthur08701244/Intern-in-CTBC@258f42aa（鉅亨網）EB_EventExtraction2（News id＋All）與 EB_SegAndRawNews（date＋all）
  BOF ：bofrankbo/LLM-Predict-Stock history_data/tw/stock_news/2311/2311_news（日月光逐檔新聞，檔名＝日期）
  ELD ：roccqqck/stock_eland 5pa_article.csv（Yahoo 股市＋PTT 2016～2018，post_time）
規則（寫死，v2）：文章依「。」「；」「\n」切段；保留同時含「營收」「月」與「<數字>(億|萬)元」的段（解析器 P2 起只用這種段，不用前一段）。
（v1 另保留含營業收入／業績的段與前一段，26MB，只放 scratchpad；P1～P4 用 v1，P5 在 v1 與 v2 上輸出逐列相同。）
同 (日期, 段文字) 去重。輸出 raw/news_rev_segments.csv.gz（src,art,date,seg_i,seg）與 raw/news_source_manifest.json（commit、檔案 sha256）。
python3 -I -B s2_extract_raw.py"""
import re, json, hashlib, glob, csv, gzip, subprocess
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
CT = AG / 'o_v4_revenue_patch_official/raw/ctbc'
CTI = CT / 'AI_Investment/OpenIEStanza_EventEmbedding/input'
BOF = AG / 'o_rev_gap_news_2013/raw/bofrankbo_LLM-Predict-Stock'
ELD = AG / 'o_rev_gap_news_2013/raw/roccqqck_stock_eland'
KEY = re.compile(r'(營收|營業收入|業績)')
NUM = re.compile(r'\d')
SPL = re.compile(r'[。；\n]')
LVL = re.compile(r'\d\s*(億|萬)\s*元')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def head(p):
    try: return subprocess.run(['git', '-C', str(p), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception: return None
seen = set(); rows = []; man = {'CTBC': {'head': head(CT), 'files': {}}, 'BOF': {'head': head(BOF), 'files': {}}, 'ELD': {'head': head(ELD), 'files': {}}}
def emit(src, art, date, text):
    if not isinstance(text, str) or not KEY.search(text): return
    segs = [s.strip() for s in SPL.split(text)]
    for i, s in enumerate(segs):
        if s and KEY.search(s) and NUM.search(s) and '營收' in s and '月' in s and LVL.search(s):
            k = (date, s)
            if k in seen: continue
            seen.add(k); rows.append((src, art, date, i, s[:600]))
for f in sorted(glob.glob(str(CTI / 'EB_EventExtraction2/*.csv'))):
    man['CTBC']['files'][str(Path(f).relative_to(CT))] = sha(f)
    d = pd.read_csv(f, usecols=['News', 'All'], dtype=str)
    for a, t in zip(d['News'], d['All']):
        emit('CTBC', 'CTBC:' + str(a), str(a)[:8], t)
for f in sorted(glob.glob(str(CTI / 'EB_SegAndRawNews/*.csv'))):
    man['CTBC']['files'][str(Path(f).relative_to(CT))] = sha(f)
    d = pd.read_csv(f, usecols=['date', 'all'], dtype=str)
    for i, (dt, t) in enumerate(zip(d['date'], d['all'])):
        emit('CTBC', f'CTBCSR:{Path(f).stem}:{i}', str(dt).replace('-', '')[:8], t)
for f in sorted(glob.glob(str(BOF / 'history_data/tw/stock_news/2311/2311_news/*'))):
    man['BOF']['files'][str(Path(f).relative_to(BOF))] = sha(f)
    emit('BOF', 'BOF:' + Path(f).name, Path(f).name[:8], Path(f).read_text(encoding='utf-8', errors='replace'))
f = ELD / '5pa_article.csv'; man['ELD']['files']['5pa_article.csv'] = sha(f)
d = pd.read_csv(f, dtype=str)
for a, pt, ti, ct in zip(d['id'], d['post_time'], d['title'], d['content']):
    emit('ELD', 'ELD:' + str(a), re.sub(r'\D', '', str(pt))[:8], (str(ti) + '。' + re.sub(r'<BR>', '\n', str(ct))))
out = HERE / 'raw/news_rev_segments.csv.gz'
with gzip.open(out, 'wt', encoding='utf-8', newline='') as g:
    w = csv.writer(g); w.writerow(['src', 'art', 'date', 'seg_i', 'seg']); w.writerows(rows)
man['n_segments'] = len(rows); man['out_sha256'] = sha(out)
man['rule'] = __doc__
(HERE / 'raw/news_source_manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=0), encoding='utf-8')
print(len(rows), out.stat().st_size, man['out_sha256'])

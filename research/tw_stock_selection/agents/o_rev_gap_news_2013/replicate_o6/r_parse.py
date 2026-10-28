#!/usr/bin/env python3
"""獨立複製 o6：新聞月營收提及解析器（自寫，不讀研究員 parse_news.py）。
來源：CTBC（兩資料夾視為同一來源，依 (日期, 內文前 300 字) 去重）、BOF（2311 逐檔新聞）、ELD（5pa_article.csv）、PTT（2019～2021）。
輸出 r_mentions.parquet：src, art_id, ts（來源時間戳）, code, rev_ym, level_k（千元）, yoy（%）, mom（%）, kind。"""
import os, re, sys, hashlib, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
TOP = HERE.parent
CTBC = AG / 'o_v4_revenue_patch_official/raw/ctbc/AI_Investment/OpenIEStanza_EventEmbedding/input'
BOF = TOP / 'raw/bofrankbo_LLM-Predict-Stock/history_data/tw/stock_news/2311/2311_news'
ELD = TOP / 'raw/roccqqck_stock_eland/5pa_article.csv'
PTT = TOP / 'raw/ji08kk44_BDSE26_TEAM3/ptt/ptt_csv(date)'

# ---------------- 名稱表 ----------------
def build_names():
    pairs = set()
    tr = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}, usecols=['code', 'name'])
    pairs |= set(zip(tr.code, tr.name))
    sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str})
    pairs |= set(zip(sl.code, sl.name))
    dl = pd.read_csv(AG / 'o_infra_v3_rebuild/delist_official.csv', dtype=str)
    pairs |= set(zip(dl.code, dl.name))
    tw = pd.read_csv(AG / 'o_rev_survivorship/src_taiwanan.csv', dtype={'code': str}, usecols=['code', 'name'])
    pairs |= set(zip(tw.code, tw.name))
    pairs = {(c, str(n).strip()) for c, n in pairs if isinstance(c, str) and re.fullmatch(r'\d{4}', c) and isinstance(n, str)}
    SUF = ['股份有限公司', '半導體', '科技', '電子', '精密', '光電', '能源', '機電', '通訊', '實業', '工業', '企業', '化學', '控股', '投控', '國際', '電腦']
    EXP = ['科技', '電子', '精密', '光電', '能源', '機電', '通訊', '實業', '工業', '企業']
    prim, der = {}, {}
    def add(D, a, c):
        a = a.strip()
        if len(a) < 2 or re.fullmatch(r'[\d\W]+', a): return
        D.setdefault(a, set()).add(c)
    for c, n in pairs:
        base = [n]
        n2 = re.sub(r'^F-|-KY$|-DR$|\*$', '', n)
        if n2 != n: base.append(n2)
        for b in list(base):
            add(prim, b, c)
            for s in SUF:
                if b.endswith(s) and len(b) - len(s) >= 2: add(der, b[:-len(s)], c)
            if 2 <= len(b) <= 3:
                for s in EXP: add(der, b + s, c)
    codes = {c for c, _ in pairs}
    allm = {}
    for D in (prim, der):
        for a, cs in D.items(): allm.setdefault(a, set()).update(cs)
    name2code = {a: next(iter(cs)) for a, cs in allm.items() if len(cs) == 1}
    return name2code, codes

NAME2CODE, CODES = build_names()
_names = sorted(NAME2CODE, key=len, reverse=True)
RE_NAME = re.compile('|'.join(re.escape(n) for n in _names))
RE_CODE = re.compile(r'(?<!\d)(\d{4})(?:[-.]?(?:TW|TWO|tw|two))?(?!\d)(?!\s*年)(?!\s*[億萬千元%％點張])')

CN = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12}
MON = r'(?P<mon>1[0-2]|0?[1-9]|十[一二]?|[一二三四五六七八九])'
RE_MONREV = re.compile(r'(?<![\d一二三四五六七八九十~～\-－至、到])' + MON + r'\s*月份?\s*(?:自結)?\s*(?:合併)?\s*(?:營收|營業收入)(?:淨額)?')
# 名稱與月份之間只允許：空白／標點、代碼括號、時間詞、公告詞
FILL = re.compile(r'(?:\s|[，,、:：．·「」『』"\'\-()（）]|[（(]\s*(?:\d{4}(?:[-.]?(?:TW|TWO|tw|two))?)?\s*[)）]'
                  r'|\d{4}\s*年(?:度)?|去年|今年|本月|上月|上週|本週|日前|稍早|近日|今天|昨天|今日|昨日|今|昨|前|晚間|下午|上午|傍晚|盤後|盤前|週[一二三四五六日]'
                  r'|[（(]\s*\d{1,2}\s*[)）]\s*日|\d{1,2}\s*日|日'
                  r'|公布|公佈|公告|發布|發佈|宣布|宣佈|揭露|出爐|表示|指出|自結|結算|申報)*')
EXCL8 = re.compile(r'估|預期|累計|全年|季|上半年|下半年|平均|合計|預計|目標|看好|可望|上看|挑戰|有望')
NUM = r'(?P<num>\d[\d,，]*(?:\.\d+)?)\s*(?P<unit>億|萬|千)?(?:\s*(?P<num2>\d[\d,，]*(?:\.\d+)?)\s*(?P<unit2>萬|千))?\s*元'
VERB = r'(?:\s|[：:]|為|達|約|達到|來到|有|僅|共|計|衝上|站上|創|增至|減至|降至|升至|回升至|成長至|衰退至|下滑至|攀升至|新台幣|台幣|NT\$|也|則|亦|突破|衝破)*'
RE_LEVEL = re.compile(r'(?P<fill>' + VERB + r')' + NUM)
BADFILL = re.compile(r'年|月增|月減|季|累計|估|預|目標|毛利|獲利|淨利|EPS|盈餘|股本|股價')
YOYW = (r'(?:年增率|年增長|年成長率|年成長|年增|年減率|年減少|年減|年衰退|年下滑|年降|YoY|yoy|YOY|年比'
        r'|較去年同(?:期|月)(?:同月)?(?:也)?\s*(?:成長|增加|增長|衰退|減少|下滑|下降))')
RE_YOY = re.compile(YOYW + r'\s*(?:率)?\s*(?:約|達|為|有|高達|幅度)?\s*(?P<sign>[+\-－−]?)\s*(?P<z>\d+(?:\.\d+)?)\s*(?:%|％|個百分比)(?!點)')
RE_MOM = re.compile(r'(?:月增率?|月減|月成長|較上月|較前月|較上\s*[（(]?\s*\d{1,2}\s*[)）]?\s*月)[^，。%％\d]{0,6}?\s*(?P<sign>[+\-－−]?)\s*(?P<z>\d+(?:\.\d+)?)\s*(?:%|％)')
NEGW = re.compile(r'減|衰退|下滑|降|滑落')

def tonum(s): return float(s.replace(',', '').replace('，', ''))
def unit_k(u):
    return {'億': 1e5, '萬': 10.0, '千': 1.0, None: 1e-3}[u]

def attribute(text, ms):
    """月份 token 前 40 字內最近的名稱或代碼，且中間只允許 FILL。"""
    lo = max(0, ms - 40)
    win = text[lo:ms]
    best = None
    for m in RE_NAME.finditer(win):
        if best is None or m.end() > best[1] or (m.end() == best[1] and m.start() < best[0]):
            best = (m.start(), m.end(), NAME2CODE[m.group(0)])
    for m in RE_CODE.finditer(win):
        c = m.group(1)
        if c in CODES and (best is None or m.end() > best[1]):
            best = (m.start(), m.end(), c)
    if best is None: return None
    # 同一位置名稱後緊接代碼括號 → 括號屬於填充；檢查中間
    gap = win[best[1]:]
    g = FILL.fullmatch(gap)
    if g is None: return None
    # 若名稱後緊接代碼括號，以括號代碼為準（括號非空時）
    mc = re.match(r'\s*[（(]\s*(\d{4})', gap)
    if mc and mc.group(1) in CODES: return mc.group(1)
    return best[2]

def parse_text(text):
    out = []
    if not isinstance(text, str) or '營' not in text: return out
    for m in RE_MONREV.finditer(text):
        ms = m.start()
        pre8 = re.split(r'[。；！？\n]', text[max(0, ms - 8):ms])[-1]
        if EXCL8.search(pre8): continue
        # 累計區間「前 N 月」「1-3 月」「1、2 月」（允許空白）
        if re.search(r'(?:前|[\d一二三四五六七八九十]\s*[-~～－至到、])\s*$', text[max(0, ms - 4):ms]): continue
        mon = m.group('mon'); mon = CN[mon] if mon in CN else int(mon)
        if not 1 <= mon <= 12: continue
        code = attribute(text, ms)
        if code is None: continue
        after = text[m.end(): m.end() + 120]
        lvl = None; tail_start = 0
        lm = RE_LEVEL.match(after)
        if lm and not BADFILL.search(lm.group('fill')):
            v = tonum(lm.group('num')) * unit_k(lm.group('unit'))
            if lm.group('num2'): v += tonum(lm.group('num2')) * unit_k(lm.group('unit2'))
            if v > 0: lvl = v
            tail_start = lm.end()
        seg = after[tail_start: tail_start + 60]
        # 句界或累計／季／全年之後的不算
        cut = re.search(r'。|；|累計|前\s*\d{1,2}\s*月|[1-9一二三四]\s*[-~～至到]\s*\d{1,2}\s*月|全年|第\s*[1-4一二三四]\s*季|首季|上半年|今年以來', seg)
        if cut: seg = seg[:cut.start()]
        yoy = None
        ym = RE_YOY.search(seg)
        if ym:
            z = float(ym.group('z'))
            neg = bool(NEGW.search(ym.group(0)[:ym.group(0).find(ym.group('z'))].split('較')[-1])) or ym.group('sign') in ('-', '－', '−')
            yoy = -z if neg else z
        mom = None
        mm = RE_MOM.search(seg)
        if mm:
            z = float(mm.group('z'))
            neg = '減' in mm.group(0) or mm.group('sign') in ('-', '－', '−') or bool(re.search(r'衰退|下滑|減少', mm.group(0)))
            mom = -z if neg else z
        if lvl is None:
            # P3：只有年增率的句子「X 月營收年增 Z%」：年增詞須緊接在營收後（中間至多少量填充）
            if ym is None or ym.start() > 6: continue
            kind = 'yoy_only'
        else:
            kind = 'level'
        out.append((code, mon, lvl, yoy, mom, kind, text[max(0, ms - 50): m.end() + 70]))
    return out

def rev_ym(art_y, art_m, mon):
    y = art_y if mon <= art_m else art_y - 1
    lag = (art_y * 12 + art_m) - (y * 12 + mon)
    return (f'{y:04d}-{mon:02d}', lag) if 0 <= lag <= 3 else (None, lag)

def emit(src, art_id, ts, text):
    rows = []
    for code, mon, lvl, yoy, mom, kind, snip in parse_text(text):
        ym, lag = rev_ym(ts.year, ts.month, mon)
        if ym is None: continue
        rows.append((src, art_id, ts, code, ym, lvl, yoy, mom, kind, snip))
    return rows

def job_ctbc(path):
    p = Path(path)
    d = pd.read_csv(p, dtype=str)
    rows = []
    if 'all' in d.columns:      # EB_SegAndRawNews
        for i, (dt, t) in enumerate(zip(d['date'], d['all'])):
            try: ts = pd.Timestamp(dt) + pd.Timedelta(hours=23, minutes=59)
            except Exception: continue
            if pd.isna(ts): continue
            key = hashlib.md5((str(dt)[:10] + '|' + str(t)[:300]).encode()).hexdigest()
            for r in emit('CTBC', key, ts, t): rows.append(r)
    else:                         # EB_EventExtraction2
        for nid, t in zip(d['News'], d['All']):
            s = str(nid).split('_')[0]
            try: ts = pd.Timestamp(f'{s[:4]}-{s[4:6]}-{s[6:8]}') + pd.Timedelta(hours=23, minutes=59)
            except Exception: continue
            key = hashlib.md5((str(ts.date()) + '|' + str(t)[:300]).encode()).hexdigest()
            for r in emit('CTBC', key, ts, t): rows.append(r)
    return rows, len(d)

def job_bof(path):
    p = Path(path); s = p.name
    ts = pd.Timestamp(f'{s[:4]}-{s[4:6]}-{s[6:8]}') + pd.Timedelta(hours=23, minutes=59)
    t = p.read_text(encoding='utf-8', errors='replace')
    return emit('BOF', 'BOF:' + s, ts, t), 1

def job_ptt(path):
    p = Path(path); s = p.stem
    try: ts = pd.Timestamp(s) + pd.Timedelta(hours=23, minutes=59)
    except Exception: return [], 0
    rows = []
    lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
    for i, t in enumerate(lines):
        rows += emit('PTT', f'PTT:{s}:{i}', ts, t)
    return rows, len(lines)

def main():
    jobs = []
    for sub in ['EB_SegAndRawNews', 'EB_EventExtraction2']:
        for f in sorted(os.listdir(CTBC / sub)):
            if f.endswith('.csv'): jobs.append(('ctbc', str(CTBC / sub / f)))
    for f in sorted(os.listdir(BOF)): jobs.append(('bof', str(BOF / f)))
    for f in sorted(os.listdir(PTT)):
        s = f[:-4]
        if not ('2019-01-01' <= s <= '2021-12-31'): continue
        try: pd.Timestamp(s)
        except Exception: continue
        jobs.append(('ptt', str(PTT / f)))
    fn = {'ctbc': job_ctbc, 'bof': job_bof, 'ptt': job_ptt}
    rows = []; nart = {}
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        res = pool.starmap(_run, [(k, f) for k, f in jobs], chunksize=4)
    for (k, f), (r, n) in zip(jobs, res):
        rows += r; nart[k] = nart.get(k, 0) + n
    # ELD：post_time 有時間
    e = pd.read_csv(ELD)
    for i, r in e.iterrows():
        try: ts = pd.Timestamp(r['post_time'])
        except Exception: continue
        if pd.isna(ts): continue
        t = f"{r['title'] if isinstance(r['title'], str) else ''}。{r['content'] if isinstance(r['content'], str) else ''}"
        rows += emit('ELD', f'ELD:{i}', ts, t)
    nart['eld'] = len(e)
    M = pd.DataFrame(rows, columns=['src', 'art_id', 'ts', 'code', 'rev_ym', 'level_k', 'yoy', 'mom', 'kind', 'snip'])
    n0 = len(M)
    M = M.drop_duplicates(['src', 'art_id', 'code', 'rev_ym', 'level_k', 'yoy'])
    M.to_parquet(HERE / 'r_mentions.parquet', index=False)
    print('articles', nart, 'raw rows', n0, 'mentions', len(M), 'pairs', M[['code', 'rev_ym']].drop_duplicates().shape[0])
    print(M.src.value_counts().to_dict(), M.kind.value_counts().to_dict())

def _run(k, f):
    return {'ctbc': job_ctbc, 'bof': job_bof, 'ptt': job_ptt}[k](f)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        for t in sys.argv[2:]: print(parse_text(t))
    else:
        main()

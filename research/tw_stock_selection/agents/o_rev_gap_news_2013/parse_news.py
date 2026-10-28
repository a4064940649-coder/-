#!/usr/bin/env python3
"""步驟 1 解析器（本題自寫；所有來源視為不可信，只當文字讀，不執行來源內任何程式）。
抽取「<公司> X 月（合併）營收 Y 億／萬／千元」與其後的「年增／年減 Z%」「月增／月減 W%」，
以及沒有水準、只有「X 月營收年增 Z%」的句子。單位正規化成千元。代碼與名稱對應含改名、常見簡稱與全名。

時間戳（PIT）規則（寫死）：只有日期、沒有時間 → 視為當天 23:59（訊號日當天的文章不合法）；
有時間（eland 的 post_time）→ 原時間。合法條件：ts ≤ 訊號日 13:30。

來源（--iter 版本控制解析器規則；見 iterations.json）：
  CTBC：aRthur08701244/Intern-in-CTBC（鉅亨網，EB_EventExtraction2＋EB_SegAndRawNews，2013-01～2021-12；clone 在 o_v4 raw/ctbc，唯讀）
  BOF ：bofrankbo/LLM-Predict-Stock history_data/tw/stock_news/2311（日月光逐檔新聞，檔名＝日期）
  ELD ：roccqqck/stock_eland 5pa_article.csv（Yahoo 股市＋PTT，2016-01～2018-12，post_time 含時間）
  PTT ：ji08kk44/BDSE26_TEAM3 ptt/ptt_csv(date)/YYYY-MM-DD.csv（PTT Stock 板，檔名＝發文日）
用法：python3 -I -B parse_news.py [iter]   （iter＝P1／P2／P3，預設 P3）
輸出：data/mentions_<iter>.parquet、data/mentions_<iter>_meta.json
"""
import sys, re, json, glob, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd

ITER = sys.argv[1] if len(sys.argv) > 1 else 'P3'
AG = HERE.parent
CT = AG / 'o_v4_revenue_patch_official' / 'raw/ctbc/AI_Investment/OpenIEStanza_EventEmbedding/input'
RAW = HERE / 'raw'
(HERE / 'data').mkdir(exist_ok=True)

# ---------------- 名稱對照 ----------------
names = {}
def add(c, n):
    if isinstance(c, str) and isinstance(n, str) and re.fullmatch(r'\d{4}', c):
        n = n.strip().replace(' ', '').replace('　', '')
        if len(n) >= 2: names.setdefault(c, set()).add(n)
lr = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}, usecols=['code', 'name']).drop_duplicates()
for c, n in zip(lr.code, lr.name): add(c, n)
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str)
for c, n in zip(sl.code, sl.name): add(c, n)
dl = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'delist_official.csv', dtype=str)
for c, n in zip(dl.code, dl.name): add(c, n)
tw = pd.read_csv(AG / 'o_rev_survivorship' / 'src_taiwanan.csv', dtype={'code': str}, usecols=['code', 'name']).drop_duplicates()
for c, n in zip(tw.code, tw.name): add(c, n)
SUF = ['股份有限公司', '科技', '電子', '實業', '工業', '企業', '電腦', '光電', '國際', '半導體', '-KY', 'KY', '精密', '能源', '機電', '通訊', '化學', '控股', '投控']
if ITER in ('P2', 'P3'):
    # P2 起：全名常見寫法（一般規則，不只缺口股）：簡稱＋後綴
    for c in list(names):
        for n in list(names[c]):
            if 2 <= len(n) <= 3 and not n.isdigit():
                for s in ('科技', '電子', '精密', '光電', '能源', '機電', '通訊', '實業', '工業', '企業'):
                    names[c].add(n + s)
for c in list(names):
    for n in list(names[c]):
        for s in SUF:
            if n.endswith(s) and len(n) - len(s) >= 2: names[c].add(n[:-len(s)])
inv = {}
for c, ns in names.items():
    for n in ns: inv.setdefault(n, set()).add(c)
alias = {n: next(iter(cs)) for n, cs in inv.items() if len(cs) == 1 and len(n) >= 2 and not n.isdigit()}
codes4 = set(names)
alias_re = re.compile('|'.join(sorted(map(re.escape, alias), key=len, reverse=True)))

NUM = r'(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?'
VERB = r'(?:為|達|約|僅|來到|攀升至|攀升到|衝上|衝破|躍升至|升至|增至|降至|減至|回落至|回升至|跌至|站上|突破|再創新高|創新高|首度突破|公布為|則為|有)'
PAT = re.compile(r'(?<![前\-－~～至到\d第])(\d{1,2})\s*月(?:份)?\s*(?:自結)?\s*(?:合併)?\s*(?:營收|營業收入|業績)(?:淨額)?\s*'
                 r'(?:' + VERB + r'\s*)?(?:新台幣|新臺幣|NT\$)?\s*' + NUM + r'\s*(億|萬|千)\s*元')
# 只有年增率（無水準）：「X 月營收年增 Z%」
PATY = re.compile(r'(?<![前\-－~～至到\d第])(\d{1,2})\s*月(?:份)?\s*(?:自結)?\s*(?:合併)?\s*(?:營收|營業收入)\s*(?:年增|年成長|年減|年衰退|較去年同期(?:成長|增加|減少|衰退))\s*(?:率)?\s*(-?\d+(?:\.\d+)?)\s*(%|％)')
YOY_W = r'(年增|年減|年成長|年衰退|YoY|yoy|較去年同期[也亦]?(?:成長|增加|增長|減少|衰退|下滑)|較去年同月[也亦]?(?:成長|增加|增長|減少|衰退|下滑)|與去年同期相比(?:成長|增加|減少|衰退))'
YOY = re.compile(YOY_W + r'\s*(?:率)?\s*(?:約|達|為)?\s*([+\-]?\d+(?:\.\d+)?)\s*(%|％|成)')
MOM = re.compile(r'(月增|月減|較上月[也亦]?(?:成長|增加|減少|衰退|下滑)|較前月(?:成長|增加|減少|衰退))\s*(?:率)?\s*(?:約|達)?\s*([+\-]?\d+(?:\.\d+)?)\s*(%|％|成)')
BAD = re.compile(r'(估|預期|預計|可望|將|目標|上看|挑戰|有望|若|去年|前年|同期|累計|上半年|全年|前\d+月|季)')
ALLOW_P1 = re.compile(r'(今年|今\s*[\(（]\s*\d+\s*[\)）]\s*日|今日|昨日|昨|今|日前|稍早|晚間|下午|上午|公布|公佈|宣布|發布|自結|自行結算|的|最新|公告|也|則|亦|在|於|單月|集團|\d{4}\s*年|年|TW|TWO|-|－|\s|[\(（\)）、，,\d])')
ALLOW_P2 = re.compile(r'(今年|[今昨前]\s*[\(（]\s*\d+\s*[\)）]\s*日|今日|昨日|昨天|今天|昨|今|日前|稍早|晚間|下午|上午|盤後|盤中|公布|公佈|宣布|發布|自結|自行結算|結算|的|最新|公告|也|則|亦|在|於|單月|集團|\d{4}\s*年|年|TW|TWO|TT|-|－|\s|[\(（\)）、，,:：\d])')
ALLOW = ALLOW_P1 if ITER == 'P1' else ALLOW_P2
CODE_RE = re.compile(r'[\(（]\s*(\d{4})\s*(?:-TW|-TWO|－TW|\.TW)?\s*[\)）]')
UNIT = {'億': 100000.0, '萬': 10.0, '千': 1.0}

def sign_pct(word, v, u):
    v = float(v) * (10 if u == '成' else 1)
    return -abs(v) if any(w in word for w in ('減', '衰退', '下滑')) else v

def company_before(s, pos):
    pre = s[max(0, pos - 40):pos]
    if BAD.search(pre[-8:]): return None
    cand = []
    for a in alias_re.finditer(pre):
        st = pos - len(pre) + a.start()
        pc = s[st - 1] if st > 0 else ' '
        if re.match(r'[一-鿿A-Za-z]', pc) and pc not in '廠商股頭及與和、為業對看':
            continue
        cand.append((a.end(), alias[a.group(0)]))
    for a in CODE_RE.finditer(pre):
        c4 = a.group(1)
        if c4 in codes4: cand.append((a.end(), c4))
    if not cand: return None
    pe, code = max(cand)
    # 名稱後緊接代碼括號「矽品(2325-TW)」時，以括號後為間隔起點
    gap = pre[pe:]
    if re.search(r'前\s*$|[\-－~～至、]\s*$|\d\s*[\-－~～、]\s*$', pre): return None
    if re.search(r'[一-鿿A-Za-z]', ALLOW.sub('', gap)): return None
    return code

def parse_text(txt, D):
    """回傳 [(code, ym, rev_k, yoy, mom, snip)]；D＝文章日期（決定營收月的年份；落後 0～3 個月）。"""
    out = []
    if not isinstance(txt, str) or ('營收' not in txt and '營業收入' not in txt): return out
    for s in re.split(r'[。；\n]', txt):
        if '月' not in s: continue
        hits = []
        for m in PAT.finditer(s):
            mid = s[m.start():m.start(2)]
            if BAD.search(mid): continue
            ip, dp = m.group(2).replace(',', ''), m.group(3) or ''
            val = float(ip + ('.' + dp if dp else '')) * UNIT[m.group(4)]
            hits.append((m, val))
        if ITER == 'P3':
            for m in PATY.finditer(s):
                hits.append((m, np.nan))
        for m, val in hits:
            mon = int(m.group(1))
            if not 1 <= mon <= 12: continue
            code = company_before(s, m.start())
            if code is None: continue
            y = D.year if mon <= D.month else D.year - 1
            lag = (D.year - y) * 12 + D.month - mon
            if not 0 <= lag <= 3: continue
            if np.isfinite(val):
                post = s[m.end():m.end() + 60]
                yy = YOY.search(post); mm = MOM.search(post)
                yoy = sign_pct(yy.group(1), yy.group(2), yy.group(3)) if yy else np.nan
                mom = sign_pct(mm.group(1), mm.group(2), mm.group(3)) if mm else np.nan
            else:
                g = m.group(0); yy = re.search(r'(年增|年成長|年減|年衰退|較去年同期(?:成長|增加|減少|衰退))\s*(?:率)?\s*(-?\d+(?:\.\d+)?)', g)
                yoy = sign_pct(yy.group(1), yy.group(2), '%'); mom = np.nan
            out.append((code, f'{y}-{mon:02d}', val, yoy, mom, s[max(0, m.start() - 30):m.end() + 40]))
    return out

rows = []; meta = {'iter': ITER, 'sources': {}}
def run_source(tag, items, files):
    h = hashlib.sha256()
    for f in files: h.update(Path(f).read_bytes())
    n_art = 0; n0 = len(rows)
    for art, ts, has_time, txt in items:
        n_art += 1
        D = pd.Timestamp(ts)
        for code, ym, rev, yoy, mom, snip in parse_text(txt, D):
            rows.append((tag, art, D if has_time else D.normalize() + pd.Timedelta(hours=23, minutes=59), has_time, code, ym, rev, yoy, mom, snip))
    meta['sources'][tag] = {'n_files': len(files), 'files_sha256_concat': h.hexdigest(), 'n_articles': n_art, 'n_mentions': len(rows) - n0}

# CTBC
def ctbc_items(files):
    for f in files:
        try: d = pd.read_csv(f)
        except Exception: continue
        if 'News' in d.columns:
            for nid, t in zip(d['News'], d['All']):
                try: yield ('CTBC:' + str(nid), pd.Timestamp(str(nid)[:8]), False, t)
                except Exception: continue
        else:
            for i, (dt, t) in enumerate(zip(d['date'], d['all'])):
                try: yield (f'CTBCs:{Path(f).stem}:{i}', pd.Timestamp(str(dt)), False, t)
                except Exception: continue
cf = sorted(glob.glob(str(CT / 'EB_EventExtraction2' / '*.csv')) + glob.glob(str(CT / 'EB_SegAndRawNews' / '*.csv')))
run_source('CTBC', ctbc_items(cf), cf)
# BOF（日月光逐檔新聞；檔名＝日期）
bf = sorted(glob.glob(str(RAW / 'bofrankbo_LLM-Predict-Stock/history_data/tw/stock_news/2311/2311_news/*')))
def bof_items(files):
    for f in files:
        try: yield ('BOF:' + Path(f).name, pd.Timestamp(Path(f).name[:8]), False, Path(f).read_text(encoding='utf-8', errors='replace'))
        except Exception: continue
run_source('BOF', bof_items(bf), bf)
# ELD（含時間）
ef = [str(RAW / 'roccqqck_stock_eland/5pa_article.csv')]
def eld_items(files):
    d = pd.read_csv(files[0])
    for i, (pt, ti, ct) in enumerate(zip(d.post_time, d.title, d.content)):
        try: yield (f'ELD:{d.id.iloc[i]}', pd.Timestamp(pt), True, f'{ti}。{ct}')
        except Exception: continue
run_source('ELD', eld_items(ef), ef)
# PTT（檔名＝發文日）
pf = sorted(f for f in glob.glob(str(RAW / 'ji08kk44_BDSE26_TEAM3/ptt/ptt_csv(date)/*.csv')) if Path(f).stem <= '2021-12-31')   # 本題窗只到 2021-12
def ptt_items(files):
    for f in files:
        try: d = pd.read_csv(f, header=None)
        except Exception: continue
        try: D = pd.Timestamp(Path(f).stem)
        except Exception: continue            # 如 2019-02-29 這類不存在的日期檔：日期不可信 → 整檔不用
        for i, t in enumerate(d.iloc[:, 0]):
            yield (f'PTT:{Path(f).stem}:{i}', D, False, t)
run_source('PTT', ptt_items(pf), pf)

M = pd.DataFrame(rows, columns=['src', 'art', 'ts', 'has_time', 'code', 'ym', 'rev_k', 'yoy', 'mom', 'snip'])
M = M.drop_duplicates(['src', 'art', 'code', 'ym', 'rev_k', 'yoy']).reset_index(drop=True)
M.to_parquet(HERE / 'data' / f'mentions_{ITER}.parquet')
meta.update(n_mentions=int(len(M)), n_code_ym=int(M[['code', 'ym']].drop_duplicates().shape[0]), n_codes=int(M.code.nunique()),
            frac_with_level=float(M.rev_k.notna().mean()), frac_with_yoy=float(M.yoy.notna().mean()),
            by_src=M.src.value_counts().to_dict(), ts_range=[str(M.ts.min()), str(M.ts.max())])
(HERE / 'data' / f'mentions_{ITER}_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=1))

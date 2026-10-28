#!/usr/bin/env python3
"""新來源 N：鉅亨網新聞鏡像（GitHub aRthur08701244/Intern-in-CTBC，EB_EventExtraction2，2013-01～2021-09，視為不可信、只當 CSV 讀）
抽出「公司 X 月（合併）營收 Y 億元／萬元」與同句的「年增／年減 Z%」。不看任何報酬。
輸出 raw_news_mentions.parquet：每筆提及一列（文章日期、代碼、營收月、千元值、有效位數、年增率、原文片段）。
用法：python3 -I -B extract_news.py <新聞 csv 目錄>
"""
import sys, re, json, glob, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd

SRCS = [Path(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [HERE / 'raw/ctbc/AI_Investment/OpenIEStanza_EventEmbedding/input/EB_EventExtraction2', HERE / 'raw/ctbc/AI_Investment/OpenIEStanza_EventEmbedding/input/EB_SegAndRawNews']

# ---- 名稱對照（代碼 → 名稱集合）：tw_revenue、stock_list、終止上市名單、taiwanan ----
names = {}
def add(c, n):
    if isinstance(c, str) and isinstance(n, str):
        n = n.strip().replace(' ', '')
        if len(n) >= 2: names.setdefault(c, set()).add(n)
lr = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}, usecols=['code', 'name']).drop_duplicates()
for c, n in zip(lr.code, lr.name): add(c, n)
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str)
for c, n in zip(sl.code, sl.name): add(c, n)
dl = pd.read_csv(HERE.parent / 'o_infra_v3_rebuild' / 'delist_official.csv', dtype=str)
for c, n in zip(dl.code, dl.name): add(c, n)
tw = pd.read_csv(HERE.parent / 'o_rev_survivorship' / 'src_taiwanan.csv', dtype={'code': str}, usecols=['code', 'name']).drop_duplicates()
for c, n in zip(tw.code, tw.name): add(c, n)
# 常見簡稱（只加在全名後綴為「科技／電子／實業…」時去掉後綴的版本，長度 ≥ 2）
SUF = ['股份有限公司', '科技', '電子', '實業', '工業', '企業', '電腦', '光電', '國際', '半導體', '-KY', 'KY']
for c in list(names):
    for n in list(names[c]):
        for s in SUF:
            if n.endswith(s) and len(n) - len(s) >= 2: names[c].add(n[:-len(s)])
# 一名多碼的別名不用（避免誤配）
inv = {}
for c, ns in names.items():
    for n in ns: inv.setdefault(n, set()).add(c)
alias = {n: next(iter(cs)) for n, cs in inv.items() if len(cs) == 1 and len(n) >= 2 and not n.isdigit()}
codes4 = {c for c in names if re.fullmatch(r'\d{4}', c)}
alias_re = re.compile('|'.join(sorted(map(re.escape, alias), key=len, reverse=True)))

NUM = r'(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?'
PAT = re.compile(r'(?<![前\-－~～至到\d])(\d{1,2})\s*月(?:份)?\s*(?:自結)?\s*(?:合併)?\s*(?:營收|營業收入|業績)\s*'
                 r'(?:(?:為|達|約|僅|來到|攀升至|攀升到|衝上|躍升至|增至|降至|減至|回落至|回升至|跌至|站上|突破|再創新高|創新高)\s*)?'
                 r'(?:新台幣|新臺幣)?\s*' + NUM + r'\s*(億|萬)\s*元')
YOY = re.compile(r'(年增|年減|年成長|年衰退|較去年同期(?:成長|增加|增長|減少|衰退|下滑))\s*(?:率)?\s*(?:約|達)?\s*(-?\d+(?:\.\d+)?)\s*(%|％|成)')
MOM = re.compile(r'(月增|月減|較上月(?:成長|增加|減少|衰退|下滑))\s*(?:率)?\s*(?:約|達)?\s*(-?\d+(?:\.\d+)?)\s*(%|％|成)')
BAD = re.compile(r'(估|預期|預計|可望|將|目標|上看|挑戰|有望|若|去年|前年|同期|累計|上半年|全年)')
ALLOW = re.compile(r'(今年|今\s*[\(（]\s*\d+\s*[\)）]\s*日|今日|昨日|昨|今|日前|稍早|晚間|下午|上午|公布|公佈|宣布|發布|自結|自行結算|的|最新|公告|也|則|亦|在|於|單月|集團|\d{4}\s*年|年|TW|TWO|-|－|\s|[\(（\)）、，,\d])')
CODE_RE = re.compile(r'[\(（]\s*(\d{4})\s*(?:-TW|-TWO|－TW)?\s*[\)）]')

rows = []
files = sorted(f for S_ in SRCS for f in glob.glob(str(S_ / '*.csv')))
fsha = hashlib.sha256()
for f in files:
    fsha.update(Path(f).read_bytes())
    try:
        dfa = pd.read_csv(f)
    except Exception:
        continue
    if 'News' in dfa.columns: ids, txts = dfa['News'], dfa['All']
    else: ids, txts = dfa['date'].astype(str).str.replace('-', ''), dfa['all']
    for nid, txt in zip(ids, txts):
        if not isinstance(txt, str) or ('營收' not in txt and '營業收入' not in txt): continue
        ds = str(nid)[:8]
        try: D = pd.Timestamp(ds)
        except Exception: continue
        for s in re.split(r'[。；\n]', txt):
            if '月' not in s: continue
            for m in PAT.finditer(s):
                mon = int(m.group(1))
                if not 1 <= mon <= 12: continue
                mid = s[m.start():m.start(2)]
                if BAD.search(mid): continue
                # 該月之前 40 字內最近的公司名或代碼
                pre = s[max(0, m.start() - 40):m.start()]
                if BAD.search(pre[-8:]): continue
                cand = []
                for a in alias_re.finditer(pre):
                    st = m.start() - len(pre) + a.start()
                    pc = s[st - 1] if st > 0 else ' '
                    # 名稱左邊界：非中英文字，或常見描述詞尾（廠、商、股…）；避免「世界先進」→「先進」
                    if re.match(r'[\u4e00-\u9fffA-Za-z]', pc) and pc not in '廠商股頭及與和、為業對看':
                        continue
                    cand.append((a.end(), alias[a.group(0)], a.group(0)))
                for a in CODE_RE.finditer(pre):
                    c4 = a.group(1)
                    if c4 in codes4 and not ('2000' <= c4 <= '2030' and 'TW' not in a.group(0)):
                        cand.append((a.end(), c4, c4))
                if not cand: continue
                pos, code, who = max(cand)
                gap = pre[pos:]
                if re.search(r'前\s*$|[\-－~～至、]\s*$|\d\s*[\-－~～、]\s*$', pre): continue
                g2 = ALLOW.sub('', gap)
                if re.search(r'[\u4e00-\u9fffA-Za-z]', g2): continue
                # 公司名與月份之間不能夾另一家公司（取最近者已處理）；也不能夾「、」列舉的其他數字
                ip, dp = m.group(2).replace(',', ''), m.group(3) or ''
                val = float(ip + ('.' + dp if dp else ''))
                unit = 100000.0 if m.group(4) == '億' else 10.0
                k = val * unit
                sig = len((ip + dp).lstrip('0')) or 1
                prec = 0.5 * unit * (10 ** -len(dp)) / k if k > 0 else np.inf     # 四捨五入造成的最大相對誤差
                # 營收月的年份：營收月 ≤ 文章月 → 同年；否則前一年；落後 > 3 個月不收
                y = D.year if mon <= D.month else D.year - 1
                lag = (D.year - y) * 12 + D.month - mon
                if not 0 <= lag <= 3: continue
                post = s[m.end():m.end() + 60]
                mm = MOM.search(post)
                mom = np.nan
                if mm:
                    v = float(mm.group(2)) * (10 if mm.group(3) == '成' else 1)
                    mom = -abs(v) if any(w in mm.group(1) for w in ('減', '衰退', '下滑')) else v
                yy = YOY.search(post)
                yoy = np.nan
                if yy:
                    v = float(yy.group(2)) * (10 if yy.group(3) == '成' else 1)
                    yoy = -abs(v) if any(w in yy.group(1) for w in ('減', '衰退', '下滑')) else v
                rows.append((ds, code, who, f'{y}-{mon:02d}', k, sig, prec, yoy, mom, lag, s[max(0, m.start() - 30):m.end() + 40]))
M = pd.DataFrame(rows, columns=['date', 'code', 'who', 'ym', 'rev_k', 'sig', 'prec', 'yoy_news', 'mom_news', 'lag', 'snip'])
M = M.drop_duplicates(['date', 'code', 'ym', 'rev_k', 'snip']).reset_index(drop=True)
M.to_parquet(HERE / 'raw_news_mentions.parquet')
meta = {'source': 'github.com/aRthur08701244/Intern-in-CTBC @258f42aa（EB_EventExtraction2＋EB_SegAndRawNews，鉅亨網新聞 2013-01～2021-12；clone 在 raw/ctbc）',
        'n_files': len(files), 'files_sha256_concat': fsha.hexdigest(), 'n_mentions': int(len(M)), 'n_code_ym': int(M[['code', 'ym']].drop_duplicates().shape[0]),
        'n_codes': int(M.code.nunique())}
(HERE / 'raw_news_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=1))

#!/usr/bin/env python3
"""數據稽核部 2026-11-21：舊版 xq_scripts/策略利潤圖.html（2015 起、修正前口徑）的新口徑候選檔。
倉庫與暫存區都找不到這張圖的產生程式（git 只有一次提交、沒有對應的 .py），所以用「原檔當樣板、只換資料與寫死的數字」的方式重產：
  1. const D＝xq_scripts/site/profit.json（make_profit.py 2026-11-21 由正式淨值檔產生；去掉 source 鍵，其餘鍵名與原圖相同）；
  2. const PUB（方塊上的公布數字）＝agents/official_nav/out/res_official.json；
  3. 提醒文字只改數字與資料來源：中位數 46.2→46.4、大贏家例子改成目前損益前三名、資料檔路徑。
python3 -I -B make_old_chart.py → public/策略利潤圖.html（不覆蓋原檔）"""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
XQ = Path('/home/user/-/xq_scripts')
h = (XQ / '策略利潤圖.html').read_text(encoding='utf-8')
P = json.loads((XQ / 'site' / 'profit.json').read_text(encoding='utf-8'))
R = json.loads((HERE.parent / 'official_nav' / 'out' / 'res_official.json').read_text(encoding='utf-8'))['strats']
m = re.search(r'const D=(\{.*?\});\n', h, re.S)
Dold = json.loads(m.group(1))
Dnew = {k: P[k] for k in Dold}                                  # 鍵與順序同原圖
assert set(Dold) == {'dates', 'cur', 'xq', 'etf', 'stats', 'years'}
h = h[:m.start(1)] + json.dumps(Dnew, ensure_ascii=False, separators=(',', ':')) + h[m.end(1):]
pc = lambda x: round(x * 100, 2)
e0, xt = R['E0|W10_70'], R['XT|W10_70']; ex = e0['extras']
pub = ('const PUB={cur:{c19:%.2f,full:%.2f,dd:%.2f,path:%.2f},xq:{c19:%.2f,full:%.2f,dd:%.2f,path:%.2f},etf:{c19:%.2f,full:%.2f,dd:%.2f,path:null}};' % (
    pc(e0['std']['cagr_since2019']), pc(e0['std']['cagr_full']), pc(e0['std']['maxdd']), pc(e0['paths_summary']['cagr19_median']),
    pc(xt['std']['cagr_since2019']), pc(xt['std']['cagr_full']), pc(xt['std']['maxdd']), pc(xt['paths_summary']['cagr19_median']),
    pc(ex['e50_since2019']), pc(ex['e50_full']), pc(ex['e50_maxdd'])))
REP = [
    ('const PUB={cur:{c19:48.13,full:34.08,dd:-33.40,path:46.23},xq:{c19:46.47,full:32.32,dd:-38.58,path:44.01},etf:{c19:28.91,full:21.14,dd:-33.96,path:null}};', pub),
    ('現行策略 2019 年起每年的中位數是 46.2%；偶爾晚 0～4 天下單約 43%。', '現行策略 2019 年起每年的中位數是 %.1f%%；偶爾晚 0～4 天下單約 43%%。' % pc(e0['paths_summary']['cagr19_median'])),
    ('（例如 2408、3653、1325）', '（例如 %s）' % '、'.join(ex['top5'][:3])),
    ('・資料：research/tw_stock_selection/agents/s19_xq_threshold/navs.csv.gz（E0|W10_70、XT|W10_70、0050）。',
     '・資料：xq_scripts/site/navs_official.csv.gz（E0|W10_70、XT|W10_70；2026-11-21 起的正式口徑）；0050 用官方含息序列。XQ 版的換股日和現行相同。'),
]
for a, b in REP:
    assert h.count(a) == 1, a
    h = h.replace(a, b)
(HERE / 'public').mkdir(exist_ok=True)
(HERE / 'public' / '策略利潤圖.html').write_text(h, encoding='utf-8')
print(pub); print({k: (Dold['stats'][k]['final'], Dnew['stats'][k]['final']) for k in ('cur', 'xq', 'etf')}, len(Dold['dates']), len(Dnew['dates']))

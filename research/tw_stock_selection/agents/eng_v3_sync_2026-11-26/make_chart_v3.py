#!/usr/bin/env python3
"""ENG-V3-SYNC 2026-11-26：用 S75 v3 淨值（漲跌幅照當時制度）重產 xq_scripts/策略利潤圖_2008_2026.html。
算法與 s75_continuous_2008_2026/make_chart.py 相同（該檔不改），差別：
  1. 淨值讀 v3_2026-11-25_limit7/nav_2008_2026_v3.csv.gz；樣板用 s75 的 chart_template.html（2026-11-26 已把寫死的文字改成 v3）；
  2. 寫檔前先核對：算出的年化／回撤／期末與 external_numbers_v3.json 相符，樣板裡寫死的數字（回撤、2012 年落後、20 條路徑、
     2008～2015 年中超額、大跌段幾段比 0050 深）都能由資料算回來；不符就報錯、不寫檔。
用法：python3 -I -B make_chart_v3.py [輸出 html]（預設 xq_scripts/策略利潤圖_2008_2026.html）"""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
S75 = HERE.parent / 's75_continuous_2008_2026'
V3 = S75 / 'v3_2026-11-25_limit7'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'xq_scripts' / '策略利潤圖_2008_2026.html'
d = pd.read_csv(V3 / 'nav_2008_2026_v3.csv.gz', parse_dates=['date']).set_index('date')
d = d[['V_BRIDGE', 'V_MASK', '0050']].dropna()
def dd_in(s, a, b):
    w = s[a:b]; pk = w.cummax(); r = w / pk - 1; t = r.idxmin(); p = w[:t].idxmax()
    return float(r.min()), p, t
bears = []
for name, a, b in [("2008 金融海嘯", "2008-01-01", "2008-12-31"), ("2011 歐債", "2011-01-01", "2011-12-31"), ("2015", "2015-01-01", "2015-12-31"),
                   ("2018", "2018-01-01", "2018-12-31"), ("2020 疫情", "2020-01-01", "2020-06-30"), ("2022", "2022-01-01", "2022-12-31"),
                   ("2024-07～2025-04", "2024-07-01", "2025-04-30")]:
    e, p, t = dd_in(d['0050'], a, b); s, sp, st = dd_in(d['V_BRIDGE'], a, b); m, _, _ = dd_in(d['V_MASK'], a, b)
    bears.append(dict(n=name, a=str(min(p, sp).date()), b=str(max(t, st).date()), s=round(s * 100, 2), m=round(m * 100, 2), e=round(e * 100, 2)))
w = d.groupby([d.index.isocalendar().year, d.index.isocalendar().week]).tail(1)
if w.index[0] != d.index[0]: w = pd.concat([d.iloc[[0]], w])
yr = {}
for c in d.columns:
    y = d[c].groupby(d.index.year).last(); prev = y.shift(1); prev.iloc[0] = 1_000_000
    yr[c] = {str(k): round(float(v), 2) for k, v in ((y / prev - 1) * 100).items()}
n = (d.index[-1] - d.index[0]).days / 365.25
def mdd(s): return float((s / s.cummax() - 1).min())
st = {c: dict(final=float(d[c].iloc[-1]), cagr=float((d[c].iloc[-1] / 1e6) ** (1 / n) - 1), dd=mdd(d[c])) for c in d.columns}
D = dict(dates=[str(x.date()) for x in w.index], br=[round(float(v)) for v in w['V_BRIDGE']], mk=[round(float(v)) for v in w['V_MASK']],
         etf=[round(float(v)) for v in w['0050']], years=dict(br=yr['V_BRIDGE'], mk=yr['V_MASK'], etf=yr['0050']),
         stats=dict(br=st['V_BRIDGE'], mk=st['V_MASK'], etf=st['0050']), bears=bears, start=str(d.index[0].date()), end=str(d.index[-1].date()))

# ---- 核對 1：與 external_numbers_v3.json（對外數字的唯一來源）相符
X = json.loads((V3 / 'external_numbers_v3.json').read_text(encoding='utf-8'))
xb = {r['項目']: r['v3新值'] for r in X['V_BRIDGE']}; xm = {r['項目']: r['v3新值'] for r in X['V_MASK']}
K_C, K_F, K_D = '全期年化（2008-01-02～2026-10-22，100 萬、位移 0）', '100 萬期末金額（策略）', '最大回撤（策略）'
chk = {
    '年化_策略': (round(st['V_BRIDGE']['cagr'] * 100, 4), xb[K_C]), '期末_策略': (round(st['V_BRIDGE']['final']), xb[K_F]), '回撤_策略': (round(st['V_BRIDGE']['dd'] * 100, 4), xb[K_D]),
    '年化_保守版': (round(st['V_MASK']['cagr'] * 100, 4), xm[K_C]), '期末_保守版': (round(st['V_MASK']['final']), xm[K_F]),
    '年化_0050': (round(st['0050']['cagr'] * 100, 2), xb['0050 全期年化']), '期末_0050': (round(st['0050']['final']), xb['100 萬期末金額（0050）']),
    '回撤_0050': (round(st['0050']['dd'] * 100, 4), xb['最大回撤（0050）']),
    '逐年_策略': ([round(v, 2) for v in yr['V_BRIDGE'].values()], [r['新'] for r in X['逐年_V_BRIDGE']]),
    '逐年_0050': ([round(v, 2) for v in yr['0050'].values()], [r['e0050'] for r in X['逐年_V_BRIDGE']]),
}
bad = {k: v for k, v in chk.items() if v[0] != v[1]}
assert not bad, bad
# ---- 核對 2：樣板寫死的文字要能由 v3 數字算回來
tpl = (S75 / 'chart_template.html').read_text(encoding='utf-8')
y_b, y_e = d['V_BRIDGE'].groupby(d.index.year).last(), d['0050'].groupby(d.index.year).last()
lag2012 = ((y_b[2012] / y_b[2011]) - (y_e[2012] / y_e[2011])) * 100
lose = [k for k in yr['V_BRIDGE'] if yr['V_BRIDGE'][k] < yr['0050'][k]]
deeper = sum(1 for b in bears if b['s'] < b['e'])
pk, lo, rec = xb['最大回撤期間（策略：高點→低點→收復）'].split('→')
ex15 = xb['2008～2015 年中年化（策略，2008-01-02→2015-06-30）'] - xb['2008～2015 年中年化（0050）']
need = [
    f'2012 年輸 {-lag2012:.1f} 個百分點',
    f'7 段裡有 {deeper} 段跌得比 0050 深',
    f'（−{-xb[K_D]:.1f}%，低點在 {int(lo[:4])} 年 {int(lo[5:7])} 月），到 {int(rec[:4])} 年 {int(rec[5:7])} 月底才回到原本的水位',
    '中位數是 %.1f%%（%.1f%%～%.1f%%）' % (xb['20 條路徑全期年化：中位數'], xb['20 條路徑全期年化：最少'], xb['20 條路徑全期年化：最多']),
    f'每年只多約 {ex15:.1f} 個百分點',
    '漲跌停已照當時制度處理（2015 年 6 月前 7%、2008 年 10 月 13～24 日跌幅 3.5%',
    '2026-11-25 起漲跌停改照當時制度',
]
miss = [s for s in need if tpl.count(s) != 1]
assert not miss, miss
assert lose == [str(v) for v in xb['輸 0050 的年份']] and [k for k in ('2011', '2012', '2014', '2016') if k not in lose] == [], lose
assert tpl.count('/*DATA*/null') == 1
page = tpl.replace('/*DATA*/null', json.dumps(D, ensure_ascii=False, separators=(',', ':')))
OUT.write_text('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + page + '</body></html>', encoding='utf-8')
print(json.dumps(dict(out=str(OUT), bytes=OUT.stat().st_size, stats=D['stats'], bears=bears, checks={k: v[0] for k, v in chk.items() if not isinstance(v[0], list)}, template_text_ok=need), ensure_ascii=False, indent=1))

#!/usr/bin/env python3
"""數據稽核部｜已發布數字總對帳（第 1、4 部分）：正式回測總重跑對照＋網站資料對帳。
前置：先跑 rerun_official.py（official_nav/run_official.py 的機械轉出版，只改輸出路徑）產生 out/rerun_*.
本程式不 import 任何研究引擎：績效指標一律從淨值序列（csv）與交易紀錄（csv）用本檔自己的算法重算。
  年化＝exp(Σ ln 日報酬 × 365.25 ÷ 日曆天數) − 1（引擎是 (尾÷頭)^(1/年)；兩種寫法數學等價但程式獨立）
  回撤＝逐日迴圈記最高點（不用 pandas cummax）
  逐年＝每年最後一個交易日淨值相除
用法：python3 -I -B audit_core.py   → out/core.json、table_a_official_rerun.csv、table_site.csv"""
import json, gzip, csv, math, io, subprocess, os
from pathlib import Path
from datetime import date

HERE = Path(__file__).resolve().parent
AG = HERE.parent
ROOT = Path('/home/user/-')
SITE = ROOT / 'xq_scripts' / 'site'
OUT = HERE / 'out'
TSD = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsd'
J = lambda p: json.load(open(p, encoding='utf-8'))


def read_nav(p):
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        rd = csv.reader(f); hdr = next(rd); rows = [r for r in rd]
    dates = [date.fromisoformat(r[0][:10]) for r in rows]
    cols = {h: [float(r[i]) for r in rows] for i, h in enumerate(hdr) if i}
    return dates, cols


def ann(dates, v, a, b):                       # a、b＝索引
    s = 0.0
    for i in range(a + 1, b + 1):
        s += math.log(v[i] / v[i - 1])
    return math.exp(s * 365.25 / (dates[b] - dates[a]).days) - 1


def ann252(v, a, b):
    return (v[b] / v[a]) ** (252 / (b - a)) - 1


def mdd(dates, v, a=0, b=None):
    b = len(v) - 1 if b is None else b
    pk, pki, worst, wi, wpk = v[a], a, 0.0, a, a
    for i in range(a, b + 1):
        if v[i] > pk:
            pk, pki = v[i], i
        d = v[i] / pk - 1
        if d < worst:
            worst, wi, wpk = d, i, pki
    return worst, dates[wpk], dates[wi]


def last_before(dates, y):                     # y 年 1/1 以前最後一個交易日的索引
    k = None
    for i, d in enumerate(dates):
        if d < date(y, 1, 1):
            k = i
    return k


def yearly(dates, v):
    out = {}; prev = v[0]
    ys = sorted({d.year for d in dates})
    for y in ys:
        k = max(i for i, d in enumerate(dates) if d.year == y)
        out[y] = v[k] / prev - 1; prev = v[k]
    return out


def metrics(dates, v):
    n = len(v) - 1; i19 = last_before(dates, 2019); i24 = last_before(dates, 2024); mid = n // 2
    dd, pk, tr = mdd(dates, v)
    return dict(c19=ann(dates, v, i19, n), full=ann(dates, v, 0, n), c24=ann(dates, v, i24, n), maxdd=dd, dd_peak=str(pk), dd_trough=str(tr),
                final=v[-1], yearly=yearly(dates, v), half1=ann(dates, v, 0, mid), half2=ann(dates, v, mid, n),
                c19_252=ann252(v, i19, n), full_252=ann252(v, 0, n), d19=str(dates[i19]), d24=str(dates[i24]), start=str(dates[0]), end=str(dates[-1]))


R = {}
A = []                                          # 表 a 的列


def row(group, item, published, recomputed, tol, method, unit='%', note=''):
    if isinstance(published, (int, float)) and isinstance(recomputed, (int, float)):
        diff = recomputed - published; ok = abs(diff) <= tol + 1e-9
    else:
        diff = ''; ok = (str(published) == str(recomputed))
    A.append(dict(分組=group, 項目=item, 已發布=published, 重算=round(recomputed, 4) if isinstance(recomputed, float) else recomputed,
                  差=round(diff, 4) if diff != '' else '', 容許=tol, 單位=unit, 一致='一致' if ok else '不一致', 重算方法=method, 備註=note))
    return ok


pc = lambda x: x * 100
# ---------------------------------------------------------------- 1a 引擎重跑 vs 已發布結果檔
pub = J(AG / 'official_nav' / 'out' / 'res_official.json'); rr = J(OUT / 'rerun_res_official.json')


def deep(a, b, p='', out=None, skip=('peak_mb', 'secs')):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in skip:
                continue
            if k not in a or k not in b:
                out.append((p + '/' + str(k), '缺', 0)); continue
            deep(a[k], b[k], p + '/' + str(k), out, skip)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((p, '長度', 0))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deep(x, y, f'{p}[{i}]', out, skip)
    elif a != b:
        out.append((p, 'float' if isinstance(a, float) else '值', abs(a - b) if isinstance(a, float) and isinstance(b, float) else 0))
    return out


dd_ = deep(pub, rr)
R['rerun_vs_res_official'] = dict(n_diff=len(dd_), max_float_abs_diff=max([x[2] for x in dd_], default=0.0), first=[x[0] for x in dd_[:10]],
                                  rerun_secs=rr['secs'], rerun_peak_mb=rr['peak_mb'], acceptance=rr['acceptance'])
# ---------------------------------------------------------------- 1b 重跑淨值 vs 網站 navs_official 逐日
dp, cp = read_nav(SITE / 'navs_official.csv.gz'); dr, cr = read_nav(OUT / 'rerun_navs_official.csv.gz')
R['nav_daily_compare'] = dict(same_dates=dp == dr, n_days=len(dp), columns=list(cp),
                              max_abs_diff={k: max(abs(x - y) for x, y in zip(cp[k], cr[k])) for k in cp})
# ---------------------------------------------------------------- 1c 獨立重算指標（讀已發布的 navs_official.csv.gz）
M = {k: metrics(dp, cp[k]) for k in cp}
R['independent_metrics'] = M
E, XA, XB, B0 = M['E0|W10_70'], M['XT|W10_70'], M['XT_B11|W10_70'], M['0050']
e0 = rr['strats']['E0|W10_70']; ex = e0['extras']; ps = e0['paths_summary']
g = '現行（E0｜W10_70，100 萬，標準路徑）'
m1 = '獨立：讀 navs_official.csv.gz 自算'
row(g, '2019 年起每年', 48.76, pc(E['c19']), 0.005, m1)
row(g, '2015-07 起每年（全期）', 34.40, pc(E['full']), 0.005, m1)
row(g, '2024 年起每年', 70.46, pc(E['c24']), 0.005, m1)
row(g, '最大回撤', -32.1, pc(E['maxdd']), 0.05, m1)
row(g, '最大回撤起點（2024-07）', '2024-07', E['dd_peak'][:7], 0, m1, '', E['dd_peak'])
row(g, '最大回撤低點（2025-04）', '2025-04', E['dd_trough'][:7], 0, m1, '', E['dd_trough'])
row(g, '期末淨值（元，res_official）', round(pub['strats']['E0|W10_70']['std']['final_nav']), round(E['final']), 1, m1, '元')
YP = {2015: -10.82, 2016: 15.44, 2017: 22.24, 2018: 0.95, 2019: 31.17, 2020: 100.75, 2021: 44.94, 2022: -17.45, 2023: 57.72, 2024: 63.38, 2025: 56.69, 2026: 75.24}
for y, v in YP.items():
    row(g + '｜逐年', f'{y} 年', v, pc(E['yearly'][y]), 0.005, m1)
row(g, '前半段每年', 26.07, pc(E['half1']), 0.005, m1)
row(g, '後半段每年', 43.22, pc(E['half2']), 0.005, m1)
# 交易統計：讀重跑的交易紀錄，自己算
with gzip.open(OUT / 'rerun_trades_E0.csv.gz', 'rt') as f:
    T = list(csv.DictReader(f))
cl = [t for t in T if t['why'] != 'open_end']
ret = [float(t['proceeds']) / float(t['cost']) - 1 for t in cl]          # 不用引擎的 ret 欄，從成本與得款自己算
R['trade_ret_vs_engine_max_abs'] = max(abs(a - float(t['ret'])) for a, t in zip(ret, cl))
pnl = [float(t['proceeds']) - float(t['cost']) for t in cl]
w = [x for x in ret if x > 0]; l = [x for x in ret if x <= 0]
order = sorted(range(len(cl)), key=lambda i: -pnl[i])[:5]
top5 = [cl[i]['code'] for i in order]; top5share = sum(pnl[i] for i in order) / sum(pnl)
m2 = '獨立：讀重跑交易紀錄，由成本／得款自算'
row(g + '｜交易', '已出場筆數', 381, len(cl), 0, m2, '筆')
row(g + '｜交易', '全部筆數（含期末未平倉）', pub['strats']['E0|W10_70']['std']['n_trades_all'], len(T), 0, m2, '筆')
row(g + '｜交易', '勝率', 50.4, pc(len(w) / len(cl)), 0.05, m2)
row(g + '｜交易', '賺的平均', 27.43, pc(sum(w) / len(w)), 0.005, m2)
row(g + '｜交易', '賠的平均', -9.11, pc(sum(l) / len(l)), 0.005, m2)
exp_ = len(w) / len(cl) * sum(w) / len(w) + len(l) / len(cl) * sum(l) / len(l)
row(g + '｜交易', '每筆期望值（README 寫 +9.30）', 9.30, pc(exp_), 0.005, m2, '%', f'精確 {pc(exp_):.4f}；用四捨五入後的 50.4×27.43−49.6×9.11＝{0.504*27.43-0.496*9.11:.3f}')
row(g + '｜交易', '平均抱幾個交易日', 34, sum(float(t['hold_days']) for t in cl) / len(cl), 0.5, m2, '日')
row(g + '｜交易', '最賺 5 筆占已平倉損益', 36, pc(top5share), 0.5, m2)
row(g + '｜交易', '最賺 5 筆代號', '2408、3653、3017、2344、2059', '、'.join(top5), 0, m2, '')
R['trades'] = dict(n_closed=len(cl), n_all=len(T), win_rate=len(w) / len(cl), avg_win=sum(w) / len(w), avg_loss=sum(l) / len(l), expectancy=exp_, top5=top5, top5_share=top5share)
# 20 條路徑（引擎重跑；統計自己算）
pv = sorted(e0['paths'].values()); pdv = sorted(e0['paths_dd'].values())
med = lambda s: (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2 if len(s) % 2 == 0 else s[len(s) // 2]
m3 = '同引擎重跑 20 條；最少／中位數／最多自己算'
row(g + '｜20 條路徑', '2019 起：最少', 42.01, pc(pv[0]), 0.005, m3)
row(g + '｜20 條路徑', '2019 起：中位數', 46.40, pc(med(pv)), 0.005, m3)
row(g + '｜20 條路徑', '2019 起：最多', 49.29, pc(pv[-1]), 0.005, m3)
row(g + '｜20 條路徑', '最大回撤中位數', -33.4, pc(med(pdv)), 0.05, m3)
row(g + '｜20 條路徑', '最大回撤最深', -39.4, pc(pdv[0]), 0.05, m3)
row(g + '｜20 條路徑', '全部高於 0050（條數）', 20, sum(1 for x in pv if x > B0['c19']), 0, m3, '條')
m4 = '同引擎重跑（rerun_official.py 的 extras）'
row(g + '｜八項檢查', '成本加倍：2019 起', 43.11, pc(ex['costx2_since2019']), 0.005, m4)
row(g + '｜八項檢查', '成本加倍：全期', 30.18, pc(ex['costx2_full']), 0.005, m4)
row(g + '｜八項檢查', '不買最賺 5 檔：2019 起', 44.88, pc(ex['block5_since2019']), 0.005, m4)
row(g + '｜八項檢查', '不買最賺 5 檔：全期', 31.97, pc(ex['block5_full']), 0.005, m4)
# 對 0050：滾動 252 個交易日
s, b = cp['E0|W10_70'], cp['0050']
rrl = [(dp[i], (s[i] / s[i - 252] - 1) - (b[i] / b[i - 252] - 1)) for i in range(252, len(s)) if dp[i] >= date(2016, 3, 1)]
worst = min(rrl, key=lambda x: x[1])
row(g + '｜對 0050', '滾動 12 個月最多落後（個百分點）', 15.3, -pc(worst[1]), 0.05, m1, '個百分點')
row(g + '｜對 0050', '最多落後的結束日', '2025-10-07', str(worst[0]), 0, m1, '')
row(g + '｜對 0050', '落後時間比例', 20.2, pc(sum(1 for x in rrl if x[1] < 0) / len(rrl)), 0.05, m1)
by = B0['yearly']; beat = sum(1 for y in YP if y != 2015 and E['yearly'][y] > by[y])
row(g + '｜對 0050', '11 年裡贏 0050 的年數', 9, beat, 0, m1, '年', '輸的年份：' + '、'.join(str(y) for y in YP if y != 2015 and E['yearly'][y] <= by[y]))
g0 = '0050（含息）'
row(g0, '2019 年起每年', 28.91, pc(B0['c19']), 0.005, m1 + '（navs 的 0050 欄）')
row(g0, '全期每年', 21.14, pc(B0['full']), 0.005, m1)
row(g0, '2024 年起每年', 54.48, pc(B0['c24']), 0.005, m1)
row(g0, '最大回撤', -34.0, pc(B0['maxdd']), 0.05, m1, '%', f"{B0['dd_peak']}～{B0['dd_trough']}")
gx = 'XQ 門檻版（衛星 70%＋0050 底倉，100 萬）'
xa = rr['strats']['XT|W10_70']; xb = rr['strats']['XT_B11|W10_70']
row(gx, '統一換股日 A10：2019 起', 45.09, pc(XA['c19']), 0.005, m1)
row(gx, 'A10：最大回撤', -38.8, pc(XA['maxdd']), 0.05, m1)
row(gx, 'A10：20 條中位數', 43.90, pc(med(sorted(xa['paths'].values()))), 0.005, m3)
row(gx, '舊換股日 B11：2019 起', 46.36, pc(XB['c19']), 0.005, m1)
row(gx, 'B11：最大回撤', -38.1, pc(XB['maxdd']), 0.05, m1)
row(gx, 'B11：20 條中位數', 44.04, pc(med(sorted(xb['paths'].values()))), 0.005, m3)
# 只修除權息（上一版）對照：差
de, ce = read_nav(SITE / 'navs_exright.csv.gz'); ME = metrics(de, ce['E0|W10_70'])
R['exright_only_metrics'] = ME
gd = '期限修正的影響（正式 − 只修除權息）'
m5 = '獨立：navs_official 與 navs_exright 各自重算後相減'
row(gd, '只修除權息 2019 起（README 48.99）', 48.99, pc(ME['c19']), 0.005, m5)
row(gd, '2019 起差（個百分點）', -0.24, pc(E['c19'] - ME['c19']), 0.005, m5, '個百分點')
row(gd, '全期差', -0.14, pc(E['full'] - ME['full']), 0.01, m5, '個百分點', f"只修除權息全期 {pc(ME['full']):.2f}；精確差 −0.147，README 是兩個四捨五入後數字相減（34.40−34.54），不算錯")
row(gd, '2024 起差', -0.29, pc(E['c24'] - ME['c24']), 0.01, m5, '個百分點', f"只修除權息 2024 起 {pc(ME['c24']):.2f}；精確差 −0.297，同上是四捨五入後相減")
px = J(AG / 'exright_engine_fix' / 'out' / 'res_fix1.json')['strats']['E0']['paths']
dpath = sorted(e0['paths'][k] - px[k] for k in px)
row(gd, '20 條逐條差的中位數', -0.35, pc(med(dpath)), 0.005, '同引擎重跑 20 條 − exright_engine_fix/out/res_fix1.json 的 20 條', '個百分點')
row(gd, '20 條中位數的差', -0.14, pc(med(pv) - med(sorted(px.values()))), 0.006, '同上', '個百分點')
R['conv_252'] = dict(說明='若改用「252 個交易日＝1 年」的年化慣例（不是發布口徑，只列敏感度）', E0_c19=pc(E['c19_252']), E0_full=pc(E['full_252']))
R['deadline_dates'] = rr['av_changed_vs_rule']
# ---------------------------------------------------------------- 4 網站資料
W = []


def wrow(f, item, pubv, calc, tol, note=''):
    ok = abs(calc - pubv) <= tol + 1e-9 if isinstance(pubv, (int, float)) else pubv == calc
    W.append(dict(檔案=f, 項目=item, 網站值=pubv, 重算=round(calc, 4) if isinstance(calc, float) else calc, 容許=tol, 一致='一致' if ok else '不一致', 備註=note))
    return ok


nav = J(SITE / 'data' / 'nav.json')['rows']; ix = {str(d): i for i, d in enumerate(dp)}
mx = {'現行策略': 0.0, 'XQ版': 0.0, '0050': 0.0}; miss = 0; notweek = 0
keymap = {'現行策略': 'E0|W10_70', 'XQ版': 'XT|W10_70', '0050': '0050'}
for r in nav:
    if r['日期'] not in ix:
        miss += 1; continue
    i = ix[r['日期']]
    if i + 1 < len(dp) and dp[i + 1].isocalendar()[:2] == dp[i].isocalendar()[:2]:
        notweek += 1
    for k, c in keymap.items():
        mx[k] = max(mx[k], abs(r[k] / cp[c][i] - 1))
wrow('nav.json', f'{len(nav)} 列的日期都在 navs_official 裡（缺幾列）', 0, miss, 0)
wrow('nav.json', '不是當週最後一個交易日的列數（說明寫「每週取最後一個交易日」）', 0, notweek, 8,
     f'{notweek} 列是補行交易的週六那幾週（2016-01-30 等），網站取週五；數值本身正確，只是說明文字不精確')
wrow('nav.json', '現行策略欄 vs navs_official E0（最大相對差）', 0.0, mx['現行策略'], 1e-6)
wrow('nav.json', 'XQ版欄 vs navs_official XT（A10）（最大相對差）', 0.0, mx['XQ版'], 1e-6)
wrow('nav.json', '0050 欄 vs navs_official 0050 欄（最大相對差）', 0.0, mx['0050'], 1e-3,
     f"網站 0050 用精確含息價、另一來源；最後一列 {nav[-1]['0050']:,} 對 navs 欄 {cp['0050'][-1]:,.0f}")
wrow('nav.json', '最後一列日期', str(dp[-1]), nav[-1]['日期'], 0)
yj = J(SITE / 'data' / 'yearly.json')['rows']
for r in yj:
    y = r['年']
    wrow('yearly.json', f'{y} 現行策略', r['現行策略'], pc(E['yearly'][y]), 0.05)
    wrow('yearly.json', f'{y} XQ版', r['XQ版'], pc(XA['yearly'][y]), 0.05)
    wrow('yearly.json', f'{y} 0050', r['0050'], pc(B0['yearly'][y]), 0.05, '0050 欄來源不同（精確含息價），容許 0.05' if abs(r['0050'] - pc(B0['yearly'][y])) > 0.05 else '')
# live.json：state 持股檔＋官方日 K
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')


def daily(d):
    txt = subprocess.run(['git', '-C', TSD, 'show', f'HEAD:data/universe/daily/{d}.csv'], capture_output=True, text=True, check=True, env=env).stdout
    return {r['stock_id']: r for r in csv.DictReader(io.StringIO(txt))}


days = sorted(x.rsplit('/', 1)[1][:-4] for x in subprocess.run(['git', '-C', TSD, 'ls-tree', '--name-only', 'HEAD', 'data/universe/daily/'],
                                                                 capture_output=True, text=True, check=True, env=env).stdout.split())
hold = {r['代號']: float(r['股數']) for r in csv.DictReader(open(SITE / 'state' / 'holdings_after_2026-11.csv', encoding='utf-8-sig'))}
cash = float([x for x in open(SITE / 'state' / 'cash_after_2026-11.txt', encoding='utf-8').read().split('\n') if x.strip() and not x.startswith('#')][0])
live = J(SITE / 'data' / 'live.json')['rows']
d0 = daily('2026-11-12')
base = sum(n * float(d0[c]['open']) for c, n in hold.items()) + cash
f = 1e6 / base
R['live'] = dict(base_open_value=base, factor=f, cash=cash, holdings=hold, rows=[])
for r in live:
    dd = daily(r['日期'])
    tot = (sum(n * float(dd[c]['close']) for c, n in hold.items()) + cash) * f
    b50 = 1e6 / float(d0['0050']['open']) * float(dd['0050']['close'])
    R['live']['rows'].append(dict(日期=r['日期'], 策略_網站=r['策略'], 策略_重算=round(tot, 2), etf_網站=r['0050'], etf_重算=round(b50, 2)))
    wrow('live.json', f"{r['日期']} 策略總資產（元）", r['策略'], tot, 1.0)
    wrow('live.json', f"{r['日期']} 只買 0050（元）", r['0050'], b50, 1.0)
wrow('live.json', '持股檔起始日開盤總值（status.json 917,733.4）', 917733.4, base, 0.06)
wrow('live.json', '換算倍數（1.089641）', 1.089641, f, 5e-7)
wrow('live.json', '最後一列＝官方日 K 最新一天？', days[-1], live[-1]['日期'], 0, f'倉庫最新日 K {days[-1]}')
st = J(SITE / 'data' / 'status.json')['rows'][0]; hj = J(SITE / 'data' / 'holdings.json')['rows']
tot_row = [x for x in hj if x['列別'] == '合計'][0]
wrow('holdings.json', '合計列最新市值＝live.json 最後一天', live[-1]['策略'], tot_row['最新市值'], 0)
wrow('holdings.json', '個股＋0050＋現金 加總＝合計', tot_row['最新市值'], sum(x['最新市值'] for x in hj if x['列別'] != '合計' and x['列別'] != '已實現'), 0)
wrow('status.json', '目前總資產＝live.json 最後一天', live[-1]['策略'], st['目前總資產'], 0)
wrow('status.json', '領先 0050（個百分點）', st['領先0050_個百分點'], (live[-1]['策略'] - live[-1]['0050']) / 1e4, 0.005)
# 期間持股有沒有除權息（用日 K 自己的漲跌價差檢查：前收 ≠ 收盤 − 漲跌價差 就是事件日）
# 季線價位：抽 2 檔，用原始收盤 60 日簡單平均（期間沒有除權息事件時＝還原口徑）
last = live[-1]['日期']; k = days.index(last); win = days[k - 60:k + 1]          # 61 天：多 1 天用來驗前收
dayc = {d: daily(d) for d in win}
for code in ('2480', '4938', '3311', '1802'):
    cl_ = [float(dayc[d][code]['close']) for d in win]
    chg = [float(dayc[d][code]['change'] or 0) for d in win]
    breaks = [win[i] for i in range(1, len(win)) if abs((cl_[i] - chg[i]) - cl_[i - 1]) > 0.011]
    ma = sum(cl_[1:]) / 60
    hrow = [x for x in hj if x['代號'] == code][0]
    wrow('holdings.json', f'{code} 季線價位（60 日原始收盤平均；窗內除權息／漲跌價差斷點 {len(breaks)} 個）', hrow['季線價位'], ma, 0.011 if not breaks else 0.5,
         '斷點日：' + '、'.join(breaks) if breaks else '窗內無除權息，原始價＝還原價')
    wrow('holdings.json', f'{code} 距季線%', hrow['距季線%'], (cl_[-1] / ma - 1) * 100, 0.011 if not breaks else 0.5)
R['site_n'] = len(W); R['site_bad'] = [w_ for w_ in W if w_['一致'] != '一致']
R['table_a_n'] = len(A); R['table_a_bad'] = [a for a in A if a['一致'] != '一致']


def dump_csv(p, rows):
    with open(p, 'w', newline='', encoding='utf-8-sig') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


dump_csv(HERE / 'table_a_official_rerun.csv', A); dump_csv(HERE / 'table_site.csv', W)
json.dump(R, open(OUT / 'core.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('引擎重跑 vs res_official：不同的欄位數', R['rerun_vs_res_official']['n_diff'], '最大浮點差', R['rerun_vs_res_official']['max_float_abs_diff'])
print('淨值逐日：', R['nav_daily_compare'])
print('表 a：', len(A), '列，不一致', len(R['table_a_bad']))
for a in R['table_a_bad']:
    print('  ✗', a)
print('網站：', len(W), '列，不一致', len(R['site_bad']))
for a in R['site_bad']:
    print('  ✗', a)
print('252 慣例', R['conv_252'], '交易報酬自算 vs 引擎欄 最大差', R['trade_ret_vs_engine_max_abs'])

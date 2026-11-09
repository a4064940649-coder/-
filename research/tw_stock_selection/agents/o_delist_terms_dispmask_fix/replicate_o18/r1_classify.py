#!/usr/bin/env python3
"""o18 題 2 獨立複製（步驟 1）：市場初篩＋盲分類＋δ。
只讀 raw/market_*.csv、raw/ws_*.txt、raw/lc_*.txt、event_list_frozen.csv、o17 events_v2.csv、V7X 面板（x_data.C）。
文字分類是複製員讀 raw 文字後手寫在 TEXT 表（每條附逐字片段），程式斷言片段逐字出現在對應 raw 檔。"""
import json, math, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TOP = HERE.parent; RAW = TOP / 'raw'; AG = TOP.parent
EV = pd.read_csv(TOP / 'event_list_frozen.csv', dtype=str)
assert len(EV) == 22 and (EV.status == 'NOT_SEARCHED').all()
E2 = pd.read_csv(AG / 'o_engine_delist_consideration' / 'events_v2.csv', dtype=str)

# ---------- 文字盲分類（複製員依 raw 文字判定；片段必須逐字出現在 raw 檔） ----------
# kind: cash（每股現金）、swap（ratio＝每 1 股目標換存續股數）、mixed（cash＋ratio）
TEXT = {
 '2432': ('VERIFIED_TEXT', 'ws_2432_0.txt', '倚天1.07股換發宏碁1股', dict(kind='swap', ratio=1/1.07, succ='2353')),
 '2403': ('TYPE_TEXT', 'ws_2403_0.txt', '大聯大擬以發行新股股份轉換方式取得友尚100%普通股股票', dict(kind='swap_type', succ='3702')),
 '2452': ('TYPE_TEXT', 'ws_2452_1.txt', '2009年台達電收購被動元件廠乾坤科技', dict(kind='acq_type', succ='2308')),
 '1716': ('TYPE_TEXT', 'ws_1716_0.txt', '透過股份轉換方式設立', dict(kind='conv_newsucc', succ='3705')),
 '1520': ('TYPE_TEXT', 'ws_1520_1.txt', '2007 年引進私募基金下市', dict(kind='cash_type')),
 '3007': ('TENDER_TEXT', 'ws_3007_0.txt', '以每股新台幣 109 元現金收購綠點', dict(kind='tender', tender=109.0)),
 '8078': ('VERIFIED_TEXT', 'ws_8078_0.txt', '未來合併對價亦為50.8元', dict(kind='cash', cash=50.8)),
 '6280': ('TENDER_TEXT', 'ws_6280_1.txt', 'tender offer at NT$93 per share', dict(kind='tender', tender=93.0)),
 '8008': ('TENDER_TEXT', 'lc_8008_0.txt', '以每股32.75元公開收購建興電', dict(kind='tender', tender=32.75)),
 '3559': ('TENDER_TEXT', 'lc_3559_0.txt', '以每股24元公開收購全智科', dict(kind='tender', tender=24.0)),
 '2473': ('VERIFIED_TEXT', 'ws_2473_1.txt', 'remaining holders receiving NT$57.00 per share', dict(kind='cash', cash=57.0)),
 '3063': ('TYPE_TEXT', 'lc_3063_0.txt', '2010 年推動旗下飛信與頎邦合併', dict(kind='swap_type', succ='6147')),
 '3009': ('VERIFIED_TEXT', 'ws_3009_0.txt', '奇美電股東每 2.05 股可換得群創 1 股', dict(kind='swap', ratio=1/2.05, succ='3481')),
 '2446': ('TYPE_TEXT', 'ws_2446_0.txt', '欣興合併全懋', dict(kind='swap_type', succ='3037')),
 '2394': ('VERIFIED_TEXT', 'ws_2394_0.txt', '一股換普立爾三.九股', dict(kind='swap', ratio=1/3.9, succ='2317')),
 '3012': ('VERIFIED_TEXT', 'ws_3012_0.txt', '每 3.5 股廣輝股票換發 1 股友達股票', dict(kind='swap', ratio=1/3.5, succ='2409')),
 '2456': ('VERIFIED_TEXT', 'ws_2456_0.txt', '每1股奇力新普通股將換發0.2002股國巨普通股', dict(kind='swap', ratio=0.2002, succ='2327')),
 '2809': ('VERIFIED_TEXT', 'ws_2809_0.txt', '京城銀普通股每股可換得現金26.75元及1.15股永豐金普通股', dict(kind='mixed', cash=26.75, ratio=1.15, succ='2890')),
 '2833': ('VERIFIED_TEXT', 'ws_2833_0.txt', '1股台壽保換發1.44股中信金', dict(kind='swap', ratio=1.44, succ='2891')),
 '2888': ('VERIFIED_TEXT', 'ws_2888_0.txt', '每1股新光金換得台新金0.672股普通股加0.175股辛種特別股', dict(kind='mixed_pref', ratio=0.672, pref=0.175, succ='2887')),
 '5305': ('VERIFIED_TEXT', 'ws_5305_0.txt', '每股現金對價為新台幣42.5元', dict(kind='cash', cash=42.5)),
 '6004': ('SEARCHED_NOT_FOUND', 'ws_6004_0.txt', '沒有找到 2007 年元京證券股東的換股比例', dict(kind='none')),
}
assert set(TEXT) == set(EV.code)
for c, (st, f, snip, _) in TEXT.items():
    txt = (RAW / f).read_text()
    assert snip in txt, (c, f, snip)

# ---------- 市場初篩 ----------
def screen(code):
    m = pd.read_csv(RAW / f'market_{code}.csv', parse_dates=['date'])
    tg = m.target.to_numpy(float)
    c11 = tg[-11:]; r = c11[1:] / c11[:-1] - 1
    frac = float(np.mean(np.abs(r) <= 0.005 + 1e-12)); c10 = tg[-10:]
    width = float((c10.max() - c10.min()) / c10.min())
    lock = frac >= 0.8 and width <= 0.02
    out = dict(lt=str(m.date.iloc[-1].date()), lc=float(tg[-1]), frac=frac, width=width, lock=lock, r2=None, ratio=None)
    if 'succ' in m.columns and m.succ.notna().sum() >= 21:
        s = m.succ.to_numpy(float)
        a = tg[-21:]; b = s[-21:]
        ra = a[1:] / a[:-1] - 1; rb = b[1:] / b[:-1] - 1
        ok = np.isfinite(ra) & np.isfinite(rb)
        X = np.c_[np.ones(ok.sum()), rb[ok]]; beta, *_ = np.linalg.lstsq(X, ra[ok], rcond=None)
        res = ra[ok] - X @ beta; r2 = 1 - res.var() / ra[ok].var()
        out['r2'] = float(r2); out['ratio'] = float(np.median(tg[-10:] / s[-10:]))
        out['succ_lt'] = float(s[-1])
    out['market'] = 'MARKET_TRACK' if (out['r2'] is not None and out['r2'] >= 0.6) else ('MARKET_LOCK' if lock else 'MARKET_NONE')
    return out

MK = {c: screen(c) for c in EV.code}

# 斷價：用 V7X 面板
sys.path.insert(0, str(AG / 'o_exec_engine_rewrite_v7x'))
import x_data as D
sys.path.pop(0)
for r in EV.itertuples():
    j = D.CI.get(r.code); ltU = pd.Timestamp(r.last_trade)
    term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) else ltU + pd.Timedelta(days=14)
    w = np.flatnonzero(D.TR[:, j]); w = w[D.IDX[w] <= max(term, ltU)]
    lt = w[-1]; last20 = np.arange(lt - 19, lt + 1)
    MK[r.code]['panel_lt'] = str(D.IDX[lt].date())
    MK[r.code]['gap'] = bool(D.IDX[lt] < ltU - pd.Timedelta(days=7) or (~D.TR[last20, j]).sum() >= 5)
    MK[r.code]['panel_lc'] = float(D.C[lt, j])
    assert MK[r.code]['lt'] == MK[r.code]['panel_lt'], r.code

# ---------- δ ----------
def succ_close(code, d):
    j = D.CI.get(code)
    if j is None: return None
    i = D.TPOS.get(pd.Timestamp(d))
    v = D.C[i, j] if i is not None else np.nan
    return float(v) if np.isfinite(v) else None

devs = {}
for c, (st, f, snip, t) in TEXT.items():
    if st != 'VERIFIED_TEXT': continue
    lc = MK[c]['lc']; lt = MK[c]['lt']
    if t['kind'] == 'cash': v = t['cash']
    elif t['kind'] in ('swap', 'mixed'):
        sc = succ_close(t['succ'], lt)
        if sc is None: devs[c] = None; continue
        v = t['ratio'] * sc + t.get('cash', 0.0)
    else: devs[c] = None; continue   # 2888 特別股價值未核實 → 不進 δ
    devs[c] = v / lc - 1
# o17 已核實 11 起（events_v2 的條款；在 lt 有存續股價者才進 δ）
o17 = E2[E2.verified == '已核實（新聞）']
for r in o17.itertuples():
    j = D.CI[r.code]; ltU = pd.Timestamp(r.last_trade)
    term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) else ltU + pd.Timedelta(days=14)
    w = np.flatnonzero(D.TR[:, j]); w = w[D.IDX[w] <= max(term, ltU)]; lt = w[-1]; lc = float(D.C[lt, j])
    cash = float(r.cash) if isinstance(r.cash, str) else 0.0
    if isinstance(r.ratio, str):
        sc = succ_close(r.successor, D.IDX[lt])
        if sc is None: devs[r.code] = None; continue
        v = cash + float(r.ratio) * sc + (float(r.pref_ratio) * float(r.pref_value) if isinstance(r.pref_ratio, str) else 0.0)
    else: v = cash
    devs[r.code] = v / lc - 1
dv = {k: v for k, v in devs.items() if v is not None}
kmax = max(dv, key=lambda k: abs(dv[k]))
delta = math.ceil(abs(dv[kmax]) * 1000 - 1e-9) / 1000
out = dict(text={c: dict(status=v[0], file=v[1], snippet=v[2], terms=v[3]) for c, v in TEXT.items()},
           market=MK, devs=devs, delta=delta, delta_pre2013=delta * 1.25, dmax_code=kmax,
           counts=pd.Series([v[0] for v in TEXT.values()]).value_counts().to_dict())
V20 = ["1520","1716","2311","2325","2394","2403","2432","2446","2452","2473","3007","3009","3012","3063","3474","3559","6280","6286","8008","8078"]
ver = [c for c in V20 if c in ('2311', '2325', '3474', '6286') or TEXT.get(c, ('',))[0] == 'VERIFIED_TEXT']
out['v7x20_verified'] = ver
out['uopt_verified'] = [c for c in ["2403","2432","2473","3009","3012","3063"] if TEXT[c][0] == 'VERIFIED_TEXT']
out['verbatim_only_v7x'] = [c for c in ver if c in ('2311', '2325', '3474', '6286') or TEXT[c][1].startswith('lc_')]
(HERE / 'r1_classify.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for c in EV.code:
    m = MK[c]; print(c, TEXT[c][0], m['market'], round(m['frac'], 2), round(m['width'] * 100, 2), m['r2'] and round(m['r2'], 3), m['ratio'] and round(m['ratio'], 4), m['gap'])
print('devs', {k: (round(v, 4) if v is not None else None) for k, v in devs.items()})
print('delta', delta, delta * 1.25, kmax, out['counts'], len(ver), ver, out['uopt_verified'], out['verbatim_only_v7x'])

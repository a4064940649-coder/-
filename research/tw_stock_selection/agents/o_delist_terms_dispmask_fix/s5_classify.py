#!/usr/bin/env python3
"""步驟 4b／5：依搜尋紀錄（search_log_web.csv、search_log_local.csv、raw/ws_*.txt、raw/lc_*.txt）與市場初篩（market_screen.json）判定 22 起事件狀態，
建 events_v3.csv（只是對價表；不計算任何報酬）。狀態只能由搜尋紀錄改變（斷言每起事件都有 ≥1 筆搜尋紀錄）。δ 依 SPEC_draft 規則估計。
python3 -I -B s5_classify.py"""
import json, hashlib, math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent; CZ = AG / 'o_engine_delist_consideration'
sys.path.insert(0, str(HERE)); import mkt as M; sys.path.pop(0)
D = M.D
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
E2 = pd.read_csv(CZ / 'events_v2.csv', dtype=str)
LST = pd.read_csv(HERE / 'event_list_frozen.csv', dtype=str)
MS = json.loads((HERE / 'market_screen.json').read_text(encoding='utf-8'))
SW = pd.read_csv(HERE / 'search_log_web.csv', dtype=str); SL = pd.read_csv(HERE / 'search_log_local.csv', dtype=str)
for c in LST.code:
    assert (SW.event_code == c).any() and (SL.event_code == c).any(), f'{c} 沒有搜尋紀錄'

# ---- 文字證據（逐筆指名來源；數字由程式斷言必須逐字出現在 raw 檔中）
TXT = {
 '2432': dict(text='VERIFIED_TEXT', kind='swap', ratio=1 / 1.07, succ='2353', raw=['ws_2432_0.txt'], must=['倚天1.07股換發宏碁1股'], src_kind='websearch_summary', cite='自由時報財經 2008-03-04（ec.ltn.com.tw/article/paper/193332）'),
 '3009': dict(text='VERIFIED_TEXT', kind='swap', ratio=1 / 2.05, succ='3481', raw=['ws_3009_0.txt'], must=['奇美電股東每 2.05 股可換得群創 1 股'], src_kind='websearch_summary', cite='TVBS／經濟日報（bnext 12836）；查詢詞含 2.05'),
 '3012': dict(text='VERIFIED_TEXT', kind='swap', ratio=1 / 3.5, succ='2409', raw=['ws_3012_0.txt'], must=['每 3.5 股 QDI 普通股換發 AUO 一股普通股'], src_kind='websearch_summary', cite='AUO 2006-04 新聞稿、AUO 20-F（capedge AUOTY-20F-2006FY）'),
 '2394': dict(text='VERIFIED_TEXT', kind='swap', ratio=1 / 3.9, succ='2317', raw=['ws_2394_0.txt'], must=['一股換普立爾三.九股'], src_kind='websearch_summary', cite='bnext 10106／moneyweekly；文字比例早於鴻海 2006-08 除權'),
 '2473': dict(text='VERIFIED_TEXT', kind='cash', cash=57.0, raw=['ws_2473_1.txt'], must=['remaining holders receiving NT$57.00 per share'], src_kind='websearch_summary', cite='Synopsys SEC 8-K（000119312512488345）；查詢詞含 57'),
 '8078': dict(text='VERIFIED_TEXT', kind='cash', cash=50.8, raw=['ws_8078_0.txt'], must=['未來合併對價亦為50.8元'], src_kind='websearch_summary', cite='鉅亨／wantgoo（仁寶公告）'),
 '5305': dict(text='VERIFIED_TEXT', kind='cash', cash=42.5, raw=['ws_5305_0.txt'], must=['每位LSC股東獲得新台幣42.50元'], src_kind='websearch_summary', cite='CNA 201908090142、Diodes SEC 文件'),
 '2833': dict(text='VERIFIED_TEXT', kind='swap', ratio=1.44, succ='2891', raw=['ws_2833_0.txt'], must=['1股台壽保換發1.44股中信金'], src_kind='websearch_summary', cite='鉅亨 963536 等；文字比例早於中信金 2015-08 除權'),
 '2456': dict(text='VERIFIED_TEXT', kind='swap', ratio=0.2002, succ='2327', raw=['ws_2456_0.txt'], must=['每1股奇力新普通股將換發0.2002股國巨普通股'], src_kind='websearch_summary', cite='今周刊 202106300075、自由 3587335'),
 '2809': dict(text='VERIFIED_TEXT', kind='mixed', cash=26.75, ratio=1.15, succ='2890', raw=['ws_2809_0.txt'], must=['京城銀普通股每股可換得現金26.75元及1.15股永豐金普通股'], src_kind='websearch_summary', cite='經濟日報 5613/9034041'),
 '2888': dict(text='VERIFIED_TEXT', kind='mixed', ratio=0.672, pref_ratio=0.175, succ='2887', raw=['ws_2888_0.txt'], must=['每1股新光金換得台新金0.672股普通股加0.175股辛種特別股'], src_kind='websearch_summary', cite='永豐 sinotrade；特別股價值未核實（不進 δ）'),
 '3007': dict(text='TENDER_TEXT', kind='cash_acq', tender=109.0, raw=['ws_3007_0.txt'], must=['以每股新台幣 109 元現金收購綠點 100% 已發行股份', '剩餘少數股權預計在收購定案時'], src_kind='websearch_summary', cite='Jabil 8-K（sec.gov 000119312507061716）；第二段對價沒有數字'),
 '6280': dict(text='TENDER_TEXT', kind='cash_acq', tender=93.0, raw=['ws_6280_1.txt'], must=['at NTD 93 per share in cash', 'acquire the remaining shares through a share swap and merger'], src_kind='websearch_summary', cite='Fairchild 8-K（capedge 0001181431-07-009702）；第二段為 share swap and merger、對價無數字'),
 '8008': dict(text='TENDER_TEXT', kind='cash_acq', tender=32.75, raw=['lc_8008_0.txt'], must=['以每股32.75元，公開收購', '發行可贖回特別股為對價'], src_kind='local_corpus_verbatim', cite='鉅亨 2013-01-31（CTBC 鏡像 20131_18.csv 第 86 行）；後續合併「改現金收購」只有標題（ws_8008_0）'),
 '3559': dict(text='TENDER_TEXT', kind='cash_acq', tender=24.0, raw=['lc_3559_0.txt'], must=['以每股24元收購全智科'], src_kind='local_corpus_verbatim', cite='鉅亨 2016-07-25（CTBC 鏡像 20167_13.csv 第 101 行）；2017 第二次收購／合併對價無數字'),
 '1520': dict(text='TYPE_TEXT', kind='cash_acq', raw=['ws_1520_1.txt'], must=['引進私募基金下市'], src_kind='websearch_summary', cite='鉅亨 2017（復盛應用 2007 引進私募基金下市）'),
 '1716': dict(text='TYPE_TEXT', kind='conv_newsucc', succ='3705', raw=['ws_1716_0.txt'], must=['透過股份轉換方式設立', '上市日期 20110103'], src_kind='websearch_summary', cite='technews 公司資料、cmoney；換股比例無數字'),
 '2403': dict(text='TYPE_TEXT', kind='swap', succ='3702', raw=['ws_2403_0.txt'], must=['大聯大擬以發行新股股份轉換方式取得友尚100%普通股股票'], src_kind='websearch_summary', cite='大聯大 IR REPORT_3397_1（獨立專家意見書）；比例無數字'),
 '2446': dict(text='TYPE_TEXT', kind='swap', succ='3037', raw=['ws_2446_0.txt', '../o_engine_delist_consideration/raw/ev_2446_0.txt'], must=[], src_kind='websearch_summary', cite='今周刊 200904020003 標題（欣興合併全懋）；o17 ev_2446_0'),
 '2452': dict(text='TYPE_TEXT', kind='acq', raw=['ws_2452_1.txt', '../o_engine_delist_consideration/raw/ev_2452_0.txt'], must=['2009年台達電收購被動元件廠乾坤科技'], src_kind='websearch_summary', cite='NYCU 個案論文 11536/143400；o17 ev_2452_0（台達電公開收購，金額未核實）'),
 '3063': dict(text='TYPE_TEXT', kind='acq', succ='6147', raw=['lc_3063_0.txt', '../o_engine_delist_consideration/raw/ev_3063_0.txt'], must=['2010 年推動旗下飛信與頎邦合併'], src_kind='local_corpus_verbatim', cite='鉅亨 2018-10（CTBC 鏡像 201810_10.csv 第 46 行）；存續頎邦為上櫃股、不在面板'),
 '6004': dict(text='SEARCHED_NOT_FOUND', kind=None, raw=['ws_6004_0.txt'], must=[], src_kind=None, cite='WebSearch 1 次＋本地語料 10 個，皆無'),
}
assert set(TXT) == set(LST.code), set(LST.code) ^ set(TXT)
for c, x in TXT.items():
    for f in x['raw']:
        p = (HERE / 'raw' / f) if not f.startswith('..') else (HERE / f).resolve()
        assert p.exists(), (c, f)
    s = ''.join(((HERE / 'raw' / f) if not f.startswith('..') else (HERE / f).resolve()).read_text(encoding='utf-8') for f in x['raw'])
    for m in x['must']:
        assert m in s, (c, m)
    x['raw_sha256'] = [sha((HERE / 'raw' / f) if not f.startswith('..') else (HERE / f).resolve())[:16] for f in x['raw']]

def close_at(code, t):
    j = D.CI.get(code)
    return float(D.C[t, j]) if j is not None and np.isfinite(D.C[t, j]) else np.nan

rows = []
for c in list(LST.code):
    r = E2[E2.code == c].iloc[0]; x = TXT[c]; m = MS[c]
    j, lt = M.lt_index(c, r.last_trade, r.terminate_date)
    lc = float(D.C[lt, j])
    succ = x.get('succ') or (r.successor if isinstance(r.successor, str) else None)
    sfirst = r.succ_first_trade if isinstance(r.succ_first_trade, str) else (r.terminate_date if isinstance(r.terminate_date, str) else None)
    if c == '1716': sfirst = '2011-01-03'
    sc = close_at(succ, lt) if succ else np.nan
    dev = None
    if x['text'] == 'VERIFIED_TEXT':
        if x['kind'] == 'cash': val = x['cash']
        elif x['kind'] == 'swap': val = x['ratio'] * sc
        elif x['kind'] == 'mixed': val = x.get('cash', 0.0) + x['ratio'] * sc if 'pref_ratio' not in x else np.nan
        dev = val / lc - 1 if np.isfinite(val) else None
    mk = m['market'].replace('GAP|', '')
    # 結算形式（SPEC 狀態表）
    if x['text'] == 'VERIFIED_TEXT':
        settle = x['kind']; status = 'VERIFIED_TEXT'
    elif x['text'] in ('TENDER_TEXT', 'TYPE_TEXT'):
        status = x['text'] + '+' + mk
        if x['kind'] in ('swap',) and mk == 'MARKET_TRACK': settle = 'track'
        elif x['kind'] == 'swap' and mk != 'MARKET_TRACK': settle = 'fixed'          # 換股但無法估比例 → 空窗前固定（acq_ur 型）
        elif x['kind'] == 'conv_newsucc': settle = 'conv_newsucc'
        else: settle = 'fixed'
    else:
        status = 'SEARCHED_NOT_FOUND+' + mk
        settle = 'fixed' if mk == 'MARKET_LOCK' else ('track' if mk == 'MARKET_TRACK' else 'unknown')
    conflict = bool(dev is not None and abs(dev) > 0.10)
    rows.append(dict(code=c, name=LST.loc[LST.code == c, 'name'].iloc[0], priority=int(LST.loc[LST.code == c, 'priority'].iloc[0]), status=status,
                     text_status=x['text'], market=m['market'], settle=settle, cash=x.get('cash'), ratio_text=x.get('ratio'), pref_ratio=x.get('pref_ratio'),
                     tender=x.get('tender'), ratio_market=m.get('ratio_med10'), R2=m.get('track_R2'), lock_frac=m['lock_frac'], lock_width=m['lock_width'],
                     successor=succ, succ_first_trade=sfirst, last_trade=r.last_trade, terminate_date=r.terminate_date, lt_panel=str(D.IDX[lt].date()), lc=lc,
                     succ_close_lt=sc, dev_text_vs_lc=dev, conflict=conflict, src_kind=x['src_kind'], cite=x['cite'], raw_files=';'.join(x['raw']), raw_sha256=';'.join(x['raw_sha256']),
                     v7x_settled=LST.loc[LST.code == c, 'v7x_settled'].iloc[0] == 'True', uopt=LST.loc[LST.code == c, 'uopt_settled'].iloc[0] == 'True',
                     n_search_web=int((SW.event_code == c).sum()), n_search_local_sources=int((SL.event_code == c).sum())))
V3 = pd.DataFrame(rows)
# ---- δ：全部 VERIFIED_TEXT（o17 已核實 11 起 ＋ 本題新核實）lt 當日條款價值 ÷ lc − 1
DEVS = {}
for c in E2.code[E2.verified == '已核實（新聞）']:
    v = E2[E2.code == c].iloc[0]['dev_vs_lastclose_lt']
    if isinstance(v, str) and v: DEVS[c] = float(v)
for r in V3.itertuples():
    if r.text_status == 'VERIFIED_TEXT' and r.dev_text_vs_lc is not None and np.isfinite(r.dev_text_vs_lc): DEVS[r.code] = float(r.dev_text_vs_lc)
dmax = max(abs(v) for v in DEVS.values()); d = math.ceil(dmax * 1000 - 1e-9) / 1000
V3['delta_evt'] = [round(d * (1.25 if pd.Timestamp(lt) < pd.Timestamp('2013-01-01') else 1.0), 6) for lt in V3.lt_panel]
V3.to_csv(HERE / 'events_v3.csv', index=False)
cashd = {k: v for k, v in DEVS.items() if v >= 0 and k in ('1704', '1902', '2325', '3474', '6286', '2473', '8078', '5305')}
OUT = dict(delta=d, delta_pre2013=round(d * 1.25, 6), devs=DEVS, dmax_code=max(DEVS, key=lambda k: abs(DEVS[k])), n_devs=len(DEVS),
           desc_separate=dict(cash_max=max(cashd.values()), swap_absmax=max(abs(v) for k, v in DEVS.items() if k not in cashd)),
           counts=V3.status.value_counts().to_dict(), text_counts=V3.text_status.value_counts().to_dict(),
           v7x20_verified=sorted(set(V3.code[(V3.text_status == 'VERIFIED_TEXT') & V3.v7x_settled]) | {'2311', '2325', '6286', '3474'}),
           uopt6_verified=sorted(V3.code[(V3.text_status == 'VERIFIED_TEXT') & V3.uopt]), not_searched=int((V3.status == 'NOT_SEARCHED').sum()),
           events_v3_sha256=sha(HERE / 'events_v3.csv'))
(HERE / 'classify.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding='utf-8')
pd.set_option('display.width', 250)
print(V3[['code', 'status', 'settle', 'cash', 'ratio_text', 'ratio_market', 'R2', 'lc', 'succ_close_lt', 'dev_text_vs_lc', 'conflict', 'delta_evt', 'v7x_settled']].to_string())
print(json.dumps(OUT, ensure_ascii=False, indent=0))

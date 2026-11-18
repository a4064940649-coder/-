#!/usr/bin/env python3
"""S68 共用程式庫：把凍結的現行規則（STD SURGE、F1、E0、A10、W10_70、嚴成本、EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開）
原封不動套到「只上櫃」股池——股池上的樣本外（上櫃從沒用來定參數）。
鏈：s47lib → s20lib → s17lib → s10lib → … → cslib（全部不改檔案）。本檔只在執行期做四件「資料層」的事：
  (1) 各層快取目錄 → scratchpad/s68/cache（面板 npz、0050 開高低收、創新板快取從官方快取複製並驗 sha256，不呼叫 git）；
  (2) 月營收：tw_revenue.csv ＋ S16 的上櫃 2018-10 補洞檔（只補原檔沒有的代號，同 s35lib）→ scratchpad，stlib.REV_CSV 指過去；
  (3) 上櫃除權息：包住 cslib.exright_fix_chg（原函式先跑完，上市結果逐位不動），再依 OTC_EXR 處理上櫃格子：
        'none'＝不處理（＝S16／共用引擎現況：上櫃除權息日漲跌價差是缺值 → 引擎以收盤÷前收計，把股利當下跌，偏悲觀）；
        'main'＝(a) 上櫃、有成交、漲跌價差缺值、前一日有成交：本地 exdiv_5y.csv.gz（TPEX，2021-10-07 起）有同日同代號參考價、
                     且收盤落在參考價 ±10.1% 內 → 漲跌價差＝收盤−參考價；否則 → 漲跌價差＝0（當日含息報酬記 0）；
                 (b) 上櫃、漲跌價差＝0、收盤對前一成交收盤有變、前一日無成交、漲跌 ≤10.5% → 漲跌價差＝收盤−前一成交收盤（同上市規則 b）；
        'zero'＝(a) 一律記 0（不讀 exdiv_5y）＋(b)。
  (4) 引擎副本 simulate_s68：s10lib.simulate_s10 原始碼只加一個計數器 st['n_sized']（5% 成交值上限檢查的分母；S37／S51／S67 口徑），
      其餘逐位相同（驗收 A3）。
股池：CUR＝P.liq & 當日上市 & 非創新板（現行）；OTC＝P.liq & 當日上櫃（MK＝2）& 非創新板；ALL＝兩者聯集。P.liq＝60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10、當日有成交、非 91xx。
一律 python3 -I -B。"""
import sys, shutil, hashlib, inspect, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s68')
CACHE = TMP / 'cache'
OFFCACHE = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/eng/cache')
OFF_SHA = '434f64222b8fd1f10e1e25c9ac4553e7e6f6d11305896a3ceabb3b212650dfab'       # 官方面板快取（同 S55）
sys.path.insert(0, str(AG / 's47_allocation_family'))
import s47lib as L47                                       # noqa: E402  （import 時把三個正式開關設成 True）
from s47lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
_MODS = ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
         (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP'))
for _m, _a in _MODS:
    setattr(_m, _a, CACHE)
L47.CACHE = CACHE
assert cslib.EXRIGHT_FIX is True and cslib.DEADLINE_FIX is True and cslib.INS15_MASK is True, '正式開關必須全開'

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PATCH_CSV = AG / 's16_surge_otc' / 'rev_patch_otc_201810.csv'
PATCH_SHA = '011f0b0accedf45a26a0f6ceb048f5fb1d80ea671ced544b0bbd326e818d7ab5'
REV_ORIG = stlib.REV_CSV
REV_PATCHED = TMP / 'tw_revenue_patched.csv'
EXDIV_5Y = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/exdiv_5y.csv.gz')   # 上市＋上櫃除權息參考價（2021-10-07～2026-10-07），只讀
CELLS = ('CUR', 'OTC', 'ALL')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, maxdd=-0.32119049020243395, n_closed=381)   # 現行（deadline_engine_fix ALL）
OTC_EXR = 'main'                                           # 'none'／'main'／'zero'；load(mode) 設定
_ORIG_FIX = cslib.exright_fix_chg
_OTC_INFO = {}


def load_exdiv_tpex():
    e = pd.read_csv(EXDIV_5Y, dtype={'code': str})
    e = e[(e['market'] == 'TPEX') & e['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
    e['date'] = pd.to_datetime(e['date']); e['ref_price'] = pd.to_numeric(e['ref_price'], errors='coerce')
    return e.dropna(subset=['ref_price']).drop_duplicates(['date', 'code'], keep='last')


def otc_cells(IDX, COLS, C, CHG, MK):
    """上櫃要處理的格子（只看當日與過去）。回傳 dict(a＝漲跌價差缺值且前一日有成交, b＝漲跌價差 0、價變、前一日無成交、≤10.5%, prevC, REFX＝exdiv_5y 參考價)。CHG 用原始值。"""
    T, N = C.shape
    has = np.isfinite(C) & (C > 0)
    CF = pd.DataFrame(np.where(has, C, np.nan)).ffill().to_numpy()
    prevC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
    prev_tr = np.vstack([np.zeros((1, N), bool), has[:-1]])
    otc = MK == 2
    a = otc & has & np.isnan(CHG) & prev_tr & np.isfinite(prevC)
    with np.errstate(invalid='ignore', divide='ignore'):
        b = otc & has & (CHG == 0) & np.isfinite(prevC) & (np.abs(C - prevC) > 1e-9) & ~prev_tr & (np.abs(C / prevC - 1) <= 0.105)
    REFX = np.full((T, N), np.nan)
    ev = load_exdiv_tpex()
    ci = {c: j for j, c in enumerate(COLS)}; di = {d: i for i, d in enumerate(pd.DatetimeIndex(IDX))}
    e = ev[ev['code'].isin(ci) & ev['date'].isin(di)]
    REFX[e['date'].map(di).to_numpy(), e['code'].map(ci).to_numpy()] = e['ref_price'].to_numpy()
    return dict(a=a, b=b, prevC=prevC, REFX=REFX, n_ev=int(len(e)))


def _fix_wrapper(IDX, COLS, C, CHG, MK, ev):
    """先跑共用引擎的上市修正（原函式，不改），再處理上櫃。上市欄位／格子不受影響（上櫃格子 MK＝2，原函式不動它們）。"""
    raw = np.array(CHG, dtype=float, copy=True)
    out, info = _ORIG_FIX(IDX, COLS, C, CHG, MK, ev)
    _OTC_INFO.clear(); _OTC_INFO['mode'] = OTC_EXR
    if OTC_EXR == 'none':
        return out, info
    z = otc_cells(IDX, COLS, C, raw, MK)
    a, b, prevC, REFX = z['a'], z['b'], z['prevC'], z['REFX']
    with np.errstate(invalid='ignore', divide='ignore'):
        rr = C / REFX
        use_ref = a & np.isfinite(REFX) & (rr >= 0.899) & (rr <= 1.101) if OTC_EXR == 'main' else np.zeros_like(a)
    out[a & ~use_ref] = 0.0
    out[use_ref] = C[use_ref] - REFX[use_ref]
    out[b] = C[b] - prevC[b]
    _OTC_INFO.update(n_a=int(a.sum()), n_a_ref=int(use_ref.sum()), n_a_zero=int((a & ~use_ref).sum()), n_b=int(b.sum()), n_exdiv_tpex_in_panel=z['n_ev'])
    return out, info


cslib.exright_fix_chg = _fix_wrapper                       # 執行期包裝；cslib.py 檔案不動

# ---- 引擎副本：只加 n_sized 計數器
_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [("def simulate_s10(", "def simulate_s68("),
         ("""            if cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / o)""",
          """            st['n_sized'] = st.get('n_sized', 0) + 1
            if cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / o)""")]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s68 simulate_s68>', 'exec'), _G)
simulate_s68 = _G['simulate_s68']
SRC_SHA = dict(simulate_s10=hashlib.sha256(_SRC0.encode()).hexdigest(), simulate_s68=hashlib.sha256(_SRC.encode()).hexdigest())
assert SRC_SHA['simulate_s10'] == L47.SRC_SHA['simulate_s10']


def setup_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    assert sha(OFFCACHE / 'panel_cs.npz') == OFF_SHA, '官方面板快取不是預期版本'
    for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):
        if not (CACHE / f).exists():
            shutil.copy2(OFFCACHE / f, CACHE / f)
    return {f: sha(CACHE / f) for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz')}


def build_patched_rev():
    """tw_revenue.csv ＋ S16 補洞檔（只補 2018-10 原檔沒有的代號；同 s35lib.build_patched_rev）。回傳（新增列數, sha256）。"""
    TMP.mkdir(parents=True, exist_ok=True)
    assert sha(PATCH_CSV) == PATCH_SHA, '補洞檔 sha256 不符'
    r = pd.read_csv(REV_ORIG, dtype={'code': str})
    pt = pd.read_csv(PATCH_CSV, dtype={'code': str})
    have = set(r.loc[r['month'].astype(str).str[:7] == '2018-10', 'code'])
    pt = pt[~pt['code'].isin(have)][['month', 'market', 'code', 'name', 'revenue_k_twd']]
    pd.concat([r, pt.reindex(columns=r.columns)], ignore_index=True).to_csv(REV_PATCHED, index=False)
    return int(len(pt)), sha(REV_PATCHED)


def load(mode='main', patch=True):
    """mode＝上櫃除權息處理（見檔頭）。回傳 P（含 P.otc_exr、P.otc_info、P.n_patch_rows）。"""
    global OTC_EXR
    assert mode in ('none', 'main', 'zero')
    OTC_EXR = mode
    setup_cache()
    n_add = 0
    if patch:
        n_add, _ = build_patched_rev(); stlib.REV_CSV = str(REV_PATCHED)
    else:
        stlib.REV_CSV = REV_ORIG
    P = s20lib.load()
    assert P.exright_fix is True
    P.otc_exr = mode; P.otc_info = dict(_OTC_INFO); P.n_patch_rows = n_add
    P.e50_fix = s17lib.e50_fix_suspension(P)
    return P


def base(P):
    """回傳 (UU＝{'CUR','OTC','ALL'} 股池, RT＝STD SURGE)。"""
    INNOV, _ = s4lib.innov_panel(P)
    b = P.liq & ~INNOV
    UU = {'CUR': b & (P.MK == 1), 'OTC': b & (P.MK == 2), 'ALL': b & (P.MK > 0)}
    assert np.array_equal(UU['CUR'], stlib.universe(P, 'twse') & ~INNOV)
    F = s17lib.rev_family(P, defs=['STD'])
    return UU, dict(S=F['S']['STD'], av=F['av'])


def sim(P, S, U, cap=1e6, cost='strict', orig=False):
    """F1＋E0、嚴成本、W10_70、空位 0050、5% 成交值上限。orig=True 走原引擎 simulate_s10（驗收用）。"""
    S2, ekw = s10lib.variant_S(S, 'E0')
    kw = dict(cost=COSTS[cost], capital=cap, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False, **ekw)
    return (s10lib.simulate_s10 if orig else simulate_s68)(P, S2, U, **kw)


def monthly(nav, e50):
    """日淨值 → 日曆月報酬（同 s55lib.monthly）。DataFrame(strategy, b0050, excess)，PeriodIndex。"""
    nav = nav.dropna(); b = e50.reindex(nav.index)
    me = nav.groupby(nav.index.to_period('M')).tail(1)
    s = pd.concat([nav.iloc[:1], me]); bb = pd.concat([b.iloc[:1], b.reindex(me.index)])
    r = pd.DataFrame(dict(strategy=s.pct_change().iloc[1:].to_numpy(), b0050=bb.pct_change().iloc[1:].to_numpy()), index=me.index.to_period('M'))
    r['excess'] = r['strategy'] - r['b0050']
    return r


def nw_t(x, L=6):
    """Newey-West t（Bartlett 核、落後 6 個月；同 S55 run.py）。"""
    x = np.asarray(x, float); T = len(x); mu = x.mean(); e = x - mu; s = (e @ e) / T
    for i in range(1, L + 1):
        s += 2 * (1 - i / (L + 1)) * (e[i:] @ e[:-i]) / T
    return float(mu / math.sqrt(s / T))


def excess_stats(M):
    x = M['excess'].to_numpy(float); T = len(x); mu, sd = x.mean(), x.std(ddof=1); yrs = T / 12
    cg = lambda r: float(np.prod(1 + r) ** (1 / yrs) - 1)
    return dict(months=int(T), period=f'{M.index[0]}～{M.index[-1]}', cagr_strategy=cg(M['strategy']), cagr_0050=cg(M['b0050']),
                cagr_diff_pp=(cg(M['strategy']) - cg(M['b0050'])) * 100, ann_excess_pp=float(mu * 12 * 100), ann_te_pp=float(sd * math.sqrt(12) * 100),
                IR_ann=float(mu / sd * math.sqrt(12)), t_iid=float(mu / sd * math.sqrt(T)), t_nw6=nw_t(x, 6), hit_rate=float((x > 0).mean()),
                beta=float(np.polyfit(M['b0050'], M['strategy'], 1)[0]))


def capped(st):
    n = int(st.get('n_sized', 0))
    return dict(n_sig=int(st['n_sig']), n_buy=int(st['n_buy']), n_sized=n, n_capped=int(st['blk_cap']), capped_rate=(float(st['blk_cap'] / n) if n else None),
                blk_limup=int(st['blk_limup']), limup_rate=(float(st['blk_limup'] / st['n_sig']) if st['n_sig'] else None),
                blk_noopen=int(st['blk_noopen']), blk_min=int(st['blk_min']), blk_lot=int(st['blk_lot']), noslot=int(st['noslot']),
                sell_partial_cap=int(st['sell_partial_cap']), sell_defer_limdn=int(st['sell_defer_limdn']), n_delist=int(st['n_delist']))

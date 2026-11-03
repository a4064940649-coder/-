#!/usr/bin/env python3
"""o_lc_sector_strong_stocks 共用庫：PIT 產業、PIT 市值前 100 宇宙、個股與產業訊號、策略權重、向量化月頻引擎、統計。

唯讀 import o_0050_tilt_pit 的 tlib／tcore（V7 凍結面板＋下市股價、gapfix 52 週比、V4 已公告營收 y3_ab、PIT 股數），
python3 -I -B 執行，不寫 pycache、不改其他資料夾。

時點：訊號日 d＝營收期限日 D_m 之後第 j 個面板交易日（j＝0..12，tlib.jscan_days），d 收盤算訊號，下一交易日 e 開盤成交；
評估窗＝d 收盤 → 下一個訊號日收盤（0050 含息同窗）。
"""
import sys, json, math, hashlib, pickle
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
O8 = AG / 'o_0050_tilt_pit'
DATA = HERE / 'data'; OUT = HERE / 'out'; RAW = HERE / 'raw'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/lcss')
SCR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(O8))
import tlib as L      # noqa: E402
import tcore as TC    # noqa: E402
sys.path.pop(0)
IDX, COLS, N, CI = L.IDX, list(L.COLS), L.N, L.CI

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def canon(o): return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
def dumps(o): return json.dumps(o, ensure_ascii=False, indent=1, default=lambda z: float(z) if isinstance(z, (np.floating, np.integer)) else (bool(z) if isinstance(z, np.bool_) else (z.tolist() if isinstance(z, np.ndarray) else str(z))))

# ---------------- 股數與市值 ----------------
PIT_START = pd.Timestamp('2013-01-02')     # shares_pit 起點（2014-01 起通過 o8 閘門）
GATE_START = pd.Timestamp('2014-01-01')    # D1：PIT 市值可用段
FIX4763 = dict(a0='2019-09-04', a1='2023-07-27', div_a=12.65, b1='2025-06-30', div_b=10.0)   # 同 o_capmatched_pit_close/pcore.py
FIX2317 = dict(pre_until='2018-10-25', eff='2018-10-26', pre_val_date='2018-06-29')

def _shares():
    SP = pd.read_parquet(O8 / 'data' / 'shares_pit.parquet').reindex(columns=COLS)
    SP = SP.reindex(IDX.union(SP.index)).sort_index().ffill(limit=10).reindex(IDX)
    ix = SP.index
    a = (ix >= pd.Timestamp(FIX4763['a0'])) & (ix < pd.Timestamp(FIX4763['a1']))
    b = (ix >= pd.Timestamp(FIX4763['a1'])) & (ix < pd.Timestamp(FIX4763['b1']))
    SP.loc[a, '4763'] = SP.loc[a, '4763'] / FIX4763['div_a']
    SP.loc[b, '4763'] = SP.loc[b, '4763'] / FIX4763['div_b']
    pre_val = float(SP.loc[:pd.Timestamp(FIX2317['pre_val_date']), '2317'].dropna().iloc[-1])
    m = (ix > pd.Timestamp(FIX2317['pre_val_date'])) & (ix <= pd.Timestamp(FIX2317['pre_until']))
    SP.loc[m, '2317'] = pre_val
    SE = pd.read_parquet(O8 / 'raw' / 'agents' / 'capmatched_shares_est.parquet').reindex(index=IDX, columns=COLS)
    early = np.asarray(IDX < PIT_START)
    S = np.where(early[:, None], SE.to_numpy(), SP.to_numpy())
    Lt = pd.read_parquet(AG / 'o_v7_listing_exec_engine' / 'data' / 'listed_twse.parquet').reindex(columns=COLS).astype(float)
    Lt = Lt.reindex(IDX.union(Lt.index)).sort_index().ffill().reindex(IDX).fillna(1.0).to_numpy() > 0.5
    S = np.where(Lt, S, np.nan)
    return S
SH = _shares()
DRMASK = np.array([c.startswith('91') for c in COLS])
def mcap_row(t): return SH[t] * L.CFF[t]

# ---------------- PIT 產業 ----------------
REG_2007 = pd.Timestamp('2007-07-02')    # 電子細分 8 類、化學／生技醫療分立、油電燃氣（研究員依證交所沿革；2023 年報導稱「16 年來首次調整」佐證 2007 年那次）
REG_2023 = pd.Timestamp('2023-07-03')    # 新增綠能環保、數位雲端、運動休閒、居家生活；觀光事業→觀光餐旅（GVM、商周報導核實）
NEW4 = {'綠能環保', '數位雲端', '運動休閒', '居家生活'}
ELEC8 = {'半導體業', '電腦及週邊設備業', '光電業', '通信網路業', '電子零組件業', '電子通路業', '資訊服務業', '其他電子業'}
PREFIX_IND = {'11': '水泥工業', '12': '食品工業', '13': '塑膠工業', '14': '紡織纖維', '15': '電機機械', '16': '電器電纜',
              '17': '化學工業', '18': '玻璃陶瓷', '19': '造紙工業', '20': '鋼鐵工業', '21': '橡膠工業', '22': '汽車工業',
              '25': '建材營造業', '26': '航運業', '27': '觀光餐旅', '28': '金融保險業', '29': '貿易百貨業', '99': '其他業'}
MANUAL_DELIST = {'1704': '化學工業', '2311': '半導體業', '2325': '半導體業', '2448': '光電業', '2456': '電子零組件業',
                 '2823': '金融保險業', '2833': '金融保險業', '2847': '金融保險業', '3474': '半導體業', '5264': '其他電子業',
                 '6452': '生技醫療業'}
SNAP_FILE = AG / 'o_v7_listing_exec_engine' / 'raw' / 'isin' / 'snap' / 'isin_02_64993c2e1699ae56b9752b47e87f90721be5fd50.txt'   # ISIN 2023/01/16
SL_FILE = O8 / 'raw' / 'screener' / 'stock_list.csv'

def _ind_tables():
    sl = pd.read_csv(SL_FILE, dtype=str).drop_duplicates('code').set_index('code')['industry']
    d = json.loads(SNAP_FILE.read_text(encoding='utf-8')); snap = {}
    for g in d['list']:
        for x in g['list']:
            if x.get('market') == '上市' and x.get('industry'): snap[x['code']] = x['industry'].replace('觀光事業', '觀光餐旅')
    cur, mid, src = [], [], []
    for c in COLS:
        v = sl.get(c)
        if isinstance(v, str) and v: s = 'stock_list'
        elif c in MANUAL_DELIST: v = MANUAL_DELIST[c]; s = 'manual_delist'
        elif c[:2] in PREFIX_IND and not c.startswith('91'): v = PREFIX_IND[c[:2]]; s = 'prefix'
        else: v = '未分類'; s = 'none'
        cur.append(v); src.append(s)
        if c in snap: mid.append(snap[c])
        elif v in NEW4: mid.append('其他業')
        else: mid.append(v)
    early = []
    for v in mid:
        if v in ELEC8: early.append('電子工業')
        elif v in ('化學工業', '生技醫療業'): early.append('化學生技醫療業')
        elif v == '油電燃氣業': early.append('其他業')
        else: early.append(v)
    return np.array(cur, dtype=object), np.array(mid, dtype=object), np.array(early, dtype=object), np.array(src, dtype=object), len(snap)
IND_CUR, IND_MID, IND_EARLY, IND_SRC, N_SNAP = _ind_tables()
def ind_at(d):
    d = pd.Timestamp(d)
    if d >= REG_2023: return IND_CUR
    if d >= REG_2007: return IND_MID
    return IND_EARLY
FINMASK = TC.FINMASK | (IND_CUR == '金融保險業')

# ---------------- 題 1 產業排名（切換後版本）----------------
# MODE 'COMP'：題 1 事前預設版（COMP、k=5、無產業緩衝，本題自行實作，用本題 PIT 產業表）
# MODE 'T1'  ：題 1 凍結選定版（H52、k=3、產業緩衝 k+2；產業排名與產業表直接取自題 1 程式 o_lc_sector_rotation_pit/lclib.py，
#              由 s0b_t1_adapter.py 預先算好存 data/t1_sector_order.json；個股宇宙仍是本題的 PIT 前 100）
MODE = {'m': 'COMP'}
T1DIR = AG / 'o_lc_sector_rotation_pit'
_T1 = {}
def t1_module():
    if 'm' not in _T1:
        import importlib.util
        sp = importlib.util.spec_from_file_location('t1_lclib_ro', T1DIR / 'lclib.py'); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
        _T1['m'] = m
    return _T1['m']
def t1_order(d):
    if 'ord' not in _T1:
        _T1['ord'] = json.loads((DATA / 't1_sector_order.json').read_text(encoding='utf-8'))['order']
    return _T1['ord'][str(pd.Timestamp(d).date())]
def set_mode(m):
    assert m in ('COMP', 'T1'); MODE['m'] = m; _CS.clear()
def mode_params():
    if MODE['m'] == 'COMP': return dict(k=5, sbuf=0)
    if 'meta' not in _T1:
        z = json.loads((DATA / 't1_sector_order.json').read_text(encoding='utf-8')); _T1['meta'] = dict(k=z['k'], sbuf=z['params']['buf'])
    return dict(_T1['meta'])

# ---------------- 訊號 ----------------
GOODI = np.where(L.GOOD)[0]
GPOS = np.full(len(IDX), -1); GPOS[GOODI] = np.arange(len(GOODI))
FIRST = np.where(L.TRADED.any(0), np.argmax(L.TRADED, axis=0), 10 ** 9)
CUMG = TC.CUMG          # CUMG[b+1]-CUMG[a+1] = Σ_{a+1..b} log GX
S52 = TC.S52
def _nh():
    s = np.nan_to_num(S52[GOODI], nan=0.0) >= 0.99999
    r = pd.DataFrame(s.astype(np.int8)).rolling(21, min_periods=1).max().to_numpy() > 0
    out = np.zeros((len(IDX), N), bool); out[GOODI] = r
    return out
NH21 = _nh()

def gback(t, k):
    """t（GOOD 日）往前第 k 個 GOOD 日的面板索引。"""
    p = GPOS[t]; assert p >= 0
    return GOODI[max(p - k, 0)]
def cumret(a, b):
    """a 收盤 → b 收盤（含息）。"""
    return np.exp(CUMG[b + 1] - CUMG[a + 1]) - 1
def to_good(d):
    t = IDX.get_loc(pd.Timestamp(d))
    while not L.GOOD[t]: t -= 1
    return t

def pct(x):
    s = pd.Series(x); return s.rank(pct=True).to_numpy()

SECT_MIN = 2
Y3_CLIP = (-80.0, 300.0)
def cross_section(d):
    """訊號日 d 收盤：宇宙（PIT 市值前 100、非 DR、近 6 日有成交、當日有收盤）、個股三分數、產業四訊號與 COMP。"""
    t = IDX.get_loc(pd.Timestamp(d)); tg = to_good(d)
    mc = mcap_row(t)
    alive = L.TRADED[max(t - 5, 0):t + 1].any(axis=0) & np.isfinite(L.CFF[t])
    el = np.where(~DRMASK & np.isfinite(mc) & (mc > 0) & alive, mc, -np.inf)
    U = np.argsort(-el, kind='stable')[:100]; U = U[np.isfinite(el[U])]
    U = U[np.argsort(-mc[U], kind='stable')]
    mcU = mc[U]
    ind = (ind_at(d) if MODE['m'] == 'COMP' else t1_module().ind_at(d))[U]
    fin = FINMASK[U]
    s52 = S52[tg, U]
    a, b = gback(tg, 126), gback(tg, 21)
    rs = np.where(FIRST[U] <= a, cumret(a, b)[U], np.nan)
    y3 = TC.y3_ab(pd.Timestamp(d))[U].astype(float)
    y3 = np.where(fin, np.nan, y3)
    P = np.vstack([pct(s52), pct(rs), pct(y3)])
    nok = np.isfinite(P).sum(0)
    with np.errstate(invalid='ignore'):
        Psum = np.nansum(P, axis=0); score = np.where(nok >= 2, Psum / np.maximum(nok, 1), np.nan)
    # 產業訊號
    y3w = np.clip(y3, Y3_CLIP[0], Y3_CLIP[1])     # 產業聚合用：年增率縮尾（2013 IFRS 斷點與極端值）
    rh = {}
    for h in (63, 126, 252):
        a2 = gback(tg, h); rh[h] = np.where(FIRST[U] <= a2, cumret(a2, b)[U], np.nan)
    nh = NH21[tg, U]
    secs = {}
    for s in sorted(set(ind)):
        m = np.where(ind == s)[0]
        if len(m) < SECT_MIN: continue
        w = mcU[m]
        def cw(x):
            ok = np.isfinite(x[m]); return float((w[ok] * x[m][ok]).sum() / w[ok].sum()) if ok.sum() and w[ok].sum() > 0 else np.nan
        secs[s] = dict(members=m, cap=float(w.sum()), REV=cw(y3w), RS63=cw(rh[63]), RS126=cw(rh[126]), RS252=cw(rh[252]),
                       H52=cw(s52), NH=float(nh[m].mean()))
    names = list(secs)
    if names:
        T = pd.DataFrame({k: [secs[s][k] for s in names] for k in ('REV', 'RS63', 'RS126', 'RS252', 'H52', 'NH', 'cap')}, index=names)
        rr = T[['RS63', 'RS126', 'RS252']].rank(pct=True).mean(axis=1, skipna=True)
        R = pd.DataFrame({'REV': T['REV'].rank(pct=True), 'RS': rr.rank(pct=True), 'H52': T['H52'].rank(pct=True), 'NH': T['NH'].rank(pct=True)})
        T['COMP'] = R.mean(axis=1, skipna=True)
        for c in R.columns: T['r_' + c] = R[c]
        T = T.sort_values(['COMP', 'cap'], ascending=[False, False])
    else:
        T = pd.DataFrame()
    if MODE['m'] == 'T1':
        od = [x for x in t1_order(d) if x in secs]           # 題 1 排序（H52 由高到低、同分比 B 權重），只留本題宇宙內 ≥2 家者
        T = T.reindex(od) if len(T) else T
    return dict(d=pd.Timestamp(d), t=t, U=U, mc=mcU, ind=ind, fin=fin, s52=s52, rs=rs, y3=y3, score=score, P=P, secs=secs, ST=T)

_CS = {}
def cs_cached(d):
    d = pd.Timestamp(d)
    if d in _CS: return _CS[d]
    f = SCR / ('cs' if MODE['m'] == 'COMP' else 'cs_T1') / f'{d.date()}.pkl'
    if f.exists():
        x = pickle.loads(f.read_bytes())
    else:
        x = cross_section(d); f.parent.mkdir(exist_ok=True); f.write_bytes(pickle.dumps(x))
    _CS[d] = x
    return x

def stock_order(cs, idx, score=None):
    """idx（宇宙內位置）依分數由高到低；同分先比營收、再比代碼；分數缺值放最後。"""
    sc = cs['score'] if score is None else score
    keys = []
    for i in idx:
        s = sc[i]; y = cs['y3'][i]
        keys.append((0 if np.isfinite(s) else 1, -(s if np.isfinite(s) else 0), -(y if np.isfinite(y) else -1e9), COLS[cs['U'][i]]))
    o = sorted(range(len(idx)), key=lambda q: keys[q])
    return [idx[q] for q in o]

# ---------------- 權重 ----------------
def cap_weights(w, cap):
    """單檔上限：超過部分依比例分給未觸頂者；不可行（K×cap<1）時有效上限＝1/K（等權）。"""
    w = np.asarray(w, float); w = w / w.sum(); K = len(w)
    c = max(cap, 1.0 / K)
    for _ in range(100):
        over = w > c + 1e-12
        if not over.any(): break
        ex = (w[over] - c).sum(); w[over] = c
        free = w < c - 1e-12
        if not free.any(): break
        w[free] += ex * w[free] / w[free].sum()
    return w / w.sum()

def cap_weights_sector(w, cap, secs):
    """單檔上限：超過部分先分給同產業未觸頂的選中股（依比例），同產業都觸頂才分給其他未觸頂者。"""
    w = np.asarray(w, float); w = w / w.sum(); K = len(w); c = max(cap, 1.0 / K)
    for _ in range(200):
        over = np.where(w > c + 1e-12)[0]
        if len(over) == 0: break
        for i in over:
            ex = w[i] - c; w[i] = c
            same = np.where((secs == secs[i]) & (w < c - 1e-12))[0]
            tgt = same if len(same) else np.where(w < c - 1e-12)[0]
            if len(tgt) == 0: break
            w[tgt] += ex * w[tgt] / w[tgt].sum()
    return w / w.sum()

SAT_W = 0.30
class Strat:
    """kind：B（宇宙市值加權）｜IND_ONLY（前 k 產業全體成員市值加權）｜IND（前 k 產業內各取分數前 n）｜CTRL（宇宙內分數前 k×n）。
    carrier：PURE｜SAT（0.7×B＋0.3×PURE）。buf：緩衝帶寬度（IND：產業內名次 ≤ n＋buf 續抱；CTRL：名次 ≤ k×(n＋buf)）。
    sector_rank_fn(cs) → 產業名單（由強到弱），預設真 COMP；score_fn(cs) → 分數向量，預設真分數（安慰劑用）。"""
    def __init__(self, kind, n=2, wt='cap', cap=0.15, carrier='SAT', k=None, buf=1, sector_rank_fn=None, score_fn=None, overflow='prop', sbuf=None):
        mp = mode_params()
        k = mp['k'] if k is None else k; sbuf = mp['sbuf'] if sbuf is None else sbuf
        self.kind, self.n, self.wt, self.cap, self.carrier, self.k, self.buf, self.sbuf = kind, n, wt, cap, carrier, k, buf, sbuf
        self.held_secs = []
        self.srf, self.sf, self.overflow = sector_rank_fn, score_fn, overflow
        self.held = set(); self.last = None
    def top_sectors(self, cs):
        order = self.srf(cs) if self.srf is not None else (list(cs['ST'].index) if len(cs['ST']) else [])
        if self.sbuf <= 0:
            top = order[:self.k]
        else:
            rank = {s: r + 1 for r, s in enumerate(order)}
            top = [s for s in order if s in self.held_secs and rank[s] <= self.k + self.sbuf]
            for s in order:
                if len(top) >= self.k: break
                if s not in top: top.append(s)
            top = top[:self.k]
        self.held_secs = list(top)
        return top
    def sleeve(self, cs):
        U = cs['U']; mc = cs['mc']
        if self.kind == 'IND_ONLY':
            secs = self.top_sectors(cs)
            m = np.concatenate([cs['secs'][s]['members'] for s in secs]) if secs else np.array([], int)
            return m, mc[m] / mc[m].sum(), {'sectors': secs}
        sc = cs['score'] if self.sf is None else self.sf(cs)
        sel = []
        if self.kind == 'IND':
            secs = self.top_sectors(cs)
            for s in secs:
                mem = [i for i in cs['secs'][s]['members'] if np.isfinite(sc[i])]
                order = stock_order(cs, mem, sc)
                rank = {i: r + 1 for r, i in enumerate(order)}
                keep = [i for i in order if U[i] in self.held and rank[i] <= self.n + self.buf][:self.n]
                for i in order:
                    if len(keep) >= self.n: break
                    if i not in keep: keep.append(i)
                sel += keep
            info = {'sectors': secs}
        elif self.kind == 'CTRL':
            K = self.k * self.n
            mem = [i for i in range(len(U)) if np.isfinite(sc[i])]
            order = stock_order(cs, mem, sc)
            rank = {i: r + 1 for r, i in enumerate(order)}
            keep = [i for i in order if U[i] in self.held and rank[i] <= self.k * (self.n + self.buf)][:K]
            for i in order:
                if len(keep) >= K: break
                if i not in keep: keep.append(i)
            sel = keep; info = {}
        sel = np.array(sel, int)
        if len(sel) == 0: return sel, np.array([]), info
        w = mc[sel] / mc[sel].sum() if self.wt == 'cap' else np.full(len(sel), 1.0 / len(sel))
        if self.overflow == 'prop':
            w = cap_weights(w, self.cap)
        elif self.overflow == 'sector':
            w = cap_weights_sector(w, self.cap, cs['ind'][sel])
        elif self.overflow == 'skeleton':
            c = max(self.cap, 0.0); w = np.minimum(w, c)      # 超過上限部分交給骨架 B（info['to_B']）
            info = dict(info); info['to_B'] = float(1 - w.sum())
        self.held = set(U[sel].tolist())
        return sel, w, info
    def step(self, cs):
        U = cs['U']; mc = cs['mc']
        wB = mc / mc.sum()
        if self.kind == 'B':
            return U.copy(), wB, {}
        sel, w, info = self.sleeve(cs)
        toB = info.get('to_B', 0.0) if isinstance(info, dict) else 0.0
        if self.carrier == 'PURE' and toB <= 1e-12:
            return U[sel], w, info
        sw = SAT_W if self.carrier == 'SAT' else 1.0
        tot = (1 - sw) * wB + sw * toB * wB
        np.add.at(tot, sel, sw * w)
        return U.copy(), tot / tot.sum(), info

# ---------------- 日期 ----------------
def sig_days(j, start, end, clip=None):
    """j：0..12（jscan）或 'off'（官方日）。回傳 [訊號日..., 終點]；終點＝下一個訊號日，或 clip（該日或之前最後一個面板日）。"""
    ds = L.official_days('2005-01-01') if j == 'off' else L.jscan_days(int(j), '2005-01-01')
    a = [d for d in ds if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    nxt = [d for d in ds if d > pd.Timestamp(end)]
    term = nxt[0] if nxt else IDX[-1]
    if clip is not None:
        c = IDX[IDX <= pd.Timestamp(clip)][-1]
        if c < term: term = c
    a = [d for d in a if d < term]
    # PREREG_AMEND v1（資料錯誤）：成交日 d+1 不是 GOOD 日（2018-01-13 週六只有 46 檔有值的殘列）→ 該訊號日不換股（併入下一窗）
    a = [d for d in a if L.GOOD[IDX.get_loc(d) + 1]]
    return a + [term]

PERIODS = {
    'SEL': ('2014-01-01', '2016-12-31', '2016-12-30'),
    'VAL': ('2017-01-01', '2026-10-31', None),
    'DESC': ('2006-04-01', '2013-12-31', '2013-12-31'),
    'FULL': ('2006-04-01', '2026-10-31', None),
    'PIT': ('2014-01-01', '2026-10-31', None),
}
def days_for(j, per):
    a, b, c = PERIODS[per]; return sig_days(j, a, b, c)

# ---------------- 向量化引擎 ----------------
FEE = 0.001425 * 0.6; TAX = 0.003
def rates(cost):
    if cost == 'normal': return FEE + 0.001, FEE + TAX + 0.001
    if cost == 'stress': return FEE + 0.006, FEE + TAX + 0.006
    if cost == 'zero': return 0.0, 0.0
    raise ValueError(cost)
GX = TC.GX
def f_c2o(t, e): return TC.factor_close_to_open(t, e)
def f_o2c(e, t2): return TC.factor_open_to_close(e, t2)
def f_c2c(a, b): return TC.factor_close_to_close(a, b)

def vrun(days, W, cost='normal'):
    """W：list（長度＝窗數），每元素為 P×N 權重矩陣或 (idx, w)。回傳 (R: P×T 窗報酬, TO: P×T 單邊周轉)。
    與 tcore.run 同口徑：d 收盤→e 開盤舊持股承擔；e 開盤換股（成本＝buy·Σ(tgt−h)+ ＋ sell·Σ(h−tgt)+，T＝Vt−c 兩次固定點）；
    e 開盤不能交易的持股凍結；已下市持股扣賣出成本轉現金。"""
    buy, sell = rates(cost)
    ti = [IDX.get_loc(d) for d in days]
    first = W[0]
    P = 1 if isinstance(first, tuple) else first.shape[0]
    H = np.zeros((P, N)); cash = np.ones(P)
    R = np.zeros((P, len(ti) - 1)); TO = np.zeros((P, len(ti) - 1))
    for k in range(len(ti) - 1):
        t, t2 = ti[k], ti[k + 1]; e = t + 1
        fo = f_c2o(t, e); tr = np.isfinite(fo)
        f2 = np.nan_to_num(f_o2c(e, t2), nan=1.0); fcc = f_c2c(t, t2)
        dead = L.LAST_TRADE < e
        V0 = H.sum(1) + cash; H = H / V0[:, None]; cash = cash / V0
        cash = cash + H[:, dead].sum(1) * (1 - sell); H[:, dead] = 0.0
        FRZ = (H > 0) & ~tr[None, :]
        H = np.where(tr[None, :], H * np.nan_to_num(fo, nan=1.0)[None, :], H)
        Hz = np.where(FRZ, H, 0.0); Htr = H - Hz
        Vt = Htr.sum(1) + cash
        wk = W[k]
        if isinstance(wk, tuple):
            WV = np.zeros((1, N)); np.add.at(WV[0], wk[0], wk[1])
        else:
            WV = wk.copy()
        WV[FRZ] = 0.0; WV[:, ~tr] = 0.0
        s = WV.sum(1, keepdims=True); WV = np.where(s > 0, WV / np.where(s > 0, s, 1), 0.0)
        T = Vt.copy()
        for _ in range(3):
            tgt = WV * T[:, None]
            c = buy * np.clip(tgt - Htr, 0, None).sum(1) + sell * np.clip(Htr - tgt, 0, None).sum(1)
            T = Vt - c
        tgt = WV * T[:, None]
        TO[:, k] = np.clip(Htr - tgt, 0, None).sum(1) / np.maximum(Vt + Hz.sum(1), 1e-12)
        H = tgt * f2[None, :] + Hz * fcc[None, :]; cash = np.zeros(P)
        R[:, k] = H.sum(1) - 1.0
    return R, TO

def bench_win(days):
    return TC.bench_win(days).to_numpy()

def stock_win_ret(days):
    """每窗每檔 d 收盤→下一訊號日收盤的含息報酬（N 向量列表）。"""
    ti = [IDX.get_loc(d) for d in days]
    return [cumret(a, b) for a, b in zip(ti[:-1], ti[1:])]

# ---------------- 統計 ----------------
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')
def stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return dict(n=len(x), mean_pp=float('nan'), t=float('nan'), TE=float('nan'), IR=float('nan'))
    te = float(x.std(ddof=1) * math.sqrt(12))
    return dict(n=int(len(x)), mean_pp=float(x.mean() * 100), t=tstat(x), TE=te * 100, IR=float(x.mean() * 12 / te) if te > 0 else float('nan'))
def cagr(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    return float(np.prod(1 + r) ** (12 / len(r)) - 1) if len(r) else float('nan')
def mdd(r):
    v = np.cumprod(1 + np.asarray(r, float)); return float((v / np.maximum.accumulate(v) - 1).min()) if len(v) else float('nan')
def drop_best_years_t(x, days, nb=3, rb=None):
    """拿掉超額（幾何：組合年報酬−0050 年報酬）最好的 nb 個日曆年後的月 t。"""
    s = pd.Series(np.asarray(x, float), index=pd.DatetimeIndex(days[:len(x)]))
    if rb is None: yr = s.groupby(s.index.year).sum()
    else:
        b = pd.Series(np.asarray(rb, float), index=s.index); rs = s + b
        yr = rs.groupby(rs.index.year).apply(lambda g: np.prod(1 + g)) - b.groupby(b.index.year).apply(lambda g: np.prod(1 + g))
    best = yr.sort_values(ascending=False).index[:nb]
    return tstat(s[~s.index.year.isin(best)].to_numpy()), [int(y) for y in best]

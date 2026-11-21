#!/usr/bin/env python3
"""S87 共用程式庫：講義單元 02「風格 ETF 順風檢查」用真實台股 ETF 做成規則，放在現行策略的 0050 部位（底倉＋空位）。
鏈：s20lib → s17lib → s10lib → cslib（全部不改）；正式開關 EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開；不開 REDUCE_FIX。
引擎把「底倉＋空位」當成同一個 0050 部位，價格來自 P.E50（含息收盤）、P.E50O（含息開盤）。本檔不改引擎：
    模擬時用 P 的淺複本，把 E50／E50O 換成「載具指數」（每月依順風檢查決定放哪幾檔 ETF；ETF 之間換倉的成本直接扣在載具指數上）。
    E50O_raw／E50C_raw（只用來算 0050 進出的半跳動點成本）不換。換手成本另加（見 COST）。
順風檢查（講義）：ETF 自己過去 L 個月含息報酬 R_etf、同期 0050 含息報酬 R_0050；R_etf > 0 且 R_etf > R_0050 ＝順風。
決定日＝每月最後一個交易日收盤（只用當天以前的資料），下一個交易日開盤換倉。
格子（16 格）：池 {S＝風格, SX＝風格＋產業}× 回看 {6, 12 個月}× 規則 {TOP1＝順風中 R 最高一檔, EW＝所有順風等權}× 範圍 {100, 50（一半留 0050）}。
沒有任何順風 ETF → 全放 0050（＝現行）。一律 python3 -I -B。"""
import sys, glob, copy, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s87')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False and cslib.EXRIGHT_FIX is False, '共用引擎預設應全關（本題自己開）'
cslib.EXRIGHT_FIX = True
cslib.DEADLINE_FIX = True
cslib.INS15_MASK = True
assert not getattr(cslib, 'REDUCE_FIX', False), '本題不開 REDUCE_FIX'

TWT49U = '/home/user/-/research/tw_stock_selection/raw/twt49u/twse-exright'
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH
fresh_from = s20lib.fresh_from
DROP = ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq')

# ---- ETF 分類（只看名稱，在算任何報酬之前寫死；台股、上市、非槓桿反向、非債券、非海外）
STYLE = {  # 風格：規模、股息／價值、低波、多因子
    '0051': '中型', '00733': '中小＋動能', '00894': '高價小資', '00921': '龍頭等權',
    '0056': '高股息', '00701': '低波股利', '00713': '高息低波', '00730': '優質高息', '00731': '高息低波', '00742': '內需收益', '00878': '高股息',
    '00900': '高股息', '00907': '優息存股', '00915': '高股息', '00918': '高股息', '00919': '高股息', '00929': '科技優息', '00930': '低碳高息',
    '00932': '高息等權', '00934': '成長高股息', '00936': '高息中小', '00939': '高息動能', '00940': '價值高息', '00943': '電子高息等權',
    '00944': '動能高息', '00946': '科技高息成長', '00961': '永續高息', '00962': 'AI 優息動能',
    '00690': '藍籌', '00905': '多因子', '00912': '多因子', '00938': '多因子', '009803': '市值動能', '009808': '多因子',
}
SECTOR = {  # 產業／主題
    '0052': '科技', '0053': '科技', '0055': '金融', '0059': '金融', '00881': '科技龍頭', '00891': '半導體', '00892': '半導體', '00904': '半導體',
    '00913': '晶圓製造', '00927': '半導體收益', '00935': '新科技', '00941': '上游半導體', '00947': 'IC 設計', '00917': '金融', '00728': '工業',
    '00896': '綠能電動車', '00901': '智能車供應鏈', '00920': '綠色電力', '00952': 'AI',
}
assert not set(STYLE) & set(SECTOR)
POOLS = {'S': tuple(STYLE), 'SX': tuple(STYLE) + tuple(SECTOR)}
LOOKS = (6, 12)
RULES = ('TOP1', 'EW')
SCOPES = (100, 50)
CELLS = [dict(id=f'{p}_L{l}_{r}_W{w}', pool=p, look=l, rule=r, scope=w) for p in POOLS for l in LOOKS for r in RULES for w in SCOPES]
assert len(CELLS) == 16
LIQ_MIN, LIQ_WIN, LIQ_MINOBS = 2e7, 60, 40                 # 決定日前 60 個交易日平均成交值 ≥ 2,000 萬，且至少 40 天有資料
FEE, TAX_ETF = stlib.FEE, stlib.TAX_ETF
XSL = 0.001                                                # 非 0050 的 ETF 換倉每邊另加 0.1% 滑價（0050 照引擎口徑不加）


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True
    return P


def base(P):
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    return U, RT


def slim(P):
    for a in DROP:
        if hasattr(P, a):
            delattr(P, a)


# ---------------------------------------------------------------- ETF 含息報酬（同 S66 的做法，另算開盤）
def etf_panel(P):
    """回傳 dict：codes、GO（隔夜：開盤 ÷ 參考價；T×M）、GI（盤中：收盤 ÷ 開盤）、TR（含息還原收盤指數，上市前 NaN）、AMT（成交值）、
    OPEN（原始開盤）、HAS（當天有收盤價）。對齊 P.IDX。參考價＝前收 − 息值（twt49u）；分割日（|收盤÷前收−1| > 30%）參考價＝收盤 − 官方漲跌。"""
    d = pd.read_parquet(TMP / 'etf_daily.parquet'); d['date'] = pd.to_datetime(d['date'])
    codes = ['0050'] + list(STYLE) + list(SECTOR)
    d = d[d.stock_id.isin(codes)]
    e = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in sorted(glob.glob(TWT49U + '/*/*.csv.gz'))], ignore_index=True)
    e['date'] = pd.to_datetime(e['date']); e = e[e.code.isin(codes)]
    idx = pd.DatetimeIndex(P.IDX); T = len(idx); M = len(codes)
    GO = np.ones((T, M)); GI = np.ones((T, M)); TR = np.full((T, M), np.nan); AMT = np.full((T, M), np.nan); OPEN = np.full((T, M), np.nan)
    HAS = np.zeros((T, M), bool); info = {}
    for m, c in enumerate(codes):
        s = d[d.stock_id == c].sort_values('date').set_index('date')
        s = s[~s.index.duplicated()]
        if not len(s):
            info[c] = dict(n=0); continue
        full = s.reindex(idx[(idx >= s.index[0]) & (idx <= s.index[-1])])
        C = full['close'].ffill(); O = full['open']; chg = full['change']
        ev = e[(e.code == c) & (e.date > C.index[0])]
        D = pd.Series(0.0, index=C.index); unm = 0
        for t, v in zip(ev.date, ev.value):
            i = C.index.searchsorted(t)
            if i >= len(C):
                unm += 1; continue
            D.iloc[i] += float(v)
        pc = C.shift(1); ref = pc - D; splits = []
        raw = C / pc - 1
        for t in raw.index[raw.abs() > 0.30]:
            ref[t] = C[t] - chg[t]; splits.append(str(t.date()))
        r = (C / ref - 1).fillna(0.0)
        traded = full['close'].notna() & O.notna() & (O > 0)
        go = (O / ref).where(traded & ref.notna() & (ref > 0), 1.0)
        gi = (1 + r) / go
        pos = idx.get_indexer(C.index)
        GO[pos, m] = go.to_numpy(); GI[pos, m] = gi.to_numpy(); TR[pos, m] = (1 + r).cumprod().to_numpy()
        AMT[pos, m] = full['amount'].to_numpy(); OPEN[pos, m] = O.to_numpy(); HAS[pos, m] = full['close'].notna().to_numpy()
        info[c] = dict(n=int(len(s)), first=str(s.index[0].date()), last=str(s.index[-1].date()), n_div=int((D > 0).sum()), div_sum=float(D.sum()),
                       splits=splits, unmapped=unm)
    return dict(codes=codes, GO=GO, GI=GI, TR=TR, AMT=AMT, OPEN=OPEN, HAS=HAS, info=info)


def month_ends(P):
    idx = pd.DatetimeIndex(P.IDX)
    return pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period('M')).max().to_numpy()


def decisions(P, E, cell, first_dec_t=0):
    """回傳 {換倉列 t（決定日的下一個交易日）: {代號: 權重}}（權重含 0050，合計 1）與每月紀錄。只用決定日（含）以前的資料。"""
    codes = E['codes']; ci = {c: i for i, c in enumerate(codes)}
    pool = [c for c in POOLS[cell['pool']]]
    me = month_ends(P); L = cell['look']; out = {}; rec = []
    for q in range(L, len(me)):
        d = int(me[q]); d0 = int(me[q - L])
        if d + 1 >= P.T or d < first_dec_t:
            continue
        r50 = E['TR'][d, 0] / E['TR'][d0, 0] - 1
        tw = []
        for c in pool:
            m = ci[c]
            if not (E['HAS'][d, m] and np.isfinite(E['TR'][d0, m])):
                continue
            a = E['AMT'][max(d - LIQ_WIN + 1, 0):d + 1, m]
            if np.isfinite(a).sum() < LIQ_MINOBS or np.nanmean(a) < LIQ_MIN:
                continue
            r = E['TR'][d, m] / E['TR'][d0, m] - 1
            if r > 0 and r > r50:
                tw.append((float(r), c))
        tw.sort(reverse=True)
        if not tw:
            w = {}
        elif cell['rule'] == 'TOP1':
            w = {tw[0][1]: 1.0}
        else:
            w = {c: 1.0 / len(tw) for _, c in tw}
        sc = cell['scope'] / 100.0
        tgt = {c: v * sc for c, v in w.items()}
        tgt['0050'] = 1.0 - sum(tgt.values())
        out[d + 1] = tgt
        rec.append(dict(dec=str(P.IDX[d].date()), r50=float(r50), tailwind=[(c, round(r, 4)) for r, c in tw], hold=sorted(k for k, v in tgt.items() if v > 1e-12 and k != '0050')))
    return out, rec


def vehicle(P, E, dec):
    """載具指數（Dc＝含息收盤、Do＝含息開盤，尺度接在 P.E50 上）。第一次換倉列之前＝P.E50／P.E50O。
    0050 那一份用引擎自己的 P.E50／P.E50O；其他 ETF 用 etf_panel 的隔夜／盤中報酬。換倉在開盤：賣方成本＝手續費＋0.1% 證交稅＋半跳動點（＋非 0050 另加 XSL），
    買方＝手續費＋半跳動點（＋XSL）。回傳 dict(Dc, Do, cost_sum（累計成本占淨值）, n_trades（有換手的月數）, turnover（單邊累計）)。"""
    codes = E['codes']; ci = {c: i for i, c in enumerate(codes)}
    T = P.T
    Dc = np.array(P.E50, float, copy=True); Do = np.array(P.E50O, float, copy=True)
    if not dec:
        return dict(Dc=Dc, Do=Do, cost_sum=0.0, n_trades=0, turnover=0.0)
    first = min(dec)
    v = {'0050': float(P.E50[first - 1])}
    cs = 0.0; nt = 0; to = 0.0

    def side(c, t):
        if c == '0050':
            po = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
            return FEE + 0.5 * stlib.etf_tick(po) / po
        po = E['OPEN'][t, ci[c]]
        po = po if np.isfinite(po) and po > 0 else 100.0
        return FEE + 0.5 * stlib.etf_tick(po) / po + XSL

    for t in range(first, T):
        for c in v:                                                   # 隔夜
            v[c] *= (P.E50O[t] / P.E50[t - 1]) if c == '0050' else E['GO'][t, ci[c]]
        W = sum(v.values())
        Do[t] = W
        if t in dec:
            tgt = dec[t]
            allc = sorted(set(v) | set(tgt))
            cost = 0.0
            for _ in range(4):
                W2 = W - cost; cost = 0.0; sold = 0.0
                for c in allc:
                    dlt = tgt.get(c, 0.0) * W2 - v.get(c, 0.0)
                    if abs(dlt) < 1e-9 * W:
                        continue
                    if dlt < 0:
                        cost += -dlt * (side(c, t) + TAX_ETF); sold += -dlt
                    else:
                        cost += dlt * side(c, t)
            W2 = W - cost
            if cost > 0:
                cs += cost / W; nt += 1; to += sold / W
            v = {c: tgt[c] * W2 for c in tgt if tgt[c] > 1e-12}
        for c in v:                                                   # 盤中
            v[c] *= (P.E50[t] / P.E50O[t]) if c == '0050' else E['GI'][t, ci[c]]
        Dc[t] = sum(v.values())
    return dict(Dc=Dc, Do=Do, cost_sum=cs, n_trades=nt, turnover=to)


def swapped(P, V):
    """P 的淺複本，E50／E50O 換成載具指數（其他陣列共用、不複製）。V=None → 原 P。"""
    if V is None:
        return P
    P2 = copy.copy(P)
    P2.E50 = V['Dc']; P2.E50O = V['Do']
    return P2


def sim(P, S, U, V=None, cap=1e6, cost='strict', block=None, fresh=False):
    """F1＋E0、嚴成本、W10_70、空位放載具（V=None＝0050＝現行，直接走原引擎）。"""
    S1 = fresh_from(S, VALID_FIRST_MONTH) if fresh else S
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(swapped(P, V), S2, U, cost=COSTS[cost], capital=cap, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False, block=block, **ekw)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

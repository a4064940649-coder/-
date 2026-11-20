#!/usr/bin/env python3
"""S80 共用：在現行策略（選股規則完全不動，V_BRIDGE 口徑）上加一層「市場狀態」規則。
共用引擎與既有研究一律不改；重用 agents/s75_continuous_2008_2026/s75lib.py（連續面板、SURGE B2、F1 訊號、統計）。
本檔新增
- simulate_s80：用 inspect 取 s10lib.simulate_s10 原始碼做字串替換後 exec（原檔不動），多四個參數：
    cashmode[t]（t＝成交日）：0＝原行為（閒錢全掃 0050）；1＝衛星閒錢留現金（0050 只維持底倉簿記 B；寫法同 S47／S54）；2＝全部現金（0050 目標 0，底倉簿記 B 在全現金期間不隨 0050 變動）；
    szm[k]（k＝訊號日）：當月每檔目標金額的倍數（0.5＝減半）；
    tend：只跑到第 tend 列（不含）——訓練段用，結果與全期跑的前段逐日相同；tstart：起跑列（後段重新起跑的補充分析用）。
    四個參數都不給時與 simulate_s10 逐位相同（驗收 A0）。
- 市場指標 indicator()、狀態 state_series()、規則套用 apply_rule()、格子清單 CELLS、統計。
一律 python3 -I -B；不呼叫 git；不連外網。規格見 PREREG.json。"""
import sys, inspect, hashlib, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s80')
sys.path.insert(0, str(AG / 's75_continuous_2008_2026'))
import s75lib as K                                           # noqa: E402
from s75lib import R, L, S71, s10lib, stlib, cslib           # noqa: E402
K.TMP = TMP; S71.TMP = TMP; R.TMP = TMP
COSTS = L.COSTS

TRAIN_END = '2015-12-31'                                     # 訓練段 2008-01-02～2015-12-31；驗證段 2016-01～2026-10-22
CAPS = K.CAPS; SHIFTS = K.SHIFTS

# ---------------------------------------------------------------- 引擎副本
_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s80("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, cashmode=None, szm=None, tend=None, tstart=None):"),
    ("    T, t0 = P.T, P.t0\n", "    T, t0 = (tend if tend is not None else P.T), (tstart if tstart is not None else P.t0)\n"),
    ("            target = u\n", "            target = u * (szm[t - 1] if szm is not None else 1.0)\n"),
    ("""                v50 *= P.E50O[t] / P.E50[t - 1]
            B *= P.E50O[t] / P.E50[t - 1]
""",
     """                v50 *= P.E50O[t] / P.E50[t - 1]
            if not (cashmode is not None and cashmode[t - 1] == 2):
                B *= P.E50O[t] / P.E50[t - 1]
"""),
    ("""            v50 *= P.E50[t] / P.E50O[t]
            B *= P.E50[t] / P.E50O[t]
""",
     """            v50 *= P.E50[t] / P.E50O[t]
            if not (cashmode is not None and cashmode[t] == 2):
                B *= P.E50[t] / P.E50O[t]
"""),
    ("""        elif cash >= sweep_min:
            f = max(minfee, cash * fee)""",
     """        elif cashmode is not None and cashmode[t]:
            gap = (B if cashmode[t] == 1 else 0.0) - v50
            if gap >= sweep_min and cash >= sweep_min:
                x = min(cash, gap)
                f = max(minfee, x * fee); val = (x - f) / (1 + sl50)
                st['e50_cost'] += x - val; st['e50_orders'] += 1
                v50 += val; cash -= x
            elif -gap >= sweep_min or (cashmode[t] == 2 and v50 > 0):
                y = -gap
                g = y * (1 - sl50); f = max(minfee, g * fee); net = g - f - g * TAX_ETF * cm
                v50 -= y; cash += net; st['e50_cost'] += y - net; st['e50_orders'] += 1
        elif cash >= sweep_min:
            f = max(minfee, cash * fee)"""),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]!r}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s80 simulate_s80>', 'exec'), _G)
simulate_s80 = _G['simulate_s80']
SRC_SHA = dict(simulate_s10=hashlib.sha256(_SRC0.encode()).hexdigest(), simulate_s80=hashlib.sha256(_SRC.encode()).hexdigest())


def sim(P, sg, U, cap=1e6, cashmode=None, szm=None, tend=None, tstart=None):
    """現行格參數（W10_70、E0、嚴成本、5% 成交值上限）＋本題掛鉤。"""
    S2, ekw = s10lib.variant_S(sg, 'E0')
    return simulate_s80(P, S2, U, cost=COSTS['strict'], capital=cap, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False,
                        cashmode=cashmode, szm=szm, tend=tend, tstart=tstart, **ekw)


# ---------------------------------------------------------------- 格子（事前寫死；PREREG.json 逐格列出）
INDS = ('E50MA60', 'E50MA120', 'E50MA200', 'BR40', 'BR50')
IND_NEIGHBORS = {'E50MA60': ('E50MA120',), 'E50MA120': ('E50MA60', 'E50MA200'), 'E50MA200': ('E50MA120',), 'BR40': ('BR50',), 'BR50': ('BR40',)}
CUR = 'CURRENT'


def cells():
    out = []
    for act in ('NOBUY', 'HALF'):
        for dest in ('E50', 'CASH'):
            for ind in INDS:
                out.append(dict(id=f'{act}_{dest}_M_{ind}', act=act, dest=dest, freq='M', re='NEXT', ind=ind))
    for dest in ('E50', 'CASH', 'ALLCASH'):
        for freq, re in (('M', 'NEXT'), ('W', 'NEXT'), ('W', 'REBUY')):
            for ind in INDS:
                out.append(dict(id=f'EXIT_{dest}_{freq}{"R" if re == "REBUY" else ""}_{ind}', act='EXIT', dest=dest, freq=freq, re=re, ind=ind))
    out.append(dict(id='IDLE_CASH_ALWAYS', act='NONE', dest='CASH', freq='M', re='NEXT', ind=None))
    return out


CELLS = cells()
CELL = {c['id']: c for c in CELLS}
assert len(CELLS) == 66 and len(CELL) == 66


def family_key(c):
    return (c['act'], c['dest'], c['freq'], c['re'])


def neighbors(cid):
    c = CELL[cid]
    if c['ind'] is None:
        return []
    return [x['id'] for x in CELLS if family_key(x) == family_key(c) and x['ind'] in IND_NEIGHBORS[c['ind']]]


# ---------------------------------------------------------------- 市場指標（第 t 列只用 ≤ t 的收盤資料）
def indicators(P, U):
    """回傳 {指標: bool[T]，True＝市場弱}。資料不足的日子＝False（不觸發）。
    E50MAn：0050 含息淨值（P.E50）≤ 自己近 n 日簡單平均（含當日）。
    BRxx：市場寬度＝股池（上市、非創新板，U）內「還原收盤 > 60 日均線」（P.F_MA60，現行選股用的同一個旗標）的檔數 ÷ 股池檔數 < xx%。"""
    E = np.asarray(P.E50, float); T = P.T
    out = {}
    s = pd.Series(E)
    for n in (60, 120, 200):
        ma = s.rolling(n, min_periods=n).mean().to_numpy()
        out[f'E50MA{n}'] = np.isfinite(ma) & (E <= ma)
    nu = U.sum(1).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        br = np.where(nu >= 50, (U & P.F_MA60).sum(1) / nu, np.nan)
    br[:61] = np.nan
    out['BR40'] = np.isfinite(br) & (br < 0.40)
    out['BR50'] = np.isfinite(br) & (br < 0.50)
    return out, br


def week_ends(P):
    """每個 ISO 週最後一個交易日的列號（bool[T]）。"""
    iso = P.IDX.isocalendar()
    key = (iso['year'].to_numpy().astype(np.int64) * 100 + iso['week'].to_numpy().astype(np.int64))
    we = np.zeros(P.T, bool); we[:-1] = key[1:] != key[:-1]; we[-1] = True
    return we


def state_series(raw, checks):
    """checks（bool[T]）＝檢查日。狀態在檢查日收盤更新成 raw[檢查日]，維持到下一個檢查日；第一個檢查日前＝False。"""
    st = np.zeros(len(raw), bool); cur = False
    for t in range(len(raw)):
        if checks[t]:
            cur = bool(raw[t])
        st[t] = cur
    return st


def apply_rule(P, U, sg, cell, IND, WE, raw_override=None):
    """把一格市場狀態規則套到現行訊號 sg 上。回傳 (sg2, cashmode, szm, state, checks)。
    時序：檢查日 c 收盤知道狀態 → c＋1 開盤執行（賣出、0050／現金切換、買進），與現行「訊號日收盤→次日開盤」相同。
    NOBUY：弱勢的訊號日不買新股。HALF：弱勢的訊號日每檔目標金額減半。EXIT：弱勢的檢查日把全部持股標記賣出，且弱勢的訊號日不買新股。
    dest：E50＝空出的錢照現行掃進 0050；CASH＝弱勢期間衛星閒錢留現金（底倉 0050 不動）；ALLCASH＝弱勢期間連底倉 0050 都賣掉，全部現金（利率 0）。
    freq：M＝只在訊號日檢查；W＝每週最後一個交易日＋訊號日檢查。
    re（只有 EXIT×W）：NEXT＝轉強後等下一個訊號日才照名單買；REBUY＝在非訊號日的檢查日由弱轉強時，把最近一個訊號日的原始買進名單中
        當天仍在股池且仍站上 60 日線的股票買回（次日開盤，名次照原訊號日 SURGE）。
    IDLE_CASH_ALWAYS：不看市場，衛星閒錢一律留現金（＝S47 idle='cash'）。"""
    T = P.T
    sigrows = np.asarray(sg['sigrows'], bool)
    if cell['act'] == 'NONE':
        cm = np.ones(T + 1, np.int8)
        return sg, cm, None, np.zeros(T, bool), sigrows
    raw = IND[cell['ind']] if raw_override is None else raw_override
    checks = sigrows | WE if cell['freq'] == 'W' else sigrows
    st = state_series(raw, checks)
    sig = sg['sig'].copy(); rank = sg['rank'].copy(); X = sg['XV']['E0'].copy()
    szm = None
    ks = np.flatnonzero(sigrows)
    if cell['act'] in ('NOBUY', 'EXIT'):
        sig[ks[st[ks]], :] = False
    if cell['act'] == 'HALF':
        szm = np.ones(T, float); szm[ks[st[ks]]] = 0.5
    if cell['act'] == 'EXIT':
        cs = np.flatnonzero(checks)
        X[cs[st[cs]], :] = True
        if cell['re'] == 'REBUY':
            prev = False; lastk = -1
            for c in cs:
                if sigrows[c]:
                    lastk = c
                elif prev and not st[c] and lastk >= 0:
                    tg = sg['sig'][lastk] & U[c] & P.F_MA60[c]
                    sig[c, tg] = True; rank[c, tg] = sg['rank'][lastk, tg]
                prev = bool(st[c])
    code = {'E50': 0, 'CASH': 1, 'ALLCASH': 2}[cell['dest']]
    cm = np.zeros(T + 1, np.int8)
    cm[1:T + 1] = np.where(st, code, 0)                       # 成交日 t 用 t−1 收盤的狀態
    sg2 = dict(sg); sg2['sig'] = sig; sg2['rank'] = rank
    XV = dict(sg['XV']); XV['E0'] = X; sg2['XV'] = XV; sg2['X'] = X
    return sg2, (cm if code else None), szm, st, checks


# ---------------------------------------------------------------- 統計
def cagr(s):
    s = s.dropna()
    return float((s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1)


def maxdd(s):
    s = s.dropna()
    return float((s / s.cummax() - 1).min())


def longest_underwater_months(s):
    """視窗內：淨值低於視窗內先前最高點的最長連續期間（日曆天 ÷ 30.4375；期末未收復算到最後一天）。"""
    s = s.dropna(); pk = s.cummax(); uw = (s < pk * (1 - 1e-12)).to_numpy()
    best = 0; i = 0; n = len(uw); idx = s.index
    while i < n:
        if uw[i]:
            j = i
            while j + 1 < n and uw[j + 1]:
                j += 1
            end = idx[j + 1] if j + 1 < n else idx[j]
            best = max(best, (end - idx[i - 1 if i > 0 else i]).days)
            i = j + 1
        else:
            i += 1
    return best / 30.4375


def window(nav, a, b):
    """a＝基底日（含），b＝結束日（含）。"""
    return nav.loc[pd.Timestamp(a):pd.Timestamp(b)].dropna()


def win_stats(nav, bench, a, b):
    s = window(nav, a, b); q = window(bench.reindex(nav.index), a, b)
    c, d = cagr(s), maxdd(s)
    rs, rb = R.monthly_returns(s, q)
    blk = R.stats_block(rs, rb)
    return dict(cagr=c, maxdd=d, calmar=(c / abs(d) if d < 0 else float('nan')), cagr_0050=cagr(q), maxdd_0050=maxdd(q), ex_ann_pp=blk['ex_ann_pp'], t_nw6=blk['t_nw6'],
                IR=blk['IR'], underwater_m=longest_underwater_months(s), months=blk['months'])


def paired_t(nav_a, nav_b, a, b):
    """規則 a 減現行 b 的月報酬差：年化平均（個百分點）與 Newey-West t（滯後 6）。"""
    s = window(nav_a, a, b); q = window(nav_b, a, b)
    ra, rb = R.monthly_returns(s, q)
    d = (ra - rb).to_numpy()
    if np.allclose(d, 0):
        return dict(diff_ann_pp=0.0, t_nw6=0.0)
    return dict(diff_ann_pp=float(d.mean() * 1200), t_nw6=R.nw_t(d, 6))


def flat_stats(P, lg, st, a, b):
    """空手月份比例＝視窗內各月最後交易日「持股檔數＝0」的比例；另報平均持股檔數、平均股票占淨值比例、進場日在視窗內的已平倉筆數。"""
    A_, B_ = pd.Timestamp(a), pd.Timestamp(b)
    npos = K.npos_series(P, lg).loc[A_:B_]
    npos = npos[npos.index > A_]
    me = npos.groupby(npos.index.to_period('M')).tail(1)
    sf = st['stockfrac'].loc[A_:B_].dropna()
    x = lg[(lg['why'] != 'open_end') & (lg['ent_date'] > A_) & (lg['ent_date'] <= B_)]
    return dict(flat_month_frac=float((me == 0).mean()), avg_npos=float(npos.mean()), avg_stock_frac=float(sf.mean()), n_closed=int(len(x)),
                win_rate=float((x['ret'] > 0).mean()) if len(x) else None, mean_ret=float(x['ret'].mean()) if len(x) else None)

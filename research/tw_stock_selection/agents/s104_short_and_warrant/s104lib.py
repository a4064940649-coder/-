#!/usr/bin/env python3
"""S104 共用程式庫：空方與多空雙向（K1 營收負驚奇放空腿、K2 小哥式雙向布林、K3 合成權證槓桿）。
資料：只用 repo 既有正式面板（s10lib/cslib 面板：日價量、含息報酬、除權息參考價、漲跌停價；月營收 SURGE＝s19lib.rev_std；
PIT 市值代理＝s12lib 做法：o_lc_sector_rotation_pit/data/shares_full.parquet 股數落後 63 日 × 價格）。不抓外部資料。
現行策略＝正式 E0｜W10_70（五個開關全開），由 s10lib.simulate_s10 原樣呼叫（prep.py 驗收與 xq_scripts/site/navs_official.csv.gz 逐日差 0）。
本檔：load_full()（只給 prep.py 用，峰值約 1.9GB）、load_cache()（之後各步驟用的精簡陣列）、三個簡化套筒（sleeve）引擎、統計。
一律 python3 -I -B。暫存：scratchpad/s104/cache（用完可刪）。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s104/cache')
TMP.mkdir(parents=True, exist_ok=True)
CACHE = TMP / 'data.npz'
OFFNAV = '/home/user/-/xq_scripts/site/navs_official.csv.gz'
SHARES = AG / 'o_lc_sector_rotation_pit' / 'data' / 'shares_full.parquet'
LAG = 63

FEE, TAX = 0.001425 * 0.28, 0.003           # 同正式（stlib）：手續費 0.1425%×2.8 折、賣出證交稅 0.3%
XSLIP = 0.003                               # 正式嚴成本：每邊另加 0.3% 滑價（再加半個跳動點）
SBL_FEE = 0.0008                            # 融券手續費（借券費）0.08%，放空進場時一次
WTAX = 0.001                                # 權證賣出證交稅 0.1%
RF = 0.01                                   # BS 無風險利率
TRAIN = ('2016-02-01', '2022-01-01')        # 訓練窗：以「< 日期」的最後一個交易日為端點
VALID_END = '2026-10-22'
SHIFTS = tuple(range(20))


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


# ---------------------------------------------------------------- 資料
def load_full():
    """同 s100lib.load（正式五開關），暫存路徑換成本題。回傳 (P, U, RT)。"""
    sys.path.insert(0, str(AG / 's19_xq_threshold'))
    import s19lib
    from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib
    for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                      (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
        setattr(mod, attr, TMP)
    cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
    cslib.REDUCE_FIX = True; cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV
    P = s19lib.load()
    s19lib.e50_fix_suspension(P)
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    S, _ = s19lib.rev_std(P)
    avA = s19lib.a10_rows(P, list(S.index))
    mods = dict(s19lib=s19lib, s10lib=s10lib, stlib=stlib, cslib=cslib, COSTS=s19lib.COSTS)
    return P, U, dict(S=S, av=avA), mods


def mcap_panel(P):
    sh = pd.read_parquet(SHARES)
    sh = sh.reindex(sh.index.union(P.IDX)).ffill().reindex(P.IDX).reindex(columns=P.COLS).to_numpy(np.float64)
    now = sh * P.CF
    MC = np.full((P.T, P.N), np.nan)
    MC[LAG:] = now[:-LAG] * (P.G[LAG:] / P.G[:-LAG])
    return np.where(np.isfinite(MC), MC, now)


class D:
    pass


def load_cache():
    z = np.load(CACHE, allow_pickle=False)
    d = D()
    for k in z.files:
        setattr(d, k, z[k])
    d.IDX = pd.DatetimeIndex(d.IDX)
    d.T, d.N = d.O.shape
    lastb = lambda s: int(np.flatnonzero(d.IDX < pd.Timestamp(s))[-1])
    d.TA, d.TB = lastb(TRAIN[0]), lastb(TRAIN[1])
    d.TE = int(np.flatnonzero(d.IDX <= pd.Timestamp(VALID_END))[-1])
    d.r50 = np.r_[0.0, d.E50[1:] / d.E50[:-1] - 1]
    return d


def official_nav(d):
    r = pd.read_csv(OFFNAV, index_col=0, parse_dates=True)['E0|W10_70']
    return r.reindex(d.IDX)


# ---------------------------------------------------------------- 統計
def cagr(nav, a, b):
    v = np.asarray(nav, float)
    if not (np.isfinite(v[a]) and np.isfinite(v[b]) and v[a] > 0):
        return float('nan')
    yrs = (DCACHE['IDX'][b] - DCACHE['IDX'][a]).days / 365.25
    return float((v[b] / v[a]) ** (1 / yrs) - 1) if v[b] > 0 else -1.0


DCACHE = {}


def set_idx(d):
    DCACHE['IDX'] = d.IDX


def maxdd(nav, a, b):
    v = np.asarray(nav, float)[a:b + 1]
    return float((v / np.maximum.accumulate(v) - 1).min())


def sharpe(nav, a, b):
    v = np.asarray(nav, float)[a:b + 1]
    r = v[1:] / v[:-1] - 1
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(252)) if sd > 0 else float('nan')


def rets(nav):
    v = np.asarray(nav, float)
    return np.r_[0.0, v[1:] / v[:-1] - 1]


def corr(nav1, nav2, a, b):
    r1, r2 = rets(nav1)[a + 1:b + 1], rets(nav2)[a + 1:b + 1]
    return float(np.corrcoef(r1, r2)[0, 1])


def overlay(navL, navS, w):
    """多方淨值＋w 倍的套筒日報酬（每日再平衡的近似）。"""
    r = rets(navL) + w * rets(navS)
    return np.cumprod(1 + r)


def blend(nav1, nav2, w2):
    r = (1 - w2) * rets(nav1) + w2 * rets(nav2)
    return np.cumprod(1 + r)


def yearly(nav, d, a, b):
    s = pd.Series(np.asarray(nav, float), index=d.IDX).iloc[a:b + 1]
    y = s.groupby(s.index.year).last()
    first = s.iloc[0]
    out = {}
    prev = first
    for yr, v in y.items():
        out[str(yr)] = float(v / prev - 1); prev = v
    return out


def metrics(nav, d, a, b, trades=None, navref=None):
    e50 = d.E50
    o = dict(cagr=cagr(nav, a, b), mdd=maxdd(nav, a, b), sharpe=sharpe(nav, a, b), ex0050=cagr(nav, a, b) - cagr(e50, a, b))
    if navref is not None:
        o['corr_official'] = corr(nav, navref, a, b)
    if trades is not None:
        tr = trades[(trades['ent_t'] > a) & (trades['ent_t'] <= b)]
        o['n_trades'] = int(len(tr)); o['win_rate'] = float((tr['ret'] > 0).mean()) if len(tr) else float('nan')
        o['avg_ret'] = float(tr['ret'].mean()) if len(tr) else float('nan')
    return o


# ---------------------------------------------------------------- 套筒引擎（多或空，等額槽位）
def sleeve(d, side, cand, exitf, t_start, t_end, slots=10, borrow=0.02, cost='strict', hold_to=None):
    """通用套筒：資金 1，最多 slots 檔，每檔＝前一日收盤套筒淨值 ÷ slots（進場時決定，之後隨價格漂移）。
    side：+1 做多、−1 放空。cand(t)→ 訊號日 t 收盤後的候選 j 陣列（已依優先順序排好）；exitf(t, j, e)→ True 表示 t 收盤標記出場。
    hold_to：dict t→set（K1：換股日重選時，仍在新名單者續抱不換手）。成交一律 t+1 開盤。
    成本（strict）：做多同正式（手續費、賣出稅 0.3%、每邊半跳動點＋0.3%）；放空另加：進場（賣）稅 0.3%＋手續費＋融券手續費 0.08%＋滑價、
    回補（買）手續費＋滑價、借券費 borrow（年化，按持有交易日 /252 計在負債市值上）。
    開盤無價→延後；做多開盤漲停不買、跌停不賣（延後）；放空開盤跌停不空、漲停不補（延後）。停牌 > 20 個交易日且已無報價→以最後收盤結清。
    回傳 nav（長度 T，t_start 之前 NaN）、trades DataFrame（j, ent_t, exit_t, ret 淨報酬, why）。"""
    O, CF, R, REF, LU, LD = d.O, d.CF, d.R, d.REF, d.LIMUP, d.LIMDN
    cm = 0.0 if cost == 'zero' else 1.0
    xs = XSLIP * cm
    cash = 1.0; pos = {}                         # j -> [notional_entry(after cost basis), value_now, e, flag, ret_base]
    nav = np.full(d.T, np.nan); nav[t_start] = 1.0
    pend = []; log = []
    for t in range(t_start + 1, t_end + 1):
        # 1) 出場（前一日收盤標記）
        for j in list(pos):
            p = pos[j]
            if not p['flag']:
                continue
            o = O[t, j]
            stale = t - p['last'] > 20
            if not (np.isfinite(o) and o > 0):
                if stale:
                    val = p['v']; why = p['why'] + '|stale'
                else:
                    continue
            else:
                if side > 0 and o <= LD[t, j] + 1e-9:
                    continue
                if side < 0 and o >= LU[t, j] - 1e-9:
                    continue
                val = p['v'] * o / REF[t, j] if np.isfinite(REF[t, j]) and REF[t, j] > 0 else p['v']
                why = p['why']
            sl = (0.5 * float(tick(o)) / o + xs) * cm if np.isfinite(o) and o > 0 else xs
            if side > 0:
                net = val * (1 - FEE * cm - TAX * cm - sl)
                cash += net
                ret = net / p['n0'] - 1
            else:
                buy = val * (1 + FEE * cm + sl)
                cash += p['n0'] - buy               # 放空損益（進場成本已在進場時扣）
                ret = (p['n0'] - buy - p['c0'] - p['bf']) / p['n0']
            log.append((j, p['e'], t, ret, why)); del pos[j]
        # 2) 進場（前一日收盤的候選）
        if pend:
            eq = nav[t - 1]
            tgt = eq / slots
            for j in pend:
                if len(pos) >= slots:
                    break
                if j in pos:
                    continue
                o = O[t, j]
                if not (np.isfinite(o) and o > 0):
                    continue
                if side > 0 and o >= LU[t, j] - 1e-9:
                    continue
                if side < 0 and o <= LD[t, j] + 1e-9:
                    continue
                sl = (0.5 * float(tick(o)) / o + xs) * cm
                if side > 0:
                    if cash < tgt * 0.5:
                        continue
                    n = min(tgt, cash)
                    cash -= n
                    v = n / (1 + FEE * cm + sl)
                    pos[j] = dict(n0=n, v=v * CF[t, j] / o, e=t, flag=False, why='', last=t, c0=0.0, bf=0.0)
                else:
                    c = tgt * (FEE * cm + TAX * cm + SBL_FEE * cm + sl)
                    cash -= c
                    pos[j] = dict(n0=tgt, v=tgt * CF[t, j] / o, e=t, flag=False, why='', last=t, c0=c, bf=0.0)
            pend = []
        # 3) 收盤估值（持有中第 2 天起乘 R）
        tot = cash
        for j, p in pos.items():
            if p['e'] < t:
                r = R[t, j]
                if np.isfinite(r):
                    p['v'] *= (1 + r)
                if np.isfinite(O[t, j]) and O[t, j] > 0:
                    p['last'] = t
            if side > 0:
                tot += p['v']
            else:
                bf = p['v'] * borrow / 252 * cm
                p['bf'] += bf; cash -= bf; tot -= bf
                tot += p['n0'] - p['v']
        nav[t] = tot
        if t == t_end:
            break
        # 4) 標記出場、產生候選
        keep = hold_to.get(t) if hold_to is not None else None
        for j, p in pos.items():
            if p['flag']:
                continue
            w = exitf(t, j, p['e'])
            if w:
                if keep is not None and w == 'rebal' and j in keep:
                    continue
                p['flag'] = True; p['why'] = w
        c = cand(t)
        if c is not None and len(c):
            pend = [int(j) for j in c if int(j) not in pos]
    # 期末未出場：以收盤價值計（不扣出場成本），why=open_end
    for j, p in pos.items():
        if side > 0:
            ret = p['v'] / p['n0'] - 1
        else:
            ret = (p['n0'] - p['v'] - p['c0'] - p['bf']) / p['n0']
        log.append((j, p['e'], t_end, ret, 'open_end'))
    return nav, pd.DataFrame(log, columns=['j', 'ent_t', 'exit_t', 'ret', 'why'])


# ---------------------------------------------------------------- Black-Scholes
def _ncdf(x):
    from scipy.special import ndtr
    return ndtr(np.asarray(x, float))


def bs_call(S, K, T, sig, r=RF):
    S = np.asarray(S, float); T = np.asarray(T, float)
    out = np.maximum(S - K, 0.0).astype(float)
    m = T > 1e-9
    if m.any():
        s = sig * np.sqrt(T[m])
        d1 = (np.log(S[m] / K) + (r + 0.5 * sig * sig) * T[m]) / s
        d2 = d1 - s
        out[m] = S[m] * _ncdf(d1) - K * np.exp(-r * T[m]) * _ncdf(d2)
    return out

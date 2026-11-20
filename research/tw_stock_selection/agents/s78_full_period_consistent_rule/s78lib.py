#!/usr/bin/env python3
"""S78 共用：全期一致性規則搜尋（2008-01～2026-10 連續帳戶，交叉驗證）。共用引擎與既有研究一律不改；重用
- s75_continuous_2008_2026/s75lib.py：load_cont()（連續面板、股池＝上市 & ~創新板、0050 停牌修正）、sim()（→ s47lib.sim → s10lib.simulate_s10；嚴成本、空位買 0050）；
- s71lib.rev_table／scope_ratio（B2 口徑比例 k_n）、s17lib.zscore、cslib.rev_avail_days／ins15_mask_row；r55lib 的月度統計。
本檔新增
- scores()：6 種營收分數（全部用 B2 水準校正後的營收；2013 年各月只用當時已公布的 n 個月來估 k，同 S71）；
- make_signal()：s10lib.make_signal_s10 的一般化（分數、趨勢均線、進場深度與動能門檻、出場、大盤過濾都是參數）；
  現行格（STD24｜MA60｜前10且20日漲幅>10%｜掉出前20或≤均線｜無大盤過濾）必須與 s10lib.make_signal_s10(B2) 逐格相同（驗收 A1）；
- 區塊統計、交叉驗證、CSCV、排列／自助法。
一律 python3 -I -B；不呼叫 git。"""
import sys, math, itertools
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s78')
sys.path.insert(0, str(AG / 's75_continuous_2008_2026'))
import s75lib as K                                           # noqa: E402
from s75lib import S71, R, L, s20lib, s17lib, s10lib, s4lib, stlib, cslib   # noqa: E402
K.TMP = TMP; S71.TMP = TMP; R.TMP = TMP

FIRST_MONTH = pd.Period('2008-01', 'M')
BLOCKS = {'B1 2008-2012': ('2008-01', '2012-12'), 'B2 2013-2016': ('2013-01', '2016-12'),
          'B3 2017-2021': ('2017-01', '2021-12'), 'B4 2022-2026': ('2022-01', '2026-10')}
HALVES = {'前段 2008-2015': ('2008-01', '2015-12'), '後段 2016-2026': ('2016-01', '2026-10')}

SCORES = ('STD24', 'W12', 'W36', 'YOY', 'ACC', 'COMBO')
TRENDS = (60, 120)
ENTRIES = {'T10M10': dict(depth=10, mom_lo=0.10), 'T10M0': dict(depth=10, mom_lo=-np.inf), 'T20M10': dict(depth=20, mom_lo=0.10)}
EXITS = ('X20', 'X20D')
REGIMES = ('NONE', 'NOENTRY')
WEIGHTS = {'W70': 0.7, 'W100': 1.0}
CUR = 'STD24|MA60|T10M10|X20|NONE|W70'


def grid():
    out = []
    for sc, tr, en, ex, rg, w in itertools.product(SCORES, TRENDS, ENTRIES, EXITS, REGIMES, WEIGHTS):
        out.append(dict(id=f'{sc}|MA{tr}|{en}|{ex}|{rg}|{w}', score=sc, trend=tr, entry=en, exit=ex, regime=rg, weight=w))
    return out


# ---------------------------------------------------------------- 營收分數
def _adj(Rv, k):
    """B2：2012-12 以前營收 × k（k 缺值的公司不乘）。回傳 (校正後營收, k 缺值欄)。"""
    pre = np.asarray(Rv.index < S71.BREAK)
    kv = k.reindex(Rv.columns).to_numpy(float)
    A = Rv.to_numpy(float).copy()
    A[pre] = A[pre] * np.where(np.isfinite(kv), kv, 1.0)[None, :]
    return pd.DataFrame(A, index=Rv.index, columns=Rv.columns), ~np.isfinite(kv)


def _raw_scores(Radj, knan):
    """由校正後營收算各分數（月份×股票）。k 缺值的公司，2013 年的年增量／年增率一律缺值（同 S71 B2）。"""
    y13 = np.asarray(Radj.index.year == 2013)
    Dv = np.array((Radj - Radj.shift(12)).to_numpy(float), copy=True)
    Dv[np.ix_(y13, knan)] = np.nan
    D = pd.DataFrame(Dv, index=Radj.index, columns=Radj.columns)
    out = {}
    for name, w in (('STD24', 24), ('W12', 12), ('W36', 36)):
        out[name], _ = s17lib.zscore(D, w, 'mean')
    with np.errstate(invalid='ignore', divide='ignore'):
        Yv = np.array((Radj / Radj.shift(12) - 1).to_numpy(float), copy=True)
    Yv[~np.isfinite(Yv)] = np.nan
    Yv[np.ix_(y13, knan)] = np.nan
    Y = pd.DataFrame(Yv, index=Radj.index, columns=Radj.columns)
    out['YOY'] = Y
    out['ACC'] = Y - (Y.shift(1) + Y.shift(2) + Y.shift(3)) / 3.0      # 四個月的年增率都要有值
    return out


def scores(P):
    """回傳 dict(S={分數: DataFrame（index＝營收月份、columns＝P.COLS）}, av＝訊號日列號)。COMBO 在 make_signal 內由 STD24 與 YOY 的當日橫斷面百分位平均而得。
    2013 年的 PIT 做法同 S71.rev_rule('B2')：營收月 ≤ 2013-01 用 k_1、2013-02 用 k_2、2013-03 起用 k_3。"""
    Rv = S71.rev_table(P.COLS)
    per = {}
    for n in (1, 2, 3):
        k, _ = S71.scope_ratio(Rv, n)
        per[n] = _raw_scores(*_adj(Rv, k))
    out = {}
    idx = Rv.index
    r1 = np.asarray(idx <= pd.Period('2013-01', 'M')); r2 = np.asarray(idx == pd.Period('2013-02', 'M'))
    for name in ('STD24', 'W12', 'W36', 'YOY', 'ACC'):
        Z = per[3][name].copy()
        Z.loc[r1] = per[1][name].loc[r1]; Z.loc[r2] = per[2][name].loc[r2]
        Z[Z.index < FIRST_MONTH] = np.nan
        out[name] = Z
    av = cslib.rev_avail_days(P, list(idx), True)
    return dict(S=out, av=av, Rv=Rv)


def _pct(x, m):
    """m 內的百分位名次（0～1，值大者高；平均名次法）。m 外 NaN。"""
    o = np.full(x.shape, np.nan)
    if m.sum() > 1:
        o[m] = pd.Series(x[m]).rank(method='average').to_numpy() / m.sum()
    return o


# ---------------------------------------------------------------- 訊號（s10lib.make_signal_s10 的一般化）
def prep_panel(P):
    P.F_MA = {60: P.F_MA60, 120: P.G > stlib.rmean(P.G, 120)}
    e = pd.Series(P.E50)
    P.E50_ABOVE = (e > e.rolling(60, min_periods=60).mean()).to_numpy()      # 0050 還原收盤 > 自己的 60 日均線
    return P


def make_signal(P, univ, F, score, trend, entry, exit_, regime, shift=0, nsel=10):
    """score∈SCORES；trend∈{60,120}（還原收盤 > N 日均線才排名）；entry＝ENTRIES 的鍵（只看排名前 depth 名、近 20 日漲幅 > mom_lo、最多 nsel 檔、不遞補）；
    exit_：'X20'＝訊號日掉出（趨勢合格者中）前 20 名或收盤 ≤ N 日均線；'X20D'＝X20 再加「任何交易日收盤 ≤ N 日均線」；
    regime：'NONE'；'NOENTRY'＝訊號日 0050 還原收盤 ≤ 60 日均線時當月不進新倉（出場照常）。"""
    av = F['av']; FM = P.F_MA[trend]
    depth, mom_lo = ENTRIES[entry]['depth'], ENTRIES[entry]['mom_lo']
    Sx = F['S']['STD24'] if score == 'COMBO' else F['S'][score]
    Sy = F['S']['YOY'] if score == 'COMBO' else None
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan)
    X = np.zeros((P.T, P.N), bool)
    info = []; RQ = {}
    for m in Sx.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 121:
            continue
        s = Sx.loc[m].to_numpy()
        if score == 'COMBO':
            y = Sy.loc[m].to_numpy()
            both0 = np.isfinite(s) & np.isfinite(y)
            if int((univ[k0] & both0).sum()) < 50:
                continue
        elif int((univ[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        if score == 'COMBO':
            y = cslib.ins15_mask_row(P, m, k, y)
            mm = univ[k] & np.isfinite(s) & np.isfinite(y)
            s = (_pct(s, mm) + _pct(y, mm)) / 2.0
        fin = univ[k] & np.isfinite(s)
        ok_ma = fin & FM[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        pool = o_ma[:depth]
        okr = P.RET20[k][pool] > mom_lo
        tgt = pool[okr][:nsel]
        if regime == 'NOENTRY' and not P.E50_ABOVE[k]:
            tgt = tgt[:0]
        sig[k, tgt] = True; rank[k, tgt] = s[tgt]
        RQ[k] = set(int(j) for j in tgt)
        X[k, :] = True; X[k, o_ma[:20]] = False
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok_ma.sum()), n_tgt=int(len(tgt)),
                         targets=[str(P.COLS[j]) for j in tgt]))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    if exit_ == 'X20D':
        X = X | ~FM
    else:
        assert exit_ == 'X20'
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, RQ=RQ, sigrows=sigrows)


def simulate(P, sg, U, weight, cap=1e6):
    """嚴成本、W10（每檔＝衛星 ÷ 10、最多 10 檔）、空位買 0050；weight＝衛星比例（0.7＝W10_70；1.0＝不留底倉）。sg['X'] 直接當出場矩陣。"""
    kw = dict(cost=L.COSTS['strict'], capital=cap, cap_pct=0.05, sat_w=WEIGHTS[weight], sizing='W10', lock=False)
    return s10lib.simulate_s10(P, sg, U, **kw)


# ---------------------------------------------------------------- 統計
def monthly(nav, bench):
    return R.monthly_returns(nav, bench)


def win(x, a, b):
    return x[(x.index >= pd.Period(a, 'M')) & (x.index <= pd.Period(b, 'M'))]


def ex_ann(ex):
    return float(np.mean(ex) * 12 * 100)


def block_stats(rs, rb, nav=None, a=None, b=None):
    s, q = win(rs, a, b), win(rb, a, b)
    blk = R.stats_block(s, q)
    o = dict(months=blk['months'], cagr=blk['cagr_strat'], cagr_0050=blk['cagr_0050'], ex_ann_pp=blk['ex_ann_pp'], t_nw6=blk['t_nw6'], t_iid=blk['t_iid'], IR=blk['IR'], te_ann_pp=blk['te_ann_pp'])
    if nav is not None:
        o['maxdd'] = dd_window(nav, a, b)
    return o


def dd_window(nav, a, b):
    pa, pb = pd.Period(a, 'M'), pd.Period(b, 'M')
    idx = nav.index
    pre = idx[idx < pa.to_timestamp()]
    d0 = pre[-1] if len(pre) else idx[0]
    d1 = idx[idx <= pb.to_timestamp(how='end')][-1]
    return R.maxdd(nav.loc[d0:d1])


def crit_min_block(EX, months_by_block):
    """EX：月份×試驗 的月超額矩陣（ndarray）。months_by_block：list of 布林遮罩或整數索引。回傳 (每個試驗的『各塊年化超額最小值』, 各塊年化超額平均)。"""
    v = np.stack([EX[m].mean(axis=0) * 1200.0 for m in months_by_block])
    return v.min(axis=0), v.mean(axis=0)


def pick(EX, months_by_block):
    """挑選準則：各塊年化超額的最小值最大；同分時取各塊平均較大者，再同分取序號小者。"""
    mn, av = crit_min_block(EX, months_by_block)
    return int(np.lexsort((np.arange(len(mn)), -av, -mn))[0]), mn, av

#!/usr/bin/env python3
"""S77 共用：只看 2008-01～2015-12 訂規則、2016-01～2026-10 驗證一次。
重用（全部不改）：s75lib（連續面板載入、成績單統計）→ s71lib（B2 口徑比例 scope_ratio、月營收表 rev_table）→ r55lib（月度統計）→ s47lib.simulate_s47／s10lib（引擎、嚴成本、空位買 0050）。
本檔只新增：(1) 截斷面板 truncate()；(2) 8 種營收訊號（跨 IFRS 斷點一律用 S71 B2 的水準校正）；(3) 一般化的訊號產生器 make_signal()（現行格＝s10lib.make_signal_s10 逐格相同）；
(4) 執行包裝 run_cfg()；(5) 訓練段評分。一律 python3 -I -B；不呼叫 git、不連網。規格見 PREREG.json。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s77')
sys.path.insert(0, str(AG / 's75_continuous_2008_2026'))
import s75lib as K                                           # noqa: E402
from s75lib import S71, R, L, s17lib, s10lib, stlib, cslib   # noqa: E402
K.TMP = TMP; S71.TMP = TMP; R.TMP = TMP
COSTS = L.COSTS

TRAIN_END = '2015-12-31'
TRAIN = ('2008-01', '2015-12'); VALID = ('2016-01', '2026-10')
SIGNALS = ('Z24', 'Z12', 'Z36', 'YOY', 'DYOY', 'MOM', 'NH', 'ZY')
FILTERS = ('NONE', 'MA60', 'MA120', 'MA60R20')
DEPTHS = (5, 10, 15, 20)
EXITS = ('RANK2N', 'MAX60', 'H1', 'H2', 'H3')
MARKETS = ('NOMKT', 'MKT60')
WEIGHTS = ('W10_70', 'EQ')
CURRENT = dict(sig='Z24', filt='MA60R20', n=10, exit='RANK2N', mkt='NOMKT', wt='W10_70')
SIG_NEIGH = {'Z24': ('Z12', 'Z36', 'ZY'), 'Z12': ('Z24',), 'Z36': ('Z24',), 'YOY': ('DYOY', 'ZY'), 'DYOY': ('YOY',), 'MOM': (), 'NH': (), 'ZY': ('Z24', 'YOY')}


def cid(c):
    return f"{c['sig']}|{c['filt']}|N{c['n']}|{c['exit']}|{c['mkt']}|{c['wt']}"


def stage1_cfgs():
    return [dict(sig=s, filt=f, n=n, exit='RANK2N', mkt='NOMKT', wt='W10_70') for s in SIGNALS for f in FILTERS for n in DEPTHS]


def stage2_cfgs(base):
    return [dict(sig=base['sig'], filt=base['filt'], n=base['n'], exit=e, mkt=m, wt=w) for e in EXITS for m in MARKETS for w in WEIGHTS]


def neighbors1(c):
    """第一階段鄰居：只差一個維度——名次差一級、趨勢過濾任一個其他選項、訊號依 SIG_NEIGH。"""
    out = []
    i = DEPTHS.index(c['n'])
    for j in (i - 1, i + 1):
        if 0 <= j < len(DEPTHS):
            out.append(dict(c, n=DEPTHS[j]))
    out += [dict(c, filt=f) for f in FILTERS if f != c['filt']]
    out += [dict(c, sig=s) for s in SIG_NEIGH[c['sig']]]
    return out


def neighbors2(c):
    """第二階段鄰居：出場、大盤過濾、權重三個維度中只差一個。"""
    return [dict(c, exit=e) for e in EXITS if e != c['exit']] + [dict(c, mkt=m) for m in MARKETS if m != c['mkt']] + [dict(c, wt=w) for w in WEIGHTS if w != c['wt']]


# ---------------------------------------------------------------- 面板
def load(truncate_at=None):
    P, U = K.load_cont()
    P.F_MA120 = P.G > stlib.rmean(P.G, 120)                  # 還原收盤 > 120 日均線（同 F_MA60 的算法，只換天數）
    e = pd.Series(P.E50)
    P.MKT_OK = (e > e.rolling(60, min_periods=60).mean()).to_numpy()   # 0050 含息淨值 > 自己的 60 日均線
    if truncate_at is not None:
        n = int(np.searchsorted(P.IDX.values, np.datetime64(pd.Timestamp(truncate_at)), side='right'))
        T0 = P.T
        for a, v in list(vars(P).items()):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T0:
                setattr(P, a, v[:n].copy())
        P.IDX = P.IDX[:n]; P.T = n
        U = U[:n].copy()
        assert str(P.IDX[-1].date()) <= truncate_at and all(not (isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T0) for v in vars(P).values())
    return P, U


# ---------------------------------------------------------------- 營收訊號
def _raw_signals(Rv):
    """Rv：月營收（月份 × 代號，千元，>0）。回傳 {訊號: DataFrame}，全部「越大越好」。"""
    D = Rv - Rv.shift(12)
    out = {}
    for w in (24, 12, 36):
        out[f'Z{w}'], _ = s17lib.zscore(D, w, 'mean')
    yoy = (Rv / Rv.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
    out['YOY'] = yoy
    out['DYOY'] = yoy - (yoy.shift(1) + yoy.shift(2) + yoy.shift(3)) / 3
    out['MOM'] = (Rv / Rv.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
    pmax = Rv.shift(1).rolling(12, min_periods=12).max()
    hi = (Rv > pmax) & pmax.notna() & Rv.notna()             # 當月營收創近 12 個月新高
    st = np.zeros(Rv.shape); h = hi.to_numpy()
    for i in range(Rv.shape[0]):
        st[i] = np.where(h[i], (st[i - 1] if i else 0) + 1, 0)
    st = np.minimum(st, 12)
    nh = pd.DataFrame(st, index=Rv.index, columns=Rv.columns) + yoy.clip(lower=0, upper=9) / 10   # 連續創高月數（上限 12）＋年增率小數（同分時年增率高者先）
    out['NH'] = nh.where((st > 0) & yoy.notna().to_numpy())
    both = out['Z24'].notna() & yoy.notna()
    out['ZY'] = (out['Z24'].where(both).rank(axis=1, pct=True) + yoy.where(both).rank(axis=1, pct=True)) / 2
    return out


def signals(P, first_month='2008-01', last_month=None):
    """8 種訊號，跨 IFRS 斷點（2013-01 起合併營收）照 S71 B2 的水準校正：
    營收月 ≤ 2012-12 → 用原始營收；2013-01 → 2012-12 以前營收 × k_1；2013-02 → × k_2；2013-03 起 → × k_3（k_n＝s71lib.scope_ratio）；
    k 缺值的公司，2012-12 以前營收當缺值（＝任何用到斷點前營收的量都缺值）。Z24 與 s71lib.rev_rule('B2') 逐格相同（驗收 A1）。
    last_month：只讀到這個營收月（訓練段截斷用）。"""
    Rv = S71.rev_table(P.COLS)
    if last_month is not None:
        Rv = Rv.loc[:pd.Period(last_month, 'M')]
    pre = np.asarray(Rv.index < S71.BREAK)
    tabs = {0: _raw_signals(Rv)}
    for n in (1, 2, 3):
        k, _ = S71.scope_ratio(Rv, n)
        kv = k.reindex(Rv.columns).to_numpy(float)
        A = Rv.to_numpy(float).copy(); A[pre] = A[pre] * kv[None, :]
        tabs[n] = _raw_signals(pd.DataFrame(A, index=Rv.index, columns=Rv.columns))
    P_ = lambda s: pd.Period(s, 'M')
    out = {}
    for name in SIGNALS:
        Z = tabs[0][name].copy()
        Z.loc[Z.index == P_('2013-01')] = tabs[1][name].loc[Z.index == P_('2013-01')]
        Z.loc[Z.index == P_('2013-02')] = tabs[2][name].loc[Z.index == P_('2013-02')]
        Z.loc[Z.index >= P_('2013-03')] = tabs[3][name].loc[Z.index >= P_('2013-03')]
        Z[Z.index < P_(first_month)] = np.nan
        out[name] = Z
    av = cslib.rev_avail_days(P, list(Rv.index), True)
    return out, av


# ---------------------------------------------------------------- 訊號產生器
def make_signal(P, univ, Z, av, c, shift=0):
    """一般化的 s10lib.make_signal_s10（現行格逐格相同，驗收 A1）。訊號日 k＝營收可得日 av[m]＋shift。
    合格＝股池內 & 分數有值 & 趨勢過濾（NONE 無／MA60 還原收盤>60 日線／MA120／MA60R20＝MA60）；依分數由大到小排名（穩定排序）。
    進場＝合格排名前 n 名（MA60R20 另要近 20 日漲幅 > 10%）；MKT60：訊號日 0050 含息淨值 ≤ 60 日均線時當月不買新股（出場照常）。
    出場（只在訊號日收盤判定、次日開盤賣）：RANK2N＝不在合格排名前 2n 名；MAX60＝還原收盤 ≤ 60 日線（不看名次）；
    H1／H2／H3＝進場後經過 1／2／3 個訊號日起，不在當月進場名單就賣（引擎 requal 掛鉤）。"""
    n = c['n']; filt = c['filt']
    FM = {'NONE': None, 'MA60': P.F_MA60, 'MA120': P.F_MA120, 'MA60R20': P.F_MA60}[filt]
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan); X = np.zeros((P.T, P.N), bool)
    info = []; RQ = {}
    for m in Z.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 121:
            continue
        s = Z.loc[m].to_numpy()
        if int((univ[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = univ[k] & np.isfinite(s)
        ok = fin & FM[k] if FM is not None else fin
        o = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
        pool = o[:n]
        if filt == 'MA60R20':
            pool = pool[P.RET20[k][pool] > 0.10]
        tgt = pool[:n]
        RQ[k] = set(int(j) for j in tgt)
        mkt_ok = True if c['mkt'] == 'NOMKT' else bool(P.MKT_OK[k])
        if mkt_ok:
            sig[k, tgt] = True
        rank[k, tgt] = s[tgt]
        if c['exit'] == 'RANK2N':
            X[k, :] = True; X[k, o[:2 * n]] = False
        elif c['exit'] == 'MAX60':
            X[k, :] = ~P.F_MA60[k]
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ok=int(ok.sum()), n_tgt=int(len(tgt)) if mkt_ok else 0, mkt_ok=mkt_ok,
                         targets=[str(P.COLS[j]) for j in tgt] if mkt_ok else []))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, RQ=RQ, sigrows=sigrows)


def run_cfg(P, U, ZS, av, c, cap=1e6, shift=0, block=None):
    """權重：W10_70＝衛星 70%、每檔＝衛星 ÷ 10、最多 10 檔（現行）；EQ＝全部資金等分 n 份、最多 n 檔（sat_w＝1）。空位與底倉一律買 0050；嚴成本；5% 成交值上限。"""
    sg = make_signal(P, U, ZS[c['sig']], av, c, shift)
    w = dict(sat_w=0.7, ndiv=10, slots=10) if c['wt'] == 'W10_70' else dict(sat_w=1.0, ndiv=c['n'], slots=c['n'])
    rq = {'H1': 1, 'H2': 2, 'H3': 3}.get(c['exit'])
    nav, lg, st = L.simulate_s47(P, sg, U, idle='e50', cost=COSTS['strict'], capital=cap, cap_pct=0.05, sizing='W10', lock=False, requal=rq, block=block, **w)
    return sg, nav, lg, st


# ---------------------------------------------------------------- 統計
def bench_of(P, cap=1e6):
    b = pd.Series(P.E50, index=P.IDX).iloc[P.t0:]
    return b / b.iloc[0] * cap


def turnover(lg, nav, a, b):
    """年化單邊換手＝窗內進場的買進成本合計 ÷ 窗內平均淨值 ÷ 年數。"""
    s = nav.loc[a:b]
    x = lg[(lg['ent_date'] >= s.index[0]) & (lg['ent_date'] <= s.index[-1])]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float(x['cost'].sum() / s.mean() / yrs)


def train_metrics(P, nav, lg, st, bench):
    """訓練段（整條淨值＝2008-01-02～2015-12-31）評分用統計。"""
    yt = K.yearly_table(nav, bench)
    ex = np.array([y['diff_pp'] for y in yt])
    blk = K.window_block(nav, bench, *TRAIN)
    x = lg[lg['why'] != 'open_end']
    return dict(yearly_excess_pp=[float(v) for v in ex], yr_median=float(np.median(ex)), yr_min=float(ex.min()), score=float(0.5 * np.median(ex) + 0.5 * ex.min()),
                cagr=blk['cagr'], cagr_0050=blk['cagr_0050'], cagr_diff_pp=blk['cagr_diff_pp'], ex_ann_pp=blk['ex_ann_pp'], IR=blk['IR'], t_nw6=blk['t_nw6'], t_iid=blk['t_iid'],
                maxdd=blk['maxdd'], maxdd_0050=blk['maxdd_0050'], calmar=float(blk['cagr'] / abs(blk['maxdd'])), n_closed=int(len(x)),
                avg_npos=float(st['avg_npos']), avg_idle_frac=float(st['avg_idle_frac']), turnover=turnover(lg, nav, None, None))

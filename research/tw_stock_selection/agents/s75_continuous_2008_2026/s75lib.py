#!/usr/bin/env python3
"""S75 共用：2008-01～2026-10 連續帳戶。共用引擎與既有研究一律不改；重用
- agents/s71_ifrs_bridge_oos/s71lib.py：rev_rule('MASK'｜'B2')（MASK 與 s17lib.rev_family 逐格相同＝S71 驗收 A3；B2＝S71 事前登記的水準校正）；
- agents/r55_replicate_s55/r55lib.py：快取改道 point_cache、月度統計 monthly_returns／stats_block／nw_t；
- s47lib.sim（→ s10lib.simulate_s10：W10_70、E0、嚴成本、空位買 0050）、s10lib.make_signal_s10（F1、A10 換股日、位移 shift）。
本檔只新增：連續面板的載入、(規則 × 位移 × 本金) 的執行包裝、成績單統計。一律 python3 -I -B；不呼叫 git（快取先由 build_panel.py 放好）。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s75')
sys.path.insert(0, str(AG / 's71_ifrs_bridge_oos'))
import s71lib as S71                                         # noqa: E402
from s71lib import R, L, s20lib, s17lib, s10lib, s4lib, stlib, cslib   # noqa: E402
S71.TMP = TMP; R.TMP = TMP

START = '2008-01-02'
FIRST_MONTH = '2008-01'
VERSIONS = {'V_MASK': 'MASK', 'V_BRIDGE': 'B2'}
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
ACC = L.ACC                                                  # 正式引擎：2019 起 48.756…%、全期 34.397…%、最大回撤、381 筆
PERIODS = {'2008～2012': ('2008-01', '2012-12'), '2013～2015': ('2013-01', '2015-12'), '2016～2018': ('2016-01', '2018-12'),
           '2019～2023': ('2019-01', '2023-12'), '2024～2026': ('2024-01', '2026-10')}
BEARS = {'2008 金融海嘯': ('2008-01-01', '2008-12-31'), '2011': ('2011-01-01', '2011-12-31'), '2015': ('2015-01-01', '2015-12-31'),
         '2018': ('2018-01-01', '2018-12-31'), '2020-03': ('2020-01-01', '2020-06-30'), '2022': ('2022-01-01', '2022-12-31'),
         '2024-07～2025-04': ('2024-07-01', '2025-04-30')}   # 後四段＝s17lib.SEGMENTS 原定義；前三段＝整個日曆年


def load_cont():
    """連續面板＋股池（上市 & ~創新板）＋ 0050 停牌修正。"""
    R.point_cache(TMP / 'cache_cont')
    stlib.START = pd.Timestamp(START)
    assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK
    P = s20lib.load()
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    s17lib.e50_fix_suspension(P)
    for a in ('H', 'L', 'VOL', 'HF', 'VOLF', 'adjH', 'HI240', 'IDXC', 'NEARHI', 'F_HI'):   # 現行格用不到的陣列（省記憶體；同 run_official.py 的做法）
        if hasattr(P, a):
            delattr(P, a)
    return P, U


def surge(P, rule):
    F = S71.rev_rule(P, rule, FIRST_MONTH)
    return F, dict(S=F['S']['STD'], av=F['av'])


def signal(P, U, RT, shift=0):
    return s10lib.make_signal_s10(P, U, RT, shift=shift)


def sim(P, sg, U, cap=1e6):
    return L.sim(P, sg, U, cap=cap)


# ---------------------------------------------------------------- 成績單統計
def cagr(s, a=None, b=None):
    return stlib.cagr(s, a, b)


def dd_info(s):
    s = s.dropna(); dd = s / s.cummax() - 1
    tr = dd.idxmin(); pk = s.loc[:tr].idxmax()
    rec = s.loc[tr:][s.loc[tr:] >= s.loc[pk]]
    return dict(maxdd=float(dd.min()), peak=str(pk.date()), trough=str(tr.date()), recovered=str(rec.index[0].date()) if len(rec) else None)


def ann_vol(s):
    r = s.dropna().pct_change().dropna()
    return float(r.std(ddof=1) * math.sqrt(252))


def last_before(idx, d):
    return idx[idx < pd.Timestamp(d)][-1]


def yearly_table(nav, bench):
    """逐年報酬：年底（該年最後交易日）÷ 前一年底 − 1；第一年以起日為基底；最後一年到資料末日。"""
    out = []
    y1 = nav.groupby(nav.index.year).last(); y2 = bench.groupby(bench.index.year).last()
    p1, p2 = nav.iloc[0], bench.iloc[0]
    for y in y1.index:
        a, b = float(y1[y] / p1 - 1), float(y2[y] / p2 - 1)
        out.append(dict(year=int(y), strat=a, e0050=b, diff_pp=(a - b) * 100))
        p1, p2 = y1[y], y2[y]
    return out


def window_block(nav, bench, a, b):
    """a、b＝'YYYY-MM'（含）。基底＝a 月之前最後一個交易日（沒有就用第一天）。"""
    pa, pb = pd.Period(a, 'M'), pd.Period(b, 'M')
    idx = nav.index
    pre = idx[idx < pa.to_timestamp()]
    d0 = pre[-1] if len(pre) else idx[0]
    d1 = idx[idx <= pb.to_timestamp(how='end')][-1]
    s, q = nav.loc[d0:d1], bench.loc[d0:d1]
    rs, rb = R.monthly_returns(nav, bench)
    m = (rs.index >= pa) & (rs.index <= pb)
    blk = R.stats_block(rs[m], rb[m])
    return dict(base=str(d0.date()), end=str(d1.date()), cagr=cagr(s), cagr_0050=cagr(q), cagr_diff_pp=(cagr(s) - cagr(q)) * 100,
                total_ret=float(s.iloc[-1] / s.iloc[0] - 1), total_ret_0050=float(q.iloc[-1] / q.iloc[0] - 1),
                maxdd=R.maxdd(s), maxdd_0050=R.maxdd(q), months=blk['months'], ex_ann_pp=blk['ex_ann_pp'], te_ann_pp=blk['te_ann_pp'], IR=blk['IR'],
                t_nw6=blk['t_nw6'], t_iid=blk['t_iid'], beta=blk['beta'])


def bear_block(nav, bench):
    o = {}
    for k, (a, b) in BEARS.items():
        s, q = nav.loc[a:b].dropna(), bench.loc[a:b].dropna()
        ds, dq = dd_info(s), dd_info(q)
        o[k] = dict(window=[a, b], strat=ds['maxdd'], strat_peak=ds['peak'], strat_trough=ds['trough'], e0050=dq['maxdd'], e0050_peak=dq['peak'], e0050_trough=dq['trough'],
                    diff_pp=(ds['maxdd'] - dq['maxdd']) * 100)
    return o


def npos_series(P, lg):
    """每個交易日收盤的持股檔數（進場日起算、出場日不算；期末未平倉算到最後一天）。"""
    d = np.zeros(P.T + 1)
    for e, x, w in zip(lg['ent_t'].to_numpy(), lg['exit_t'].to_numpy(), lg['why'].to_numpy()):
        d[e] += 1; d[(x + 1) if w == 'open_end' else x] -= 1
    return pd.Series(np.cumsum(d)[:P.T], index=P.IDX)


def trade_block(P, lg, st, a=None, b=None):
    """a ≤ 進場日 ≤ b 的已平倉交易；平均持股檔數與閒置資金比例（現金＋0050 占淨值）取 a～b 的交易日平均。"""
    x = lg[lg['why'] != 'open_end']
    A_ = pd.Timestamp(a) if a else P.IDX[P.t0]; B_ = pd.Timestamp(b) if b else P.IDX[-1]
    x = x[(x['ent_date'] >= A_) & (x['ent_date'] <= B_)]
    n_open = int(((lg['why'] == 'open_end') & (lg['ent_date'] >= A_) & (lg['ent_date'] <= B_)).sum())
    npos = npos_series(P, lg).loc[A_:B_]; npos = npos[npos.index > P.IDX[P.t0]]
    sf = st['stockfrac'].loc[A_:B_].dropna()
    w = x[x['ret'] > 0]['ret']; l = x[x['ret'] <= 0]['ret']
    return dict(n_closed=int(len(x)), n_open_end=n_open, win_rate=float((x['ret'] > 0).mean()) if len(x) else None, mean_ret=float(x['ret'].mean()) if len(x) else None,
                median_ret=float(x['ret'].median()) if len(x) else None, avg_win=float(w.mean()) if len(w) else None, avg_loss=float(l.mean()) if len(l) else None,
                avg_hold_trading_days=float(x['hold_days'].mean()) if len(x) else None, median_hold_trading_days=float(x['hold_days'].median()) if len(x) else None,
                avg_npos=float(npos.mean()), avg_idle_frac=float(1 - sf.mean()), pnl_sum=float(x['pnl'].sum()))


def limit7_diag(P, lg):
    """2015-06-01 以前實際漲跌停 7%（引擎用 10%）：開盤 ≥ 7% 漲停價的買進筆數、開盤 ≤ 7% 跌停價的賣出筆數（不改結果）。"""
    cut = int(np.searchsorted(P.IDX.values, np.datetime64('2015-06-01'), side='left'))
    tick = cslib.tick
    def lim(t, j, up):
        ref = P.REF[t, j]
        if up:
            tk = tick(ref * 1.07); return np.floor(ref * 1.07 / tk + 1e-9) * tk
        tk = tick(ref * 0.93); return np.ceil(ref * 0.93 / tk - 1e-9) * tk
    nb = ns = 0; b_tot = s_tot = 0
    for j, e, x, w in zip(lg['j'].to_numpy(), lg['ent_t'].to_numpy(), lg['exit_t'].to_numpy(), lg['why'].to_numpy()):
        if e < cut:
            b_tot += 1; nb += int(P.O[e, j] >= lim(e, j, True) - 1e-9)
        if x < cut and w != 'open_end':
            s_tot += 1; ns += int(P.O[x, j] <= lim(x, j, False) + 1e-9)
    return dict(buys_before_2015_06=b_tot, buys_open_at_7pct_limit_up=nb, sells_before_2015_06=s_tot, sells_open_at_7pct_limit_down=ns)

#!/usr/bin/env python3
"""S81 共用：照 S76 診斷方向（名單放寬／空名額遞補／拉長持有）設計規則並驗證。共用引擎與既有研究一律不改；重用
- s75_continuous_2008_2026/s75lib.py：load_cont()（連續面板、股池＝上市 & ~創新板、0050 停牌修正）、surge()（V_BRIDGE＝B2 口徑 SURGE）；
- s47lib.simulate_s47（s10lib.simulate_s10 的字串替換副本，多 ndiv＝每檔除數；ndiv=10 與原引擎逐位相同）＋引擎原有的 slots（檔數上限）、requal（持有滿 N 期才檢查名單）掛鉤；
- r55lib 的月度統計。
本檔新增
- grid()：192 格（事前登記）；make_signal()：s10lib.make_signal_s10 的一般化（候選深度、動能門檻、續抱名次、最少持有期）；
- simulate()：每檔＝選股資金 ÷ 持股上限；odd=True 時把 cslib.ODD_DATE 暫時改到 2000-01-01（全期允許零股），跑完還原；
- 統計與挑選準則。一律 python3 -I -B；不呼叫 git。"""
import sys, math, itertools
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 's75_continuous_2008_2026'))
import s75lib as K                                           # noqa: E402
from s75lib import S71, R, L, s10lib, stlib, cslib           # noqa: E402

TRAIN = ('2008-01', '2015-12'); VALID = ('2016-01', '2026-10'); FULL = ('2008-01', '2026-10')
SUBS = {'2008-2012': ('2008-01', '2012-12'), '2013-2015': ('2013-01', '2015-12'), '2016-2018': ('2016-01', '2018-12'), '2019-2026': ('2019-01', '2026-10')}
WINDOWS = dict({'前段 2008-2015': TRAIN, '後段 2016-2026': VALID, '全期 2008-2026': FULL}, **SUBS)

# ---- 搜尋空間（事前寫死）
PAIRS = [(10, 10), (10, 20), (10, 30), (10, 50), (15, 20), (15, 30), (15, 50), (20, 20), (20, 30), (20, 50), (30, 30), (30, 50)]   # (持股上限, 候選深度)；深度 ≥ 上限
MOMS = {'M10': 0.10, 'M5': 0.05, 'M0': 0.0, 'NOMOM': -np.inf}      # 近 20 日漲幅門檻（NOMOM＝不設，依名次補滿）
MOM_ORDER = ['M10', 'M5', 'M0', 'NOMOM']
EXIT_OPTS = ['H1', 'H2', 'H3', 'K30', 'K50']                       # H1＝基準續抱名次、每期檢查；H2／H3＝持有滿 2／3 期才檢查名次（不合格＝破季線等仍照出）；K30／K50＝續抱名次放寬到前 30／50（每期檢查）
EXIT_ADJ = {'H1': ['H2', 'K30'], 'H2': ['H1', 'H3'], 'H3': ['H2'], 'K30': ['H1', 'K50'], 'K50': ['K30']}
CUR = 'C10|D10|M10|H1'


def base_keep(depth):
    return max(20, depth)                                          # 基準續抱名次：不小於候選深度（不然買進就不合格）


def exit_spec(depth, opt):
    """回傳 (續抱名次, 最少持有期)；不在搜尋空間回傳 None。"""
    bk = base_keep(depth)
    if opt == 'H1':
        return bk, 1
    if opt == 'H2':
        return bk, 2
    if opt == 'H3':
        return bk, 3
    k = int(opt[1:])
    return (k, 1) if k > bk else None


def grid():
    out = []
    for (cap, depth), mom, ex in itertools.product(PAIRS, MOM_ORDER, EXIT_OPTS):
        sp = exit_spec(depth, ex)
        if sp is None:
            continue
        out.append(dict(id=f'C{cap}|D{depth}|{mom}|{ex}', cap=cap, depth=depth, mom=mom, exit=ex, keep=sp[0], hold=sp[1]))
    return out


def neighbors(cells):
    """鄰近格：只有一個維度差一級（持股上限、候選深度、動能門檻、出場選項各自的相鄰值），而且在搜尋空間內。"""
    ids = {c['id'] for c in cells}
    caps = [10, 15, 20, 30]; deps = [10, 20, 30, 50]
    nb = {}
    for c in cells:
        cand = []
        i = caps.index(c['cap'])
        for i2 in (i - 1, i + 1):
            if 0 <= i2 < 4:
                cand.append(f"C{caps[i2]}|D{c['depth']}|{c['mom']}|{c['exit']}")
        i = deps.index(c['depth'])
        for i2 in (i - 1, i + 1):
            if 0 <= i2 < 4:
                cand.append(f"C{c['cap']}|D{deps[i2]}|{c['mom']}|{c['exit']}")
        i = MOM_ORDER.index(c['mom'])
        for i2 in (i - 1, i + 1):
            if 0 <= i2 < 4:
                cand.append(f"C{c['cap']}|D{c['depth']}|{MOM_ORDER[i2]}|{c['exit']}")
        for e2 in EXIT_ADJ[c['exit']]:
            cand.append(f"C{c['cap']}|D{c['depth']}|{c['mom']}|{e2}")
        nb[c['id']] = [x for x in cand if x in ids]
    return nb


# ---------------------------------------------------------------- 訊號
def make_signal(P, univ, RT, depth=10, mom_lo=0.10, keep=20, hold=1, shift=0):
    """s10lib.make_signal_s10 的一般化。每個訊號日：股池內、SURGE 有值、還原收盤 > 60 日線者依 SURGE 由大到小排名（o_ma）。
    進場名單＝前 depth 名中近 20 日漲幅 > mom_lo 的全部（依名次；引擎依空位由前往後買，買不到就換下一名）。
    hold＝1：X＝不在 o_ma 前 keep 名（＝現行寫法；depth=10、keep=20 時與 make_signal_s10 的 sig／X(E0) 逐格相同）。
    hold≥2：X＝不在 o_ma（破季線、失去分數或出股池，每期都檢查）；另給 RQ＝o_ma 前 keep 名，引擎 requal=hold：進場後經過的訊號日數 ≥ hold 才檢查是否在 RQ。"""
    S, av = RT['S'], RT['av']
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan); X = np.zeros((P.T, P.N), bool)
    info = []; RQ = {}
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((univ[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = univ[k] & np.isfinite(s)
        ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        pool = o_ma[:depth]
        tgt = pool[P.RET20[k][pool] > mom_lo]
        sig[k, tgt] = True; rank[k, tgt] = s[tgt]
        X[k, :] = True
        if hold == 1:
            X[k, o_ma[:keep]] = False
        else:
            X[k, o_ma] = False
            RQ[k] = set(int(j) for j in o_ma[:keep])
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok_ma.sum()), n_tgt=int(len(tgt)), targets=[str(P.COLS[j]) for j in tgt]))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, RQ=RQ, sigrows=sigrows, hold=hold)


def cell_signal(P, U, RT, c, shift=0):
    return make_signal(P, U, RT, depth=c['depth'], mom_lo=MOMS[c['mom']], keep=c['keep'], hold=c['hold'], shift=shift)


def simulate(P, sg, U, cap_n=10, odd=False, capital=1e6, cost='strict', block=None, force_requal=False):
    """W10_70 底倉不動（sat_w=0.7、±10 帶狀拉回、空位買 0050）；每檔目標＝選股資金 ÷ cap_n、最多 cap_n 檔。
    cap_n=10、hold=1 → 原引擎 simulate_s10(sizing='W10')；其餘 → s47lib.simulate_s47(ndiv=cap_n, slots=cap_n, requal=hold)。
    odd=True：全期允許零股（cslib.ODD_DATE 暫改 2000-01-01；成本仍是 strict，零股不另加價，同 S76 的做法）。"""
    kw = dict(cost=s10lib.COSTS[cost], capital=capital, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False, block=block)
    hold = sg.get('hold', 1)
    odd0 = cslib.ODD_DATE
    if odd:
        cslib.ODD_DATE = pd.Timestamp('2000-01-01')
    try:
        if cap_n == 10 and hold == 1 and not force_requal:
            return s10lib.simulate_s10(P, sg, U, **kw)
        return L.simulate_s47(P, sg, U, ndiv=cap_n, slots=cap_n, requal=(hold if (hold > 1 or force_requal) else None), **kw)
    finally:
        cslib.ODD_DATE = odd0


# ---------------------------------------------------------------- 統計
def win(x, a, b):
    return x[(x.index >= pd.Period(a, 'M')) & (x.index <= pd.Period(b, 'M'))]


def dd_window(nav, a, b):
    pa, pb = pd.Period(a, 'M'), pd.Period(b, 'M')
    idx = nav.index
    pre = idx[idx < pa.to_timestamp()]
    d0 = pre[-1] if len(pre) else idx[0]
    d1 = idx[idx <= pb.to_timestamp(how='end')][-1]
    return R.maxdd(nav.loc[d0:d1])


def yearly_ex(rs, rb):
    ys = (1 + rs).groupby(rs.index.year).prod() - 1; yb = (1 + rb).groupby(rb.index.year).prod() - 1
    return {int(y): float((ys[y] - yb[y]) * 100) for y in ys.index}


def block(rs, rb, nav, a, b):
    """月報酬統計（a、b＝'YYYY-MM'，含）：年化（月複利）、超額＝月超額平均×12、t（Newey-West 6 期）、資訊比率、追蹤誤差、區間內最大跌幅、逐年超額最差值。"""
    s, q = win(rs, a, b), win(rb, a, b)
    blk = R.stats_block(s, q)
    ye = yearly_ex(s, q)
    return dict(months=blk['months'], cagr=blk['cagr_strat'], cagr_0050=blk['cagr_0050'], cagr_diff_pp=blk['geo_diff_pp'], ex_ann_pp=blk['ex_ann_pp'], t_nw6=blk['t_nw6'], t_iid=blk['t_iid'],
                IR=blk['IR'], te_ann_pp=blk['te_ann_pp'], maxdd=dd_window(nav, a, b), worst_year_ex_pp=min(ye.values()), worst_year=min(ye, key=ye.get), yearly_ex_pp=ye)


def acct_block(P, lg, st, nav, a, b):
    """帳戶面：平均持股檔數、持股比例、交易數（進場日在區間內的已平倉交易）、換手（買進金額無法分段 → 用交易成本法：區間內進場交易的買進成本合計 ÷ 區間平均淨值 ÷ 年數）。"""
    A_ = pd.Period(a, 'M').to_timestamp(); B_ = pd.Period(b, 'M').to_timestamp(how='end')
    npos = K.npos_series(P, lg).loc[A_:B_]; npos = npos[npos.index > P.IDX[P.t0]]
    sf = st['stockfrac'].loc[A_:B_].dropna()
    x = lg[(lg['ent_date'] >= A_) & (lg['ent_date'] <= B_)]
    xc = x[x['why'] != 'open_end']
    yrs = len(sf) / 252.0
    navm = float(nav.loc[A_:B_].mean())
    hold_m = float(xc['hold_days'].mean()) if len(xc) else None
    return dict(avg_npos=float(npos.mean()), stockfrac=float(sf.mean()), n_closed=int(len(xc)), n_entries=int(len(x)), trades_per_year=float(len(x) / yrs),
                turnover_buy_per_year=float(x['cost'].sum() / navm / yrs), avg_hold_days=hold_m, win_rate=float((xc['ret'] > 0).mean()) if len(xc) else None,
                mean_ret=float(xc['ret'].mean()) if len(xc) else None, avg_ticket=float(x['cost'].mean()) if len(x) else None)


def pct_rank(v):
    """0～1 百分位名次（值大者高，平均名次法）。"""
    return (pd.Series(v).rank(method='average').to_numpy() - 1) / (len(v) - 1)

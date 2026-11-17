#!/usr/bin/env python3
"""S44 共用程式庫：「名次＋最低 SURGE 門檻」——F1（L_MA 依 SURGE 前 10 名中 RET20 > 10%）再加 SURGE 下限，能不能挑掉弱訊號？
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎）之上，本檔只新增：
- 開除權息修正：cslib.EXRIGHT_FIX = True（必須在 load() 之前；驗收 2019 起 48.99%、全期 34.54%、回撤 −32.11%、已平倉 381 筆）。
- f1_rows(P, U, RT, S)：F1 目標逐筆表（month、k、j、code、surge、當月全股池 SURGE 分位 pct、L_MA 第 10 名 SURGE、全股池檔數）。
  「當月全股池」＝訊號列 k 的股池（只上市、排除創新板，stlib.universe & ~INNOV）且當月 SURGE 有值——就是 make_signal_s10 排名前、
  還沒套季線（F_MA60）的 fin 集合；pct＝股池內 SURGE ≤ 該股的比例（0～1）；第 q 百分位用 numpy linear。
- keep(row, rule)：ABS＝surge ≥ θ；REL＝surge ≥ 當月股池第 q 百分位。
- apply_rule(P, S, F1, rule)：把 F1 目標中不合規則的 sig 拿掉（不遞補、不改出場、不改名額；空出的錢照引擎掃進 0050）。
- random_drop(P, S, removed_by_month, seed)：每個月隨機剔除「與規則同檔數」的 F1 目標（對照用）。
中間檔放 scratchpad/s44/cache。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S44 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s44/cache')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S44; s17lib.S17 = S44; s10lib.S10 = S44; s9lib.S9 = S44; s8lib.S8 = S44; s7lib.S7 = S44; s6lib.S6 = S44
s5lib.S5 = S44; s4lib.S4 = S44; s3lib.S3 = S44; s2lib.S2 = S44; stlib.S1 = S44; cslib.TMP = S44
cslib.EXRIGHT_FIX = True

ACC = dict(since2019=0.4899, full=0.3454, maxdd=-0.3211, n_closed=381)
TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
fresh_from = s20lib.fresh_from
outcomes = s20lib.outcomes
month_cluster_diff = s20lib.month_cluster_diff
THETAS = (1.5, 2.0, 2.5)

RULES = [
    dict(id='ABS_1.5', kind='ABS', theta=1.5, family='R1 絕對門檻',
         desc='F1 且 SURGE ≥ 1.5：F1 目標中 SURGE < 1.5 的不買（不遞補，空位照引擎放 0050）；出場 E0 不動。'),
    dict(id='ABS_2.0', kind='ABS', theta=2.0, family='R1 絕對門檻',
         desc='F1 且 SURGE ≥ 2.0：同上，門檻 2.0。'),
    dict(id='ABS_2.5', kind='ABS', theta=2.5, family='R1 絕對門檻',
         desc='F1 且 SURGE ≥ 2.5：同上，門檻 2.5（＝XQ 20 號 MinSurge 預設值）。'),
    dict(id='REL_P95', kind='REL', col='p95', q=95, family='R2 相對門檻（使用者例）',
         desc='F1 且 SURGE ≥ 當月全股池（只上市、排除創新板、SURGE 有值，未套季線）SURGE 第 95 百分位（numpy linear）。'),
    dict(id='REL_P975', kind='REL', col='p975', q=97.5, family='R3 相對門檻（描述後加嚴）',
         desc='F1 且 SURGE ≥ 當月全股池 SURGE 第 97.5 百分位。'),
]

DROP = ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq')


def load():
    S44.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True, '除權息修正沒有開'
    return P


def base(P):
    """股池 U 與現行 SURGE（STD）。"""
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    return U, RT


def f1_rows(P, U, RT, S):
    """S（make_signal_s10 輸出）→ F1 目標逐筆表＋每月摘要表。"""
    Sv = RT['S']
    rows, mons = [], []
    for x in S['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        s = Sv.loc[m].to_numpy()
        fin = U[k] & np.isfinite(s)
        pool = np.sort(s[fin])
        ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        s10 = float(s[o_ma[9]]) if len(o_ma) >= 10 else np.nan
        s20 = float(s[o_ma[19]]) if len(o_ma) >= 20 else np.nan
        q = {f'p{int(qq*10) if qq*10 % 10 else int(qq)}': float(np.quantile(pool, qq / 100)) for qq in (90, 95, 97.5, 99)}
        mons.append(dict(month=x['month'], k=k, sig_date=x['sig_date'], n_pool=int(fin.sum()), n_ma=int(ok_ma.sum()), s10=s10, s20=s20,
                         s1=float(s[o_ma[0]]) if len(o_ma) else np.nan, n_tgt=int(x['n_tgt']), pool_ge2=float((pool >= 2).mean()),
                         pool_ge25=float((pool >= 2.5).mean()), **q))
        for r, c in enumerate(x['targets']):
            j = P.ci[c]
            v = float(s[j])
            pct = float(np.searchsorted(pool, v, side='right') / len(pool))
            rank_ma = int(np.flatnonzero(o_ma == j)[0]) + 1
            rows.append(dict(month=x['month'], k=k, j=j, code=c, surge=v, pct=pct, rank_ma=rank_ma, n_pool=int(fin.sum()), **{kk: vv for kk, vv in q.items()}))
    df = pd.DataFrame(rows); mo = pd.DataFrame(mons)
    if len(df):
        df['mper'] = pd.PeriodIndex(df['month'], freq='M')
        df['sig_date'] = P.IDX[df['k'].to_numpy()]
    mo['mper'] = pd.PeriodIndex(mo['month'], freq='M')
    return df, mo


def keep(F1, rule):
    """F1 表 → bool Series（True＝留下）。rule＝None（現行）、dict(kind='ABS', theta)、dict(kind='REL', q, col)。"""
    if rule is None:
        return pd.Series(True, index=F1.index)
    if rule['kind'] == 'ABS':
        return F1['surge'] >= rule['theta']
    if rule['kind'] == 'REL':
        return F1['surge'] >= F1[rule['col']]
    raise ValueError(rule)


def apply_rule(P, S, F1, rule):
    """回傳 (S 的淺拷貝：sig／rank 複製後拿掉不合規則的 F1 目標, removed 清單 [(month, code)])。X、info 不動。"""
    S2 = dict(S)
    if rule is None:
        return S2, []
    kp = keep(F1, rule).to_numpy()
    sig = S['sig'].copy(); rank = S['rank'].copy()
    d = F1[~kp]
    sig[d['k'].to_numpy(), d['j'].to_numpy()] = False
    rank[d['k'].to_numpy(), d['j'].to_numpy()] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2, list(zip(d['month'], d['code']))


def random_drop(P, S, F1, n_by_month, seed):
    """每個月從 F1 目標中隨機剔除 n_by_month[month] 檔（同檔數對照）。"""
    rng = np.random.default_rng(seed)
    sig = S['sig'].copy(); rank = S['rank'].copy()
    for m, g in F1.groupby('month'):
        n = int(n_by_month.get(m, 0))
        if n <= 0:
            continue
        pick = rng.choice(len(g), size=n, replace=False)
        kk = g['k'].to_numpy()[pick]; jj = g['j'].to_numpy()[pick]
        sig[kk, jj] = False; rank[kk, jj] = np.nan
    S2 = dict(S); S2['sig'] = sig; S2['rank'] = rank
    return S2


def seg_dd(nav):
    return s17lib.seg_dd(nav)


def top5_share(lg, nav, cap):
    return s17lib.top5_share(lg, nav, cap)

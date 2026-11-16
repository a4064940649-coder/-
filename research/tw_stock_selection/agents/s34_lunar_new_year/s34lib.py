#!/usr/bin/env python3
"""S34 共用程式庫：農曆年（春節）讓 1、2 月營收年增金額失真——事前定義的校正能不能讓 SURGE 更好？
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎）之上，本檔只新增：
- 開除權息修正：cslib.EXRIGHT_FIX = True（必須在 load() 之前；現行驗收 2019 起 48.99%、全期 34.54%、回撤 −32.11%、已平倉 381 筆）。
- LNY：2012～2026 春節（農曆正月初一）日期表（寫死；來源見 LNY_SOURCE）。
- trading_days()：每個日曆月的交易日數，取自加權指數日資料（o_concentration_regime/raw/taiex_twii.csv，只讀；1999 起，涵蓋 2012～2014 的基期）。
- d_series(P, mode)：與 s17lib.rev_family 同一份月營收（同代碼優先上市、營收 ≤ 0 缺值、IFRS 斷點 2013 年缺值）算 D＝營收(m)−營收(m−12)，再依 mode 改 1、2（、3）月：
    'none'：不改（＝STD，驗收用）；
    'jf'  ：同一年 1 月與 2 月的 D 都換成兩者平均（＝「1＋2 月合計 − 去年 1＋2 月合計」÷ 2）；任一缺值 → 用另一個；
    'adj2'：1、2、3 月的 D 換成「當月與上月 D 的平均」（1 月＝(12 月＋1 月)÷2，2 月＝(1＋2)÷2，3 月＝(2＋3)÷2）；上月缺值 → 用當月；只用已公告資料；
    'wd'  ：1、2 月的 D 改成工作天數校正：營收(m) − 營收(m−12) × 交易日(m) ÷ 交易日(m−12)。
  D 序列整條替換（μ、σ 視窗裡的 1、2 月也用校正值），再照 STD 的 z 分數（s17lib.zscore，視窗 24、減均值、σ > 0）。
- surge_rule(P, rule)：回傳 dict(S, av)。'jf' 的 1 月營收月份整列設為缺值（1 月訊號時 2 月營收還沒公告）→ make_signal_s10 跳過該列＝「跳過換股」（不買也不賣，持股抱到 2 月營收的訊號日）。
- nobuy(S0, months)：現行訊號，但營收月份在 months 的訊號列不新買（sig 清空）；E0 出場照常。
- random_months()：隨機對照用，每年隨機挑一組相鄰兩個月（3～11 月起頭，不碰 12、1、2 月）套用同機制。
中間檔放 scratchpad/s34/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S34 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s34/cache')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S34; s17lib.S17 = S34; s10lib.S10 = S34; s9lib.S9 = S34; s8lib.S8 = S34; s7lib.S7 = S34; s6lib.S6 = S34
s5lib.S5 = S34; s4lib.S4 = S34; s3lib.S3 = S34; s2lib.S2 = S34; stlib.S1 = S34; cslib.TMP = S34
cslib.EXRIGHT_FIX = True

ACC = dict(since2019=0.4899, full=0.3454, maxdd=-0.3211, n_closed=381)
TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
SEGMENTS = s17lib.SEGMENTS
fresh_from = s20lib.fresh_from
TAIEX_CSV = AG / 'o_concentration_regime' / 'raw' / 'taiex_twii.csv'

LNY = {2012: '2012-01-23', 2013: '2013-02-10', 2014: '2014-01-31', 2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
       2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25', 2021: '2021-02-12', 2022: '2022-02-01', 2023: '2023-01-22',
       2024: '2024-02-10', 2025: '2025-01-29', 2026: '2026-02-17'}
LNY_SOURCE = ('農曆正月初一的國曆日期：交通部中央氣象署天文站「農曆與國曆對照」（公開曆書）、行政院人事行政總處各年「政府行政機關辦公日曆表」。'
              '日期為天文曆法決定，事前（前一年）即公告；本表只用來描述，不參與任何規則（規則只看日曆月份與交易日數）。'
              '休市期間另由加權指數交易日推得（1～2 月最長的無交易空檔），兩者互相核對。')

RULES = [
    dict(id='JF_SKIP', kind='surge', mode='jf',
         desc='(a) 1＋2 月合併：同一年 1、2 月的 D 都換成「(1＋2 月合計) − (去年 1＋2 月合計)」÷ 2（μ、σ 視窗內的 1、2 月也換）；'
              '2 月營收訊號（約 3/10）用校正後 SURGE 排名買賣；1 月營收訊號（約 2/10）跳過換股（不買也不賣，持股抱到 3 月訊號日）。'),
    dict(id='ADJ2', kind='surge', mode='adj2',
         desc='(b) 相鄰兩月平均：營收月份 1、2、3 月的 D 換成「當月與上月 D 的平均」（1 月＝(12＋1)÷2、2 月＝(1＋2)÷2、3 月＝(2＋3)÷2），'
              '只用已公告資料、不跳過換股；μ、σ 視窗內也換。'),
    dict(id='WD', kind='surge', mode='wd',
         desc='(d) 工作天數校正：營收月份 1、2 月的 D 改成「營收(m) − 營收(m−12) × 交易日(m) ÷ 交易日(m−12)」（交易日取自加權指數，事前公告的休市日曆）；μ、σ 視窗內也換；不跳過換股。'),
    dict(id='NOBUY', kind='nobuy', months=[1, 2],
         desc='(c) 1、2 月營收訊號不新買：營收月份 1、2 月的訊號列照常算 SURGE 與 E0 出場（跌出 L_MA 前 20 名賣出），但不買進新股票；空出的錢照引擎放 0050。'),
]


def load():
    S34.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True, '除權息修正沒有開'
    return P


def trading_days():
    d = pd.read_csv(TAIEX_CSV, usecols=['date'], parse_dates=['date'])['date']
    i = pd.DatetimeIndex(sorted(set(d)))
    return pd.Series(1, index=i).groupby(i.to_period('M')).sum()


def lny_closure():
    """每年 1～2 月加權指數最長的無交易空檔（休市日曆日數、最後交易日、開紅盤日）。"""
    d = pd.read_csv(TAIEX_CSV, usecols=['date'], parse_dates=['date'])['date']
    i = pd.DatetimeIndex(sorted(set(d)))
    out = {}
    for y in LNY:
        x = i[(i >= pd.Timestamp(f'{y}-01-01')) & (i <= pd.Timestamp(f'{y}-03-10'))]
        g = np.diff(x.values).astype('timedelta64[D]').astype(int)
        a = int(np.argmax(g))
        out[y] = dict(last_trade=str(x[a].date()), reopen=str(x[a + 1].date()), gap_days=int(g[a]))
    return out


def revenue(P):
    """同 s17lib.rev_family 的營收讀法。回傳 Rv（月份 Period × P.COLS，千元；≤ 0 缺值）。"""
    r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
    r['pri'] = (r['market'] != 'sii').astype(int)
    r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
    Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
    Rv.index = pd.PeriodIndex(Rv.index, freq='M')
    Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')).reindex(columns=P.COLS)
    return Rv.where(Rv > 0)


def d_series(Rv, mode, TD=None, months=(1, 2)):
    """D 序列（DataFrame）。months 只給隨機對照用（'jf'／'adj2' 的機制套到別的月份），規則本身一律用預設。"""
    D = Rv - Rv.shift(12)
    if mode == 'wd':
        td = TD.reindex(Rv.index).to_numpy(float)
        ratio = td / np.roll(td, 12)
        ratio[:12] = np.nan
        if isinstance(months[0], tuple):
            ym = set((y, a) for y, a, b in months) | set((y, b) for y, a, b in months)
            sel = np.array([(p.year, p.month) in ym for p in Rv.index])
        else:
            sel = np.isin(Rv.index.month, months)
        D2 = Rv - Rv.shift(12) * ratio[:, None]
        D = D.where(np.repeat(~sel[:, None], D.shape[1], axis=1), D2)
    D[D.index.year == 2013] = np.nan                       # IFRS 斷點（同 s17lib）
    A = D.to_numpy(float)
    if mode == 'none' or mode == 'wd':
        return pd.DataFrame(A, index=D.index, columns=D.columns)
    out = A.copy()
    per = list(D.index)
    if mode == 'jf':
        # months＝(a, b) 為一組（預設 1、2）：同一年 a、b 兩月都換成兩者平均；一個缺值用另一個
        if isinstance(months[0], tuple):
            pairs = list(months)
        else:
            pairs = [(y, months[0], months[1]) for y in sorted(set(D.index.year))]
        pos = {(p.year, p.month): i for i, p in enumerate(per)}
        for y, a, b in pairs:
            ia, ib = pos.get((y, a)), pos.get((y, b))
            if ia is None or ib is None:
                continue
            m = np.nanmean(np.vstack([A[ia], A[ib]]), axis=0)
            out[ia] = m; out[ib] = m
    elif mode == 'adj2':
        # 預設：1、2、3 月換成「當月與上月平均」；隨機對照 months＝[(y, a, b)] → b 月換成 (a＋b)÷2、a 月換成 (a−1＋a)÷2（同「春節所在附近」的三個月相對位置）
        if isinstance(months[0], tuple):
            tgt = []
            for y, a, b in months:
                tgt += [(y, a), (y, b), (y + (b == 12), b % 12 + 1)]
        else:
            tgt = [(p.year, p.month) for p in per if p.month in (1, 2, 3)]
        pos = {(p.year, p.month): i for i, p in enumerate(per)}
        for key in tgt:
            i = pos.get(key)
            if i is None or i == 0:
                continue
            prev = A[i - 1]
            out[i] = np.where(np.isfinite(prev), np.nanmean(np.vstack([A[i], prev]), axis=0), A[i])
    else:
        raise ValueError(mode)
    return pd.DataFrame(out, index=D.index, columns=D.columns)


def surge_from_D(D, av_months=None, P=None):
    Z, nz = s17lib.zscore(D, 24, 'mean')
    Z[Z.index < s17lib.FIRST_MONTH] = np.nan
    return Z, nz


def surge_rule(P, Rv, mode, TD=None, months=(1, 2), skip=None):
    """mode → dict(S, av, n_sigma0)。skip：要整列設缺值的營收月份（'jf' 預設＝1 月；隨機對照＝每組的第一個月）。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        D = d_series(Rv, mode, TD, months)
    Z, nz = surge_from_D(D)
    if mode == 'jf':
        if skip is None:
            skip = [p for p in Z.index if p.month == 1]
        Z.loc[[p for p in skip if p in Z.index]] = np.nan
    av = cslib.rev_avail_days(P, list(Z.index), True)
    return dict(S=Z, av=av, n_sigma0=nz)


def nobuy(S0, months):
    """現行訊號 S0（make_signal_s10），營收月份的月在 months（int 月或 Period 集合）的訊號列：sig 清空（不新買），X（E0）不動。"""
    S1 = dict(S0); sig = S0['sig'].copy(); rank = S0['rank'].copy(); removed = []
    for x in S0['info']:
        m = pd.Period(x['month'], 'M')
        hit = (m in months) if isinstance(next(iter(months)), pd.Period) else (m.month in months)
        if hit:
            k = x['k']
            removed += [(x['month'], c) for c in x['targets']]
            sig[k, :] = False; rank[k, :] = np.nan
    S1['sig'] = sig; S1['rank'] = rank
    return S1, removed


def build(P, U, Rv, TD, rule, shift=0, S_cur=None, RT_cur=None):
    """規則 → (S1（make_signal_s10 格式）, RT1)。rule＝None → 現行。"""
    if rule is None:
        return (S_cur if S_cur is not None else s10lib.make_signal_s10(P, U, RT_cur, shift=shift)), RT_cur
    if rule['kind'] == 'surge':
        RT1 = surge_rule(P, Rv, rule['mode'], TD)
        return s10lib.make_signal_s10(P, U, RT1, shift=shift), RT1
    if rule['kind'] == 'nobuy':
        S0 = S_cur if S_cur is not None else s10lib.make_signal_s10(P, U, RT_cur, shift=shift)
        return nobuy(S0, set(rule['months']))[0], RT_cur
    raise ValueError(rule['kind'])


def random_pairs(seed, years):
    """隨機對照：每年隨機挑相鄰兩月 (a, a+1)，a ∈ 3～10（第三個月 a+2 ≤ 12，不碰 12、1、2 月的春節窗）。回傳 [(y, a, a+1)]。"""
    rng = np.random.default_rng(seed)
    return [(int(y), int(a), int(a) + 1) for y, a in zip(years, rng.integers(3, 11, size=len(years)))]


def list_diff(S0, S1):
    """兩個訊號的逐月買進名單差異：{month: (orig, new)}（只列有差的月份），以及每月是否存在。"""
    a = {x['month']: x['targets'] for x in S0['info']}
    b = {x['month']: x['targets'] for x in S1['info']}
    out = {}
    for m in sorted(set(a) | set(b)):
        if a.get(m) != b.get(m):
            out[m] = (a.get(m), b.get(m))
    return out


def seg_dd(nav):
    return s17lib.seg_dd(nav)


def top5_share(lg, nav, cap):
    return s17lib.top5_share(lg, nav, cap)

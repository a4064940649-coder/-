#!/usr/bin/env python3
"""o_rev_event_trigger 核心（後備題：J0／J+1 固定換股日正式化與前視清理）。

資料一律用 V4 凍結版：只讀 import o_v4_revenue_patch_official/v4lib.py（它再只讀 import
o_disposal_rejudge/drj_core → main_v2/v2_core → independent_audit/ia_core，以及 o_infra_v3_rebuild/o3_core）。
策略規格（股池、分數、同分序、n10、處置成交日口徑、fin2、KY/DR、P、X、成本）完全沿用 v4lib.V4，本模組只改兩件事：
  1) 換股訊號日：錨定月 m（營收月 m−1）的法定期限 D_m＝第 m 月 10 日起第一個交易日（加權指數日曆）；
     特例 2019-02 → 2019-02-15、2023-04 → 2023-04-12。J{j}＝D_m 之後第 j 個交易日（J0＝D_m 當天）。
     錨定月集合＝V4 官方訊號日的月份（v2 的營收可用條件）；訊號日不在面板或不是 GOOD 日、或次一面板日不是 GOOD 日 → 該月跳過（續抱）。
     成交日＝訊號日的次一面板交易日開盤（＝v4lib 的 e＝t+1）。
     J0N（「期限日盤後完成率 ≥ 99% 的最早日」）：訊號日＝D_m，價格腿用 D_m 收盤，營收腿用 D_m 當晚（次一交易日開盤前）已公告的營收。
  2) 營收可得性（AB）：
     day 模式（訊號在收盤時算）：s < D → 全部未公告；s = D → 依 rev_announce 產業延後機率 p（金融 p＝1）抽樣；
        2020-02 錨定月（疫情可延到 2/15）：D < s ≤ 2020-02-14 也抽樣；s > D → 全部已公告。
     night 模式（J0N）：s = D → 依「期限日當晚仍未公告」機率 q 抽樣（q 由 2026-09 首見時間估計，見 explore）；2020-02 同 day 模式抽 p。
     2026-01～08 營收：INS15 名單（含非金融 2207、2905）期限 15 日：s < D15 → 未公告；s = D15 → day 抽 p、night 視為已公告。
     ABN（只當迭代版本）：day 模式下，新聞日期（鉅亨網，o_v4 raw_news_mentions.parquet）早於訊號日者 → 確定已公告；
        其他股票改用 p_cond＝min(1, p／(1−f))，f＝當月有營收資料的非金融股中「新聞確定早公告」的比例（保持未公告期望值不變）。
未公告者照 v4lib：改用 m−2..m−4 的三月平均年增率（＝最近已公告值）。
"""
import sys, json, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
V4DIR = AG / 'o_v4_revenue_patch_official'
sys.path.insert(0, str(V4DIR))
import numpy as np, pandas as pd
import v4lib as L            # noqa: E402（只讀 import）
sys.path.pop(0)
K, V, R = L.K, L.V, L.R
IDX, COLS, N, CI = K.IDX, K.COLS, K.N, K.CI
DATA = HERE / 'data'

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

# ---------------- 日曆 ----------------
_TX = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']
TCAL = pd.DatetimeIndex(sorted(set(_TX)))
IPOS = {d: i for i, d in enumerate(IDX)}
SPECIAL_DL = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12')}

def first_td(ts):
    k = TCAL.searchsorted(ts); return TCAL[k] if k < len(TCAL) else None

def deadline(m):
    s = str(m)
    if s in SPECIAL_DL: return first_td(SPECIAL_DL[s])
    return first_td(pd.Timestamp(m.year, m.month, 10))

def deadline15(m): return first_td(pd.Timestamp(m.year, m.month, 15))

def day_for(m, j):
    D = deadline(m)
    if D is None: return None
    k = TCAL.get_loc(D) + j
    if k < 0 or k >= len(TCAL): return None
    s = TCAL[k]
    if s not in IPOS: return None
    t = IPOS[s]
    if not K.GOOD[t] or t + 1 >= len(IDX) or not K.GOOD[t + 1]: return None
    return s

# ---------------- 資料（V4 凍結）----------------
_ST = {}
def setup():
    if _ST: return _ST
    fz = L.check_freeze(require_rfix=True)
    XR = L.setup_px()
    YFX, Y5F = L.load_rfix()
    DMT = L.disp_mask_fn('trade')
    _ST.update(fz=fz, XR=XR, YFX=YFX, Y5F=Y5F, DMT=DMT)
    S0 = L.V4(YFX, Y5F, DMT)
    _ST['off_days'] = list(S0.sig_days); _ST['off_end'] = S0.end_day
    _ST['anchors'] = [pd.Period(d, 'M') for d in S0.sig_days]
    return _ST

def version_days(j):
    """J{j}：[(m, d)] 與評價終點（2026-10 錨定月的 J{j} 日；超出面板 → 面板最後一日）。"""
    st = setup()
    out = [(m, day_for(m, j)) for m in st['anchors']]
    out = [(m, d) for m, d in out if d is not None]
    de = day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > IDX[-1]: de = IDX[-1]
    out = [(m, d) for m, d in out if d < de]
    return out, de

def official_days():
    st = setup()
    return [(pd.Period(d, 'M'), d) for d in st['off_days']], st['off_end']

def jcase_official():
    """官方日（每月第一個日 ≥ 11 的面板日）屬於 J0 還是 J+1（或其他）。"""
    days, _ = official_days()
    cnt = {}
    for m, d in days:
        D = deadline(m)
        j = TCAL.get_loc(d) - TCAL.get_loc(D) if d in TCAL else None
        cnt[j] = cnt.get(j, 0) + 1
    return cnt

# ---------------- AB 可得性 ----------------
PCOL = V.PCOL.copy(); IS15 = V.IS15.copy()
def _rng(seed, d, tag=0): return np.random.default_rng([int(seed), int(d.strftime('%Y%m%d')), int(tag)])

_NEWS = {}
def news_first():
    """{(code, 營收月 str): 最早新聞日期}（data/ann_news.csv）。"""
    if 'x' not in _NEWS:
        f = DATA / 'ann_news.csv'
        A = pd.read_csv(f, dtype={'code': str, 'rev_month': str}, parse_dates=['first_news'])
        _NEWS['x'] = {(c, m): d for c, m, d in zip(A.code, A.rev_month, A.first_news)}
        _NEWS['df'] = A
    return _NEWS['x']

def make_unavail(mode='day', seed=0, q=None, news=False, kind='mc'):
    """回傳 callable(d)→bool 陣列（True＝m−1 月營收在可用時點前未確定公告）。
    mode：'day'（收盤算訊號）、'night'（J0N：期限日當晚）、'none'（全部已公告＝非 AB）。kind：'mc' 抽樣、'worst' 期限日全部未公告。"""
    cache = {}
    nf = news_first() if news else None
    def U(d):
        if d in cache: return cache[d]
        u = np.zeros(N, bool)
        if mode == 'none': cache[d] = u; return u
        m = pd.Period(d, 'M'); D = deadline(m); rm = str(m - 1)
        covid = (str(m) == '2020-02' and D < d <= pd.Timestamp('2020-02-14'))
        if d < D:
            u[:] = True
        elif d == D or covid:
            if mode == 'night' and not covid:
                u = (_rng(seed, d, 7).random(N) < q) if kind == 'mc' else np.zeros(N, bool)
            else:
                if kind == 'worst': u[:] = True
                else:
                    p = PCOL
                    if nf is not None:
                        known = np.array([(c, rm) in nf and nf[(c, rm)] < d for c in COLS])
                        base = np.isfinite(V.K.yoy3_for_month(setup()['YFX'] if d < V.SWITCH else setup()['Y5F'], m)) & ~V.FINMASK
                        f = known[base].mean() if base.any() else 0.0
                        p = np.minimum(1.0, PCOL / max(1e-9, 1 - f))
                        u = (_rng(seed, d).random(N) < p) & ~known
                    else:
                        u = _rng(seed, d).random(N) < p
        if '2026-01' <= rm <= '2026-08':
            D15 = deadline15(m)
            if d < D15: u |= IS15
            elif d == D15 and mode != 'night': u |= IS15 & (_rng(seed, d, 15).random(N) < PCOL)
        cache[d] = u
        return u
    return U

# ---------------- 策略物件 ----------------
class V4J(L.V4):
    """v4lib.V4，訊號日改成指定的日曆。"""
    def __init__(self, days, end_day, unavail=None, **kw):
        st = setup()
        super().__init__(st['YFX'], st['Y5F'], st['DMT'], unavail=unavail, **kw)
        self.sig_days = [d for _, d in days]; self.end_day = end_day
        self.anchor = {d: m for m, d in days}

def build(days, end_day, unavail=None):
    S = V4J(days, end_day, unavail=unavail)
    t0, t1, sd = S.bounds()
    pl = S.plan(10, 'official')
    return S, pl, t0, t1, sd

def sim(pl, t0, t1, cost='normal'):
    return L.sim_plan(pl, t0, t1, cost, setup()['XR'])

def ew_nav(pl, t0, t1):
    nav, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    return nav

def ab_navs(days, end_day, mode='day', seeds=range(40), q=None, news=False, costs=('normal',), keep_seed=True, t1_cap=None):
    """AB：每個 seed 一張名單與一條淨值；回傳 dict(mean={cost: nav}, seeds={cost: [nav...]}, ew=nav, plans=[...], nold=float)。
    mean＝各 seed 日報酬平均後連乘（同 run_v4）。"""
    rets = {c: [] for c in costs}; sn = {c: [] for c in costs}; ewr = []; pls = []; nold = []
    t0 = t1 = None
    for s in seeds:
        S, pl, a, b, sd = build(days, end_day, make_unavail(mode, s, q=q, news=news))
        if t1_cap is not None: b = min(b, t1_cap)
        t0, t1 = a, b
        for c in costs:
            nav, lg = sim(pl, a, b, c); rets[c].append(nav.pct_change())
            if keep_seed: sn[c].append(nav)
        ew = ew_nav(pl, a, b); ewr.append(ew.pct_change()); pls.append(pl)
        nold.append(np.mean([S.cross_section(d)['nold'] for d in S.sig_days if IDX.get_loc(d) <= b]))
    mean_nav = lambda X: (1 + pd.concat(X, axis=1).mean(axis=1).fillna(0)).cumprod()
    return dict(mean={c: mean_nav(rets[c]) for c in costs}, seeds=sn, ew=mean_nav(ewr), plans=pls, nold=float(np.mean(nold)), t0=t0, t1=t1)

# ---------------- 統計 ----------------
E50 = K.E50
P = lambda s: pd.Period(s, 'M')
PERIODS = {'SEL': (P('2006-04'), P('2018-12')), 'VAL': (P('2019-01'), P('2023-12')), 'HOLD': (P('2024-01'), P('2026-10')),
           'POST': (P('2019-01'), P('2026-10')), 'ALL': (P('2006-04'), P('2026-10')),
           'S1': (P('2006-04'), P('2013-12')), 'S2': (P('2014-01'), P('2020-12')), 'S3': (P('2021-03'), P('2026-10')),
           'S4': (P('2024-01'), P('2026-10')), 'S2a': (P('2014-01'), P('2018-12')), 'S11_18': (P('2011-01'), P('2018-12'))}
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def windows(sig_days, end_day):
    """[(m, d0, d1)]：m＝d0 的月份（＝錨定月）。"""
    sd = list(sig_days) + [end_day]
    return [(pd.Period(sd[i], 'M'), sd[i], sd[i + 1]) for i in range(len(sd) - 1)]

def wret(nav, W):
    a = np.array([nav.get(w[1], np.nan) for w in W]); b = np.array([nav.get(w[2], np.nan) for w in W])
    return pd.Series(b / a - 1, index=pd.PeriodIndex([w[0] for w in W], freq='M'))

def selp(s, key):
    a, b = PERIODS[key]
    return s[(s.index >= a) & (s.index <= b)]

def cal_ret(nav):
    """日曆月簡單報酬（上月最後一個淨值日收盤 → 本月最後一個淨值日收盤）；只取完整月：
    第一個月＝淨值起點月的下一個月；最後一個月＝淨值終點是該月最後一個面板交易日時含該月，否則到前一個月。"""
    pm = nav.index.to_period('M')
    me = nav.groupby(pm).last()
    r = me.pct_change()
    first_full = pm[0] + 1
    lastm = pm[-1]
    last_panel_day = IDX[IDX.to_period('M') == lastm][-1]
    last_full = lastm if nav.index[-1] == last_panel_day else lastm - 1
    r = r[(r.index >= first_full) & (r.index <= last_full)]
    return r.dropna()

def cal_ex(nav, bench=None):
    bench = E50 if bench is None else bench
    b = bench.reindex(nav.index)
    return (cal_ret(nav) - cal_ret(b)).dropna()

def span_cagr(nav, W, key):
    w = [x for x in W if PERIODS[key][0] <= x[0] <= PERIODS[key][1]]
    if not w: return np.nan, np.nan
    a, b = w[0][1], w[-1][2]
    s = nav[(nav.index >= a) & (nav.index <= b)]
    yrs = (b - a).days / 365.25
    eq = s / s.iloc[0]
    return float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, float((eq / eq.cummax() - 1).min()) * 100

def block(nav, W, keys=('ALL', 'SEL'), ew=None):
    e50 = E50.reindex(nav.index)
    ex = wret(nav, W) - wret(e50, W)
    cx = cal_ex(nav)
    o = {}
    for k in keys:
        x = selp(ex, k).dropna(); c = selp(cx, k)
        cg, mdd = span_cagr(nav, W, k); cb, mdb = span_cagr(e50, W, k)
        d = dict(CAGR=cg, maxDD=mdd, E50_CAGR=cb, E50_maxDD=mdb, ex_mean_pp=float(x.mean() * 100), ex_sd_pp=float(x.std(ddof=1) * 100),
                 t_0050=tstat(x), n=int(len(x)), t_0050_cal=tstat(c), cal_mean_pp=float(c.mean() * 100), n_cal=int(len(c)))
        if ew is not None:
            y = selp(wret(nav, W) - wret(ew.reindex(nav.index), W), k).dropna(); d['t_EW'] = tstat(y); d['EW_CAGR'] = span_cagr(ew, W, k)[0]
        o[k] = d
    return o

def paired_cal(nav_a, nav_b, key):
    """日曆月配對差（a−b）。"""
    i = nav_a.index.intersection(nav_b.index)
    d = selp((cal_ret(nav_a.reindex(i)) - cal_ret(nav_b.reindex(i))).dropna(), key)
    return _pd(d)

def paired_win(nav_a, Wa, nav_b, Wb, key):
    """錨定月配對差：各自的訊號窗報酬，依錨定月對齊（同 o_stagger_tranche）。"""
    a = wret(nav_a, Wa); b = wret(nav_b, Wb)
    d = selp((a - b.reindex(a.index)).dropna(), key)
    return _pd(d)

def _pd(d):
    nz = d[d.abs() > 1e-12]
    out = dict(mean_pp=float(d.mean() * 100), t=tstat(d), n=int(len(d)), sd_pp=float(d.std(ddof=1) * 100) if len(d) > 1 else np.nan,
               n_nonzero=int(len(nz)), MDE80_pp=float(2.8 * d.std(ddof=1) / math.sqrt(len(d)) * 100) if len(d) > 2 else np.nan)
    if len(d) > 5:
        x = d.sort_values(ascending=False)
        out['drop_best1_t'] = tstat(x.iloc[1:]); out['drop_best3_t'] = tstat(x.iloc[3:])
        out['drop_best3_mean_pp'] = float(x.iloc[3:].mean() * 100)
    return out

def endpoint(nav, W, key='ALL'):
    e50 = E50.reindex(nav.index)
    ex = selp(wret(nav, W) - wret(e50, W), key).dropna()
    o = {'t': tstat(ex), 'n': int(len(ex)), 'drop_last1': tstat(ex.iloc[:-1])}
    xs = ex.sort_values(ascending=False)
    for q in (1, 3, 5, 10): o[f'drop_best{q}'] = tstat(xs.iloc[q:])
    yr = ex.groupby(ex.index.year).sum(); by = int(yr.idxmax())
    o['best_year'] = by; o['drop_best_year'] = tstat(ex[ex.index.year != by])
    b3 = list(yr.sort_values(ascending=False).index[:3]); o['best3_years'] = [int(x) for x in b3]
    o['drop_best3_years'] = tstat(ex[~ex.index.year.isin(b3)])
    o['drop_first_year_2006'] = tstat(ex[ex.index.year != 2006])
    o['start_2007'] = tstat(ex[ex.index >= P('2007-01')]); o['start_2008'] = tstat(ex[ex.index >= P('2008-01')])
    o['best_month'] = str(xs.index[0]); o['best_month_pp'] = float(xs.iloc[0] * 100)
    c = selp(cal_ex(nav), key); o['t_cal'] = tstat(c)
    cs = c.sort_values(ascending=False); o['t_cal_drop_best1'] = tstat(cs.iloc[1:])
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); r50 = E50.reindex(nav.index).pct_change()
    return {int(y): {'strat': float((1 + g).prod() - 1) * 100, '0050': float((1 + r50[g.index].fillna(0)).prod() - 1) * 100} for y, g in r.groupby(r.index.year)}

def p_one(t): return 0.5 * math.erfc(t / math.sqrt(2)) if np.isfinite(t) else float('nan')

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, float): return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Period)): return str(o)
    return o

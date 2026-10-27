#!/usr/bin/env python3
"""main_v2 核心：在 independent_audit/ia_core.py（只讀 import）之上，實作正式版 v2 的規格。

v2 自己的部分：
  * 滾動窗三版 strict／gapfix／lenient（gapfix＝只豁免面板缺口日的缺值）
  * 營收來源依訊號日切換（< 2021-03-01 長期營收；之後 rev_5y yoy）
  * 同分次排序（營收 3 月年增率高者 → 代碼小者）與隨機同分
  * 金融保險業排除、只用已公告營收（AB，照 rev_announce 的期限日曆與延後機率）
  * 終點：2026-10-12（不執行 10 月換股）或最新資料日（執行 10 月換股）
  * 日曆月 t、最大單股貢獻
"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
IA = HERE.parent / 'independent_audit'
sys.path.insert(0, str(IA))
import numpy as np, pandas as pd
import ia_core as K
sys.path.pop(0)

IDX, COLS, N = K.IDX, K.COLS, K.N
SWITCH = pd.Timestamp('2021-03-01')
END1012 = pd.Timestamp('2026-10-12')
FEE = K.FEE

# ---------------- 滾動窗 ----------------
EXEMPT_DAYS = [pd.Timestamp('2017-08-03'), pd.Timestamp('2018-07-12')]
EXEMPT_RANGE = (pd.Timestamp('2024-02-15'), pd.Timestamp('2024-03-15'))
_RS = {}

def rolling_signals(mode):
    if mode in _RS: return _RS[mode]
    g = np.where(K.GOOD)[0]                      # 剔除有值檔數 < 300 的日子（2018-01-13）
    A = K.MYADJ.to_numpy()[g]; Vv = K.V.to_numpy()[g]
    Ad, Vd = pd.DataFrame(A), pd.DataFrame(Vv)
    if mode == 'strict':
        hi = Ad.rolling(252, min_periods=252).max().to_numpy()
        v60 = Vd.rolling(60, min_periods=60).mean().to_numpy()
    elif mode == 'lenient':
        hi = Ad.rolling(252, min_periods=200).max().to_numpy()
        v60 = Vd.rolling(60, min_periods=50).mean().to_numpy()
    elif mode == 'gapfix':
        dg = IDX[g]
        ex = np.isin(dg, pd.DatetimeIndex(EXEMPT_DAYS)) | ((dg >= EXEMPT_RANGE[0]) & (dg <= EXEMPT_RANGE[1]))
        missA = pd.DataFrame(np.isnan(A) & ~ex[:, None]).astype(np.int16)
        missV = pd.DataFrame(np.isnan(Vv) & ~ex[:, None]).astype(np.int16)
        okA = (missA.rolling(252, min_periods=252).sum().to_numpy() == 0)
        okV = (missV.rolling(60, min_periods=60).sum().to_numpy() == 0)
        hi = np.where(okA, Ad.rolling(252, min_periods=1).max().to_numpy(), np.nan)
        v60 = np.where(okV, Vd.rolling(60, min_periods=1).mean().to_numpy(), np.nan)
    else:
        raise ValueError(mode)
    with np.errstate(invalid='ignore', divide='ignore'):
        s52 = A / hi
    S52 = np.full((len(IDX), N), np.nan); V60 = np.full((len(IDX), N), np.nan)
    S52[g] = s52; V60[g] = v60
    _RS[mode] = (S52, V60)
    return _RS[mode]

# ---------------- 營收 ----------------
def load_revenue():
    YL = K.yoy_from_long()
    Y5 = K.yoy_from_rev5y()
    # 「到最新」版本：2026-09 由 rev_hist 補（MOPS 公告值，與 rev_5y 同口徑）
    rh = pd.read_csv(K.X / 'rev_hist.csv.gz', dtype={'code': str})
    rh['p'] = pd.PeriodIndex(rh['ym'], freq='M')
    rh = rh.drop_duplicates(['p', 'code'], keep='last').pivot(index='p', columns='code', values='yoy')
    Y5f = Y5.copy()
    allc = Y5f.columns.union(rh.columns)
    Y5f = Y5f.reindex(columns=allc)
    for p in rh.index:
        if p not in Y5f.index: Y5f.loc[p] = np.nan
        row = Y5f.loc[p]
        add = rh.loc[p].reindex(allc)
        Y5f.loc[p] = row.where(row.notna(), add)
    Y5f = Y5f.sort_index()
    return YL, Y5, Y5f

def last_full_month(Y):
    cnt = Y.notna().sum(axis=1)
    return cnt[cnt >= 0.8 * cnt.median()].index.max()

# ---------------- 金融保險業 ----------------
SL = pd.read_csv(K.X.parent / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
FIN_IND = '金融保險業'
FIN = set(SL.index[SL['industry'] == FIN_IND]) | {c for c in COLS if c[:2] == '28' or c in ('5876', '5880')}
FINMASK = np.array([c in FIN for c in COLS])

# ---------------- AB：已公告營收 ----------------
RA = HERE.parent / 'rev_announce'
PDL = json.loads((RA / 'p_delay.json').read_text(encoding='utf-8'))
INS15 = set(PDL['INS15'])
FS = pd.read_csv(RA / 'firstseen_2026-09.csv', dtype=str, index_col=0)
ANN_OK_2609 = set(FS.index[FS['cat'] == 'A_before'])
IND = [SL['industry'].get(c) if isinstance(SL['industry'].get(c), str) else '（stock_list 無）' for c in COLS]
PCOL = np.array([1.0 if c in FIN else PDL['p_by_industry'].get(i, PDL['p_overall']) for c, i in zip(COLS, IND)])
IS15 = np.array([c in INS15 for c in COLS])
_tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']
TCAL = pd.DatetimeIndex(sorted(set(_tx) | set(IDX)))

def deadline_case(d):
    rm = pd.Period(d, 'M') - 1
    if str(rm) in ('2019-01', '2023-03'): return 'D>S'
    if str(rm) == '2020-01': return 'D=S'
    D = TCAL[TCAL >= pd.Timestamp(d.year, d.month, 10)][0]
    return 'D<S' if D < d else ('D=S' if D == d else 'D>S')

def make_unavail(mode, seed=None):
    """mode：'mc'（延後機率抽樣）或 'worst'（期限＝訊號日全部未公告）。回傳 callable(d)→bool 陣列（True＝未公告）。"""
    rng = np.random.default_rng(seed) if seed is not None else None
    cache = {}
    def U(d):
        if d in cache: return cache[d]
        rm = str(pd.Period(d, 'M') - 1); c = deadline_case(d)
        u = np.zeros(N, bool)
        if rm == '2026-09':
            u = ~np.array([x in ANN_OK_2609 for x in COLS])
        elif c == 'D>S':
            u[:] = True
        elif c == 'D=S':
            if mode == 'worst': u[:] = True
            else: u = rng.random(N) < PCOL
        if '2026-01' <= rm <= '2026-08': u |= IS15
        cache[d] = u
        return u
    return U

# ---------------- 策略 ----------------
class V2:
    def __init__(self, sigmode, YL, Y5, latest=False, exfin=False, unavail=None, start='2005-01-01', rev_min_names=100):
        self.YL, self.Y5 = YL, Y5
        self.exfin, self.unavail = exfin, unavail
        self.S52, self.V60 = rolling_signals(sigmode)
        self.sigmode = sigmode
        lm5 = Y5.index.max() if latest else last_full_month(Y5)
        lmL = last_full_month(YL)
        days = K.rebalance_days(start)
        keep = []
        for d in days:
            m = pd.Period(d, 'M')
            Y, lm = (self.Y5, lm5) if d >= SWITCH else (self.YL, lmL)
            if (m - 1) > lm: continue
            if np.isfinite(K.yoy3_for_month(Y, m)).sum() > rev_min_names: keep.append(d)
        if latest:
            self.sig_days = keep; self.end_day = IDX[-1]
        else:
            keep = [d for d in keep if d < END1012]
            self.sig_days = keep; self.end_day = END1012
        self._cs = {}

    def y3(self, d, back=0):
        Y = self.Y5 if d >= SWITCH else self.YL
        y = K.yoy3_for_month(Y, pd.Period(d, 'M') - back)
        if self.exfin: y = np.where(FINMASK, np.nan, y)
        return y

    def cross_section(self, d):
        if d in self._cs: return self._cs[d]
        t = IDX.get_loc(d); e = t + 1
        v60 = self.V60[t]; px = K.Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True
        top &= base
        y3 = self.y3(d)
        nold = 0
        if self.unavail is not None:
            U = self.unavail(d)
            if U is not None and U.any():
                y3 = np.where(U, self.y3(d, 1), y3)
                nold = int((U & top).sum())
        s52 = self.S52[t]
        el = top & np.isfinite(y3) & np.isfinite(s52)
        el &= ~K.disposal_mask(d)
        el &= np.isfinite(K.On[e]) & K.TRADED[e]
        def pr(x):
            return pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()
        sc = 0.5 * pr(s52) + 0.5 * pr(y3)
        sc[~el] = np.nan
        cs = dict(t=t, e=e, el=el, sc=sc, s52=s52, y3=y3, nold=nold)
        self._cs[d] = cs
        return cs

    @staticmethod
    def order(cs, tie='official'):
        sc = np.round(np.nan_to_num(cs['sc'], nan=-9), 10)
        col = np.arange(N)
        if tie == 'official':
            keys = (col, -np.nan_to_num(cs['y3'], nan=-1e18), -sc)
        elif tie.startswith('rand:'):
            rng = np.random.default_rng(int(tie[5:]) * 100003 + cs['t']); keys = (rng.permutation(N), -sc)
        elif tie == 'code':
            keys = (col, -sc)
        else:
            raise ValueError(tie)
        return np.lexsort(keys)

    def plan(self, n, tie='official'):
        out = []; held = set(); nblk = 0
        for d in self.sig_days:
            cs = self.cross_section(d)
            el = cs['el']
            if el.sum() < 50: continue
            e = cs['e']; gu = K.gap_up_open(e)
            cand = self.order(cs, tie)[:int(el.sum())]
            pick = []
            for j in cand:
                if gu[j] and j not in held: nblk += 1; continue
                pick.append(j)
                if len(pick) == n: break
            out.append((e, np.array(pick, int), np.where(el)[0], d))
            held = set(pick)
        self.nblk = nblk
        return out

    def bounds(self):
        return IDX.get_loc(self.sig_days[0]), IDX.get_loc(self.end_day), self.sig_days + [self.end_day]

# ---------------- 統計 ----------------
# (訊號窗起點 a, 終點 b, 日頻起點 a_daily)：S3 的日頻從 2021-03 訊號日收盤之後的第一筆報酬起算
SEGS = {'S1': ('2006-04-01', '2013-12-31', '2006-04-01'), 'S2': ('2014-01-01', '2020-12-31', '2014-01-01'),
        'S3': ('2021-03-01', '2026-10-12', '2021-03-12'), 'S4': ('2024-01-01', '2026-10-12', '2024-01-01'),
        'P1': ('2006-04-01', '2020-12-31', '2006-04-01'), 'ALL': ('2006-04-01', '2026-10-12', '2006-04-01')}
SEGS_LATEST = {'S3': ('2021-03-01', '2026-10-22', '2021-03-12'), 'S4': ('2024-01-01', '2026-10-22', '2024-01-01'),
               'ALL': ('2006-04-01', '2026-10-22', '2006-04-01')}

def cal_t(nav, ew, ad, b):
    me = nav.groupby(nav.index.to_period('M')).last()
    be = K.E50.reindex(nav.index).groupby(nav.index.to_period('M')).last()
    ee = ew.groupby(ew.index.to_period('M')).last()
    a0 = pd.Period(ad, 'M') if pd.Timestamp(ad).day == 1 else pd.Period(ad, 'M') + 1
    sel = lambda s: s[(s.index >= a0) & (s.index <= pd.Period(b, 'M'))]
    mr, br, er = me.pct_change(), be.pct_change(), ee.pct_change()
    # 第一個月：若段起點不在月初，第一個月的報酬起點是段起點前的月底（nav 起點月份會是 NaN → 自動略過）
    return K.tstat(sel(mr - br).dropna()), K.tstat(sel(mr - er).dropna())

def seg_stats(nav, ew, a, b, sd, yearly=False, ad=None):
    ad = a if ad is None else ad
    o = K.stats(nav, ad, b, sd, K.E50, ew)          # 日頻（年化、回撤、0050 年化、逐年）
    ot = K.stats(nav, a, b, sd, K.E50, ew)          # 訊號窗 t（窗起點落在 [a, b]）
    o['t_0050'], o['t_EW'], o['n_m'] = ot['t_0050'], ot['t_EW'], ot['n_m']
    oe = K.stats(ew, ad, b, sd)
    tc0, tce = cal_t(nav, ew, ad, b)
    r = {'CAGR': o['CAGR'] * 100, 'maxDD': o['maxDD'] * 100, 'Calmar': o['CAGR'] / abs(o['maxDD']) if o['maxDD'] < 0 else np.nan,
         't_0050': o['t_0050'], 't_EW': o['t_EW'], 't_0050_cal': tc0, 't_EW_cal': tce,
         'E50_CAGR': o['0050_CAGR'] * 100, 'EW_CAGR': oe['CAGR'] * 100, 'ex_0050': (o['CAGR'] - o['0050_CAGR']) * 100,
         'ex_EW': (o['CAGR'] - oe['CAGR']) * 100, 'n_m': o['n_m'], 'years': o['years']}
    if yearly: r['yearly'] = o['yearly']
    return r

def stock_contrib(plan, t_end, G=None, a=None, b=None):
    """每個持有期（成交日開盤→下一成交日開盤，最後一期到 t_end 收盤）每檔貢獻＝(1/k)×該期報酬。回傳 (dict code→加總貢獻, 期數清單)。"""
    G = K.GROW if G is None else G
    ex = [p[0] for p in plan]
    con = {}; per = []
    for i, (t, pk, _, d) in enumerate(plan):
        if a is not None and not (pd.Timestamp(a) <= d <= pd.Timestamp(b)): continue
        t2 = ex[i + 1] if i + 1 < len(ex) else None
        k = len(pk)
        if k == 0: continue
        last = (t2 if t2 is not None else t_end)
        if last > t_end: last = t_end
        r = (K.Cn[t, pk] / K.On[t, pk])
        if last > t:
            r = r * np.prod(G[t + 1:last + 1, pk], axis=0)
        if t2 is not None and t2 <= t_end:
            tr = K.TRADED[t2, pk] & np.isfinite(K.On[t2, pk])
            r = np.where(tr, r * K.On[t2, pk] / np.where(tr, K.Cn[t2, pk], 1.0), r)
        r = r - 1.0
        per.append((d, pk, r))
        for j, x in zip(pk, r):
            con[COLS[j]] = con.get(COLS[j], 0.0) + x / k
    return con, per

def top_contrib(plan, t_end, a, b, nav_cagr_fn=None):
    con, per = stock_contrib(plan, t_end, a=a, b=b)
    if not con: return {}
    top = max(con, key=con.get); tot = sum(con.values())
    # 剔除該股：該股該期報酬換成同期其他持股平均，重算各期組合報酬（無成本）後的年化，對照同法的含該股年化
    def cagr(excl):
        g = 1.0
        for d, pk, r in per:
            rr = r.copy()
            if excl is not None:
                m = np.array([COLS[j] == excl for j in pk])
                if m.any() and (~m).any(): rr[m] = rr[~m].mean()
            g *= 1 + rr.mean()
        yrs = (pd.Timestamp(b) - per[0][0]).days / 365.25
        yrs = min(yrs, (IDX[t_end] - per[0][0]).days / 365.25)
        return (g ** (1 / yrs) - 1) * 100
    c_all, c_ex = cagr(None), cagr(top)
    return {'top_code': top, 'top_name': SL['name'].get(top, ''), 'top_contrib_pp': con[top] * 100, 'sum_contrib_pp': tot * 100,
            'top_share': con[top] / tot if tot > 0 else np.nan, 'gross_CAGR_approx': c_all, 'gross_CAGR_ex_top': c_ex, 'ex_top_drop_pp': c_all - c_ex,
            'top5': sorted(((c, round(v * 100, 1)) for c, v in con.items()), key=lambda x: -x[1])[:5]}

def exdiv_override():
    """flag 1 除權息日：ret 原本 0 → 收盤/開盤 ×(1+平均跳空)。平均跳空在 2021-10 起 flag 2 日重算。"""
    F = K.Fn
    f2 = (F == 2) & K.TRADED & np.isfinite(K.On) & (IDX >= pd.Timestamp('2021-10-01'))[:, None]
    ref = K.Cn / (1 + K.Rn)
    gap = (K.On / ref - 1)[f2]
    gap = gap[np.isfinite(gap)]
    gm = float(gap.mean())
    G1 = K.GROW.copy()
    m1 = (F == 1) & K.TRADED & np.isfinite(K.On)
    G1[m1] = (K.Cn[m1] / K.On[m1]) * (1 + gm)
    return G1, gm, int(m1.sum())

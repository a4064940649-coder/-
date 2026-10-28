#!/usr/bin/env python3
"""o_v5_pit_revenue 核心（只讀 import o_v4_revenue_patch_official/v4lib.py；後者再 import v2_core／o3_core／drj_core）。
本模組只加：資料凍結檢查（data_freeze_v5.json）、V5 營收面板載入、缺口格列舉工具。策略規格完全沿用 v4lib.V4。"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official'))
import v4lib as L4          # noqa: E402（只讀 import）
sys.path.pop(0)
import numpy as np, pandas as pd
K, V, R, O3 = L4.K, L4.V, L4.R, L4.O3
IDX, COLS, N, CI = K.IDX, K.COLS, K.N, K.CI
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
V4DIR = AG / 'o_v4_revenue_patch_official'

def gap_cells(S, d0='2006-04-01', d1='2012-12-31', yfun=None):
    """回傳缺口格 DataFrame：訊號日 d（d0～d1），營收以外都合格（S.cross_section 的 ok_other）但 y3 缺值。
    欄位：sig、code、s52、s52_pct（在 ok_other 池內的百分位）、liq_rank、delisted（訊號日 ≤ 終止日）、pool_ok。"""
    dl = O3.delist_codes()
    rows = []
    for d in S.sig_days:
        if not (pd.Timestamp(d0) <= d <= pd.Timestamp(d1)): continue
        cs = S.cross_section(d)
        ok = cs['ok_other']; y3 = S.y3(d) if yfun is None else yfun(d)
        miss = ok & ~np.isfinite(y3)
        p52 = pd.Series(np.where(ok, cs['s52'], np.nan)).rank(pct=True).to_numpy()
        v60 = S.V60[cs['t']]
        order = np.argsort(-np.where(np.isfinite(v60), v60, -np.inf), kind='stable')
        rank = np.empty(N, int); rank[order] = np.arange(N)
        for j in np.where(miss)[0]:
            c = COLS[j]
            rows.append(dict(sig=str(d.date()), code=c, s52=float(cs['s52'][j]), s52_pct=float(p52[j]), liq_rank=int(rank[j]),
                             delisted=bool(c in dl and d <= dl[c]), pool_ok=int(ok.sum()), n_el=int((ok & np.isfinite(y3)).sum())))
    return pd.DataFrame(rows)

def needed_levels(sig, code):
    """缺口格 (sig, code) 計算 y3 需要的營收水準月份：m−1..m−3 及其前 12 個月。"""
    m = pd.Period(sig, 'M')
    ks = [m - 1, m - 2, m - 3]
    return ks + [k - 12 for k in ks]

# ======================= V5：首次值替換 =======================
DATA = HERE / 'data'
FREEZE5 = HERE / 'data_freeze_v5.json'

def check_freeze5():
    """斷言 data_freeze_v5.json 存在，且其中每個檔案（原始、處理後、腳本）sha256 都沒變。"""
    assert FREEZE5.exists(), '先執行 freeze_v5.py'
    fz = json.loads(FREEZE5.read_text(encoding='utf-8'))
    bad = [k for k, v in fz['files'].items() if sha(v['path']) != v['sha256']]
    assert not bad, f'data_freeze_v5 不符：{bad[:10]}'
    return fz

def load_fv_replace():
    F = pd.read_csv(DATA / 'fv_replace.csv', dtype={'code': str}, parse_dates=['first_date'])
    return F

def load_mc(tag):
    """tag：'s0'..'s19'、'neg'、'pos'。回傳 {(code, ym): ratio}。"""
    f = DATA / (f'fv_mc_{tag}.parquet' if tag in ('neg', 'pos') else f'fv_mc_s{tag[1:]}.parquet')
    M = pd.read_parquet(f)
    return {(c, y): float(r) for c, y, r in zip(M.code, M.ym, M.ratio)}

class FVAdj:
    """首次值調整：對訊號日 d、營收月 k、股票 c：f(c,k;d)＝ratio（新聞替換格且 first_date < d；MC 擾動格不論日期）否則 1；兩者相乘。
    年增率（%）調整：y' ＝ ((1＋y/100)·f(c,k;d)／f(c,k−12;d) − 1)·100（base＝'first'）；base＝'latest' 時分母恆為 1。
    y3＝調整後 m−1..m−3 三個月的平均。panel：'L'（長期 R_FIX，新聞用 ratio_L）或 '5'（Y5_FIX，新聞用 ratio_5）；MC 兩個面板同一 ratio。"""
    PIDX = pd.period_range('2003-01', '2027-12', freq='M')
    def __init__(self, use_news=True, mc=None, base='first'):
        self.base = base
        self.news = []
        if use_news:
            F = load_fv_replace()
            for r in F.itertuples():
                if r.code in CI: self.news.append((CI[r.code], pd.Period(r.ym, 'M'), float(r.ratio_L), float(r.ratio_5), pd.Timestamp(r.first_date)))
        self.M = None
        if mc:
            M = np.ones((len(self.PIDX), N))
            pos = {p: i for i, p in enumerate(self.PIDX)}
            for (c, y), r in mc.items():
                if c in CI: M[pos[pd.Period(y, 'M')], CI[c]] *= r
            self.M, self.pos = M, pos
        self.active = bool(self.news) or self.M is not None

    def adjust_y3(self, Y, m, d, panel, y3, fin):
        if not self.active: return y3
        ks = [m - 1, m - 2, m - 3]
        if any(x not in Y.index for x in ks): return y3
        Z = Y.reindex(index=ks, columns=COLS).to_numpy()
        fa = np.ones((3, N)); fb = np.ones((3, N))
        if self.M is not None:
            fa *= self.M[[self.pos[k] for k in ks]]
            if self.base == 'first': fb *= self.M[[self.pos[k - 12] for k in ks]]
        for j, k, rl, r5, fd in self.news:
            if not fd < d: continue
            r = rl if panel == 'L' else r5
            for i, kk in enumerate(ks):
                if k == kk: fa[i, j] *= r
                if self.base == 'first' and k == kk - 12: fb[i, j] *= r
        ch = ((fa != 1.0) | (fb != 1.0)).any(axis=0)
        if not ch.any(): return y3
        Za = ((1 + Z / 100.0) * fa / fb - 1) * 100.0
        yn = Za.mean(axis=0); yn[np.isnan(Za).any(axis=0)] = np.nan
        return np.where(ch & np.isfinite(y3) & ~fin, yn, y3)

class V5(L4.V4):
    """V4 規格不變，只把 y3 換成首次值調整後的 y3（fv＝FVAdj 或 None）。"""
    def __init__(self, YL, Y5, dmask, fv=None, **kw):
        super().__init__(YL, Y5, dmask, **kw)
        self.fv = fv
    def y3(self, d, back=0):
        y = super().y3(d, back)
        if self.fv is None: return y
        Y, panel = (self.Y5, '5') if d >= V.SWITCH else (self.YL, 'L')
        fin = V.FINMASK if self.exfin else np.zeros(N, bool)
        return self.fv.adjust_y3(Y, pd.Period(d, 'M') - back, d, panel, y, fin)

# ======================= 執行與統計工具 =======================
import math
A0, B0 = '2006-04-01', '2026-10-12'
_G = {}
def setup():
    """凍結檢查＋P／X（只呼叫一次）＋營收面板。"""
    if _G: return _G
    L4.check_freeze()
    XR = L4.setup_px()
    YL, Y5 = L4.load_rfix()
    _G.update(XR=XR, YL=YL, Y5=Y5, DM=L4.disp_mask_fn('trade'), GAP=L4.gap_dict())
    return _G

def make(fv=None, unavail=None, bound=None, **kw):
    g = setup()
    gap, pfix = (g['GAP'], {'B05': 0.5, 'B10': 1.0}[bound]) if bound else (None, None)
    return V5(g['YL'], g['Y5'], g['DM'], fv=fv, unavail=unavail, gap=gap, pfix=pfix, **kw)

def sim(pl, cost, t0, t1):
    return L4.sim_plan(pl, t0, t1, cost, setup()['XR'])[0]

def ew_of(pl, t0, t1):
    return K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)[0]

def mean_nav(rets):
    return (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()

def tstat(x): return L4.tstat(x)
def xs(nav, sd, a=A0, b=B0): return L4.excess(nav, K.E50, sd, a, b)

def nw_t(x, lag):
    x = np.asarray(x, float); n = len(x); mu = x.mean(); e = x - mu
    s = (e @ e) / n
    for l in range(1, lag + 1): s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return float(mu / math.sqrt(s / n))

def block_boot(x, B=10000, blk=12, seed=7):
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed); c = x - x.mean(); mu = x.mean()
    nb = int(math.ceil(n / blk)); cnt = 0
    for _ in range(B):
        st_ = rng.integers(0, n, nb); idx = (st_[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n
        if c[idx].mean() >= mu: cnt += 1
    return cnt / B

def p1(t): return 0.5 * math.erfc(t / math.sqrt(2)) if np.isfinite(t) else float('nan')

def endpoint(nav, sd):
    x = xs(nav, sd); o = {'t_all': tstat(x), 'n': int(len(x)), 'drop_last1': tstat(x.iloc[:-1])}
    srt = x.sort_values(ascending=False)
    for k in (1, 3, 5, 10): o[f'drop_best{k}'] = tstat(x.drop(srt.index[:k]))
    yr = x.groupby(x.index.year).sum(); by = int(yr.idxmax())
    o['best_year'] = by; o['drop_best_year'] = tstat(x[x.index.year != by])
    o['drop_2006'] = tstat(x[x.index.year != 2006]); o['drop_2007'] = tstat(x[x.index.year != 2007])
    for s in ('2007-01-01', '2008-01-01'):
        o[f'start_{s[:7]}'] = {'t': tstat(x[x.index >= pd.Timestamp(s)]), 'CAGR': R.cagr_mdd(nav, s, B0)[0], 'E50_CAGR': R.cagr_mdd(K.E50.reindex(nav.index), s, B0)[0]}
    o['best_month'] = str(srt.index[0].date()); o['best_month_pp'] = float(srt.iloc[0] * 100)
    me = nav.groupby(nav.index.to_period('M')).last(); be = K.E50.reindex(nav.index).groupby(nav.index.to_period('M')).last()
    r = (me.pct_change() - be.pct_change()).dropna(); r = r[(r.index >= pd.Period('2006-04', 'M')) & (r.index <= pd.Period('2026-10', 'M'))]
    o['cal_month_t'] = tstat(r.to_numpy())
    return o

def yearly(nav):
    r = nav.pct_change(); r50 = K.E50.reindex(nav.index).pct_change(); m = r.index >= pd.Timestamp(A0)
    return {int(y): {'strat': float((1 + g).prod() - 1) * 100, '0050': float((1 + r50[g.index].fillna(0)).prod() - 1) * 100} for y, g in r[m].groupby(r[m].index.year)}

def full(nav, ew, sd, extra=True):
    o = L4.allseg(nav, ew, sd)
    if extra:
        x = xs(nav, sd); t = o['ALL']['t_0050']
        o['ALL'].update(NW_t={l: nw_t(x, l) for l in (3, 6, 12)}, boot12_p_one_sided=block_boot(x), p_one_sided=p1(t),
                        bonferroni={m: min(1.0, m * p1(t)) for m in (10, 20, 50)})
        o['endpoint'] = endpoint(nav, sd)
        y = yearly(nav); o['yearly'] = y; o['yearly_lose_to_0050'] = [k for k, v in y.items() if v['strat'] < v['0050']]
    return o

def paired(a, b, sd, x0=A0, x1=B0):
    ra, rb = L4.win_ret(a, sd), L4.win_ret(b.reindex(a.index), sd)
    d = (ra - rb); d = d[(d.index >= pd.Timestamp(x0)) & (d.index <= pd.Timestamp(x1))].dropna()
    nz = d[d.abs() > 1e-12]
    o = dict(mean_pp=float(d.mean() * 100), t=tstat(d), n=int(len(d)), n_nonzero=int(len(nz)), sd_pp=float(d.std(ddof=1) * 100))
    o['MDE_pp_80pow_5pct'] = float(2.8016 * d.std(ddof=1) / math.sqrt(len(d)) * 100) if len(d) > 2 else None
    return o

def stock_win_ret(e, e2, j):
    """個股：成交日 e 開盤 → 下一成交日 e2 開盤（e2 無成交則以收盤計；停牌期間以最後收盤計，不含 X 對價）。"""
    r = K.Cn[e, j] / K.On[e, j]
    if e2 > e: r *= np.prod(K.GROW[e + 1:e2 + 1, j])
    if np.isfinite(K.On[e2, j]) and K.TRADED[e2, j]: r *= K.On[e2, j] / K.Cn[e2, j]
    return r - 1

def marginal_swaps(plans_new, plans_old, t1):
    """逐期比較兩份名單：被換進（new−old）與被換出（old−new）名額的下一期報酬差，按訊號月平均（多 seed 時先逐 seed 再逐月平均）。"""
    per = {}
    for pn, po in zip(plans_new, plans_old):
        ex = [p[0] for p in pn]
        for i, (a, b) in enumerate(zip(pn, po)):
            assert a[3] == b[3]
            sa, sb = set(a[1].tolist()), set(b[1].tolist())
            if sa == sb: continue
            e = a[0]; e2 = ex[i + 1] if i + 1 < len(ex) else t1
            ins = [stock_win_ret(e, e2, j) for j in sa - sb]; outs = [stock_win_ret(e, e2, j) for j in sb - sa]
            ins = [x for x in ins if np.isfinite(x)]; outs = [x for x in outs if np.isfinite(x)]
            if ins and outs: per.setdefault(a[3], []).append(np.mean(ins) - np.mean(outs))
    s = pd.Series({d: np.mean(v) for d, v in per.items()}).sort_index()
    o = {'n_months': int(len(s)), 'mean_pp': float(s.mean() * 100) if len(s) else None, 't': tstat(s.to_numpy()) if len(s) >= 3 else None,
         'sd_pp': float(s.std(ddof=1) * 100) if len(s) >= 3 else None}
    o['MDE_pp_80pow_5pct'] = float(2.8016 * s.std(ddof=1) / math.sqrt(len(s)) * 100) if len(s) >= 3 else None
    o['by_segment'] = {k: {'n': int(((s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))).sum()),
                           'mean_pp': float(s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))].mean() * 100) if ((s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))).any() else None}
                       for k, (a, b) in L4.PER.items() if k in ('S1', 'S2', 'S3', 'S4', 'SEL', 'VAL', 'HOLD')}
    return o

#!/usr/bin/env python3
"""o_v4_revenue_patch_official 核心（只讀 import，不改其他資料夾）：
  main_v2/v2_core.py（→ independent_audit/ia_core.py）：策略規格、滾動窗 gapfix、同分序、AB、統計
  o_infra_v3_rebuild/o3_core.py：P（下市股價格）、X（併購出清）、simulate
  o_disposal_rejudge/drj_core.py：官方處置名單（disposal_hist_rejudge.csv ∪ disposal_5y 獨有列）

本模組只做三件事：
  1) 凍結檢查：data_freeze_v4.json 的每個輸入檔 sha256（官方處置名單、P、X、R_DEL、R_FIX…）
  2) 處置日口徑：'trade'（V4 主口徑：成交日 e＝訊號日次一交易日仍在處置期間 → 排除；公告日 ≤ 訊號日）
                 'signal'（現行 v2／o3：訊號日 d 在處置期間 → 排除）
  3) V4 類別：V2 ＋ 處置遮罩 ＋ 明文排除 KY／DR ＋ AB（unavail）＋ 補不到缺口的界限（pfix）
策略規格（股池、分數、n10、換股日、成交、成本）一律沿用 v2_core，不在這裡改。
"""
import sys, json, hashlib, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_disposal_rejudge'))
import numpy as np, pandas as pd
import drj_core as R          # noqa: E402（只讀 import；它再 import main_v2/v2_core）
sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_infra_v3_rebuild'))
import o3_core as O3          # noqa: E402（只讀 import）
sys.path.pop(0)
K, V = R.K, R.V
assert O3.K is K and O3.V is V
IDX, COLS, N, CI = K.IDX, K.COLS, K.N, K.CI

FREEZE_F = HERE / 'data_freeze_v4.json'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

# 凍結的 V3 底層輸入（o4 各題共用）：路徑 → 預期 sha256（取自各題自己的凍結檔；這裡再斷言一次）
V3_INPUTS = {
    'official_disposal': (AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv', '2c171924096ecf4c0d1da9c3c0db12002c38ace46864dede615496cbb1abe20d'),
    'P_delist_prices': (AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet', 'bc4526b4424eaf9b0d13dcc378ce1ccf54a3f3a35e57d414101ea587aa6abf04'),
    'X_merger_consideration': (AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv', '262e92568a0162e99241c16f1de309df3736c670447b436b91d12b6f69cfd8fd'),
    'delist_official': (AG / 'o_infra_v3_rebuild' / 'delist_official.csv', '41f51cb7e1aadbf1d28244f26cb6452f3ede20f2fbb594350848ed8f55a1f2bd'),
    'R_DEL': (AG / 'o_rev_survivorship' / 'yoy_long_R_DEL.parquet', '3e4b72e454af89cfb38640c5a2c5c146d9b00ff2c24d115e9927d332d6d59a97'),
    'tw_revenue_long': (K.LONG, None),
    'rev_5y': (K.X / 'rev_5y.csv.gz', None),
    'disposal_5y': (K.X / 'disposal_5y.csv.gz', None),
    'src_taiwanan': (AG / 'o_rev_survivorship' / 'src_taiwanan.csv', '6eacc673463e3150d37b1a6ad6e9329a51ccae5ecf223f5385e5503062e90211'),
}

def check_v3_inputs():
    out = {}
    for k, (f, s) in V3_INPUTS.items():
        h = sha(f)
        if s is not None: assert h == s, f'{k} sha256 不符：{h}'
        out[k] = {'path': str(f), 'sha256': h}
    return out

def freeze():
    return json.loads(FREEZE_F.read_text(encoding='utf-8')) if FREEZE_F.exists() else None

def check_freeze(require_rfix=True):
    fz = freeze(); assert fz is not None, '先執行 freeze_v4.py'
    for k, v in fz['v3_base'].items():
        assert sha(v['path']) == v['sha256'], f'{k} 與 data_freeze_v4 不符'
    if require_rfix:
        r = fz['R_FIX']; f = HERE / r['file']
        assert sha(f) == r['sha256'], 'R_FIX 與 data_freeze_v4 不符'
    return fz

# ---------------- 外國公司（KY／DR）----------------
FOREIGN = set(json.loads((AG / 'o_rev_survivorship' / 'data_freeze.json').read_text(encoding='utf-8'))['foreign_codes'])
FOREIGN |= {c for c in COLS if c.startswith('91')}
_dl = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'delist_official.csv', dtype=str)
FOREIGN |= {c for c, n in zip(_dl.code, _dl.name) if isinstance(n, str) and (n.startswith('F-') or 'KY' in n.upper())}   # 舊名「F-」＝外國公司（2014 起改稱 -KY）
DL_NAME = dict(zip(_dl.code, _dl.name))
FIN_WORDS = ('銀', '證', '金控', '證金', '產保', '壽', '票券', '產險', '人壽', '投信', '合庫', '信託')
def is_fin(c):
    n = DL_NAME.get(c, '') if isinstance(DL_NAME.get(c, ''), str) else ''
    return c in V.FIN or any(w in n for w in FIN_WORDS)
FIN2MASK = np.array([is_fin(c) for c in COLS])      # v2 的 FINMASK 再加終止上市的金融股（合庫 5854、元京證 6004、金鼎證 6012…，stock_list 已無其產業別）
def gap_dict():
    """界限用的缺營收股：終止上市名單（4 碼）中，非 KY／DR／F-、非金融者 → {代碼: 終止上市日}。"""
    return {c: d for c, d in O3.delist_codes().items() if c not in FOREIGN and not is_fin(c)}
FORMASK = np.array([c in FOREIGN for c in COLS])

# ---------------- 價格 P、出清 X（只能呼叫一次，必須在建立任何策略物件之前）----------------
_STATE = {'patched': False}
def setup_px():
    if not _STATE['patched']:
        n = O3.apply_patch(require_freeze=True)
        E = O3.load_events(require_freeze=True)
        _STATE.update(patched=True, npatch=n, XR=O3.exit_rules(E, 'main'), E=E)
    return _STATE['XR']

# ---------------- 官方處置名單 ----------------
_D = {}
def disposal_table():
    if not _D:
        D = R.load_list()
        _D['D'] = D
        _D['ann'] = D['announce'].to_numpy(dtype='datetime64[ns]')
        _D['st'] = D['start'].to_numpy(dtype='datetime64[ns]')
        _D['en'] = D['end'].to_numpy(dtype='datetime64[ns]')
        _D['j'] = np.array([CI.get(c, -1) for c in D['code']])
    return _D['D']

def disp_mask_fn(rule):
    """rule：'trade'（成交日仍在處置 → 排除；公告日 ≤ 訊號日）、'signal'（訊號日在處置 → 排除；＝drj E1）、'none'。"""
    disposal_table()
    cache = {}
    def f(d):
        if d in cache: return cache[d]
        m = np.zeros(N, bool)
        if rule == 'none': cache[d] = m; return m
        t = IDX.get_loc(d); e = t + 1
        dd = np.datetime64(d); de = np.datetime64(IDX[e]) if e < len(IDX) else dd
        ref = de if rule == 'trade' else dd
        sel = (_D['ann'] <= dd) & (_D['st'] <= ref) & (_D['en'] >= ref) & (_D['j'] >= 0)
        m[_D['j'][sel]] = True
        cache[d] = m
        return m
    return f

# ---------------- 營收 ----------------
def load_rdel():
    f, s = V3_INPUTS['R_DEL']; assert sha(f) == s
    Y = pd.read_parquet(f); Y.index = pd.PeriodIndex(Y.index, freq='M')
    return Y

def load_rfix():
    fz = check_freeze(require_rfix=True)
    Y = pd.read_parquet(HERE / fz['R_FIX']['file']); Y.index = pd.PeriodIndex(Y.index, freq='M')
    Y5 = None
    if fz.get('Y5_FIX'):
        f5 = HERE / fz['Y5_FIX']['file']; assert sha(f5) == fz['Y5_FIX']['sha256']
        Y5 = pd.read_parquet(f5); Y5.index = pd.PeriodIndex(Y5.index, freq='M')
    return Y, Y5

def y5_orig():
    YL, Y5 = R.revenue()
    return Y5

def delist_codes():
    return O3.delist_codes()

# ---------------- 策略 ----------------
class V4(V.V2):
    """V2（gapfix、正式同分序、排除金融）＋：
       dmask：處置遮罩 callable(d)；exclude_foreign：明文排除 KY／DR；
       gap：{代碼: 終止上市日}，pfix：補不到的缺營收股（營收以外都合格、訊號日 ≤ 終止日）營收百分位固定為 pfix 放回（界限 B05／B10）；
       unavail（AB）：沿用 v2_core 的 make_unavail。"""
    def __init__(self, YL, Y5, dmask, exclude_foreign=True, gap=None, pfix=None, unavail=None, sigmode='gapfix', fin2=True, yproxy=None):
        super().__init__(sigmode, YL, Y5, exfin=True, unavail=unavail)
        self.dmask, self.exfor, self.fin2, self.yproxy = dmask, exclude_foreign, fin2, yproxy
        self.gap, self.pfix = gap, pfix
        self.gm_codes = np.array([gap is not None and c in gap for c in COLS])
        self.gm_until = np.array([np.datetime64(gap[c]) if (gap is not None and c in gap) else np.datetime64('2100-01-01') for c in COLS])
        self.ngap_used = {}

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
        ok_other = top & np.isfinite(s52) & ~self.dmask(d) & np.isfinite(K.On[e]) & K.TRADED[e] & ~(FIN2MASK if self.fin2 else V.FINMASK)
        if self.exfor: ok_other &= ~FORMASK
        self.nproxy = getattr(self, 'nproxy', {})
        if self.yproxy is not None:
            yp = self.yproxy(d)
            cand = self.gm_codes & ok_other & ~np.isfinite(y3) & (np.datetime64(d) <= self.gm_until) & np.isfinite(yp)
            y3 = np.where(cand, yp, y3)
            self.nproxy[d] = [COLS[j] for j in np.where(cand)[0]]
        el_rev = ok_other & np.isfinite(y3)
        gm = np.zeros(N, bool)
        if self.pfix is not None and self.gap is not None:
            gm = self.gm_codes & ok_other & ~np.isfinite(y3) & (np.datetime64(d) <= self.gm_until)
        self.ngap_used[d] = [COLS[j] for j in np.where(gm)[0]]
        el = el_rev | gm
        pr52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
        prr = pd.Series(np.where(el_rev, y3, np.nan)).rank(pct=True).to_numpy()
        if gm.any(): prr = np.where(gm, self.pfix, prr)
        sc = 0.5 * pr52 + 0.5 * prr
        sc[~el] = np.nan
        y3o = y3
        if gm.any():
            y3o = np.where(gm, np.inf if self.pfix >= 1 else np.nan, y3)
        cs = dict(t=t, e=e, el=el, sc=sc, s52=s52, y3=y3o, nold=nold, gm=gm, top=top, ok_other=ok_other)
        self._cs[d] = cs
        return cs

def run_nav(S, n=10, cost='normal', xr=None, tie='official'):
    """回傳 (nav, ew, sig_days, plan, log)。cost：'normal'＝滑價 0.1%＋手續費 6 折；'stress'＝每邊 0.5%；
    ('slip', x)＝單邊滑價 x；('fullfee', x)＝手續費全額＋滑價 x；('extra', x)＝一般成本每邊再加 x。"""
    t0, t1, sd = S.bounds()
    pl = S.plan(n, tie)
    b, s = cost_rates(cost)
    nav, lg = O3.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1, xrules=xr)
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    return nav, ew, sd, pl, lg

def sim_plan(pl, t0, t1, cost, xr):
    b, s = cost_rates(cost)
    nav, lg = O3.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1, xrules=xr)
    return nav, lg

def cost_rates(cost):
    FEE = K.FEE; TAX = 0.003
    if cost == 'normal': return K.rates(0.001)
    if cost == 'stress': return K.rates('stress')
    if cost == 'zero': return 0.0, 0.0
    kind, x = cost
    if kind == 'slip': return FEE + x, FEE + TAX + x
    if kind == 'fullfee': return 0.001425 + x, 0.001425 + TAX + x
    if kind == 'extra': return FEE + 0.001 + x, FEE + TAX + 0.001 + x
    raise ValueError(cost)

# ---------------- 統計 ----------------
PER = {'S1': ('2006-04-01', '2013-12-31'), 'S2': ('2014-01-01', '2020-12-31'), 'S3': ('2021-03-01', '2026-10-12'),
       'S4': ('2024-01-01', '2026-10-12'), 'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31'),
       'HOLD': ('2024-01-01', '2026-10-12'), 'ALL': ('2006-04-01', '2026-10-12')}

def win_ret(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]

def excess(nav, bench, sd, a, b):
    ra, rb = win_ret(nav, sd), win_ret(bench.reindex(nav.index), sd)
    x = (ra - rb)
    return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()

def tstat(x): return R.tstat(x)

def seg(nav, ew, sd, a, b):
    cg, mdd = R.cagr_mdd(nav, a, min(pd.Timestamp(b), nav.index[-1]))
    e50 = K.E50.reindex(nav.index)
    x50 = excess(nav, e50, sd, a, b); xew = excess(nav, ew, sd, a, b)
    return dict(CAGR=cg, maxDD=mdd, E50_CAGR=R.cagr_mdd(e50, a, min(pd.Timestamp(b), nav.index[-1]))[0],
                EW_CAGR=R.cagr_mdd(ew, a, min(pd.Timestamp(b), nav.index[-1]))[0],
                t_0050=tstat(x50), mean_pp=float(x50.mean() * 100), sd_pp=float(x50.std(ddof=1) * 100), n=int(len(x50)), t_EW=tstat(xew))

def allseg(nav, ew, sd):
    return {k: seg(nav, ew, sd, a, b) for k, (a, b) in PER.items()}

def p_one_sided(t):
    return 0.5 * math.erfc(t / math.sqrt(2)) if np.isfinite(t) else float('nan')

def qproxy_fn():
    """B_Q：季營收代理 y3（yoy_qproxy.parquet；訊號月 m 用 m−1..m−3 的代理年增率平均）。"""
    fz = check_freeze(require_rfix=True)
    f = HERE / 'yoy_qproxy.parquet'; assert sha(f) == fz['this_topic_files']['yoy_qproxy.parquet']['sha256']
    yq = pd.read_parquet(f); yq.index = pd.PeriodIndex(yq.index, freq='M')
    cache = {}
    def g(d):
        if d in cache: return cache[d]
        m = pd.Period(d, 'M'); ms = [m - 1, m - 2, m - 3]
        if any(x not in yq.index for x in ms): out = np.full(N, np.nan)
        else:
            Z = yq.reindex(index=ms, columns=COLS).to_numpy()
            out = Z.mean(axis=0); out[np.isnan(Z).any(axis=0)] = np.nan
        cache[d] = out
        return out
    return g

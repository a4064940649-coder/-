#!/usr/bin/env python3
"""o_infra_v3_rebuild 核心：只讀 import main_v2/v2_core.py（它再只讀 import independent_audit/ia_core.py）。
策略規格與 main_v2 正式版 v2（gapfix 窗、正式次排序、排除金融、成本、期間、統計）完全相同，只加三個修正開關：
  P：下市股價格（delist_prices.parquet；apply_patch() 寫進 K 的原始表並重算衍生陣列，必須在建立任何 V2 物件之前呼叫）
  X：永久停止交易的持股（之後再也沒有成交）照 merger_consideration.csv 出清：
       現金併購 → 價值 ×（每股現金對價 ÷ 最後收盤），在付款日之後第一個換股日轉回現金重新投入；
       換股／控股公司轉換／轉上櫃 → 價值 ×1（最後收盤），在終止上市日之後第一個換股日轉回現金；
       經營惡化下市 → 價值歸零（第一個換股日沖銷）。
     沒有對價資料者依 xmode：'upper'＝最後收盤、終止日後第一個換股日；'lower'＝最後收盤 −5%、延到終止日＋60 天。
  D：處置股濾網的資料來源（disposal_hist.csv 的 2005～2021-09-23 近似名單＋disposal_5y 的 2021-09-24 起真實名單），
     以 set_disposal(mode) 切換：'v2'（現行：只有 2021-09-24 起）、'all'（全期一律排除）、'none'（全期一律不排除）、
     'all_plus'（近似＋依召回率推估的漏網名額：排除同訊號日 52 週比最高的合格股）。
"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
K = V.K
sys.path.pop(0)

FREEZE_F = HERE / 'data_freeze.json'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def freeze():
    return json.loads(FREEZE_F.read_text(encoding='utf-8')) if FREEZE_F.exists() else None
def check(name):
    fz = freeze(); assert fz is not None, '先執行 freeze_data.py'
    f = HERE / name; assert sha(f) == fz['files'][name]['sha256'], f'{name} 與凍結的 sha256 不符'
    return f

# ---------------- P ----------------
PATCHED = {'n': 0}
def apply_patch(require_freeze=True):
    f = check('delist_prices.parquet') if require_freeze else HERE / 'delist_prices.parquet'
    Pt = pd.read_parquet(f)
    Cn, On, Hn, Vn, Rn, Fn = (K.C.to_numpy().copy(), K.O.to_numpy().copy(), K.H.to_numpy().copy(), K.V.to_numpy().copy(),
                              K.RET.to_numpy().copy(), K.FLAG.to_numpy().copy())
    ii = K.IDX.get_indexer(pd.DatetimeIndex(Pt['date'])); jj = np.array([K.CI.get(c, -1) for c in Pt['code']])
    ok = (ii >= 0) & (jj >= 0)
    ii, jj, Pt = ii[ok], jj[ok], Pt[ok]
    new = ~np.isfinite(Cn[ii, jj])
    ii, jj, Pt = ii[new], jj[new], Pt[new]
    Cn[ii, jj] = Pt['close'].to_numpy(); On[ii, jj] = Pt['open'].to_numpy(); Hn[ii, jj] = Pt['high'].to_numpy()
    Vn[ii, jj] = Pt['value'].to_numpy(); Rn[ii, jj] = Pt['ret'].to_numpy(); Fn[ii, jj] = 0
    K.C = pd.DataFrame(Cn, index=K.IDX, columns=K.COLS); K.O = pd.DataFrame(On, index=K.IDX, columns=K.COLS)
    K.H = pd.DataFrame(Hn, index=K.IDX, columns=K.COLS); K.V = pd.DataFrame(Vn, index=K.IDX, columns=K.COLS)
    K.RET = pd.DataFrame(Rn, index=K.IDX, columns=K.COLS); K.FLAG = pd.DataFrame(Fn, index=K.IDX, columns=K.COLS)
    K.Cn, K.On, K.Rn, K.Fn, K.Hn = Cn, On, Rn, Fn, Hn
    K.TRADED = np.isfinite(K.Cn)
    K.GROW = np.where(K.TRADED, 1.0 + np.nan_to_num(K.Rn, nan=0.0), 1.0)
    K.MYADJ = pd.DataFrame(np.where(K.TRADED, np.cumprod(K.GROW, axis=0), np.nan), index=K.IDX, columns=K.COLS)
    K.NOBS = K.TRADED.sum(axis=1); K.GOOD = K.NOBS >= 300
    K.PREVC = K.C.ffill().shift(1).to_numpy()
    V._RS.clear(); K._RS.clear()
    PATCHED['n'] = int(len(ii))
    return PATCHED['n']

def last_trade_idx():
    T = K.TRADED
    return np.where(T.any(axis=0), len(K.IDX) - 1 - np.argmax(T[::-1], axis=0), -1)

# ---------------- X ----------------
def load_events(require_freeze=True):
    f = check('merger_consideration.csv') if require_freeze else HERE / 'merger_consideration.csv'
    E = pd.read_csv(f, dtype={'code': str, 'successor': str})
    for c in ('delist_date', 'pay_date', 'announce_date'):
        E[c] = pd.to_datetime(E[c], errors='coerce')
    return E.set_index('code')

def exit_rules(E, xmode='main'):
    """回傳 {欄位 j: (factor, release_date, type[, valid_until])}，只對「面板中最後一筆之後再也沒有成交」且最後一筆早於面板終點 30 天的股票。"""
    lt = last_trade_idx(); end = K.IDX[-1]
    out = {}
    # 代碼重用（終止上市後同代碼由新公司使用，例如 2432 倚天→倚天酷碁）：舊證券視為在終止上市日前最後一筆下市，
    # 規則只在「新證券第一筆成交」之前有效（第 4 個元素＝有效期終點日期）
    dl = pd.read_csv(HERE / 'delist_official.csv', dtype=str)
    dl = dl[dl['four'] == 'True']
    until = {}
    for c, dd in zip(dl['code'], pd.to_datetime(dl['date'])):
        j = K.CI.get(c)
        if j is None or lt[j] < 0 or K.IDX[lt[j]] <= dd: continue
        tr = np.where(K.TRADED[:, j])[0]
        old_ = tr[K.IDX[tr] <= dd]; new_ = tr[K.IDX[tr] > dd]
        if len(old_) == 0 or len(new_) == 0: continue
        lt[j] = old_[-1]; until[j] = K.IDX[new_[0]]
    for j, c in enumerate(K.COLS):
        if lt[j] < 0 or K.IDX[lt[j]] > end - pd.Timedelta(days=30): continue
        lc = K.Cn[lt[j], j]
        ltd = K.IDX[lt[j]]
        if c in E.index:
            e = E.loc[c]
            typ = e['type']; dd = e['delist_date'] if pd.notna(e['delist_date']) else ltd + pd.Timedelta(days=14)
            if typ == 'cash' and pd.notna(e['cash']) and xmode == 'main':
                pay = e['pay_date'] if pd.notna(e['pay_date']) else dd + pd.Timedelta(days=14)
                out[j] = (float(e['cash']) / lc, pay, typ)
            elif typ in ('swap', 'transfer', 'cash') and xmode == 'main':
                out[j] = (1.0, dd, typ)
            elif typ == 'failure' and xmode == 'main':
                out[j] = (0.0, ltd, typ)
            elif xmode == 'upper':
                out[j] = (1.0, dd, typ)
            elif xmode == 'lower':
                out[j] = (0.95 if typ != 'failure' else 0.0, dd + pd.Timedelta(days=60), typ)
            elif xmode == 'lastclose':      # 只解凍：所有永久停牌一律以最後收盤出清（含經營惡化），終止日後第一個換股日
                out[j] = (1.0, dd, typ)
            elif xmode == 'main':           # 型態未知（unknown）：最後收盤、終止日後第一個換股日
                out[j] = (1.0, dd, typ)
        else:
            if xmode == 'main': out[j] = (1.0, ltd + pd.Timedelta(days=14), 'unknown')   # 主版本：無資料者（不應出現）＝最後收盤
            elif xmode in ('upper', 'lastclose'): out[j] = (1.0, ltd + pd.Timedelta(days=14), 'unknown')
            elif xmode == 'lower': out[j] = (0.95, ltd + pd.Timedelta(days=74), 'unknown')
    for j in until:
        if j in out: out[j] = out[j][:3] + (until[j],)
    return out

def simulate(plan, buy, sell, t0, t1, xrules=None, ret_override=None):
    """K.simulate（freeze=True）＋ X 出清。xrules=None 時與 K.simulate 逐日相同（程式內斷言於 validate）。"""
    G = K.GROW if ret_override is None else ret_override
    N = K.N
    reb = {p[0]: p[1] for p in plan}
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1); tov = []; frozen_ev = []; exits = []
    rel = None
    if xrules:
        rel = {j: (v[0], int(np.searchsorted(K.IDX, v[1])), int(np.searchsorted(K.IDX, v[3])) if len(v) > 3 else 10 ** 9) for j, v in xrules.items()}
        typs = {j: v[2] for j, v in xrules.items()}
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t]
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            gap = np.where(tr, G[t] * K.On[t] / np.where(tr, K.Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            if rel is not None:
                for j in np.where(frz)[0]:
                    if j in rel and rel[j][1] <= t < rel[j][2]:
                        f = rel[j][0]
                        exits.append((str(K.IDX[t].date()), K.COLS[j], typs[j], float(f), float(h[j] / (val.sum() + cash))))
                        cash += h[j] * f; h[j] = 0.0; frz[j] = False
            Vfz = h[frz].sum()
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            k = len(pk)
            def cost_of(T):
                tgt = np.zeros(N); tgt[pk] = T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * k + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            T = lo
            tgt = np.zeros(N); tgt[pk] = T
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz))
            newval = np.where(frz, h, 0.0)
            newval[pk] += T
            oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            cash = 0.0
            if frz.any(): frozen_ev.append((str(K.IDX[t].date()), [K.COLS[j] for j in np.where(frz)[0]], float(Vfz / (Vt + Vfz))))
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=K.IDX[t0:t1 + 1]), {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan, 'frozen': frozen_ev, 'exits': exits}

# ---------------- D ----------------
DSP5 = K.DSP.copy()
D5_START = pd.Timestamp('2021-09-24')
_DM = {'mode': 'v2', 'hist': None, 'extra': {}}

def load_disposal_hist(require_freeze=True):
    f = check('disposal_hist.csv') if require_freeze else HERE / 'disposal_hist.csv'
    H = pd.read_csv(f, dtype={'code': str}, parse_dates=['start', 'end'])
    return H

def _mask_from(T, d):
    m = np.zeros(K.N, bool)
    x = T[(T['start'] <= d) & (T['end'] >= d)]
    for c in x['code']:
        j = K.CI.get(c)
        if j is not None: m[j] = True
    return m

def set_disposal(mode, hist=None, extra=None):
    """mode：v2／all／none／all_plus。extra＝{訊號日: 額外排除名額數}（all_plus 用）。"""
    _DM['mode'] = mode; _DM['hist'] = hist; _DM['extra'] = extra or {}
    def dm(d):
        if _DM['mode'] == 'none': return np.zeros(K.N, bool)
        m = _mask_from(DSP5, d)               # 真實名單（2021-09-24 起）
        if _DM['mode'] in ('all', 'all_plus') and d < D5_START:
            m |= _mask_from(_DM['hist'], d)
        return m
    K.disposal_mask = dm

# ---------------- 營收：R_DEL（o_rev_survivorship 凍結的修補版長期營收）----------------
ORS = HERE.parent / 'o_rev_survivorship'
def load_rdel():
    fz = json.loads((ORS / 'data_freeze.json').read_text(encoding='utf-8'))['outputs']['R_DEL']
    f = ORS / fz['file']; assert sha(f) == fz['sha256'], 'R_DEL sha256 不符'
    Y = pd.read_parquet(f); Y.index = pd.PeriodIndex(Y.index, freq='M')
    return Y

# 終止上市名單（4 碼、2006 起）：缺營收界限只套用在這些股票
def delist_codes():
    """回傳 {代碼: 終止上市日}（4 碼）。缺營收界限只套用在訊號日 ≤ 終止上市日（代碼重用後的新公司不算）。"""
    d = pd.read_csv(HERE / 'delist_official.csv', dtype=str)
    d = d[d['four'] == 'True']
    return dict(zip(d['code'], pd.to_datetime(d['date'])))

class V3(V.V2):
    """V2 加：(1) gap：終止上市名單內、營收以外條件都合格但缺營收的股票，營收百分位固定為 pfix 後放回合格池（界限）；
    (2) all_plus：2021-09-24 以前另外剔除同訊號日 52 週比最高的 k 檔（k＝依召回率推估的漏網處置股數）。"""
    def __init__(self, *a, gapcodes=None, pfix=None, foreign=frozenset(), **kw):
        super().__init__(*a, **kw)
        self.gapcodes, self.pfix = gapcodes, pfix
        self.gm_mask = np.array([c in (gapcodes or ()) and c not in foreign for c in K.COLS])
        self.ngap_used = {}; self.plus_drop = {}

    def cross_section(self, d):
        if d in self._cs: return self._cs[d]
        t = K.IDX.get_loc(d); e = t + 1
        v60 = self.V60[t]; px = K.Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        top = np.zeros(K.N, bool); top[order[:min(300, base.sum())]] = True
        top &= base
        y3 = self.y3(d)
        s52 = self.S52[t]
        ok_other = top & np.isfinite(s52) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e]
        el_rev = ok_other & np.isfinite(y3)
        gm = np.zeros(K.N, bool)
        if self.pfix is not None and self.gapcodes:
            gm = self.gm_mask & ok_other & ~np.isfinite(y3)
            if isinstance(self.gapcodes, dict):
                gm &= np.array([d <= self.gapcodes.get(c, pd.Timestamp('2100-01-01')) for c in K.COLS])
            if self.exfin: gm &= ~V.FINMASK
        self.ngap_used[d] = [K.COLS[j] for j in np.where(gm)[0]]
        el = el_rev | gm
        k = _DM['extra'].get(d, 0) if (_DM['mode'] == 'all_plus' and d < D5_START) else 0
        if k > 0:
            s = np.where(el, np.nan_to_num(s52, nan=-1), -np.inf)
            drop = np.argsort(-s, kind='stable')[:k]
            el[drop] = False; el_rev[drop] = False; gm[drop] = False
            self.plus_drop[d] = [K.COLS[j] for j in drop]
        pr52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
        prr = pd.Series(np.where(el_rev, y3, np.nan)).rank(pct=True).to_numpy()
        if gm.any(): prr = np.where(gm, self.pfix, prr)
        sc = 0.5 * pr52 + 0.5 * prr
        sc[~el] = np.nan
        y3o = np.where(gm, np.inf if (self.pfix or 0) >= 1 else np.nan, y3)   # 次排序：pfix=1 視為營收最高
        cs = dict(t=t, e=e, el=el, sc=sc, s52=s52, y3=np.where(gm, np.nan, y3) if (self.pfix or 0) < 1 else y3o, nold=0, gm=gm, top=top)
        self._cs[d] = cs
        return cs

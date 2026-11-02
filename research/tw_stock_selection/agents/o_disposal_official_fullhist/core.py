#!/usr/bin/env python3
"""o_disposal_official_fullhist 共用核心（只讀 import，不寫入其他資料夾）。

資料：
  - 價格：V7X 快取（o_exec_engine_rewrite_v7x/x_data：V7 上市身分面板＋o_infra_v3_rebuild 下市價 P）。
  - 名單：o_disposal_rejudge/disposal_hist_rejudge.csv（TWSE rwd punish 全年封存 2002-12～2026）。
  - 0050 含息、加權指數。
事件規則：disposal_v2 官方版 V_OFF 逐字（R1 鏈、R2 空頭、R3 進場漲幅、R4 流動性、R5 出場延後、R6 成本 0.8/1.2），
2015-06-01 前漲跌幅 7%，門檻改用 6%（漲跌幅 − 1%）。
可執行層：V7X 成本口徑（手續費 0.1425%×折扣、最低 20 元、證交稅 0.3%）＋每邊 max(一跳/價, 0.1%)＋K5T 量相依衝擊
κ·σ60·√(q/Qref)，Qref＝當日成交值 ÷ 當日撮合盤數（連續撮合期＝當日成交值的 1%）；參與上限 q ≤ cap×Qref。
"""
import sys, math, json, hashlib
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_exec_engine_rewrite_v7x'))
import x_data as D  # noqa: E402  只讀快取

LIST_F = AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv'
PROXY_F = AG / 'o_disposal_rejudge' / 'proxy_events.csv'
TAIEX_F = Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')
E50_F = Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')
CPI = json.loads((AG / 'o_v7_listing_exec_engine' / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text())['index_2006_100']
SPEC7 = D.SPEC7
KAPPA = SPEC7['kappa']            # 0.5
STRESS_EXTRA = SPEC7['stress']['extra_per_side']   # 0.004
TAX = 0.003
ETF_FEE = 0.001425 * 0.6
ETF_BUY, ETF_SELL = ETF_FEE, ETF_FEE + 0.001

IDX = D.IDX; T = len(IDX); CI = D.CI; COLS = D.COLS
C, O, V, R, F, PC = D.C, D.O, D.V, D.R, D.F, D.PC
V60p, SIGp, REF = D.V60p, D.SIGp, D.REF
TR = np.isfinite(C)
LIM_SWITCH = pd.Timestamp('2015-06-01'); I_LIM = int(IDX.searchsorted(LIM_SWITCH))
FEE_SWITCH = pd.Timestamp('2015-01-01'); I_FEE = int(IDX.searchsorted(FEE_SWITCH))
CHANGE = pd.Timestamp('2026-08-10')
PERIODIC_START = pd.Timestamp('2010-12-01')

PER = {
    'PREPAY_2004_2006-03': ('2004-01-01', '2006-03-31'),
    'OOS_2006-04_2021-09': ('2006-04-01', '2021-09-30'),
    'SEL_2021-10_2023': ('2021-10-01', '2023-12-31'),
    'DESC_2024_2026-10': ('2024-01-01', '2026-10-22'),
    'NEWREG_2026-08-10_': ('2026-08-10', '2026-10-22'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def limit_rate(i):
    return 0.07 if i < I_LIM else 0.10


def tick(p):
    return D.tick(p)


with np.errstate(invalid='ignore', divide='ignore'):
    _lim = np.where(np.arange(T)[:, None] < I_LIM, 0.07, 0.10)
    _up = REF * (1 + _lim); _dn = REF * (1 - _lim)
    LIMUP = np.floor(_up / tick(_up) + 1e-9) * tick(_up)
    LIMDN = np.ceil(_dn / tick(_dn) - 1e-9) * tick(_dn)
    del _lim, _up, _dn
    LA = np.log1p(np.nan_to_num(R, nan=0.0).clip(-0.99, None)).cumsum(0)
    # 開盤水準（對數）：LA[t] − log(C/O)
    LOGCO = np.where(TR & np.isfinite(O) & (O > 0), np.log(C / O), 0.0)
LASTV = np.array([np.max(np.where(TR[:, j])[0]) if TR[:, j].any() else -1 for j in range(len(COLS))])
FIRSTV = np.array([np.min(np.where(TR[:, j])[0]) if TR[:, j].any() else T for j in range(len(COLS))])

# 流動性合格（前一日判定）等權
LIQL = np.nan_to_num(V60p, nan=0.0) >= 1e7
LIQL &= np.nan_to_num(PC, nan=0.0) >= 10
with np.errstate(invalid='ignore'):
    _m = LIQL & TR & np.isfinite(R)
    EWR = np.where(_m, R, 0.0).sum(1) / np.maximum(_m.sum(1), 1)
    _mo = _m & np.isfinite(O) & (O > 0)
    EW_INTRA = np.where(_mo, C / O - 1, 0.0).sum(1) / np.maximum(_mo.sum(1), 1)
LEW = np.log1p(EWR.clip(-0.99, None)).cumsum()
LEW_OPEN = LEW - np.log1p(EW_INTRA.clip(-0.99, None))
E50 = D.E50
LE5 = np.log(pd.Series(E50).ffill().bfill().to_numpy())
with np.errstate(invalid='ignore', divide='ignore'):
    E50_INTRA = np.zeros(T)   # 0050 沒有開盤價：開盤點以前一日收盤近似（只用於開盤進出版本的對 0050 比較）

# 加權指數 200 日均線（空頭開關）
_tx = pd.read_csv(TAIEX_F, parse_dates=['date']).set_index('date')['close'].sort_index()
_bear = (_tx < _tx.rolling(200, min_periods=200).mean())
_TXI = _tx.index


def bear_before(d):
    k = _TXI.searchsorted(d) - 1
    return bool(_bear.iloc[k]) if k >= 0 else False


def tdays_tx(s, e):
    return int(_TXI.searchsorted(e, 'right') - _TXI.searchsorted(s))


def cpi_factor(t):
    y = str(min(max(IDX[t].year, 2006), 2026))
    return CPI[y] / CPI['2026']


# ---------------- 名單 ----------------
def load_list():
    d = pd.read_csv(LIST_F, dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
    d['ordr'] = np.arange(len(d))
    d['name'] = d['name'].fillna('')
    d['is_common'] = (d['code'].str.len() == 4) & (~d['code'].str.startswith('0')) & (~d['name'].str.contains('DR'))
    return d


def universe(d=None):
    d = load_list() if d is None else d
    u = d[d['is_common']].copy()
    u['in_D'] = u['code'].isin(CI)
    u['s_i'] = IDX.searchsorted(u['start'].values)
    u['e_i'] = IDX.searchsorted(u['end'].values, 'right') - 1
    u['ann_i'] = IDX.searchsorted(u['announce'].values, 'right') - 1
    u = u.sort_values(['code', 'start', 'announce', 'ordr'], kind='stable').reset_index(drop=True)
    # R6 再處置（30 個交易日）：本筆 start 前 30 個交易日內同檔有先前記錄結束
    sec = []; pe = {}
    for r in u.itertuples():
        p = pe.get(r.code, -10 ** 9)
        sec.append(bool(p >= r.s_i - 30 and p < r.s_i))
        pe[r.code] = max(p, r.e_i)
    u['second30'] = sec
    # R1 鏈（disposal_today.py）：start ≤ 上一筆採用記錄的 end + 3 日曆日 → 不採用
    take = []; last = {}
    for r in u.itertuples():
        le = last.get(r.code)
        if le is not None and r.start <= le + pd.Timedelta(days=3):
            take.append(False)
        else:
            take.append(True); last[r.code] = r.end
    u['take_R1'] = take
    u['n_auct'] = np.where(u['regime'] == 'prepay_only', np.nan,
                           np.where(u['match_min'].fillna(20) > 0, 270.0 / u['match_min'].fillna(20).replace(0, 20), np.nan))
    return u


def qref(t, j, ev_nauct, in_disp):
    """單一撮合盤的可成交值參考：處置期（分盤）＝V/盤數；連續撮合處置（2010-11 前）＝1%×V；非處置日＝5%×V（遞延賣出用）。"""
    v = V[t, j]
    if not np.isfinite(v) or v <= 0: return 0.0
    if not in_disp: return 0.05 * v / 0.2       # 乘以參與上限 0.2 後＝5%×V
    if ev_nauct is None or not np.isfinite(ev_nauct): return 0.01 * v
    return v / ev_nauct


def fee_rate(t, stress=False):
    if stress or t < I_FEE: return 0.001425 * 0.6
    return 0.001425 * 0.28


def side_cost(t, j, q, Q, side, stress=False):
    """單邊成本率（不含稅）：手續費（最低 20 元）＋max(一跳/價, 0.1%)＋κσ√(q/Q)（＋壓力 0.4%）。"""
    if q <= 0: return 0.0
    px = C[t, j]
    tk = float(tick(px)) / px if np.isfinite(px) and px > 0 else 0.001
    sg = SIGp[t, j] if np.isfinite(SIGp[t, j]) else 0.03
    imp = max(tk, 0.001) + (KAPPA * sg * math.sqrt(q / Q) if Q > 0 else 0.0)
    fr = fee_rate(t, stress)
    fee = max(20.0, fr * q) / q
    return fee + imp + (STRESS_EXTRA if stress else 0.0)


# ---------------- 事件建構 ----------------
DEF_CFG = dict(entry='C1', exit='CL', bear=True, liq=1e7, px_min=10.0, thr_off=0.01, first_only=False,
               kinds=('regular', 'changed_method', 'supervisory'), chain=True)


def entry_index(cfg, s_i):
    return s_i if cfg['entry'] == 'C1' else s_i + 1


def entry_ok(cfg, i, j):
    """可否在進場點成交（C*：收盤漲幅 < 漲跌幅 − thr_off；O2：開盤未鎖漲停）。"""
    if i >= T or not TR[i, j]: return False
    if cfg['entry'] in ('C1', 'C2'):
        r = R[i, j]
        return bool(np.isfinite(r) and r < limit_rate(i) - cfg['thr_off'])
    o = O[i, j]
    return bool(np.isfinite(o) and o > 0 and not (np.isfinite(LIMUP[i, j]) and o >= LIMUP[i, j] - 1e-9))


def exit_index(cfg, e_i, j):
    """回傳 (出場日 index, 出場點種類 'C' 或 'O')；延後規則：跌幅 ≤ −(漲跌幅−1%) 或無價 → 順延到下一個可賣日收盤。"""
    lim = min(T - 1, LASTV[j])
    if cfg['exit'] == 'OA':
        k = e_i + 1
        if k > lim: return lim, 'C'
        o = O[k, j]
        if TR[k, j] and np.isfinite(o) and o > 0 and not (np.isfinite(LIMDN[k, j]) and o <= LIMDN[k, j] + 1e-9):
            return k, 'O'
        k0 = k
    else:
        k0 = e_i if cfg['exit'] == 'CL' else e_i - 1
    k = min(k0, lim)
    while k < lim and not (TR[k, j] and np.isfinite(R[k, j]) and R[k, j] > -(limit_rate(k) - 0.01)):
        k += 1
    return k, 'C'


def lvl(LAarr, t, j, kind):
    return LAarr[t, j] - (LOGCO[t, j] if kind == 'O' else 0.0)


def build(cfg=None, u=None, period=None):
    cfg = dict(DEF_CFG, **(cfg or {}))
    u = universe() if u is None else u
    st = dict(records=0, not_common_or_kind=0, chain_R1=0, first_only_drop=0, not_in_panel=0, invalid=0, busy=0, bear=0,
              untradable=0, illiquid=0, no_exit=0, kept=0, delayed=0, delisted_in_hold=0)
    rows = []; busy = {}
    uu = u.sort_values(['s_i', 'announce', 'ordr'], kind='stable')
    for r in uu.itertuples():
        st['records'] += 1
        if r.kind not in cfg['kinds']: st['not_common_or_kind'] += 1; continue
        if cfg['chain'] and not r.take_R1: st['chain_R1'] += 1; continue
        if cfg['first_only'] and r.second30: st['first_only_drop'] += 1; continue
        if not r.in_D: st['not_in_panel'] += 1; continue
        if r.s_i >= T or r.e_i <= r.s_i: st['invalid'] += 1; continue
        if r.e_i >= T - 1 and cfg['exit'] == 'OA': st['invalid'] += 1; continue
        j = CI[r.code]
        i0 = entry_index(cfg, r.s_i)
        if busy.get(r.code, -1) >= r.s_i: st['busy'] += 1; continue
        if cfg['bear'] and bear_before(IDX[r.s_i]): st['bear'] += 1; continue
        if not entry_ok(cfg, i0, j): st['untradable'] += 1; continue
        v60 = V60p[r.s_i, j]; pc = PC[r.s_i, j]
        if not (np.isfinite(v60) and v60 >= cfg['liq'] and np.isfinite(pc) and pc >= cfg['px_min']): st['illiquid'] += 1; continue
        i1, xk = exit_index(cfg, r.e_i, j)
        ek = 'O' if cfg['entry'] == 'O2' else 'C'
        if i1 < i0 or (i1 == i0 and not (ek == 'O' and xk == 'C')): st['no_exit'] += 1; continue
        busy[r.code] = i1; st['kept'] += 1
        planned = r.e_i if cfg['exit'] == 'CL' else (r.e_i - 1 if cfg['exit'] == 'CL1' else r.e_i + 1)
        dl = bool(i1 >= LASTV[j] and LASTV[j] < r.e_i + 1 and LASTV[j] < T - 1)
        st['delayed'] += int(i1 > planned); st['delisted_in_hold'] += int(dl)
        a = lvl(LA, i0, j, ek); b = lvl(LA, i1, j, xk)
        raw = math.expm1(b - a)
        ew0 = LEW_OPEN[i0] if ek == 'O' else LEW[i0]; ew1 = LEW_OPEN[i1] if xk == 'O' else LEW[i1]
        ew = math.expm1(ew1 - ew0)
        e5 = math.expm1(LE5[i1 - (1 if xk == 'O' else 0)] - LE5[i0 - (1 if ek == 'O' else 0)])
        c_flat = 0.012 if r.second30 else 0.008
        run6 = (math.exp(LA[r.s_i - 1, j] - LA[r.s_i - 7, j]) - 1) if r.s_i >= 7 and TR[r.s_i - 1, j] else np.nan
        pre = V[max(r.s_i - 20, 0):r.s_i, j]; dur = V[r.s_i:r.e_i + 1, j]
        vdrop = (np.nanmean(dur) / np.nanmean(pre) - 1) if np.isfinite(pre).sum() >= 10 and np.isfinite(dur).sum() >= 2 else np.nan
        rows.append(dict(code=r.code, j=j, ordr=r.ordr, announce=r.announce, start=r.start, end=r.end, s_i=r.s_i, e_i=r.e_i,
                         i0=i0, i1=i1, ek=ek, xk=xk, date=IDX[i0], exit_date=IDX[i1], hold=i1 - i0, plan_n=r.e_i - r.s_i,
                         second=bool(r.second30), nth=int(r.nth), match_min=r.match_min, n_auct=r.n_auct, regime=r.regime,
                         kind=r.kind, raw=raw, ew=ew, e50=e5, gross=raw - ew, cost=c_flat, net=raw - ew - c_flat,
                         net50=raw - e5 - c_flat, raw_net=raw - c_flat, run6=run6, vdrop=vdrop, delayed=bool(i1 > planned),
                         delisted=dl, v60=v60, px=C[i0, j], ann_ok=bool(r.ann_i < r.s_i)))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(['date', 'code']).reset_index(drop=True)
        if period is not None:
            a, b = PER[period] if isinstance(period, str) else period
            df = df[(df['date'] >= pd.Timestamp(a)) & (df['date'] <= pd.Timestamp(b))].reset_index(drop=True)
    return df, st


def in_disp(t, ev):
    return ev['s_i'] <= t <= ev['e_i']


def exec_cost_event(df, slot_2026=1e5, cap=0.2, stress=False, defer=True):
    """逐筆可執行成本（以 2026 幣值每槽 slot_2026 元、CPI 換算）；回傳加上 cost_x、net_x、fill_ratio 的新表。
    進場：處置第 1 日（或第 2 日）最後一盤（收盤）或開盤；量上限 cap×Qref；超過縮單。出場：同理，超過的部分順延到下一日（處置結束後 5%×V）。"""
    out = df.copy()
    cx, fr, nx = [], [], []
    for r in df.itertuples():
        j, i0, i1 = r.j, r.i0, r.i1
        target = slot_2026 * cpi_factor(i0)
        Q0 = qref(i0, j, r.n_auct, r.s_i <= i0 <= r.e_i)
        q = min(target, cap * Q0) if Q0 > 0 else 0.0
        if q <= 0:
            cx.append(np.nan); fr.append(0.0); nx.append(np.nan); continue
        cb = side_cost(i0, j, q, Q0, +1, stress)
        # 出場：依序在 i1, i1+1, ... 賣出（以 i1 出場點的價值計量，之後日子以收盤水準相對 i1 出場點換算）
        base = lvl(LA, i1, j, r.xk)
        val = q * (1 - cb) * (1 + r.raw); rem = val; t = i1; got = 0.0; first = True
        lim = min(T - 1, LASTV[j])
        while rem > 1e-6:
            if t > lim: t = lim
            if first and r.xk == 'O':
                Q1 = 0.05 * V[t, j] / 0.2 if np.isfinite(V[t, j]) else 0.0; pxr = 1.0
            else:
                Q1 = qref(t, j, r.n_auct, r.s_i <= t <= r.e_i); pxr = math.exp(LA[t, j] - base)
            sell = rem if (t >= lim or not defer) else (min(rem, cap * Q1) if Q1 > 0 else 0.0)
            if sell > 0:
                c1 = side_cost(t, j, sell, Q1 if Q1 > 0 else sell, -1, stress) + TAX
                got += sell * pxr * (1 - c1); rem -= sell
            first = False
            if t >= lim: break
            t += 1
            while t < lim and not (TR[t, j] and np.isfinite(R[t, j]) and R[t, j] > -(limit_rate(t) - 0.01)):
                t += 1
        ret_x = got / q - 1
        cx.append((1 + r.raw) - (1 + ret_x)); fr.append(q / target); nx.append(ret_x - r.ew)
    out['cost_x'] = cx; out['fill'] = fr; out['net_x'] = nx; out['raw_x'] = out['net_x'] + out['ew']
    out['net50_x'] = out['raw_x'] - out['e50']
    return out


# ---------------- 統計 ----------------
def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std(ddof=1) == 0: return None
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)


def S(df, col='net'):
    df = df.dropna(subset=[col]) if len(df) else df
    o = {'n': int(len(df))}
    if len(df) == 0: return o
    x = df[col]
    o['mean_pct'] = round(float(x.mean() * 100), 2)
    if len(df) < 3: return o
    m = x.groupby(df['date'].dt.to_period('M')).mean()
    xs = x.sort_values(); k5 = max(1, int(math.ceil(len(xs) * 0.05)))
    o.update({'median_pct': round(float(x.median() * 100), 2), 'hit': round(float((x > 0).mean()), 3),
              't_month': tstat(m), 'n_months': int(len(m)), 't_event': tstat(x),
              'ex_top5pct_mean_pct': round(float(xs.iloc[:-k5].mean() * 100), 2)})
    return o


def sub(df, a, b):
    return df[(df['date'] >= pd.Timestamp(a)) & (df['date'] <= pd.Timestamp(b))]


def cluster_t(df, flag, col='net'):
    d = df.dropna(subset=[col, flag]); y = d[col].to_numpy(float); X = np.c_[np.ones(len(d)), d[flag].astype(float).to_numpy()]
    if X[:, 1].sum() < 3 or (1 - X[:, 1]).sum() < 3: return None
    XtX = np.linalg.inv(X.T @ X); b = XtX @ X.T @ y; e = y - X @ b
    g = d['date'].dt.to_period('M').to_numpy(); G = pd.unique(g); meat = np.zeros((2, 2))
    for k in G:
        s = (X[g == k] * e[g == k, None]).sum(0); meat += np.outer(s, s)
    n, kk, nG = len(y), 2, len(G)
    Vv = XtX @ meat @ XtX * (nG / (nG - 1)) * ((n - 1) / (n - kk))
    return {'diff_pct': round(float(b[1] * 100), 2), 't_cluster_month': round(float(b[1] / math.sqrt(Vv[1, 1])), 2),
            'n1': int(X[:, 1].sum()), 'n0': int((1 - X[:, 1]).sum()), 'n_clusters': int(nG)}


# ---------------- 10 槽組合（閒置 0050） ----------------
def sleeve(trades, tier_2026=1e6, mode='FIX', K=10, cap=0.2, stress=False, t_start='2006-04-01', t_end=None, idle='0050', min_fill=0.0, defer=True, buffer_slots=0):
    """trades：build() 的表（不需 exec 欄）。回傳 (日 NAV Series, 統計 dict, 逐筆成交 DataFrame)。
    FIX：每槽目標＝tier×CPI(年)/CPI(2026)/K；GROW：起始資金＝tier×CPI(起始年)/CPI(2026)，每槽目標＝權益/K。
    預收款券：買進資金在委託日前一交易日收盤從 0050 贖回（委託日當天現金不計息）。出場超過量上限的部分順延。"""
    t0 = int(IDX.searchsorted(pd.Timestamp(t_start))); t1 = T - 1 if t_end is None else int(IDX.searchsorted(pd.Timestamp(t_end), 'right') - 1)
    ev = {}
    for r in trades.itertuples():
        if t0 + 1 <= r.i0 <= t1: ev.setdefault(r.i0 - (1 if r.ek == 'C' else 1), []).append(r)
    # 委託日前一日（i0−1）先保留現金；C1 的委託日＝i0；O2／C2 委託日＝s_i+1＝i0
    cap0 = tier_2026 * cpi_factor(t0)
    u50 = cap0 * (1 - ETF_BUY) / E50[t0 - 1]; cash = 0.0
    pos = {}   # key -> dict
    resv = []  # 已保留現金、待進場：(r, amt)
    nav = np.empty(t1 - t0 + 1); inv = np.empty(t1 - t0 + 1)
    fills = []; stat = dict(entered=0, no_slot=0, held=0, capped=0, zero_cap=0, deferred_sell_days=0)

    def equity(t):
        pv = 0.0
        for p in pos.values():
            pv += p['units'] * math.exp(LA[t, p['j']] - LA[p['i0'], p['j']]) * p['px0']
        return cash + u50 * E50[t] + pv, pv

    for t in range(t0, t1 + 1):
        # 1) 進場（今天收盤或開盤）——用昨日保留的現金
        newresv = []
        for r, amt in resv:
            if r.i0 != t: newresv.append((r, amt)); continue
            j = r.j
            Q0 = qref(t, j, r.n_auct, r.s_i <= t <= r.e_i)
            q = min(amt, cap * Q0) if Q0 > 0 else 0.0
            if q < amt: stat['capped'] += 1
            if q < min_fill * amt: q = 0.0
            if q <= 0: stat['zero_cap'] += 1; u50 += amt * (1 - ETF_BUY) / E50[t]; continue
            cb = side_cost(t, j, q, Q0, +1, stress)
            px0 = C[t, j] if r.ek == 'C' else C[t, j] / math.exp(LOGCO[t, j])
            # 開盤進場：持有單位的價格基準以收盤水準換算
            units_val = q * (1 - cb)
            if r.ek == 'O': units_val *= math.exp(LOGCO[t, j])  # 開盤到收盤的漲跌
            pos[(r.code, r.ordr)] = dict(j=j, i0=t, px0=1.0, units=units_val, r=r, rem_frac=1.0, selling=False, inv=q)
            if amt - q > 0: u50 += (amt - q) * (1 - ETF_BUY) / E50[t]
            stat['entered'] += 1
        resv = newresv
        # 2) 出場
        for key in list(pos.keys()):
            p = pos[key]; r = p['r']; j = p['j']
            if t < r.i1: continue
            lim = min(T - 1, LASTV[j])
            sellable = t >= lim or t == r.i1 or (TR[t, j] and np.isfinite(R[t, j]) and R[t, j] > -(limit_rate(t) - 0.01))
            if not sellable: stat['deferred_sell_days'] += 1; continue
            first = (t == r.i1)
            val = p['units'] * math.exp(LA[t, j] - LA[p['i0'], j])
            if first and r.xk == 'O':
                val = val / math.exp(LOGCO[t, j]); Q1 = 0.05 * V[t, j] / 0.2 if np.isfinite(V[t, j]) else 0.0
            else:
                Q1 = qref(t, j, r.n_auct, r.s_i <= t <= r.e_i)
            sell = val if (t >= lim or not defer) else (min(val, cap * Q1) if Q1 > 0 else 0.0)
            if sell <= 0: stat['deferred_sell_days'] += 1; continue
            c1 = side_cost(t, j, sell, Q1 if Q1 > 0 else sell, -1, stress) + TAX
            proceeds = sell * (1 - c1)
            u50 += proceeds * (1 - ETF_BUY) / E50[t]
            frac = sell / val
            if frac >= 1 - 1e-9:
                fills.append(dict(code=r.code, date=IDX[p['i0']], exit=IDX[t], inv=p['inv'], ret=None))
                pos.pop(key)
            else:
                p['units'] *= (1 - frac); stat['deferred_sell_days'] += 1
        # 3) 今日收盤後：為明天（或今天之後）的委託保留現金
        E, pv = equity(t)
        for r in ev.get(t, []):
            if any(pp['r'].code == r.code for pp in pos.values()) or any(rr.code == r.code for rr, _ in resv):
                stat['held'] += 1; continue
            if len(pos) + len(resv) >= K: stat['no_slot'] += 1; continue
            tgt = (E / K) if mode == 'GROW' else tier_2026 * cpi_factor(t) / K
            avail = u50 * E50[t] * (1 - ETF_SELL)
            amt = min(tgt, avail)
            if amt <= 0: continue
            u50 -= amt / (1 - ETF_SELL) / E50[t]; cash += 0.0
            resv.append((r, amt))
        # 現金緩衝（嚴格預收款券敏感度）：閒置中 buffer_slots 槽留現金（0 報酬），其餘 0050；不計調整成本
        if buffer_slots:
            idle_tot = cash + u50 * E50[t]
            tgt_s = (equity(t)[0] / K) if mode == 'GROW' else tier_2026 * cpi_factor(t) / K
            want = min(buffer_slots * tgt_s, idle_tot)
            cash = want; u50 = (idle_tot - want) / E50[t]
        # 保留中的現金
        E, pv = equity(t)
        E += sum(a for _, a in resv)
        nav[t - t0] = E; inv[t - t0] = pv / E if E > 0 else 0
    s = pd.Series(nav, index=IDX[t0:t1 + 1])
    stat['mean_invested'] = round(float(inv.mean()), 3)
    return s, stat


def monthly(nav):
    m = nav.resample('ME').last()
    return m.pct_change().dropna()


def e50_nav(idx):
    return pd.Series(E50, index=IDX).reindex(idx)


def ann(nav):
    n = len(nav); yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else float('nan')


def mdd(nav):
    return float((nav / nav.cummax() - 1).min())


def vs0050(nav, a=None, b=None):
    nav = nav.dropna()
    if a: nav = nav[nav.index >= pd.Timestamp(a)]
    if b: nav = nav[nav.index <= pd.Timestamp(b)]
    if len(nav) < 40: return {'n_days': int(len(nav))}
    e = e50_nav(nav.index)
    ms, me = monthly(nav), monthly(e)
    x = (ms - me).dropna()
    return {'ann': round(ann(nav) * 100, 2), 'ann_0050': round(ann(e) * 100, 2), 'mdd': round(mdd(nav) * 100, 1),
            'mdd_0050': round(mdd(e) * 100, 1), 'ex_month_mean_pp': round(float(x.mean() * 100), 3), 't_vs_0050': tstat(x),
            'n_months': int(len(x))}


# ---------------- 安慰劑（P3） ----------------
def run6_vec(s):
    with np.errstate(invalid='ignore', over='ignore'):
        v = np.exp(LA[s - 1] - LA[s - 7]) - 1
    return np.where(TR[s - 1] & TR[s - 7], v, np.nan)


def run6_cuts(U):
    vals = []
    for r in U.itertuples():
        if r.in_D and 7 <= r.s_i < T:
            v = run6_vec(r.s_i)[CI[r.code]]
            if np.isfinite(v): vals.append(v)
    return np.quantile(np.array(vals), [0.2, 0.4, 0.6, 0.8])


def disp_intervals(U):
    iv = {}
    for r in U.itertuples():
        if r.in_D: iv.setdefault(CI[r.code], []).append((r.s_i, r.e_i))
    return iv


def placebo(df, U, cuts, iv=None):
    """回傳 df 加上 pl_raw（對照平均原始報酬）、pl_n、pl_mode（'bucket'／'pm5'／None）、bucket。"""
    iv = disp_intervals(U) if iv is None else iv
    out = df.copy(); pr, pn, pm, bk = [], [], [], []
    for r in df.itertuples():
        s, i0, i1 = r.s_i, r.i0, r.i1
        rv = run6_vec(s)
        b_ev = int(np.searchsorted(cuts, r.run6, side='right')) if np.isfinite(r.run6) else -1
        thr = limit_rate(s) - 0.01
        with np.errstate(invalid='ignore'):
            ok = LIQL[s] & TR[s] & np.isfinite(R[s]) & (R[s] < thr) & np.isfinite(rv)
        ok[r.j] = False
        cand = np.where(ok)[0]
        cand = [j for j in cand if not any(a <= i1 and b >= s - 30 for a, b in iv.get(j, []))]
        cand = np.array(cand, int)
        mode = None; sel = np.array([], int)
        if len(cand) and b_ev >= 0:
            bb = np.searchsorted(cuts, rv[cand], side='right')
            sel = cand[bb == b_ev]; mode = 'bucket'
            if len(sel) == 0:
                sel = cand[np.abs(rv[cand] - r.run6) <= 0.05]; mode = 'pm5'
        if len(sel) == 0:
            pr.append(np.nan); pn.append(0); pm.append(None); bk.append(b_ev); continue
        e = np.minimum(i1, LASTV[sel])
        rr = np.exp(LA[e, sel] - LA[i0, sel]) - 1
        pr.append(float(np.mean(rr))); pn.append(int(len(sel))); pm.append(mode); bk.append(b_ev)
    out['pl_raw'] = pr; out['pl_n'] = pn; out['pl_mode'] = pm; out['bucket'] = bk
    out['ev_minus_pl'] = out['raw'] - out['pl_raw']
    return out


def path_split(df):
    """P5：每筆從進場收盤到第 k 日收盤的累積超額（毛）；early＝到 floor(N/2)，late＝其餘（N＝計畫出場與進場的交易日差）。"""
    E, Lt, N = [], [], []
    for r in df.itertuples():
        n = r.e_i - r.i0
        if n < 2 or r.ek != 'C':
            E.append(np.nan); Lt.append(np.nan); N.append(n); continue
        h = n // 2; j = r.j
        a = lambda k: math.expm1(LA[r.i0 + k, j] - LA[r.i0, j]) - math.expm1(LEW[r.i0 + k] - LEW[r.i0])
        e = a(h); f = a(n)
        E.append(e); Lt.append(f - e); N.append(n)
    o = df.copy(); o['early'] = E; o['late'] = Lt; o['late_m_early'] = o['late'] - o['early']; o['N'] = N
    return o


# ---------------- PREREG 斷言 ----------------
RES_F = HERE / 'results.json'


def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def check_prereg():
    res = json.loads(RES_F.read_text(encoding='utf-8'))
    assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：先跑 s3_prereg.py'
    assert canon_sha(res['PREREG']) == res['PREREG_sha256'], 'PREREG 被改過'
    for rel, h in res['PREREG']['hashes'].items():
        p = Path(rel) if rel.startswith('/') else HERE / rel
        assert sha(p) == h, f'{rel} 與 PREREG 登記的 sha256 不符'
    return res

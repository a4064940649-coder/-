#!/usr/bin/env python3
"""S89 共用程式庫：「進場後很快就不對勁」的部位提早處理（時間停損／沒動就走／減碼版）。
引擎：正式管線（agents/official_nav/run_official.py 的 E0 寫法；五個開關全開：EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK、REDUCE_FIX、0050 官方含息）。
非訊號日出場：s10lib／s8lib 沒有「進場後第 N 天看累計報酬」的掛鉤（只有 trail 移動停損、requal、每日 X）。
  → 照 S57 的做法：inspect 取 s10lib.simulate_s10 原始碼，字串替換加掛鉤，exec 成 simulate_s89（原檔不動）。掛鉤全關（tstop=None）時要與 simulate_s10 逐日差 0（驗收 A1）。
掛鉤 tstop＝dict(n, x, strict, half)：
  進場日算第 1 個交易日；第 n 個交易日（列號 t＝進場列 e＋n−1）收盤時，引擎的持有期含息累計指數 g（＝進場日收盤÷進場日開盤，之後每天 ×(1＋R)；不含買進成本）
  若 g ≤ 1−x（strict＝True 時改為 g < 1）→ 標記 'tstop'，次一交易日開盤賣（無開盤價或開盤 ≤ 跌停價順延，同引擎原規則）。
  half＝False：全賣；half＝True：只賣一半股數（2020-10-26 盤中零股上路前，股數向下取整到 1000 股，不足一張就不賣、標記取消），賣完解除標記，剩下一半照現行規則。
  賣得的錢照引擎原邏輯當天掃進 0050，到下個訊號日才可能再買股票（下個訊號日若仍在買進名單可買回）。
  同一天訊號列的現行出場（X）優先；被標記減碼但尚未成交時若遇到現行出場，升級成全賣。
暫存：scratchpad/s89/cache（面板快取，只讀原始日 K；用完刪）。一律 python3 -I -B。"""
import sys, gc, hashlib, inspect
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s89/cache')
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)

CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
SEGMENTS = {'2018': ('2018-01-01', '2018-12-31'), '2020-03': ('2020-01-01', '2020-06-30'),
            '2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}      # 同 s17lib.SEGMENTS
SEGV = ('2022', '2024-07~2025-04')
VALID_FIRST_MONTH = pd.Period('2021-12', 'M')              # 驗證期：營收月份 ≥ 2021-12（訊號日 2022-01 起；同 s20lib）
TARGET = dict(since2019=0.4886246519182116, full=0.3446294016617564, maxdd=-0.3160938700293674, n_closed=381, paths_median=0.4641920784536262)   # official_nav/out/res_official.json

# ---------------------------------------------------------------- 格子（26 格，事前固定）
CELLS = {}
for _n in (3, 5, 10):
    for _x in (3, 5, 8, 10):
        CELLS[f'T{_n}_{_x}'] = dict(n=_n, x=_x / 100, strict=False, half=False)
CELLS['NM10'] = dict(n=10, x=0.0, strict=True, half=False)
for _k in list(CELLS):
    CELLS[_k + 'h'] = dict(CELLS[_k], half=True)
assert len(CELLS) == 26

# ---------------------------------------------------------------- 引擎副本（原檔不動）
_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s89("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, tstop=None):"),
    ("""            sh = int(round(p['sh'] * frac))
            sl = 0.5 * cm * float(tick(o)) / o + xs
            if ox > 0 and sh > 0 and t >= odd_t:
""",
     """            if p.get('half'):
                frac = min(frac, 0.5)
            sh = int(round(p['sh'] * frac))
            if p.get('half'):
                if t < odd_t:
                    sh = (sh // 1000) * 1000
                if sh <= 0 or sh >= p['sh']:
                    p['half'] = False; p['flag'] = False; p['why'] = ''; st['half_skip_lot'] = st.get('half_skip_lot', 0) + 1
                    continue
                frac = sh / p['sh']
            sl = 0.5 * cm * float(tick(o)) / o + xs
            if ox > 0 and sh > 0 and t >= odd_t:
"""),
    ("""                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0)
        if sat_w < 1.0 and t in exec_days:
""",
     """                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0)
                if p.get('half'):
                    p['half'] = False; p['flag'] = False; p['why'] = ''; st['half_sold'] = st.get('half_sold', 0) + 1
                    st.setdefault('ts_fills', []).append((int(t), int(j), int(p['e']), float(frac), float(gross - f - tx)))
        if sat_w < 1.0 and t in exec_days:
"""),
    ("""            if not p['flag']:
                if X[t, j]:
""",
     """            if p['flag'] and p.get('half') and X[t, j]:
                p['half'] = False; p['why'] = 'rule' if is_sig else 'ma_daily'
            if not p['flag']:
                if X[t, j]:
"""),
    ("""                elif requal is not None and is_sig and p['nm'] >= requal and j not in RQ.get(t, ()):
""",
     """                elif tstop is not None and t - p['e'] == tstop['n'] - 1 and (p['g'] < 1.0 if tstop['strict'] else p['g'] <= 1.0 - tstop['x']):
                    p['flag'] = True; p['why'] = 'tstop'; p['half'] = bool(tstop['half'])
                    st.setdefault('ts_events', []).append((int(t), int(j), int(p['e']), float(p['g']), bool(tstop['half'])))
                elif requal is not None and is_sig and p['nm'] >= requal and j not in RQ.get(t, ()):
"""),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:50]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s89 simulate_s89>', 'exec'), _G)
simulate_s89 = _G['simulate_s89']
SRC_SHA = dict(simulate_s10=hashlib.sha256(_SRC0.encode()).hexdigest(), simulate_s89=hashlib.sha256(_SRC.encode()).hexdigest())


def load():
    """逐行照 official_nav/run_official.py 的載入（只留 E0 需要的）。回傳 P, U, RT。"""
    assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
    assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
    cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
    cslib.REDUCE_FIX = True; cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV
    P = s19lib.load()
    assert P.exright_fix and P.reduce_fix and P.e50_source == str(cslib.E50_OFFICIAL_CSV)
    s19lib.e50_fix_suspension(P)
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    del INNOV
    S, _ = s19lib.rev_std(P)
    avA = s19lib.a10_rows(P, list(S.index))
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
        if hasattr(P, a):
            delattr(P, a)
    gc.collect()
    return P, U, dict(S=S, av=avA)


def signal(P, U, RT, shift):
    return s10lib.make_signal_s10(P, U, RT, shift=shift)


def fresh_from(S, first_month=VALID_FIRST_MONTH):
    """同 s20lib.fresh_from：只保留營收月份 ≥ first_month 的進場訊號；出場 X 不動。"""
    S2 = dict(S)
    sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first_month:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2


def sim(P, S, U, cell=None, cap=1e6, cost='strict', block=None, orig=False):
    """W10_70、嚴成本、空位 0050。cell＝None → 現行（tstop 關）。orig＝True 用原引擎 s10lib.simulate_s10（驗收用）。"""
    kw = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], capital=cap, block=block)
    if orig:
        assert cell is None
        return s10lib.simulate_s10(P, S, U, **kw)
    return simulate_s89(P, S, U, tstop=(CELLS[cell] if cell else None), **kw)


def seg_dd(nav, keys=tuple(SEGMENTS)):
    o = {}
    for k in keys:
        a, b = SEGMENTS[k]
        s = nav.loc[a:b].dropna()
        o[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return o


def early_path(P, lg, days=(3, 5, 10)):
    """每筆交易在進場後第 n 個交易日（進場日＝第 1 個）收盤的累計報酬（＝引擎的 g − 1：進場日開盤起、含息、不含成本），
    以及頭 n 個交易日內逐日收盤累計報酬的最低值。該日已不在手上（已賣或資料結束）→ NaN。"""
    out = {f'r{n}': [] for n in days}; out.update({f'min{n}': [] for n in days})
    for e, x, w, j in zip(lg['ent_t'], lg['exit_t'], lg['why'], lg['j']):
        last = x if w == 'open_end' else x - 1             # 最後一個收盤時仍持有的列
        g = [P.CF[e, j] / P.O[e, j]]
        for t in range(e + 1, min(e + max(days), last + 1)):
            g.append(g[-1] * (1 + P.R[t, j]))
        g = np.array(g) - 1
        for n in days:
            ok = len(g) >= n
            out[f'r{n}'].append(float(g[n - 1]) if ok else np.nan)
            out[f'min{n}'].append(float(g[:n].min()) if ok else np.nan)
    return pd.DataFrame(out, index=lg.index)


# ---------------------------------------------------------------- 逐筆對照工具（描述用，不影響淨值）
def gpath(P, j, e, t):
    """進場列 e 開盤起到列 t 收盤的含息累計指數（同引擎的 g）。"""
    g = P.CF[e, j] / P.O[e, j]
    for u in range(e + 1, t + 1):
        g *= (1 + P.R[u, j])
    return float(g)


def sell_open(P, j, d):
    """標記後的賣出列：從 d 起第一個有開盤價且開盤 > 跌停價的列；到資料結束都沒有 → None。"""
    while d < P.T:
        o = P.O[d, j]
        if np.isfinite(o) and o > 0 and not (o <= P.LIMDN[d, j] + 1e-9):
            return d
        d += 1
    return None


def gross_at_sell(P, j, e, d):
    """列 d 開盤賣出的毛報酬（不含成本）＝g(d−1) × 開盤 ÷ 參考價 − 1。"""
    return gpath(P, j, e, d - 1) * float(P.O[d, j] / P.REF[d, j]) - 1


def current_exit(P, S, j, e):
    """照現行規則（訊號列 X 成立 → 次日開盤賣）抱到出場：回傳 (賣出列或 None＝資料結束仍持有, 毛報酬)。不處理下市。"""
    ks = np.flatnonzero(S['sigrows'] & S['X'][:, j])
    ks = ks[ks >= e]
    d = sell_open(P, j, int(ks[0]) + 1) if len(ks) else None
    if d is None:
        return None, gpath(P, j, e, P.T - 1) - 1
    return d, gross_at_sell(P, j, e, d)


def stop_effect(P, S, j, e, n, x, strict, cost, frac=1.0):
    """單筆、靜態（不考慮資金與名額連動）的「第 n 天觸發就賣」對照：沒觸發回傳 None。
    回傳 dict：觸發時累計報酬 r_trig、賣出列 ds、賣出毛報酬 g_stop、現行出場列 dc 與毛報酬 g_cur、
    金額差 d_stock＝成本 × frac ×（g_stop − g_cur）、d_e50＝賣得金額放 0050（開盤到開盤）到現行出場日的報酬、d_total。"""
    t = e + n - 1
    dc, g_cur = current_exit(P, S, j, e)
    last = (dc - 1) if dc is not None else P.T - 1
    if t > last:
        return None
    r = gpath(P, j, e, t) - 1
    if not ((r < 0) if strict else (r <= -x)):
        return None
    if S['sigrows'][t] and S['X'][t, j]:
        return None                                        # 同一天現行出場優先
    ds = sell_open(P, j, t + 1)
    if ds is None or (dc is not None and ds >= dc):
        return dict(t=int(t), r_trig=r, ds=None, dc=dc, g_stop=g_cur, g_cur=g_cur, d_stock=0.0, d_e50=0.0, d_total=0.0)
    g_stop = gross_at_sell(P, j, e, ds)
    e_end = float(P.E50O[dc]) if dc is not None else float(P.E50[P.T - 1])
    d_stock = cost * frac * (g_stop - g_cur)
    d_e50 = cost * frac * (1 + g_stop) * (e_end / float(P.E50O[ds]) - 1)
    return dict(t=int(t), r_trig=r, ds=int(ds), dc=dc, g_stop=g_stop, g_cur=g_cur, d_stock=d_stock, d_e50=d_e50, d_total=d_stock + d_e50)

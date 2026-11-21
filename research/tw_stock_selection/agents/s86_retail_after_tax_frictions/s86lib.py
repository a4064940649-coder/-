#!/usr/bin/env python3
"""S86 共用程式庫：回測沒算、散戶實際會付的摩擦（股利稅、補充保費、股利入帳時間差、手續費折扣與低消、零股價差）。
描述性研究，不改策略規則。鏈同正式管線 official_nav/run_official.py（s19lib → s10lib；三個正式開關全開），原檔一律不動。
simulate_s86：用 inspect 取 s10lib.simulate_s10 原始碼做字串替換後 exec（寫法同 s62lib）。新參數全部預設＝原行為（驗收：逐位相同）：
  fee_rate ：手續費率（None＝0.1425%×0.28）。 minfee_v：單筆最低手續費（None＝20 元）。
  odd_from ：'hist'（2020-10-26 起才可買零股＝正式）、'all'（全期可買零股）。
  DY, dy50 ：現金股利率矩陣（T×N；除息日那格＝現金股利 ÷ 除息前收盤）與 0050 的同義序列（長度 T）。給了才記錄股利。
  div_tax  ：有效股利稅率（假設情境）；除息日從該部位扣「股利 × 稅率」。
  nhi_rate, nhi_thr：補充保費假設費率與單筆門檻；單筆（同一檔同一次）股利 ≥ 門檻時扣「股利 × 費率」。
  delay    ：股利延後幾個交易日入帳（0＝回測原假設：除息日以參考價再投入）。>0 時除息日把現金股利從部位移出、掛應收，
             到期日開盤前變成現金（達閒錢門檻就照引擎規則掃進 0050；淨值含應收）。
  start_t  ：起跑列號（None＝正式 2015-07-01；小帳戶「第一年」分析用，全部現金新起跑，寫法同 s62lib）。
  EXPY     ：T×N「該股當年現金股利率合計 ÷ 當年交易日數」，用來算「按持有時間比例應收到多少」（第 7 題）。
回傳 nav, lg, st；st['div'] 為逐日 DataFrame，st['pay'] 為逐筆股利（列號、欄號（−1＝0050）、金額、前一日淨值）。"""
import sys, inspect, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s86')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
FEE_FULL = 0.001425
ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, n_closed=381)

_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s86("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, fee_rate=None, minfee_v=None, odd_from='hist',\n"
     "                 DY=None, dy50=None, div_tax=0.0, nhi_rate=0.0, nhi_thr=20000.0, delay=0, EXPY=None, start_t=None):"),
    ("    T, t0 = P.T, P.t0\n", "    T, t0 = P.T, (P.t0 if start_t is None else int(start_t))\n"),
    ("    fee, tax, minfee = FEE * cm, TAX * cm, MINFEE * cm\n",
     "    fee, tax, minfee = (FEE if fee_rate is None else fee_rate) * cm, TAX * cm, (MINFEE if minfee_v is None else minfee_v) * cm\n"),
    ("    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))\n",
     "    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))\n"
     "    assert odd_from in ('hist', 'all')\n"
     "    if odd_from == 'all':\n"
     "        odd_t = 0\n"
     "    PT = [0.0]; pend = []; pay = []\n"
     "    dv = {k: np.zeros(T) for k in ('stk_gross', 'e50_gross', 'tax', 'nhi', 'stk_exp', 'stk_val', 'v50_val', 'nav_prev')}\n"),
    ("        return cash + v50 + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())\n",
     "        return PT[0] + cash + v50 + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())\n"),
    ("    for t in range(t0 + 1, T):\n        if CLOSE:                                              # close：0050、底倉、持股直接走到 t 日收盤\n",
     "    for t in range(t0 + 1, T):\n"
     "        while pend and pend[0][0] <= t:\n"
     "            _d, _a, _b = pend.pop(0); cash += _a; B += _b; PT[0] -= _a\n"
     "        if DY is not None:\n"
     "            dv['nav_prev'][t] = nav[t - 1]; dv['v50_val'][t] = v50; dv['stk_val'][t] = sum(p['v'] for p in pos.values())\n"
     "            d5 = dy50[t]\n"
     "            if d5 > 0 and v50 > 0:\n"
     "                g = v50 * d5; tx_ = g * div_tax; nh_ = g * nhi_rate if g >= nhi_thr else 0.0; ded = tx_ + nh_\n"
     "                dv['e50_gross'][t] += g; dv['tax'][t] += tx_; dv['nhi'][t] += nh_; pay.append((t, -1, g, nav[t - 1]))\n"
     "                bs = min(1.0, B / v50)\n"
     "                if delay > 0:\n"
     "                    v50 -= g; B -= bs * g; pend.append((t + delay, g - ded, bs * (g - ded))); PT[0] += g - ded\n"
     "                else:\n"
     "                    v50 -= ded; B -= bs * ded\n"
     "            for j, p in pos.items():\n"
     "                if EXPY is not None:\n"
     "                    dv['stk_exp'][t] += p['v'] * EXPY[t, j]\n"
     "                d = DY[t, j]\n"
     "                if d > 0:\n"
     "                    g = p['v'] * d; tx_ = g * div_tax; nh_ = g * nhi_rate if g >= nhi_thr else 0.0; ded = tx_ + nh_\n"
     "                    dv['stk_gross'][t] += g; dv['tax'][t] += tx_; dv['nhi'][t] += nh_; pay.append((t, int(j), g, nav[t - 1]))\n"
     "                    if delay > 0:\n"
     "                        p['v'] -= g; pend.append((t + delay, g - ded, 0.0)); PT[0] += g - ded\n"
     "                    else:\n"
     "                        p['v'] -= ded\n"
     "            pend.sort()\n"
     "        if CLOSE:\n"),
    ("        nav[t] = cash + v50 + tot\n", "        nav[t] = cash + v50 + tot + PT[0]\n"),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:50]}'
    _SRC = _SRC.replace(_a, _b)
# 把逐日股利紀錄塞進回傳的 st（找到函式最後的 st 組裝處較脆弱，改用包一層）
_G = dict(s10lib.__dict__)
_SRC = _SRC.replace("    st = dict(n_sig=0,", "    st = dict(_dv=None, _pay=None, n_sig=0,", 1)
assert "_dv=None" in _SRC
assert _SRC.count("    def px_of(t, k):\n") == 1
_SRC = _SRC.replace("    def px_of(t, k):\n", "    st['_dv'] = dv; st['_pay'] = pay\n\n    def px_of(t, k):\n", 1)
exec(compile(_SRC, '<s86 simulate_s86>', 'exec'), _G)
simulate_s86 = _G['simulate_s86']


def setup(shifts=(0,)):
    """同 run_official.py 的載入順序。回傳 P, U, {位移: 訊號}。"""
    import gc
    P = s19lib.load(); assert P.exright_fix
    s19lib.e50_fix_suspension(P)
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    S, _ = s19lib.rev_std(P)
    avA = s19lib.a10_rows(P, list(S.index))
    SG = {s: s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=s) for s in shifts}
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
        if hasattr(P, a):
            delattr(P, a)
    gc.collect()
    return P, U, SG


def div_arrays(P):
    """現金股利率：DY（種類＝息，值＝(前收−參考價)÷前收）、DYM（種類＝權息，權值＋息值合計 ÷ 前收，現金與股票無法拆開，只當上限用）、
    NQ（種類＝權 的事件旗標）、dy50（0050；eng_audit_fixes_2026-11/e50/e50_events.json 的 exright 息事件）。"""
    ev = cslib.load_twt49u()
    ci = {c: j for j, c in enumerate(P.COLS)}; di = {d: i for i, d in enumerate(P.IDX)}
    e = ev[ev['code'].isin(ci) & ev['date'].isin(di)].copy()
    e['t'] = e['date'].map(di); e['j'] = e['code'].map(ci)
    e['y'] = (e['prev_close'] - e['ref_price']) / e['prev_close']
    DY = np.zeros((P.T, P.N)); DYM = np.zeros((P.T, P.N)); NQ = np.zeros((P.T, P.N), bool)
    a = e[e['type'] == '息']; a = a[(a['y'] > 0) & (a['y'] < 0.5)]
    DY[a['t'].to_numpy(), a['j'].to_numpy()] = a['y'].to_numpy()
    b = e[e['type'] == '權息']; b = b[(b['y'] > 0) & (b['y'] < 0.6)]
    DYM[b['t'].to_numpy(), b['j'].to_numpy()] = b['y'].to_numpy()
    c = e[e['type'] == '權']
    NQ[c['t'].to_numpy(), c['j'].to_numpy()] = True
    evs = json.loads((AG / 'eng_audit_fixes_2026-11' / 'e50' / 'e50_events.json').read_text())['events']
    dy50 = np.zeros(P.T); e50ev = []
    for x in evs:
        if x['kind'] == 'exright' and x['label'] == '息':
            d = pd.Timestamp(x['date'])
            if d in di:
                dy50[di[d]] = x['value'] / x['pre_close']; e50ev.append((x['date'], x['value'], x['pre_close']))
    info = dict(n_cash=int(len(a)), n_mixed=int(len(b)), n_stock_only=int(len(c)), n_e50=int((dy50 > 0).sum()), types=e['type'].value_counts().to_dict())
    return DY, DYM, NQ, dy50, e50ev, info


def expy(P, DY):
    """每格＝該股該曆年現金股利率合計 ÷ 該年交易日數（面板內）。"""
    yrs = P.IDX.year.to_numpy(); out = np.zeros_like(DY)
    for y in np.unique(yrs):
        m = yrs == y
        out[m] = DY[m].sum(0)[None, :] / m.sum()
    return out


def bench50(P, dy50, cap=1e6, div_tax=0.0, nhi_rate=0.0, nhi_thr=20000.0, delay=0, fee_rate=None, minfee_v=None):
    """只買 0050 並抱著（起始日以還原價全額持有，同正式基準；不計初次買進成本）。
    delay＝0：稅與保費在除息日從部位扣，其餘以參考價再投入（全 0 時＝正式基準 P.E50）。
    delay>0：除息日把現金股利移出，到期日開盤買回（付手續費（有低消）＋半個跳動點）。"""
    fee = stlib.FEE if fee_rate is None else fee_rate; mf = stlib.MINFEE if minfee_v is None else minfee_v
    T, t0 = P.T, P.t0
    nav = np.full(T, np.nan); V = float(cap); nav[t0] = V; pend = []; PT = 0.0
    tot = dict(gross=0.0, tax=0.0, nhi=0.0); pay = []
    for t in range(t0 + 1, T):
        add = 0.0
        while pend and pend[0][0] <= t:
            _, a = pend.pop(0); PT -= a
            po = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
            add += (a - max(mf, a * fee)) / (1 + 0.5 * stlib.etf_tick(po) / po)
        d5 = dy50[t]
        if d5 > 0:
            g = V * d5; tx = g * div_tax; nh = g * nhi_rate if g >= nhi_thr else 0.0
            tot['gross'] += g; tot['tax'] += tx; tot['nhi'] += nh; pay.append((t, -1, g, nav[t - 1]))
            if delay > 0:
                V -= g; pend.append((t + delay, g - tx - nh)); PT += g - tx - nh
            else:
                V -= tx + nh
        V = V * P.E50[t] / P.E50[t - 1] + add * P.E50[t] / P.E50O[t]
        nav[t] = V + PT
    return pd.Series(nav, index=P.IDX), tot, pay

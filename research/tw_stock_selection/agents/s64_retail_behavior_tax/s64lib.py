#!/usr/bin/env python3
"""S64 共用程式庫：散戶行為偏差與稅後報酬（描述性，不改策略）。
鏈：s20lib → s17lib → s10lib（全部 import、不改）；正式開關 cslib.EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開；載入與訊號逐行照
agents/s56_industry_cap_family/run.py（其現行格逐位重現 deadline_engine_fix ALL＝48.76%／381 筆）。
引擎副本 simulate_s64：inspect 取 s10lib.simulate_s10 原始碼，只做字串替換後 exec（原檔不動），新參數 beh（dict，None＝與原引擎逐位相同）：
    gap＝x：買進時當日開盤 ÷ 參考價 − 1 > x → 不買（不補買）。
    stop＝(X, RES)：收盤淨值 < 自己的高點 ×(1−X) → 停手：所有持股標記賣出（次日開盤賣，跌停或無開盤順延）、不買、不做底倉拉回，
          賣得的錢照原引擎掃進 0050；RES[t]＝停手日 t 起滿 M 個月後第一個交易日列號；收盤淨值 > 停手前高點或 t ≥ RES → 恢復
          （高點重設為恢復當天淨值；之後遇到訊號列才照當月名單買）。
    DY／DY50：除息事件（只記錄、不改任何成交）：DY[t]＝{j: 現金股利 ÷ 除息前收盤}、DY50[t]＝0050 同義；記在 st['div_ev']、st['div50_ev']。
漏做與挑股不需要改引擎：直接改訊號（skip_months／pick_half）。
中間檔放 scratchpad/s64/cache（結束刪除）。規格見 SPEC.json、PREREG.json。一律 python3 -I -B。"""
import sys, gc, inspect, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s64')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False and cslib.EXRIGHT_FIX is False, '共用引擎預設應全關（本題自己開）'

CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
PATHS = [(c, s) for s in SHIFTS for c in CAPS]             # 20 條；路徑編號 i → PATHS[i]
pkey = lambda c, s: f'{int(c/1e4)}萬|+{s}'
ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, maxdd=-0.32119049020243395, n_closed=381)   # deadline_engine_fix ALL

# ---- 引擎副本（原始碼字串替換，原檔不動）
_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s64("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, beh=None):"),
    ("    assert exec_mode == 'open'\n",
     """    assert exec_mode == 'open'
    beh = beh or {}
    gap = beh.get('gap'); stop = beh.get('stop'); DY = beh.get('DY'); DY50 = beh.get('DY50')
    stopped = False; peak = float(capital); resume_t = -1
    div_ev = []; div50_ev = []; stop_ev = []
"""),
    ("""            if o >= P.LIMUP[t, j] - 1e-9:
                st['blk_limup'] += 1; continue
""",
     """            if o >= P.LIMUP[t, j] - 1e-9:
                st['blk_limup'] += 1; continue
            if gap is not None and o / REF[t, j] - 1.0 > gap:
                st['blk_gap'] = st.get('blk_gap', 0) + 1; continue
"""),
    ("""    for t in range(t0 + 1, T):
""",
     """    for t in range(t0 + 1, T):
        if DY50 is not None and v50 > 0 and DY50[t] > 0:
            div50_ev.append((int(t), float(v50 * DY50[t])))
        if DY is not None and t in DY:
            for j_, y_ in DY[t].items():
                if j_ in pos:
                    div_ev.append((int(t), int(j_), float(pos[j_]['v'] * y_)))
"""),
    ("""        if sat_w < 1.0 and t in exec_days:
            rebalance(t)
""",
     """        if sat_w < 1.0 and t in exec_days and not stopped:
            rebalance(t)
"""),
    ("""        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
""",
     """        cand = np.flatnonzero(sig[t - 1] & univ[t - 1]) if not stopped else np.zeros(0, int)
"""),
    ("""        nav[t] = cash + v50 + tot
""",
     """        nav[t] = cash + v50 + tot
        if stop is not None:
            if not stopped:
                if nav[t] > peak:
                    peak = nav[t]
                elif nav[t] < peak * (1.0 - stop[0]):
                    stopped = True; resume_t = int(stop[1][t])
                    stop_ev.append([int(t), float(nav[t] / peak - 1), None, None])
            if stopped:
                for p in pos.values():
                    if not p['flag']:
                        p['flag'] = True; p['why'] = 'panic'
                if t > stop_ev[-1][0] and (nav[t] > peak or t >= resume_t):
                    stopped = False; stop_ev[-1][2] = int(t); stop_ev[-1][3] = 'newhigh' if nav[t] > peak else 'time'
                    peak = nav[t]
"""),
    ("""    navs = pd.Series(nav[t0:T], index=P.IDX[t0:T])
""",
     """    st['div_ev'] = div_ev; st['div50_ev'] = div50_ev; st['stop_ev'] = stop_ev
    navs = pd.Series(nav[t0:T], index=P.IDX[t0:T])
"""),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s64 simulate_s64>', 'exec'), _G)
simulate_s64 = _G['simulate_s64']
SRC_SHA = dict(simulate_s10=hashlib.sha256(_SRC0.encode()).hexdigest(), simulate_s64=hashlib.sha256(_SRC.encode()).hexdigest())


def load():
    """逐行照 s56 run.py 的載入順序。回傳 (P, U, RT)。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    cslib.EXRIGHT_FIX = True
    P = s20lib.load()
    assert P.exright_fix
    assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    del INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    s17lib.e50_fix_suspension(P)
    P.AMT20 = pd.DataFrame(np.where(np.isfinite(P.AMT), P.AMT, 0.0)).rolling(20, min_periods=1).mean().to_numpy()   # 挑股「成交值最大」用（訊號日（含）前 20 日平均成交值）
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
        if hasattr(P, a):
            delattr(P, a)
    S0 = F['S']['STD']; months = list(S0.index)
    del F; gc.collect()
    cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
    AV = cslib.rev_avail_days(P, months, True)
    return P, U, dict(S=S0, av=AV)


def base_signal(P, U, RT, shift):
    sg = s10lib.make_signal_s10(P, U, RT, shift=shift, depth=10, mom_lo=0.10, nsel=10)
    S2, ekw = s10lib.variant_S(sg, 'E0')
    assert ekw == {}
    return S2


def skip_months(S, skip_k):
    """漏做：skip_k（訊號列集合）那幾個換股月完全沒操作——不買（sig 清空）、不賣（X 清空）、不做底倉拉回（從 info 拿掉 → 不是執行日）。"""
    if not skip_k:
        return S
    S2 = dict(S)
    sig = S['sig'].copy(); X = S['X'].copy(); sr = S['sigrows'].copy()
    ks = sorted(skip_k)
    sig[ks, :] = False; X[ks, :] = False; sr[ks] = False
    S2['sig'] = sig; S2['X'] = X; S2['sigrows'] = sr
    S2['info'] = [x for x in S['info'] if int(x['k']) not in skip_k]
    return S2


def pick_half(P, S, mode, rng=None):
    """挑股：每個訊號列的 F1 名單（n 檔）只留 ceil(n/2) 檔可買（其餘 sig 清掉；出場 X、名次、每檔金額都不動）。
    mode：'rand'（rng 隨機）、'lowpx'（訊號日原始收盤最低）、'highpx'（最高）、'bigamt'（前 20 日平均成交值最大）、'smallamt'（最小）。"""
    S2 = dict(S); sig = S['sig'].copy()
    nfull = nkeep = 0
    for x in S['info']:
        k = int(x['k'])
        tgt = np.flatnonzero(S['sig'][k])
        n = len(tgt)
        if n == 0:
            continue
        m = (n + 1) // 2
        if mode == 'rand':
            keep = rng.choice(tgt, size=m, replace=False)
        else:
            v = {'lowpx': P.CF[k, tgt], 'highpx': -P.CF[k, tgt], 'bigamt': -P.AMT20[k, tgt], 'smallamt': P.AMT20[k, tgt]}[mode]
            keep = tgt[np.argsort(np.nan_to_num(v, nan=np.inf), kind='stable')[:m]]
        sig[k, :] = False; sig[k, keep] = True
        nfull += n; nkeep += m
    S2['sig'] = sig; S2['pick_stats'] = dict(n_full=nfull, n_keep=nkeep)
    return S2


def resume_rows(P, months):
    """RES[t]＝P.IDX[t] 加 months 個月（日曆）當天（含）以後第一個交易日的列號。"""
    idx = pd.DatetimeIndex(P.IDX)
    tgt = (idx + pd.DateOffset(months=months)).values
    return np.searchsorted(idx.values, tgt, side='left')


def load_dividends(P):
    """TWT49U → 股票 DY_main／DY_low（dict t → {j: 殖利率}）與 0050 的 DY50（長度 T 陣列）、事件表。
    殖利率＝value ÷ prev_close（除權息日在面板交易日、代號在面板內才收）。
    main：type＝'息' 全額＋'權息' 全額當現金（上界）；low：只算 '息'（'權息' 當 0，下界）；'權' 兩者都不算。"""
    ev = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in sorted(cslib.TWT49U_DIR.glob('*/*.csv.gz'))], ignore_index=True)
    ev['date'] = pd.to_datetime(ev['date'])
    ev = ev.drop_duplicates(['date', 'code'], keep='last')
    tix = {d: i for i, d in enumerate(pd.DatetimeIndex(P.IDX))}
    col = {str(c): j for j, c in enumerate(P.COLS)}
    DYm, DYl = {}, {}
    DY50 = np.zeros(P.T)
    n = dict(cash=0, mixed=0, right=0, off_calendar=0)
    for d, code, pc, val, typ in ev[['date', 'code', 'prev_close', 'value', 'type']].itertuples(index=False):
        t = tix.get(d)
        if t is None:
            n['off_calendar'] += int(code in col or code == '0050'); continue
        if not (pc > 0 and val > 0):
            continue
        y = float(val / pc)
        if code == '0050':
            assert typ == '息'; DY50[t] = y; continue
        j = col.get(code)
        if j is None:
            continue
        if typ == '息':
            DYm.setdefault(t, {})[j] = y; DYl.setdefault(t, {})[j] = y; n['cash'] += 1
        elif typ == '權息':
            DYm.setdefault(t, {})[j] = y; n['mixed'] += 1
        else:
            n['right'] += 1
    return DYm, DYl, DY50, n

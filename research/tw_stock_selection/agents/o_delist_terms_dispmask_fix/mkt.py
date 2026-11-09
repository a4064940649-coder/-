#!/usr/bin/env python3
"""市場證據判準（SPEC_draft.json market_criteria 的實作；只讀 import V7X x_data）。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
AG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AG / 'o_exec_engine_rewrite_v7x')); import x_data as D; sys.path.pop(0)
IDX = D.IDX


def lt_index(code, last_trade, term):
    j = D.CI.get(code)
    if j is None: return None, None
    lim = max(pd.Timestamp(last_trade), pd.Timestamp(term)) if isinstance(term, str) and term else pd.Timestamp(last_trade) + pd.Timedelta(days=14)
    w = np.flatnonzero(D.TR[:, j]); w = w[IDX[w] <= lim]
    return j, (int(w[-1]) if len(w) else None)


def screen(code, last_trade, term, succ=None):
    j, lt = lt_index(code, last_trade, term)
    if j is None or lt is None: return dict(code=code, status='NO_PANEL')
    c = D.C[:, j]
    win30 = np.arange(max(0, lt - 29), lt + 1)
    out = dict(code=code, lt_panel=str(IDX[lt].date()), lc=float(c[lt]))
    gap_days = (pd.Timestamp(last_trade) - IDX[lt]).days
    w20 = np.arange(lt - 19, lt + 1)
    nmiss20 = int(np.sum(~np.isfinite(c[w20])))
    out['gap'] = bool(gap_days >= 7 or nmiss20 >= 5); out['gap_days'] = int(gap_days); out['nmiss20'] = nmiss20
    # 鎖價：最後 10 個「有收盤」交易日
    tv = np.flatnonzero(np.isfinite(c[:lt + 1]))[-11:]
    cl = c[tv]; r = cl[1:] / cl[:-1] - 1
    out['lock_frac'] = float(np.mean(np.abs(r) <= 0.005)); out['lock_width'] = float((cl[1:].max() - cl[1:].min()) / cl[1:].min())
    out['lock'] = bool(out['lock_frac'] >= 0.8 and out['lock_width'] <= 0.02)
    out['last10_close'] = [float(x) for x in cl[1:]]
    out['track_R2'] = None; out['track'] = False; out['ratio_med10'] = None
    rows = pd.DataFrame({'date': IDX[win30], 'target': c[win30]})
    if succ and succ in D.CI:
        s = D.C[:, D.CI[succ]]
        rows['succ'] = s[win30]
        tv = np.flatnonzero(np.isfinite(c[:lt + 1]) & np.isfinite(s[:lt + 1]))[-21:]
        tv = tv[tv > lt - 40]
        if len(tv) < 11:
            out['track_note'] = f'存續公司在 lt 前 40 日內共同有價日只有 {len(tv)} 日（新設控股或不在面板）→ 無法檢驗跟隨'
            out['market'] = 'MARKET_LOCK' if out['lock'] else 'MARKET_NONE'
            if out['gap']: out['market'] = 'GAP|' + out['market']
            return out, rows
        rt = c[tv][1:] / c[tv][:-1] - 1; rs = s[tv][1:] / s[tv][:-1] - 1
        if np.std(rs) > 0 and np.std(rt) > 0:
            R2 = float(np.corrcoef(rt, rs)[0, 1] ** 2)
        else:
            R2 = 0.0
        out['track_R2'] = R2; out['track'] = bool(R2 >= 0.6)
        out['beta'] = float(np.polyfit(rs, rt, 1)[0]) if np.std(rs) > 0 else None
        t10 = tv[-10:]
        out['ratio_med10'] = float(np.median(c[t10] / s[t10]))
        out['succ_ret_last20'] = float(s[tv[-1]] / s[tv[0]] - 1); out['target_ret_last20'] = float(c[tv[-1]] / c[tv[0]] - 1)
        out['succ_lt_close'] = float(s[lt])
    out['market'] = 'MARKET_TRACK' if out['track'] else ('MARKET_LOCK' if out['lock'] else 'MARKET_NONE')
    if out['gap']: out['market'] = 'GAP|' + out['market']
    return out, rows

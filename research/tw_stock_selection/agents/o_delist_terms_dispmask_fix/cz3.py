#!/usr/bin/env python3
"""consideration_v3 結算規則建構（SPEC 狀態表的實作）。輸出與 o_engine_delist_consideration/cz_engine.build_settlement 相同格式的 rule dict，
可直接餵 cz_engine._sim_cons（ia 紙上）與 cz_v7x.CSim（V7X）。不修改任何 o17 檔案（只讀 import cz_engine）。
build(E2, E3, bound, TR, IDX, Cn, CI, On, scenario=None)
  bound ∈ {'main','lo','hi'}；δb＝0／−δ_evt／＋δ_evt（只作用於非 VERIFIED_TEXT 的市場估計值）。
  scenario（事前可行性檢查用，不是正式口徑）：'best'＝16 起未核實事件一律 fixed lc×(1+δb)、不歸零；'worst'＝非 VERIFIED_TEXT 一律 lo 歸零。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
CZ = Path(__file__).resolve().parent.parent / 'o_engine_delist_consideration'
sys.path.insert(0, str(CZ)); import cz_engine as CE; sys.path.pop(0)


def _lt(j, term, TR, IDX):
    w = np.flatnonzero(TR[:, j])
    if term is not None: w = w[IDX[w] <= term]
    return int(w[-1]) if len(w) else -1


def build(E2, E3, bound, TR, IDX, Cn, CI, On, scenario=None, d_other=0.096):
    out = CE.build_settlement(E2[~E2.code.isin(set(E3.code))], bound, d=d_other, TR=TR, IDX=IDX, Cn=Cn, CI=CI, On=On)
    for r in E3.itertuples():
        j = CI.get(r.code)
        if j is None: continue
        ltU = pd.Timestamp(r.last_trade)
        term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) and r.terminate_date else ltU + pd.Timedelta(days=14)
        lt = _lt(j, max(term, ltU), TR, IDX)
        if lt < 0 or lt + 1 >= len(IDX): continue
        lc = float(Cn[lt, j]); ti = int(np.searchsorted(IDX.values, np.datetime64(term))); pay = min(ti + 20, len(IDX) - 1)
        de = {'main': 0.0, 'lo': -r.delta_evt, 'hi': r.delta_evt}[bound]
        rule = dict(code=r.code, lt=lt, trig=lt + 1, lc=lc, type=f'v3:{r.status}:{r.settle}', pay=pay, term=ti, cash_0050=False, parts=[])
        if IDX[lt] < ltU - pd.Timedelta(days=7):
            rule['type'] = 'price_gap'; rule['parts'].append(dict(kind='fixed_sell', per_share=lc, rel=lt + 1))
            out.setdefault(j, []).append(rule); continue
        sj = CI.get(r.successor) if isinstance(r.successor, str) and r.successor else None
        sfirst = None
        if sj is not None and isinstance(r.succ_first_trade, str) and r.succ_first_trade:
            w = np.flatnonzero(TR[:, sj] & np.isfinite(On[:, sj])); w = w[w >= int(np.searchsorted(IDX.values, np.datetime64(pd.Timestamp(r.succ_first_trade))))]
            sfirst = int(w[0]) if len(w) else None
        st = r.settle
        if scenario == 'best' and r.text_status != 'VERIFIED_TEXT': st = 'fixed'
        if scenario == 'worst' and r.text_status != 'VERIFIED_TEXT': st = 'unknown'
        if st == 'cash':
            rule['parts'].append(dict(kind='cash', per_share=float(r.cash), rel=pay))
        elif st in ('swap', 'mixed') and sj is not None and sfirst is not None:
            rule['parts'].append(dict(kind='stock', succ=sj, ratio=float(r.ratio_text), rel=sfirst, per_share_window=lc))
            if st == 'mixed' and np.isfinite(r.cash if r.cash is not None else np.nan): rule['parts'].append(dict(kind='cash', per_share=float(r.cash), rel=pay))
            if st == 'mixed' and bound == 'hi' and np.isfinite(r.pref_ratio if r.pref_ratio is not None else np.nan):
                rule['parts'].append(dict(kind='fixed_sell', per_share=float(r.pref_ratio) * 10.0, rel=sfirst))   # 特別股價值未核實：只在 hi 以面額 10 計
        elif st == 'track' and sj is not None and sfirst is not None:
            rule['parts'].append(dict(kind='stock', succ=sj, ratio=float(r.ratio_market) * (1 + de), rel=sfirst, per_share_window=lc))
        elif st == 'conv_newsucc' and sj is not None and sfirst is not None:
            rule['parts'].append(dict(kind='stock', succ=sj, ratio=lc * (1 + de) / float(On[sfirst, sj]), rel=sfirst, per_share_window=lc))
        elif st in ('fixed', 'swap', 'mixed', 'track', 'conv_newsucc'):
            rule['type'] += '->fixed'; rule['parts'].append(dict(kind='cash', per_share=lc * (1 + de), rel=pay))
        elif st == 'unknown':
            if bound == 'lo': rule['parts'].append(dict(kind='fixed_sell', per_share=lc, rel=lt + 1, zero=True))
            else: rule['parts'].append(dict(kind='cash', per_share=lc, rel=pay))
        else:
            raise ValueError(st)
        out[j] = [x for x in out.get(j, []) if x['lt'] != lt] + [rule]
    return out


def load_E():
    H = Path(__file__).resolve().parent
    E2 = pd.read_csv(CZ / 'events_v2.csv', dtype={'code': str, 'successor': str})
    E3 = pd.read_csv(H / 'events_v3.csv', dtype={'code': str, 'successor': str})
    E3 = E3.astype(object).where(pd.notna(E3), None)
    for k in ('cash', 'ratio_text', 'ratio_market', 'pref_ratio', 'delta_evt'):
        E3[k] = [float(x) if x is not None else np.nan for x in E3[k]]
    return E2, E3

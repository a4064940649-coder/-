#!/usr/bin/env python3
"""反方審查（成本與可成交）o17：只讀，不改研究員檔案。python3 -I -B review_exec_o17.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'o_exec_engine_rewrite_v7x'))
import x_data as D
sys.path.pop(0)
E = pd.read_csv(HERE / 'events_v2.csv', dtype=str)
IDX = D.IDX
def px(code, a, b):
    j = D.CI.get(code)
    if j is None: return None
    s = pd.Series(D.C[:, j], index=IDX).loc[a:b].dropna()
    return s
out = {}
# 1) acq_ur 有存續公司在面板者：若實際是換股，空窗後存續股價變動（固定現金請求權 vs 換股）
for code in ['2432', '2403', '2452', '2473', '3063', '2446', '3009', '3012', '2394', '2833']:
    r = E[E.code == code].iloc[0]
    succ = r.successor if isinstance(r.successor, str) else None
    lt = pd.Timestamp(r.last_trade); term = pd.Timestamp(r.terminate_date)
    ti = int(np.searchsorted(IDX.values, np.datetime64(term))); pay = IDX[min(ti + 20, len(IDX) - 1)]
    row = dict(type=r.type, lt=str(lt.date()), term=str(term.date()), pay=str(pay.date()), succ=succ)
    if succ and succ in D.CI:
        s = px(succ, lt - pd.Timedelta(days=3), pay + pd.Timedelta(days=3))
        if len(s):
            a = s.loc[:lt].iloc[-1] if len(s.loc[:lt]) else np.nan
            b = s.loc[term:].iloc[0] if len(s.loc[term:]) else np.nan
            c = s.loc[:pay].iloc[-1]
            row.update(succ_lt=float(a), succ_term=float(b), succ_pay=float(c), succ_ret_lt_to_term=float(b / a - 1), succ_ret_lt_to_pay=float(c / a - 1))
    j = D.CI.get(code)
    if j is not None:
        s = pd.Series(D.C[:, j], index=IDX).loc[:lt].dropna().tail(6)
        v = pd.Series(D.V[:, j] if hasattr(D, 'V') else np.full(len(IDX), np.nan), index=IDX).loc[:lt].tail(6) if hasattr(D, 'V') else None
        row['last6_close'] = [float(x) for x in s.values]
    e50 = pd.Series(D.E50, index=IDX)
    row['0050_lt_to_pay'] = float(e50.loc[:pay].iloc[-1] / e50.loc[:lt].iloc[-1] - 1)
    out[code] = row
# 2) 執行層才出現的 unknown 事件：最後幾日價量
for code in ['3007', '6280', '1520', '1716', '8008', '8078', '3559']:
    r = E[E.code == code].iloc[0]; lt = pd.Timestamp(r.last_trade)
    j = D.CI.get(code)
    if j is None: out['x' + code] = 'not in panel'; continue
    s = pd.Series(D.C[:, j], index=IDX).loc[:lt].dropna().tail(8)
    out['x' + code] = dict(type=r.type, lt=str(lt.date()), close=[round(float(x), 2) for x in s.values], dates=[str(x.date()) for x in s.index])
if "--v7x" not in sys.argv: print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

# ---- 第二部分：2432／2403／2452 改為 swap_ur（市場隱含比例；最後 6 日價格跟著存續股走、沒有鎖價 → 換股型證據）
#      只跑 V7X 100 萬一般成本，orig 與改分類後 cons_main；也跑 unknown 鎖價事件全部視為 cash@lc（＝main）不變。
if __name__ == '__main__' and '--v7x' in sys.argv:
    from multiprocessing import Pool
    sys.path.insert(0, str(HERE)); import cz_v7x as CV, cz_engine as CE; sys.path.pop(0)
    sys.path.insert(0, str(CV.V7X)); import x_stats as S; sys.path.pop(0)
    VR = json.loads((CV.V7X / 'results.json').read_text())
    BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
    E2 = pd.read_csv(HERE / 'events_v2.csv', dtype={'code': str, 'successor': str})
    E3 = E2.copy(); E3.loc[E3.code.isin(['2432', '2403', '2452']), 'type'] = 'swap_ur'
    kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
    ST = {'cons_main': CE.build_settlement(E2, 'main', d=0.057, **kw), 'swapfix_main': CE.build_settlement(E3, 'main', d=0.057, **kw),
          'swapfix_lo': CE.build_settlement(E3, 'lo', d=0.057, **kw)}
    globals()['ST'] = ST
    cfg0 = dict(tier=1e6, mode='GROW', model='K5T', cost='normal'); cfg0.update(BASE)
    def job(a):
        key, pl = a; o = {}
        for eg in ['orig'] + list(ST):
            o[eg] = CV.CSim(pl, dict(cfg0), settle=ST.get(eg), cmode='orig' if eg == 'orig' else 'consideration').run()
        return key, o
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    navs = {}
    with Pool(4) as pool:
        for key, o in pool.imap_unordered(job, sorted(P.items(), key=lambda kv: kv[0]), chunksize=2):
            for eg, nv in o.items(): navs.setdefault((key[0], eg), []).append((key[1], nv))
    TAB = {}
    for eg in ['orig'] + list(ST):
        per = [S.stats(S.avg_nav([nv for s, nv in sorted(navs[(j, eg)], key=lambda x: x[0])]), 'AB_V7_B10', j) for j in range(13)]
        TAB[eg] = {k: v for k, v in S.jsumm(per).items() if k in ('CAGR_med', 'exc_med', 't_0050_med', 'exc_p10', 'exc_Y19on_med')}
    print(json.dumps(TAB, indent=1))

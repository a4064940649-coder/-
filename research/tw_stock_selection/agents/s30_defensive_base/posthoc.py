#!/usr/bin/env python3
"""S30 事後描述（看過驗證結果後才寫；未登記、不進判定）：
(a) 其餘 3 條規則（R_LBQ、R_HDLV、R_MIX）與 R_LOWB 的驗證窗重新起跑、全期連續 20 條路徑中位數（年化、回撤、四段空頭、CAPM β）。
(b) 成本拆解：R_LOWB 的資金進出成本改成 0050 等級（稅 0.1%、滑價約 0.02%）→ 落後有多少是進出成本、多少是組合本身。
(c) 同樣 β 的笨方法：底倉＋空位＝0050 與定存（年 1%）混合（50／50、35／65，每月拉回），看低 β 組合是否比「少放一點 0050」更有效率。
輸出 results_posthoc.json。python3 -I -B posthoc.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s30lib
from s30lib import s17lib, s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
assert (H / 'results.json').exists(), '先跑 validate.py'
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s30lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
RT = s30lib.surge(P)
s17lib.e50_fix_suspension(P)
Hd, info, FR = s30lib.holdings(P, INNOV)
IDXS = {b: s30lib.basket_index(P, Hd[b], FR) for b in s30lib.BASKETS}
IDXS['MIX'] = s30lib.mix_index(P, IDXS['LBQ'])


def cash_mix(w):
    """0050 w＋定存（年 1%）1−w，每月最後交易日拉回；0050 進出計手續費＋0.1% 稅＋半個跳動點。"""
    T = P.T; idx = pd.DatetimeIndex(P.IDX)
    me = set(pd.Series(np.arange(T), index=idx).groupby(idx.to_period('M')).max().tolist())
    Dc = np.full(T, np.nan); Do = np.full(T, np.nan)
    t0 = P.t0
    Dc[:t0 + 1] = P.E50[:t0 + 1]; Do[:t0 + 1] = P.E50O[:t0 + 1]
    a = w * P.E50[t0]; b = (1 - w) * P.E50[t0]
    for t in range(t0 + 1, T):
        rf = 0.01 * (idx[t] - idx[t - 1]).days / 365.0
        tot0 = a + b
        Do[t] = Dc[t - 1] * (a * P.E50O[t] / P.E50[t - 1] + b) / tot0
        a *= P.E50[t] / P.E50[t - 1]; b *= 1 + rf
        Dc[t] = Dc[t - 1] * (a + b) / tot0
        if t in me:
            W = a + b; x = abs(a - w * W)
            po = P.E50C_raw[t] if np.isfinite(P.E50C_raw[t]) and P.E50C_raw[t] > 0 else 100.0
            c = x * (s30lib.FEE + 0.5 * stlib.etf_tick(po) / po + (s30lib.TAX_ETF0 if a > w * W else 0.0))
            W2 = W - c; a = w * W2; b = (1 - w) * W2; Dc[t] *= W2 / W
        s = Dc[t] / (a + b); a *= s; b *= s
    return dict(Dc=Dc, Do=Do)


IDXS['CASH50'] = cash_mix(0.5); IDXS['CASH35'] = cash_mix(0.35)
s30lib.FLOW['CASH50'] = dict(sb=0.5 * 0.0002, tax=0.5 * s30lib.TAX_ETF0)
s30lib.FLOW['CASH35'] = dict(sb=0.35 * 0.0002, tax=0.35 * s30lib.TAX_ETF0)
CFG = {'CUR': None}
for r, b in (('R_LOWB', 'LOWB'), ('R_LBQ', 'LBQ'), ('R_HDLV', 'HDLV'), ('R_MIX', 'MIX'), ('CASH50', 'CASH50'), ('CASH35', 'CASH35')):
    CFG[r] = s30lib.base_arrays(P, b, IDXS)
B = dict(CFG['R_LOWB']); B['sb'] = 0.0002; B['tax'] = s30lib.TAX_ETF0
CFG['R_LOWB_cheapflow'] = B
del INNOV
gc.collect()
bench = pd.Series(P.E50, index=P.IDX)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, D24, TA, TB = lastb('2019-01-01'), lastb('2024-01-01'), lastb('2016-02-01'), lastb('2022-01-01')
cg = stlib.cagr
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
sim = lambda S, c, cap=1e6, **kw: s30lib.sim(P, S, U, B=CFG[c], cap=cap, **kw)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
FCAP = {}
for c in CFG:
    nav1 = sim(S0, c, 1e6, fresh=True)[0]
    FCAP[c] = 1e6 / float(nav1.loc[TB])
PATH = {c: {} for c in CFG}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            navf, _, stf = sim(S, c, cap * FCAP[c], fresh=True)
            nav, _, st = sim(S, c, cap)
            sgf = s17lib.seg_dd(navf); sg = s17lib.seg_dd(nav)
            PATH[c][key] = dict(valid=cg(navf, TB, DE), valid_dd=stlib.maxdd(navf, TB, DE), valid_beta=s30lib.capm_beta(navf, bench, TB, DE)['beta'],
                                seg2022=sgf['2022'], seg2425=sgf['2024-07~2025-04'],
                                train=cg(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), since2019=cg(nav, D19, DE), full=cg(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE),
                                beta=s30lib.capm_beta(nav, bench, D0, DE)['beta'], seg2018=sg['2018'], seg2020=sg['2020-03'],
                                sharpe=stlib.sharpe(nav, D0, DE), stockfrac=float(np.nanmean(st['stockfrac'].to_numpy())))
            del navf, nav
    if s:
        del S
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
keys = list(PATH['CUR'])
out = {'note': '事後描述，未登記、不進判定', 'median': {}, 'vs_CUR': {}}
for c in CFG:
    out['median'][c] = {f: float(np.median([PATH[c][k][f] for k in keys])) for f in PATH[c][keys[0]]}
    if c != 'CUR':
        out['vs_CUR'][c] = {f: dict(median_pp=float(np.median([(PATH[c][k][f] - PATH['CUR'][k][f]) * 100 for k in keys])),
                                    npos=int(sum((PATH[c][k][f] - PATH['CUR'][k][f]) > 0 for k in keys))) for f in PATH[c][keys[0]] if f not in ('sharpe', 'stockfrac', 'beta', 'valid_beta')}
        out['vs_CUR'][c]['beta_diff_median'] = float(np.median([PATH[c][k]['beta'] - PATH['CUR'][k]['beta'] for k in keys]))
for nm in ('CASH50', 'CASH35'):
    s = pd.Series(IDXS[nm]['Dc'], index=P.IDX)
    out.setdefault('cash_mix_index', {})[nm] = dict(cagr_2016=cg(s, TA, DE), maxdd_2016=stlib.maxdd(s, TA, DE), beta=s30lib.capm_beta(s, bench, TA, DE)['beta'], seg=s17lib.seg_dd(s))
out['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
for c in CFG:
    m = out['median'][c]
    print(c, {k: round(v * (100 if k not in ('beta', 'valid_beta', 'sharpe') else 1), 2) for k, v in m.items()})
print(json.dumps(out['cash_mix_index'], ensure_ascii=False))
print(out['meta'])

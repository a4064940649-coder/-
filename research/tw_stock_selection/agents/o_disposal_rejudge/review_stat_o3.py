#!/usr/bin/env python3
"""反方審查（資料與統計）o_disposal_rejudge：不修改研究員檔案，只讀。
1) 從 navs_full_A.parquet 重算 E1/E0/E5 跨循環年化、配對差、逐月刪一（leave-one-month-out）t、單月尖峰造成的 t≈1 假象。
2) 2021-06-11 那個月的處置公告日是否 ≤ 訊號日（前視）。
3) 「公告日 ≤ d 但 start ∈ (d, e]」：換股日 e 開盤前已知、成交當天就進處置，但 E1 仍買進的名額（SOP 定義漏洞）。
4) 全期 E0−E1 配對 t（研究員只在表中列挑選期）。
"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import drj_core as R
K = R.K
out = {}
navs = pd.read_parquet(HERE / 'navs_full_A.parquet')
S = R.make_strategy('E0', R.Disp(R.load_list()))
t0, t1, sd = S.bounds()
def wr(col):
    return R.win_returns(navs[col].dropna(), sd)
r1 = wr('E1|n10|normal')
for v in ('E0', 'E5', 'E2', 'E3'):
    dd = (wr(f'{v}|n10|normal') - r1).dropna()
    res = {}
    for lab, a, b in (('SEL', R.SEL_A, R.SEL_B), ('ALL', R.SEL_A, R.HOLD_B), ('POST', R.VAL_A, R.HOLD_B)):
        x = dd[(dd.index >= a) & (dd.index <= b)]
        nz = x[x.abs() > 1e-10]
        top = x.abs().sort_values(ascending=False)
        lomo = {}
        for k in top.index[:3]:
            y = x.drop(k); lomo[str(k.date())] = (round(float(x[k] * 100), 3), round(float(y.mean() * 100), 4), round(R.tstat(y.values), 2))
        # 單月尖峰 t 假象：最大月份單獨一筆、其他設 0 的 t
        spike = pd.Series(0.0, index=x.index); spike[top.index[0]] = x[top.index[0]]
        # 非零月的符號檢定
        npos = int((nz > 0).sum()); nn = int(len(nz))
        from math import comb
        p_sign = sum(comb(nn, k) for k in range(npos, nn + 1)) / 2 ** nn if nn else float('nan')
        # 自助法（月份重抽）P(mean<=0)
        rng = np.random.default_rng(7); xv = x.values
        bs = np.array([rng.choice(xv, len(xv)).mean() for _ in range(20000)])
        res[lab] = dict(mean_pp=round(float(x.mean() * 100), 4), t=round(R.tstat(x.values), 2), n=len(x), n_nonzero=nn, n_pos=npos, sign_p_one_sided=round(p_sign, 3),
                        boot_p_mean_le0=round(float((bs <= 0).mean()), 3), drop_one_month=lomo, t_if_only_top_month=round(R.tstat(spike.values), 2),
                        top_month_share_of_sum=round(float(x[top.index[0]] / x.sum()), 2) if x.sum() != 0 else None)
    out[f'{v}-E1'] = res
# 年化重算
def cagr(nav, a, b):
    return R.cagr_mdd(nav.dropna(), a, min(pd.Timestamp(b), nav.dropna().index[-1]))
out['cagr_recalc'] = {v: [round(x, 2) for x in cagr(navs[f'{v}|n10|normal'], R.SEL_A, R.HOLD_B)] for v in ('E1', 'E0', 'E5')}

# 2) 2021-06-11
DS = R.Disp(R.load_list())
d = pd.Timestamp('2021-06-11')
D = DS.D
x = D[D.code.isin(['2465', '2615', '2636', '3530', '2603', '2609'])]
x = x[(x.start <= d + pd.Timedelta(days=40)) & (x.end >= d - pd.Timedelta(days=40))]
out['2021-06-11_disposals'] = x[['code', 'announce', 'start', 'end', 'nth', 'src']].astype(str).values.tolist()
out['2021-06-11_status_at_d'] = {K.COLS[j]: dict(rem=i['rem'], thr=i['thr'], nth=i['nth']) for j, i in DS.at(d).items() if K.COLS[j] in ('2465', '2615', '2636', '3530', '2603', '2609')}

# 3) 已公告、e 當天開始處置但 E1 仍買進
rows = []
for ver in ('E1', 'E0'):
    Sv = R.make_strategy(ver, DS)
    pl = Sv.plan(10, 'official')
    for (e, pk, _, dd_) in pl:
        cur = DS.at(dd_)
        y = D[(D.announce <= dd_) & (D.start > dd_) & (D.start <= K.IDX[e])]
        codes = set(y.code)
        for j in pk:
            c = K.COLS[j]
            if c in codes and j not in cur:
                rows.append(dict(ver=ver, sig=str(dd_.date()), code=c))
ann = pd.DataFrame(rows)
if len(ann):
    ann['y'] = ann.sig.str[:4].astype(int)
    out['announced_by_d_start_on_e_held'] = {v: dict(total=int((ann.ver == v).sum()), SEL=int(((ann.ver == v) & (ann.sig <= '2021-08-31')).sum()),
                                                     by_year=ann[ann.ver == v].groupby('y').size().to_dict()) for v in ('E1', 'E0')}
else:
    out['announced_by_d_start_on_e_held'] = 0
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

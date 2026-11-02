"""P1（0056 臺灣高股息）：截止日當時資料重建預測殖利率排名，檢查實際納入／刪除股的名次。不讀報酬。
限制：沒有 0056 歷史成分快照 → 無法分辨「非成分的前 35 名」，precision 無法計算，只報 recall 與名次分布。
母體：截止日 PIT 市值前 150 名上市普通股（近似臺灣50＋中型100）。
殖利率代理（FTSE 用的「未來一年預測現金股利」來自不公開的共識預估）：
  y1＝pe_5y 月初殖利率（證交所 BWIBBU，最近年度現金股利／收盤）；
  y2＝截止日前 365 日除息現金（exdiv_5y 息／權息 value）÷ 截止日收盤；
  y3＝max(y1, y2)；y4＝配發率÷本益比（前瞻代理）；y5＝y1 與 y4 名次平均。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import eflib as E
R = E.reviews(); R = R[R.etf == '0056']
ev = E.events(); ev = ev[ev.etf == '0056']
pe = pd.read_csv(E.EX / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
xd = pd.read_csv(E.EX / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
xd = xd[xd.kind.isin(['息', '權息'])]
def yields(cut):
    d = E.tday_le(cut)
    rk, mc = E.mcap_rank(d, patch=True)
    U = list(rk.index[:150])
    p = pe[pe.date <= d]; p = p[p.date == p.date.max()].set_index('code')['yield']
    y1 = p.reindex(U)
    x = xd[(xd.date <= d) & (xd.date > d - pd.Timedelta(days=365))].groupby('code')['value'].sum()
    cl = E.P('close_raw').loc[d]
    y2 = (x.reindex(U).fillna(0) / cl.reindex(U)) * 100
    y3 = pd.concat([y1, y2], axis=1).max(axis=1)
    # y4：前瞻代理＝配發率 × 近四季 EPS ÷ 價格＝配發率 ÷ PE_now；配發率＝最近年度股利 ÷ 一年前的近四季 EPS
    pm = pe[pe.date <= d]; dm = pm.date.max()
    p0 = pm[pm.date == dm].set_index('code'); p12 = pm[pm.date == pm[pm.date <= dm - pd.Timedelta(days=360)].date.max()].set_index('code')
    cl12 = E.P('close_raw').loc[:E.tday_le(dm - pd.Timedelta(days=365))].ffill().iloc[-1]
    eps12 = cl12.reindex(U) / p12['pe'].reindex(U).replace(0, np.nan)
    dlast = p0['yield'].reindex(U) / 100 * E.P('close_raw').loc[:dm].ffill().iloc[-1].reindex(U)
    payout = (dlast / eps12).clip(0, 1.5)
    y4 = payout / p0['pe'].reindex(U).replace(0, np.nan) * 100
    y5 = (y1.rank(ascending=False) + y4.rank(ascending=False)) * -1.0
    return U, {'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4, 'y5': y5}, rk
if __name__ == '__main__':
    out = {}
    for yk in ['y1', 'y2', 'y3', 'y4', 'y5']:
        rows = []; ra = []; rd = []
        for r in R.itertuples():
            U, Y, rk = yields(r.cutoff)
            y = Y[yk].rank(ascending=False, method='first')
            A = list(ev[(ev.review == r.review) & (ev.action == 'add')].code)
            D = list(ev[(ev.review == r.review) & (ev.action == 'del')].code)
            a_r = {c: (int(y[c]) if c in y.index and np.isfinite(y[c]) else None) for c in A}
            d_r = {c: (int(y[c]) if c in y.index and np.isfinite(y[c]) else None) for c in D}
            rows.append(dict(review=r.review, cutoff=r.cutoff, add_rank=a_r, del_rank=d_r))
            ra += [v for v in a_r.values()]; rd += [v for v in d_r.values()]
        nA = len(ra); inU = [v for v in ra if v is not None]
        res = dict(n_add=nA, add_in_universe=len(inU), add_recall_top35=round(sum(v <= 35 for v in inU) / nA, 3),
                   add_recall_top50=round(sum(v <= 50 for v in inU) / nA, 3), add_rank_median=float(np.median(inU)) if inU else None,
                   n_del=len(rd), del_recall_rank66_or_out=round(sum((v is None) or v >= 66 for v in rd) / max(1, len(rd)), 3),
                   precision='無法計算（無 0056 歷史成分快照）', rows=rows)
        out[yk] = res
        print(yk, {k: v for k, v in res.items() if k != 'rows'})
    json.dump(out, open(E.HERE / 'p1_0056.json', 'w'), ensure_ascii=False, indent=1)

"""P1（臺灣50）：只用截止日當時的收盤價 × PIT 股數，依公開規則（非成分 ≤40 納入、成分 ≥61 刪除、納入多於刪除時刪最低名次成分）預測名單。
不讀任何報酬。版本：v1 PIT 股數當日；v2 PIT 股數落後 63 日（保守 PIT）；v3 v1＋流動性（截止日前 60 日平均成交值 ≥ 全體第 300 名，近似 FTSE 流動性檢驗）；v4 v1＋修正 4763 股數；v5 v4＋落後 63 日。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import eflib as E
R = E.reviews(); R = R[(R.etf == 'TW50')].copy()
ev = E.events(); ev = ev[ev.etf == 'TW50']
def predict(cut, lag=0, liq=False, patch=False):
    d = E.tday_le(cut)
    rk, mc = E.mcap_rank(d, lag=lag, patch=patch)
    if liq:
        v = E.P('value').loc[:d].tail(60).mean()
        ok = set(v[v >= v.sort_values(ascending=False).iloc[min(299, len(v)-1)]].index)
        rk = mc[mc.index.isin(ok)].rank(ascending=False, method='first').sort_values()
    mem = E.tw50_members_at(pd.Timestamp(cut))
    adds = [c for c in rk.index if c not in mem and rk[c] <= 40]
    dels = [c for c in mem if (c not in rk.index) or rk[c] >= 61]
    keep = [c for c in mem if c not in dels]
    extra = len(keep) + len(adds) - 50
    if extra > 0:
        keep_sorted = sorted(keep, key=lambda c: rk.get(c, 9999), reverse=True)
        dels += keep_sorted[:extra]
    return adds, dels, rk, mem
if __name__ == '__main__':
    out = {}
    for ver, kw in [('v1_pit', {}), ('v2_pit_lag63', {'lag': 63}), ('v3_pit_liq300', {'liq': True}), ('v4_pit_fix4763', {'patch': True}), ('v5_pit_fix4763_lag63', {'patch': True, 'lag': 63})]:
        rows = []
        TPa = FPa = FNa = TPd = FPd = FNd = 0
        for r in R.itertuples():
            if pd.Timestamp(r.eff) < pd.Timestamp('2015-09-18'): continue   # 成分名單起點
            adds, dels, rk, mem = predict(r.cutoff, **kw)
            A = set(ev[(ev.review == r.review) & (ev.action == 'add')].code)
            D = set(ev[(ev.review == r.review) & (ev.action == 'del')].code)
            pa, pd_ = set(adds), set(dels)
            rows.append(dict(review=r.review, cutoff=r.cutoff, verify=r.verify, pred_add=sorted(pa), act_add=sorted(A), pred_del=sorted(pd_), act_del=sorted(D),
                             act_add_rank={c: (int(rk[c]) if c in rk.index else None) for c in A}, act_del_rank={c: (int(rk[c]) if c in rk.index else None) for c in D},
                             miss_add=sorted(A - pa), false_add=sorted(pa - A)))
            TPa += len(pa & A); FPa += len(pa - A); FNa += len(A - pa)
            TPd += len(pd_ & D); FPd += len(pd_ - D); FNd += len(D - pd_)
        prec = TPa / max(1, TPa + FPa); rec = TPa / max(1, TPa + FNa)
        out[ver] = dict(n_reviews=len(rows), add=dict(TP=TPa, FP=FPa, FN=FNa, precision=round(prec, 4), recall=round(rec, 4)),
                        dele=dict(TP=TPd, FP=FPd, FN=FNd, precision=round(TPd / max(1, TPd + FPd), 4), recall=round(TPd / max(1, TPd + FNd), 4)), rows=rows)
        print(ver, out[ver]['add'], out[ver]['dele'])
    json.dump(out, open(E.HERE / 'p1_tw50.json', 'w'), ensure_ascii=False, indent=1, default=str)

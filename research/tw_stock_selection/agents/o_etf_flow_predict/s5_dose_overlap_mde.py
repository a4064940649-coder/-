"""描述性（不讀事件報酬）：(a) 0056 實際納入股的被動需求劑量 D（公開規模點 × 等權 1/50 假設 ÷ 截止日前 20 日均成交值）；
(b) 臺灣50 預測納入／刪除與 0056 實際事件股和主策略（V7 AB_V7_B10 seed 0 名單）的重疊；(c) P4 規模通道的最小可偵測效果（MDE）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import eflib as E
import s2_tw50_p1 as S2
VAL, CL, ADJ = E.P('value'), E.P('close_raw'), E.P('adj')
# 公開規模點（新聞／證交所頁面；WebSearch 2026-11-02 取得，見 gate.json 來源）——不是月資料
AUM0056 = {'2017-12': 106.5, '2021-10': 1000.0, '2022-10': 1591.0, '2026-03': 5210.0, '2026-07': 7184.0}   # 億元
def aum_interp(d):
    s = pd.Series({pd.Timestamp(k + '-28'): v for k, v in AUM0056.items()}).sort_index()
    x = np.log(s); t = (s.index - s.index[0]).days
    td = (pd.Timestamp(d) - s.index[0]).days
    return float(np.exp(np.interp(td, t, x.values)))
ev = E.events(); R = E.reviews()
out = {'aum_points_0056_億': AUM0056}
rows = []
for r in R[R.etf == '0056'].itertuples():
    C = E.tday_le(r.cutoff); aum = aum_interp(C) * 1e8
    for c in ev[(ev.etf == '0056') & (ev.review == r.review) & (ev.action == 'add')].code:
        if c not in VAL.columns: continue
        adv = VAL.loc[:C, c].tail(20).mean()
        if not np.isfinite(adv) or adv <= 0: continue
        rows.append(dict(review=r.review, code=c, aum_億=round(aum / 1e8), adv20_億=round(adv / 1e8, 2), D=round(aum / 50 / adv, 2)))
d = pd.DataFrame(rows)
out['D_0056_adds'] = dict(n=len(d), median=float(d.D.median()), p25=float(d.D.quantile(.25)), p75=float(d.D.quantile(.75)), max=float(d.D.max()),
                          by_review=d.groupby('review').D.median().round(2).to_dict())
# 臺灣50 對照：新納入股權重≈市值/前 50 名市值合計；0050+006208 規模用 0056 同年點的 2 倍以內粗估不可得 → 只報 w/ADV（每 1 億元被動規模對應的成交值倍數）
rows = []
for r in R[(R.etf == 'TW50') & (R.eff >= '2015-09-18')].itertuples():
    adds, dels, rk, mem = S2.predict(r.cutoff, patch=True)
    C = E.tday_le(r.cutoff); _, mc = E.mcap_rank(C, patch=True)
    top = mc.sort_values(ascending=False).head(50).sum()
    for c in adds:
        adv = VAL.loc[:C, c].tail(20).mean()
        rows.append(dict(review=r.review, code=c, w=mc[c] / top, adv20=adv, D_per_1000億=(mc[c] / top) * 1e11 / adv))
t = pd.DataFrame(rows)
out['D_tw50_pred_adds_per_1000億AUM'] = dict(n=len(t), median=round(float(t['D_per_1000億'].median()), 3), w_median=round(float(t.w.median()), 4))
# (b) 重疊
pl = pd.read_csv(E.AG / 'o_v7_listing_exec_engine/data/plans_ab7.csv', dtype=str)
pl = pl[(pl.strat == 'AB_V7_B10') & (pl.seed == '0')].copy(); pl['exec'] = pd.to_datetime(pl['exec'])
def held(d):
    q = pl[pl['exec'] <= d]
    return set(q.iloc[-1].picks.split(',')) if len(q) else set()
ov = {'tw50_pred_add': [0, 0], 'tw50_pred_del': [0, 0], '0056_add': [0, 0], '0056_del': [0, 0]}
for r in R.itertuples():
    if r.etf not in ('TW50', '0056') or not isinstance(r.cutoff, str): continue
    if r.etf == 'TW50' and r.eff < '2015-09-18': continue
    h = held(pd.Timestamp(r.ann))
    if r.etf == 'TW50':
        adds, dels, rk, mem = S2.predict(r.cutoff, patch=True)
        for k, l in (('tw50_pred_add', adds), ('tw50_pred_del', dels)):
            ov[k][0] += len(set(l) & h); ov[k][1] += len(l)
    else:
        for k, a in (('0056_add', 'add'), ('0056_del', 'del')):
            l = set(ev[(ev.etf == '0056') & (ev.review == r.review) & (ev.action == a)].code)
            ov[k][0] += len(l & h); ov[k][1] += len(l)
out['overlap_with_main_B10_seed0'] = {k: dict(hit=v[0], n=v[1], rate=round(v[0] / max(1, v[1]), 3)) for k, v in ov.items()}
# (c) MDE：前 1/5 流量壓力股 vs 同池等權的月超額。用前 150 大市值股隨機 1/5（30 檔）等權減 150 等權的月超額標準差估 σ。
me = pd.DatetimeIndex(sorted(set(ADJ.index.to_period('M').map(lambda p: ADJ.index[ADJ.index.to_period('M') == p][-1]))))
me = me[(me >= '2019-01-01') & (me <= '2026-09-30')]
rng = np.random.default_rng(0); ex = []
for a, b in zip(me[:-1], me[1:]):
    rk, mc = E.mcap_rank(a, patch=True)
    U = list(rk.index[:150]); r = (ADJ.loc[b, U] / ADJ.loc[a, U] - 1).dropna()
    s = rng.choice(r.index, 30, replace=False)
    ex.append(r[s].mean() - r.mean())
sd = float(np.std(ex, ddof=1))
out['P4_MDE'] = {'sigma_monthly_excess_random30_of150': round(sd * 100, 3),
                 'MDE_pct_per_month_80pow_5pct': {str(T): round(2.8 * sd / np.sqrt(T) * 100, 3) for T in (36, 48, 90)},
                 'note': 'T＝36（2023～2025 申購潮）、48、90（2019～2026）。月規模資料不可得，P4 無法估計。'}
json.dump(out, open(E.HERE / 'dose_overlap_mde.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, default=str)[:2500])

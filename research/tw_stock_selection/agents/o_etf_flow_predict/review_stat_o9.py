"""反方審查（o9 資料與統計）：只讀研究員輸出，不改其檔案。"""
import sys, ast, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
H = Path('/home/user/-/research/tw_stock_selection/agents/o_etf_flow_predict')
sys.path.insert(0, str(H))
df = pd.read_csv(H / 'tw50_control_events.csv', dtype={'code': str, 'review': str})
def ct(d, col):
    g = d.groupby('review')[col].mean().dropna(); n = len(g)
    t = g.mean() / (g.std(ddof=1) / np.sqrt(n)); p = 2 * stats.t.sf(abs(t), n - 1)
    return dict(n_ev=int(d[col].notna().sum()), n_rev=n, ev_mean=round(d[col].mean()*100, 2), rev_mean=round(g.mean()*100, 2), t=round(t, 2), p=round(p, 4))
out = {}
a = df[df.side == 'add']; dl = df[df.side == 'del']
cells = {}
for s, d in (('add', a), ('del', dl)):
    for w in ('W1', 'W2', 'W3'):
        cells[f'{s}|{w}|net'] = ct(d, f'{w}_net_ex')
out['cells'] = cells
ps = sorted([(v['p'], k) for k, v in cells.items()])
m = len(ps); holm = {}
for i, (p, k) in enumerate(ps):
    holm[k] = round(min(1, p * (m - i)), 4)
out['holm6'] = holm
# 擴大到研究員實際計算的所有格（2 side × 3 per × 3 win × 3 cost = 54）
out['n_cells_computed'] = 54
# 年別、逐年刪除
a = a.copy(); a['yr'] = a.review.str[:4]
out['W2_net_by_year_eventmean'] = a.groupby('yr').W2_net_ex.mean().mul(100).round(2).to_dict()
out['W2_net_by_year_n'] = a.groupby('yr').size().to_dict()
loyo = {}
for y in sorted(a.yr.unique()):
    loyo[y] = ct(a[a.yr != y], 'W2_net_ex')['t']
out['W2_leave_one_year_out_t'] = loyo
# 去掉最差 2 個審核期
g = a.groupby('review').W2_net_ex.mean().sort_values()
out['W2_rev_means_sorted'] = g.mul(100).round(2).to_dict()
for k in (1, 2, 3):
    keep = g.index[k:]
    out[f'W2_drop_worst{k}_rev_t'] = ct(a[a.review.isin(keep)], 'W2_net_ex')['t']
# 公告日單日：W2-W1
a['annday'] = a.W2_gross_ex - a.W1_gross_ex
out['ann_day_gross'] = dict(mean=round(a.annday.mean()*100, 2), t=ct(a.assign(x=a.annday), 'x')['t'])
a['postann'] = a.W3_gross_ex - a.W2_gross_ex
out['ann_to_eff_gross(approx diff)'] = dict(mean=round(a.postann.mean()*100, 2), t=ct(a.assign(x=a.postann), 'x')['t'])
# 對照組規模：用研究員 eflib 算截止日市值名次
try:
    import eflib as E
    R = E.reviews(); R = R[R.etf == 'TW50'].set_index('review')
    rows = []
    for r in a.itertuples():
        C = E.tday_le(R.loc[r.review, 'cutoff'])
        rk, mc = E.mcap_rank(C, patch=True)
        ctl = ast.literal_eval(r.ctl)
        rows.append(dict(ev_rank=r.mcap_rank, ctl_rank_med=float(np.median([rk.get(c, np.nan) for c in ctl])),
                         ctl_mc_ratio=float(np.nanmedian([mc.get(c, np.nan) for c in ctl]) / mc.get(r.code, np.nan))))
    z = pd.DataFrame(rows)
    out['control_size'] = dict(ev_rank_median=float(z.ev_rank.median()), ctl_rank_median=float(z.ctl_rank_med.median()),
                               ctl_rank_p25=float(z.ctl_rank_med.quantile(.25)), ctl_rank_p75=float(z.ctl_rank_med.quantile(.75)),
                               ctl_mcap_over_event_median=round(float(z.ctl_mc_ratio.median()), 3))
    # 不同基準：相對大盤（前 50 大市值等權）
    ADJ, OP, CL = E.P('adj'), E.P('open'), E.P('close_raw')
    rr = []
    for r in a.itertuples():
        C = E.tday_le(R.loc[r.review, 'cutoff']); t0 = E.tday_after(C, 1); A = E.tday_le(R.loc[r.review, 'ann'])
        rk, mc = E.mcap_rank(C, patch=True); U = list(rk.index[30:60])
        def wr(c):
            o, c0, a0, a1 = OP.at[t0, c], CL.at[t0, c], ADJ.at[t0, c], ADJ.at[A, c]
            return a1 / a0 * c0 / o - 1
        own = wr(r.code); bm = np.nanmean([wr(c) for c in U if c != r.code])
        rr.append(dict(review=r.review, x=own - bm))
    zz = pd.DataFrame(rr)
    out['W2_gross_vs_rank31_60_ew'] = ct(zz, 'x')
except Exception as e:
    out['control_size_err'] = repr(e)
# 0056 P1：實際納入股是否在母體、名次分布
p = json.load(open(H / 'p1_0056.json'))
for yk in ('y1', 'y4'):
    rows = p[yk]['rows']; ranks = [v for r in rows for v in r['add_rank'].values()]
    out[f'0056_{yk}_add_none'] = sum(v is None for v in ranks)
    out[f'0056_{yk}_by_review_recall35'] = {r['review']: round(np.mean([(v is not None and v <= 35) for v in r['add_rank'].values()]), 2) for r in rows if r['add_rank']}
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

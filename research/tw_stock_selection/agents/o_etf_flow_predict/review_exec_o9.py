"""反方審查（o9 成本與可成交）：只讀研究員輸出與共用面板，不改其檔案。"""
import sys, ast, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path('/home/user/-/research/tw_stock_selection/agents/o_etf_flow_predict')
sys.path.insert(0, str(H))
import eflib as E
import s2_tw50_p1 as S2
import s4_tw50_control as S4
CL, OP, ADJ = E.P('close_raw'), E.P('open'), E.P('adj')
ev = pd.read_csv(H / 'tw50_control_events.csv', dtype={'code': str, 'review': str})
R = E.reviews(); R = R[R.etf == 'TW50'].set_index('review')
EX = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
disp = pd.read_csv(EX / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['start', 'end'])
out = {}
idx = CL.index
def nd(a, b): return int(((idx > a) & (idx <= b)).sum())
rows = []
for r in ev[ev.side == 'add'].itertuples():
    rv = R.loc[r.review]
    C = E.tday_le(rv.cutoff); t0 = E.tday_after(C, 1)
    A = E.tday_le(rv.ann); A1 = E.tday_le(pd.Timestamp(rv.ann) - pd.Timedelta(days=1)); E1 = E.tday_le(pd.Timestamp(rv.eff) - pd.Timedelta(days=1))
    c = r.code
    pc = CL.at[C, c]; o = OP.at[t0, c]
    gap = o / pc - 1 if np.isfinite(o) and np.isfinite(pc) else np.nan
    # 出場日跌停/漲停近似
    prevA = idx[idx < A][-1]; ra = CL.at[A, c] / CL.at[prevA, c] - 1
    dd = disp[(disp.code == c) & (disp.start <= E1) & (disp.end >= t0)]
    # 規模配對對照：截止日市值名次最近的 3 檔非成分、非事件股（名次 30～90）
    adds, dels, rk, mem = S2.predict(rv.cutoff, patch=True)
    excl = mem | set(adds) | set(dels)
    pool = [x for x in rk.index if x not in excl and 30 <= rk[x] <= 90 and x in OP.columns]
    myr = rk.get(c, np.nan)
    pool = sorted(pool, key=lambda x: abs(rk[x] - myr))[:3]
    g2 = S4.wret(c, t0, A); cr = np.nanmean([S4.wret(x, t0, A) for x in pool]) if pool else np.nan
    g3 = S4.wret(c, t0, E1); cr3 = np.nanmean([S4.wret(x, t0, E1) for x in pool]) if pool else np.nan
    cost = S4.side_cost(c, t0) + S4.side_cost(c, A) + 0.003
    rows.append(dict(review=r.review, code=c, nW1=nd(t0, A1) + 1, nW2=nd(t0, A) + 1, nW3=nd(t0, E1) + 1, gap_open=gap, ret_annday=ra,
                     disposal=len(dd) > 0, cost_rt=cost, W2_gross_ctl=r.W2_gross_ex, W2_net_ctl=r.W2_net_ex,
                     W2_gross_size=g2 - cr, W2_net_size=g2 - cr - cost, W3_gross_size=g3 - cr3, W3_net_size=g3 - cr3 - (S4.side_cost(c, t0) + S4.side_cost(c, E1) + 0.003), raw_W2=g2))
d = pd.DataFrame(rows)
def ct(x, col):
    g = x.groupby('review')[col].mean().dropna(); n = len(g)
    return dict(n_ev=int(x[col].notna().sum()), n_rev=n, rev_mean_pct=round(g.mean() * 100, 2), t=round(g.mean() / (g.std(ddof=1) / np.sqrt(n)), 2))
out['window_days_median'] = {k: float(d[k].median()) for k in ('nW1', 'nW2', 'nW3')}
out['window_days_range_W2'] = [int(d.nW2.min()), int(d.nW2.max())]
out['cost_rt_pct_median'] = round(d.cost_rt.median() * 100, 3)
out['open_gap_ge_9pct'] = int((d.gap_open >= 0.09).sum()); out['open_gap_median_pct'] = round(d.gap_open.median() * 100, 2)
out['annday_limitdown'] = int((d.ret_annday <= -0.09).sum())
out['disposal_overlap'] = int(d.disposal.sum())
for col in ('W2_gross_ctl', 'W2_net_ctl', 'W2_gross_size', 'W2_net_size', 'W3_gross_size', 'W3_net_size', 'raw_W2'):
    out[col] = ct(d, col)
post = d[d.review >= '2019-06']
out['post2019_W2_net_size'] = ct(post, 'W2_net_size'); out['post2019_W2_gross_size'] = ct(post, 'W2_gross_size')
out['medians_pct'] = {k: round(d[k].median() * 100, 2) for k in ('W2_gross_size', 'W2_net_size', 'W3_gross_size', 'W3_net_size')}
out['W3_net_size_pos_share'] = round(float((d.W3_net_size > 0).mean()), 2)
# 劑量：TW50 預測納入股的被動需求 ÷ ADV20
print(json.dumps(out, ensure_ascii=False, indent=1))

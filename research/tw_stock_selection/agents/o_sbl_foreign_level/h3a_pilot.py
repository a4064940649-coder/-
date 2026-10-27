#!/usr/bin/env python3
"""H3a 描述性試算（不是檢定、不用來選版）：借券只有 2026-07-30 起；融券 2024-01 起。

必須在 PREREG（含 H3a 前瞻追蹤規格）寫入之後才執行。輸出 h3a_pilot.json。
  * 借券 20 日增量 ÷ 發行股數（訊號日前一交易日 vs 再往前 20 個交易日）：只有 2026-09-11、2026-10-12（到 10-22，不完整）兩個橫斷面
  * 融券 20 日增量 ÷ 發行股數：2024-02～2026-09 訊號日（P2 對照的描述）
  * 借券與融券、52 週高點、營收、過去 20 日報酬的相關
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import sbl_lib as L

H = L.HERE
R = json.loads((H / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in R and 'H3a_forward' in R['PREREG']['spec'], 'H3a 前瞻追蹤規格尚未登記'
L.check_freeze()
K, IDX, COLS, N = L.K, L.IDX, L.COLS, L.N
S = L.strategy(True)
SB = pd.read_parquet(H / 'sbl_panel.parquet').reindex(columns=COLS)
SH = pd.read_parquet(H / 'short_panel.parquet').reindex(columns=COLS) * 1000.0   # 張 → 股
Rq, Sq, Fq = L.load_qfiis()
shares_anchor = L.fo_data()['shares']

def incr(P, t, lag=20):
    d1, d0 = IDX[t - 1], IDX[t - 1 - lag]
    if d1 not in P.index or d0 not in P.index: return None
    sh = Sq.reindex(columns=COLS).loc[:d1].iloc[-1].to_numpy(float) if d1 >= Sq.index.min() else shares_anchor
    return (P.loc[d1].to_numpy(float) - P.loc[d0].to_numpy(float)) / sh

REB = K.rebalance_days('2024-01-01')          # 每月第一個 ≥ 11 日的交易日（到 2026-10-12）
latest_end = len(IDX) - 1
PL10 = {p[3]: set(p[1]) for p in S.plan(10)}
def cross(d, P, lag=20):
    cs = S.cross_section(d); el = cs['el']; t, e = cs['t'], cs['e']
    x = incr(P, t, lag)
    if x is None: return None
    nd = [z for z in REB if z > d]
    if nd:
        e2 = IDX.get_loc(nd[0]) + 1; full = True
    else:
        e2 = latest_end; full = False
    idx = np.where(el & np.isfinite(x))[0]
    r = L.open_to_open(e, e2, idx, to_open=full)
    q = pd.qcut(pd.Series(x[idx]).rank(method='first'), 5, labels=False).to_numpy()
    top = q == 4
    past20 = K.MYADJ.to_numpy()[t - 1, idx] / K.MYADJ.to_numpy()[t - 21, idx] - 1
    pr = lambda v: pd.Series(np.where(el, v, np.nan)).rank(pct=True).to_numpy()[idx]
    picks = PL10.get(d, set())
    return dict(d=str(d.date()), exec=str(IDX[e].date()), end=str(IDX[e2].date()), full_month=full, n=int(len(idx)),
                top_minus_rest_pct=round(float(r[top].mean() - r[~top].mean()) * 100, 3),
                spearman_incr_vs_ret=round(float(pd.Series(x[idx]).rank().corr(pd.Series(r).rank())), 3),
                spearman_incr_vs_s52=round(float(pd.Series(x[idx]).rank().corr(pd.Series(pr(cs['s52'])).rank())), 3),
                spearman_incr_vs_past20=round(float(pd.Series(x[idx]).rank().corr(pd.Series(past20).rank())), 3),
                main_picks_in_top_quintile=[COLS[j] for j in idx[top] if j in picks],
                _x=x, _idx=idx, _r=r)

out = {'note': '描述性；不構成檢定。借券只有 2 個橫斷面（第 2 個只到 2026-10-22）。'}
sbl_rows = []
for d in [d for d in REB if d >= pd.Timestamp('2026-08-01')]:      # 2026-08-11（20 日窗不夠 → None）、09-11、10-12
    c = cross(d, SB)
    if c: sbl_rows.append(c)
# 借券 vs 融券增量的橫斷面相關
for c in sbl_rows:
    t = IDX.get_loc(pd.Timestamp(c['d']))
    y = incr(SH, t)
    if y is not None:
        c['spearman_sbl_vs_short_incr'] = round(float(pd.Series(c['_x'][c['_idx']]).rank().corr(pd.Series(y[c['_idx']]).rank())), 3)
out['SBL_cross_sections'] = [{k: v for k, v in c.items() if not k.startswith('_')} for c in sbl_rows]
# 10 日增量版本（2026-08-11 訊號可用）
c = cross(pd.Timestamp('2026-08-11'), SB, lag=8)
out['SBL_8day_incr_2026-08-11'] = {k: v for k, v in c.items() if not k.startswith('_')} if c else None

# 融券（P2 對照的描述）：2024-02～2026-09
sp = {}
rows = []
for d in [d for d in S.sig_days if pd.Timestamp('2024-02-01') <= d <= pd.Timestamp('2026-09-30')]:
    c = cross(d, SH)
    if c is None: continue
    sp[d] = c['top_minus_rest_pct'] / 100
    rows.append({k: v for k, v in c.items() if not k.startswith('_') and k != 'main_picks_in_top_quintile'})
s = pd.Series(sp)
out['short_balance_P2_top_quintile_minus_rest'] = {'all': L.summ(s), '2024-02_2025-09': L.summ(s[s.index < pd.Timestamp('2025-10-01')]),
                                                   '2025-10_2026-09': L.summ(s[s.index >= pd.Timestamp('2025-10-01')])}
out['short_balance_monthly'] = rows
(H / 'h3a_pilot.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k != 'short_balance_monthly'}, ensure_ascii=False, indent=1, default=str))

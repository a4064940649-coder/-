#!/usr/bin/env python3
"""反方審查（資料與統計）o13：不修改研究員檔案，只讀其程式與 raw/，結果印到 stdout。
檢核：
 A. D1a 留一的「難度」：拿掉 ±20 曆日錨點後，真值股最近前後剩餘錨點的間距（月資料 tejm 讓留一變成 2 個月內插）。
 B. 加嚴留一：drop ±100 曆日；只留 12 月錨點（模擬下市股只有年底股本）；去掉 tejm 來源。看誤差與重疊。
 C. D1a 重疊的機械成分：真值前 100 中無真值者兩邊同值 → 只看有真值股的有效重疊。
 D. D2 嚴格只往回看（≤ d 的快照）口徑的 dated 覆蓋；往後看快照占 dated 的比例。
 E. 真值的存活者偏誤：檢核日前 100 中「2021 前下市」股的真值有無、來源。
 F. 真值來源別誤差（tejm vs 其他）。
只讀 close_raw/open/value/flag 與 raw/；不讀 adj、ret；不算報酬。"""
import sys, json
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/o_pit_shares_industry_2006_2013')
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import plib as P, pbuild as B

SPEC = json.loads((P.DATA / 'GATE_SPEC.json').read_text(encoding='utf-8'))
VER = SPEC['construction']['version']
A = B.load_anchor_rows(); M = B.merge_anchors(A)
tlo, thi = P.ti('2005-06-01'), P.ti('2021-06-30')
mult = B.event_multipliers(tlo, thi)
ADV = P.adv120()
S_full, SRC = B.build(M, tlo, thi, mult, VER)
cps = [P.ti(d) for d in SPEC['D1a_2006']['checkpoints']]
out = {}

# ---- A. 間距 ----
gaps = []
for tq in cps:
    d = P.IDX[tq]
    tru = M[M.t == tq]
    keep = M[(M.date - d).abs() > pd.Timedelta(days=20)]
    al = P.alive_at(tq)
    base = np.where(np.isin(np.arange(P.N), [P.CI[c] for c in tru.code]), np.nan, np.nan)
    for r in tru.itertuples():
        g = keep[keep.code == r.code]
        prev = g[g.date < d].date.max(); nxt = g[g.date > d].date.min()
        gaps.append(dict(d=str(d.date()), code=r.code, src=r.src,
                         prev_days=(d - prev).days if pd.notna(prev) else None, next_days=(nxt - d).days if pd.notna(nxt) else None))
G = pd.DataFrame(gaps)
G['span'] = G.prev_days.fillna(9999) + G.next_days.fillna(9999)
out['A_truth_src_share'] = G.src.value_counts(normalize=True).round(3).to_dict()
out['A_span_le_70d_share'] = round(float((G.span <= 70).mean()), 3)
out['A_span_le_100d_share'] = round(float((G.span <= 100).mean()), 3)
out['A_span_median_days'] = float(G.span.median())

# ---- B/C/F. 加嚴留一 ----
def summarize(rows, det, label):
    E = pd.DataFrame(det); ov = np.array([r['overlap'] for r in rows])
    ntl = np.array([r['n_truthless_in_top'] for r in rows])
    eff = ov - ntl  # 有真值股中的重疊（無真值股兩邊同值，機械性重疊）
    res = dict(med=round(float(E.abs_log.median()), 5), p75=round(float(E.abs_log.quantile(.75)), 4), p90=round(float(E.abs_log.quantile(.9)), 4),
               share_gt_5pct=round(float((E.abs_log > 0.05).mean()), 4),
               overlap_med=float(np.median(ov)), overlap_min=int(ov.min()), n_q_overlap_lt93=int((ov < 93).sum()),
               eff_overlap_rate_min=round(float(np.min(eff / (100 - ntl))), 3),
               by_year_overlap_min=pd.Series(ov, index=[r['d'][:4] for r in rows]).groupby(level=0).min().to_dict())
    print(label, res, flush=True)
    return res, E

r0, d0 = B.loo_eval(M, cps, mult, VER, S_full, ADV)
out['B0_reproduce'], E0 = summarize(r0, d0, 'B0 reproduce ±20d')
# F: 真值來源別
src_at = M.set_index(['code', 't']).src
E0['src'] = [src_at.get((c, P.ti(d)), None) for c, d in zip(E0.code, E0.d)]
out['F_err_by_truth_src'] = E0.groupby('src').abs_log.agg(['median', lambda s: s.quantile(.9), 'count']).round(5).astype(str).to_dict('index')

r1, d1 = B.loo_eval(M, cps, mult, VER, S_full, ADV, drop_days=100)
out['B1_drop100d'], _ = summarize(r1, d1, 'B1 drop ±100d')
dec_only = lambda K: K[K.date.dt.month == 12]
r2, d2 = B.loo_eval(M, cps, mult, VER, S_full, ADV, anc_filter=dec_only)
out['B2_dec_only'], E2 = summarize(r2, d2, 'B2 Dec-only anchors')
out['B2_dec_only_by_year_med'] = E2.groupby(E2.d.str[:4]).abs_log.median().round(4).to_dict()
out['B2_dec_only_by_year_p90'] = E2.groupby(E2.d.str[:4]).abs_log.quantile(.9).round(4).to_dict()
# 注意：B3 無效（審查員自己的設計錯誤）：M 已依優先序合併，季末 tejm 與 tejq 同日時只留 tejm，
# 過濾 tejm 等於連同日 tejq 一起拿掉 → 大多真值股變 ADV 插補，誤差是假象。結論不採用 B3。
no_tejm = lambda K: K[K.src != 'tejm']
r3, d3 = B.loo_eval(M, cps, mult, VER, S_full, ADV, anc_filter=no_tejm)
out['B3_no_tejm'], _ = summarize(r3, d3, 'B3 no tejm in rebuild')

# ---- E. 真值存活者偏誤 ----
last_trade = pd.Series([P.IDX[np.where(P.TRADED[:, j])[0].max()] if P.TRADED[:, j].any() else pd.NaT for j in range(P.N)], index=P.COLS)
erows = []
for tq in cps:
    al = P.alive_at(tq)
    mc = np.where(al, S_full[tq] * P.CFF[tq], np.nan); mc, imp = B.impute_mcap(tq, mc, al, ADV)
    T = B.top_k(mc, 100)
    tru = set(M[M.t == tq].code)
    dl = [P.COLS[j] for j in T if last_trade[P.COLS[j]] < pd.Timestamp('2021-01-01')]
    erows.append(dict(d=str(P.IDX[tq].date()), n_delisted_in_top=len(dl), n_delisted_with_truth=sum(c in tru for c in dl),
                      n_survivor_without_truth=sum((P.COLS[j] not in tru) and (P.COLS[j] not in dl) for j in T),
                      delisted_no_truth=[c for c in dl if c not in tru]))
ER = pd.DataFrame(erows)
out['E_delisted_in_top100_mean'] = round(float(ER.n_delisted_in_top.mean()), 2)
out['E_delisted_with_truth_share'] = round(float(ER.n_delisted_with_truth.sum() / max(ER.n_delisted_in_top.sum(), 1)), 3)
out['E_by_year'] = ER.groupby(ER.d.str[:4])[['n_delisted_in_top', 'n_delisted_with_truth', 'n_survivor_without_truth']].mean().round(2).to_dict('index')
out['E_delisted_no_truth_codes'] = pd.Series([c for x in ER.delisted_no_truth for c in x]).value_counts().head(25).to_dict()

# ---- D. D2 嚴格往回看 ----
snaps = [s for s in B.load_snapshots() if not s['evidence'].startswith('C')]
V60 = P.v60_main(); rows = []
for d in P.main_rebalance_days('2006-01-01', '2013-12-31'):
    t = P.IDX.get_loc(d); v = V60[t]; px = P.Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10) & ~P.DRMASK
    o = np.argsort(-np.where(base, v, -np.inf), kind='stable')[:min(300, int(base.sum()))]
    for j in o:
        c = P.COLS[j]
        before = [s for s in snaps if c in s['map'] and s['date'] <= d and (d - s['date']).days <= 731]
        after = [s for s in snaps if c in s['map'] and s['date'] > d and (s['date'] - d).days <= 731]
        rows.append(dict(y=d.year, back=bool(before), fwd_only=(not before) and bool(after)))
DR = pd.DataFrame(rows)
out['D_strict_backward_dated_share'] = round(float(DR.back.mean()), 4)
out['D_forward_only_share_of_all'] = round(float(DR.fwd_only.mean()), 4)
out['D_by_year_backward'] = DR.groupby('y').back.mean().round(4).to_dict()
out['D_by_year_forward_only'] = DR.groupby('y').fwd_only.mean().round(4).to_dict()

print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

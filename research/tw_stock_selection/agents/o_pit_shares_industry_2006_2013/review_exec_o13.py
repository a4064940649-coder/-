#!/usr/bin/env python3
"""反方審查（成本與可成交／資料取得實際時點）o13：不修改研究員檔案，只讀其程式與 raw/，結果印到 stdout。
X1. 公告時滯（真正 PIT）重建：錨點只能在「公告日」之後使用（不用任何未來錨點），錨點後只用除權事件乘數外推；
    在 D1a 的 32 個季末與主策略換股日（每月第一個日 ≥ 11）比較真值前 100 與因果重建前 100。
    時滯口徑：
      L_rep（法定申報期限）：2012 以前 Q1/Q3 +1 個月、Q2 +2 個月、Q4 +4 個月；2013 起 Q1–Q3 +45 天、Q4 +3 個月；
            tejm（TEJ 月底流通股數）+15 天；ga（年底股本）同 Q4；o8 同季報。
      L_strict：tejm 也當季報口徑（不知道 TEJ 何時入庫）。
X2. 因果口徑下，真值前 100 中「沒有任何已公告錨點 → ADV 插補」的檔數（D1c 的因果版）。
X3. 金融旗標不一致對主策略池的實際影響：2841、5854 在主策略前 300 池的月數、60 日均成交值。
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
S_full, _ = B.build(M, tlo, thi, mult, VER)

def pub_date(d, src, strict):
    d = pd.Timestamp(d)
    if src == 'tejm' and not strict: return d + pd.Timedelta(days=15)
    q = (d.month - 1) // 3 + 1
    if d.month == 12 and src in ('ga',): q = 4
    if d.year <= 2012:
        lag = {1: 1, 2: 2, 3: 1, 4: 4}[q]; return d + pd.offsets.MonthEnd(lag)
    return d + (pd.Timedelta(days=45) if q < 4 else pd.offsets.MonthEnd(3))

out = {}
for label, strict in (('L_rep', False), ('L_strict', True)):
    A2 = A.copy(); A2['pub'] = [pub_date(d, s, strict) for d, s in zip(A2.date, A2.src)]
    def causal_at(tq):
        d = P.IDX[tq]
        K = A2[(A2.pub <= d) & (A2.t >= max(tq - 520, 0)) & (A2.t <= tq)].drop(columns='pub')
        K = B.merge_anchors(K)
        S, _ = B.build(K, max(tq - 520, 0), tq, mult, VER)   # 修正：tq<520 時負索引會清空整張表
        return S[tq]
    def evaluate(days, truth_from_anchor):
        rows, errs = [], []
        for tq in days:
            al = P.alive_at(tq); px = P.CFF[tq]
            tv = np.full(P.N, np.nan)
            if truth_from_anchor:
                for c, v in M[M.t == tq].set_index('code')['shares'].items(): tv[P.CI[c]] = v
            has = np.isfinite(tv)
            base = np.where(has, tv, S_full[tq])          # 真值：當日錨點，否則全錨點建構（近似真值，含前視）
            mt = np.where(al, base * px, np.nan); mt, imp_t = B.impute_mcap(tq, mt, al, ADV)
            sc = causal_at(tq)
            mc = np.where(al, sc * px, np.nan); mc, imp_c = B.impute_mcap(tq, mc, al, ADV)
            T = B.top_k(mt); H = B.top_k(mc)
            ok = [j for j in T if np.isfinite(base[j]) and not imp_t[j]]
            e = np.abs(np.log(mc[ok] / mt[ok]))
            errs.extend(e[np.isfinite(e)].tolist())
            rows.append(dict(d=str(P.IDX[tq].date()), overlap=len(set(T) & set(H)), med=float(np.nanmedian(e)),
                             n_causal_imputed_in_true_top=int(imp_c[T].sum()) - int((imp_c & imp_t)[T].sum()),
                             n_causal_imputed_total_top=int(imp_c[H].sum()),
                             missed=[P.COLS[j] for j in set(T) - set(H)]))
        R = pd.DataFrame(rows); E = np.array(errs)
        return dict(overlap_med=float(R.overlap.median()), overlap_min=int(R.overlap.min()), n_lt93=int((R.overlap < 93).sum()), n_lt95=int((R.overlap < 95).sum()),
                    pooled_med=round(float(np.median(E)), 5), p90=round(float(np.percentile(E, 90)), 4), share_gt_5pct=round(float((E > 0.05).mean()), 4),
                    by_year_overlap_min=R.groupby(R.d.str[:4]).overlap.min().to_dict(),
                    by_year_extra_imputed_mean=R.groupby(R.d.str[:4]).n_causal_imputed_in_true_top.mean().round(2).to_dict(),
                    worst=R.sort_values('overlap').head(6)[['d', 'overlap', 'missed']].to_dict('records'))
    cps = [P.ti(d) for d in SPEC['D1a_2006']['checkpoints']]
    out[label + '_qend'] = evaluate(cps, True)
    print(label, 'qend', {k: v for k, v in out[label + '_qend'].items() if k != 'worst'}, flush=True)
    print('  worst', out[label + '_qend']['worst'], flush=True)
    rb = [P.IDX.get_loc(d) for d in P.main_rebalance_days('2006-01-01', '2013-12-31')]
    out[label + '_rebal'] = evaluate(rb, False)
    print(label, 'rebal', {k: v for k, v in out[label + '_rebal'].items() if k != 'worst'}, flush=True)

# X3：2841、5854 在主策略池
V60 = P.v60_main(); x3 = {}
for d in P.main_rebalance_days('2006-01-01', '2013-12-31'):
    t = P.IDX.get_loc(d); v = V60[t]; px = P.Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10) & ~P.DRMASK
    o = list(np.argsort(-np.where(base, v, -np.inf), kind='stable')[:min(300, int(base.sum()))])
    for c in ('2841', '5854'):
        j = P.CI.get(c)
        if j is not None and j in o:
            x3.setdefault(c, []).append((str(d.date()), o.index(j) + 1, round(float(v[j]) / 1e8, 2)))
out['X3'] = {c: dict(n=len(v), first=v[0], last=v[-1], rank_med=float(np.median([r for _, r, _ in v])), adv60_e8_med=float(np.median([a for _, _, a in v]))) for c, v in x3.items()}
print('X3', out['X3'])
print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, default=str)[:200])

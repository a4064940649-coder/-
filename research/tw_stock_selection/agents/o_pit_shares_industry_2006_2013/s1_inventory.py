#!/usr/bin/env python3
"""第 0 步：來源盤點與覆蓋率（凍結閘門協定之前，不算任何閘門統計）＋ 非閘門診斷（來源間一致性 D1b）。輸出 explore.json。
事前定（寫在執行之前）：
  候選池＝每月最後交易日、存活上市普通股中 ADV120（120 日均成交值）前 150 檔（不用要驗證的股數）。
  覆蓋＝該股該月最後交易日到最近「直接錨點」（任一來源）的曆日距離：≤ 92 天（1 季）與 ≤ 366 天（1 年）兩種口徑。
  提前結束規則：若 ≤ 1 年口徑覆蓋 < 90%，且路徑 B 事件來源（2014 年前的現增／CB 轉換／減資事件表）不存在 → 判「資料不足、只存檔」。
  產業：主策略前 300 流動性池（換股日＝每月第一個日 ≥ 11 的交易日；60 日均成交值 ≥ 1,000 萬、股價 ≥ 10、前 300）股月，
        最近 dated 快照（evidence A/B、且快照內有該股）的距離；≤ 24 個月的比例。
  D1b（來源間一致性）：同股同日兩來源股數的 |對數差| 中位數（門檻 ≤ 1%），2006～2020 分期間報。
只讀 close_raw/value/flag 與 raw/；不讀 adj、ret；不算任何報酬。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import plib as P, pbuild as B
t0 = time.time()
out = {'rule': __doc__}

A = B.load_anchor_rows('2004-06-01', '2021-12-31')
A.to_parquet(P.SCR / 'anchors_all.parquet')
last_trade = {c: P.IDX[np.where(P.TRADED[:, j])[0].max()] if P.TRADED[:, j].any() else None for c, j in P.CI.items()}
dead2021 = {c for c, d in last_trade.items() if d is not None and d < pd.Timestamp('2021-01-01')}

# ---------------- 1. 來源盤點 ----------------
inv = {}
for s, g in A.groupby('src'):
    g6 = g[(g.date >= '2006-01-01') & (g.date <= '2013-12-31')]
    mon = g6.date.dt.month.value_counts().sort_index().to_dict()
    inv[s] = dict(first=str(g.date.min().date()), last=str(g.date.max().date()), n_codes=int(g.code.nunique()), n_rows=len(g),
                  n_codes_2006_2013=int(g6.code.nunique()), n_rows_2006_2013=len(g6), months_2006_2013=mon,
                  n_codes_delisted_before_2021=int(g6.code.isin(dead2021).groupby(g6.code).first().sum()) if len(g6) else 0)
inv_meta = {
 'tejm': dict(src='JunJie-Chang/ff5_replication data/raw/price.xlsx（TEJ Pro 月資料）', freq='月', delisted='否（2025-12 存活者）', evidence='A'),
 'tejq': dict(src='JackWang0102/...Taiwan Data/市值.xlsx（TEJ 期末市值）', freq='2005～2007 半年；2008 起季', delisted='否（2021-01 存活者）', evidence='A'),
 'gq': dict(src='yhzbambu/Lab_project goodinfo BzPerformance', freq='2006～2007 半年；2008 起季', delisted='否（460 檔 1xxx/2xxx 存活股）', evidence='A'),
 'ga': dict(src='henry753951/finance goodinfo StockAssetsStatus 年度股本', freq='年（12-31）', delisted='部分（goodinfo 仍保留頁面者）', evidence='A'),
 'twq': dict(src='taiwanan/TW_Stock_Exchange_Crawler 季股本', freq='季，2013Q1 起', delisted='是（out/）', evidence='A')}
for k in inv: inv[k].update(inv_meta.get(k, {}))
out['inventory'] = inv
out['inventory_not_found'] = {
 '(a) MOPS 2006～2012 財報／公司基本資料實收資本額封存': 'GitHub 搜尋 t05st22、t163sb01、t05st03、實收資本額、普通股股本：只有 IFRS 後（2013+）或單一公司教學檔；shinbutou/nccu_bigdata 只有 2011+ 重大訊息標題。找到的替代：TEJ（tejm／tejq）與 goodinfo（gq／ga）二手封存。',
 '(b) 證交所統計月報／上市公司市值逐檔表': '找不到（搜尋「上市公司市值 發行股數」只有新聞文字）。',
 '(c) 現增、CB 轉換、減資事件封存（2014 年前）': '找不到：TWTAUU（減資）只有程式碼；david26984741-cell/tw-cb-data 只有 CB 日行情與債息表（不含轉換股數）；TWT49U 只有 2014-01 起（joeycchen72）。o12 說的「icoolhome TWT49U 2003～2026 封存」實為 3 個 2026-01 單日檔（o8 manifest 已註明），2014 年前 TWT49U 不存在於本中心。',
 '(d) 2007～2014 上市清單／產業別': '找到 toomore/goristock、toomore/grs 的 stock_no.csv／twse_list.csv 6 個版本（UPDATE 2011-07-28、2011-09-26、2012-03-28、2012-07-31、2013-12-18；另一版無 UPDATE 列，commit 2011-08-17）。2011-07 以前沒有任何 dated 快照（FinMind TaiwanStockInfo 類清單的 date 欄全為 2023 抓取日，不能當 PIT）。',
 '(e) 臺灣50 歷次審核名單 2006～2013': '找不到（本中心 tw50_members 只從 2015-09 起；GitHub 搜尋只有新聞與 2026 成分）。P5 不做。'}
print('inventory', {k: (v['n_codes_2006_2013'], v['n_rows_2006_2013']) for k, v in inv.items()}, flush=True)

# ---------------- 2. 覆蓋率（ADV 前 150 候選池） ----------------
ADV = P.adv120()
anc_by = {c: np.array(sorted(g.date.values)) for c, g in A.groupby('code')}
src_by = {c: g for c, g in A.groupby('code')}
rows = []
for d in P.month_last_days('2006-01-01', '2013-12-31'):
    t = P.ti(d); al = P.alive_at(t)
    a = np.where(al & np.isfinite(ADV[t]), ADV[t], -1); o = np.argsort(-a)[:150]
    for rk, j in enumerate(o):
        if a[j] <= 0: continue
        c = P.COLS[j]; ds = anc_by.get(c)
        dist = int(np.min(np.abs((ds - np.datetime64(d)).astype('timedelta64[D]').astype(int)))) if ds is not None and len(ds) else 99999
        g = src_by.get(c); srcs = ''
        if g is not None:
            near = g[(g.date - d).abs() <= pd.Timedelta(days=92)]; srcs = ','.join(sorted(set(near.src)))
        rows.append(dict(d=d, y=d.year, code=c, adv_rank=rk + 1, dist=dist, srcs_q=srcs, delisted_before_2021=c in dead2021))
CV = pd.DataFrame(rows)
CV.to_parquet(P.SCR / 'coverage_rows.parquet')
def cov(x): return dict(n=len(x), le_1q=round(float((x.dist <= 92).mean()), 4), le_1y=round(float((x.dist <= 366).mean()), 4))
out['coverage_adv150'] = dict(all=cov(CV), top120_by_adv=cov(CV[CV.adv_rank <= 120]),
                             by_year={int(y): cov(g) for y, g in CV.groupby('y')},
                             survivors=cov(CV[~CV.delisted_before_2021]), delisted_before_2021=cov(CV[CV.delisted_before_2021]),
                             delisted_share_of_stock_months=round(float(CV.delisted_before_2021.mean()), 4),
                             src_share_le_1q={s: round(float(CV.srcs_q.str.contains(s).mean()), 4) for s in ['tejm', 'tejq', 'gq', 'ga', 'twq']},
                             uncovered_1y_codes=CV[CV.dist > 366].groupby('code').size().sort_values(ascending=False).head(40).to_dict(),
                             uncovered_1q_codes=CV[CV.dist > 92].groupby('code').size().sort_values(ascending=False).head(40).to_dict())
c1y = out['coverage_adv150']['all']['le_1y']
out['early_stop'] = dict(coverage_le_1y=c1y, path_b_event_source_exists=False,
                         stop=bool(c1y < 0.90), note='Path B 事件來源（2014 年前現增／CB／減資）不存在；≤1 年覆蓋 ≥ 90% → 繼續')
print('coverage', out['coverage_adv150']['all'], out['coverage_adv150']['top120_by_adv'], flush=True)

# ---------------- 3. 產業快照距離（主策略前 300 流動性池） ----------------
snaps = B.load_snapshots()
out['industry_snapshots'] = [dict(date=str(s['date'].date()), src=s['src'], evidence=s['evidence'], n=len(s['map'])) for s in snaps]
V60 = P.v60_main()
irows = []
for d in P.main_rebalance_days('2006-01-01', '2013-12-31'):
    t = P.IDX.get_loc(d); v = V60[t]; px = P.Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10) & ~P.DRMASK
    o = np.argsort(-np.where(base, v, -np.inf), kind='stable')[:min(300, int(base.sum()))]
    for j in o:
        c = P.COLS[j]
        ds = [abs((s['date'] - d).days) for s in snaps if c in s['map'] and not s['evidence'].startswith('C')]
        irows.append(dict(d=d, y=d.year, code=c, dist_days=min(ds) if ds else 99999))
IR = pd.DataFrame(irows); IR.to_parquet(P.SCR / 'industry_pool_rows.parquet')
out['industry_coverage_pool300'] = dict(n=len(IR), le_24m=round(float((IR.dist_days <= 731).mean()), 4),
                                       by_year={int(y): round(float((g.dist_days <= 731).mean()), 4) for y, g in IR.groupby('y')},
                                       never_in_any_snapshot=int((IR.dist_days == 99999).sum()),
                                       never_codes=IR[IR.dist_days == 99999].code.value_counts().head(30).to_dict())
print('industry', out['industry_coverage_pool300']['le_24m'], out['industry_coverage_pool300']['by_year'], flush=True)

# ---------------- 4. D1b：來源間一致性（非閘門診斷） ----------------
W = A.pivot_table(index=['code', 'date'], columns='src', values='shares', aggfunc='last')
d1b = {}
pairs = [('tejm', 'tejq'), ('tejm', 'gq'), ('tejq', 'gq'), ('tejm', 'ga'), ('tejq', 'ga'), ('gq', 'ga'), ('tejm', 'twq'), ('tejq', 'twq'), ('ga', 'twq')]
for a_, b_ in pairs:
    if a_ not in W or b_ not in W: continue
    x = W[[a_, b_]].dropna(); x = x[(x[a_] > 0) & (x[b_] > 0)]
    if not len(x): continue
    ld = np.log(x[a_] / x[b_]); dt = x.index.get_level_values('date')
    seg = {}
    for lab, lo, hi in [('2006_2007', '2006-01-01', '2007-12-31'), ('2008_2010', '2008-01-01', '2010-12-31'), ('2011_2013', '2011-01-01', '2013-12-31'), ('2014_2020', '2014-01-01', '2020-12-31')]:
        m = (dt >= lo) & (dt <= hi)
        if m.sum(): seg[lab] = dict(n=int(m.sum()), med_abs=round(float(ld[m].abs().median()), 5), p90_abs=round(float(ld[m].abs().quantile(0.9)), 4), share_gt_2pct=round(float((ld[m].abs() > 0.02).mean()), 4), med_signed=round(float(ld[m].median()), 5))
    d1b[f'{a_}_vs_{b_}'] = seg
out['D1b_source_consistency'] = d1b
# 大型股限定（ADV 前 150 股月對應的季末）
big = set(zip(CV.code, CV.d.dt.to_period('Q')))
xs = []
for (c, d), r in W.iterrows():
    if (c, pd.Timestamp(d).to_period('Q')) in big: xs.append((c, d, r.get('tejm'), r.get('tejq'), r.get('gq'), r.get('ga')))
XB = pd.DataFrame(xs, columns=['code', 'date', 'tejm', 'tejq', 'gq', 'ga'])
d1b_big = {}
for a_, b_ in [('tejm', 'tejq'), ('tejm', 'gq'), ('tejq', 'gq'), ('tejm', 'ga'), ('tejq', 'ga')]:
    x = XB[[a_, b_, 'code', 'date']].dropna()
    if len(x):
        ld = np.log(x[a_].astype(float) / x[b_].astype(float))
        d1b_big[f'{a_}_vs_{b_}'] = dict(n=len(x), med_abs=round(float(ld.abs().median()), 5), p90_abs=round(float(ld.abs().quantile(0.9)), 4), share_gt_2pct=round(float((ld.abs() > 0.02).mean()), 4),
                                         worst=x.assign(ld=ld.round(4)).reindex(ld.abs().sort_values(ascending=False).index).head(8)[['code', 'date', 'ld']].astype(str).to_dict('records'))
out['D1b_large_caps_2006_2013'] = d1b_big
print('D1b big', {k: (v['n'], v['med_abs']) for k, v in d1b_big.items()}, flush=True)
out['elapsed_s'] = round(time.time() - t0)
(P.HERE / 'explore.json').write_text(P.dumps(out), encoding='utf-8')
print('done', out['elapsed_s'])

#!/usr/bin/env python3
"""獨立複製（第 1 部分）：從原始股本來源自己重建「當時已公告」股數（版本 c），不讀 shares_pit、也不讀 o7 shares_est。
來源（都在 o_0050_tilt_pit/raw，唯讀）：
  - twsec（taiwanan/TW_Stock_Exchange_Crawler）上市非金融季股本（千元），2013Q1～2021Q1，含 out/ 下市股。
  - hawan（hawan0228/fintech_ai，FinMind OrdinaryShare，元），2015Q1～2025Q4，112 檔大型股（含金融）。
  - TDCC 集保總股數快照 2021-07-02、2021-09-24；weihautin 股本快照 2015-05-22、2017-07-06。
  - MI_QFIIS 每日發行股數（o7 raw/agents/vf_twse_foreign.parquet）2025-10-07 起。
  - 事件（除權日當天即知）：TWT49U（icoolhome）純「權」；exdiv.csv（2024-08 起）stock_ratio＋rights_ratio；corp_actions 減資；面板價格比 ≥2 倍的分割。
規則（嚴格 PIT）：季末股本在法定公告期限才算已知（Q1→5/15、Q2→8/14、Q3→11/14、Q4→次年 3/31）；快照在快照日即知。
  每天的股數＝「已知錨點中錨點日最新者」× 錨點日之後、當天以前的事件因子。第一個錨點以前：用第一個錨點值（新上市公司的公開說明書股數，近似）。
  面額：預設 10；若 TDCC 2021-07-02 ÷（2021Q1 股本/10）或 MI_QFIIS ÷（最新股本/10）接近 2、4、10，視為面額 5、2.5、1（改面額日＝面板偵測到的分割日，找不到就全期）。
輸出 replicate/shares_c.parquet。python3 -I -B r_shares.py"""
import sys, json, glob, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
RAW8 = AG / 'o_0050_tilt_pit' / 'raw'
O7 = AG / 'o_capmatched_selection'
C = pd.read_parquet(O7 / 'raw' / 'panel' / 'close_raw.parquet')
IDX, COLS = C.index, list(C.columns); CS = set(COLS)

def qend(y, q): return pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0)
def known(qe):
    y, m = qe.year, qe.month
    return {3: pd.Timestamp(y, 5, 15), 6: pd.Timestamp(y, 8, 14), 9: pd.Timestamp(y, 11, 14), 12: pd.Timestamp(y + 1, 3, 31)}[m]

anc = {}   # code -> list of (asof, known_date, capital_or_shares, kind, src)
def add(c, asof, kd, val, kind, src):
    if c in CS and np.isfinite(val) and val > 0: anc.setdefault(c, []).append((pd.Timestamp(asof), pd.Timestamp(kd), float(val), kind, src))
# twsec
for f in glob.glob(str(RAW8 / 'twsec' / 'tse_earning_raw_data' / '*.csv')) + glob.glob(str(RAW8 / 'twsec' / 'tse_earning_raw_data' / 'out' / '*.csv')):
    c = Path(f).stem
    if c not in CS: continue
    try: d = pd.read_csv(f, skiprows=1)
    except Exception: continue
    if '股本' not in d.columns: continue
    for q, v in zip(d['季度'], pd.to_numeric(d['股本'], errors='coerce')):
        try: y, qq = str(q).split(' Q'); qe = qend(int(y), int(qq))
        except Exception: continue
        add(c, qe, known(qe), v * 1000.0, 'cap', 'twsec')
# hawan
H = pd.read_csv(RAW8 / 'hawan' / 'capital_rows.csv', dtype={'stock_id': str}); H = H[H.type == 'OrdinaryShare']
for r in H.itertuples():
    qe = pd.Timestamp(r.date); add(r.stock_id, qe, known(qe), r.value, 'cap', 'hawan')
# 快照
for f in ['2021-07-02', '2021-09-24']:
    td = pd.read_csv(RAW8 / 'tdcc' / f'total_{f}.csv', dtype={'code': str})
    for c, v in zip(td.code, td.shares): add(c, f, f, float(v), 'shares', 'tdcc')
w15 = pd.read_csv(RAW8 / 'weihautin' / 'update_csv' / 'StockList_20150522.csv', dtype={'代號': str})
for c, v in zip(w15['代號'], w15['股本(億)']): add(c, '2015-05-22', '2015-05-22', float(v) * 1e8, 'cap', 'weihautin')
w17 = pd.read_csv(RAW8 / 'weihautin' / 'update_csv' / 'StockList.csv', dtype={'代號': str})
for c, v, p in zip(w17['代號'], w17['股本(億)'], w17['面值(元)']):
    try: add(c, '2017-07-06', '2017-07-06', float(v) * 1e8 / float(p), 'shares', 'weihautin')
    except Exception: pass
Q = pd.read_parquet(O7 / 'raw' / 'agents' / 'vf_twse_foreign.parquet')
Q = Q[Q.total_shares > 0]
QP = Q.pivot_table(index=pd.to_datetime(Q.date), columns='code', values='total_shares', aggfunc='last')

# 面額
def latest_cap(c, before):
    xs = [a for a in anc.get(c, []) if a[3] == 'cap' and a[0] <= before]
    return max(xs, key=lambda a: a[0])[2] if xs else None
close = C
def split_day(c, ratio, a, b):
    s = close[c].loc[a:b].dropna()
    if len(s) < 2: return None
    r = (s.shift(1) / s).dropna()
    hit = r[(r > ratio * 0.8) & (r < ratio * 1.25)]
    return hit.index[0] if len(hit) else None
PAR = {}   # c -> list of (from_date, par)
def near(x):
    for k, par in ((2.0, 5.0), (4.0, 2.5), (10.0, 1.0)):
        if abs(x / k - 1) < 0.15: return par
    return 10.0
for c in COLS:
    p21 = 10.0; p25 = 10.0
    tdv = [a for a in anc.get(c, []) if a[4] == 'tdcc' and a[0] == pd.Timestamp('2021-07-02')]
    cp = latest_cap(c, pd.Timestamp('2021-06-30'))
    if tdv and cp: p21 = near(tdv[0][2] / (cp / 10))
    if c in QP.columns:
        qv = QP[c].dropna()
        cp2 = latest_cap(c, pd.Timestamp('2025-09-30'))
        if len(qv) and cp2: p25 = near(qv.iloc[0] / (cp2 / 10))
    if p21 == 10.0 and p25 == 10.0: continue
    if p21 == p25: PAR[c] = [(IDX[0], p21)]
    else:
        sd = split_day(c, p21 / p25, '2021-07-01', '2025-10-07')
        PAR[c] = [(IDX[0], p21), (sd if sd is not None else pd.Timestamp('2025-10-07'), p25)]
def par_at(c, d):
    if c not in PAR: return 10.0
    p = 10.0
    for fd, pp in PAR[c]:
        if d >= fd: p = pp
    return p

# 事件因子（日期 → 股數倍數）
ev = {}
def addev(c, d, g, src):
    if c in CS and np.isfinite(g) and g > 0 and abs(g - 1) > 1e-6: ev.setdefault(c, []).append((pd.Timestamp(d), float(g), src))
# TWT49U 封存（joeycchen72，2014～2026）＋exdiv_5y（2021-10 起）：權＝prev/ref；權息：現金部分 D＝同股最近一次純「息」金額（400 日內，≤ value），找不到就 0.5×value
tws = []
for f in glob.glob(str(RAW8 / 'jc_exright' / 'data' / 'twse-exright' / '*' / '*.csv.gz')):
    try: tws.append(pd.read_csv(f, dtype={'code': str}))
    except Exception: pass
TW = pd.concat(tws, ignore_index=True)[['date', 'code', 'prev_close', 'ref_price', 'value', 'type']]
e5 = pd.read_csv(RAW8 / 'screener' / 'exdiv_5y.csv.gz', dtype={'code': str})
e5 = e5[e5.market == 'TWSE'].rename(columns={'kind': 'type'})[['date', 'code', 'prev_close', 'ref_price', 'value', 'type']]
TW = pd.concat([TW, e5[~e5.set_index(['date', 'code']).index.isin(TW.set_index(['date', 'code']).index)]], ignore_index=True)
TW['date'] = pd.to_datetime(TW['date']); TW = TW.sort_values('date')
for col in ('prev_close', 'ref_price', 'value'): TW[col] = pd.to_numeric(TW[col], errors='coerce')
n_tw = 0; lastcash = {}
for r in TW.itertuples(index=False):
    if r.type == '息': lastcash[r.code] = (r.date, r.value); continue
    if not (np.isfinite(r.prev_close) and np.isfinite(r.ref_price) and r.ref_price > 0): continue
    if r.type == '權': g = r.prev_close / r.ref_price
    elif r.type == '權息':
        lc = lastcash.get(r.code)
        D = min(lc[1], r.value) if (lc and (r.date - lc[0]).days <= 400 and np.isfinite(lc[1])) else 0.5 * r.value
        g = (r.prev_close - D) / r.ref_price
    else: continue
    if 1.0 < g < 3.0: addev(r.code, r.date, g, 'twt49u'); n_tw += 1
# 面板分割偵測：前一有效收盤 / 當日收盤 ≈ 2、4、5、10（±8%），且之後 10 日收盤中位數 / 之前 10 日中位數 ≈ 1/m（±15%）
n_split = 0
for c in COLS:
    v = C[c].dropna()
    if len(v) < 22: continue
    a = v.to_numpy(); r = np.r_[np.nan, a[:-1] / a[1:]]
    for k in np.where(np.isfinite(r) & (r > 1.8))[0]:
        if k < 10 or k + 10 > len(a): continue
        q = np.median(a[k:k + 10]) / np.median(a[k - 10:k])
        if any(e[2] == 'split' and abs((v.index[k] - e[0]).days) < 120 for e in ev.get(c, [])): continue   # 價格資料來回跳動：120 日內只認第一次
        for m in (2.0, 4.0, 5.0, 10.0):
            if abs(r[k] / m - 1) < 0.08 and abs(q * m - 1) < 0.15: addev(c, v.index[k], m, 'split'); n_split += 1
# 去重（同股同日只留一筆）
for c in ev:
    seen = {}
    for d, g, s in sorted(ev[c], key=lambda x: (x[0], x[2] != 'exdiv')):
        seen.setdefault(d, (d, g, s))
    ev[c] = sorted(seen.values())

# 建表（逐股向量化）
S = np.full((len(IDX), len(COLS)), np.nan)
stats = dict(n_anchor_stocks=len(anc), n_events=sum(len(v) for v in ev.values()), n_twt49u=n_tw, n_split=n_split, par_overrides={c: [(str(a.date()), p) for a, p in v] for c, v in PAR.items()})
IDXn = IDX.values
for jc, c in enumerate(COLS):
    A = anc.get(c)
    if not A: continue
    par = np.full(len(IDX), 10.0)
    for fd, pp in PAR.get(c, []): par[IDXn >= np.datetime64(fd)] = pp
    E = ev.get(c, [])
    lg = np.zeros(len(IDX))
    for d, g, _ in E:
        k = IDX.searchsorted(d)
        if k < len(IDX): lg[k] += math.log(g)
    cum = np.cumsum(lg)                              # cum[t]＝Σ log g（事件日 ≤ IDX[t]）
    def cum_at(ts):
        k = IDX.searchsorted(ts, side='right') - 1
        return cum[k] if k >= 0 else 0.0
    A = sorted(A, key=lambda a: a[1])
    seg = []; best = None
    for a in A:
        if best is None or a[0] >= best[0]: best = a
        seg.append((a[1], best))
    first = min(A, key=lambda a: a[1])
    def base_vec(a, sl):   # 股本型錨點用「錨點日」的面額（之後的分割由事件處理）
        return np.full(sl.stop - sl.start, a[2] if a[3] == 'shares' else a[2] / par[min(IDX.searchsorted(a[0]), len(IDX) - 1)])
    k0 = IDX.searchsorted(seg[0][0])
    S[:k0, jc] = base_vec(first, slice(0, k0))       # 第一個錨點公告前：用第一個錨點值
    for i, (kd, a) in enumerate(seg):
        s0 = IDX.searchsorted(kd); s1 = IDX.searchsorted(seg[i + 1][0]) if i + 1 < len(seg) else len(IDX)
        if s1 <= s0: continue
        sl = slice(s0, s1)
        S[sl, jc] = base_vec(a, sl) * np.exp(cum[sl] - cum_at(a[0]))
    if c in QP.columns:
        q = QP[c].dropna()
        if len(q):
            qq = q.reindex(IDX).ffill(); m = IDXn >= np.datetime64(q.index[0])
            S[m, jc] = qq.to_numpy()[m]
out = pd.DataFrame(S, index=IDX, columns=COLS)
out.loc[out.index < pd.Timestamp('2013-01-01')] = np.nan
out.to_parquet(HERE / 'shares_c.parquet')
# 檢核：TDCC 真值（同日快照為錨點，屬自我檢核，另列 2021-08-02 前一日用 07-02 值）與 4763
chk = {}
for d in ['2019-06-28', '2021-06-30', '2023-06-30', '2025-06-27', '2025-07-15', '2025-10-08']:
    t = IDX[IDX <= pd.Timestamp(d)][-1]; chk[d] = float(out.loc[t, '4763']) if '4763' in out else None
stats['4763'] = chk
stats['2317_2018'] = {str(d.date()): float(out.loc[d, '2317']) for d in IDX[(IDX >= '2018-06-25') & (IDX <= '2018-11-05')][::10]}
(HERE / 'shares_c_meta.json').write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in stats.items() if k != 'par_overrides'}, ensure_ascii=False, indent=1, default=str)); print('par overrides', len(PAR))

#!/usr/bin/env python3
"""第 0 步：PIT 股數重建（看任何報酬之前完成並凍結）。只用收盤價（市值名次）與公開股本／事件資料，不讀 ret、不算任何報酬。

方法（真值路徑重建）：
  錨點（季末普通股股數＝普通股股本 ÷ 面額）：hawan（FinMind OrdinaryShare，2015Q1～2025Q4，112 檔）＞ twsec（上市非金融每季股本，2013Q1～2021Q1，含 out/ 下市股）
    ＞ jhc（74 檔，日期為公告日 → 對應季末，只補缺）。2025-10-07 起 MI_QFIIS 實際發行股數（日）為錨點。
  面額：預設 10 元；若 MI_QFIIS（2025-10）或 TDCC（2021-07）與報表股本的比例接近 2.5／5／1 等面額，改用該面額（只用最早可得的比對，見 par_log）。
  兩錨點之間：比例 r＝S1/S0。若期間有除權（TWT49U 權／權息、exdiv_5y、exdiv、corp_actions 減資、面板 flag 3 參考價重設）→ 整段變動放在第一個事件日（階梯）；
    沒有事件 → 依交易日對數線性內插（可轉債轉換、員工認股等逐步變動）。
  第一個錨點之前：用 TWT49U 事件的價格比反推（權：s＝前收/參考價−1；權息：扣掉當年度同股或市場息值殖利率中位數）。
  最後錨點之後（twsec-only 股票 2021Q1 以後）：接 o7 回推股數（capmatched shares_est，2021-10-07 起為 R1/R2 實際事件段，錨定 MI_QFIIS），
    兩者之間的缺口用同樣的事件／內插規則橋接。o7 回推的 2021-10-07 以前（R3 啟發式）一律不用。
  沒有任何錨點的股票（下市金融股等）：TDCC 2021 不用來建構（保留為獨立檢核）；改用 o7 的 0050 報酬校準股數（tw50_fill）或標記「無 PIT 股數」。
輸出：data/shares_pit.parquet（日 × 股，2013-01-02 起）、data/shares_src.parquet（來源代碼）、results.json['gate']。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tlib as L
RAW, DATA = L.RAW, L.DATA
IDX, COLS, CI, N = L.IDX, L.COLS, L.CI, L.N
T0 = pd.Timestamp('2013-01-02')
ID = IDX[IDX >= T0]; NT = len(ID)
OUT = {}

def qend(y, q): return pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0)
# ---------------- 錨點 ----------------
anc = {}   # code -> {date: (shares_raw_ntd_capital, src)}
def put(c, d, cap, src, prio):
    if c not in CI or not np.isfinite(cap) or cap <= 0: return
    a = anc.setdefault(c, {})
    if d in a and a[d][2] >= prio: return
    a[d] = (float(cap), src, prio)
# hawan（元）
H = pd.read_csv(RAW / 'hawan' / 'capital_rows.csv', dtype={'stock_id': str})
Ho = H[H.type == 'OrdinaryShare']
for r in Ho.itertuples(): put(r.stock_id, pd.Timestamp(r.date), r.value, 'hawan', 3)
# twsec（千元）
tw_files = list((RAW / 'twsec' / 'tse_earning_raw_data').glob('*.csv')) + list((RAW / 'twsec' / 'tse_earning_raw_data' / 'out').glob('*.csv'))
n_tw = 0
for f in tw_files:
    c = f.stem
    if c not in CI: continue
    try: d = pd.read_csv(f, skiprows=1)
    except Exception: continue
    if '股本' not in d.columns: continue
    for q, v in zip(d['季度'], pd.to_numeric(d['股本'], errors='coerce')):
        try: y, qq = str(q).split(' Q'); dt = qend(int(y), int(qq))
        except Exception: continue
        put(c, dt, v * 1000.0, 'twsec', 2); n_tw += 1
# jhc（元；日期是公告日 → 對應季末）
def pub2q(d):
    d = pd.Timestamp(d)
    if d.month in (3, 4): return qend(d.year - 1, 4)
    if d.month == 5: return qend(d.year, 1)
    if d.month in (8,): return qend(d.year, 2)
    if d.month in (11,): return qend(d.year, 3)
    return None
for f in (RAW / 'jhc' / 'GAT-main' / 'data').glob('*.csv'):
    c = f.stem
    try: d = pd.read_csv(f, usecols=['date', 'OrdinaryShare'])
    except Exception: continue
    for dt, v in zip(d['date'], d['OrdinaryShare']):
        q = pub2q(dt)
        if q is not None: put(c, q, float(v), 'jhc', 1)
# 沒有任何報表錨點的股票：用 TDCC 2021-07-02／09-24 合計股數與 weihautin 2015-05-22／2017-07-06 股本快照當錨點
# （這些股票在對應檢核點的閘門統計中排除，避免用真值檢核真值）。
SNAP_USED = {}
_snaps = []
for f in ['2021-07-02', '2021-09-24']:
    td = pd.read_csv(RAW / 'tdcc' / f'total_{f}.csv', dtype={'code': str}); _snaps.append((pd.Timestamp(f), td.set_index('code')['shares'].astype(float), f'TDCC_{f}'))
w15 = pd.read_csv(RAW / 'weihautin' / 'update_csv' / 'StockList_20150522.csv', dtype={'代號': str}).set_index('代號')
_snaps.append((pd.Timestamp('2015-05-22'), w15['股本(億)'].astype(float) * 1e8 / 10, 'weihautin_2015-05-22'))
w17 = pd.read_csv(RAW / 'weihautin' / 'update_csv' / 'StockList.csv', dtype={'代號': str}).set_index('代號')
_snaps.append((pd.Timestamp('2017-07-06'), w17['股本(億)'].astype(float) * 1e8 / w17['面值(元)'].astype(float), 'weihautin_2017-07-06'))
_noanc = [c for c in COLS if c not in anc and not c.startswith('91')]
for d, ser, nm in _snaps:
    for c in _noanc:
        if c in ser.index and ser[c] > 0 and np.isfinite(L.CFF[min(IDX.searchsorted(d), len(IDX) - 1), CI[c]]):
            put(c, d, float(ser[c]) * 10.0, nm, 0)   # 存成「股本元」口徑（÷面額 10 還原股數）
            SNAP_USED.setdefault(nm, []).append(c)
# 只有 MI_QFIIS（2025-10-07 起）的股票（近年上市、無 hawan）：首個 MI 值也當錨點，往前用 TWT49U 價格比回推到上市日
_vf0 = pd.read_parquet(RAW / 'agents' / 'vf_twse_foreign.parquet'); _vf0 = _vf0[_vf0.code.isin(CI) & (_vf0.total_shares > 0)].sort_values('date').drop_duplicates('code')
for r in _vf0.itertuples():
    if r.code not in anc and not r.code.startswith('91'):
        put(r.code, pd.Timestamp(r.date), float(r.total_shares) * 10.0, 'MIQFIIS_first', 0); SNAP_USED.setdefault('MIQFIIS_first', []).append(r.code)
OUT['snapshot_anchors_used'] = {k: len(v) for k, v in SNAP_USED.items()}
OUT['anchor_counts'] = {s: int(sum(1 for a in anc.values() for v in a.values() if v[1] == s)) for s in ['hawan', 'twsec', 'jhc']}
OUT['snapshot_anchor_codes'] = SNAP_USED
OUT['n_stocks_with_statement_anchor'] = len(anc)

# 來源一致性（同一季末兩個來源都有）：hawan vs twsec
cons = []
tw_only = {}
for f in tw_files:
    pass
for c, a in anc.items():
    pass
# 重新讀一次 twsec 值做比對
twv = {}
for f in tw_files:
    c = f.stem
    try: d = pd.read_csv(f, skiprows=1)
    except Exception: continue
    if '股本' not in d.columns: continue
    for q, v in zip(d['季度'], pd.to_numeric(d['股本'], errors='coerce')):
        try: y, qq = str(q).split(' Q'); twv[(c, qend(int(y), int(qq)))] = v * 1000.0
        except Exception: pass
hwv = {(r.stock_id, pd.Timestamp(r.date)): r.value for r in Ho.itertuples()}
for k in set(twv) & set(hwv):
    if twv[k] > 0 and hwv[k] > 0: cons.append(abs(math.log(twv[k] / hwv[k])))
cons = np.array(cons)
OUT['source_consistency_hawan_vs_twsec'] = {'n': int(len(cons)), 'median_abs_log': float(np.median(cons)), 'p90': float(np.quantile(cons, 0.9)), 'share_gt_2pct': float((cons > 0.02).mean())}

# ---------------- MI_QFIIS 實際發行股數（2025-10-07 起）----------------
vf = pd.read_parquet(RAW / 'agents' / 'vf_twse_foreign.parquet'); vf['date'] = pd.to_datetime(vf['date'])
vf = vf[vf.code.isin(CI) & (vf.total_shares > 0)]
VF = vf.pivot_table(index='date', columns='code', values='total_shares', aggfunc='last').sort_index()

# ---------------- 面額（隨時間）：偵測面額變更（分割）----------------
# 價格跳動超出漲跌幅（|log(收盤/前收)| > log 1.25）且非除權息日 → 公司行動事件日（也加入階梯位置）；
# 其中比例 < 0.4（或 > 2.5）、且前後季末報表股本（元）變動 < 10% → 面額變更，k＝round(1/比例) ∈ {2,4,5,10}，之後面額＝前面額/k。
td21 = pd.read_csv(RAW / 'tdcc' / 'total_2021-07-02.csv', dtype=str); td21 = td21.set_index('code')['shares'].astype(float)
with np.errstate(invalid='ignore', divide='ignore'):
    JR = L.Cn / L.PREVC
_lj = np.abs(np.log(np.where(np.isfinite(JR) & (JR > 0), JR, 1)))
JUMP = L.TRADED & np.isfinite(JR) & (_lj > math.log(1.25))   # 純現金除息極少 > 20%；超過 25% 的跳動一律視為股數事件候選
jump_dates = {}
for j in np.where(JUMP.any(axis=0))[0]:
    for t in np.where(JUMP[:, j])[0]: jump_dates.setdefault(COLS[j], []).append((IDX[t], float(JR[t, j])))
par_steps = {}; par_log = {}
for c, lst in jump_dates.items():
    a = anc.get(c)
    if not a: continue
    ks = sorted(a)
    for d, r in lst:
        if not (r < 0.4 or r > 2.5): continue
        b4 = [k for k in ks if k < d]; af = [k for k in ks if k >= d]
        if not b4 or not af: continue
        capr = a[af[0]][0] / a[b4[-1]][0]
        if abs(math.log(capr)) < 0.1:
            k = min([2, 4, 5, 10], key=lambda x: abs(math.log(x * r)))
            if abs(math.log(k * r)) < 0.35:
                par_steps.setdefault(c, []).append((d, k)); par_log.setdefault(c, []).append({'date': str(d.date()), 'price_ratio': r, 'k': k, 'cap_ratio': float(capr)})
def par_at(c, d):
    p = 10.0
    for ds, k in par_steps.get(c, []):
        if d >= ds: p /= k
    return p
# 核對：2025-09-30 報表股本 ÷ MI_QFIIS 首日股數 ≈ 面額
par_check = {}
for c, a in anc.items():
    if c in VF.columns and VF[c].notna().any():
        q = [k for k in sorted(a) if pd.Timestamp('2025-06-30') <= k <= pd.Timestamp('2025-09-30')]
        if q:
            ratio = a[q[-1]][0] / VF[c].dropna().iloc[0]
            if abs(math.log(ratio / par_at(c, q[-1]))) > 0.15: par_check[c] = {'implied_par': float(ratio), 'model_par': par_at(c, q[-1])}
OUT['par_steps'] = par_log; OUT['par_mismatch_vs_MIQFIIS_2025'] = par_check

# ---------------- 事件日（只定階梯位置；第一錨點之前另用價格比）----------------
ev_dates = {}; ev_px = []   # (code, date, kind, prev, ref)
import glob, gzip
for f in sorted((RAW / 'jc_exright' / 'data' / 'twse-exright').glob('*/*.csv.gz')):
    d = pd.read_csv(f, dtype={'code': str})
    d = d[d.code.isin(CI)]
    for r in d.itertuples():
        if '權' in str(r.type):
            ev_dates.setdefault(r.code, set()).add(pd.Timestamp(r.date))
            ev_px.append((r.code, pd.Timestamp(r.date), str(r.type), float(r.prev_close), float(r.ref_price)))
JC = pd.DataFrame(ev_px, columns=['code', 'date', 'kind', 'prev', 'ref'])
e5 = pd.read_csv(RAW / 'screener' / 'exdiv_5y.csv.gz', dtype={'code': str}); e5 = e5[(e5.market == 'TWSE') & e5.code.isin(CI) & e5.kind.astype(str).str.contains('權')]
for r in e5.itertuples(): ev_dates.setdefault(r.code, set()).add(pd.Timestamp(r.date))
ex = pd.read_csv(RAW / 'screener' / 'exdiv.csv', dtype={'code': str}); ex = ex[(ex.market == 'TWSE') & ex.code.isin(CI) & ((ex.stock_ratio.fillna(0) > 0) | (ex.rights_ratio.fillna(0) > 0))]
for r in ex.itertuples(): ev_dates.setdefault(r.code, set()).add(pd.Timestamp(r.date))
ca = pd.read_csv(RAW / 'screener' / 'corp_actions.csv', dtype={'code': str}); ca = ca[ca.code.isin(CI)]
for r in ca.itertuples(): ev_dates.setdefault(r.code, set()).add(pd.Timestamp(r.date))
F3 = (L.FLAG == 3)
for c in COLS[F3.any().to_numpy()]:
    for d in L.FLAG.index[F3[c].to_numpy()]: ev_dates.setdefault(c, set()).add(d)
for c, lst in jump_dates.items():
    for d, r in lst: ev_dates.setdefault(c, set()).add(d)
OUT['n_event_dates'] = int(sum(len(v) for v in ev_dates.values()))

# 息值殖利率（權息事件的現金部分估計）：同年 TWSE 純「息」事件的 value/prev 中位數（jc 資料）
jcall = []
for f in sorted((RAW / 'jc_exright' / 'data' / 'twse-exright').glob('*/*.csv.gz')):
    d = pd.read_csv(f, dtype={'code': str}); jcall.append(d)
JCA = pd.concat(jcall); JCA['date'] = pd.to_datetime(JCA['date'])
pure = JCA[JCA.type == '息']; DY = (pure.value / pure.prev_close).groupby(pure.date.dt.year).median().to_dict()
def stock_ratio_from_px(kind, prev, ref, year):
    g = prev / ref
    if kind == '權': return g
    dy = DY.get(year, 0.035)
    return max(g * (1 - dy), 1.0)

# ---------------- o7 回推股數（只用 2021-10-07 以後段）----------------
SE = pd.read_parquet(RAW / 'agents' / 'capmatched_shares_est.parquet')
SE.index = pd.to_datetime(SE.index)
SE = SE.reindex(columns=COLS)
SE_OK_FROM = pd.Timestamp('2021-10-07')

# ---------------- 逐股建路徑 ----------------
S = np.full((NT, N), np.nan); SRC = np.zeros((NT, N), np.int8)
# 來源碼：1 錨點／錨點間階梯或內插（報表）；2 第一錨點前的價格比反推；3 o7 回推（2021-10 後段）；4 MI_QFIIS 實值；5 0050 報酬校準（tw50_fill）；0 無
pos = {d: i for i, d in enumerate(ID)}
def tpos(d):
    k = ID.searchsorted(pd.Timestamp(d));  return min(k, NT - 1)
def fill_between(j, t0, s0, t1, s1, evs, code_src):
    """ID 位置 t0（含）到 t1（不含）的路徑；t0 值 s0，t1 值 s1。"""
    if t1 <= t0: return
    r = s1 / s0
    if abs(math.log(r)) < 1e-6:
        S[t0:t1, j] = s0; SRC[t0:t1, j] = code_src; return
    e = sorted(x for x in evs if ID[t0] < x <= ID[min(t1, NT - 1)])
    if e:
        k = tpos(e[0]); k = min(max(k, t0 + 1), t1)
        S[t0:k, j] = s0; S[k:t1, j] = s1
    else:
        w = np.arange(t1 - t0) / max(t1 - t0, 1)
        S[t0:t1, j] = s0 * np.exp(w * math.log(r))
    SRC[t0:t1, j] = code_src
for c, a in anc.items():
    j = CI[c]
    pts = sorted((d, v[0] / par_at(c, d)) for d, v in a.items() if d >= pd.Timestamp('2012-12-31'))
    # MI_QFIIS 實值點（2025-10-07 起，每週取一點，避免過密）
    if c in VF.columns:
        vv = VF[c].dropna(); vv = vv[vv.index >= pd.Timestamp('2025-10-07')]
        pts = [x for x in pts if x[0] < pd.Timestamp('2025-10-07')] + [(d, float(s)) for d, s in vv.iloc[::5].items()] + ([(vv.index[-1], float(vv.iloc[-1]))] if len(vv) else [])
    # o7 回推後段：只給「最後報表錨點 ≤ 2021-06-30」的股票（hawan 不需要）
    last_stmt = max(d for d, v in a.items())
    if last_stmt <= pd.Timestamp('2021-06-30') and c in SE.columns:
        se = SE[c].dropna(); se = se[(se.index >= SE_OK_FROM) & (se.index < pd.Timestamp('2025-10-07'))]
        if len(se):
            # 取 o7 路徑的每個變動點與月初點
            chg = se[se.diff().fillna(1) != 0]
            mo = se.groupby(se.index.to_period('M')).head(1)
            sel = pd.concat([chg, mo]).sort_index(); sel = sel[~sel.index.duplicated()]
            pts += [(d, float(v)) for d, v in sel.items()]
            pts = sorted(pts)
    if not pts: continue
    evs = ev_dates.get(c, set())
    # 報表記錄的股本變動早於交易面（減資換發、配股基準日在季末後）：若 (d0, d1] 沒有事件、但 (d1, d1+110 天] 且早於下一錨點有事件，
    # 把 s1 延到該事件日才生效（價格在同一天調整，市值才一致）。
    adj_pts = []; k = 0
    while k < len(pts):
        d1, s1 = pts[k]
        if k > 0 and abs(math.log(s1 / adj_pts[-1][1])) > 0.005:
            d0 = adj_pts[-1][0]
            if not any(d0 < x <= d1 for x in evs):
                dn = pts[k + 1][0] if k + 1 < len(pts) else d1 + pd.Timedelta(days=110)
                later = sorted(x for x in evs if d1 < x <= min(d1 + pd.Timedelta(days=110), dn))
                if later:
                    adj_pts.append((d1, adj_pts[-1][1])); adj_pts.append((later[0], s1)); k += 1; continue
        adj_pts.append((d1, s1)); k += 1
    pts = adj_pts
    tl = [(tpos(d) if d >= ID[0] else 0, s, d) for d, s in pts]
    # 季末錨點 → 季末當日或之後第一個交易日
    for (ta, sa, da), (tb, sb, db) in zip(tl[:-1], tl[1:]):
        srcc = 3 if (da >= SE_OK_FROM and last_stmt <= pd.Timestamp('2021-06-30') and da < pd.Timestamp('2025-10-07')) else (4 if da >= pd.Timestamp('2025-10-07') else 1)
        fill_between(j, ta, sa, tb, sb, evs, srcc)
    tb, sb, db = tl[-1]
    # 最後錨點之後：下市股到最後成交日、其他到面板終點；之後的除權事件用價格比
    endt = NT
    S[tb:endt, j] = sb; SRC[tb:endt, j] = 4 if db >= pd.Timestamp('2025-10-07') else 1
    after = JC[(JC.code == c) & (JC.date > db)].sort_values('date')
    for r in after.itertuples():
        k = tpos(r.date); g = stock_ratio_from_px(r.kind, r.prev, r.ref, r.date.year)
        S[k:endt, j] *= g; SRC[k:endt, j] = 2
    # 第一錨點之前：用 TWT49U 價格比往回推（2014 起有資料；2013 無事件資料 → 維持常數並標記）
    ta, sa, da = tl[0]
    if ta > 0:
        S[:ta, j] = sa; SRC[:ta, j] = 2
        before = JC[(JC.code == c) & (JC.date <= da)].sort_values('date', ascending=False)
        for r in before.itertuples():
            k = tpos(r.date); g = stock_ratio_from_px(r.kind, r.prev, r.ref, r.date.year)
            S[:k, j] /= g
        OUT.setdefault('first_anchor_after_2013q1', {})[c] = str(da.date())

# ---------------- 沒有報表錨點的股票：o7 的 0050 報酬校準股數（2311/2325/3474 已有報表錨點時不用）----------------
fill = json.loads((AG_CM := (L.AG / 'o_capmatched_selection' / 'data' / 'tw50_fill.json')).read_text(encoding='utf-8'))
OUT['tw50_fill_used'] = {}
for c, v in fill.items():
    j = CI.get(c)
    if j is None or c in anc: continue
    if v['mode'] != 'calibrated': continue
    w0, w1 = pd.Timestamp(v['window'][0]), pd.Timestamp(v['window'][1])
    m = (ID >= w0 - pd.Timedelta(days=400)) & (ID <= w1)
    S[m, j] = v['shares']; SRC[m, j] = 5; OUT['tw50_fill_used'][c] = v['window']

# 只保留有收盤（或之前有收盤）的日子
live = np.zeros((NT, N), bool)
tr = L.TRADED[IDX >= T0]
first = np.where(tr.any(axis=0), np.argmax(tr, axis=0), NT)
lastt = np.where(tr.any(axis=0), NT - 1 - np.argmax(tr[::-1], axis=0), -1)
for j in range(N): live[first[j]:lastt[j] + 1, j] = True
S[~live] = np.nan; SRC[~live] = 0
Sdf = pd.DataFrame(S, index=ID, columns=COLS)
Sdf.to_parquet(DATA / 'shares_pit.parquet'); pd.DataFrame(SRC, index=ID, columns=COLS).to_parquet(DATA / 'shares_src.parquet')
(DATA / 's1_build.json').write_text(L.dumps(OUT), encoding='utf-8')
print(L.dumps({k: v for k, v in OUT.items() if k not in ('first_anchor_after_2013q1',)})[:3000])
print('first anchors after 2013Q1:', len(OUT.get('first_anchor_after_2013q1', {})))

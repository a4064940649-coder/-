#!/usr/bin/env python3
"""資料合併規則的迭代（只看資料品質與覆蓋率，不看任何報酬）與 R_FIX 產出。

輸入：R_DEL（o_rev_survivorship，sha 3e4b72e4…，正式底稿，既有格子一律不覆寫）、tw_revenue（sii 水準）、taiwanan 月營收水準（src_taiwanan.csv）、
      新來源 N（extract_news.py 抽出的鉅亨新聞營收提及）、rev_5y（2021-03 起的訊號）。
只補「國內、非金融、終止上市名單內（訊號日 ≤ 終止上市日）」股票的缺格；KY／DR、F- 股、金融業一律不補。

版本（iterations.json）：
  M1：N 的同股同月提及 ≥ 2 則、彼此差 ≤ 0.5%（共識），有效精度 ≤ 0.25%、文章落後 ≤ 1 個月；年增率＝水準 ÷ 12 個月前水準（與面板同法）
  M2：M1 ∪「單則但內部一致」：同句的月增率或年增率，與已知的上月／去年同月水準（面板或已接受的 N）算出來的差 ≤ 1pp（迭代到收斂）
  M3：M2 ＋ 沒有 12 個月前水準時，用同一則已接受新聞自己報的年增率（年增率格）
  M4：M3，但 2013 年各月一律用新聞年增率（IFRS 合併對合併；面板存活股 2013 年是合併對個體）——IFRS 斷點的另一種處理
  M5：M2，有效精度放寬到 ≤ 0.5%
  M6：全部單則（精度 ≤ 0.25%，不要求共識或內部一致）
選版規則（事前寫定）：重疊月（N 與面板都有值）水準一致率（差 < 0.5%）≥ 99%；用到新聞年增率的版本，年增率一致率（差 < 0.5pp，2014 年起）也要 ≥ 99%。
  通過者中，取「池內缺口股票月（終止上市股、訊號日 ≤ 終止日）新增可算 y3 的個數」最多者；同分取新增格子較少者。
更正公告：N 是公司第一次公告（自結）的數字；面板（MOPS 現行頁）是最新更正值。兩者並存時一律保留面板。
"""
import sys, json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v4lib as L
K, V = L.K, L.V

L.check_v3_inputs()
YRD = L.load_rdel()
Y5 = L.y5_orig()
MEN = pd.read_parquet(HERE / 'raw_news_mentions.parquet')

# ---------------- 水準面板 L0（面板＋taiwanan）----------------
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
d['ym'] = d['month'].str[:7]
d = d[d.market == 'sii']
T = pd.read_csv(L.V3_INPUTS['src_taiwanan'][0], dtype={'code': str})
r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str})
L0 = {}
for c, y, v in zip(r5.code, r5.ym, r5.revenue): L0[(c, y)] = float(v)
for c, y, v in zip(T.code, T.ym, T.rev): L0[(c, y)] = float(v)
for c, y, v in zip(d.code, d.ym, d.revenue_k_twd): L0[(c, y)] = float(v)          # 面板優先
def sh(ym, k): return str(pd.Period(ym, 'M') - k)

# ---------------- 可補的股票 ----------------
DLD = L.delist_codes()                                  # {代碼: 終止上市日}
def patchable(c):
    return c in DLD and c not in L.FOREIGN and not c.startswith('91') and not L.is_fin(c)
FIN_DELISTED = sorted(c for c in DLD if L.is_fin(c))
FOREIGN_DELISTED = sorted(c for c in DLD if c in L.FOREIGN)

# ---------------- N：同股同月彙整 ----------------
def agg_news(prec_max):
    Z = MEN[(MEN.prec <= prec_max) & (MEN.lag <= 1) & (MEN.ym >= '2013-01')]
    g = Z.groupby(['code', 'ym'])
    A = pd.DataFrame({'med': g.rev_k.median(), 'n': g.size(), 'spread': g.rev_k.agg(lambda v: v.max() / v.min() - 1),
                      'yoy': g.yoy_news.median(), 'mom': g.mom_news.median()}).reset_index()
    return A[A.spread <= 0.005].reset_index(drop=True)

def accept(A, rule):
    """回傳被接受的 (code, ym) → 水準。rule：'cons'（n ≥ 2）、'cons_or_int'（加內部一致，迭代）、'all'。內部一致檢查用 L0 ∪ 已接受。"""
    acc = {}
    if rule == 'all':
        return {(c, y): v for c, y, v in zip(A.code, A.ym, A.med)}
    for c, y, v, n in zip(A.code, A.ym, A.med, A.n):
        if n >= 2: acc[(c, y)] = v
    if rule == 'cons': return acc
    rows = list(zip(A.code, A.ym, A.med, A.yoy, A.mom))
    changed = True
    while changed:
        changed = False
        for c, y, v, yo, mo in rows:
            if (c, y) in acc: continue
            ok = False
            b12 = L0.get((c, sh(y, 12)), acc.get((c, sh(y, 12))))
            b1 = L0.get((c, sh(y, 1)), acc.get((c, sh(y, 1))))
            if b12 and np.isfinite(yo) and abs((v / b12 - 1) * 100 - yo) <= 1: ok = True
            if b1 and np.isfinite(mo) and abs((v / b1 - 1) * 100 - mo) <= 1: ok = True
            if ok: acc[(c, y)] = v; changed = True
    return acc

def gate(A, acc, use_news_yoy):
    """重疊月一致率（面板有值的月份，水準差 < 0.5%）；新聞年增率一致率（2014 起，對 R_DEL／Y5 的年增率，差 < 0.5pp）。"""
    lv = [(v, L0[k]) for k, v in acc.items() if k in L0 and k[1] >= '2013-01']
    rel = np.array([abs(a / b - 1) for a, b in lv if b > 0])
    out = {'n_overlap_level': int(len(rel)), 'level_agree_0.5pct': float((rel < 0.005).mean()) if len(rel) else np.nan}
    if use_news_yoy:
        yy = []
        Ay = A.set_index(['code', 'ym'])
        for k in acc:
            if k[1] < '2014-01' or k not in Ay.index: continue
            yn = Ay.loc[k, 'yoy']
            if not np.isfinite(yn): continue
            p = pd.Period(k[1], 'M')
            Yb = YRD if p < pd.Period('2021-01', 'M') else Y5
            if k[0] in Yb.columns and p in Yb.index and np.isfinite(Yb.at[p, k[0]]): yy.append(abs(yn - Yb.at[p, k[0]]))
        yy = np.array(yy)
        out.update({'n_overlap_yoy': int(len(yy)), 'yoy_agree_0.5pp': float((yy < 0.5).mean()) if len(yy) else np.nan})
    return out

def build(acc, A, yoy_only, ifrs_news_2013):
    """在 R_DEL／Y5 的缺格補年增率（只補 patchable、訊號窗內：營收月 ≤ 終止上市月）。回傳 (YL_new, Y5_new, cells)。"""
    YL = YRD.copy(); Y5n = Y5.copy()
    LV = dict(L0); LV.update({k: v for k, v in acc.items() if k not in L0})
    Ay = A.set_index(['code', 'ym'])
    cells = []
    for (c, y) in sorted(set(acc)):
        if not patchable(c): continue
        p = pd.Period(y, 'M')
        if p > pd.Period(DLD[c], 'M'): continue
        for tgt in (y, sh(y, -12)):          # 本月的年增率，以及 12 個月後那一月的年增率（基期）
            pt = pd.Period(tgt, 'M')
            if pt > pd.Period(DLD[c], 'M') or pt > pd.Period('2026-09', 'M'): continue
            tabs = [X for X in (YL, Y5n) if pt in X.index and not (c in X.columns and np.isfinite(X.at[pt, c]))]
            if not tabs: continue
            cur, base = LV.get((c, tgt)), LV.get((c, sh(tgt, 12)))
            val, how = np.nan, ''
            if ifrs_news_2013 and tgt[:4] == '2013' and (c, tgt) in acc and (c, tgt) in Ay.index and np.isfinite(Ay.loc[(c, tgt), 'yoy']):
                val, how = float(Ay.loc[(c, tgt), 'yoy']), 'news_yoy_ifrs2013'
            elif cur and base and base > 0:
                val, how = (cur / base - 1) * 100, 'level'
            elif yoy_only and (c, tgt) in acc and (c, tgt) in Ay.index and np.isfinite(Ay.loc[(c, tgt), 'yoy']):
                val, how = float(Ay.loc[(c, tgt), 'yoy']), 'news_yoy'
            if not np.isfinite(val): continue
            for Yb in tabs:
                if c not in Yb.columns: Yb[c] = np.nan
                Yb.at[pt, c] = val
            cells.append((c, tgt, val, how))
    return YL, Y5n, pd.DataFrame(cells, columns=['code', 'ym', 'yoy', 'how']).drop_duplicates(['code', 'ym'])

# 池內缺口股票月（explore_gap.py；終止上市股、訊號日 ≤ 終止日）
G = pd.read_csv(HERE / 'gap_pool_stockmonths.csv', dtype={'code': str})
G = G[G.delisted].copy()
def y3_ok(YL, Y5n, c, sig):
    m = pd.Period(sig, 'M'); Y = Y5n if pd.Timestamp(sig) >= V.SWITCH else YL
    ms = [m - 1, m - 2, m - 3]
    if c not in Y.columns or any(x not in Y.index for x in ms): return False
    return bool(np.isfinite(Y.loc[ms, c].to_numpy(dtype=float)).all())

VERS = {'M1': dict(prec=0.0025, rule='cons', yoy_only=False, ifrs=False),
        'M2': dict(prec=0.0025, rule='cons_or_int', yoy_only=False, ifrs=False),
        'M3': dict(prec=0.0025, rule='cons_or_int', yoy_only=True, ifrs=False),
        'M4': dict(prec=0.0025, rule='cons_or_int', yoy_only=True, ifrs=True),
        'M5': dict(prec=0.005, rule='cons_or_int', yoy_only=False, ifrs=False),
        'M6': dict(prec=0.0025, rule='all', yoy_only=False, ifrs=False)}
its = []; built = {}
for k, cfg in VERS.items():
    A = agg_news(cfg['prec'])
    acc = accept(A, cfg['rule'])
    gt = gate(A, acc, cfg['yoy_only'] or cfg['ifrs'])
    YL, Y5n, cells = build(acc, A, cfg['yoy_only'], cfg['ifrs'])
    G['ok_' + k] = [y3_ok(YL, Y5n, c, s) for c, s in zip(G.code, G.sig)]
    passed = gt['level_agree_0.5pct'] >= 0.99 and (not (cfg['yoy_only'] or cfg['ifrs']) or gt.get('yoy_agree_0.5pp', 0) >= 0.99)
    it = dict(version=k, cfg=cfg, gate=gt, passed=bool(passed), n_news_months_accepted=len(acc),
              n_cells_added=int(len(cells)), cells_by_how=cells.how.value_counts().to_dict() if len(cells) else {},
              n_codes_patched=int(cells.code.nunique()) if len(cells) else 0,
              pool_gap_stockmonths=int(len(G)), pool_gap_newly_covered=int(G['ok_' + k].sum()),
              newly_covered_by_year=G[G['ok_' + k]].groupby(G.sig.str[:4]).size().to_dict())
    its.append(it); built[k] = (YL, Y5n, cells)
    print(k, json.dumps({x: it[x] for x in ('gate', 'passed', 'n_cells_added', 'n_codes_patched', 'pool_gap_newly_covered')}, ensure_ascii=False, default=float), flush=True)
ok = [i for i in its if i['passed']]
best = max(ok, key=lambda i: (i['pool_gap_newly_covered'], -i['n_cells_added']))
sel = best['version']
YL, Y5n, cells = built[sel]
f = HERE / 'yoy_long_R_FIX.parquet'
Yw = YL.copy(); Yw.index = Yw.index.astype(str); Yw.to_parquet(f)
f5 = HERE / 'yoy_rev5y_FIX.parquet'
Y5w = Y5n.copy(); Y5w.index = Y5w.index.astype(str); Y5w.to_parquet(f5)
cells.to_csv(HERE / 'rfix_cells.csv', index=False)
G.to_csv(HERE / 'gap_pool_stockmonths_cov.csv', index=False)
# 與 R_DEL 差異斷言：只有新增，沒有覆寫
a, b = YRD.reindex(columns=YL.columns).to_numpy(dtype=float), YL.reindex(index=YRD.index).to_numpy(dtype=float)
assert np.all(np.isnan(a) | (a == b)), 'R_FIX 覆寫了 R_DEL 的既有格子'
y5a = Y5.reindex(columns=Y5n.columns).to_numpy(dtype=float); y5b = Y5n.to_numpy(dtype=float)
assert np.all(np.isnan(y5a) | (y5a == y5b))
out = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
       'note': '只用資料品質（重疊月一致率）與覆蓋率選版，不看任何報酬。選版規則在跑任何版本之前寫在本檔 docstring。',
       'rule': '重疊月水準一致率（差 < 0.5%）≥ 99%；用新聞年增率者，年增率一致率（差 < 0.5pp）≥ 99%；通過者取池內缺口新增覆蓋最多者，同分取新增格子較少者',
       'versions': its, 'selected': sel, 'fin_delisted_not_patched': FIN_DELISTED, 'foreign_delisted_not_patched': FOREIGN_DELISTED,
       'R_FIX': {'file': f.name, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()},
       'Y5_FIX': {'file': f5.name, 'sha256': hashlib.sha256(f5.read_bytes()).hexdigest(), 'cells_added': int((cells.ym >= '2021-01').sum())}}
(HERE / 'iterations_data.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print('selected', sel, out['R_FIX'])

#!/usr/bin/env python3
"""第 0 步：資料關卡（不看任何策略報酬）。
1) 交叉核對：來源 A（taiwanan）與既有 tw_revenue.csv 在存活公司重疊月份逐筆比對（營收水準、年增率）。
2) 資料建構版本（iterations）：三種「補進的年增率」算法，只用重疊期一致率挑選（事前規則：一致率最高者；同分取 V1）。
3) 產出修補後的長期年增率表（只補原本 NaN 的格子，不覆寫既有值）：
     R_DEL：只補非 KY／非 DR（國內公司；＝存活者偏誤修補，不改變股池定義）
     R_ALL：另補 KY／DR（外國公司；現行工具與回測都因資料缺漏而完全排除它們，這是股池定義問題）
4) 覆蓋率＝已補的缺口池內股票月 ÷ 缺口池內股票月（2013-01 起的換股日；gap.py 的 NOLEVEL）。
5) 凍結：輸出 parquet 與 sha256、時間戳 → data_freeze.json（之後所有回測斷言 sha256 相同）。
"""
import sys, json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
K = V.K
sys.path.pop(0)

YL, Y5, Y5f = V.load_revenue()
T = pd.read_csv(HERE / 'src_taiwanan.csv', dtype={'code': str})
T['p'] = pd.PeriodIndex(T['ym'], freq='M')
T = T[T['code'].isin(K.COLS)]

# ---- 名稱與 KY 判定 ----
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
SLs = pd.read_csv(K.X.parent / 'stock_list.csv', dtype=str)
names = {}
for c, nm in list(zip(T['code'], T['name'])) + list(zip(d['code'], d['name'])) + list(zip(SLs['code'], SLs['name'])):
    if isinstance(nm, str): names.setdefault(c, set()).add(nm)
def is_foreign(c):
    return c.startswith('91') or any(('KY' in n.upper()) or ('-DR' in n.upper()) for n in names.get(c, ()))
FOREIGN = {c for c in K.COLS if is_foreign(c)}

# ---- 1) 交叉核對（重疊：既有 tw_revenue 有值的 code×月） ----
LV = d[d['market'] == 'sii'].pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='first')
ov = T[T.apply(lambda r: r['code'] in LV.columns and r['p'] in LV.index and np.isfinite(LV.at[r['p'], r['code']]), axis=1)].copy()
ov['lv_old'] = [LV.at[p, c] for p, c in zip(ov['p'], ov['code'])]
rel = (ov['rev'] / ov['lv_old'] - 1).abs()
lev_agree = float((rel <= 0.005).mean())
lev_exact = float((rel <= 1e-9).mean())

# ---- 2) 三種年增率算法 ----
def yoy_v1(T):   # 當月 ÷ 去年同月（MOPS 當期公告的去年同月，已是同口徑）
    y = (T['rev'] / T['rev_last_year'] - 1) * 100
    return y.where(T['rev_last_year'] > 0)
def yoy_v2(T):   # 檔案內 YOY(%) 欄
    return T['yoy_pct']
def yoy_v3(T):   # 水準自算：當月 ÷ 12 個月前「當月」（只用來源 A 自己的水準）
    s = T.set_index(['code', 'p'])['rev']
    prev = [s.get((c, p - 12), np.nan) for c, p in zip(T['code'], T['p'])]
    prev = np.array(prev, float)
    return pd.Series(np.where(prev > 0, (T['rev'].to_numpy() / prev - 1) * 100, np.nan), index=T.index)
def yoy_v4(T):   # V3，來源內沒有 12 個月前水準時退回 V1（2015 年各月）
    a = yoy_v3(T); return a.where(a.notna(), yoy_v1(T))
VERS = {'V1_rev_over_lastyear': yoy_v1, 'V2_yoy_column': yoy_v2, 'V3_level_shift12': yoy_v3, 'V4_V3_fallback_V1': yoy_v4}
iters = []
ovy = T[T.apply(lambda r: r['code'] in YL.columns and r['p'] in YL.index and np.isfinite(YL.at[r['p'], r['code']]), axis=1)].copy()
ovy['y_old'] = [YL.at[p, c] for p, c in zip(ovy['p'], ovy['code'])]
for k, f in VERS.items():
    yn = f(T).reindex(ovy.index)
    diff = (yn - ovy['y_old']).abs()
    ok = diff <= 0.5          # 年增率差 ≤ 0.5 個百分點視為一致
    iters.append({'version': k, 'n_overlap': int(yn.notna().sum()), 'agree_rate_0.5pp': float(ok[yn.notna()].mean()),
                  'agree_rate_2pp': float((diff <= 2)[yn.notna()].mean()), 'coverage_of_overlap': float(yn.notna().mean()),
                  'median_abs_diff_pp': float(diff.median())})
# 規則（第 2 版，在任何回測之前修訂；第 1 版「只看一致率」會選到覆蓋 2015 年為 0 的 V3）：
#   先過閘門（重疊期一致率 ≥ 98%），再取「重疊期覆蓋率」最高者；同分取一致率高者。
passing = [r for r in iters if r['agree_rate_0.5pp'] >= 0.98]
best = max(passing, key=lambda r: (round(r['coverage_of_overlap'], 6), r['agree_rate_0.5pp']))
T['yoy_new'] = VERS[best['version']](T)

# 不一致的例子（給 FINDINGS）
yb = T['yoy_new'].reindex(ovy.index); bad = ovy[(yb - ovy['y_old']).abs() > 0.5].assign(y_new=yb)
bad_ex = bad[['code', 'ym', 'rev', 'rev_last_year', 'y_old', 'y_new']].head(15).to_dict(orient='records')
bad_by_year = bad['ym'].str[:4].value_counts().sort_index().to_dict()

# ---- 3) 修補 ----
add = T.pivot_table(index='p', columns='code', values='yoy_new', aggfunc='last')
def repaired(foreign_ok):
    Y = YL.copy()
    cols = [c for c in add.columns if (foreign_ok or c not in FOREIGN)]
    Y = Y.reindex(columns=Y.columns.union(cols))
    A = add[cols].reindex(index=Y.index)
    m = Y[cols].isna() & A.notna()
    Y[cols] = Y[cols].where(~m, A)
    return Y, int(m.to_numpy().sum())
R_DEL, n_del = repaired(False)
R_ALL, n_all = repaired(True)

# ---- 4) 覆蓋率 ----
G = pd.read_csv(HERE / 'gap_stockmonths.csv', dtype={'code': str})
G = G[(G['type'] == 'NOLEVEL') & (G['sig'] >= '2013-01-01')].copy()
G['foreign'] = G['code'].isin(FOREIGN)
def y3_ok(Y, code, sig):
    m = pd.Period(sig, 'M'); ms = [m - 1, m - 2, m - 3]
    if code not in Y.columns or any(x not in Y.index for x in ms): return False
    return bool(np.isfinite(Y.loc[ms, code].to_numpy()).all())
G['era'] = np.where(G['sig'] >= str(V.SWITCH.date()), 'rev5y_2021+', 'long_2013_2021')
G['fill_del'] = [(not f) and e == 'long_2013_2021' and y3_ok(R_DEL, c, s) for c, s, f, e in zip(G['code'], G['sig'], G['foreign'], G['era'])]
G['fill_all'] = [e == 'long_2013_2021' and y3_ok(R_ALL, c, s) for c, s, e in zip(G['code'], G['sig'], G['era'])]
G.to_csv(HERE / 'gap_stockmonths_filled.csv', index=False)
def cov(mask, col):
    g = G[mask]; return {'gap_stockmonths': int(len(g)), 'filled': int(g[col].sum()), 'coverage': float(g[col].mean()) if len(g) else np.nan,
                         'codes': int(g['code'].nunique())}
COV = {
    'domestic_all_2013_2026': cov(~G['foreign'], 'fill_del'),
    'domestic_2013_2021_long_era': cov(~G['foreign'] & (G['era'] == 'long_2013_2021'), 'fill_del'),
    'domestic_2015_04_2019_05': cov(~G['foreign'] & (G['sig'] >= '2015-04-01') & (G['sig'] < '2019-06-01'), 'fill_del'),
    'domestic_2013_01_2015_03': cov(~G['foreign'] & (G['sig'] < '2015-04-01'), 'fill_del'),
    'domestic_2021_plus': cov(~G['foreign'] & (G['era'] == 'rev5y_2021+'), 'fill_del'),
    'foreign_all_2013_2026': cov(G['foreign'], 'fill_all'),
    'foreign_long_era': cov(G['foreign'] & (G['era'] == 'long_2013_2021'), 'fill_all'),
    'total_all_2013_2026': cov(G['code'].notna(), 'fill_all'),
}
G['y'] = G['sig'].str[:4]
COV['by_year'] = {y: {'dom_gap': int((~g['foreign']).sum()), 'dom_filled': int(g.loc[~g['foreign'], 'fill_del'].sum()),
                      'for_gap': int(g['foreign'].sum()), 'for_filled': int(g.loc[g['foreign'], 'fill_all'].sum())} for y, g in G.groupby('y')}
unfilled_dom = G[~G['foreign'] & ~G['fill_del']].groupby('code').size().sort_values(ascending=False)
COV['unfilled_domestic_top'] = [(c, sorted(names.get(c, {''}))[0] if names.get(c) else '', int(n)) for c, n in unfilled_dom.head(30).items()]

# ---- 5) 凍結 ----
out = {}
for nm, Y in (('R_DEL', R_DEL), ('R_ALL', R_ALL)):
    f = HERE / f'yoy_long_{nm}.parquet'
    Yw = Y.copy(); Yw.index = Yw.index.astype(str)
    Yw.to_parquet(f)
    out[nm] = {'file': f.name, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'cells_added': n_del if nm == 'R_DEL' else n_all}
src_sha = hashlib.sha256((HERE / 'src_taiwanan.csv').read_bytes()).hexdigest()
freeze = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
          'note': '資料在任何回測之前凍結；之後所有回測斷言下列 sha256。',
          'source_priority': ['A: taiwanan/TW_Stock_Exchange_Crawler monthly_raw_data（git 全歷史、每檔最後存在版）', '其餘來源（MOPS、FinMind API、archive.org、鉅亨／MoneyDJ）在本環境連不到；GitHub 搜尋未找到 2013～2014 或 2019-09 之後含下市公司的月檔'],
          'src_taiwanan_sha256': src_sha, 'outputs': out,
          'level_agree_0.5pct': lev_agree, 'level_exact': lev_exact, 'n_level_overlap': int(len(ov)),
          'yoy_iterations': iters, 'chosen': best['version'], 'yoy_bad_by_year': bad_by_year, 'yoy_bad_examples': bad_ex,
          'n_foreign_codes_in_panel': len(FOREIGN), 'foreign_codes': sorted(FOREIGN), 'coverage': COV,
          'gate': {'coverage_rule': '覆蓋率 ≥ 80%', 'consistency_rule': '重疊期一致率 ≥ 98%'}}
(HERE / 'data_freeze.json').write_text(json.dumps(freeze, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
(HERE / 'iterations.json').write_text(json.dumps({'note': '本題不改策略參數，迭代只在「補進資料的年增率算法」，且只用重疊期一致率（不看任何報酬）挑選。共 3 版。',
                                                 'versions': iters, 'chosen': best['version'], 'rule_v1_superseded': '一致率（≤0.5pp）最高者（會選 V3，但 V3 對 2015 年完全沒有值）', 'rule': '先過閘門（一致率 ≥ 98%），再取重疊期覆蓋率最高者；同分取一致率高者。規則在任何回測之前修訂，只看資料品質。'}, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in freeze.items() if k not in ('foreign_codes', 'yoy_bad_examples')}, ensure_ascii=False, indent=1, default=str))

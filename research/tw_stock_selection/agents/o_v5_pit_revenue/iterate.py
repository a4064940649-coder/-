#!/usr/bin/env python3
"""資料合併規則迭代（只看資料品質指標，不看報酬）。首次值判定規則 F1～F6、缺口代理規則 G1～G2。輸出 iterations.json。
選版規則（寫在跑之前）：
  首次值：(i) 不可有 |首次/最新−1| > 50% 的替換（單位或抽取錯誤）；(ii) 替換格中有獨立佐證（≥2 則共識，或同句年增／月增率與面板去年同月／上月水準一致）的比例 ≥ 95%；
          通過者取替換格最多者；同分取規則較嚴者（編號較大者）。
  缺口：G1（嚴格 PIT：可得時點 ≤ 訊號日的新聞、封存快照、已公告季報）與 G2（放寬：任何事後抓取的官方月營收值，視為法定期限公告）各算覆蓋率；
        閘門＝全部缺口格與 52 週比百分位 ≥ 0.7 的高風險格覆蓋率都 ≥ 60%（以 G1 判定，G2 只報告）。"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd
A = pd.read_csv(HERE / 'data' / 'fv_cells.csv', dtype={'code': str}, parse_dates=['first_date'])
T30 = pd.read_csv(HERE / 'data' / 'top30_cells.csv', dtype={'code': str})
t30 = set(zip(T30[T30.role == 'cur'].code, T30[T30.role == 'cur'].ym))
def deadline_ok(fd, ym):
    p = pd.Period(ym, 'M') + 1
    return fd <= pd.Timestamp(p.year, p.month, 15)
A['in_window'] = [deadline_ok(f, y) for f, y in zip(A.first_date, A.ym)]
A['confirmed'] = A.consensus | A.internal_ok
def version(tol=0.005, maxrel=None, confirm=False, no2013=False, window=False):
    m = A.rel_first.abs() >= tol
    if maxrel is not None: m &= A.rel_first.abs() <= maxrel
    if confirm: m &= A.confirmed
    if no2013: m &= A.ym.str[:4] != '2013'
    if window: m &= A.in_window
    return m
VERS = {
 'F1': dict(desc='單則即可：精度 ≤ 0.25%、落後 ≤ 1 個月、最早一則與面板差 ≥ 0.5% 就替換', kw=dict()),
 'F2': dict(desc='F1＋|差| ≤ 50%（排除單位／抽取錯誤）', kw=dict(maxrel=0.5)),
 'F3': dict(desc='F2＋獨立佐證（≥2 則共識或同句年增／月增率內部一致）', kw=dict(maxrel=0.5, confirm=True)),
 'F4': dict(desc='F3＋排除 2013 年（IFRS 斷點：合併／個體口徑可能混用）', kw=dict(maxrel=0.5, confirm=True, no2013=True)),
 'F5': dict(desc='F3＋最早一則須在營收月次月 15 日以前（確保是初次公告的版本）', kw=dict(maxrel=0.5, confirm=True, window=True)),
 'F6': dict(desc='F3，但差 ≥ 1.0% 才替換', kw=dict(tol=0.01, maxrel=0.5, confirm=True)),
}
it = {'rule': __doc__, 'first_value_versions': {}, 'gap_versions': {}}
for k, v in VERS.items():
    m = version(**v['kw']); x = A[m]
    it['first_value_versions'][k] = {
        'desc': v['desc'], 'n_replaced': int(m.sum()),
        'n_implausible_gt50pct': int((x.rel_first.abs() > 0.5).sum()),
        'confirmed_share': float(x.confirmed.mean()) if len(x) else None,
        'n_in_top30_current_cells': int(sum((c, y) in t30 for c, y in zip(x.code, x.ym))),
        'n_2013': int((x.ym.str[:4] == '2013').sum()), 'n_after_window': int((~x.in_window).sum()),
        'dir_first_lt_latest': int((x.rel_first < 0).sum()), 'dir_first_gt_latest': int((x.rel_first > 0).sum()),
        'median_abs_rel': float(x.rel_first.abs().median()) if len(x) else None}
ok = {k: v for k, v in it['first_value_versions'].items() if v['n_implausible_gt50pct'] == 0 and (v['confirmed_share'] or 0) >= 0.95}
sel = sorted(ok, key=lambda k: (-ok[k]['n_replaced'], -int(k[1])))[0]
it['first_value_selected'] = sel
G = pd.read_csv(HERE / 'data' / 'gap_cells_2006_2012.csv', dtype={'code': str})
src = json.load(open(HERE / 'sources_log.json'))
for k, desc in (('G1', '嚴格 PIT：鉅亨新聞（2013 起，不涵蓋 2006～2012）、web.archive.org 快照（連不到）、季報（2006～2012 無可得來源）'),
                ('G2', '放寬：任何事後抓取的官方月營收／季營收值（GitHub 搜尋無 2006～2012 下市股資料）')):
    cov = np.zeros(len(G), bool)      # 兩版都沒有任何來源格
    it['gap_versions'][k] = {'desc': desc, 'n_cells': int(len(G)), 'n_covered': int(cov.sum()), 'coverage_all': float(cov.mean()),
                             'n_risk': int(G.risk.sum()), 'coverage_risk': float(cov[G.risk.to_numpy()].mean())}
g1 = it['gap_versions']['G1']
it['gap_gate'] = {'threshold': 0.6, 'coverage_all': g1['coverage_all'], 'coverage_risk': g1['coverage_risk'],
                  'pass': bool(g1['coverage_all'] >= 0.6 and g1['coverage_risk'] >= 0.6),
                  'consequence_if_fail': '缺口部分以 AB＋B10（V5 資料下為 AB_V5_B10）為正式保守口徑結案'}
E = json.load(open(HERE / 'explore_fv.json'))
it['first_value_gate'] = {'threshold': 0.8, 'top30_stockmonths': E['top30_stockmonths'], 'coverage_all6_levels': E['top30_cov_all6'],
                          'coverage_cur3_levels': E['top30_cov_cur3'], 'coverage_all6_2013_2021': E['top30_cov_all6_2013_2021'],
                          'pass': bool(E['top30_cov_all6'] >= 0.8),
                          'consequence_if_fail': '有首次值的格照替換；其餘格以「首次值取最壞／最好」給界限（FV 擾動 Monte Carlo＋固定方向版）'}
it['n_versions_total'] = len(VERS) + 2
json.dump(it, open(HERE / 'iterations.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in it.items() if k != 'rule'}, ensure_ascii=False, indent=1))

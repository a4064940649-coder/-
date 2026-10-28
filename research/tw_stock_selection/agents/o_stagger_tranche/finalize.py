#!/usr/bin/env python3
"""收尾：事後（post hoc，標註）描述統計＋判定＋檔案 sha256，寫入 results.json。不新增任何策略版本。"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import st_core as C
res = json.loads((C.HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in res
oos = json.loads((C.HERE / 'oos.json').read_text(encoding='utf-8'))
it = json.loads((C.HERE / 'iterations.json').read_text(encoding='utf-8'))
ex = json.loads((C.HERE / 'explore.json').read_text(encoding='utf-8'))
NAV = pd.read_parquet(C.HERE / 'navs_oos.parquet')
JS = list(range(-8, 13))
# 事後：各 J 的自己窗月超額
EX = {}
for j in JS:
    days, end = C.version_days(j)
    W = C.windows_from(days, end)
    nv = NAV[f'J{j:+d}'].dropna()
    EX[j] = C.wret(nv, W) - C.wret(C.E50.reindex(nv.index), W)
M = pd.DataFrame(EX)
def contrast(a, b, key):
    dd = C.sel(M[a].mean(axis=1) - M[b].mean(axis=1), key).dropna()
    return dict(diff_pp=float(dd.mean() * 100), t=C.tstat(dd), n=int(len(dd)))
post = {}
for key in ('SEL', 'POST', 'S3', 'S4', 'ALL'):
    post[key] = {'J0..3_minus_J9..12': contrast([0, 1, 2, 3], [9, 10, 11, 12], key), 'J0..3_minus_J-8..-1': contrast([0, 1, 2, 3], list(range(-8, 0)), key),
                 'J1..5_minus_J6..12': contrast([1, 2, 3, 4, 5], list(range(6, 13)), key)}
def tdist(js, key='ALL'):
    x = np.array([oos['scan'][f'J{j:+d}']['stats'][key]['t_0050'] for j in js])
    return dict(js=[min(js), max(js)], min=float(x.min()), p10=float(np.percentile(x, 10)), median=float(np.median(x)), max=float(x.max()))
sub = {'fresh_J0..4': {k: tdist(range(0, 5), k) for k in ('ALL', 'SEL', 'POST')}, 'post_deadline_J1..12': {k: tdist(range(1, 13), k) for k in ('ALL', 'SEL', 'POST')},
       'pre_deadline_J-8..-1': {k: tdist(range(-8, 0), k) for k in ('ALL', 'SEL', 'POST')}}
# 為什麼分批的標準差降不多：共同窗上 J+1 與其他批的月超額相關
days_off, end_off = C.official_list(); WO = C.windows_from(days_off, end_off)
e50 = C.wret(C.E50.reindex(NAV.index), WO)
cor = {}
for j in (6, 11, 16 if 'J+16' in NAV else 12):
    if f'J{j:+d}' not in NAV: continue
    a = C.sel(C.wret(NAV['J+1'].dropna(), WO) - e50, 'SEL'); b = C.sel(C.wret(NAV[f'J{j:+d}'].dropna(), WO) - e50, 'SEL')
    cor[f'J+1_vs_J{j:+d}_SEL'] = float(pd.concat([a, b], axis=1).dropna().corr().iloc[0, 1])
# 周轉修正：oos/explore 的 turnover 依「成交日的日曆月」加總，j＝−8、−7 時同一日曆月常有兩次換股而被高估。
# 這裡改用「每次換股（每批）的賣出額 ÷ 該批總值」平均（不含各批第一次），與單日版的月單邊周轉同義。
import iterate as IT
st = C.setup()
def ev_tov(lg):
    tv = lg['tov']; first = tv.groupby('i').head(1).index
    return float(tv.drop(first)['tov'].mean())
TOV = {}
for j in JS:
    days, end = C.version_days(j)
    pl = C.plan_single(days)
    t0 = C.IDX.get_loc(days[0][1]); t1 = C.IDX.get_loc(end)
    TOV[f'J{j:+d}'] = ev_tov(C.tsim([pl], *C.NORMAL, t0, t1, xrules=st['XR'])[2])
plo = C.plan_single(days_off, ab=False)
TOV['OFF'] = ev_tov(C.tsim([plo], *C.NORMAL, C.IDX.get_loc(days_off[0][1]), C.IDX.get_loc(end_off), xrules=st['XR'])[2])
for nm in ('V08', 'V12', 'V03'):
    TOV[nm] = ev_tov(IT.run_spec(IT.SPECS[nm], C.NORMAL, max_m=None, t_end=end_off)[2])
print('TOV', {k: round(x, 3) for k, x in TOV.items()})
v = oos['versions']; SELV = res['PREREG']['selected_version']; sd = oos['selected_detail']
B = sd['judgment_B']
DECISION = {
 'verdict': '不可用（SOP 不採用分批）；主策略級別加註「時點敏感」',
 'judgment_A': dict(flag=oos['judgment_A']['flag_timing_sensitive'], p10_t_ALL=oos['judgment_A']['p10_t_ALL'], median_t_ALL=oos['dist']['ALL']['t']['median'],
                    OFF_t=oos['dist']['ALL']['OFF_t'], OFF_pctile=oos['dist']['ALL']['OFF_pctile'], interval_CAGR=oos['judgment_A']['interval_CAGR_p10_p90'],
                    interval_t=oos['judgment_A']['interval_t_p10_p90']),
 'judgment_B': {k: B[k] for k in ('B1', 'B2', 'B3', 'B4', 'B5', 'all_pass')},
 'mechanism': {'M1_timing_luck': '支持：21 個單日版本全期 t 0.23～2.37，官方日 2.55 高於全部單日版（百分位 1.00）；同一天 AB 抽樣不同 seed 也讓 t 在 2.03～2.37 間變動。',
               'M2_freshness': '部分支持：挑選期 J+1～+5 對 J−8～−1 只高 0.12pp（t 0.32），未達 0.5pp；2019 後高 0.64pp（t 0.89），S4 2.45pp（t 1.91），且 2019 後 J0～J3 明顯優於 J9～J12。',
               'M3_capacity': '支持：1,000 萬帳戶超過 60 日均成交值 1% 的新買名額，挑選期由 38.6% 降到 1.9%（4 批），單筆占比中位數 0.74% → 0.18%（約 1/4）。'},
 'mechanism_overall': '部分支持',
}
SUMMARY = {
 'selected': SELV, 'spec': res['PREREG']['spec'],
 'selected_ALL': v[SELV]['normal']['stats']['ALL'], 'selected_SEL': v[SELV]['normal']['stats']['SEL'], 'selected_VAL': v[SELV]['normal']['stats']['VAL'], 'selected_HOLD': v[SELV]['normal']['stats']['HOLD'],
 'selected_paired_J1': v[SELV]['normal']['paired_J1'], 'selected_paired_OFF': v[SELV]['normal']['paired_OFF'],
 'OFF_ALL': oos['OFF']['stats']['ALL'], 'J1_ALL': oos['J1_common']['stats']['ALL'],
 'scan_dist_ALL': oos['dist']['ALL'], 'post_hoc_contrasts': post, 'subset_t_dist': sub, 'corr_common_windows': cor, 'turnover_per_event': TOV,
 'n_versions': dict(J_scan=21, tranche=len(it['versions']), main_history=10, total=res['PREREG']['multiple_testing']['m_total'], rounds=len(it['rounds'])),
}
res['DECISION'] = DECISION; res['SUMMARY'] = SUMMARY
files = ['st_core.py', 'validate.py', 'plan.py', 'explore.py', 'iterate.py', 'prereg.py', 'oos.py', 'finalize.py', 'validate.json', 'explore.json', 'iterations.json', 'oos.json', 'navs_oos.parquet', 'explore_rows.csv']
res['FILES_sha256'] = {f: hashlib.sha256((C.HERE / f).read_bytes()).hexdigest() for f in files}
res['FINAL_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
(C.HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps({'post': post, 'sub': sub, 'cor': cor}, ensure_ascii=False, indent=1, default=float))

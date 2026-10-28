#!/usr/bin/env python3
"""凍結：依 PLAN 的選版規則（iterations.json 的 selection）寫入 PREREG（含 sha256 與時間戳）。已存在就拒絕覆寫。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE / 'results.json'
res = json.loads(RES.read_text(encoding='utf-8'))
assert 'PLAN_sha256' in res and 'PREREG' not in res
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
sel = it['selection']['selected']
spec = {k: v for k, v in it['versions'][sel]['spec'].items() if k != 'family'}
def sha(p): return hashlib.sha256((HERE / p).read_bytes()).hexdigest()
PREREG = {
 'selected_version': sel,
 'selected_spec': spec,
 'selected_by': {k: it['selection'][k] for k in ('rule', 'candidates_in_band', 'chain', 'selected', 'n_versions')},
 'frozen_spec': {
   'strategy': '主策略 V3：main_v2/v2_core.V2(gapfix)，排除金融保險（stock_list 金融保險業＋28xx、5876、5880）；前 300 流動性池（60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10、隔日有開盤成交、非處置＝o_disposal_rejudge 官方名單 E1 全部排除）；'
               '分數＝0.5×52 週比百分位＋0.5×近 3 月營收年增率平均百分位（R_DEL 長期營收；2021-03 起 rev_5y）；每月第一個 ≥ 11 日的交易日收盤算訊號、隔日開盤成交；n10；同分先比營收再比代碼；開盤漲停（跳動點）且未持有者遞補；停牌凍結；P 下市價格、X 併購出清。',
   'buffer_rule': f'{sel}：' + {'A': '上月實際持股若本月仍合格、隔日可交易且排名 ≤ K 就續抱（rank 1 起算），其餘依排名補滿 10 檔；全部（含續抱股）調回等權。',
                                'B': '上月持股排名 ≤ K 且 52 週比 ≥ h 才續抱。', 'C': '上月持股分數 +δ 後重排。', 'D': '漂移帶。', 'E': '兩期重疊。', 'X': '組合。'}[it['versions'][sel]['spec']['family']] + f' 參數 {spec}',
   'costs': res['PLAN']['costs'],
   'engine': 'hb_core.sim（band=None 時＝o3_core.simulate 逐日相同）、hb_core.planner（K=None 時＝V2.plan）',
 },
 'oos_runs': {'validation': '訊號日 2019-01～2023-12（只報告）', 'holdout': '訊號日 2024-01～2026-10-12（只報告）', 'cross_cycle': '2006-04～2026-10-12',
              'compare': ['主策略 V3（本題重建＝o_disposal_rejudge v3_results R_DEL|E1，逐位相同）', '0050 含息', '同池等權（零成本）', 'main_v2 navs_gapfix EXFIN|n10|normal（v2 舊正式）'],
              'report_only': '其餘 19 個版本的 2019+ 配對差只報告，不參與判定、不改選版'},
 'final_thresholds': res['PLAN']['final_thresholds'],
 'explore_P2_result': '探索 P2a（排名 11～30 且 52 週比 ≥0.90 的舊股 − 新進前 10）＝−1.64pp／月、t −2.05（< −2）：依 PLAN，慢漂移機制在本策略不成立、緩衝題永久關閉；模型族照事前登記全部跑完並凍結，只作紀錄。',
 'code_sha256': {p: sha(p) for p in ['hb_core.py', 'validate.py', 'plan.py', 'explore.py', 'iterate.py']},
 'iterations_sha256': sha('iterations.json'), 'explore_sha256': sha('explore.json'),
}
txt = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
res['PREREG'] = PREREG
res['PREREG_sha256'] = hashlib.sha256(txt.encode('utf-8')).hexdigest()
res['PREREG_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(sel, spec, res['PREREG_sha256'], res['PREREG_time_utc'])

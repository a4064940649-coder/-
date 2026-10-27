#!/usr/bin/env python3
"""凍結：依 PLAN 的選版規則（iterations.json）寫入 PREREG（含 sha256 與時間）。已存在就拒絕覆寫。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE / 'results.json'
res = json.loads(RES.read_text(encoding='utf-8'))
assert 'PLAN_sha256' in res and 'PREREG' not in res
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
selA, selAP = it['A']['selection']['selected'], it['AP']['selection']['selected']
assert selA == selAP, (selA, selAP)
def sha(p): return hashlib.sha256((HERE / p).read_bytes()).hexdigest()
PREREG = {
 'selected_version': selA,
 'selected_by': it['A']['selection'],
 'frozen_spec': {
   'strategy': '主策略 v2（main_v2/v2_core.V2，gapfix 滾動窗、排除金融保險（含 28xx、5876、5880）、前 300 流動性池：60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10、隔日有開盤成交；分數＝0.5×52 週比百分位＋0.5×近 3 月營收年增率平均百分位；每月第一個 ≥ 11 日的交易日收盤算訊號、隔日開盤成交、等權 n10；開盤漲停（跳動點）未持有者遞補；同分先比營收再比代碼）',
   'disposal_rule': 'E5：訊號日 d 若某股處於處置期間（公告日 ≤ d 且 start ≤ d ≤ end），只有在「第二次以上處置（第二次處置／收足全部款券／內容含曾發布處置／督導會報）」或「剩餘處置天數 rem ≥ thr」時才排除；rem＝成交日 e 到 end 的交易日數，thr＝ceil(0.6×該次處置交易日數)（10 日制 6、5 日制與 2026-08-10 新制 3）。其餘處置股照常進入股池與排名。',
   'list': 'disposal_hist_rejudge.csv（TWSE 官方封存，sha256 見 list_sha256）∪ disposal_5y 獨有列；A+P 版本另補近似規則（只作穩健性）',
   'costs': '一般：買 0.0855%+滑價 0.1%、賣 0.0855%+稅 0.3%+滑價 0.1%；壓力：一般＋處置中交易每邊再加 0.5%；另報 v2 全面壓力 0.5%/0.5%',
   'engine': 'drj_core.simulate2（＝ia_core.simulate 的逐檔成本版；停牌凍結）',
 },
 'oos_runs': {'validation_seen': '訊號日 2021-09～2023-12（只報告）', 'holdout_seen': '2024-01～2026-10-12（只報告）', 'cross_cycle': '2006-04～2026-10-12',
              'compare': ['E1（現行 SOP）', 'E0', '0050 含息', '同池等權', 'main_v2 navs_gapfix EXFIN|n10|normal（官方主策略：2021-09 前無處置濾網）']},
 'judgement_from_PLAN': res['PLAN']['judgement'],
 'list_sha256': json.loads((HERE / 'data_freeze.json').read_text(encoding='utf-8'))['list_sha256'],
 'code_sha256': {p: sha(p) for p in ['build_list.py', 'drj_core.py', 'explore.py', 'iterate.py', 'plan.py']},
 'iterations_sha256': sha('iterations.json'),
}
txt = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
res['PREREG'] = PREREG
res['PREREG_sha256'] = hashlib.sha256(txt.encode('utf-8')).hexdigest()
res['PREREG_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(selA, res['PREREG_sha256'], res['PREREG_time_utc'])

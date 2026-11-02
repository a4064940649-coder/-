#!/usr/bin/env python3
"""事後附記（PREREG／RESULTS 之後，不改凍結判定）：run.py 的 R_V7X 條件 (ii) 把 gate_zero.json 內所有數值（含 runtime_s、off_n、j_n 等非差值欄位與 ideal_variants）
一律當成差值取最大，屬實作瑕疵。本檔只重算「零衝擊閘門主體（off 160 條、j 52 條）」與「ideal_variants（併購延遲等變體）」兩部分，寫進 POSTHOC_gate，不動 RESULTS。"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
G = json.loads((HERE.parent / 'o_exec_engine_rewrite_v7x/out/gate_zero.json').read_text())
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'RESULTS' in R and 'POSTHOC_gate' not in R
def mx(d):
    m = 0.0
    for v in d.values(): m = max(m, mx(v)) if isinstance(v, dict) else (max(m, abs(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) else m)
    return m
off, jj, iv = mx(G['off']), mx(G['j']), mx(G['ideal_variants'])
c = R['RESULTS']['R_V7X']['conditions']
others = all(v for k, v in c.items() if isinstance(v, bool) and k != 'V7X_zero_impact_gate_le_1e-9')
R['POSTHOC_gate'] = {
    'note': '事後附記，不改 RESULTS 的凍結判定（不取代）。',
    'frozen_condition_ii_max': c['V7X_zero_impact_gate_max'], 'frozen_max_came_from': 'summary.runtime_s（178.2 秒，非差值；實作瑕疵）',
    'gate_off_max': off, 'gate_j_max': jj, 'gate_summary_PASS': G['summary']['PASS'],
    'ideal_variants_max': iv, 'ideal_variants_detail': G['ideal_variants'],
    'ideal_variants_cause': 'B10 J12 併購延遲變體（mdelay／both）淨值差約 0.89%：2325 撥款日 2018-05-29 恰為換股日，V7X 用嚴格大於、原版用大於等於（V7X RECON_TRACE trace4，規格模糊）。',
    'condition_ii_gate_body_only_le_1e-9': bool(max(off, jj) <= 1e-9),
    'condition_ii_including_ideal_variants_le_1e-9': bool(max(off, jj, iv) <= 1e-9),
    'other_conditions_all_true': bool(others),
    'posthoc_reading': ('若條件 (ii) 只看閘門主體（off＋j），四個條件全成立 → 會是「取代」；若連 ideal_variants 一起看，仍因 0.89% 的規格模糊而「不取代」。'
                        '兩種讀法都不改凍結判定；建議 SOP 表把 V7X 列為「四項修正後」欄，與 V7 原表並列，量級取兩者較低值（K5T 8 格全是 V7X 較低）。')}
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in R['POSTHOC_gate'].items() if k != 'ideal_variants_detail'}, ensure_ascii=False, indent=1))

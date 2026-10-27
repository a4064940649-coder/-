# -*- coding: utf-8 -*-
"""閘門判定（不計算任何報酬）。(a) 不過 → (b) 不計算、主題關閉。"""
import os, sys, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import hashlib
HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'build_summary.json').read_text(encoding='utf-8'))
ga = S['gate_a']
sha = lambda p: hashlib.sha256((HERE / p).read_bytes()).hexdigest()
G = dict(
    time_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    gate_a=dict(rule=ga['rule'], count=ga['n_liquid_exfin'], count_incl_fin=ga['n_liquid'], threshold=200, passed=ga['passed'],
                by_year=ga['n_liquid_exfin_by_year'], by_outcome=ga['n_liquid_exfin_by_outcome'],
                upper_bound_if_every_FSC_case_had_a_board_date=ga['upper_bound']['total'],
                incl_ECB_not_counted=ga['n_liquid_exfin_incl_ECB']),
    gate_b=dict(computed=False, reason='(a) 未通過；依規定 (a) 過了才算 (b)，因此沒有計算任何事件報酬。'),
    decision='主題關閉' if not ga['passed'] else '進入 (b)',
    reason=('挑選期 2010～2018 可用董事會日錨點、60 日均成交值 ≥ 3,000 萬、非金融的上市公司國內 CB 事件只有 %d 筆（< 200）。'
            '即使每一件 FSC 申報都找得到董事會日（含 2010 年，實際沒有來源），上界也只有 %d 筆。'
            '海外 CB（ECB）不屬於本題機制（國內基準價、CBAS、競拍／詢圈），事前就不計入。' % (ga['n_liquid_exfin'], ga['upper_bound']['total'])),
    sha256={p: sha(p) for p in ['events_board.csv', 'build_summary.json', 'build_events.py', 'mops_titles.parquet']},
)
(HERE / 'gate.json').write_text(json.dumps(G, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(G, ensure_ascii=False, indent=1))

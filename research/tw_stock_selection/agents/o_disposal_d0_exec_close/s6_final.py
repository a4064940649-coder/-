#!/usr/bin/env python3
"""事後彙整（不重算報酬）：把判定、G1～G8、自我交叉核對與揭露寫進 results.json 的 FINAL。用法：python3 -I -B s6_final.py"""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
res = json.loads((HERE / 'results.json').read_text())
R = res['RESULTS']
rep = json.loads(subprocess.run([sys.executable, '-I', '-B', str(HERE / 'replicate' / 'rep_min.py')], capture_output=True, text=True).stdout.strip().splitlines()[-1])
res['FINAL'] = {
    'verdict': '不可用（結案）', 'trigger': 'K1：驗證期主欄 9 時點對 0050 月 t 中位數 −0.16 < 1.0',
    'K': {k: R['DECISION'][k] for k in ('K1', 'K2', 'K3')},
    'G': {k: v.get('pass') for k, v in R['G'].items()},
    'self_crosscheck_rep_min': {'rep': rep, 's4_FULL_C1xCL': R['TRADES']['100萬|normal']['FULL']['by_timing']['C1xCL'],
                                's4_VAL_C1xCL': R['TRADES']['100萬|normal']['VAL']['by_timing']['C1xCL'],
                                'note': '同一代理重寫（不 import dlib），只核對 C1×CL 每槽 10 萬；不等於 G6'},
    'disclosures': [
        '嚴格預收現金帳在少數部分成交時有微小透支（主欄最低 −241 元；GROW 100 萬最低 −8,806 元），原因是預算以限價計但未預留手續費與衝擊成本；量級 < 0.1% 權益，不影響判定',
        'K1 在所有預收口徑都觸發：none（o9 口徑，無現金池）驗證期 t 中位數 0.78，threshold −0.27，k1 −0.06，k3 −0.37',
        '版本數：4 個迭代版本（E1～E4）＋207 條衛星變體（23 種設定 × 9 時點）＋1 條焦點，全部併入 G3 族群',
        'FULL 2006-04～2026-10 為描述段（o9 已看過）'],
    'action': '依事前規則結案：建議主研究員從 SOP 永久刪除處置配置列；前瞻只保留事件時戳存檔；不再開處置變體。本題未修改 SOP、o_forward_infra_live、live_tracking/main_forward。'}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(res['FINAL'], ensure_ascii=False, indent=1)[:2500])

#!/usr/bin/env python3
"""步驟 9：凍結（判定器測試全過、FINDINGS 機器比對通過後）：status_chain.jsonl GENESIS、FREEZE_MANIFEST_v2a.json、BACKLOG_ANCHOR_LINE_v2a.txt。
用法：python3 -I -B s9_freeze.py <findings 草稿檔（scratchpad）>"""
import sys, json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g8alib as L

R = L.load_res()
assert not (HERE / 'FREEZE_MANIFEST_v2a.json').exists(), '已凍結'
L.check_reg('PREREG', files=False)
T = R['EVALUATOR_TEST']; S = T['summary']
proto = HERE / 'G8_PROTOCOL_v2a.json'; ev = HERE / 'g8_evaluate_v2a.py'
assert S['protocol_sha256'] == L.sha(proto) and S['evaluator_sha256'] == L.sha(ev), '測試用的協定／判定器與現檔不符'
assert S['consistency_mismatch'] == 0 and S['pathwise_violations'] == 0 and S['governance_tests_ok'].split('/')[0] == S['governance_tests_ok'].split('/')[1]
assert T['attack_summary']['n_attacks_change_v2a'] == 0
OCs = R['OC_FINAL']['summary']; assert OCs['qualification'][OCs['selected']]['qualified']
PA = json.loads(proto.read_text(encoding='utf-8'))
# FINDINGS 機器比對：草稿中每個小數都要在協定、oc 摘要、results.json 或 explore 找得到
fd = Path(sys.argv[1]).read_text(encoding='utf-8')
hay = proto.read_text(encoding='utf-8') + json.dumps(R, ensure_ascii=False) + (HERE / 'explore.json').read_text(encoding='utf-8') + \
      (HERE / 'K_ALIGNMENT.json').read_text(encoding='utf-8') + (HERE / 'replicate_self/rep_ba.json').read_text(encoding='utf-8') + \
      (HERE / 'evaluator_tests.json').read_text(encoding='utf-8') + (L.V2_DIR / 'FINDINGS.md').read_text(encoding='utf-8')
nums = sorted(set(re.findall(r'(?<![\d.])\d+\.\d+(?![\d.])', fd)))
FL = sorted({abs(float(m)) for m in re.findall(r'-?\d+\.\d+(?:e-?\d+)?', hay)})
def found(x):
    if x in hay: return True
    v = float(x); d = len(x.split('.')[1])
    return any(round(m, d) == v for m in FL)
missing = [x for x in nums if not found(x)]
unres = [x for x in missing if not found(str(round(float(x) / 100, len(x.split('.')[1]) + 2)))]   # 百分比寫法 → 比對 /100
check = dict(n_numbers=len(nums), n_direct_missing=len(missing), n_resolved_as_percent=len(missing) - len(unres), unresolved=unres,
             findings_draft_sha256=L.sha(sys.argv[1]))
print('FINDINGS check', check)
assert not unres, f'FINDINGS 有數字找不到：{unres}'

ZERO = '0' * 64
import importlib.util
spec = importlib.util.spec_from_file_location('ev_f', ev); EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
utc = L.now_utc()
g = dict(seq=0, type='GENESIS', utc=utc, prev_sha=ZERO, protocol_sha256=L.sha(proto), evaluator_sha256=L.sha(ev),
         note='G8P-2a status 雜湊鏈起點（只能追加；只有 label＝OFFICIAL 的正式呼叫能追加）')
g['entry_sha'] = EV.entry_sha(g)
(HERE / 'status_chain.jsonl').write_text(EV.canon(g) + '\n', encoding='utf-8')
man = dict(protocol='G8P-2a', frozen_utc=utc, protocol_frozen_utc=PA['frozen_utc'], deadline_utc='2026-11-10T05:30:00+00:00',
           protocol_sha256=L.sha(proto), evaluator_sha256=L.sha(ev), status_chain_genesis_entry_sha=g['entry_sha'],
           status_chain_genesis_file_sha256=L.sha(HERE / 'status_chain.jsonl'),
           code_sha256={f: L.sha(HERE / f) for f in ('g8alib.py', 'refimpl.py', 'synth.py', 's7_test_evaluator.py', 's8_protocol.py', 's9_freeze.py')},
           results_registrations=dict(PREREG_DESIGN=R['PREREG_DESIGN_sha256'], PREREG_DESIGN_utc=R['PREREG_DESIGN_utc'], PREREG=R['PREREG_sha256'], PREREG_utc=R['PREREG_utc']),
           smoke_manifest_sha256=L.sha(HERE / 'smoke/SMOKE_MANIFEST_v2a.json'), oc_final_sha256=L.sha(HERE / 'oc_final.json'),
           evaluator_tests_sha256=L.sha(HERE / 'evaluator_tests.json'), k_alignment_sha256=L.sha(HERE / 'K_ALIGNMENT.json'),
           addendum_to=dict(protocol_sha256=L.V2_PROTO_SHA, manifest_sha256=L.V2_MANIFEST_SHA, evaluator_sha256=L.V2_EVAL_SHA),
           g8p1_sha256=L.G8P1_SHA, selected=PA['params']['selected'], b_a=PA['params']['b_a'],
           effective_rule='origin push 時間（manifest 首次 commit 與 BACKLOG 錨定行首次 commit 取較晚者）< 2026-11-10T05:30Z → 生效錨定月 2026-11；否則 push 後第一個完整錨定月',
           findings_machine_check=check)
(HERE / 'FREEZE_MANIFEST_v2a.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
ms = L.sha(HERE / 'FREEZE_MANIFEST_v2a.json')
line = (f"- [G8P-2a 凍結錨定] o_g8p2a_protocol_addendum：FREEZE_MANIFEST_v2a.json sha256 {ms}；G8_PROTOCOL_v2a.json sha256 {L.sha(proto)}；"
        f"g8_evaluate_v2a.py sha256 {L.sha(ev)}；status_chain GENESIS {g['entry_sha']}；凍結 UTC {utc}（期限 2026-11-10T05:30Z）。"
        f"G8P-2（{L.V2_PROTO_SHA[:8]}…）與 G8P-1 保留不刪；v2a 以 origin push 時間判生效月（早於期限 → 2026-11），判定＝G8P-2 原樣與 v2a 規則的較不利者。\n")
(HERE / 'BACKLOG_ANCHOR_LINE_v2a.txt').write_text(line, encoding='utf-8')
L.save_key('G8_FREEZE', dict(manifest_sha256=ms, protocol_sha256=L.sha(proto), evaluator_sha256=L.sha(ev), frozen_utc=utc, genesis=g['entry_sha'],
                             backlog_line_sha256=L.sha(HERE / 'BACKLOG_ANCHOR_LINE_v2a.txt')))
print('FROZEN', utc, 'manifest', ms)
print(line)

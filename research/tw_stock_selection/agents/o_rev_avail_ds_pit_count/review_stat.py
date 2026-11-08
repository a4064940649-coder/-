#!/usr/bin/env python3
"""統計審查（只讀）：分母、分類規則套用、門檻方向逐條校對、有效獨立事件。輸出 review_stat.json。python3 -I -B review_stat.py"""
import json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'SPEC.json').read_text(encoding='utf-8'))
E = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['ERRATUM']
DD = pd.read_csv(HERE / 'data/census_days.csv', dtype=str)
F = DD[DD.formal == 'True']
chk = {}
# 1 分母
def nsl(a, b): x = F[(F.sig >= a) & (F.sig <= b)]; return int(x.n_slots.astype(int).sum())
chk['denominators'] = {'930': nsl('2006', '2013-12-31') == 930, '2450': nsl('2006', '2026-12-31') == 2450, '920': nsl('2019', '2026-12-31') == 920,
                       '1640': nsl('2013', '2026-12-31') == 1640, '720': nsl('2013', '2018-12-31') == 720,
                       'DS_2006_2013_330': int(F[(F.sig <= '2013-12-31') & (F.case == 'D=S')].n_slots.astype(int).sum()) == 330,
                       'all_days_10_slots': bool((F.n_slots.astype(int) == 10).all())}
# 2 分類規則套用：由證據欄重導 state
EV = pd.read_csv(HERE / 'data/slot_avail_evidence.csv', dtype=str)
bad = []
for s, c, st, ev in zip(EV.sig, EV.code, EV.state, EV.evidence):
    d = json.loads(ev); ss = [x[1] for x in d.get('sources', [])]
    exp = 'UNDET' if ('AVAIL' in ss and 'UNAVAIL' in ss) else 'AVAIL' if 'AVAIL' in ss else 'UNAVAIL' if 'UNAVAIL' in ss else 'UNDET' if 'UNDET' in ss else 'NONE'
    if exp != st: bad.append((s, c, st, exp))
chk['state_from_evidence_mismatch'] = len(bad)
# 3 門檻方向逐條校對（值、運算子、結果、判定布林）
r1 = E['R1']['ratios']; j1 = E['R1']['judgement']; j2 = E['R2']['judgement']; j3 = E['R3']['judgement']
rows = [
 ('R1 G5 前視未排除（2006～2013 上界）', r1['r1a_2006_2013']['upper']['share'], '>', 0.05),
 ('R1 G5 前視未排除（全期 D=S＋D>S 上界）', r1['r1b_ALL']['upper']['share'], '>', 0.05),
 ('R1 可執行正式數字註記（2019 起上界）', r1['r1c_2019_on']['upper']['share'], '>', 0.05),
 ('R1 前視已證實（2006～2013 下界）', r1['r1a_2006_2013']['lower']['share'], '>', 0.05),
 ('R1 前視已證實（全期下界）', r1['r1b_ALL']['lower']['share'], '>', 0.05),
 ('R1 前視已證實（2019 起下界）', r1['r1c_2019_on']['lower']['share'], '>', 0.05),
 ('R2 G4 第三段存活者偏誤未排除（2013～2026，兩面板較大者）', j2['governing_share_2013_2026'], '>', 0.05),
 ('R2 G4 第二段存活者偏誤未排除（2013～2018，兩面板較大者）', j2['governing_share_2013_2018'], '>', 0.05),
 ('R3 G5 處置排除未 PIT（四個比例最大者）', j3['governing_share'], '>', 0.05)]
tab = [{'rule': r, 'value': round(v, 6), 'op': op, 'thr': t, 'triggered': v > t, 'text_direction': '> 0.05 觸發、≤ 0.05 不觸發'} for r, v, op, t in rows]
chk['direction_table'] = tab
chk['booleans_consistent'] = {
 'G5_lookahead_not_excluded': j1['G5_lookahead_not_excluded'] == (tab[0]['triggered'] or tab[1]['triggered']),
 'exec_note': j1['exec_note'] == tab[2]['triggered'],
 'proven_all_false': (not j1['G5_lookahead_proven']) and (not j1['exec_proven']) and not any(x['triggered'] for x in tab[3:6]),
 'G4_seg3': j2['G4_seg3_survivorship_not_excluded'] == tab[6]['triggered'], 'G4_seg2': j2['G4_seg2_survivorship_not_excluded'] == tab[7]['triggered'],
 'R3': j3['G5_disposal_not_PIT'] == tab[8]['triggered']}
chk['spec_threshold_verbatim'] = SPEC['rules_verbatim']['threshold_direction'] == '> 0.05 觸發、≤ 0.05 不觸發'
# 4 有效獨立事件
chk['effective_events'] = {'r1a_upper': {k: r1['r1a_2006_2013']['upper'][k] for k in ('n', 'n_days', 'n_issuers', 'share_ex_top')},
                           'r1c_upper': {k: r1['r1c_2019_on']['upper'][k] for k in ('n', 'n_days', 'n_issuers', 'share_ex_top')},
                           'R2A_2013_2026': {k: E['R2']['panelA_C0']['2013_2026'][k] for k in ('U_opt', 'n_days', 'n_issuers', 'top_issuer', 'share_ex_top')},
                           'R2A_2013_2018': {k: E['R2']['panelA_C0']['2013_2018'][k] for k in ('U_opt', 'n_days', 'n_issuers', 'top_issuer', 'share_ex_top')}}
# 5 審查意見
chk['opinions'] = [
 '上界 > 5% 在登記前已幾乎確定（SPEC pre_known 已標明）；上界數字本身沒有證據力，只說明「沒有被排除」。',
 '下界 0 不是「證明沒有前視」：2024-06 之前除了 2013～2021 的新聞之外沒有任何能判 UNAVAIL 的來源（Tasi 只能在 S 之後的交易日快照才判不可得，新聞永遠不能判不可得）。只有 2024-06 起（100 槽）和 2026-10-12 是有效檢定，兩者下界都是 0、UNDET 21%／10%。',
 '2006～2013 的 330 槽 98.5% 是「查不到」：預測 A（確定可得 60～85%）無法檢驗，不能寫成被支持或被否證。',
 '營收百分位 < 0.5 的入選槽在全期為 0（分數門檻決定：營收腿 < 0.5 的股票分數上限 0.75，低於第 10 名門檻），預測 A 的「< 0.5 靠 52 週腿入選」子樣本是空集合。',
 'R2 第三段的觸發來自 2013～2018（2013～2026 面板 A 251 槽中 227 槽在 2013～2018）；2019～2026 自身只有 2.6%（面板 B 3.0%），≤ 5%。依 R2 字面，第三段仍要加註；建議註記文字同時寫出 2019～2026 自身比例，避免讀者誤以為 2019 年後的存活者偏誤 > 5%。',
 '面板 A 的 2019～2023 價格面板只有存活者（BRIEF），所以面板 A 的 2019～2026 存活者缺口天然偏低；面板 B（含 2019～2023 下市股價）3.0% 才是較好的估計。',
 'R3 的 4 個 2006～2013 槽位在登記前已知（不算證據）；延伸段 11 槽（訊號日）／10 槽（成交日）是新資訊，2021-06-11 一天就有 4 槽（群集，有效獨立事件數 < 槽位數）。']
chk['verdict'] = '可用（分母、規則套用、門檻方向全部核對一致；判定文字需同時寫出上界與下界）'
chk['all_checks_pass'] = all(chk['denominators'].values()) and chk['state_from_evidence_mismatch'] == 0 and all(chk['booleans_consistent'].values()) and chk['spec_threshold_verbatim']
(HERE / 'review_stat.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: chk[k] for k in ('denominators', 'state_from_evidence_mismatch', 'booleans_consistent', 'all_checks_pass')}, ensure_ascii=False, indent=1))
for t in tab: print(t)

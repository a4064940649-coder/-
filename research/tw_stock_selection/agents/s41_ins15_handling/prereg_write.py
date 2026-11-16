#!/usr/bin/env python3
"""S41 事前登記：寫 PREREG.json 與 PREREG.sha256（describe.py 之後、train.py 之前跑一次；已存在就拒絕）。python3 -I -B prereg_write.py"""
import json, hashlib, datetime, sys
from pathlib import Path
H = Path(__file__).resolve().parent
assert not (H / 'PREREG.json').exists(), 'PREREG.json 已存在，拒絕覆寫'
sys.path.insert(0, str(H))
import s41lib as L
D = json.loads((H / 'results_describe.json').read_text())
sha = lambda p: hashlib.sha256((H / p).read_bytes()).hexdigest()
cf = D['counterfactual_cells']
PRE = {
    'topic': 'S41 2026 年起 15 日營收期限的保險業及具保險子公司（S40 名單 18 檔＋2884）在換股日看不到當月營收：make_list.py 實盤該怎麼處理？',
    'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    'status_at_write': '只跑過 describe.py：驗收 A0／A1、現行 CUR（全部看得到）的描述（這群公司 F1 4 次、前 20 名 61 次、交易損益占比約 1.8%）、反事實遮蔽格數。沒有算任何處理 A／B／C 的報酬或交易。',
    'sha256_at_write': {'s41lib.py': sha('s41lib.py'), 'describe.py': sha('describe.py'), 'results_describe.json': sha('results_describe.json')},
    'group': {'codes': L.GROUP, 'source': 'S40 s40lib.INS15（18 檔：11 家上市金控中具保險子公司者＋上市保險＋2905、2207）＋2884（S40 註明 2026-09 起實際照 15 日）；回溯 2016～2026 用同一批代碼，不做其他增減。'},
    'baseline_engine（不改定義）': 'S10 E0／W10_70（XQ 20＋21 號預設）：s20lib→s17lib→s10lib，cslib.EXRIGHT_FIX＝True，s17lib.e50_fix_suspension；股池＝stlib.universe(P,"twse") & ~創新板；SURGE＝STD；make_signal_s10(shift)；variant_S("E0")；simulate_s10(cost=strict, sat_w=0.7, sizing="W10", lock=False)。',
    'counterfactual': {
        'definition': '假設這群公司一直適用 15 日期限：營收月份 m 的訊號列 av[m]＋位移 s，若早於「次月 15 日（含）以後第一個交易日」→ 該股月在訊號日看不到；期限日＝訊號日視為當晚可見（同 S40）。2016～2026 全期套用（其他股票不動）。',
        'why': '真正受影響只有 2026 年起（10 個訊號月），樣本太少；回溯資料其實看得到，所以用反事實在全期模擬三種處理。',
        'cells_preview（describe，只數格數）': {k: {a: b for a, b in v.items() if a != 'months_visible'} for k, v in cf.items() if k.startswith('+')},
        'note': '位移 0（＝實盤 A10 換股日）幾乎每月都看不到（129 個月中 124 個）；位移 2、3 的訊號日常落在 15 日以後，遮蔽月份少很多（51、15 個月），所以 20 條路徑之間處理效果會不同。實盤 make_list.py 在 A10 執行＝位移 0。'},
    'treatments（≤ 3，只在這群公司的看不到股月上改）': {
        'A': '現狀（＝make_list.py 現行）：看不到＝SURGE 缺值 → 不進 L_MA、不能新買；持有中依 E0（不在 L_MA 前 20）於次日開盤賣出。',
        'B': '持有中不因缺值賣出，改用「最後已知 SURGE」（上個營收月份 S[m−1]，換股日一定已公開）判斷續抱：股池內、還原收盤 > 60 日均線、S[m−1] 有值，且把 S[m−1] 放進本月 L_MA（不含看不到的股票）後名次 ≤ 20（同分依欄位序）→ 續抱；否則賣。不新買；本月 L_MA 排名不含這群公司（其他股票的名次與 A 相同）。'
             '說明：題目寫「沿用上月名次與 SURGE、不賣，等下月再判斷」；因為反事實下每個月都看不到當月營收，若字面上永遠沿用上月名次會變成永不賣出，所以登記為「每個換股日用最新已公開的上月 SURGE 判斷續抱」，等於延後一個月判斷。',
        'C': '看不到的股月 SURGE 改用上月 S[m−1]，完整參與 L_MA 排名、F1 買進與 E0 續抱（＝S40 的 PST 敏感度）。',
        'equivalence': '這群公司一律看得到（遮蔽格空）時，A、B、C 的訊號、出場矩陣、交易與淨值都與現行 CUR 逐位相同（驗收 A1，describe 已通過）。'},
    'windows': {'train': '訊號日 2016-02～2021-12（營收月份 ≤ 2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出，同 S25）',
                'valid': '訊號日 2022-01～2026-10（營收月份 ≥ 2021-12）；重新起跑（只保留營收月份 ≥ 2021-12 的訊號，起始資金縮放使 2021-12-30 淨值約＝60～200 萬），驗證窗 2021-12-30 → 2026-10-22；只跑一次'},
    'paths': '20 條路徑＝訊號日位移 0～3 × 起始資金 60／80／100／150／200 萬；位移 s 的遮蔽依 av[m]＋s 重新判斷。',
    'selection（訓練期，沿用 S25）': '比較基準＝A（實盤現狀）。候選＝B、C。分數＝候選 − A 的訓練窗年化差，20 條路徑中位數（個百分點）；最高者凍結（同分依「差 ≥ +0.25 個百分點的路徑數」，再依簡單者 B 優先）。'
                                  'T0（含改善最小幅度）：分數 ≥ +1 個百分點，且差 ≥ +0.25 個百分點的路徑 ≥ 14／20。T0 不過仍凍結分數最高者，驗證期只描述、判定必為不通過。',
    'verdict（驗證期只跑一次，沿用 S25／S21，凍結者 FZ 對 A）': {
        'V1': '驗證窗（重新起跑）年化差 > 0 的路徑 ≥ 15／20',
        'V2': '配對差中位數 ≥ +2 個百分點',
        'V3': '禁買 A 與 FZ 各自（重新起跑主跑）最賺 5 筆期間的聯集後，配對差中位數 > 0',
        'V4': '兩段空頭（2022；2024-07～2025-04）段內最大回撤配對差中位數都不深於 −2 個百分點',
        'overall': 'T0 且 V1～V4 全過＝通過（實盤採 FZ）；否則不通過＝各處理差異視為雜訊內，改用下面登記的建議規則'},
    'recommendation_rule（不通過時，事前固定）': {
        'closeness': '與回測（現行 CUR，這群公司全部看得到）的距離 d＝全期連續模擬 2019 起年化與 CUR 之差的絕對值，20 條路徑中位數（個百分點）；另報驗證窗重新起跑同口徑與交易集合 Jaccard（描述）。',
        'rule': '取 d 最小者；但若較簡單的處理的 d 與最小值相差 ≤ 0.10 個百分點，選較簡單者。簡單順序：A（不改程式）< C（缺值以上月 SURGE 補，一處改動）< B（持股另判續抱，需新邏輯）。',
        'why': '差異在雜訊內時，選最簡單、最接近回測的做法；兩者衝突時以 0.10 個百分點容忍度偏向簡單。'},
    'descriptive（不參與判定）': [
        'A、B、C 對 CUR（全部看得到）：訓練窗、驗證窗、2019 起、全期年化與最大回撤（20 條路徑中位數與標準路徑）',
        'B 續抱事件數、C 多買／少買的交易明細；三者與 CUR 的交易差異',
        '實際情境（只有 2026-01 以後的營收月份看不到）：A、B、C 對 CUR 的標準路徑與 20 條路徑差（2024 起、全期）',
        '成本加倍標準路徑'],
    'acceptance': {'A0': 'CUR 重現 2019 起 48.99%、全期 34.54%、回撤 −32.11%、已出場 381 筆（±0.025 個百分點）',
                   'A1': '遮蔽格空時 A、B、C 與 CUR 逐位相同（訊號、X、交易、淨值）',
                   'A2': 'B、C 在第一個遮蔽格影響到的訊號列以前淨值與 A 逐位相同（描述檢查）'},
    'N': '資料窺探族群 N：＋2（B、C 兩條候選；A 是基準）',
    'predictions（事前預測）': {
        'P1': '這群公司 10 年只被 F1 選中 4 次，訓練期（1 次）B、C 對 A 的分數都在 ±0.3 個百分點內，T0 不過（機率 > 95%）',
        'P2': 'C 最接近 CUR（因為它讓這群公司照常參與排名），B 次之，A 最遠；但三者與 CUR 的距離都 < 0.3 個百分點',
        'P3': '最後建議很可能落在 A（距離差 ≤ 0.10 個百分點時選最簡單）或 C'},
    'discipline': '只用公開資料；python3 -I -B；峰值記憶體 ≤ 3GB；暫存 scratchpad/s41 結束刪除；不改共用引擎、make_list.py 與既有研究；不 git commit/push',
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
(H / 'PREREG.json').write_text(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{h}  PREREG.json\n')
print(h)

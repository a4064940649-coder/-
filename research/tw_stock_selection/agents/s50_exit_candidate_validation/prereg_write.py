#!/usr/bin/env python3
"""S50 事前登記：在本題算任何報酬（含驗收）之前寫 SPEC.json、PREREG.json 與各自的 sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
for n in ('SPEC.json', 'PREREG.json'):
    assert not (H / n).exists(), f'{n} 已存在，拒絕覆寫'
lib_sha = hashlib.sha256((H / 's50lib.py').read_bytes()).hexdigest()
now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

SPEC = {
 'id': 's50_exit_candidate_validation',
 'title': 'S50 出場候選（S48 事後看到的 C1／C2／C3）能不能取代現行 E0？——訓練→凍結→驗證一次（S25 改良判定）',
 'written_at': now,
 'status_at_write': '只讀過 S48（FINDINGS、run.py、results.json）、S25／S44 的 PREREG 與程式；本題沒有跑任何模擬。s50lib.py 已寫好（sha256 見下）。',
 's50lib_sha256_at_write': lib_sha,
 'engine': {
  'chain': 's20lib → s17lib → s10lib（全部 import，不改任何既有檔案）；載入與自建出場矩陣逐行照 agents/s48_exit_rule_family/run.py。',
  'flags': 'cslib.EXRIGHT_FIX＝True（load 前）；SURGE（STD）在 DEADLINE_FIX／INS15_MASK 關閉下建，之後兩者＝True、cslib.rev_avail_days(P, months, True)；s17lib.e50_fix_suspension。三個正式開關全開。',
  'universe': "stlib.universe(P, 'twse') & ~創新板",
  'entry': 'F1 不動：s10lib.make_signal_s10(P, U, RT, shift, depth=10, mom_lo=0.10, nsel=10)；進場 60 日均線（P.F_MA60）不動。',
  'exit_cell': '出場格 (N, M)＝S48 定義：每個訊號列 k，續抱＝股池內且 SURGE 有值（INS15 遮掉＝缺值）且 SURGE ≥ 當月 L_MA 第 N 名分數（N＝不用時省略）且 還原收盤 > M 日均線（M＝不用時省略）；否則次一交易日開盤賣。只在換股訊號日檢查。',
  'simulate': "s10lib.simulate_s10(P, S', U, cost=COSTS['strict'], capital, sat_w=0.7, sizing='W10', lock=False)＝W10_70、嚴成本。",
  'paths': '20 條：資金 {60, 80, 100, 150, 200} 萬 × 換股日位移 {0, 1, 2, 3}；主跑＝100 萬、位移 0。'
 },
 'candidates（事前固定，共 3 條＋現行）': {
  'CUR': '(N 20, M 60)：現行 E0',
  'C1': '(N 20, M 120)：前 20 名＋120 日均線',
  'C2': '(不用排名, M 20)：不用排名＋20 日均線',
  'C3': '(N 20, 不用均線)：前 20 名＋不用均線（S48 鄰近候選；＝S10 MANO 寫法）'
 },
 'windows': {
  'train': '訊號日 2016-02～2021-12（營收月份 ≤ 2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30，從全期連續模擬切出（同 S44／S25）。',
  'valid': '訊號日 2022-01～2026-10（營收月份 ≥ 2021-12）；重新起跑：s20lib.fresh_from 只保留營收月份 ≥ 2021-12 的訊號（出場矩陣不動），驗證窗 2021-12-30 → 2026-10-22；起始資金縮放使現行主跑 2021-12-30 淨值＝各路徑資金（同 S44）。',
  'posthoc': '事後描述（不影響判定）：全期連續模擬 2016-03～2018-12（2016-02 最後交易日 → 2018 最後交易日）與 2024 起（2023 最後交易日 → 2026-10-22）分段，四個設定 × 20 條。'
 },
 'acceptance': {
  'A0': 'rmean(P.G, 60) 重建 F_MA60 與 P.F_MA60 逐格相同。',
  'A1': '(20, 60) 自建出場矩陣與 make_signal_s10 的 XV[E0] 在位移 0～3 逐格相同。',
  'A2': '現行主跑 2019 起 48.76%（0.4875617247721724）、全期、全期回撤、已出場 381 筆與 deadline_engine_fix/out/results.json ALL 逐位相同；20 條 2019 起逐條相同。',
  'A3': '重新起跑的現行主跑：2022-01-01 前沒有任何股票進場。',
  'rule': '任一不符 → 停止、不報結果。驗收放在 train.py 開頭，只印驗收數字（現行格數字已知）。'
 },
 'scripts': 's50lib.py（共用）、prereg_write.py（本檔）、train.py（驗收＋訓練＋凍結：寫 results_train.json、FROZEN.json／FROZEN.sha256；FROZEN 已存在就拒絕）、validate.py（凍結後只跑一次：寫 results.json；已存在就拒絕；含事後分段描述）。',
 'resources': '單行程；峰值記憶體 ≤ 3GB；python3 -I -B；暫存 /tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s50/（結束刪除）；不存淨值。'
}

PRE = {
 'id': 's50_exit_candidate_validation',
 'written_at': now,
 'question': 'S48 事後看到 C1（前 20 名＋120 日均線，20/20 正、+1.90）與 C2（不用排名＋20 日均線，20/20 正、+3.83）比現行 E0 好；照研究協定在訓練期選一個、凍結、驗證期只跑一次，S25 標準全過才建議改出場。',
 'honesty_note（事前寫明的偏誤來源）': [
  '候選是 S48 在 2019-01～2026-10 全期看過結果後挑出的；S48 也已公布 2019～2023 與 2024 起兩段逐條差：C1 −1.19／+8.99、C2 +6.88／−1.45、C3 −2.48／+11.46（個百分點，20 條中位數）。所以驗證期（2022-01 起）不是盲的：2024 起的方向已知，只有 2022-01～2023-12 的部分與「重新起跑」的組合沒看過。',
  '訓練期 2016-02～2021-12 裡，2016-03～2018-12 在 S48 只有各格中位數與名次（事後描述），2019-01～2021-12 沒有單獨報過。',
  '因此本題通過也只能說「沒被否定」，不通過則可直接判不改。驗證不通過時不換候選、不換門檻重跑。'
 ],
 'candidates（事前固定）': {'C1': '前 20 名＋120 日均線', 'C2': '不用排名＋20 日均線', 'C3': '前 20 名＋不用均線（S48 鄰近候選）', 'CUR': '現行 E0（前 20 名＋60 日均線）'},
 'selection（訓練期，事前寫死）': {
  'score': '每個候選的訓練期分數＝候選 − 現行的訓練窗（2016-01-29 → 2021-12-30）年化差，20 條路徑逐條配對後取中位數（個百分點）。',
  'rule': '三個候選中分數最高者；其分數 ≥ +1 個百分點 → 凍結該候選；否則凍結「不改」（現行）。同分依「年化差 ≥ +0.25 的路徑數」多者、再依 C1→C2→C3 順序。',
  'T0（S25）': '被選中的候選：分數 ≥ +1 個百分點，且年化差 ≥ +0.25 個百分點的路徑 ≥ 14／20。',
  'if_no_change': '若凍結「不改」或 T0 不過：總判定必為「不改」；驗證期仍只跑一次，對訓練分數最高的候選照 V1～V4 算出來，只作描述。'
 },
 'verdict（驗證期只跑一次，沿用 S25）': {
  'V1': '驗證窗（重新起跑）年化差 > 0 的路徑 ≥ 15／20',
  'V2': '20 條配對差中位數 ≥ +2 個百分點',
  'V3': '禁買現行與凍結候選各自（重新起跑主跑 100 萬、位移 0）已出場交易中損益最大 5 筆的持有期（兩邊聯集，整段禁買），重跑 20 條後配對差中位數仍 > 0',
  'V4': '兩段空頭（2022-01-01～2022-12-31；2024-07-01～2025-04-30）段內最大回撤的配對差中位數都不深於 −2 個百分點（候選 − 現行 ≥ −2）',
  'overall': 'T0 且 V1～V4 全過＝建議改出場；任一不過＝不改。不通過時停在描述，不換候選、不換參數重跑。'
 },
 'posthoc（事後描述，不影響判定）': '全期連續模擬四個設定 × 20 條：2016-03～2018-12、2024 起、2019 起、2019～2021；另列三個候選在驗證窗（重新起跑）的配對差（未被選中的只描述）。',
 'acceptance': 'A0～A3 見 SPEC.json；任一不符停止。',
 'N': '資料窺探族群 N：＋3（三個候選；S48 已計 23 格，這裡是同一批格子的再檢定，研究中心可決定是否重複計）。',
 'predictions（事前預測，可被否證）': [
  'P1：訓練期 C2 分數最高（S48 2019～2023 +6.88，短均線在 2019～2021 換股快、吃到輪動），C1、C3 訓練期分數 < +1（S48 2019～2023 都是負）。',
  'P2：若凍結 C2，驗證期 V2 不過（S48 2024 起 −1.45，驗證窗一半以上在 2024 起），總判定「不改」。',
  'P3：T0 通過機率約 40%；T0 且 V1～V4 全過約 10%。',
  'P4：C1 回撤（V4 兩段）比現行淺；C2 回撤與現行差不多。'
 ],
 'resources': '單行程、峰值記憶體 ≤ 3GB；python3 -I -B；暫存 scratchpad/s50/，結束刪除。'
}
for n, o in (('SPEC', SPEC), ('PREREG', PRE)):
    txt = json.dumps(o, ensure_ascii=False, indent=1)
    (H / f'{n}.json').write_text(txt)
    s = hashlib.sha256(txt.encode()).hexdigest()
    (H / f'{n}.sha256').write_text(f'{s}  {n}.json\n')
    print(n, s)
code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(H.glob('*.py'))}
(H / 'code_prereg.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in code.items()))

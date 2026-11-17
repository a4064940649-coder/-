#!/usr/bin/env python3
"""S45 事前登記：在算任何非現行格的報酬之前寫 SPEC.json、PREREG.json 與各自的 .sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只做描述：股池檔數、L_MA 檔數、F1 目標數與重疊率；沒有算任何報酬。）"""
import sys, json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
for n in ('SPEC.json', 'PREREG.json'):
    assert not (H / n).exists(), f'{n} 已存在，拒絕覆寫'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
dc = json.loads((H / 'data_check.json').read_text())
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00')

SPEC = {
 'topic': 'S45 股池門檻家族：規格（引擎、資料、驗收、資源）',
 'written_at': now,
 'engine_chain': 's45lib → s20lib → s17lib → s10lib → s9lib → … → stlib → cslib（全部不改）；cslib.EXRIGHT_FIX＝True、DEADLINE_FIX＝True、INS15_MASK＝True（load 前設定）；'
                 's17lib.e50_fix_suspension（0050 停牌日開盤用前一日收盤）；SURGE＝s17lib.rev_family STD（σ>0）；s10lib.make_signal_s10(shift)（F1：L_MA 前 10 名中 RET20>10%、不遞補）；'
                 's10lib.variant_S(E0)；s10lib.simulate_s10(嚴成本, W10, sat_w=0.7, lock=False, cap_pct=0.05, min_frac=0.5)。',
 'current_universe_from_code': {
  'where': 'agents/s1_shortterm_leaderboard/stlib.py load()：amt＝where(P.listed, nan_to_num(P.AMT), nan)；P.V60＝amt 的 60 日滾動平均（min_periods=60）；'
           'P.liq＝(P.V60 >= 1e7) & (P.C >= 10) & P.has & P.not91；stlib.universe(P,"twse")＝P.liq & (P.MK==1)；現行引擎再 & ~創新板（s4lib.innov_panel）。',
  'units': 'P.AMT＝官方成交金額（元）；P.C＝官方原始收盤（未還原）；P.has＝當日有成交；not91＝代碼不以 91 開頭（TDR）。',
  'where_used': '股池同時用在 (1) 訊號列：make_signal_s10 的 fin＝univ[k] & SURGE 有值（排名、L_MA 前 10／前 20 出場名次、當月有值 < 50 檔不換股）；(2) 引擎買進：cand＝sig[t−1] & univ[t−1]。兩處同一個 U。',
  'wrapper': 's45lib.universe(P, INNOV, adv, px)：同一公式只換 1e7 與 10；(1e7,10) 與 stlib.universe & ~INNOV 逐位相同（第 0 步已驗：A1_universe_bitwise_equal＝%s）。' % dc['A1_universe_bitwise_equal'],
 },
 'data': '公開封存官方日 K（上市，含下市）、月營收 tw_revenue（同 s2lib）、TWT49U 除權息參考價、營收期限例外表與 15 日期限名單（raw/rev_deadline_exceptions）、0050 日 K。面板 T＝2876、N＝2142、最後一天 2026-10-22；淨值起算 2015-07-01。',
 'windows': '2019 起＝2018 最後交易日 → 2026-10-22；全期＝2015-07-01 起；2024 起＝2023 最後交易日起；分段 2019-01～2023-12、2024-01 起；最大回撤＝全期。',
 'paths': '20 條＝本金 {60, 80, 100, 150, 200} 萬 × 位移 {0,1,2,3}（訊號列往後移 s 個交易日、用移後那天的股池／均線／RET20 重算名單與出場名次；營收月份不變；INS15 遮罩依位移後日期）。',
 'acceptance': {
  'P0': '面板 T＝2876、N＝2142、最後一天 2026-10-22。',
  'A1': '現行格股池與 stlib.universe(P,"twse") & ~INNOV 逐位相同。',
  'A0': '現行格 100 萬位移 0：2019 起 48.76%、全期 34.40%、最大回撤 −32.12%、已平倉 381 筆（±0.025 個百分點），並與 agents/deadline_engine_fix/out/results.json 的 ALL 標準路徑逐位相同。',
  'A2': '現行格 20 條路徑的 2019 起年化與 deadline_engine_fix ALL 的 20 條逐條相同（差 < 1e-9）。',
  'official_nav': 'xq_scripts/site/navs_official.csv.gz 若存在就對照現行格淨值（寫題時不存在）。'
 },
 'resources': '單行程；python3 -I -B；峰值記憶體 ≤ 3GB（逐格算完即釋放訊號矩陣）；暫存 scratchpad/s45/（結束刪除）；輸出 ≤ 50MB（results.json、navs.csv.gz 只存 15 個主跑＋0050、trades.csv.gz 只存主跑）。',
 'no_touch': '不改共用引擎與既有研究；不碰 xq_scripts、s46_*、trend-radar、o_g8p2a_protocol_addendum、live_tracking/main_forward、o_forward_infra_live；不 commit。',
 'code_sha256_at_write': {n: sha(H / n) for n in ('s45lib.py', 'prep.py', 'prereg_write.py')},
}

PRE = {
 'topic': 'S45：股池門檻（60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10 元）有沒有被挑過？（家族全部列出、全部報告、不挑最好）',
 'written_at': now,
 'spec_file': 'SPEC.json（sha256 見 SPEC.sha256）',
 'status_at_write': '只跑過 prep.py（data_check.json：股池檔數、L_MA 檔數、F1 目標數與對現行重疊率；沒有任何報酬）。非現行格的報酬一個都沒算過。',
 'prior_work_check': 'grep agents/ 全部 .md/.json/.py：沒有成交值／股價門檻家族。S12 是市值分層（不同）；S16 只多一個上櫃 2,000 萬敏感度格（上櫃，不同股池）；S19 D5 只記錄 XQ 腳本用還原價算成交值的差異。',
 'seen_before_prereg': 'S17 FINDINGS／PREREG（家族判定規則）、S9／S10 PREREG decision_rules、BACKLOG 最後 80 行、deadline_engine_fix README 與 out/results.json（現行格已知值：100 萬 48.76%／34.40%／−32.12%／381 筆；20 條路徑逐條已知）、S37 FINDINGS 與 results.json part3（容量做法：單筆＝股數×開盤÷進場日成交金額；1,000 萬約 13% 買進被 5% 上限縮量）、data_check.json。',
 'question': '現行股池門檻（1,000 萬、10 元）在「合理門檻」家族裡是不是剛好最好的？若是，報酬預期要打折。另報容量隨門檻的變化。',
 'family（事前寫死，不得增減）': {
  'adv_60d': [3e6, 5e6, 1e7, 2e7, 3e7],
  'price': [5, 10, 20],
  'cells': '5 × 3＝15 格，id＝A{成交值萬}_P{價}；現行＝A1000_P10。',
  'what_changes': '只換股池公式裡的兩個數字；SURGE 定義、F1、E0、W10_70、嚴成本、5% 成交值上限、創新板排除、只上市、當月有值 < 50 檔不換股（套各格自己的股池）都不動。',
  'not_included': '不加 1 億以上（容量研究另做）、不加無門檻（含全額交割與殭屍股，不是合理候選）、不做市值門檻（S12）、不做上櫃（S16）。',
 },
 'grid': 'W10_70（每檔＝衛星 ÷ 10、0050 底倉 30%、±10 個百分點帶狀拉回、不鎖定）。',
 'per_cell_runs': '100 萬位移 0 主跑；成本加倍主跑；20 條路徑；容量：1,000 萬位移 0 無上限（cap_pct＝0）、1,000 萬與 3,000 萬位移 0 照 5% 上限。',
 'primary_statistics（事前指定）': {
  'P1 家族分布': '每格 20 條路徑 2019 起年化中位數（＝路徑中位數）；家族 15 個路徑中位數的中位數、最小、最大；家族 15×20＝300 條路徑的中位數與第 10 百分位。另報 100 萬主跑（次要）。',
  'P2 現行名次': '現行在 15 格依路徑中位數的名次（1＝最高）；依 20 條最大回撤中位數的名次（1＝最淺）。',
  'P3 現行減家族中位數': '現行路徑中位數 − 家族 15 個路徑中位數的中位數（個百分點）。',
  'P4 逐條配對': '每格 × 每條路徑（同資金、同位移）該格 − 現行的 2019 起年化差：正的條數、中位數；兩段（2019-01～2023-12、2024-01 起）逐條差中位數；最大回撤差（正＝變淺）。',
  'P5 容量': '1,000 萬無上限主跑的每筆買進「單筆占成交值」＝股數 × 開盤 ÷ 進場日成交金額（同 S37）：平均、中位數、第 95 百分位、> 1% 與 > 5% 的比例；另報單筆 ÷ 進場前一日 60 日均成交值。1,000 萬、3,000 萬 5% 上限主跑：2019 起年化、被上限縮量次數 ÷ 買進數、對同格 100 萬的差。'
 },
 'decision_rules（事前寫死，沿用 S17／S9）': {
  '偏樂觀': 'P3 ≥ 3 個百分點 →「現行門檻偏樂觀」，報酬預期改用家族 300 條路徑中位數（附第 10 百分位）；P3 < 3 →「沒被挑過（不必因門檻打折）」。',
  '現行是不是剛好最好': 'P2 報酬名次＝1 →「是」；2～4（前 1/4）→「前段」；其他 →「不是」。',
  '比現行好／差（每格）': '逐條配對 2019 起年化差：≥ 15／20 條為正而且中位數 ≥ +2 個百分點 →「比現行好」；≤ 5／20 條為正而且中位數 ≤ −2 →「比現行差」；其他 →「差不多」。',
  '穩定地比較好': '「比現行好」而且兩段逐條差中位數都 > 0 →「穩定地比較好」；只有一段 > 0 →「比現行好但只靠一段」。（本題不做 min_frac 敏感度，因此「穩定」比 S17 少一個條件，照實註明。）',
  '容量分級（每格）': '1,000 萬無上限主跑中單筆占成交值 > 5% 的買進比例：< 15% →「1,000 萬可行」；15%～< 30% →「1,000 萬受限」；≥ 30% →「只適合小資金（數百萬以下）」。',
  '容量註記': '任何「比現行好」或路徑中位數高於現行的格，若容量分級比現行差 → 註明「報酬較高但容量較差，小資金以外不可行」。',
  '預設': '不論結果，不改現行預設（家族檢查只判斷預期要不要打折）；若有「穩定地比較好」且容量不差於現行的格，只列為之後獨立題目的候選。'
 },
 'predictions（事前寫下，可被否證）': [
  '現行不會排第 1；P3 < 3 →「沒被挑過」。',
  '門檻越低股池越大，報酬可能略高（小型股），但容量分級變差；3,000 萬格容量最好、報酬略低。',
  '股價門檻 5／10 差異小（F1 重疊 98%）；20 元門檻排除較多低價股，差異較大。',
  '沒有任何格「穩定地比較好」。'
 ],
 'posthoc_planned（只描述，標「事後」，不影響判定）': ['逐年報酬（主跑）', '最賺 5 筆占淨值增加比例（主跑）', '各格與現行不同的交易中損益最大的幾檔'],
 'data_snooping': {'added': 14, 'note': '15 格扣掉現行（已計入）。研究中心 N 依 BACKLOG 彙總另加；路徑、成本加倍、容量是同一格的穩健性／描述，不另計。不做 SPA／DSR。'},
 'data_check_summary': {k: dict(univ_mean_2019=round(v['univ_mean_2019'], 1), f1_overlap_with_cur=round(v['f1_overlap_with_cur'], 3)) for k, v in dc['cells'].items()},
}

(H / 'SPEC.json').write_text(json.dumps(SPEC, ensure_ascii=False, indent=1))
(H / 'SPEC.sha256').write_text(sha(H / 'SPEC.json') + '  SPEC.json\n')
PRE['spec_sha256'] = sha(H / 'SPEC.json')
(H / 'PREREG.json').write_text(json.dumps(PRE, ensure_ascii=False, indent=1))
(H / 'PREREG.sha256').write_text(sha(H / 'PREREG.json') + '  PREREG.json\n')
print(open(H / 'SPEC.sha256').read(), open(H / 'PREREG.sha256').read())

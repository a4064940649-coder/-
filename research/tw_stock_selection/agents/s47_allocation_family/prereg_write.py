#!/usr/bin/env python3
"""S47 事前登記：在算任何非現行格的報酬之前寫 SPEC.json、PREREG.json 與各自的 .sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只算現行格：驗收 A0／A2／A3、官方淨值對照、現行格持股描述；沒有算任何非現行格的報酬。）"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
for n in ('SPEC.json', 'PREREG.json'):
    assert not (H / n).exists(), f'{n} 已存在，拒絕覆寫'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
dc = json.loads((H / 'data_check.json').read_text())
assert dc['A0']['tol_ok'] and dc['A0']['bitwise_vs_deadline_ALL'] and dc['A2']['ok'] and dc['A3_all_ok'] and dc['official_nav']['ok']
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00')

SPEC = {
 'topic': 'S47 資金配置家族：規格（引擎、資料、驗收、資源）',
 'written_at': now,
 'engine_chain': 's47lib → s20lib → s17lib → s10lib → s9lib → … → stlib → cslib（全部不改）；cslib.EXRIGHT_FIX＝True、DEADLINE_FIX＝True、INS15_MASK＝True（load 前設定）；'
                 's17lib.e50_fix_suspension（0050 停牌日開盤用前一日收盤）；SURGE＝s17lib.rev_family STD（σ>0）；股池＝stlib.universe(P,"twse") & ~創新板；'
                 's10lib.make_signal_s10(shift)（F1：L_MA 前 10 名中 RET20>10%、不遞補）；s10lib.variant_S(E0)；嚴成本；cap_pct＝0.05；min_frac＝0.5；lock＝False；band＝0.10；slots＝10；sweep_min＝2 萬。',
 'current_allocation_from_code': {
  'where': 'agents/s10_exit_details/s10lib.py simulate_s10：B＝(1−sat_w)×起始資金（底倉簿記，每天跟 0050 漲跌）；換股執行日開盤先賣、再拉回（|衛星÷淨值 − sat_w| > 10 個百分點才把 B 設回 (1−sat_w)×淨值）、再買；'
           '買進每檔目標 u＝max(開盤估淨值 − B, 0) ÷ 10（sizing＝"W10"）；最多 10 檔（slots）；股數 × 開盤 < 0.5 × u 不買；買股錢不夠可賣 0050（含底倉）；'
           'e50_net：現金 ≥ 2 萬整筆掃進 0050（底倉與空位共用一個 0050 部位）＝「空位買 0050」。',
  'name': 'W10_70：S6 的格名（sizing W10＝衛星 ÷ 10、sat_w 0.7）；S5 主格 W70_T10 事前指定衛星 70%；S6 事前主格是 WN_70、W10_70 是對照；之後研究中心以回撤較淺把 W10_70 定為現行。',
 },
 'variants': {
  'sat_w': '衛星比例 50／60／70／80／100%（直接改 simulate_s10 的 sat_w；100% 時 B 恆為 0、不拉回）。',
  'div': '每檔＝衛星 ÷ 8／÷ 10／÷ 12：÷ 10 走原引擎 simulate_s10(sizing="W10")；÷ 8、÷ 12 走 s47lib.simulate_s47(ndiv=8/12)（s10lib.simulate_s10 原始碼字串替換：只改除數；檔數上限仍 10，所以 ÷ 12 時衛星最多用 10/12）。',
  'idle_cash': '空位留現金：simulate_s47(idle="cash")：0050 部位只維持在底倉簿記 B（B − 0050 ≥ 2 萬且現金 ≥ 2 萬就用現金補買 min(現金, 差額)；0050 − B ≥ 2 萬就賣回現金，計手續費、半跳動點、0.1% 稅），其餘閒置錢留現金、利率 0；'
               '買股錢不夠仍可賣 0050（同現行）。只跑 70%×÷10。',
  'src_sha256': dc['src_sha256'],
 },
 'data': '公開封存官方日 K（上市，含下市）、月營收 tw_revenue（同 s2lib）、TWT49U 除權息參考價、營收期限例外表與 15 日期限名單（raw/rev_deadline_exceptions）、0050 日 K。面板 T＝2876、N＝2142、最後一天 2026-10-22；淨值起算 2015-07-01。',
 'windows': '2019 起＝2018 最後交易日 → 2026-10-22；全期＝2015-07-01 起；2024 起＝2023 最後交易日起；分段 2019-01～2023-12、2024-01 起；最大回撤＝全期。',
 'paths': '20 條＝本金 {60, 80, 100, 150, 200} 萬 × 位移 {0,1,2,3}（訊號列往後移 s 個交易日）；訊號與配置無關，同一位移所有格共用同一份訊號。',
 'acceptance': {
  'P0': '面板 T＝2876、N＝2142、最後一天 2026-10-22。（第 0 步已過）',
  'A0': '現行格 100 萬位移 0：2019 起 48.76%、全期 34.40%、最大回撤 −32.12%、已平倉 381 筆，且與 deadline_engine_fix ALL 逐位相同。（第 0 步已過）',
  'A2': '現行格 20 條路徑 2019 起年化與最大回撤和 deadline_engine_fix ALL 逐條相同（差 0）。（第 0 步已過）',
  'A3': 'simulate_s47(ndiv=None／10, idle="e50") 在現行格 20 條路徑與 simulate_s10 淨值、交易逐位相同（第 0 步已過）；run.py 另在 S50_D10 的 100 萬位移 0 再驗一次 ndiv=10 副本＝原引擎。',
  'official_nav': 'xq_scripts/site/navs_official.csv.gz「E0|W10_70」2,760 天對現行主跑淨值最大差 %.4f 元（檔案 2 位小數）。（第 0 步已過）' % dc['official_nav']['max_abs_diff_yuan'],
 },
 'resources': '單行程；python3 -I -B；峰值記憶體 ≤ 3GB（第 0 步 %d MB）；暫存 scratchpad/s47/（結束刪除）；輸出 ≤ 50MB（results.json、navs.csv.gz 只存 16 個主跑＋0050）。' % dc['maxrss_mb'],
 'no_touch': '不改共用引擎與既有研究；xq_scripts 只讀；不碰 s46_*、trend-radar、o_g8p2a_protocol_addendum、live_tracking/main_forward、o_forward_infra_live；不 commit／push。',
 'code_sha256_at_write': {n: sha(H / n) for n in ('s47lib.py', 'prep.py', 'prereg_write.py')},
}

PRE = {
 'topic': 'S47：資金配置 W10_70（0050 底倉 30%＋衛星 70%、每檔＝衛星 ÷ 10、空位買 0050）有沒有被挑過？（家族全部列出、全部報告、不挑最好）',
 'written_at': now,
 'spec_file': 'SPEC.json（sha256 見 SPEC.sha256）',
 'status_at_write': '只跑過 prep.py（data_check.json：現行格驗收與持股描述）。非現行格的報酬一個都沒在本題算過。',
 'prior_work_check': 'grep agents/BACKLOG.md 與各題 PREREG：沒有做過「衛星比例 × 每檔份數」的家族判定。相關但不同：S5（衛星 50／70／100% × 動能門檻 10／15%，舊引擎，判「值得上 XQ」，非家族名次）；'
                     'S6（W10／W7／WN × 衛星 70／100% ＋名次加權，舊引擎，效率前緣描述）；S7／S8（底倉鎖定、單檔上限）；S38（衛星 50～80% × 底倉 0050 占比，描述性選擇表、不進判定、不加 N；底倉與空位一起換定存，不是「只空位留現金」）。'
                     '沒有 ÷8、÷12，也沒有「只空位留現金、底倉仍 0050」。',
 'seen_before_prereg（誠實揭露）': 'S45 FINDINGS／PREREG／s45lib／run.py（格式）、S17 FINDINGS（判定規則）、BACKLOG（S5：衛星 50% 與 100% 舊引擎的主跑數字；S6：W7_70、W10_100、WN 類主跑；S8：回撤中位數 ≈ −23.1 − 1.58 × 每檔權重，W7_70 −37.34、W10_100 −37.36 路徑回撤；'
                                '**S38：衛星 50／60／70／80% × 0050 100% 這 4 格（＝本題 S50_D10～S80_D10）的 20 條中位數與 Sharpe 1.32～1.38 已知，S38 用 EXRIGHT_FIX 引擎、未開 DEADLINE_FIX／INS15_MASK**）、deadline_engine_fix 現行格數字、data_check.json。'
                                '所以 ÷10 那一列的大致結果事前已知；÷8、÷12、留現金在任何引擎都沒看過。',
 'question': '現行配置在「合理配置」家族裡是不是剛好最好的？若是，報酬預期要打折。衛星比例本質是風險偏好（報酬與回撤一起動），所以另用 Calmar（報酬÷回撤）判斷風險調整後是否被挑過。',
 'family（事前寫死，不得增減）': {
  'sat_w': [0.5, 0.6, 0.7, 0.8, 1.0],
  'div': [8, 10, 12],
  'cells': '5 × 3＝15 格，id＝S{衛星%}_D{除數}；空位一律買 0050（現行）；現行＝S70_D10。',
  'idle_pair': '另加 S70_D10_CASH（空位留現金、底倉仍 0050）只和現行比，不進 15 格家族中位數與名次。',
  'what_changes': '只換 sat_w、每檔除數、空位處理；選股（SURGE STD、F1、L_MA、RET20）、出場 E0、換股日 A10、股池、嚴成本、5% 成交值上限、min_frac 0.5、帶狀 ±10、不鎖定、檔數上限 10 都不動。',
  'not_included': '不做 WN（名單檔數）與名次加權（S6 已做）、不做底倉鎖定（S7）、不做單檔上限（S8）、不做底倉換定存（S38）、不加衛星 < 50% 或 > 100%（槓桿）。',
 },
 'per_cell_runs': '20 條路徑（含 100 萬位移 0 主跑）；主跑另算成本加倍 2019 起年化。',
 'primary_statistics（事前指定）': {
  'P1 家族分布': '每格 20 條路徑 2019 起年化中位數（路徑中位數）；家族 15 個路徑中位數的中位數、最小、最大；15×20＝300 條路徑的中位數與第 10 百分位；300 條最大回撤中位數與最深 10%。',
  'P2 現行名次': '現行在 15 格依路徑中位數的名次（1＝最高）；依 20 條最大回撤中位數的名次（1＝最淺）；依 Calmar 中位數的名次（1＝最高）。',
  'P3 現行減家族中位數': '報酬：現行路徑中位數 − 15 個路徑中位數的中位數（百分點）。Calmar：現行 Calmar 中位數 − 15 格 Calmar 中位數的中位數，另報相對值（÷ 家族中位數）。',
  'Calmar 定義': '每條路徑 Calmar＝2019 起年化 ÷ |全期最大回撤|；每格取 20 條中位數。另報（次要）全期年化 ÷ |全期最大回撤|。',
  'P4 逐條配對': '每格 × 每條路徑（同資金、同位移）該格 − 現行：2019 起年化差的正條數與中位數；兩段（2019-01～2023-12、2024-01 起）差中位數；最大回撤差（正＝變淺）中位數與變淺條數；Calmar 差的正條數與中位數。',
  'P5 相對 0050 最長落後期': '每條路徑 rel＝淨值 ÷ 0050（2015-07-01 同起點），rel 低於先前最高點的最長連續交易日數（期末未收復也算）；每格報 20 條中位數，主跑另報起訖日。',
  'P6 次要（描述）': '主跑：全期、2024 起、逐年、四段空頭跌幅（2018、2020-03、2022、2024-07～2025-04）、最賺 5 筆占淨值增加、平均持股檔數、個股占淨值、閒置占淨值、成本加倍 2019 起。子家族名次：現行在 ÷10 那一列（5 個衛星比例）與 70% 那一行（3 個除數）各自的報酬與 Calmar 名次。'
 },
 'decision_rules（事前寫死，沿用 S17／S9／S45，另加風險調整口徑）': {
  '報酬口徑': 'P3 報酬 ≥ 3 個百分點 →「報酬口徑偏樂觀」；< 3 →「報酬口徑沒被挑過」。',
  '風險調整口徑': 'P3 Calmar 相對值 ≥ 10%（現行 Calmar 比家族中位數高一成以上）→「風險調整後偏樂觀」；< 10% →「風險調整後沒被挑過」。',
  '總判定': '(a) 兩口徑都沒偏樂觀 →「沒被挑過（不必因配置打折）」；(b) 只有報酬口徑偏樂觀 →「現行報酬高是承擔較多風險換來的（風險偏好），不算挑過；報酬預期不因配置打折，但必須和回撤一起報」；'
          '(c) 風險調整口徑偏樂觀（不論報酬口徑）→「現行配置偏樂觀」：報酬預期改用家族 300 條路徑中位數（附第 10 百分位），回撤預期改用 300 條回撤中位數（附最深 10%）。',
  '現行是不是剛好最好': '報酬名次與 Calmar 名次各自判：＝1 →「是」；2～4（前 1/4）→「前段」；其他 →「不是」。',
  '比現行好／差（每格，報酬）': '逐條配對 2019 起年化差：≥ 15／20 條為正而且中位數 ≥ +2 個百分點 →「比現行好」；≤ 5／20 條為正而且中位數 ≤ −2 →「比現行差」；其他 →「差不多」。'
                         '「比現行好」而且兩段差中位數都 > 0 →「穩定地比較好」；只有一段 > 0 →「比現行好但只靠一段」。',
  '風險調整（每格）': '逐條 Calmar 差：≥ 15／20 條為正而且中位數 ≥ +0.10 →「風險調整後較好」；≤ 5／20 條為正而且中位數 ≤ −0.10 →「風險調整後較差」；其他 →「風險調整後差不多」。',
  '風險註記（每格）': '報酬判「比現行好」類而且最大回撤差中位數 ≤ −2 個百分點（較深）→ 註明「報酬較高但回撤較深（風險換報酬），不算改進」。',
  '改進候選': '「穩定地比較好」且風險調整不是「較差」且回撤差中位數 > −2 → 只列為之後獨立題目的候選。',
  '空位處理（S70_D10_CASH 對現行）': '用同一套「比現行好／差」「風險調整」「風險註記」規則判；CASH 判「比現行好」→「空位買 0050 不是較好的選擇」；否則 →「空位買 0050 沒有吃虧」。',
  '預設': '不論結果，不改現行預設（家族檢查只判斷預期要不要打折）。'
 },
 'predictions（事前寫下，可被否證）': [
  '報酬名次：現行不是第 1（衛星 100%、÷8 最高）；報酬 P3 < 3（可能是負的，因為現行在衛星比例中段）。',
  '衛星比例越高、除數越小 → 報酬越高、回撤越深（大致單調）。',
  'Calmar：各格差距小（S38 Sharpe 1.32～1.38），現行 Calmar 相對家族中位數 < 10% →「風險調整後沒被挑過」。',
  '總判定 (a)「沒被挑過」。',
  'S70_D10_CASH：報酬「比現行差」（閒置約占淨值 1/3，0050 期間年化約 29%），回撤較淺，Calmar 差不多或較差 →「空位買 0050 沒有吃虧」。',
  '沒有任何格是「改進候選」。'
 ],
 'data_snooping': {'added': 13, 'note': '15 格＋留現金 1 格＝16，扣現行（已計入）、S50_D10（≈ S5 W50_T10）、S100_D10（＝S5／S6 W10_100）兩格已計入過 → +13。研究中心 N 依 BACKLOG 彙總另加；路徑與成本加倍不另計。不做 SPA／DSR。'},
 'data_check_summary': {k: dc['A0'][k] for k in ('avg_npos', 'avg_idle_frac', 'stockfrac_mean', 'sat_idle_frac_mean')},
}

(H / 'SPEC.json').write_text(json.dumps(SPEC, ensure_ascii=False, indent=1))
(H / 'SPEC.sha256').write_text(sha(H / 'SPEC.json') + '  SPEC.json\n')
PRE['spec_sha256'] = sha(H / 'SPEC.json')
(H / 'PREREG.json').write_text(json.dumps(PRE, ensure_ascii=False, indent=1))
(H / 'PREREG.sha256').write_text(sha(H / 'PREREG.json') + '  PREREG.json\n')
print(open(H / 'SPEC.sha256').read(), open(H / 'PREREG.sha256').read())

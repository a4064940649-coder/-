#!/usr/bin/env python3
"""S53 事前登記：在算任何非現行格的報酬之前寫 SPEC.json、PREREG.json 與各自的 .sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只算現行格：驗收 A0／A2／A3、官方淨值、B 目標波動常數、A 倍數分布與 B 規則套在現行 sret 上的描述；沒有算任何非現行格的報酬。）"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
for n in ('SPEC.json', 'PREREG.json'):
    assert not (H / n).exists(), f'{n} 已存在，拒絕覆寫'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
dc = json.loads((H / 'data_check.json').read_text())
assert dc['A0']['tol_ok'] and dc['A0']['bitwise_vs_deadline_ALL'] and dc['A2']['ok'] and dc['A3_all_ok'] and dc['official_nav']['ok']
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00')
TG = dc['B_target_vol']

SPEC = {
 'topic': 'S53 波動調整部位家族：規格（引擎、資料、驗收、資源）',
 'written_at': now,
 'engine_chain': 's53lib → s51lib → s47lib → s20lib → s17lib → s10lib → … → stlib → cslib（全部不改、只讀）；cslib.EXRIGHT_FIX＝True、DEADLINE_FIX＝True、INS15_MASK＝True；'
                 's17lib.e50_fix_suspension；SURGE＝rev_family STD；股池＝上市 & ~創新板；make_signal_s10(shift)（F1）；variant_S(E0)；嚴成本；cap_pct 0.05；min_frac 0.5；lock False；slots 10；sweep_min 2 萬；底倉 0050、空位買 0050。',
 'engine_copy': 'simulate_s53＝s51lib._SRC（s10lib.simulate_s10＋S47 ndiv/idle＋S51 n_sized 計數器）再 9 處字串替換（逐字見 s53lib._REPL）：新參數 wmult（A）、vt（B）；sret＝隔夜續抱持股的市值加權收盤對收盤日報酬（R 含息；當天新買、當天開盤賣出不算；無持股＝NaN）。'
                'wmult=None、vt=None 時與 simulate_s10 逐位相同（A3，第 0 步 20 條全過）。',
 'cells': {
  'CUR': '現行 W10_70：衛星 70%、每檔＝(淨值−底倉簿記)÷10、拉回帶 ±10 個百分點。',
  'A_D6／A_D8（個股層）': '新買進目標＝(淨值−底倉簿記)÷10 × m_j；訊號列 k 的 F1 名單 L（sig[k] & 股池[k]），σ_j＝個股 60 日含息日報酬標準差（R 列 k−60..k−1、有成交日 ≥ 40，用到訊號日前一日），'
                         'σ_ref＝名單 σ 中位數，m_j＝clip(σ_ref÷σ_j, 0.5, 10÷6 或 10÷8)；σ 缺值或名單有效 < 2 檔 → m＝1。衛星比例 70%、檔數上限 10、拉回規則不動（＝單檔上限 衛星÷6／÷8，下限 ÷20）。'
                         '不做預算中性重新正規化（低波動股可加碼到上限）。',
  'B60／B120（組合層，拉回帶 ±10）': '換股執行日 t、拉回前：σ̂＝sret[t−W..t−1] 有效值（≥ W÷3 天）的標準差 × √252；sat_w(t)＝clip(0.7 × 目標 ÷ σ̂, 50%, 90%)（有效天數不足＝70%），再走原本拉回（|衛星÷淨值 − sat_w(t)| > 10 個百分點才把底倉簿記設回 (1−sat_w)×淨值；衛星過多按比例賣股；衛星不足只調簿記，之後新買每檔＝(淨值−簿記)÷10 變大）。',
  'B60_R／B120_R（組合層，每次重設）': '同上但拉回帶＝0：每個換股執行日都把衛星重設到 sat_w(t)（衛星過多就按比例賣股）。',
  'targets': {'60': TG['60'], '120': TG['120'], 'how': '依題目指定：現行格 100 萬位移 0 主跑的 sret 套同一估計式，全期每日 σ̂ 的中位數（年化）。這是用到全期資料的常數（輕微前視，題目明定，揭露）；'
              '20 條路徑共用同一常數。'},
  'REF_S70_D8（參考，不進判定）': 'S47／S51 的「單純每檔加碼」：衛星 70%、每檔÷8（simulate_s53 ndiv=8）。只拿來對照「加碼的好處」大小，不增加 N。',
 },
 'data': '公開封存官方日 K（上市，含下市）、月營收、TWT49U、營收期限例外表與 15 日期限名單、0050 日 K。面板 T＝2876、N＝2142、最後一天 2026-10-22；淨值起算 2015-07-01。',
 'windows': '2019 起＝2018 最後交易日 → 2026-10-22；全期＝2015-07-01 起；分段 2019-01～2023-12、2024-01 起；最大回撤＝全期。',
 'paths': '20 條＝本金 {60, 80, 100, 150, 200} 萬 × 位移 {0,1,2,3}；同一位移所有格共用同一份訊號。',
 'acceptance': {
  'A0': '現行 100 萬位移 0：48.76%／34.40%／−32.12%／381 筆，與 deadline_engine_fix ALL 逐位相同（第 0 步已過）。',
  'A2': '現行 20 條與 deadline_engine_fix ALL 逐條差 0（第 0 步已過）。',
  'A3': 'simulate_s53(wmult=None, vt=None) 20 條逐位＝simulate_s10（第 0 步已過）；run.py 再驗一次 100 萬位移 0。',
  'official_nav': f"navs_official.csv.gz E0|W10_70 最大差 {dc['official_nav']['max_abs_diff_yuan']:.4f} 元（第 0 步已過）。",
 },
 'capacity': '1,000 萬、5% 當日成交值上限、全期連續、嚴成本、位移 0～3：S37 口徑 capped_rate＝被上限縮量的買進嘗試 ÷ 算出股數的買進嘗試（s51 計數器 n_sized、引擎 blk_cap）；報 4 個位移中位數與位移 0。',
 'resources': '單行程；python3 -I -B；峰值記憶體 ≤ 3GB（第 0 步 %d MB）；暫存 scratchpad/s53/（結束刪除）。' % dc['maxrss_mb'],
 'no_touch': '不改共用引擎與既有研究；xq_scripts 只讀；不碰 r51_*、s52_*、s54_*、trend-radar、o_g8p2a_protocol_addendum、live_tracking/main_forward、o_forward_infra_live；不用 git。',
 'src_sha256': dc['src_sha256'],
 'code_sha256_at_write': {n: sha(H / n) for n in ('s53lib.py', 'prep.py', 'prereg_write.py')},
}
(H / 'SPEC.json').write_text(json.dumps(SPEC, ensure_ascii=False, indent=1))
spec_sha = sha(H / 'SPEC.json')
(H / 'SPEC.sha256').write_text(f'{spec_sha}  SPEC.json\n')

PREREG = {
 'topic': 'S53：波動調整部位（Moreira & Muir 2017；Barroso & Santa-Clara 2015）能否在不加深回撤下拿到「每檔多押一點」的好處？家族全部列出、全部報告。',
 'written_at': now,
 'spec_file': 'SPEC.json（sha256 見 SPEC.sha256）',
 'status_at_write': '只跑過 prep.py（現行格驗收、B 目標常數、A 倍數分布、B 規則套在現行自己 sret 的衛星目標分布）。非現行格的報酬一個都沒算過。',
 'prior_work_check': {
  'grep': 'agents/BACKLOG.md、各題 FINDINGS／PREREG：grep「波動｜風險平價｜risk parity｜反波動｜1/σ｜目標波動｜vol target｜Moreira｜Barroso」。',
  'done_before': 'S32（s32_vol_weighting，2026-11-16）：衛星內每檔依 1/σ60（IV60）、1/σ²、反 β、高波動減半，**預算中性**（Σm＝檔數）、m 夾 [0.5, 1.5]（＝單檔 ÷20～÷6.67），σ 含訊號日收盤；訓練→驗證，IV60 訓練期 −2.22、全期描述 −3.22、回撤只淺 0.6、Calmar 變差；不通過。'
                 'momentum_family P6（選股用波動調整動能，不同題）；S30／S38（底倉換低 β／定存＝降曝險，不是依波動縮放）；strategy_ensemble（事後依波動縮放只當診斷）。',
  'not_done': '(1) 組合層「衛星依自身已實現波動縮放到目標波動」（B）從沒做過——本題主體；(2) A 的「不做預算中性、只設上限 ÷6／÷8」、σ 只用到訊號日前一日、用全期家族＋Calmar 判定口徑沒做過，但 A_D6 與 S32 IV60 很接近（上限 ÷6 對 ÷6.67、是否正規化），結果大致可預期，只當確認。',
 },
 'seen_before_prereg（誠實揭露）': 'S47 FINDINGS（15 格 20 條中位數、S70_D8 50.77%／−35.18%／Calmar 1.411、現行 46.40%／−33.39%／1.379）、S51 FINDINGS（S70_D8 驗證、容量 13.2%→18.9%）、S32 FINDINGS（IV60 結果）、S38、S6 FINDINGS 摘要；data_check.json（B 目標 60 日 ' + f"{TG['60']:.4f}" + '、120 日 ' + f"{TG['120']:.4f}" + '；B 規則套在現行 sret 上約 6 成換股日會超出 ±10 帶、約 1 成貼上限、1 成貼下限；A 倍數中位數 1、上限 ÷8 時約 22%、÷6 時約 10% 貼上限）。',
 'family（事前寫死，不得增減）': {
  'cells': ['A_D6', 'A_D8', 'B60', 'B120', 'B60_R', 'B120_R'],
  'current': 'CUR（W10_70）',
  'reference': 'REF_S70_D8 只描述，不進家族、不判定、不加 N。',
  'not_included': '不做 A×B 組合、不改衛星基準 70% 與上下限 50／90%、不做 1/σ²、不做反 β（S32 已做）、不做槓桿（衛星 > 100%）。',
 },
 'per_cell_runs': '20 條路徑（含 100 萬位移 0 主跑）；主跑另算成本加倍 2019 起；容量 1,000 萬 × 位移 0～3。',
 'statistics（事前指定）': {
  'S1': '每格 20 條：2019 起年化中位數、全期最大回撤中位數、Calmar 中位數（每條 Calmar＝2019 起年化 ÷ |全期最大回撤|）、全期年化中位數、兩段年化中位數。',
  'S2 逐條配對（格 − 現行，同資金同位移）': '2019 起年化差（正條數、中位數，百分點）；兩段差中位數；最大回撤差（正＝變淺）中位數與變淺條數；Calmar 差正條數與中位數；相對 0050 最長落後天數差中位數。',
  'S3 描述': '主跑：全期、逐年、四段空頭跌幅、最賺 5 筆占淨值增加、平均持股、個股占淨值、已平倉筆數、成本加倍；B 格：換股執行日實際衛星目標的平均／中位數、貼上下限比例、拉回（向上賣股／向下）次數；A 格：主跑買進平均倍數。',
  'S4 容量': '1,000 萬 capped_rate（4 位移中位數、位移 0）；與現行差（百分點）。',
  'S5 參考': 'REF_S70_D8 同樣的 S1／S2／S4（只描述）。',
 },
 'decision_rules（事前寫死）': {
  '主口徑 Calmar（每格）': '逐條 Calmar 差 ≥ 15/20 條為正且中位數 ≥ +0.10 →「Calmar 較好」；≤ 5/20 條為正且中位數 ≤ −0.10 →「Calmar 較差」；其他 →「Calmar 差不多」。',
  '輔口徑 報酬（每格）': '逐條 2019 起年化差 ≥ 15/20 條為正且中位數 ≥ +2 個百分點 →「報酬較好」（兩段差中位數都 > 0 →「穩定」，否則「只靠一段」）；≤ 5/20 且中位數 ≤ −2 →「報酬較差」；其他 →「報酬差不多」。',
  '回撤註記（每格）': '最大回撤差中位數 ≤ −2 個百分點 →「回撤較深」；≥ +2 →「回撤較淺」；其他 →「回撤相近」。',
  '題目目標「不加深回撤拿到加碼好處」（每格）': '報酬差 ≥ 15/20 條為正且中位數 ≥ +1 個百分點，而且最大回撤差中位數 ≥ −0.5 個百分點 →「達成」；否則「未達成」。',
  '後續候選（每格）': '「Calmar 較好」且報酬不是「較差」且回撤差中位數 > −1 個百分點 → 只列為之後訓練／驗證的候選；不改預設。',
  '家族總判定': '(a) 有任何格是後續候選 →「波動調整有值得驗證的候選」；(b) 沒有候選但有格「達成」→「有加碼好處但 Calmar 沒明顯改善，屬風險偏好」；(c) 都沒有 →「波動調整在本策略沒有改善風險調整報酬，不採用」。',
  '容量註記': '格的 capped_rate 中位數比現行高 ≥ 3 個百分點 →「容量變差」。',
  '預設': '不論結果，不改現行預設。',
 },
 'predictions（事前寫下，可被否證）': [
  'A_D6、A_D8：報酬差不多或較差（S32：反波動系統性少押高波動大贏家）；Calmar 差不多；不是候選。',
  'B 類：衛星多數時間在 ±10 帶內，B60／B120 與現行很接近（報酬、回撤差 < 1.5 個百分點），Calmar 差不多。',
  'B_R 類：每次重設會增加交易與成本，Calmar 差不多；2024-07～2025-04 段可能因前期波動高而降衛星、回撤略淺，但 2024 大漲段少賺。',
  '家族總判定 (c)。',
  'REF_S70_D8 全期 20 條中位數 ≈ 50.77%／−35.18%（同 S47，用同一引擎應逐位相同）。',
 ],
 'data_snooping': {'added': 6, 'note': '6 格（現行已計入；REF 已在 S47 計入）；路徑、成本加倍、容量不另計。不做 SPA／DSR。'},
 'spec_sha256': spec_sha,
}
(H / 'PREREG.json').write_text(json.dumps(PREREG, ensure_ascii=False, indent=1))
(H / 'PREREG.sha256').write_text(f"{sha(H / 'PREREG.json')}  PREREG.json\n")
print('SPEC', spec_sha); print('PREREG', sha(H / 'PREREG.json'))

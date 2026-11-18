#!/usr/bin/env python3
"""S70：寫 SPEC.json、PREREG.json 與各自的 sha256（只在 prep.py、train.py --dry、validate.py --dry 之後、任何候選報酬之前執行一次）。python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
assert not (H / 'SPEC.json').exists() and not (H / 'PREREG.json').exists(), '已登記，拒絕重寫'
DC = json.loads((H / 'data_check.json').read_text())
assert DC['A0']['bitwise_vs_deadline_ALL'] and DC['A2']['ok'] and DC['A1_nofin_identical']
now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
sha = lambda p: hashlib.sha256((H / p).read_bytes()).hexdigest()
step0 = {k: v for k, v in DC.items() if k.startswith('step0')}
SPEC = {
 'id': 'S70',
 'title': '營收驚奇名單加一道「獲利確認」能不能改善？（訓練→凍結→驗證一次）',
 'written_at': now,
 'question': 'S60 發現現行策略偏押低獲利、高投資的股票（FF 五因子 RMW β −0.71、CMA β −0.40）。Novy-Marx (2013, JFE)、Fama & French (2015, JFE) 指出獲利能力有溢酬。在營收驚奇名單上要求「公司真的有賺錢」，能不能改善？',
 'dedup（先查過 BACKLOG.md 與 agents/；以下做過、不重做）': {
  'S11（s11_pead_second_signal）＋ o_pead_margin_surprise_low_attention': '季報驚喜 SUE⊥（EPS 驚喜）當第二訊號：單獨 23%（0/20 贏 0050）、與現行各半 −10.03，不通過。屬「獲利驚喜」，不是獲利水準。',
  'w3_pead_ttm_eps_proxy／w3_revenue_sue／w4_buffett_roe_pe_screen／w4_roe_yield_compounder／w4_peg_revenue_growth／w1_canslim_proxy／o_ext_canslim_xs': '網路偵查的 EPS／ROE／CANSLIM 選股（獨立策略，不是現行名單上的濾網）：都沒過門檻。',
  'S13（s13_big_winner_features）': '名單內 6 個特徵（含毛利率年變化）預測大贏家：驗證不通過；毛利率變化 AUC 約 0.5。',
  'S20（s20_persistent_growth）': '持續成長（μ÷σ）濾網：訓練全負、驗證不通過；濾網把錢移去 0050 有機械性拖累；由衰轉盛是大贏家來源。',
  'S28（s28_margin_confirm）': '毛利率／營業利益率「年變化」禁買（BAN_GM2、BAN_GM5、BAN_GMOP、SWAP_GM2）：驗證 0/20、−7.02，比隨機剔除差；擋到 2408 等循環翻身股。測的是「變化」，沒測「水準／正負」。',
  'S29（s29_course_qmv）': 'Q＝ROE（近四季淨利 ÷ 權益）、M、V＝E/P：FLT_Q（名單內 ROE 最差 1/3 剔除）訓練 −2.41；FLT_V（E/P 在全市場最貴 1/5，含虧損股）訓練 −6.03；RANK_QM／RANK_QMV（名次重排）+0.02／−0.87；FILL_QMV 驗證 0/20、−3.90。事後描述「負 E/P（近四季淨利 < 0）的名單股」訓練 35 筆超額 −0.9%、驗證 31 筆 +13.0%（中位數 −0.8%）。',
  'S30（s30_defensive_base）': '低 β＋高 ROE 底倉：不通過。',
  'S31（s31_industry_neutral）': '產業內排名：不通過。',
  'S35（s35_turnaround_otc）': '谷底翻揚（含 ROE 低）上櫃樣本外：不通過（t 1.18）。',
  'S60（s60_factor_attribution）': '只做因子歸因（RMW／CMA 曝險），沒有測任何獲利濾網。',
  '題目原例 C2「排除最近四季合計 EPS ≤ 0」→ 判定為已做過，不重做': f"本題 data_check：名單中近四季淨利 ≤ 0 的 {DC['overlap_with_S29']['all']['EPS_TTM']['n']} 檔，有 {DC['overlap_with_S29']['all']['EPS_TTM']['in_s29_FLT_V_removed']} 檔就是 S29 FLT_V 剔除的股票、{DC['overlap_with_S29']['all']['EPS_TTM']['s29_negative_EP']} 檔就是 S29「負 E/P」描述組（訓練、驗證兩期的報酬都已經看過）。再測等於重做，而且驗證期不乾淨。",
  '結論（沒做過、且事前有文獻理由的）': 'C1 最近一季營業利益為負就不買（Fama-French 2015 的營業獲利；S28 只測年變化）；C2 毛利／資產低就不買（Novy-Marx 2013 的毛利獲利能力；本地從未用過毛利／資產）；C3 名單不變、買進順序改依毛利／資產（題目原例 C3）。',
 },
 'definitions': {
  '名單': '當月訊號名單＝股池（上市、排除創新板）中站上 60 日均線者依 SURGE 排前 10 名，其中 RET20 > 10% 者（make_signal_s10 的 sig；各位移各自重算）。',
  'C1': '名單中，訊號日當時可得的最近一季「單季營業利益 < 0」者不買。不遞補，空出的錢照引擎放 0050；營業利益缺值（多為金融保險格式）保留。已持有的股票照 E0 出場，不因本規則賣出。',
  'C2': '名單中，「毛利／資產」＝最近四季毛利合計（連續 4 季都有值）÷ 最近一季總資產，在同期股池（訊號列的上市非創新板股池、有值者）百分位 < 0.30（最低 30%，Fama-French 30／40／30 斷點的低組）者不買。不遞補；缺值保留。',
  'C3': '名單不變，只把買進順序由 SURGE 高到低改成「毛利／資產同期股池百分位」高到低（缺值當 0.5；同分依 SURGE）。順序只有在名額（最多 10 檔）或現金不夠時才影響買到哪幾檔。',
  '共同': '只改當月買進名單或順序；出場（E0：跌出 L_MA 前 20 名）、每檔金額（衛星 ÷ 10）、底倉、成本、股池全部不動。',
 },
 'data_and_pit': {
  'file': 'agents/o_pead_margin_surprise_low_attention/data/fin_long.parquet（MOPS XBRL 單季財報，2013 起；只讀不複製）＋ avail_dates.parquet（市場可得日 A）；同 S28／S29／S60 的來源。',
  'pit': '訊號日 d：季 kq 可用 ⇔ 法定期限 deadline(kq) < d 且（A 缺值或 A < d）。取法定期限已過的最新季；該檔該季晚交（A ≥ d）或沒有資料時往回 1 季；再沒有 → 缺值（保留）。次日開盤才進場。',
  'deadlines': '證券交易法第 36 條：Q1 5/15、Q2 8/14、Q3 11/14、年報（Q4）次年 3/31。金融保險與 -KY 的 Q2 為 8/31——A10 訊號日都在 9/10 之後，不影響（同 S28 既有處理）。A10 訊號日約每月 10 日，所以 5、8、11 月訊號日用的仍是前一季。',
  'age': f"名單上資料齡（訊號日距季底）中位數 {DC['step0_list_F1（訊號名單，特徵）']['all']['age_days_median']:.0f} 天、最多 {DC['step0_list_F1（訊號名單，特徵）']['all']['age_days_max']:.0f} 天。",
  'coverage_F1': {k: {f: v[f] for f in ('n', 'cov_opi', 'cov_roe', 'cov_gpa')} for k, v in DC['step0_list_F1（訊號名單，特徵）'].items()},
  'caveat': '封存可能含更正後版本（無法逐筆驗證首次申報值）；2017-04 前的 A 是推斷值（只用來排除晚交，不用來提前）。財報 2013 起，訓練期 2016 起，近四季合計夠用。',
 },
 'engine': {
  'chain': 's70lib → s47lib（不改；只借正式開關與匯入鏈）→ s20lib → s17lib → s10lib（全部不改）。引擎＝原函式 s10lib.simulate_s10，沒有副本、沒有改寫。',
  'flags': DC['flags'],
  'fixed': '出場 E0、嚴成本（strict）、股池只上市排除創新板、W10_70（衛星 70%、每檔＝衛星 ÷ 10）、最多 10 檔、帶狀拉回 ±10 個百分點、2 萬掃 0050、空位 0050、5% 成交值上限、目標一半不買。',
  'acceptance_done_before_prereg': {
   'A0': f"現行 100 萬位移 0：2019 起 {DC['A0']['since2019']*100:.2f}%、全期 {DC['A0']['full']*100:.2f}%、回撤 {DC['A0']['maxdd']*100:.2f}%、已平倉 {DC['A0']['n_closed']} 筆，與 deadline_engine_fix ALL（＝official_nav E0）逐位相同",
   'A1': '財報全缺值時 C1／C2／C3 的交易與淨值與現行逐位相同（缺值＝保留、順序＝SURGE）',
   'A2': '現行 20 條路徑與 deadline_engine_fix ALL 逐條差 0',
   'dry_run': 'train.py --dry、validate.py --dry（三個候選都換成現行）全部差 0，程式路徑正常',
   'C3_mech（只數買進集合，不看報酬）': {k: v for k, v in DC['C3_mech'].items() if k != 'paths'},
  },
 },
 'rule_effect_preview（只數檔數，位移 0）': {c: {k: v for k, v in DC['rule_effect_preview'][c].items() if '+' not in k or k.endswith('+0')} for c in ('C1', 'C2', 'C3')},
 'overlap（名單檔數）': dict(between_rules=DC['overlap_between_rules_F1'], with_S29=DC['overlap_with_S29']),
 'paths': '20 條＝起始資金 {60, 80, 100, 150, 200} 萬 × 訊號位移 {0, 1, 2, 3}',
 'windows': {
  'train': '連續模擬（全部訊號），量 2016-01-30 → 2021-12-30（＝訊號 2016-02～2021-12）的年化（stlib.cagr）與窗內最大回撤',
  'valid': '重新起跑：只保留營收月份 ≥ 2021-12 的訊號（2022-01-01 前無交易，全在 0050），起始資金 ÷ 2.558 使 2021-12-30 淨值約＝60～200 萬；量 2021-12-30 → 2026-10-22',
  'bear_segments_V4': '2022（2022-01-01～2022-12-31）、2024-07～2025-04（2024-07-01～2025-04-30），段內高低點的最大跌幅（s17lib.seg_dd）',
 },
 'step0_descriptive（寫 PREREG 前：特徵看全樣本，報酬只看訓練期）': step0,
 'step0_reading': '見 PREREG.step0_reading',
 'files': ['s70lib.py', 'prep.py', 'prereg_write.py', 'train.py', 'validate.py', 'data_check.json', 'f1_features.csv.gz', 'results_train.json', 'FROZEN.json', 'results.json'],
 'code_sha256_at_write': {p: sha(p) for p in ('s70lib.py', 'prep.py', 'prereg_write.py', 'train.py', 'validate.py')},
 'temp': '/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s70/（結束刪除）',
}
txt = json.dumps(SPEC, ensure_ascii=False, indent=1, default=str)
(H / 'SPEC.json').write_text(txt)
ssha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'SPEC.sha256').write_text(f'{ssha}  SPEC.json\n')
B = DC['step0_bought（現行實際買進的交易，特徵）']; UV = DC['step0_universe（同期股池，各訊號月平均）']; TR = DC['step0_train_returns_by_group（只看訓練期；現行主跑交易，報酬已扣成本）']
PRE = {
 'id': 'S70',
 'written_at': now,
 'candidates（事前固定，不得增減）': {
  'C1': '名單中最近一季（當時可得）單季營業利益 < 0 者不買；不遞補；缺值保留',
  'C2': '名單中毛利／資產（近四季毛利 ÷ 最近一季總資產）在同期股池最低 30% 者不買；不遞補；缺值保留',
  'C3': '名單不變，買進順序改依毛利／資產同期股池百分位由高到低（缺值 0.5；同分依 SURGE）',
 },
 'current': '現行：名單不看財報，依 SURGE 順序買、每檔＝衛星 ÷ 10、其餘放 0050',
 'step0_reading': (f"現行全期實際買進 {B['all']['n']} 筆：最近一季營業利益為負 {B['all']['loss_op_q']*100:.1f}%（同期股池 {UV['all']['loss_op_q']*100:.1f}%）、近四季淨利 ≤ 0 {B['all']['loss_ni_ttm']*100:.1f}%（股池 {UV['all']['loss_ni_ttm']*100:.1f}%）；"
                   f"營業利益率中位數 {B['all']['opm_q']['p50']:.1f}%（股池 {UV['all']['median_opm_q']:.1f}%）、ROE 中位數 {B['all']['roe_ttm']['p50']:.1f}%（股池 {UV['all']['median_roe_ttm']:.1f}%）、毛利／資產中位數 {B['all']['gpa']['p50']:.1f}%（股池 {UV['all']['median_gpa']:.1f}%）。"
                   f"訓練期較明顯：營業虧損 {B['train']['loss_op_q']*100:.1f}%（股池 {UV['train']['loss_op_q']*100:.1f}%）、ROE 低於股池中位數的占 {B['train']['roe_below_univ_median']*100:.0f}%；驗證期營業虧損 {B['valid']['loss_op_q']*100:.1f}%（股池 {UV['valid']['loss_op_q']*100:.1f}%）。"
                   f"訓練期報酬（已看）：營業虧損 {TR['營業利益<0（最近一季）']['n']} 筆平均超額 {TR['營業利益<0（最近一季）']['mean_ex']*100:+.1f}%（損益占比 {TR['營業利益<0（最近一季）']['pnl_share']*100:+.1f}%），"
                   f"不虧損 {TR['營業利益≥0']['n']} 筆 {TR['營業利益≥0']['mean_ex']*100:+.1f}%；毛利／資產最低 30% {TR['毛利／資產最低30%']['n']} 筆 {TR['毛利／資產最低30%']['mean_ex']*100:+.1f}%（含 1325 +345%，損益占比 {TR['毛利／資產最低30%']['pnl_share']*100:.0f}%）、"
                   f"中間 {TR['毛利／資產中間40%']['mean_ex']*100:+.1f}%、最高 30% {TR['毛利／資產最高30%']['mean_ex']*100:+.1f}%。"),
 'step1_train_selection': {
  'rule': '三個候選中，取訓練窗 20 條路徑年化中位數最高者（同分依 ≥ +0.25 路徑數、再依 C1→C3 順序）。若「其中位數 − 現行中位數 ≥ +1 個百分點」且「其回撤中位數 − 現行回撤中位數 ≥ −2 個百分點（不深於現行 2 個百分點以上）」→ 選它；否則選「不改」（不往下找第二名）。',
  'frozen_for_validation': '不論選它或「不改」，都凍結中位數最高的候選跑驗證期一次；選「不改」時驗證只作描述，總判定一律不通過。',
  'T0（S25 標準，訓練期）': '凍結候選對現行的成對差（逐條路徑）中位數 ≥ +1 個百分點，且差 ≥ +0.25 個百分點的路徑 ≥ 14/20。',
 },
 'step2_validation（凍結後只跑一次）': {
  'V1': '驗證窗年化差（凍結候選 − 現行，重新起跑）> 0 的路徑 ≥ 15/20',
  'V2': '驗證窗年化差中位數 ≥ +2 個百分點',
  'V3': '兩邊（現行與凍結候選的重新起跑主跑：100 萬換算、位移 0）各自最賺 5 筆已平倉交易的聯集，整段持有期兩邊都禁買，重跑 20 條路徑後年化差中位數 > 0',
  'V4': '2022、2024-07～2025-04 兩段空頭的段內最大跌幅差中位數都 ≥ −2 個百分點（不深於現行 2 個百分點以上）',
  'overall': '訓練選擇不是「不改」且 T0、V1、V2、V3、V4 全過 → 建議改成凍結候選；否則不改。',
  'also_report': '驗證窗最大回撤與 Calmar 的成對差；2022～2023、2024 起兩段年化差；成本加倍。',
 },
 'descriptive_only（登記但不作判定）': [
  '另兩個候選在驗證期的同樣 V1～V4 統計（V3 各自用與現行的聯集）',
  '全期連續 20 條路徑（2019 起、全期、回撤、Calmar、四段空頭）；各位移',
  '第 0 步報酬分組的全樣本與驗證期版（現行主跑交易）；被各候選擋掉的現行交易（筆數、平均報酬、損益合計、最賺與最賠 5 筆）',
  '隨機剔除對照（C1、C2）：每月從名單隨機剔除相同檔數 20 次（seed 70），看候選是否落在隨機範圍外',
 ],
 'predictions（事前）': {
  'P0': '很可能「不改」（約 85%）。',
  'P1': 'C1（營業虧損不買）：訓練期年化差約 −0.3～+0.8 個百分點。理由：訓練期營業虧損的買進只有 37 筆、平均超額 −1.7%、損益占比 −2.6%，拿掉它們多賺有限，到不了 +1。',
  'P2': 'C2（毛利／資產低不買）：訓練期年化差約 −1～−4 個百分點（剔除約四成買進，其中有 1325 +345%；S20／S29 的濾網機械拖累）。',
  'P3': 'C3（買進順序）：訓練期年化差 ±0.5 個百分點內（每條路徑只影響約 7 筆買進）。',
  'P4': '驗證期：凍結候選 V2（≥ +2）不過；整體通過機率 < 5%。若凍結 C1，驗證期年化差約 −4～0（S28、S29 已知驗證期的大贏家有循環翻身、帳面仍虧損的股票，如 2408）。',
  'P5': '第 0 步的驗證期版：營業虧損組平均超額不會比不虧損組差很多，甚至更高（由少數大贏家拉高），中位數為負。',
 },
 'caveat_validation_not_fresh': ('現行策略 2022～2026 的結果早已看過。C1 的驗證期不完全乾淨：名單中營業虧損的 82 檔有 53 檔（65%）屬於 S29 事後描述過的「負 E/P」組（驗證期 31 筆平均超額 +13.0%、中位數 −0.8%），'
                                 'S28 也已知 2408（2025-09，+297.5%）是毛利率仍在谷底的翻身股；所以對 C1 的方向已有先驗。C2、C3（毛利／資產）在任何期間都沒算過。'),
 'data_snooping': {'added': 3, 'note': '三個事前固定的候選，N +3（原例的「近四季 EPS ≤ 0」因與 S29 重複而不測、不計）。'},
 'spec_sha256': ssha,
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1, default=str)
(H / 'PREREG.json').write_text(txt)
psha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{psha}  PREREG.json\n')
print('SPEC', ssha); print('PREREG', psha); print(PRE['step0_reading'])

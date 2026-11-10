#!/usr/bin/env python3
"""凍結：把 S2 選定模型（iterations.json 的 selection.chosen）與 P1～P5、判定規則、重算規格寫進 results.json 的 PREREG（sha256＋written_utc）。
只寫一次。之後 s4_predict.py 才能產生 2019 預測與逐年殘差。
python3 -I -B s3_prereg.py"""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8')) if RJ.exists() else {}
assert 'PREREG' not in R, 'PREREG 只寫一次'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
ch = IT['selection']['chosen']; V = [v for v in IT['versions'] if v['version'] == ch][0]
PR = {
 'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
 'scope': '資料勘誤（不產生選股訊號、不改主策略規則；G3 族群 N 維持 2,573）',
 'file_sha256': {f: sha(HERE / f) for f in ['SPEC_draft.json', 'iterations.json', 'data/monthly_features.csv', 'data/universe_announce.csv',
                                            's1_lineage.json', 'nblib.py', 's2_features.py', 's2_explore_iterate.py', 'raw/lineage/disposal_official_universe.csv',
                                            'raw/lineage/disposal_hist_rejudge.csv']},
 'model_frozen': dict(version=ch, form=V['form'], x=V['x'], coef=V['coef'], alpha=V['alpha'], aic=V['aic'],
                      fit_sample='2006-01～2018-12、2020-01～2023-12（204 月；排除 2019）', y='universe 普通股且 in_panel、以 start 月份計數',
                      refit_rule='s4_predict.py 以同一程式在同一擬合樣本重新估計，係數須與此處一致（|Δ| < 1e-6）'),
 'intervals': {
   'primary_PI95': '參數抽樣（係數＋log α 多元常態）× NB 月抽樣，月份獨立，年總數 10,000 次模擬的 2.5～97.5 百分位（SPEC_draft 方法）',
   'parallel_PI95_annual_calibrated': '擬合樣本逐年標準化殘差平方平均 φ（迭代時已知殘差月自相關 0.35、逐年 |z|>2 的年份 4 個）→ 年變異 × φ，常態近似；凍結前宣告，和主區間並報',
   'no_param_unc': '不含參數不確定性的 NB 區間（描述）',
   'assumption_interval': '迭代 v2～v4（NB2）三版的 2019 點預測範圍',
   'yearly_residuals': '所有年份（2004～2026）報 O、E、主區間、z（(O−E)/模擬 SD）、校準 z（÷√φ）；殘差 < −2σ（主區間口徑）的年份進入 S3 交叉搜尋'},
 'predictions': {
   'P1': 'universe 2,218 列可用 (code, start) 一對一接回 announce；100% announce < start（領先 1～2 個交易日）；0 列 start ≤ announce；以公開證據界定「公告在當日盤後、早於成交日開盤」。機率約 90%。任一列 start ≤ announce 或接回率 < 99% → 否定。（註：S1 計數已在凍結前完成，P1 的計數部分不是事前檢驗，只有「公開證據界定盤後」部分是）',
   'P2': '2019 觀察值落在主 PI95 內（H_real），機率約 60%。否定（改判 H_inc）：低於下界且嚴格 CONFIRMED_MISSING ≥ 5。逐年殘差全部報告。',
   'P3': '若有缺漏：CONFIRMED_MISSING 集中在 2019-05、06、08、09（> 50%）。',
   'P4': '補上缺列後主策略受影響訊號日槽 ≤ 3；V7X 新正式 6 格 |Δ| ≤ 0.3pp、|Δt| ≤ 0.05 → 加註非取代，機率約 75%；影響集中 2019～2021，2006～2013 接近 0。',
   'P5': '2,093 vs 2,075 的 18 組差可用 ≤ 2 條定義規則解釋；需逐案特判 → 否定，列為不可驗證缺口。'},
 'S3_search_rules': {
   'status_table': '見 SPEC_draft.status_table（CONFIRMED_MISSING／SINGLE_SOURCE／NOT_FOUND／ALREADY_LISTED／NOT_COMMON_OR_TPEX）',
   'candidates': '(a) 2019 年及主口徑殘差 < −2σ 年份，本地語料中所有含「處置」字樣且可對應到上市普通股代碼或名稱的提及；(b) 同期面板近似觸發事件（K_reg 事件）中不在 universe 的股票日，抽樣上限 40 起／年依觸發強度排序，以 WebSearch 查證（查詢詞只用公司名＋「處置」＋年份，不含目標數字或特定日期）',
   'matching': '提及日 ±10 個交易日內 universe 有同代碼 start → ALREADY_LISTED',
   'strict': '只加 CONFIRMED_MISSING（主欄）', 'lenient': '另加 SINGLE_SOURCE（並列）',
   'adverse': '每格取嚴格、寬鬆兩版中對主策略最差者'},
 'S4_recompute': {
   'DISP_OFFICIAL_v2': 'data/DISP_OFFICIAL_v2.csv＝universe 2,218 列＋announce＋lag_td＋嚴格補列（補列 announce 取來源載明之公告日，否則取 start 前一交易日；end 取來源載明，否則 start＋9 個交易日）；寬鬆版 DISP_OFFICIAL_v2_lenient.csv',
   'paper': 'C0 與 ALLPIT_REAL_{opt,p90,carry} × cons_main，一般／壓力成本（o18lib.build／win／stats_pair）；處置 union 遮罩改讀 v2（o18lib.DU 置換）',
   'v7x': '題 1 v7x_lib.make 重建 REAL_{opt,p90,carry} × 13 j × 20 seeds 名單（v7x_lib.DU 與 DMT 置換為 v2）→ cz_v7x.CSim × K5T × GROW × cons_main；格：100 萬一般、100 萬壓力、300 萬一般、1,000 萬一般、2019 起（100 萬一般）、j p10（100 萬一般）＋30 萬一般（V7X 標準：2020-10-26 前盤後零股有量上限）＋30 萬整股限制（2020-10-26 前不允許零股）',
   'identity_shortcut': '若嚴格與寬鬆補列皆 0 且 P1 成立：DISP_OFFICIAL_v2 的處置覆蓋與現行完全相同，名單 sha 必須與官方逐位相同（程式斷言），Δ＝0 為構造結果；仍重跑 30 萬兩格與 100 萬一般 REAL_opt／p90／carry 一格作管線重現核對',
   'worst_cell': '逐格取 opt／p90／carry 最差（與 o18 新正式口徑相同）',
   'breakeven': '紙上 C0 與 ALLPIT_REAL 最差格附單邊滑價兩平點（超額歸零）與 t<1 點',
   'sampling_interval': '紙上全期超額 ±1.96 SE（月份分群 t 換算）與假設區間（放回 3 種 × 嚴格／寬鬆）並報',
   'attribution': '若取代：拆解為 (i) 時點修正、(ii) 2019 補列、(iii) 其他年補列'},
 'decision_rules': {
   'replace': '補正後主策略任一正式數字 |Δ| > 0.3pp 或 |Δt| > 0.05 → 取代（附歸因拆解）；否則加註',
   'H_real': '補列 0、模型支持 H_real → 加註「官方清單完整性與公告時點已核（2019 年低列數經模型與 N 個來源支持為真實）」，G5 移除「處置清單公告時點未核」',
   'H_inc_unfixed': '模型支持 H_inc 但嚴格補列 < 5 → 加註「2019 年疑似不完整、未能補正」，G5 維持 ✘',
   'P5': '18 組可規則化解釋 → 結案；否則 G5 不可驗證缺口',
   'main_strategy': '本題只改資料不改規則；主策略預期維持不可投資'},
 'o18_lessons': ['1 不利端逐格取最差', '2 凍結前不算正式口徑統計量（2019 預測、逐年殘差、主策略重算都在本 PREREG 之後）',
                 '3 搜尋後新增狀態和嚴格版並列、嚴格版為主欄', '4 raw/ 只放未修改原始檔，查詢詞不得含目標數字，摘要和原文分開計數',
                 '5 假設區間和抽樣區間並報', '6 取代正式數字附歸因拆解', '7 紙上數字附兩平點', '8 可執行測到 30 萬', '9 G6 狀態在複製後重新核對'],
 'o1_lessons': ['事件 t 按事件日分群（本題無事件報酬；主策略 t 沿用月份分群）', '驗證期污染：2019 是唯一保留年，凍結前未產生預測',
                '格點邊緣贏家詛咒：模型只 4 版、AIC 選版規則事前寫定', '成交可行性：處置排除以「公告日 ≤ 訊號日、成交日在處置期」為準',
                't 提升拆均值與波動（如有）', '存活者偏誤：面板 2019～2023 只有存活者，Y 用 in_panel 計數使特徵與因變數同母體']}
txt = json.dumps(PR, ensure_ascii=False, sort_keys=True)
R['PREREG'] = PR; R['PREREG_sha256'] = hashlib.sha256(txt.encode()).hexdigest()
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], PR['written_utc'])

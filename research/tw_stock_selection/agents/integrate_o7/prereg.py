#!/usr/bin/env python3
"""integrate_o7 事前登記：在任何整合回測之前，把組合、比例、選版規則、判定規則、前瞻登記、雜湊範圍寫進 results.json（含 sha256 與時間）。只能執行一次。"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
V7D = AG / 'o_v7_listing_exec_engine'
SPD = AG / 'o_main_spa_breaks'
RJ = HERE / 'results.json'
assert not RJ.exists(), 'results.json 已存在：PREREG 只能寫一次'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
V7R = json.loads((V7D / 'results.json').read_text(encoding='utf-8'))
assert V7R['PREREG_sha256'] == 'e84cf8bd00d3374e914fb3b168f2a79941ed59627f00bb11c06c42d3ccee6e71'

files = sorted(str(p) for p in HERE.glob('*.py')) + sorted(str(p) for p in (HERE / 'data').glob('*')) + [str(HERE / 'raw/access_log.json')]
files += [str(V7D / x) for x in ('results.json', 'frozen_spec.txt', 'data_freeze_v7.json', 'navs_v7.parquet', 'seed_navs_v7.parquet', 'navs_exec7.parquet',
                                 'plan_V7.csv', 'v7env.py', 'v7stats.py', 'v7common.py', 'data/plans_j7.csv', 'data/jscan_end_days.json')]
files += [str(SPD / 'results.json'), str(SPD / 'spa_lib.py'), str(SPD / 'fam_common.py')]
files += list(V7R['PREREG']['hashes'].keys())
files = sorted(set(files))
HASH = {f: sha(f) for f in files}

PREREG = {
    'study': 'integrate_o7：把 o7 唯一通過審查（待觀察）的 o_v7_listing_exec_engine（V7：以全市場上市身分為母體重建股池、修補颱風除權息與 2024-01 前視；'
             '可執行引擎逐名單模擬衝擊、處置期賣出、合併下市結算延遲）和主策略（排除金融 n10）整合。',
    'context': 'V7 不是新的選股訊號，而是同一條主策略的「資料版本＋可執行引擎」。o7 的兩題原創研究（2019 年後市場結構的新優勢來源）與第 (C) 題（收縮配置表＋序貫判定）都沒有通過審查，'
               '因此本輪沒有可以當濾網、加分項或第二條腿的新策略。可執行的整合只有：(1) 取代：主策略的正式數字改用 V7（R_V7）；'
               '(2) 在 V7 版主策略上重算 SOP 已出現過的固定比例配置（0050 20%／30%／50%、0050 核心 70%＋主策略衛星 30%、2330 30%），'
               '同時報零衝擊（官方日 40 seeds）與 V7 可執行引擎（j＝0～12、兩個衝擊模型、四個資金級距、FIX／GROW）兩種口徑；'
               '(3) 依總監指示，加報 2019-01～2022-12（P19_22）與 2023-01～2026-10（P23_26）兩段，兩段都已被 o5／o6／V7 看過，不是樣本外。',
    'base': {'AB_V7': 'V7 研究 navs_v7.parquet 的 AB_V7|normal／stress／EW（V7 面板、官方日、40 seeds 日報酬平均）。驗證：40 條逐 seed 淨值（seed_navs_v7）平均＝navs_v7（≤ 1e-9）；'
                      'seeds 0、1 用 v7env.load("v7") 引擎重跑＝seed_navs_v7（≤ 1e-9）。不重跑全部 40 seeds（V7 已有獨立複製，淨值層級通過）。',
             'AB_V7_B10': '同上，B10（缺營收的下市股營收百分位固定 1.0 放回；SOP 保守口徑）。',
             'AB_V4／AB_V6': '舊版（o_v4_revenue_patch_official、o_v6_price_panel_official），只當 R_V7 的對照。'},
    'candidates': {
        'R_V7': '取代：主策略的資料版本由 V4 換成 V7。判定規則見 judgement_rule.R_V7。',
        'M_E50_20': 'AB_V7 80%＋0050 20%（SOP「選股 80%＋0050 20%」敘述）。',
        'M_E50_30': 'AB_V7 70%＋0050 30%（SOP「想少落後 0050」選項）。',
        'M_E50_50': 'AB_V7 50%＋0050 50%。',
        'C_E50_70': '0050 核心 70%＋AB_V7 衛星 30%（總監指示「0050 核心＋新策略衛星」；新策略沒有通過審查，衛星只能是主策略本身。30% 取自 SOP 現行條件式提醒「主策略比重 ≤ 30%」與 V7 研究 core_satellite 表，不另挑）。',
        'M_2330_30': 'AB_V7 70%＋2330 30%（SOP 事後觀察欄；照 integrate_o6 建議，含單一事後贏家的組合不得走 (B) 路徑、不得 SPA 升級、不得被選版，最高只寫「事後觀察」）。',
        'mix_method': '袖子法（與 integrate_o4／o5／o6 同一函式）：兩腿各自漲跌；每個 AB_V7 成交日（plan_V7 exec）收盤調回固定比例，移轉金額扣成本：'
                      '股票腿用主策略成本（v4lib.cost_rates：一般＝手續費 0.1425%×0.6＋滑價 0.1%，賣方再加證交稅 0.3%；壓力＝K.rates("stress")）；0050 腿賣方＝買方＋0.1%。'
                      '2330 腿用 V7 面板的含息日報酬（K.GROW）。先在 V6 日曆上用同一函式重算 integrate_o6 的 M_E50_20／30／50（AB、B10 × 一般、壓力），逐日相對差 ≤ 1e-8。',
        'bases': '每個組合都做 AB 與 B10 兩個基底，各一般與壓力成本。',
    },
    'executable_mix': {
        'source': 'V7 研究 navs_exec7.parquet：{AB, B10} × j＝0～12（各 20 seeds 平均）× {K5T 上界, TICK 下界} × {一般, 壓力} × {30 萬, 100 萬, 300 萬, 1,000 萬} × {FIX, GROW}。',
        'method': '在每一條可執行淨值上，以該 j 的成交日（plans_j7.csv seed 0 的 exec 欄；各 seed 相同）收盤調回比例，用上面同一個袖子函式與 0050 混合，主策略比例 w ∈ {1.0, 0.8, 0.7, 0.5, 0.3}；'
                  '移轉成本一般用 cost_rates("normal")、壓力用 cost_rates("stress")。',
        'capital_mapping': '可執行淨值的資金級距解讀為「主策略那條腿的資金」，帳戶資金＝級距 ÷ w。例如 w＝0.3、300 萬級距＝帳戶 1,000 萬。'
                           'GROW 版的腿資金在原淨值中隨主策略單獨的報酬成長，混合後實際腿資金成長較慢 → 衝擊高估，對混合是保守的。0050 腿的零股、最低手續費不另模擬。',
        'stats': '每個 j：年化（淨值首日至該 j 的終點日）、最大回撤、同期 0050 年化、對 0050 的窗 t（格點＝該 j 的訊號日＋終點）、Y19on／P19_22／P23_26／HOLD／SEL 的年化與 t；'
                 '彙整成 j 中位數與 p10、贏 0050 的 j 數。先斷言 w＝1 時每個 j 的年化、t、回撤、0050 年化、t_Y19on 與 V7 RESULTS_EXEC.table 相同（< 1e-9），j 中位數與 jsumm 相同。',
        'matched_accounts': '事前固定兩組同帳戶比較（只有這兩組能用既有級距精確對上）：帳戶 100 萬＝全押（100 萬級距，w＝1）對主策略 30%（30 萬級距，w＝0.3）；'
                            '帳戶 1,000 萬＝全押（1,000 萬，w＝1）對主策略 30%（300 萬，w＝0.3）。每組報兩個模型的 j 中位數差與「混合較高的 j 數」；'
                            '兩模型的中位數差同號才寫「主策略 30% 較高」或「全押較高」，否則寫「無法判斷」。SOP 口徑＝B10、一般成本。',
    },
    'not_applicable': {
        'V7 當濾網或加分項': 'V7 是資料版本與執行引擎，沒有個股分數；選股規則一字不改。',
        'V4／V6／V7 帳戶按比例並行': '實盤只有一份價格；三者是同一策略的三種回測量測。',
        'o7 兩題原創研究（2019 年後新優勢來源）的濾網、加分、第二條腿': '沒有通過審查；本研究不讀、不用其淨值，避免把未通過的結果帶進整合。',
        'o7 第 (C) 題收縮配置表與序貫判定規則': '沒有通過審查；本研究的配置數字只是固定比例的描述，不是最適比例，也不用它的序貫規則做判定。',
        '其他比例（例如 0050 10%、40%、60%、90%）': '事前只固定上列比例；HOLD 與 2019 年後各段不參與任何選擇。',
        '處置股 20% 袖子': '處置股策略沒有 V7 版（上市身分與價格修補只做在主策略股池），重算不在本題範圍；SOP 的 80/20 處置配置維持原註記。',
        '開盤試撮（MIS）校準': 'o6 第 2 題因證交所網站在本環境被擋改列資料任務；本研究沿用 V7 的 K5T／TICK 兩個界，不另校準。',
    },
    'spec_fixed': {
        'return_type': '一律簡單報酬。',
        'annualization': '全期年化從 2006-04-11 收盤起算，到 2026-10-12 收盤（v7stats.cagr_mdd：段內第一個面板日起算）。',
        'windows': 't 用 V7 官方訊號窗（plan_V7 訊號日收盤 → 下一訊號日收盤；246 窗），窗依起點歸段。另報日曆月 t（完整月 2006-05～2026-09）與 HOLD 日曆歸屬（窗終點 ≥ 2024-01-02）。',
        'periods': 'v7stats.PER 的 S1 2006-04～2013、S2 2014～2020、S3 2021-03～2026-10-12、SEL 2006-04～2018、VAL 2019～2023、HOLD 2024～2026-10-12、Y19on 2019～2026-10-12、ALL；'
                   '另加 P19_22＝2019-01～2022-12、P23_26＝2023-01～2026-10-12（總監指定的挑選／驗證切分；兩段都已被看過）。',
        'paired_vs_main': '組合對 AB_V7（主策略單獨）的配對差：主指標日曆月，次指標 V7 訊號窗；附 MDE（2.8×sd/√n）與拿掉絕對值最大月後的 t。相關：訊號窗、日曆月、對 0050 月超額。',
        'R_V7_pairs': 'R_V7 對 AB_V4：V4 格點訊號窗（245 窗）與日曆月（2024-01、02 合併）；對 AB_V6：V7 格點窗（日期同 V6）與日曆月。',
        'EW': '對同池等權 t：AB 基底用 AB_V7|EW，B10 基底用 AB_V7_B10|EW。',
        'jump_check': '（o5 教訓 1）查 V7 日曆對加權指數交易日的缺漏與多出、AB_V7 日報酬 |r| > 10% 的日子、2024-01 區段最大日報酬，以及 AB_V7 與 AB_V6／AB_V4 在 2023-12-29 以前的最大相對差與第一個不同日（只報告，不斷言相同：V7 改了上市身分母體）。',
    },
    'metrics': ['各期年化、最大回撤、對 0050 t（訊號窗、日曆月、HOLD 日曆歸屬）、月均超額、追蹤誤差、對同池等權 t；壓力成本年化、t、回撤',
                '和主策略單獨（AB_V7）的配對差＋MDE、相關係數', '端點（v7stats.endpoint）、逐年與輸 0050 年數、40 seeds 的 t 分布（seed s 的 AB_V7 配同一條 0050／2330 腿；全期與 Y19on）',
                '保留期預測分布：照抄 V7 s5（Σlog((1+r)/(1+r_0050))，pre＝窗起點 ≥ 2006-04-01 且終點 < 2024-01-02，HOLD＝終點 ≥ 2024-01-02，平穩 bootstrap L3／6／12，B 10,000，種子 5150＋L；'
                '原始、中心收縮 50%、零超額三版 p_lower），每個組合對 0050 與對同池等權；先斷言 AB_V7、AB_V7_B10、兩者壓力的三版 p_lower 與 V7 研究完全相同。',
                'SPA：o_main_spa_breaks F_long（2,319）× W_long（V4 格點 245 窗），14 條 o_v4 成員換 V7 版（同 V7 研究），再加入 AB_V7 壓力成本與 5 個組合 ×（AB 一般、B10 一般、AB 壓力）共 16 條（N＝2,335）；'
                'Politis-Romano 變異，B 10,000，種子 20261028＋L；焦點 p（SPA_c、l、u、RC、StepM、單一檢定）與排名。先斷言 V7 研究員版的 AB_V7／B10／壓力焦點 p 與排名完全重現。'
                'V7 的獨立複製在 L6 差 0.021～0.022（超過 0.02 容許值，V7 採不利值）；本研究組合的 SPA p 另報「加上 V7 研究員與複製員最大差」的不利版。',
                '實盤預期：見 executable_mix。'],
    'selection_rule': '只用 SEL：在 M_E50_20、M_E50_30、M_E50_50、C_E50_70 中（M_2330_30 不得被選），選 SEL 對 0050 t 高於 AB_V7、而且 SEL 年化比 AB_V7 少 ≤ 1pp 者中 t 最高的一個；'
                      '都不符合就維持主策略單獨。HOLD、Y19on、P19_22、P23_26 不參與任何選擇。',
    'judgement_rule': {
        'R_V7': '取代（資料版本），需全部成立：(i) navs_v7 的 40 seeds 平均與 seeds 0、1 引擎重跑 ≤ 1e-9；(ii) V7 閘門 PASS；(iii) V7 資料複製（逐格＋淨值）通過；(iv) V7 可執行複製通過；'
                '(v) |Δ年化 ALL（AB_V7 − AB_V4）| ≤ 1pp。統計複製超出容許值時照 V7 規則取不利值，不阻擋取代。取代只改數字與判定欄，不改選股規格。'
                '級別依 V7 規則：AB 與 B10 的 L6 焦點 p（採 V7 不利值）都 ≤ 0.10 才可提議回第 2 級，否則維持第 3 級。實盤預期表照 V7 規則（B10 j 中位數、一般成本、兩界並列、兩模型結論相反寫「無法判斷」）。',
        '組合（對 AB_V7）': {
            '可用': '(A) ALL 日曆月配對 t ≥ 1.0，而且 SEL 與 VAL 的日曆月配對平均都 > 0；或 (B) ALL 對 0050 t 高 ≥ 0.2、SEL 與 VAL 的 t 都較高、ALL 年化少 ≤ 1pp、ALL 最大回撤不更深（含 2330 的組合不得走 (B)）。',
            '待觀察': '不符合可用，但 ALL 日曆月配對 t > −1、ALL 年化少 ≤ 2pp，而且 SEL 或 VAL 至少一段的對 0050 t 高於 AB_V7。',
            '不可用': '其餘（指「不能當報酬改善寫進 SOP」；配置選項本身的追蹤誤差、回撤數字照列）。',
            'cap': '上限待觀察（主策略已改版 ≥ 10 次、2019 年後資料都已看過、沒有盲測期）；含 2330 的組合上限「事後觀察」。',
            'SPA_grade': 'L6 SPA_c 焦點 p ≤ 0.05 第 2 級；0.05～0.10 第 2 級（邊緣）；> 0.10 第 3 級；含 2330 的組合不給級別。',
        },
        'hold': 'p_lower(L6，原始版) ∈ [0.10, 0.90] 寫「與雜訊相容」；< 0.10 寫「落後超出雜訊」；收縮 50%、零超額兩版並列。不得寫「衰退已證實」，除非對同池等權也落後超出雜訊。',
        'live_expectation': '實盤預期一律用 B10 可執行 j 中位數（一般成本），K5T、TICK 兩界並列；混合的結論只在兩界同向時才下，否則「無法判斷」。',
        'director_question_2019': '總監問題「2019 年後主策略是否滿足打敗 0050」：本研究只描述各組合在 Y19on、P19_22、P23_26 的年化、t 與可執行 j 中位數；這三段都已被看過，任何差異都不能當新證據，判定上限「待觀察」。',
    },
    'forward_registration': {
        'start': '2026-11 換股訊號（約 2026-11-11）起，到 live_tracking 的紙上淨值。',
        'lines': ['主策略單獨（排除金融 n10，實盤名單）', '0050 含息', 'M_E50_30（主策略 70%＋0050 30%，每月主策略成交日收盤調回）', 'C_E50_70（0050 70%＋主策略 30%，同法調回）'],
        'rule': '描述性追蹤，不做任何改版判定（o7 序貫判定規則沒有通過審查）。每 12 個月報一次各線累積報酬、對 0050 的累積 log 超額與最大回撤；'
                '若主策略單獨的前瞻累積 log 超額低於本研究保留期預測分布（AB_V7、L6、原始版）按月數等比例縮放的 q10，於報告中標註「低於歷史 10% 分位」，但不據此改 SOP。',
        'note': '比例不會因前瞻結果調整；前瞻期也不用來挑比例。',
    },
    'effective_tests_note': '本研究新增：5 個組合 × 2 基底 × 2 成本 × 10 期間的描述統計；判定只用 5 個組合（AB 一般成本）的事前規則與 SPA 焦點；SPA 族群由 2,319 增為 2,335。'
                            '可執行混合 2 基底 × 2 模型 × 2 成本 × 4 級距 × 2 版本 × 5 比例 × 13 j，全部報告，不挑。主策略歷來改版 ≥ 10 次、V7 研究另試 6 個資料版本，有效檢定數以 SPA 族群為準。',
    'honesty_notes': ['V7 研究已公開 AB_V7、AB_V7_B10 的全期、分段、保留期、SPA 與可執行表；integrate_o4／o5／o6 已公開 V4／V6 版 0050／2330 混合。組合結果大致可由公開數字推得，本研究沒有新的樣本外證據。',
                      'V7 研究的 SUMMARY_posthoc.core_satellite 已報過 70/30、50/50、30/70 的零衝擊年化與追蹤誤差（我在寫 PREREG 前讀過）；本研究的比例沒有因此調整。',
                      'PREREG 寫入前，我用 --smoke 模式（1 seed 重跑、B=200、j 只跑 2 個）跑過 run.py 只測管線，輸出寫到 scratchpad，沒有印出或檢視任何績效數字。',
                      '2019 年後的所有數字（P19_22、P23_26、Y19on、HOLD）都已在 o5～o7 被看過。'],
    'validation': ['navs_v7 的 AB_V7、AB_V7_B10（一般、壓力）＝40 條 seed 淨值平均 ≤ 1e-9；seeds 0、1 引擎重跑＝seed_navs_v7 ≤ 1e-9。',
                   '統計函式重現 V7 RESULTS_RUN.table 的 AB_V7、AB_V7_B10、AB_V7 壓力（ALL、SEL、VAL、HOLD、Y19on 的年化、t、回撤、對等權 t）< 1e-9。',
                   '袖子函式在 V6 上重算 integrate_o6 的 M_E50_20／30／50（AB、B10 × 一般、壓力）≤ 1e-8。',
                   '保留期三版 p_lower（AB_V7、B10、兩者壓力；L3／6／12；對同池等權 L6）與 V7 研究完全相同。',
                   'SPA 焦點 p 與排名（AB_V7、B10、壓力；L3／6／12）與 V7 研究員版完全相同；AB_V7／B10 窗超額與族群成員差 < 1e-12。',
                   '可執行 w＝1 每格與 V7 RESULTS_EXEC.table、jsumm 差 < 1e-9。', 'data/ 副本與原檔 sha256 相同。'],
    'files': {'prereg.py': '本檔', 'run.py': '驗證、組合、統計、保留期、SPA、可執行 j 混合、選版、判定，寫回 RESULTS（只寫一次）', 'data/': '直接讀取的淨值、族群、計畫檔副本',
              'raw/access_log.json': '沒有外部抓取'},
    'run': 'python3 -I -B prereg.py && python3 -I -B run.py',
    'inputs': {'o_v7_listing_exec_engine_PREREG_sha256': V7R['PREREG_sha256']},
    'sha256': HASH,
}
blob = json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()
R = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(blob).hexdigest(), 'PREREG_written_at': pd.Timestamp.now(tz='UTC').isoformat()}
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_written_at'], len(HASH), 'files hashed')

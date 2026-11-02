#!/usr/bin/env python3
"""integrate_o8 事前登記：在任何整合績效數字之前，把組合、比例、統計口徑、選版規則、判定規則、前瞻登記與雜湊範圍寫進 results.json（含 sha256 與時間）。只能執行一次。"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
XD = AG / 'o_exec_engine_rewrite_v7x'
O7 = AG / 'o_v7_listing_exec_engine'
I7 = AG / 'integrate_o7'
RJ = HERE / 'results.json'
assert not RJ.exists(), 'results.json 已存在：PREREG 只能寫一次'


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()


XR = json.loads((XD / 'results.json').read_text(encoding='utf-8'))
assert XR['PREREG_sha256'] == 'faacf2bde2f13242dec52ca5ba5efa8e1eb0c000939cf584f27334b59f94a956'
files = sorted(str(p) for p in HERE.glob('*.py')) + sorted(str(p) for p in (HERE / 'data').glob('*')) + [str(HERE / 'raw/access_log.json')]
files += [str(XD / x) for x in ('results.json', 'frozen_spec.txt', 'x_data.py', 'x_stats.py', 'data/x_cache.npz', 'out/navs_oos_R4.parquet', 'out/navs_oos_R4c.parquet',
                                'out/navs_oos_R0.parquet', 'out/gate_zero.json')]
files += [str(O7 / 'navs_exec7.parquet'), str(O7 / 'data/plans_j7.csv'), str(O7 / 'data/jscan_end_days.json'), str(O7 / 'plan_V7_B10.csv'), str(O7 / 'plan_V7.csv'),
          str(I7 / 'results.json')]
files = sorted(set(files))
HASH = {f: sha(f) for f in files}

PREREG = {
    'study': 'integrate_o8：把 o8 唯一通過審查的 o_exec_engine_rewrite_v7x（V7X：V7 可執行引擎依 model_spec7.json 獨立重寫，修正 FIX 縮放成本、處置期成交上限、'
             '2020 前零股量上限、開盤量前視四項偏樂觀設定；引擎「可用」、策略「待觀察」）和主策略（排除金融 n10）整合。',
    'context': 'V7X 不是選股訊號，而是主策略的「可執行量測」（逐名單模擬實際下單、衝擊、零股、處置、提領注資成本）。本輪第 1 題（營收訊號新鮮度）、第 2 題（0050 骨架選股 2.0）、'
               '第 4 題（前瞻序貫規則）都沒有通過審查，因此沒有可以當濾網、加分項或第二條腿的新選股策略。可執行的整合只有：'
               '(1) R_V7X：SOP「實盤預期」表由 V7 原可執行表換成 V7X 正式表（R4＝四項修正全開）；'
               '(2) 在 V7X 可執行淨值上重做主策略與 0050 的固定比例混合（80/20、70/30、50/50、0050 核心 70%＋主策略 30%），報各期間年化、回撤、對 0050 t、'
               '對主策略單獨的月差配對 t 與相關；(3) 事前固定的兩組同帳戶資金配對（全押 vs 主策略 30%），這是唯一和實際決策（同一筆錢怎麼放）對應的比較。',
    'base': {
        'ALONE': 'V7X out/navs_oos_R4.parquet：AB_V7_B10（主策略排除金融 n10、B10 保守口徑：缺營收的下市股營收百分位固定 1.0 放回）× j＝0～12（各 20 seeds 日報酬平均）'
                 '× {K5T（正式）, TICK（敏感度）} × {一般, 壓力} × {30 萬, 100 萬, 300 萬, 1,000 萬} × {FIX, GROW}。V7X 只跑 B10，沒有 AB 版；本研究不另跑引擎。',
        'R4c': 'V7X 事後修正版（F2 處置賣出款隔日補買；s5c_posthoc_f2.py），只當敏感度，不參與判定。',
        'R0': 'V7X 原設定（四項修正全關），只用於 R_V7X 的挑選期對帳與不利值調整。',
        'V7ORIG': 'V7 原可執行表 o_v7_listing_exec_engine/navs_exec7.parquet（B10 欄），只用於重現 integrate_o7 的可執行混合表（驗證本檔函式）與對照。'},
    'candidates': {
        'R_V7X': '取代：SOP 實盤預期表換成 V7X 正式表（R4，B10、一般成本、13 個換股日 j 中位數；K5T 主欄、TICK 敏感度並列）。',
        'M_E50_20': '主策略 80%＋0050 20%（w＝0.8）。', 'M_E50_30': '主策略 70%＋0050 30%（w＝0.7；SOP「想少落後 0050」選項）。',
        'M_E50_50': '主策略 50%＋0050 50%（w＝0.5）。', 'C_E50_70': '0050 核心 70%＋主策略 30%（w＝0.3；SOP 現行追蹤誤差預算選項）。',
        'mix_method': '袖子法（與 integrate_o4～o7 同一函式 sleeve）：兩腿各自漲跌；每個 j 的主策略成交日（plans_j7.csv，AB_V7_B10、seed 0 的 exec 欄）收盤調回固定比例，'
                      '移轉金額扣成本：股票腿＝cost_rates（一般：手續費 0.1425%×0.6＋滑價 0.1%，賣方再加證交稅 0.3%；壓力：買賣各 0.5%）；0050 腿賣方＝買方＋0.1%。'
                      '淨值在該 j 的終點日（jscan_end_days.json）截斷。0050 用 V7X 快取 E50（先斷言與 etf_0050.csv 相同）。',
        'capital_mapping': '可執行淨值的資金級距解讀為「主策略那條腿的資金」，帳戶資金＝級距 ÷ w（例：w＝0.3、300 萬級距＝帳戶 1,000 萬）。'
                           'FIX 版主策略腿在每個換股日重設到級距（F1 起提領注資要付成本），混合後的移轉又再扣一次成本 → 對混合保守。GROW 版腿資金在原淨值中隨主策略單獨成長，'
                           '混合後實際腿資金成長較慢 → 衝擊高估，對混合保守。0050 腿的零股、最低手續費不另模擬。',
        'matched_accounts': '事前固定兩組（只有這兩組能用既有級距精確對上，與 integrate_o7 相同）：帳戶 100 萬＝全押（100 萬級距，w＝1）對主策略 30%（30 萬級距，w＝0.3）；'
                            '帳戶 1,000 萬＝全押（1,000 萬，w＝1）對主策略 30%（300 萬，w＝0.3）。'},
    'not_applicable': {
        'V7X 當濾網或加分項': 'V7X 是可執行引擎，沒有個股分數；選股規則一字不改。',
        'V7X 當第二條腿': '它量測的就是主策略本身，不是另一條報酬來源。',
        'o8 其他三題（營收新鮮度、0050 骨架 2.0、前瞻序貫）': '沒有通過審查；本研究不讀、不用其淨值或名單。V7X 的 STEP6_APPLY 也因這兩題沒有名單而未套用。',
        '其他比例（0050 10%、40%、60%、90% 等）': '事前只固定上列比例；HOLD 與 2019 年後各段不參與任何選擇。',
        '處置股 20% 袖子': '處置股策略沒有 V7／V7X 可執行版，不在本題範圍；SOP 的處置配置維持原註記。',
        'AB（非 B10）可執行版': 'V7X 只跑 B10；AB 的可執行數字仍是 V7 原表（integrate_o7），不混用。'},
    'spec_fixed': {
        'return_type': '簡單報酬；每格先算 13 個 j 各自的統計，再取 j 中位數（主欄）與 p10（o7 教訓：判定一律用 j 中位數）。',
        'periods': 'ALL＝該 j 第一個訊號日收盤至終點日；SEL＝至 2018-12-31；VAL＝2019-05-01～2023-12-31；HOLD＝2024-01-01 起；Y19on＝2019-01-01 起；'
                   'P19_22＝2019-01-01～2022-12-31；P23_26＝2023-01-01 起。年化、t 的口徑與 V7X x_stats.stats 相同（先斷言 w＝1 的 j 中位數重現 V7X STEP_TABLES R4 32 格，差 < 1e-9）。',
        't_vs_0050': '主指標：該 j 訊號日窗（訊號日收盤→下一訊號日收盤）超額對 0050 的 t，窗依起點歸段；另報日曆月 t（2006-05～2026-09）與追蹤誤差。',
        'maxDD': '全期與每一段的段內最大回撤（段內重新起算），並列同段 0050 回撤。',
        'paired_vs_alone': '同腿級距：混合 − 單獨（同 j、同級距、同版本、同模型、同成本）的日曆月報酬差，每段的平均與 t（取 j 中位數）；另報 13 個 j 平均月差序列的 t（jpool）與 MDE（2.8×sd/√n）。'
                           '同帳戶：混合（小級距 w＝0.3）− 單獨（大級距 w＝1）同法。',
        'correlation': '混合與單獨的月報酬相關、月超額（對 0050）相關（ALL、Y19on），以及各淨值與 0050 的月報酬相關。',
        'validation': ['V1：本檔袖子＋統計在 V7 原表（navs_exec7 B10）上重現 integrate_o7 RESULTS.live_exec_j 全部 B10 格（5 個 w × 2 模型 × 2 成本 × 4 級距 × 2 版本）的 '
                       'CAGR、t、maxDD、0050 年化、t_Y19on、Y19on 年化、HOLD 年化 j 中位數，差 < 1e-9。',
                       'V2：w＝1 在 R4 重現 V7X STEP_TABLES.R4 的 32 格（CAGR、t_0050、maxDD、exc_Y19on、CAGR_SEL、exc_VAL、t_VAL、CAGR_HOLD、E50_CAGR），差 < 1e-9。',
                       'V3：w＝1 在 R4c 重現 V7X FORMAL_V7Xc_posthoc（CAGR_K5T、t_med），差 < 1e-9。',
                       'E50 快取＝etf_0050.csv（< 1e-12）；plans_j7 seed 0 的訊號日＝V7X SIGD；data/ 副本 sha256 與原檔相同。']},
    'selection_rule': '只用 SEL，對 SOP 主欄 8 格（R4、B10、K5T、一般成本 × 4 級距 × FIX／GROW）逐格：在 M_E50_20／30／50、C_E50_70 中，選 SEL 對 0050 t（j 中位數）'
                      '高於單獨、而且 SEL 年化比單獨少 ≤ 1pp 者中 SEL t 最高的一個；都不符合就維持主策略單獨。VAL、HOLD、Y19on、P19_22、P23_26 不參與任何選擇。',
    'judgement_rule': {
        'R_V7X': '取代，需全部成立：(i) 本檔 w＝1 重現 V7X R4 表 ≤ 1e-9；(ii) V7X 零衝擊閘門（out/gate_zero.json 全部 ≤ 1e-9）；(iii) R0 對 V7 原表挑選期 SEL 年化 j 中位數，一般成本 16 格最大差 ≤ 0.10pp（本檔獨立重算）；'
                 '(iv) V7X VERDICT.engine 為「可用」。取代時：SOP 表用 R4（凍結版，不用事後 R4c）；並列「不利值」欄＝R4 − max(0, R0 − V7 原表) 的 j 中位數差（V7X RECON.diff_pp），'
                 '即 V7X 原設定比 V7 原表樂觀的部分全部扣回。「贏 0050」照 V7X 規則：j 中位數超額 > 0、t ≥ 1.5、2019 年起超額 ≥ 0 同時成立；超額 ≤ 0 寫「輸」；其餘「無可辨識差異」。',
        '組合（對主策略單獨，同腿級距；R4、B10、K5T、一般成本，8 格各自判，全部用 j 中位數）': {
            '可用': '(A) ALL 月差配對 t ≥ 1.0，而且 SEL 與 VAL 的月差平均都 > 0；或 (B) ALL 對 0050 t 高 ≥ 0.2、SEL 與 VAL 的對 0050 t 都較高、ALL 年化少 ≤ 1pp、ALL 最大回撤不更深。',
            '待觀察': '不符合可用，但 ALL 月差配對 t > −1、ALL 年化少 ≤ 2pp，而且 SEL 或 VAL 至少一段的對 0050 t 高於單獨。',
            '不可用': '其餘（指「不能當報酬改善寫進 SOP」；追蹤誤差、回撤數字照列）。',
            'overall': '8 格中 ≥ 5 格達到某等級（含以上）即給該等級，否則較低一級；上限「待觀察」（主策略已改版 ≥ 10 次、2019 年後資料都已看過、沒有盲測期）。'},
        '同帳戶配對': '兩個模型（K5T、TICK）的 j 中位數年化差同號才寫「主策略 30% 較高」或「全押較高」，否則「無法判斷」（同 integrate_o7）；'
                      '另標強度：兩模型的月差配對 t（j 中位數）都 ≥ 2 或都 ≤ −2 才寫「顯著」，否則「只是點估計」。SOP 口徑＝R4、一般成本；壓力成本並列。',
        'director_question_2019': '「2019 年後是否打敗 0050」：只描述 Y19on、P19_22、P23_26、HOLD；這些段都已被看過，任何差異不能當新證據，判定上限「待觀察」。'},
    'metrics': ['每格（run × 模型 × 成本 × 級距 × 版本 × w）：各段年化、0050 同段年化、超額、對 0050 t（訊號窗）、各段最大回撤與 0050 回撤、日曆月 t、追蹤誤差、與 0050 相關；'
                'j 中位數與 p10、超額為正的 j 數。', '混合 vs 單獨（同腿級距）：各段月差平均與 t、MDE、jpool t、月報酬與月超額相關。',
                '同帳戶配對：各段年化差 j 中位數、混合較高的 j 數、各段月差配對 t、相關。', 'R_V7X：SOP 表（K5T、TICK、不利值、Y19on、壓力、V7 原表對照）。',
                '零衝擊參考：直接引用 integrate_o7 的 B10 一般成本混合統計（不重算）。'],
    'forward_registration': {
        'start': '2026-11 換股訊號起，到 live_tracking 的紙上淨值。',
        'lines': ['主策略單獨（排除金融 n10，實盤名單）', '0050 含息', 'M_E50_30', 'C_E50_70'],
        'rule': '描述性追蹤，不做改版判定（本輪第 4 題序貫規則沒有通過審查）。每 12 個月報告各線累積報酬、對 0050 的累積 log 超額與最大回撤；'
                '實際成交（距開盤幾跳、占開盤量比例）累積 ≥ 60 筆後，用 V7X s8_forward_calib.py 校準 K5T 係數 κ，再回頭更新本表的實盤預期（只更新量級，不改判定規則）。',
        'note': '比例不因前瞻結果調整；前瞻期不用來挑比例。'},
    'effective_tests_note': '新增描述統計：4 runs（R4、R4c、R0、V7ORIG）× 2 模型 × 2 成本 × 4 級距 × 2 版本 × 5 比例 × 13 j，全部報告不挑；判定只用 R4、K5T、一般成本 8 格的事前規則。',
    'honesty_notes': ['寫 PREREG 前我讀過 V7X 的 FORMAL_V7X、STEP_TABLES、RECON、DECOMP 與 integrate_o7 的 0050 混合與同帳戶配對結果（含「100 萬全押較高、1,000 萬無法判斷」），'
                      '比例與規則沿用 integrate_o7，沒有因此調整。',
                      'PREREG 寫入前，我用 --smoke（j 只跑 0、6）測過管線，並在 scratchpad 跑過 V1／V2 驗證（只印出差值：V1 最大 2.2e-13、V2 0），沒有印出或檢視任何混合的績效數字。',
                      '混合 0050 的數學後果大致可由公開數字推得（同腿級距下混合＝等比例稀釋）；真正未知的是同帳戶配對在 V7X 四項修正下是否改變 integrate_o7 的結論。',
                      '2019 年後的所有數字都已在 o5～o8 被看過，不是樣本外。'],
    'files': {'prereg.py': '本檔', 'run.py': '驗證、混合、統計、同帳戶配對、選版、判定，寫回 RESULTS（只寫一次）', 'data/': 'plans_j7.csv、jscan_end_days.json、etf_0050.csv 副本',
              'raw/access_log.json': '沒有外部抓取'},
    'run': 'python3 -I -B prereg.py && python3 -I -B run.py',
    'inputs': {'o_exec_engine_rewrite_v7x_PREREG_sha256': XR['PREREG_sha256']},
    'sha256': HASH,
}
blob = json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()
R = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(blob).hexdigest(), 'PREREG_written_at': pd.Timestamp.now(tz='UTC').isoformat()}
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_written_at'], len(HASH), 'files hashed')

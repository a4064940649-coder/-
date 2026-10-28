#!/usr/bin/env python3
"""integrate_o6 事前登記：在任何整合回測之前，把組合、比例、選版規則、判定規則、雜湊範圍寫進 results.json（含 sha256 與時間）。只能執行一次。"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
V6D = AG / 'o_v6_price_panel_official'
RJ = HERE / 'results.json'
assert not RJ.exists(), 'results.json 已存在：PREREG 只能寫一次'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
V6R = json.loads((V6D / 'results.json').read_text(encoding='utf-8'))
SPR = json.loads((AG / 'o_main_spa_breaks/results.json').read_text(encoding='utf-8'))
assert V6R['PREREG_sha256'] == '609f42d0064130d492cb931d267df15b48e104607061f260ce51266478e282a6'
assert SPR['PREREG_sha256'] == 'ab57019a7397f5e47c751792b81de5eb90a2d293361f347a49edd4d55e61d75f'
PV4 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')

files = sorted(str(p) for p in HERE.glob('*.py')) + sorted(str(p) for p in (HERE / 'data').glob('*')) + [str(HERE / 'raw/access_log.json')]
files += [str(V6D / x) for x in ('results.json', 'frozen_spec.txt', 'data_freeze_v6.json', 'navs_v6.parquet', 'seed_navs_v6.parquet', 'navs_jscan_v6.parquet',
                                 'plan_V6.csv', 'v6env.py', 'v6stats.py', 'v6common.py')]
files += [str(AG / 'o_main_spa_breaks/results.json'), str(AG / 'o_main_spa_breaks/spa_lib.py'), str(AG / 'integrate_o4/navs.parquet'),
          str(AG / 'o_exec_capacity_tiers/navs_abavg.parquet'), str(AG / 'o_exec_capacity_tiers/data/bench.parquet'),
          str(PV4 / 'close_raw.parquet'), str(PV4 / 'ret.parquet')]
files += list(V6R['PREREG']['hashes'].keys())
files = sorted(set(files))
HASH = {f: sha(f) for f in files}

PREREG = {
    'study': 'integrate_o6：把 o6 唯一通過審查（待觀察）的 o_v6_price_panel_official（V6 價格面板：補 2024-01 日 K 缺口、以官方日 K 取代 2024-02～2025-09 的 5 分 K 近似）和主策略（排除金融 n10）整合',
    'context': 'V6 不是新選股訊號，而是同一條主策略的「價格資料版本」。實盤上 V4 與 V6 是同一個帳戶、同一張名單（差異只在回測用的價格量測），'
               '所以「V6 當濾網／加分項」「V4 帳戶與 V6 帳戶按比例並行」都沒有意義（見 not_applicable）。可執行的整合是：'
               '(1) 取代：主策略的正式數字與判定欄改用 V6（R_V6＝AB_V6）；(2) 在 V6 版主策略上，重算 SOP 已列出的配置選項——主策略＋0050 的 80/20、70/30、50/50，'
               '以及主策略 70%＋台積電 30%（SOP 事後觀察欄）——因為這些選項的數字原本都建在 V4 上，而且 2024～2026 正好是 V6 改動的區間。'
               '比例全部沿用 SOP／integrate_o4 已出現過的固定值，不另挑。',
    'base': {'AB_V4': '舊正式版（o_v4_revenue_patch_official 的 AB_V4|normal），只當 R_V6 的對照。',
             'AB_V6': 'V6 研究的 AB_V6|normal（V6 面板、官方日、只用已公告營收、40 seeds 日報酬平均）。本研究用 V6 研究的引擎接線（v6env.load("v6")）重跑 40 seeds，'
                      '斷言 normal／stress／EW 平均淨值與 navs_v6.parquet、40 條逐 seed 淨值與 seed_navs_v6.parquet 逐日相對差 ≤ 1e-9。AB_V6_B10 同法重跑（normal／stress／EW）。'},
    'candidates': {
        'R_V6': '取代：主策略的資料版本由 V4 換成 V6（AB_V6）。判定規則見 judgement_rule.R_V6。',
        'M_E50_20': 'AB_V6 帳戶 80%＋0050 20%（SOP「選股 80%＋0050 20%」敘述）。',
        'M_E50_30': 'AB_V6 70%＋0050 30%（SOP「想少落後 0050」選項）。',
        'M_E50_50': 'AB_V6 50%＋0050 50%。',
        'M_2330_30': 'AB_V6 70%＋2330 30%（SOP 事後觀察欄；o4 已判定為事後觀察）。',
        'mix_method': '袖子法（與 integrate_o4／o5 同一函式）：兩腿各自漲跌；每個 AB_V6 成交日（官方換股日隔天）收盤調回固定比例，移轉金額扣成本：'
                      '股票腿用主策略成本（買＝手續費 0.1425%×0.6＋滑價 0.1%；賣＝同＋證交稅 0.3%）；0050 腿賣方＝買方＋0.1%（ETF 證交稅）。'
                      '2330 腿用 V6 面板的含息日報酬（K.GROW）。先在 V4 日曆上用同一函式重算 integrate_o4 的 L_E50_20／30／50、L_2330_30（normal／stress），逐日相對差 ≤ 1e-9。',
        'bases': '每個組合都做兩個基底：AB（AB_V6）與 B10（AB_V6_B10，SOP 引用的保守口徑），各一般與壓力成本；可執行版用 V6 研究的 j 分布淨值（見 live_expectation）。',
    },
    'not_applicable': {
        'V6 當濾網或加分項': 'V6 是價格資料版本，沒有個股分數；選股規則一字不改。',
        'V4／V6 兩帳戶按比例並行（例如 80/20、50/50）': '實盤只有一份價格；V4 與 V6 是同一策略的兩種回測量測，混合沒有可交易的意義。',
        '其他比例（例如 0050 10%、40%、2330 20%／50%）': '事前只固定上列 4 種（都是 SOP 或 integrate_o4 已出現過的值）；不另挑。HOLD（2024～2026）不參與任何選擇。',
        '處置股 20% 袖子': '處置股策略沒有 V6 版（其 2024～2025 價格同樣受 5 分 K 近似影響），重算不在本題範圍；SOP 的 80/20 處置配置維持原註記。',
        '核心衛星最適比例與勝率': 'o6 第 4 題（收縮估計）沒有通過審查，本研究不回答「該放多少」，只把 SOP 既有選項的數字換到 V6。',
        '容量／零股校準': 'o6 第 3 題沒有通過審查；可執行版沿用 V6 研究的 o5 移植公式（未校準），結論只能寫條件式。',
    },
    'spec_fixed': {
        'return_type': '一律簡單報酬。',
        'annualization': '全期年化從 2006-04-11 收盤起算，到 2026-10-12 收盤。分段年化用 v4lib.seg（段內第一個面板日的前一日收盤起算），所以 HOLD＝2023-12-29 收盤起算（SOP 口徑）。',
        'windows': 't 用 V6 官方訊號窗（plan_V6 訊號日收盤 → 下一訊號日收盤；246 窗），窗依起點歸段（v4lib.PER）。另報日曆月 t（完整月 2006-05～2026-09）與 HOLD 日曆歸屬（窗終點 ≥ 2024-01-02，v6stats）。',
        'periods': 'v4lib.PER：S1 2006-04～2013、S2 2014～2020、S3 2021-03～2026-10-12、SEL 2006-04～2018、VAL 2019～2023、HOLD 2024～2026-10-12、ALL。',
        'paired_vs_main': '組合對 AB_V6（主策略單獨）的配對差：主指標日曆月，次指標 V6 訊號窗；附 MDE（2.8×sd/√n）與拿掉絕對值最大月後的 t。相關：訊號窗、日曆月、對 0050 月超額。',
        'paired_R_V6_vs_AB_V4': 'R_V6 對 AB_V4：主指標 V4 格點訊號窗（245 窗；AB_V4 在 2024-01-02～02-14 沒有價格，V6 格點多出的 2024-01 窗無法配對）；次指標日曆月，2024-01 與 2024-02 合併成一格。',
        'EW': '對同池等權 t：AB 基底用 AB_V6|EW，B10 基底用 AB_V6_B10|EW。',
        'jump_check': '（o5 教訓 1）先查淨值跳日：AB_V6 日報酬 |r| > 10% 的日子、2024-01 區段最大日報酬、V6 日曆對加權指數交易日的缺漏；並斷言 AB_V6 與 AB_V4 在 2023-12-29 以前逐日相同（相對差 ≤ 1e-12）。',
    },
    'metrics': ['各期年化、最大回撤、對 0050 t（訊號窗、日曆月、HOLD 日曆歸屬）、月均超額、追蹤誤差、對同池等權 t；壓力成本年化、t、回撤',
                '和主策略單獨（AB_V6）的配對差＋MDE、相關係數', '端點（拿掉最後 1 窗、最好 1／3／5／10 窗、最好年份、2006、起點 2007／2008、只到 2025、日曆月）',
                'AB 逐 seed 分布（seed s 的 AB_V6 配同一條 0050／2330 腿）', '成本損益兩平滑價（0.10%～0.40%，每 0.02%）', '逐年與輸 0050 年數',
                '保留期預測分布：照抄 V6 s5 演算法（Σlog((1+r)/(1+r_0050))，pre＝窗起點 ≥ 2006-04-01 且終點 < 2024-01-02，HOLD＝終點 ≥ 2024-01-02，平穩 bootstrap L3／6／12，B 10,000，種子 5150＋L），'
                '每個組合對 0050 與對同池等權；先斷言 AB_V6、AB_V6_B10、AB_V6 壓力的 p_lower 與 V6 研究完全相同',
                'SPA：o_main_spa_breaks F_long（2,319）× W_long（V4 格點 245 窗），14 條 o_v4 成員換 V6 版（同 V6 研究），再加入 AB_V6 壓力成本與 4 個組合 ×（AB 一般、B10 一般、AB 壓力）共 13 條；'
                'Politis-Romano 變異，B 10,000，種子 20261028＋L；焦點 p（SPA_c、l、u、RC、StepM、單一檢定）與排名。先斷言 V6 研究的 AB_V6／AB_V6_B10／壓力焦點 p 與排名完全重現。',
                '實盤預期（零衝擊數字不進 SOP）：V6 研究的 j＝0～12 淨值（AB_V6_B10 為 SOP 口徑，AB_V6 並列）加上 o5 移植的可執行拖累（30 萬／100 萬／300 萬／1,000 萬 × FIX／GROW），'
                '組合在每個 j 上用同一袖子法（調整日＝官方 AB_V6 成交日）；報 j 中位數與 p10 的年化、回撤中位數、對 0050 超額中位數。先斷言 x＝100% 時與 V6 研究 RESULTS_JSCAN 完全相同。'],
    'selection_rule': '只用 SEL：在 M_E50_20、M_E50_30、M_E50_50、M_2330_30 中，選 SEL 對 0050 t 高於 AB_V6、而且 SEL 年化比 AB_V6 少 ≤ 1pp 者中 t 最高的一個；都不符合就維持主策略單獨。HOLD 不參與任何選擇。',
    'judgement_rule': {
        'R_V6': '取代（資料版本），需全部成立：(i) 本研究重跑 AB_V6／AB_V6_B10 與 navs_v6 逐日相對差 ≤ 1e-9；(ii) 2023-12-29 以前與 AB_V4 逐日相同；(iii) V6 研究閘門 PASS；'
                '(iv) V6 研究獨立複製通過（面板 ≥ 99.9%、淨值 ≤ 1e-8、SPA 與保留期差 ≤ 0.02）；(v) |Δ年化 ALL| ≤ 1pp（否則先拆到逐月逐檔）。'
                '取代只改「數字與判定欄」，不改選股規格；不看方向。策略級別依 V6 規則：AB 焦點 p ≤ 0.10 且 B10 焦點 p ≤ 0.10 才可提議回第 2 級，否則維持第 3 級。',
        '組合（對 AB_V6）': {
            '可用': '(A) ALL 日曆月配對 t ≥ 1.0，而且 SEL 與 VAL 的日曆月配對平均都 > 0；或 (B) ALL 對 0050 t 高 ≥ 0.2、SEL 與 VAL 的 t 都較高、ALL 年化少 ≤ 1pp、ALL 最大回撤不更深。',
            '待觀察': '不符合可用，但 ALL 日曆月配對 t > −1、ALL 年化少 ≤ 2pp，而且 SEL 或 VAL 至少一段的對 0050 t 高於 AB_V6。',
            '不可用': '其餘（指「不能當報酬改善寫進 SOP」，配置選項本身的追蹤誤差／回撤數字照列）。',
            'cap': '上限待觀察：主策略已改版 ≥ 10 次、2019 年後資料已看過、沒有盲測期。',
            'SPA_grade': 'L6 SPA_c 焦點 p ≤ 0.05 第 2 級；0.05～0.10 第 2 級（邊緣）；> 0.10 第 3 級。含 2330 的組合，低 p 只反映台積電 2006～2026 對 0050 的事後超額，不構成選股力證據（o5 已說明）。',
        },
        'hold': 'p_lower(L6) ∈ [0.10, 0.90] 寫「與雜訊相容」；< 0.10 寫「落後超出雜訊」；不得寫「衰退已證實」，除非對同池等權也落後超出雜訊。',
        'live_expectation': '實盤預期一律用 AB_V6_B10 可執行版 j 中位數（V6 規則）；可執行拖累是未校準的 o5 移植公式，所以資金分級的結論只寫條件式。',
    },
    'effective_tests_note': '本研究新增檢定：4 個組合 × 2 基底 × 2 成本 × 7 期間的描述統計；判定只用 4 個組合（AB 基底一般成本）的事前規則與 4 個 SPA 焦點（AB 一般）；SPA 族群因此由 2,319 增為 2,332 條。'
                            '主策略歷來改版 ≥ 10 次、V6 研究另試 8 個資料版本，有效檢定數以 SPA 族群為準。',
    'honesty_notes': ['V6 研究已公開 AB_V6、AB_V6_B10 的全期、分段、保留期與 SPA 數字；integrate_o4／o5 已公開 V4 版 0050／2330 混合。組合結果大致可由公開數字推得，本研究沒有新的樣本外證據。',
                      'PREREG 寫入前，我用 2 seeds、B=200 的 --smoke 模式跑過 run.py 只測管線（輸出寫到 scratchpad，沒有印出或檢視任何績效數字；修了兩個只在 smoke 下才會觸發的斷言）。',
                      'V6 研究的 replicate_o6 資料夾本研究沒有讀；複製是否通過只讀 V6 results.json 的 SUMMARY_posthoc.replication.pass。'],
    'validation': ['AB_V6、AB_V6_B10 normal／stress／EW 重跑與 navs_v6 逐日相對差 ≤ 1e-9；AB_V6、AB_V6_B10 40 條逐 seed 淨值與 seed_navs_v6 ≤ 1e-9。',
                   '袖子函式在 V4 上重算 integrate_o4 的 L_E50_20／30／50、L_2330_30（normal／stress）≤ 1e-9。',
                   '保留期 p_lower（AB_V6、B10、壓力；L3／6／12；對同池等權 L6）與 V6 研究完全相同。',
                   'SPA 焦點 p 與排名（AB_V6、B10、壓力；L3／6／12）與 V6 研究完全相同；本研究 AB_V6／B10 窗超額與族群成員差 < 1e-9。',
                   '可執行 j 分布（x＝100%）每個 j、每個資金級距與 V6 研究 RESULTS_JSCAN 差 < 1e-9。', 'data/ 副本與原檔 sha256 相同。'],
    'files': {'prereg.py': '本檔', 'run.py': '重跑、組合、統計、保留期、SPA、可執行 j 分布、選版、判定，寫回 RESULTS（只寫一次）', 'data/': '直接讀取的淨值、族群、計畫檔副本',
              'raw/access_log.json': '沒有外部抓取'},
    'run': 'python3 -I -B prereg.py && python3 -I -B run.py',
    'inputs': {'o_v6_price_panel_official_PREREG_sha256': V6R['PREREG_sha256'], 'o_main_spa_breaks_PREREG_sha256': SPR['PREREG_sha256']},
    'sha256': HASH,
}
blob = json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()
R = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(blob).hexdigest(), 'PREREG_written_at': pd.Timestamp.now(tz='UTC').isoformat()}
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_written_at'], len(HASH), 'files hashed')

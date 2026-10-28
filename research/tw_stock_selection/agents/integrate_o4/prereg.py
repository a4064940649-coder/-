#!/usr/bin/env python3
"""integrate_o4 事前登記：在任何整合回測之前，把組合、比例、選版規則、判定規則寫進 results.json（含 sha256 與時間）。只能執行一次。"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
RJ = HERE / 'results.json'
assert not RJ.exists(), 'results.json 已存在：PREREG 只能寫一次'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
V4D = AG / 'o_v4_revenue_patch_official'
v4res = json.loads((V4D / 'results.json').read_text(encoding='utf-8'))
assert v4res['PREREG_sha256'] == '3797d0fdf9606e3cb3e32cb34329dccd54a2e0b12dddc474a4015252e57cff4e'

PREREG = {
    'study': 'integrate_o4：把 o4 唯一通過審查（待觀察）的 o_v4_revenue_patch_official（主策略單一正式數字 V4）和主策略、既有配置選項整合',
    'context': 'o4 另外三題（o_hold_buffer_fresh 換股緩衝帶、o_stagger_tranche 分批換股、o_buyback_gate_core_rebal 庫藏股／替代題）沒有通過審查，不納入整合。'
               'V4 不是新選股訊號，而是主策略本身在 V3 資料（官方處置名單、下市股價格 P、併購出清 X）＋營收修補 R_FIX 上的單一正式版本，判級基準是 AB（只用已公告營收）。'
               '因此本整合的意義是：(a) 以 AB_V4 取代 main_v2 當「主策略單獨」的基準，報告新舊基準的差；'
               '(b) 把 V4 研究產生、實盤可寫成規則的選擇（處置日口徑）當成濾網測；'
               '(c) 在 V4 上重新確認 SOP 既有的配置與風控選項（0050／2330 第二條腿固定比例、新高家數減碼）。'
               '「V4 與 main_v2 按比例混合」不測：兩者是同一策略在兩版資料上的回測，實盤只能執行一版（實盤差異只剩處置口徑，已由濾網候選涵蓋）。',
    'base': {
        'AB_V4': 'o_v4_revenue_patch_official 的 AB_V4|normal（40 條 mc 未公告抽樣的日報酬平均）＝主策略單獨、判定與選版基準。本研究重跑 AB_V4（同 seed 0～39），斷言與 navs_v4.parquet 的 AB_V4|normal 逐日相對差 ≤ 1e-9。',
        'V4': 'V4|normal（BASE，換股日完整營收），次要口徑。',
        'MAIN_V2': 'main_v2/navs_gapfix.parquet 的 EXFIN|n10|normal（舊正式 28.15%），只用來報告新舊基準的差。',
        'O3REP': 'V4 研究的 O3REP（o3 官方名單版 26.95%），只報告。',
    },
    'candidates': {
        'F_SIG': '濾網（處置口徑）：處置改用訊號日口徑（公告日 ≤ d 且 start ≤ d ≤ end；＝現行 SOP 文字「換股日當天在處置期間」），其餘同 V4，AB 40 seeds。',
        'F_DUNION': '濾網（處置口徑最嚴）：訊號日口徑或成交日口徑任一成立就排除，其餘同 V4，AB 40 seeds。',
        'O_BREADTH30': '風控疊加（SOP 既有待觀察選項，研究報告 13.19）：訊號日 d 的流動性前 300 池（V60 ≥ 1,000 萬、收盤 ≥ 10 的前 300）中 52 週比 ≥ 0.95 的檔數 B；'
                       '主策略曝險＝min(1, B/30)，其餘現金（報酬 0）；在成交日 e 收盤調到目標比例，移轉金額扣成本。主策略袖子＝AB_V4。參數 30、0.95 沿用 13.19，不調。',
        'L_E50_20': '第二條腿：AB_V4 80%＋0050 含息 20%，每個成交日 e 收盤調回比例（袖子法，移轉扣成本：股票賣方含證交稅 0.3%，0050 賣方含證交稅 0.1%）。',
        'L_E50_30': 'AB_V4 70%＋0050 30%，同上（SOP 既有選項「30% 放 0050」）。',
        'L_E50_50': 'AB_V4 50%＋0050 50%，同上。',
        'L_2330_20': 'AB_V4 80%＋2330 20%，同上（2330 含息日報酬＝面板 GROW）。',
        'L_2330_30': 'AB_V4 70%＋2330 30%（SOP 既有待觀察選項）。',
        'L_2330_50': 'AB_V4 50%＋2330 50%。',
    },
    'ratios_fixed': '比例 80/20、70/30、50/50 事前固定，不在任何期間挑；HOLD（2024～2026）不參與任何選擇。',
    'sleeve_rule': '袖子法：兩條腿各自的日報酬（主策略腿已扣自身換股成本）在每個成交日 e 收盤調回目標比例；從股票腿移出 m 元的成本＝m×(股票賣方費率＋對方買方費率)，'
                   '移入股票腿＝m×(對方賣方費率＋股票買方費率)；0050 賣方費率＝買方費率＋0.1%（ETF 證交稅）。驗證：同一函式套用在 o_largecap_leg 的 MAIN|normal 上，'
                   '必須重現其 MAIN70_0050_30|normal（最終淨值相對差 ≤ 1e-6）；套用 2330 30% 時與 o_largecap_leg 的單一帳戶 TSMC30|normal 比較並報告差距（只報告，不斷言）。',
    'not_tested': {
        'V4 與 main_v2 混合': '同一策略兩版資料，實盤不可執行；處置口徑差異由 F_SIG／F_DUNION 涵蓋。',
        'B05／B10／B_Q 界限': '是資料界限，不是實盤規則（實盤營收是完整的）；只在結論引用 V4 研究的數字。',
        '處置股策略 20%（SOP 現行配置）': '處置股策略只有 2021 起的真實名單，沒有 2006 起可比的日淨值，不在 V4 框架內整合；SOP 配置不受本研究影響。',
        '緩衝帶、分批換股、庫藏股': '本輪沒有通過審查。',
    },
    'cost': {'normal': '買＝手續費 0.1425%×0.6＋滑價 0.1%；賣＝同＋證交稅 0.3%（v4lib.cost_rates("normal")）',
             'stress': '每邊 0.5%（主策略腿用 AB_V4|stress；袖子移轉用同費率，0050 賣方再加 0.1%）',
             'breakeven': '單邊滑價格點 0.10%～0.40%（每 0.02%），主策略腿用同滑價重跑 AB（40 seeds），袖子移轉也用同滑價；報 ALL 對 0050 t 跌到 2.30 與 2.00 的滑價（線性內插）。'},
    'periods': 'v4lib.PER：S1 2006-04～2013、S2 2014～2020、S3 2021-03～2026-10-12、HOLD(=S4) 2024～2026-10-12、SEL 2006-04～2018、VAL 2019～2023、ALL 2006-04～2026-10-12。'
               't＝訊號窗（訊號日收盤→下一訊號日收盤）超額，窗起點落在區間內。',
    'metrics': ['各期年化、最大回撤（日頻）', '對 0050 含息 t、月均超額、追蹤誤差（訊號窗超額標準差×√12）', '對同池等權 t（用 AB_V4|EW）',
                '和 AB_V4（主策略單獨）、MAIN_V2 的訊號窗月報酬差：平均 pp、配對 t、非零月數', '和 AB_V4 的月報酬相關、月超額（對 0050）相關',
                '逐年、輸 0050 的年份', '壓力成本', '端點敏感度：拿掉最後 1 個月、最好 1／3／5 個月、最好年份，只算到 2025 年底，日曆月 t',
                '成本損益兩平滑價（t 2.30／2.00）', 'Bonferroni p（m=10 改版次數；另報 m=19＝10＋本研究 9 個候選）', '實盤預期＝0050 年化＋0.5×（回測年化 − 0050 年化）'],
    'selection_rule': '只用 SEL：在 F_SIG、F_DUNION、O_BREADTH30、L_2330_20、L_2330_30、L_2330_50 中，選 SEL 對 0050 t 最高、高於 AB_V4、而且 SEL 年化比 AB_V4 少 ≤ 1pp 的一個為「建議整合版」；'
                      '都不符合就維持 AB_V4。L_E50_* 不參與選版：混入基準本身時，對 0050 的 t 在數學上近似不變（超額按比例縮小、標準差同比例縮小），只當「追蹤誤差預算」選項報告。',
    'judgement_rule': {
        '可用': '下面兩條路之一成立：(A) 報酬改善：ALL 和 AB_V4 的月差配對 t ≥ 1.0，而且 SEL 與 VAL 的月差平均都 > 0；'
                '(B) 追蹤誤差改善：ALL 對 0050 t 比 AB_V4 高 ≥ 0.2、SEL 與 VAL 的對 0050 t 都高於 AB_V4、ALL 年化少 ≤ 1pp、ALL 最大回撤不深於 AB_V4。',
        '待觀察': '不符合可用，但 ALL 月差配對 t > −1、ALL 年化少 ≤ 2pp，而且 SEL、VAL 至少一段的對 0050 t 高於 AB_V4。',
        '不可用': '其餘。',
        '配置選項': 'L_E50_*：不判可用與否，只報告追蹤誤差、回撤與年化代價。',
        'caps': {
            '全部候選': '上限待觀察：主策略已改版 ≥ 10 次、2019 後資料已經看過、沒有盲測期；任何升級只能用 2026-11 起的前瞻資料。',
            'F_SIG／F_DUNION': '另註：兩種處置口徑只在少數月份名單不同，必須報非零月數與拿掉最大月後的 t（o3 方法教訓 3）。',
            'L_2330_*': '沿用 o1／o2 總監判定（待觀察）：改善的是追蹤誤差，來源是台積電 2006～2026 的事後超額。',
            'O_BREADTH30': '沿用 13.19：參數曾用 2006～2013 定、2014～2020 沒通過事前判定，最多待觀察。',
        },
    },
    'honesty_notes': ['AB_V4、V4、V4_SIG（BASE）的全期數字已在 o_v4_revenue_patch_official 公開；F_SIG 的 AB 版、F_DUNION、O_BREADTH30 與各袖子組合在 V4 上的數字是新算的，'
                      '但所有成分（0050、2330、新高家數減碼、處置口徑）都已在前幾輪看過全期結果，本研究沒有新的樣本外證據，只是用事前固定的組合方式與判定規則機械處理。',
                      'V4 尚待獨立逐日複製；本研究只驗證「能重現 V4 研究自己的 AB_V4 淨值」，不等於獨立複製。'],
    'validation': ['重跑 AB_V4（normal、stress）與 navs_v4.parquet 逐日相對差 ≤ 1e-9；V4（BASE）的計畫重跑與 V4|normal 相對差 ≤ 1e-9；V4_SIG BASE 與 V4_SIG|normal 相對差 ≤ 1e-9。',
                   '袖子函式重現 o_largecap_leg MAIN70_0050_30|normal（≤ 1e-6）。',
                   'AB_V4 的 ALL 年化／t 與 V4 研究 results.json 相同（≤ 1e-6）。'],
    'inputs': {'o_v4_results_PREREG_sha256': v4res['PREREG_sha256'], 'navs_v4.parquet_sha256': sha(V4D / 'navs_v4.parquet'),
               'v4lib.py_sha256': sha(V4D / 'v4lib.py'), 'data_freeze_v4.json_sha256': sha(V4D / 'data_freeze_v4.json'),
               'main_v2_navs_gapfix_sha256': sha(AG / 'main_v2' / 'navs_gapfix.parquet'), 'largecap_navs_sha256': sha(AG / 'o_largecap_leg' / 'navs.parquet')},
    'files': ['prereg.py（本檔）', 'run.py（重跑 AB、組合、統計、選版、判定，寫回 results.json 的 RESULTS，只寫一次）'],
    'python': 'python3 -I -B',
}
blob = json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()
R = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(blob).hexdigest(), 'PREREG_written_at': pd.Timestamp.now(tz='UTC').isoformat()}
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_written_at'])

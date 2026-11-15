#!/usr/bin/env python3
"""S6 事前登記：寫 PREREG.json 與 PREREG.sha256（已存在就拒絕覆寫）。python3 -I -B prereg_write.py"""
import json, hashlib, datetime, sys
from pathlib import Path

H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
if f.exists():
    sys.exit('PREREG.json 已存在，拒絕覆寫')

P = {
    'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'topic': 'S6：衛星資金利用率與名次加權——報酬和回撤的取捨曲線（s6_satellite_sizing）',
    'seen_before_prereg': ('看過 s5_core_satellite/FINDINGS.md（含研究中心附註）、replicate/REPLICATION_REPORT.md、replicate/review_numbers.json（審查 B 的數字）、'
                           's5lib.py、run.py、posthoc.py、prep.py、PREREG.json、xs/底倉衛星_操作說明.md、results.json 的 meta 與 acceptance，'
                           's4lib.py 的 load／innov_panel／make_signal、s3lib.extra_stats、stlib 的 window／yearly／trade_stats／sharpe。'
                           '寫這份檔案時本題的面板還沒建、沒有用本題程式算過任何報酬。'
                           '事前已知的數字（任務說明與 S5／複製報告給的）：W10_70＝S5 主格 48.13%／34.08%／70.16%、−33.4%、379 筆；'
                           'W10_100＝S5 的 F1 53.63%／37.70%、−36.1%；複製報告審查 B（每檔＝衛星 ÷ max(名單檔數, 5)、衛星 70%）62.16%／44.50%、−39.49%，'
                           '它的定義和本題 WN_70 應相同或極接近，所以 WN_70 的報酬不是盲測。其餘 5 格（W7_70、W7_100、WN_100、RW10_70、RWN_70）的報酬都還沒看過。'
                           'S5 事後觀察：名次 1～5 名對同期 0050 多賺 +6.72%、6～10 名 +4.12%（名次加權的動機，事後觀察，非盲）。'),
    'data_universe_signal': ('與 S5 完全相同：import s5lib（→ s4lib → s3lib → s2lib → stlib），只上市、排除創新板、最小買進金額、嚴成本、100 萬、淨值起算 2015-07-01。'
                             '衛星選股 s4lib.make_signal(depth=10, mom_lo=0.10, exit=\'top20\', keep_info=True)：L_MA 前 10 名中 RET20 > 10%、不遞補、依 SURGE 由大到小、最多 10 檔、'
                             '隔日開盤買；換股日不在 L_MA 前 20 名（含跌破季線）就賣。不改任何既有檔案；面板快取放 scratchpad/s6/。'),
    'engine': {
        '_impl': ('s6lib.simulate_s6：s5lib.simulate_cs 的逐行複製（底倉 B 簿記、拉回、減碼、0050 空位、成本全部相同），只把「每檔目標金額」換成下面的 sizing 規則。'
                  'sizing=W10 且不加權時必須與 s5lib.simulate_cs 日淨值（容許 1e-6 元）與交易逐筆相同（驗收 A2）。'),
        '衛星資金': 'Sv＝max(換股執行日開盤估淨值 − B, 0)，在出場賣完、拉回之後、買進之前計算（同 S5）。',
        '每檔目標金額': {
            'W10': 'Sv ÷ 10（＝S5 主格）。',
            'W7': 'Sv ÷ 7；最多仍 10 檔；依 SURGE 順序逐檔買，錢不夠時照 S5 引擎：該檔買 min(目標, 可用資金)，若買得到的金額 < 0.5 × 目標就不買、看下一檔（錢用完為止）。',
            'WN': 'Sv ÷ max(n, 5)，n＝當月新名單檔數（訊號日 L_MA 前 10 名中 RET20 > 10% 的檔數＝make_signal 的 n_tgt，含已持有、含當天要賣的）；續抱股不重新平衡。',
        },
        '名次加權（RW10、RWN）': ('基本單位 u＝Sv ÷ 10（RW10）或 Sv ÷ max(n, 5)（RWN）；名單內每檔權重 w＝1.5（L_MA 名次 pos_ma 1～5）或 1.0（pos_ma 6～10）。'
                              '總額＝u × Σw（Σ 對當月整份名單 n 檔，含已持有與當天要賣的）；縮放 s＝min(1, Sv ÷ 總額)；每檔目標＝u × w × s。'
                              '「名單第 1～5 名」定義為 L_MA 名次 1～5（＝S5 P4 的 pos_ma ≤ 5），不是名單內的序位。'),
        '可用資金': '同 S5：max(現金, 0)＋全部 0050（含底倉）可變現金額；買完後 B > 現金＋0050 時 B 降為現金＋0050（底倉被賣去買股），每天收盤同樣夾住。',
        '最小買進金額': '股數 × 開盤價 < 0.5 × 該檔自己的目標金額就不買、不占名額（同 S5 規格 2，目標改成本題的每檔目標）。',
        '拉回': '同 S5：只在換股執行日開盤、出場賣完後、買進前；|Sv ÷ eq − 衛星比例| > 10 個百分點就把 B 設回 (1 − 衛星比例) × eq；衛星偏低只改簿記；偏高先劃閒置 0050／現金，不夠再依市值等比例減碼個股。衛星 100% 時 B 恆為 0、不拉回。',
        '上限': '最多 10 檔（slots=10），單筆 ≤ 當日成交值 5%，2020-10-26 前整張。',
    },
    'grid': {
        'cells': [
            dict(id='W10_70', sizing='W10', sat_w=0.70, rank_w=False, note='＝S5 主格（對照）'),
            dict(id='W7_70', sizing='W7', sat_w=0.70, rank_w=False),
            dict(id='WN_70', sizing='WN', sat_w=0.70, rank_w=False, note='主格（事前指定）'),
            dict(id='W10_100', sizing='W10', sat_w=1.00, rank_w=False, note='＝S5 的 F1'),
            dict(id='W7_100', sizing='W7', sat_w=1.00, rank_w=False),
            dict(id='WN_100', sizing='WN', sat_w=1.00, rank_w=False),
            dict(id='RW10_70', sizing='W10', sat_w=0.70, rank_w=True, note='名次加權'),
            dict(id='RWN_70', sizing='WN', sat_w=0.70, rank_w=True, note='名次加權'),
        ],
        'main': 'WN_70',
        'main_reason': '最直接回答「把衛星用滿」：每檔＝衛星 ÷ max(名單檔數, 5)，衛星 70%。',
        'no_other_cells': '格子就是這 8 格（6＋名次加權 2），不得增減、不得事後挑。',
    },
    'acceptance（先做，通過才算新格）': {
        'A0': 's5lib.simulate_cs、衛星 70%、嚴成本：2019 起 48.13%、全期 34.08%（各 ±0.02 個百分點）、已平倉 379 筆。',
        'A2': 's6lib.simulate_s6(sizing=W10, rank_w=False) 在衛星 70% 與 100% 時，日淨值與 s5lib.simulate_cs 相同（atol 1e-6 元），交易逐筆相同（代號、進場、出場、原因）。',
    },
    'per_cell_report': {
        '成本': '嚴為主（分類、取捨表、穩健性都用嚴）；輕成本並列 2019 起／全期／2024 起／最大回撤。',
        '視窗': '2019 起＝2018-12-28 收盤起；全期＝2015-07-01 起；2024 起＝2023-12-29 收盤起；前後兩段切在面板中點（S5 為 2021-02-19）；逐年。',
        '指標': ['2019 起年化對同期 0050（放第一個）', '全期', '2024 起', '逐年', '最大回撤（全期與 2019 起）', '年化波動（全期、2019 起）', '日夏普（全期、2019 起，不扣無風險利率）',
                '平均個股部位占總資金（第一筆進場日起）', '已平倉筆數、每筆期望值＝勝率 × 平均賺 −（1 − 勝率）× 平均賠（代入數字）', '年化單邊周轉＝(買進值＋賣出值) ÷ 2 ÷ 平均淨值 ÷ 年數（同 s3lib.extra_stats）',
                '拉回次數（向上／向下）、減碼個股的天數、筆數、金額', '底倉真的被賣去買股的次數：換股執行日，收盤後 B 比「拉回後 B × 0050 當日開→收漲跌」低超過 0.01% 的天數，並列合計與最大金額、最大比例',
                '成本加倍（2019 起／全期）', '禁買最賺 5 筆：該格嚴成本主跑已平倉損益最大 5 筆，在各自整段持有期禁買該股（s2lib.block_from），重跑',
                '前後兩段', '換股日往後移 1、3、5 個交易日（make_signal shift）', '20 條路徑：60、80、100、150、200 萬 × 位移 0～3 日的 2019 起年化最小／中位數／最大與高於 0050 條數',
                'beta 與月 alpha（t）：月報酬 OLS r＝α＋β r_0050，全期與 2019 起（s5lib.beta_alpha）', '月超額 t',
                '滾動 12 個月（252 個交易日）報酬減 0050，自 2016-03-01 起：最差、最差結束日、落後時間比例、中位數'],
    },
    'classification（機械套用，嚴成本）': ('八項：2019 起、全期；成本加倍的 2019 起、全期；禁買最賺 5 筆的 2019 起、全期；前半；後半，全部高於 0050＝值得上 XQ 測；'
                                       '2019 起與全期都低於 0050＝不值得；其餘＝有條件（列出沒過的項目）。'),
    'tradeoff_table': ('報酬 vs 最大回撤：8 格＋S5 主格（＝W10_70）＋0050。以「全期最大回撤（絕對值）」對「2019 起年化」與對「全期年化」各判一次：'
                       '若沒有任何其他列的回撤 ≤ 它且報酬 > 它（或回撤 < 它且報酬 ≥ 它），就標「同樣回撤下報酬最高」（效率前緣）。'
                       '另列每格相對 0050、相對 W10_70 的「回撤多深幾個百分點 ÷ 年化多幾個百分點」。'),
    'posthoc_rule': '看過任何新格報酬之後才加的分析一律標「事後」，不影響分類與主格。',
    'resources': '單行程、峰值記憶體 2GB 內；python3 -I -B；中間檔 scratchpad/s6/，結束刪除。',
}
b = json.dumps(P, ensure_ascii=False, indent=1).encode('utf-8')
f.write_bytes(b)
h = hashlib.sha256(b).hexdigest()
(H / 'PREREG.sha256').write_text(f'{h}  PREREG.json\n')
print(h)

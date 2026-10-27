#!/usr/bin/env python3
"""integrate_o2 事前登記：在任何回測之前，把組合、選版規則、判定規則寫進 results.json（含 sha256 與時間）。只能執行一次。"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'
assert not RJ.exists(), 'results.json 已存在：PREREG 只能寫一次'

PREREG = {
    'study': 'integrate_o2：把 o2 唯一通過審查（待觀察）的 o_rev_survivorship（營收存活者偏誤修補，附 KY／DR 補資料發現）和主策略 v2 排除金融 n10 整合',
    'context': 'o2 其他三題（MSCI 新納入避開、可轉債／現增避開、融券借券外資持股水準）沒有通過審查，不納入整合。'
               'o_rev_survivorship 不是新選股訊號，而是資料修補；所以「整合」的意義是：(a) 用修補後的資料重新確認主策略與既有配置選項（台積電 30%）；'
               '(b) 把修補研究衍生的兩個可規則化的選擇——「缺營收者給中性分數」與「把 KY／DR 納入股池」——當成濾網／加分項／第二條腿測試。',
    'engine': '只讀 import o_rev_survivorship/osurv_core.py（→ main_v2/v2_core.py → independent_audit/ia_core.py）取得 V2B 與凍結的營收版本（斷言 sha256），'
              '只讀 import o_capacity_tsmc_gap/ocore.py 取得加權日頻引擎 simulate_w、tsmc_plan、segstat、periods、window_returns、paired_t。成本、漲停遞補、停牌凍結都不重寫。',
    'base': {
        'MAIN_O': '主策略正式版（現行營收 ORIG）：gapfix 窗、排除金融（金融保險業＋28xx、5876、5880）、前 300 流動性池、0.5×52 週比百分位＋0.5×近 3 月營收年增率百分位、'
                  '同分先比營收再比代碼、每月第一個 ≥11 日交易日收盤訊號、隔日開盤等權 10 檔、跳動點開盤漲停未持有者跳過遞補、停牌凍結、評價到 2026-10-12、2021-03 起營收用 rev_5y。',
        'MAIN_R': '同一規格，營收換成 R_DEL（只補國內公司原本 NaN 的格子）。判定與選版的比較基準＝MAIN_R（同一份修補資料，才能把組合效果和資料修補效果分開）；另報和 MAIN_O 的配對差。',
    },
    'combos': {
        'NEUTRAL05': '加分項／缺值規則：R_DEL ＋「池內其他條件都合格但缺營收的國內股票，營收百分位給 0.5」（＝o_rev_survivorship 的 B05_DEL）。'
                     '實盤可執行：工具遇到缺營收（例如新上市不足 15 個月）時給中性分而不是剔除。',
        'KY_ALL': '股池擴充：營收換成 R_ALL（另補 KY／DR，只有 2015-01～2019-08 有 KY 營收），其餘同 MAIN_R。',
        'KY_CAP2': '濾網版：同 KY_ALL 的排名，但 10 檔中 KY／DR 最多 2 檔（超過的跳過、往下遞補）。上限 2 事前固定，不做格點（比照處置股上限 20%）。',
        'MIX80_20_KY': '第二條腿：80% MAIN_R ＋ 20% KY_ALL，合併成單一帳戶，每次換股日調回比例，抵銷的交易只算一次（兩腿共同持股權重相加）。',
        'MIX50_50_KY': '50% MAIN_R ＋ 50% KY_ALL，同上。',
        'TSMC30_R': '既有配置選項的重新確認：MAIN_R 的 10 檔占 70%＋2330 固定 30%（integrate_o1 的 NEW，換股之間不再平衡），只把營收換成 R_DEL。',
    },
    'not_tested_reason': {
        '避開後來下市的公司': 'o_rev_survivorship P5：2015-04～2019-12 修補期內，主策略 570 個入選股月中下市前 12 個月的 0 個；而且「後來下市」事前不可知，無法寫成濾網。',
        'B10 極端最壞': '只是資料界限，不是可執行規則。',
    },
    'cost': {'normal': '買 0.0855%＋滑價 0.1%；賣 0.0855%＋證交稅 0.3%＋滑價 0.1%', 'stress': '買賣各 0.5%'},
    'mode': 'BASE（換股日可得的完整月營收）。AB（只用已公告營收）在 main_v2／o_rev_survivorship 的影響 ≤ 1pp，本研究不重跑；實盤預期用 BASE 套五折規則並註明。',
    'periods': {'SEL': '訊號窗起點 2006-04～2019-03（ocore.periods，SPLIT＝2019-04-11）', 'VAL': '2019-04-11～2026-10-12', 'VAL_A': '2019-04～2023-12',
                'HOLD': '2024-01～2026-10-12（只報告，不參與任何選擇）', 'S1': '2006-04～2013', 'S2': '2014～2020', 'S3': '2021-03～2026-10-12', 'ALL': '2006-04～2026-10-12',
                'FIX': '訊號 2015-04～2019-12（修補資料實際影響訊號的唯一期間；KY 只在這段有營收）'},
    'metrics': ['年化（日頻）', '最大回撤（日頻）', '對 0050 含息 t（訊號窗月超額）', '對同池等權 t', '追蹤誤差、資訊比率',
                '和 MAIN_R、MAIN_O 的訊號窗月報酬差：平均 pp、配對 t', '和 MAIN_R、MAIN_O 的月報酬相關、月超額（對 0050）相關', '逐年、贏 0050 年數', '壓力成本',
                'KY 持股檔數（每期平均、最多）', '和 MAIN_R 名單不同的月數'],
    'selection_rule': '只用 SEL：在 NEUTRAL05、KY_CAP2、MIX80_20_KY、MIX50_50_KY、KY_ALL、TSMC30_R 中，選 SEL 對 0050 t 最高、而且 SEL 年化比 MAIN_R 少 ≤ 1pp 的一個為「建議整合版」；'
                      '若沒有任何一個的 SEL t 高於 MAIN_R，維持 MAIN_R。HOLD 不得參與。比例（80/20、50/50）事前固定，不在任何期間挑。',
    'judgement_rule': {
        '可用': '下面兩條路之一成立：(A) 報酬改善：ALL 和 MAIN_R 的月差配對 t ≥ 1.0，而且 SEL 與 VAL 的月差平均都 > 0；'
                '(B) 追蹤誤差改善：ALL 對 0050 t 比 MAIN_R 高 ≥ 0.2、SEL 與 VAL 的對 0050 t 都高於 MAIN_R、ALL 年化少 ≤ 1pp、ALL 最大回撤不深於 MAIN_R。',
        '待觀察': '不符合可用，但 ALL 月差配對 t > −1、ALL 年化少 ≤ 2pp，而且 SEL、VAL、FIX 至少一段的對 0050 t 高於 MAIN_R。',
        '不可用': '其餘（含 ALL 配對 t ≤ −1 或 ALL 年化少 > 2pp）。',
        'caps': {'KY_ALL／KY_CAP2／MIX*_KY': '上限待觀察：rev_5y（2021 起）與實盤工具完全沒有 KY／DR 營收，KY 只在 2015-01～2019-08 有資料，且這段資料已在 o_rev_survivorship 看過；'
                                             '要升級必須先補到 KY 月營收並做前瞻追蹤。',
                 'TSMC30_R': '上限沿用 o1 總監判定（待觀察）；本研究只確認修補後結論是否改變。',
                 'NEUTRAL05': '上限待觀察：B05_DEL 的全期數字已在 o_rev_survivorship 公開，不是新的樣本外。'},
    },
    'honesty_notes': ['MAIN_O、MAIN_R、NEUTRAL05、KY_ALL、TSMC30_R 的全期數字已在 o_rev_survivorship（含 p6_R_DEL）公開；KY_CAP2、MIX80_20_KY、MIX50_50_KY 是新組合，'
                      '但兩條腿只在 FIX 期間（2015-04～2019-12 訊號）有實質差異，其他期間的差異只來自持股路徑。',
                      '修補與 KY 的影響幾乎全落在挑選期（SEL），驗證期與保留期幾乎沒有新資訊；這一點必須在結論中寫明。'],
    'validation': '執行前斷言：MAIN_O 的 ALL 年化與 integrate_o1 MAIN（28.15%）相差 ≤ 0.05pp；MAIN_R、NEUTRAL05、KY_ALL 與 o_rev_survivorship navs_R_DEL／B05_DEL／R_ALL 的 EXFIN|n10|normal 最終淨值相對差 ≤ 0.1%。',
    'expectation_rule': '實盤預期年化＝同期 0050 年化＋0.5×（回測年化 − 0050 年化）；BASE 模式，另註 AB 約再少 ≤ 1pp。',
    'files': ['prereg.py（本檔）', 'run.py（全部回測、統計、選版、判定，寫回 results.json 的 RESULTS）'],
    'python': 'python3 -I -B',
}
blob = json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()
R = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(blob).hexdigest(), 'PREREG_written_at': pd.Timestamp.now(tz='UTC').isoformat()}
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_written_at'])

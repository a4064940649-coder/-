#!/usr/bin/env python3
"""integrate_o1 事前登記：在任何回測之前，把要測的組合、選版規則、判定規則寫進 results.json（含 sha256 與時間）。
只能執行一次；results.json 已存在就中止。"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'
assert not RJ.exists(), 'results.json 已存在：PREREG 只能寫一次'

PREREG = {
    'study': 'integrate_o1：把 o_capacity_tsmc_gap 通過的「主策略＋台積電 2330 固定權重 30%」和主策略（v2 排除金融 n10）整合',
    'base_main': 'MAIN＝main_v2 正式規格：gapfix 窗、排除金融（stock_list 金融保險業＋28xx、5876、5880，營收設 NaN）、前 300 流動性池、'
                 '分數 0.5×52 週比百分位＋0.5×營收 3 月年增率百分位、同分先比營收再比代碼、每月第一個 ≥11 日交易日收盤算訊號、隔日開盤成交、'
                 '跳動點漲停未持有者跳過遞補、停牌凍結、等權 n10、評價到 2026-10-12。程式只讀 import o_capacity_tsmc_gap/ocore.py（V2R(1,300) 與 simulate_w）。',
    'new_strategy': 'NEW＝o_capacity_tsmc_gap 凍結規格：換股日開盤 2330 占 30%，其餘 70% 平分 MAIN 的 10 檔；2330 入選則相加；換股之間不再平衡。',
    'combos': {
        'MAIN': '主策略單獨（基準，2330 權重 0）',
        'NEW': '新策略單獨（2330 30%）',
        'MIX80_20': '第二條腿固定比例：80% MAIN＋20% NEW，每次換股日把兩條腿調回 80/20。合併成單一帳戶、互相抵銷的交易只算一次，等同 2330 權重 6%、10 檔各 9.4%',
        'MIX50_50': '50% MAIN＋50% NEW，同上，等同 2330 權重 15%、10 檔各 8.5%',
        'SLOT10': '濾網／加分版：2330 直接占 10 檔中的一席（等權 10%）。若 2330 本身已在前 10 名，名單不變；否則取前 9 名＋2330。2330 未持有且開盤漲停時不加入（照 MAIN 的漲停規則），名單維持前 10',
        'TREND30': '濾網版：只有在 2330 自己的 52 週比（還原收盤÷252 日最高）在當期合格池的百分位 ≥ 0.5 時，才持有 NEW 的 30% 2330；否則該期整個帳戶就是 MAIN。門檻 0.5 事前固定，不做格點',
        'C0050_30': '對照（不參選）：0050 30%＋MAIN 70%，每次成交日再平衡；0050 交易成本買 0.1855%、賣 0.1855%＋證交稅 0.1%（沿用 o_capacity mix_0050 寫法）',
    },
    'cost': {'normal': '買 0.0855%＋滑價 0.1%；賣 0.0855%＋證交稅 0.3%＋滑價 0.1%', 'stress': '買賣各 0.5%（0050 對照組另以同一 mix 寫法，只報一般成本）'},
    'periods': {'SEL': '訊號窗起點 2006-04～2019-03（日頻到 2019-04-11）', 'VAL': '訊號窗起點 2019-04-11～2026-09，日頻到 2026-10-12',
                'VAL_A': '2019-04～2023-12', 'HOLD': '2024-01～2026-10-12（只報告，不參與任何選擇）',
                'S1': '2006-04～2013', 'S2': '2014～2020', 'S3': '2021-03～2026-10-12', 'ALL': '2006-04～2026-10-12'},
    'metrics': ['年化（日頻）', '最大回撤（日頻）', '對 0050 含息 t（訊號窗月超額）', '對同池等權 t', '追蹤誤差、資訊比率',
                '和 MAIN 的訊號窗月報酬差：平均、t（配對）', '和 MAIN 的月報酬相關係數、月超額（對 0050）相關係數',
                '腿與腿的相關：MAIN vs 2330、MAIN vs 0050、(MAIN−0050) vs (2330−0050)', '逐年贏 0050 的年數', '壓力成本'],
    'selection_rule': '只用 SEL：在 NEW、MIX80_20、MIX50_50、SLOT10、TREND30 中，選 SEL「對 0050 t」最高、而且 SEL 年化比 MAIN 少 ≤ 1pp 的一個作為「建議整合版」。'
                      '都不符合就維持 MAIN。HOLD 不得用來挑比例或挑版本。C0050_30 不參選。',
    'judgement_rule': '每個組合（含建議整合版）三個條件都符合才列「可用的配置選項」：(1) VAL 對 0050 t 高於 MAIN；(2) ALL 年化比 MAIN 少 ≤ 2pp；(3) ALL 最大回撤不深於 MAIN。'
                      '另外要求 VAL 和 MAIN 的月差配對 t > −1（不顯著變差）。只有 HOLD 改善的判待觀察；其餘不可用。',
    'honesty_notes': ['NEW 與 MAIN 的 VAL／HOLD 數字已在 o_capacity_tsmc_gap 公開，MIX80_20／MIX50_50 介於已知格點 w0～w30 之間（等同 w6／w15），'
                      '因此本研究對這兩者不是全新的樣本外，只是事前固定比例、用 SEL 機械選版；SLOT10、TREND30 是本研究新加的版本，VAL 結果事前未知。',
                      'NEW 的 30% 是 o_capacity 格點上界（0/10/20/30%），本研究不再往上加格點。'],
    'validation': '執行前斷言：本程式重算的 MAIN（w0）與 NEW（w30）ALL 年化與 o_capacity_tsmc_gap/results_oos.json 相差 ≤ 0.05pp。',
    'files': ['prereg.py（本檔）', 'run.py（全部回測、統計、選版、判定，寫回 results.json 的 RESULTS）'],
    'python': 'python3 -I -B',
}
blob = json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()
R = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(blob).hexdigest(), 'PREREG_written_at': pd.Timestamp.now(tz='UTC').isoformat()}
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_written_at'])

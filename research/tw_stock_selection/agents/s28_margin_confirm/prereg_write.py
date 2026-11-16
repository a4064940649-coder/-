#!/usr/bin/env python3
"""S28 事前登記：在算任何報酬之前寫 PREREG.json、PREREG.sha256、code_prereg.sha256。PREREG.json 已存在就拒絕。python3 -I -B prereg_write.py"""
import sys, json, hashlib, datetime
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
assert not (H / 'PREREG.json').exists(), 'PREREG.json 已存在'
assert not (H / 'results_train.json').exists() and not (H / 'results.json').exists(), '已經有報酬結果，不能事前登記'
import s28lib  # noqa: E402

dc = json.loads((H / 'data_check.json').read_text())
assert dc['sufficiency']['verdict'] == '資料足夠'
sha = lambda p: hashlib.sha256((H / p).read_bytes()).hexdigest()
PRE = {
    'topic': 'S28 毛利率確認：營收爆發但毛利率在惡化的公司，是不是 SURGE 的地雷？',
    'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    'status_at_write': '只跑過 prep.py（data_check.json）：面板、季報涵蓋、PIT 季別、F1 上的 RY／DGM／DOPM 分布、各規則改動檔數、由衰轉盛股被判定比例；沒有算任何報酬。',
    's28lib_sha256_at_write': sha('s28lib.py'),
    'baseline（不改定義）': ('S10 的 E0／W10_70（XQ 20＋21 號預設），除權息修正後引擎（cslib.EXRIGHT_FIX＝True）：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、'
                        'F1 進場（L_MA 依 SURGE 前 10 名中 RET20 > 10%）、E0 出場（跌出 L_MA 前 20 名）；衛星 70%、每檔＝衛星 ÷ 10；0050 停牌日用前一日收盤（s17lib）。'
                        'SURGE＝STD（σ > 0）。引擎 s10lib.simulate_s10，經 s28lib → s21lib → s20lib → s17lib → s10lib 直接 import，不改。'),
    'motivation': 'SURGE 只看月營收；營收成長不代表獲利成長。營收增加但毛利率下降，可能是降價搶量或一次性低毛利訂單，營收爆發可能不持久（S25：報酬靠少數大贏家、中位數交易略負）。',
    'data': {
        'file': 'agents/o_pead_margin_surprise_low_attention/data/fin_long.parquet（MOPS XBRL 單季財報，公開封存；只讀不複製）＋ avail_dates.parquet（市場可得日 A）',
        'fields': 'GM＝單季毛利 ÷ 單季營收；OPM＝單季營業利益 ÷ 單季營收；RY＝單季營收年增率；DGM、DOPM＝與去年同季相比的變化（個百分點）。Q4＝全年減前三季。',
        'pit': dc['pit_rule'],
        'coverage': f"F1 的 DGM 可用率：訓練期 {dc['train']['F1']['DGM_avail']:.1%}、驗證期 {dc['valid']['F1']['DGM_avail']:.1%}（缺值多為金融保險格式、無毛利科目）；"
                    f"資料齡（訊號日距季底）中位數 {dc['train']['F1']['age_days_median']:.0f} 天、最多 {dc['train']['F1']['age_days_max']:.0f} 天。",
        'caveat': '封存可能含更正後版本（無法逐筆驗證首次申報值）；2017-04 前的 A 是推斷值（只用來排除晚交，不用來提前）。'
    },
    'rule_form': {
        'candidates（4 條，全部計入 N；參與訓練期選擇）': [{k: v for k, v in r.items()} for r in s28lib.RULES],
        'common': ('所有規則只改當月買進名單；出場 X（E0）不動。篩掉的名額不遞補（SWAP_GM2 除外），錢照引擎掃進 0050。'
                   '判定值缺（金融保險、新上市無去年同季）→ 保留，不禁買。X（毛利率年減門檻）只在 2、5 兩個值之間、只用訓練期選。'),
        'turnaround_safeguard（事前說明怎麼避免擋掉由衰轉盛的大贏家）': [
            '(1) 只在「同一季單季營收年增 > 0」時才判定毛利率惡化：營收年減的季（衰退谷底季）就算毛利率下滑也不禁買。由衰轉盛的公司在訊號日能看到的最近一季，常常還是谷底季（資料齡中位數約 104 天），這條讓它們不受影響。',
            '(2) 只看最近一季的「年變化」，不要求毛利率水準、不要求連續幾季改善、不看 μ÷σ 或營收成長的持續性（S20 持續成長濾網失敗的原因）。',
            '(3) 只做「禁買」，不做加碼、不改出場、不改每檔金額（S6、S13：加碼會變成押少數股票）。',
            f"(4) 事前特徵檢查（data_check.turnaround_check，只看特徵不看報酬）：訓練期 μ ≤ 0 的 F1（{dc['turnaround_check']['train']['turnaround_mu_le0']['n']} 筆）被 BAN_GM2 判定的比例 "
            f"{dc['turnaround_check']['train']['turnaround_mu_le0']['flag_GM2']:.1%}，低於 μ > 0 的 {dc['turnaround_check']['train']['mu_gt0']['flag_GM2']:.1%}；"
            f"若不加 (1)，μ ≤ 0 組會有 {dc['turnaround_check']['train']['turnaround_mu_le0']['DGM_lt_m2_any_RY']:.1%} 被擋。",
        ],
        'rule_effect_preview（data_check，只數檔數）': dc['rule_effect_preview'],
    },
    'windows': {
        'train': '訊號日 2016-02～2021-12（營收月份 2016-01～2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出，同 S20／S21 train.py）',
        'valid': '訊號日 2022-01～2026-10（營收月份 2021-12～2026-09）；重新起跑：只保留營收月份 ≥ 2021-12 的訊號，驗證窗 2021-12-30 → 2026-10-22（同 S20／S21 validate.py）'
    },
    'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S20、S21）',
    'selection': ('訓練期分數＝規則 − 現行 CUR 的訓練窗年化差，20 條路徑中位數（個百分點）；最高者凍結（同分依「改善 ≥ +0.25」路徑數、再依改動檔數少者）。'
                  'T0（訓練期最低要求，含改善最小幅度，沿用 S25）：分數 ≥ +1 個百分點，且年化差 ≥ +0.25 個百分點的路徑 ≥ 14／20。'
                  'T0 不過仍凍結分數最高者，驗證期只描述、判定必為不通過。'),
    'verdict（驗證期只跑一次，沿用 S25）': {
        'V1': '驗證窗（重新起跑）年化差為正的路徑 ≥ 15／20',
        'V2': '配對差中位數 ≥ +2 個百分點',
        'V3': '禁買 CUR 與凍結規則各自（重新起跑主跑）最賺 5 筆期間的聯集後，配對差中位數 > 0',
        'V4': '兩段空頭（2022；2024-07～2025-04）段內最大回撤配對差中位數都不深於 −2 個百分點',
        'overall': 'T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換規則重跑、不寫 XQ 修改建議'
    },
    'acceptance': {
        'A0': 'CUR 全期主跑（位移 0、100 萬）重現除權息修正後 W10_70：2019 起 48.99%、全期 34.54%、回撤 −32.11%（±0.025 個百分點）、381 筆交易',
        'A1': '「永不判定」的規則（gm＝+∞）交易與淨值與 CUR 逐筆相同',
        'A2': 'build_variant 每個訊號列重現的 F1 與 make_signal_s10 完全相同（assert）'
    },
    'controls_and_posthoc（未登記，只描述，不參與選擇與判定）': {
        'random_removal': '隨機剔除對照：每月從 F1 隨機剔除與凍結規則相同檔數（swap 時另從第 11～20 名 RET20 > 10% 隨機遞補相同檔數），20 次；看凍結規則是否落在隨機範圍外。',
        'event_level': '逐筆：F1 在 E0 持有期對 0050 的超額（s20lib.outcomes，不扣成本），毛利率惡化組（RY > 0 且 DGM < −2）vs 其他組：平均、中位數、去最賺 5 筆、勝率、月份群聚 t；訓練／驗證期分開；另看 RY ≤ 0 且 DGM < −2（谷底季）組。',
        'trades_diff': '凍結規則擋掉了哪些現行交易（全期連續主跑），其中有沒有大贏家、是不是由衰轉盛股。'
    },
    'N': '資料窺探族群 N：+4（4 條候選；以 BACKLOG 最近記錄 S25 的 2,861 計 → 2,865；S26、S27 若另有增加由研究中心調整）',
    'predictions（事前預測）': {
        'P1': '訓練期四條都在 −1.5～+1 個百分點之間：被擋的只有 F1 的 4～9%（訓練期 12～24 筆），錢放 0050 的機械效果約 −0.5～−1.5',
        'P2': 'SWAP_GM2 最差或接近最差（S21：L_MA 第 11～20 名遞補平均比 F1 差）',
        'P3': 'T0 通過機率約 15%；整體通過約 5%，最可能卡在 T0（改善路徑數）或 V2（+2 很難，改動檔數少）',
        'P4': '事後描述：毛利率惡化組平均超額略低於其他組，但群聚 t 的絕對值 < 1.5（樣本小、報酬由少數大贏家決定）'
    }
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
(H / 'PREREG.json').write_text(txt)
s = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{s}  PREREG.json\n')
(H / 'code_prereg.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in ('s28lib.py', 'prep.py', 'prereg_write.py')))
print(s)

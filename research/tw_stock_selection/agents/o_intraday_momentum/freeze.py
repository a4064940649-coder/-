#!/usr/bin/env python3
"""凍結：把 PREREG（含 sha256 與時間戳）寫進 results.json。之後 oos.py／p4.py oos 才能執行。
已經存在 PREREG 時拒絕覆寫。"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent
RF = HERE / 'results.json'
doc = json.loads(RF.read_text(encoding='utf-8')) if RF.exists() else {}
assert 'PREREG' not in doc, 'PREREG 已存在，不可覆寫'
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
assert it['results']['chosen'] is None   # 本次凍結文字是依「沒有版本通過」寫的
fh = lambda p: hashlib.sha256((HERE / p).read_bytes()).hexdigest()

PREREG = {
 '0_status': '挑選期探索與迭代已完成（explore.json、iterations.json、p4_sel.json）。凍結後才計算 2019-05～2026-10 的任何結果。',
 '1_combo_verdict': {
   'rule': '挑選期配對差（對原主策略）t ≥ 1.0 的版本中，取對同池等權 t 最高者；都沒有 → 不可用、不打開驗證期',
   'result': '8 個版本全部配對差為負（t −1.78～−0.21），沒有版本達到 1.0',
   'verdict': '不可用（主策略加盤中／隔夜訊號的結合版）。驗證期不計算任何結合版的數字。',
   'n_versions_tested': 8, 'n_signal_specs_explored': 12},
 '2_P1_spread_test': {
   'purpose': 'P1 是獨立的假說檢定（IMOM／OMOM 五分位價差），不是被選中的策略；規格事前固定為研究計畫原文的「12 個月、跳過最近 1 個月」，另附挑選期最強的規格當次要（揭露其選擇偏誤）。',
   'primary_spec': {'lookback_good_days': 252, 'skip_good_days': 21, 'IMOM': '累積盤中對數報酬 Σlog(close/open)', 'OMOM': '累積隔夜對數報酬 Σ[log(1+ret) − log(close/open)]'},
   'secondary_spec': {'lookback_good_days': 126, 'skip_good_days': 0, 'form': 'cum'},
   'daily_split': 'olib.daily_split：flag 3/4/5、open/close 缺、open==close 且成交值 0 → 缺值；flag 1 隔夜缺值；極端值規則同挑選期',
   'window_coverage': '盤中與隔夜有效日各 ≥ 80% 窗長；只用有值檔數 ≥ 300 的日子',
   'pool': '每月換股日：60 日均成交值前 300 名（≥1,000 萬、收盤 ≥10）、52 週比有值（gapfix）、非處置、隔日有開盤成交、排除金融保險業（v2 FINMASK）',
   'portfolio': '池內依訊號分五等分，等權；隔日開盤買、下一換股隔日開盤賣（毛報酬，扣成本前）；另報近似淨額（多空各換手比例 × 0.5%）',
   'period': '驗證期訊號 2019-05～2026-09（最後一期持有到 2026-10-13 開盤）；分段 2019-05～2023-12、2024-01～2026-09',
   'src3_rule': '主版本剔除回看窗內 src 3（2024-02-15～2025-09-23）日數 > 20% 的訊號月；敏感度版本保留',
   'criteria_validation': 'P1 成立＝ IMOM Q5−Q1 毛報酬 ≥ 0.40%／月、月份 t ≥ 1.0，而且 OMOM Q5−Q1 < 0；挑選期已成立（主規格 0.707%，t 1.98；OMOM −0.754%，t −2.21）',
   'standalone_leg_rule': '只有 P1 在兩期都成立，而且 IMOM 價差（Q5−Q1 月序列）和主策略月超額（對 0050，訊號窗）的相關在兩期都 < 0.5，才把 IMOM 前 1/5（等權、一般成本、K.simulate）當候選第二腿報告；否則只報價差'},
 '3_P2_descriptive': {
   'spec': '主策略前 30 名（正式分數順序）中，依 252 日、不跳過的盤中占比（CI/(CI+CO)，總報酬 ≤ 0 → 缺）以中位數分高低兩半，比較下月可成交報酬；另分流動性第 1～100、101～300 名',
   'role': '描述與機制判斷用（P5）；挑選期 t 0.10 已經觸發 P5「主策略盤中占比分組沒有差異」',
   'daytrade_2021': '2021-01 以後（當沖資料起點）：前 30 名依訊號日前 20 個交易日平均當沖比率分高低兩半，各自做同樣的占比分組；只當機制描述，不做判定'},
 '4_P4_execution': {
   'engine': 'olib.simulate_exec（ia_core.simulate 的改寫；open/open 與 ia_core 逐值相同），主策略 = main_v2 正式（gapfix、排除金融、n10、正式次排序），選股名單完全不變，只改換股日成交價',
   'primary_rule': '研究計畫原文：隔日收盤或 TWAP 相對隔日開盤，「每筆」（買進與賣出合併的單邊平均）在挑選期與驗證期都改善 > 0.20%，才改 SOP；挑選期 close +0.048%、TWAP +0.015%，已不通過 → 主結論預定「維持隔日開盤（就買賣合併而言）」',
   'P4b_buy_close': {
     'origin': '挑選期探索發現（買進側 close 改善 +0.446%／筆 t 3.56；賣出側 −0.351% t −3.31），屬於事後從挑選期看出的規則，凍結後用驗證期檢定',
     'rule': '買進（新進股）改在隔日收盤成交，賣出維持隔日開盤；續抱不動',
     'criteria': '驗證期主版本（排除 src 3 期間的換股日）：(1) 買進側每筆改善 > 0.20% 且月份 t ≥ 1.0；(2) 淨值配對差（對隔日開盤版，訊號窗月報酬）在一般與壓力成本下都 > 0；(3) 剔除收盤漲停的買進後 (1) 的方向不變。全部成立 → 可用（建議 SOP 改「賣出隔日開盤、買進隔日尾盤／收盤集合競價」）；只有方向為正 → 待觀察；其他 → 不可用',
     'src3': 'src 3 期間的收盤是 13:20 近似，主版本該期間的換股日一律用開盤（不套用變體）；敏感度版本全部套用'},
   'twap_definition': '(開盤＋收盤)/2，只是近似，不是真正的成交量加權'},
 '5_period_and_data': {'validation': '2019-05～2026-10-12（與 main_v2 正式終點相同）', 'value_parquet_md5_prefix': '8272e253',
                       'no_spec_change_after_freeze': True},
 '6_inputs_sha256': {'explore.json': fh('explore.json'), 'iterations.json': fh('iterations.json'), 'p4_sel.json': fh('p4_sel.json'),
                     'olib.py': fh('olib.py'), 'combo.py': fh('combo.py')},
}
sh = hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
doc['PREREG'] = PREREG; doc['PREREG_sha256'] = sh
doc['PREREG_time'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
RF.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
print(sh, doc['PREREG_time'])

#!/usr/bin/env python3
"""凍結規格（PREREG）：寫入 results.json，含規格全文 sha256、程式與資料 sha256、時間戳。之後才可跑 oos.py、h3a_pilot.py。"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

it = json.loads((H / 'iterations.json').read_text(encoding='utf-8'))
assert it['chosen_sleeve'] == 'B1' and it['chosen_combo'] == 'C2'

SPEC = {
 'data': {
  'fo_ratio': '外資持股比率：2025-10-07～2026-10-21 用 MI_QFIIS 實值（voidful/tw-institutional-stocker 鏡像）；2021-10-08～2025-10-06 用「2025-10-07 MI_QFIIS 比率 − 之後每日外資（不含自營）淨買超 ÷ 2025-10-07 發行股數」回推（inst_5y；2026-10-08 起 BlackBoxOP，逐筆一致）。回推值截在 [0,1]。資料凍結 data_freeze.json（2026-10-27T19:30:04Z）。',
  'timing': '訊號日 t 只用 t−1 收盤已公布的外資持股比率。',
  'size': '市值近似＝2025-10-07 發行股數 × 訊號日收盤（沒有歷史股數）。',
  'limit': '2015 年以前（甚至 2021-10 以前）的外資持股比率與借券賣出餘額全市場歷史都無法取得；判定上限為「待觀察」。'},
 'H3b_primary_sleeve_B1': {
  'pool': '主策略同池：前 300 名流動性（60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10、依 60 日均值排前 300）、排除金融保險、非處置、隔日有開盤成交（main_v2 gapfix）。',
  'rule': '池內依近似市值分三組（小／中／大），每組取外資持股比率最高 10 檔，共 30 檔，等權。',
  'rebalance': '每月第一個 ≥ 11 日的交易日收盤算訊號，隔日開盤成交（M2）；開盤漲停的新買進跳過、往下一名遞補（已持有者不擋）；停牌持股凍結。',
  'cost': '一般：手續費 0.1425%×0.6＋滑價 0.1%（賣再加 0.3% 證交稅）；壓力：每邊再加 0.3% 滑價。',
  'benchmarks': '同池等權（無成本，main_v2 口徑）、0050 含息、主策略 n10 排除金融（main_v2 gapfix 正式版，重建後與 main_v2 逐日一致）。'},
 'H3b_primary_combo_C2': '主策略分數改為 0.4×52 週比百分位＋0.4×營收 3 月年增率百分位＋0.2×（市值組內外資持股比率百分位；缺值給 0.5），其餘同主策略 n10 排除金融。依挑選期規則（對主策略配對差 t 最高）選出（C2 t 0.54 vs C1 0.49，差距很小）。',
 'H3b_secondary_combo_C1': '主策略 n10 80% ＋ B1 20%，每個成交日（以前一日收盤近似）重設權重。只當次要報告，不用於判定。',
 'windows': {'selection': '訊號日 2021-10-12～2023-12-11（27 期；B1 26 個完整月窗）',
             'validation_A': '訊號日 2024-02-15～2025-09（回推持股）', 'holdout_B': '訊號日 2025-10～2026-09（官方 MI_QFIIS 實值），評價到 2026-10-12',
             'judgement': '合併 2024-02～2026-09（驗證期）＋分段各報一次；之後 2026-11 起前瞻紙上追蹤'},
 'thresholds_H3b': {
  '(1) sleeve': 'B1 扣一般成本後對同池等權月超額：挑選期 t ≥ 2（已達 2.39）；驗證期均值 > 0 且 t ≥ 1.0；和主策略超額相關 < 0.5；市值組內仍成立（三組各自對組內等權月超額的三組平均 > 0；各組分別報告）。',
  '(2) combo': 'C2 對主策略配對差（月）兩期 > 0 且 t ≥ 1.0（挑選期已知 0.54，不滿足 → 至多看「相同報酬下回撤與追蹤誤差下降」：驗證期 C2 年化 ≥ 主策略 −1pp 且最大回撤較淺、對 0050 追蹤誤差較低）。',
  '(3) verdict': '資料只有 2021-10 起 → 兩假說最多「待觀察」。B1 通過 (1) 而組合沒改善 → 待觀察；B1 驗證期均值 ≤ 0 或 t < 1.0 → 不可用（第二條腿）。'},
 'H3a_forward': {
  'status': '無法測試：TWT93U 借券賣出餘額只找到 2026-07-30 起（BlackBoxOP 鏡像），挑選期與驗證期都沒有資料。',
  'signal': '訊號日 t：借券賣出當日餘額（t−1）− 借券賣出當日餘額（t−21），除以 t−1 的 MI_QFIIS 發行股數；池＝主策略同池；池內最高 1/5 為「借券增加」名單。融券（MI_MARGN 今日餘額）同法為對照。',
  'forward_tracking': '2026-11 起每個換股日記錄：(a) 池內借券增量最高 1/5 與其餘的下一期（成交日開盤→下一成交日開盤）報酬差（P1）；(b) 融券同法（P2）；(c) 非近似 0050 成分（近似市值前 50 以外）且 52 週比 ≥ 池中位數的子樣本（P3）、近似 0050 成分（P4）；(d) 主策略 n10 入選股中命中者 vs 其他入選股（P5），以及「命中跳過、往下遞補」版的紙上淨值。',
  'evaluation': '累積滿 24 個月（2028-11）才評估，用原計畫門檻：P1 毛價差 ≤ −0.5%／月且 t ≤ −2，P2 沒有同樣負效果；主策略層級配對差 t ≥ 1.5、每月平均被濾掉 ≥ 0.5 檔。2026-08～10 的描述性試算（h3a_pilot.py）不用於任何決策。'},
 'multiple_testing': 'H3b 共 6 版（B1、B2、B3、C1、C2、C3）＋ 8 個鄰近版本（每組 5／15 檔、權重 0.1／0.3）；H3a 0 版。',
}
spec_txt = json.dumps(SPEC, ensure_ascii=False, sort_keys=True)
R = json.loads((H / 'results.json').read_text(encoding='utf-8')) if (H / 'results.json').exists() else {}
assert 'PREREG' not in R, 'PREREG 已存在，不得覆寫'
R['PREREG'] = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'spec': SPEC,
               'spec_sha256': hashlib.sha256(spec_txt.encode('utf-8')).hexdigest(),
               'code_sha256': {f: sha(H / f) for f in ('sbl_lib.py', 'oos.py', 'data_freeze.json', 'iterate.py', 'explore.py')},
               'iterations_sha256': sha(H / 'iterations.json'), 'explore_sha256': sha(H / 'explore.json')}
(H / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG']['created_utc'], R['PREREG']['spec_sha256'])

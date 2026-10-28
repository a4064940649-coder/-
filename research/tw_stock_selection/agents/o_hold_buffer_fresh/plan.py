#!/usr/bin/env python3
"""事前計畫（PLAN）：在算任何探索或迭代報酬之前寫入 results.json。已存在就拒絕覆寫。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE / 'results.json'
res = json.loads(RES.read_text(encoding='utf-8')) if RES.exists() else {}
assert 'PLAN' not in res, 'PLAN 已存在，不覆寫'
assert not (HERE / 'explore.json').exists() and not (HERE / 'iterations.json').exists(), '已經有探索／迭代結果，PLAN 必須先寫'
val = json.loads((HERE / 'validate.json').read_text(encoding='utf-8'))
assert val['V3_reproduced'] and val['planner_equals_V2plan'] and val['sim_equals_o3'] < 1e-12

PLAN = {
 'question': '主策略 V3（排除金融 n10）能否用「條件式續抱＋權重漂移帶＋換股門檻差＋兩期重疊」把月單邊周轉從約 62% 降到 35～40%，又不損失扣成本前的優勢？',
 'mechanism': '報酬流失只來自價格腿已衰退（52 週比下滑）的舊股；因營收百分位被新公告洗牌而掉到 11～30 名、但仍貼近 52 週高點的舊股，下月報酬應與新進股相近（處分效應→反應不足→數月內逐步消化）。另一個不靠訊號的周轉來源是續抱股每月被調回等權。',
 'seen_before': {'turnover_buffer': '只有 2021+，n10 K=15～30 全部變差', 'main_refine R4 K=20': '長歷史挑選期 t −0.11（純排名緩衝、ORIG 營收、v1 處置）',
                 '2019+ 主策略數字': '已公開（V3：VAL 9.94%、HOLD 60.09%）；緩衝版本的 2019+ 數字本題凍結前不看',
                 'note': '因 2019+ 主策略與 turnover_buffer 的 2021+ 緩衝結果都已看過，最終判定上限為「待觀察」。'},
 'data': 'V3＝o_infra_v3_rebuild 的 P（下市股價格）、X（併購出清 main）、R_DEL 營收；o_disposal_rejudge 官方處置名單 E1（全部排除）。validate.py 重現 R_DEL|E1 跨循環 26.9459%、t 2.5486（逐位相同）。',
 'engine': 'hb_core.sim＝o3_core.simulate 的推廣（band／逐檔成本），band=None 時逐日相同（validate 斷言 max 相對差 < 1e-12）；hb_core.planner＝V2.plan 的推廣，原版參數時每期名單相同（斷言）。',
 'costs': {'gross': '零成本（扣成本前）', 'normal': '買 0.1855%、賣 0.4855%（手續費 0.0855%＋稅 0.3%＋滑價 0.1%）',
           'stress': '一般每邊再加 0.3%（買 0.4855%、賣 0.7855%）——選版與判定用', 'flat_stress': '舊口徑買賣各 0.5%（ia_core.rates("stress")，只報告）',
           'cap10m': '1,000 萬容量情境：一般成本＋每檔每邊 2.5%×sqrt(成交額/60 日均成交值)，帳戶固定 1,000 萬（只報告）',
           'breakeven': '單邊滑價 s 使挑選期（訊號窗）對 0050 月均超額＝0 的 s*（二分法）；另報 t 對 0050 降到 2.0 的 s。判定用「s*（版本）− s*（原版）」。'},
 'periods': {'selection': '訊號日 2006-01～2018-12（第一個可用訊號日 2006-04；評價到 2019-01 第一個訊號日收盤）', 'validation': '訊號日 2019-01～2023-12（只報告）',
             'holdout': '訊號日 2024-01～2026-10-12（只報告）', 'cross_cycle': '2006-04～2026-10-12'},
 'exploration': ['(1) 原版周轉拆解：賣出＝出場名額＋續抱股減碼；買進＝新進名額＋續抱股加碼（扣成本前與一般成本）',
                 '(2) 舊持股下月排名轉移矩陣（1～10、11～15、16～20、21～30、31～50、51+、不合格）',
                 '(3) 舊股（上月原版持股、本月掉出原版前 10）依排名桶（11～15、16～20、21～30、31+）× 52 週比桶（≥0.95、0.90～0.95、0.85～0.90、<0.85）× 最新單月營收年增率是否在合格池前 1/3，與當月新進前 10 名比較下一窗扣成本前報酬（成交日開盤→下一成交日開盤），按月分群 t',
                 '(4) 1,000 萬資金：新買名額部位（100 萬）超過 60 日均成交值 1% 的比例',
                 '(5) 原版持股依「已連續持有月數」分群的下一窗報酬（資訊消化速度）'],
 'predictions': {'P1': '62% 周轉中 10～15pp 來自續抱股調回等權；只做漂移帶即可降到約 50%，扣成本前配對差 |Δ| < 0.03pp／月',
                 'P2': '挑選期：排名 11～30 且 52 週比 ≥0.9 的舊股 − 新進前 10，差 ≥ −0.2pp（t > −1）；52 週比 < 0.85 的舊股顯著較差（t < −2）。若前者 t < −2 → 慢漂移機制在本策略不成立、緩衝題永久關閉',
                 'P3': '條件式續抱＋漂移帶：周轉 35～40%，壓力成本下對原版配對差 > 0，損益兩平單邊滑價提高 ≥ 0.05%',
                 'P4': '高成本情境改善較大；V 型反彈與領漲快速輪動期（2009、2020-04～06）改善較小甚至為負；S1 改善最大'},
 'model_family': {
   'A': {'A15': {'K': 15}, 'A20': {'K': 20}, 'A25': {'K': 25}, 'A30': {'K': 30}},
   'B': {'B20h85': {'K': 20, 'h': 0.85}, 'B20h90': {'K': 20, 'h': 0.90}, 'B20h95': {'K': 20, 'h': 0.95},
         'B30h85': {'K': 30, 'h': 0.85}, 'B30h90': {'K': 30, 'h': 0.90}, 'B30h95': {'K': 30, 'h': 0.95}},
   'C': {'C03': {'delta': 0.03}, 'C05': {'delta': 0.05}, 'C10': {'delta': 0.10}},
   'D': {'D20': {'band': 0.20}, 'D40': {'band': 0.40}},
   'E': {'E2': {'overlap': True}},
   'X': '跨族組合最多 4 個（B 或 C 疊 D），依挑選期探索與迭代結果決定，名稱 X1～X4',
   'max_versions': 20,
   'rules': 'B：排名 ≤K（rank 1 起算）且訊號日 52 週比 ≥h 的上月持股續抱，其餘依排名補滿 10 檔；C：上月持股綜合分數 +δ 後重排取前 10；D：續抱股市值在等權目標 ±band 內不調整，其餘調到共同目標；E：兩個各半資金的子組合交錯每兩個月換股（各自持有 10 檔兩個月）。全部沿用開盤漲停遞補（未持有者）、停牌凍結、X 出清。'},
 'iteration_limit': '迭代 3～8 輪（每輪一批版本），只能用挑選期；族內參數只能依挑選期探索調整；全部版本列入 iterations.json 與多重檢定帳。',
 'selection_rule': '挑選期、壓力成本、對原版（V3 n10）訊號窗配對 t 最高，且挑選期平均單邊周轉在 35～45%；選中版本在同族鄰近參數（同族中參數差一格者）的壓力成本配對差不得為負，否則退回較保守的鄰居（續抱較少者：K 較小、h 較高、δ 較小、band 較小）。',
 'final_thresholds': {'T1': '挑選期一般成本配對差 > 0', 'T2': '挑選期壓力成本配對差 ≥ 0', 'T3': '損益兩平單邊滑價（對 0050 超額＝0）提高 ≥ 0.05%',
                      'T4': '挑選期拿掉最好 3 個月後（一般成本）配對差 ≥ 0', 'T5': '挑選期扣成本前配對差 ≥ −0.05pp／月',
                      'cap': '全部通過時上限＝待觀察（2019 後的主策略資料已看過；正式改版要等 2026-11 起的前瞻盲測）'},
 'report': '和主策略 V3 比：跨循環年化、對 0050 t、最大回撤、2024～2026、AB（只用已公告營收，Monte Carlo 20 次）、成本損益兩平點、端點敏感度（拿掉最後 1 月、最好 1／3／5 月、最好年份、日曆月 t）、Bonferroni p（m＝主策略累計改版 10＋本題版本數）；穩健性：分年、分產業、去掉最大貢獻股。',
}
txt = json.dumps(PLAN, ensure_ascii=False, sort_keys=True)
res['PLAN'] = PLAN
res['PLAN_sha256'] = hashlib.sha256(txt.encode('utf-8')).hexdigest()
res['PLAN_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(res['PLAN_sha256'], res['PLAN_time_utc'])

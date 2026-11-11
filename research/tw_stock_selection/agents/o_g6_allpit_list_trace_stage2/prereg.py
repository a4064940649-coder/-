# -*- coding: utf-8 -*-
"""第 0 步 PREREG（A 動工前寫死）：寫入 results.json 的 PREREG 區塊並記 sha256。python3 -I -B prereg.py"""
import json, hashlib, datetime, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, 'results.json')
R = json.load(open(P)) if os.path.exists(P) else {}
assert 'PREREG' not in R, 'PREREG 已存在，不得覆寫'
PR = {
 'topic': 'o_g6_allpit_list_trace_stage2：G6 第二階段（ALLPIT 名單層獨立重建＋1,000 萬／壓力兩格逐筆追蹤＋紙上處置遮罩全期統一）',
 'S1': '不產生新選股訊號，G3 族群 N 不變（2,573）。',
 'a_thresholds': {
   'divergence': '關鍵數字差 > 0.1pp 或 t 差 > 0.05 列分歧（逐格最差層級與逐 j 層級並報）',
   'replace': '原引擎錯且修正後改變 > 0.1pp → 取代正式數字並附歸因；舊數字移附錄',
   'g6_recheck': 'G6 在複製完成後重新核對',
   'list_P1': 'REAL_{opt,p90,carry} 逐槽 Jaccard ≥ 0.99、相異 ≤ 5 個換股日×股票、相異只在 2013～2018 opt 補缺槽與 A_UB 觸發月；相異落在 2006～2012 orig 段或 2019 後 → P1 否定',
   'paper_P2': 'opt 22.06%／t 1.59、p90 23.56%、carry 25.38%；差 ≤ 0.1pp／t ≤ 0.05，最差仍是 opt',
   'P3': '1,000 萬第一筆不同交易落在參與上限／順延判定量 binding 日；壓力格落在壓力成本套用位置；若第一筆不同在 100 萬也會走到的路徑（除權息、停牌、E23）→ P3 否定',
   'P4': '修正後對正式 6 格影響 ≤ 0.15pp，G1～G8 判定不翻',
   'P5': '2026 端點與年化口徑對齊後系統差 ≤ 0.1pp；仍 > 0.3pp → 改判引擎或資料差並追到交易層',
   'P6': '兩格逐 j 差中位數 |·| ≤ 0.05pp，最大絕對值 ≤ 0.2pp',
 },
 'b_roles': {
   'rule': 'A 重建、B 追蹤與裁定、C 覆核須由三個不同子代理擔任，不得兼任；比對由研究員以獨立比對程式執行；A 凍結前看不到目標數字',
   'environment_fact': '本研究員子代理的工具清單沒有可產生子代理的工具（無 Agent/Task 工具；create_session 開的雲端容器沒有本機面板與 scratchpad 資料）。因此 A／B／C 只能由同一子代理依序扮演，角色分離不成立。',
   'consequence_prereg': '依 (b)：角色分離不成立 → 本輪 G6 第二階段最多判「部分進展」，G6 維持 ✘；所有裁定標「待不同子代理覆核」；不得以本輪結果取代正式數字（除非是可由原始資料直接驗證的帳務錯，且仍標待覆核）。',
   'contamination_disclosure': 'A 動工前，本子代理已讀過：o18 SPEC.json／FINDINGS 開頭附註（規格，允許）、o18 v7x_run.py 前 80 行（引擎呼叫，非名單邏輯）、複製員 rengine.py／rrun.py／rstats.py（引擎與統計，非名單邏輯）、任務文字中的目標數字（22.06／1.59、23.56、25.38）。A 的「凍結前看不到目標數字」因任務文字本身已列出而不成立。名單邏輯程式（pclib、o18lib、v7x_lib、ia_core、bflib、run_paper）在登記表凍結前不讀。',
 },
 'c_frozen_conventions': {
   'annualization': '日曆年化（o19 凍結口徑）：(nav_b/nav_a)^(365.25/天數)−1；紙上另報 ia 口徑（年數＝窗數/12）並列，揭盲後不得更換主口徑',
   'j_summary': 'j 中位數為主欄，同報 j p10',
   'variant_worst': 'opt／p90／carry 逐格取最差',
   'per_j_report': '逐 j 差分布：中位數、最大絕對值、最差變體是否換人',
 },
 'd_mask_assertion': '紙上與可執行都用「公告日 ≤ d 且處置期涵蓋成交日 e」；凍結前於程式內斷言 2006～2026 每個 (d,e) 的遮罩都依此口徑產生（不得只看 d）。',
 'e_lessons_o18_o19': [
   '不利端逐格取最差', '凍結前不算正式統計量', '事後新增狀態與嚴格版並列，嚴格版為主欄', 'raw/ 只放未修改原始檔',
   '假設區間與抽樣區間並報', '取代正式數字附歸因', '紙上附成本兩平點', '可執行測到 30 萬', 'G6 角色必須不同子代理',
   '關鍵中間檔不得只放 scratchpad', '比對報逐 j 差分布', '文字證據核對主詞', '紙上與可執行遮罩口徑統一', '口徑凍結後不得換'],
 'o1_lessons': ['事件 t 按事件日分群', '驗證期污染', '格點邊緣贏家詛咒', '成交可行性寫進事前登記', 't 提升要拆均值與波動', '存活者偏誤'],
 'f_readonly': 'live_tracking/main_forward、o_forward_infra_live 只讀，開始／結束各記 sha256（ro_snapshot_start/end.json），避開每日 06:49 台北排程與換股寫入時段；不碰 o_g8p2a_protocol_addendum/，不呼叫其判定器。',
 'forbidden_imports': ['pclib', 'cz_engine', 'cz_v7x', 'v7env', 'x_exec', 'ia_core', '任何研究員 .py'],
 'target_generation_note': '原引擎逐筆成交紀錄在 rep_o19 不存在（rep_o19 實為 o19 複製員 rengine 的 cache：navs_*、rdata_*、stats_*）。若需原引擎逐筆紀錄，研究員以「目標產生器」身分在獨立子行程執行原程式（標記 K 角色），其程式碼不得被重建／追蹤程式 import。',
 'plan_order': ['0 PREREG', 'A 規格解讀登記表（只讀規格文字與資料檔）→ 凍結 sha → 一次跑名單與紙上', 'B 名單層無關的引擎分歧：逐 j 淨值對齊 → 第一個分歧日 → 逐筆', 'P5 2026 端點', 'C 覆核（同一子代理，另寫帳務重算程式）', '揭盲比對', '收尾、刪 scratchpad 中間檔'],
 'disk': '磁碟約 2.9GB；大型中間檔放 scratchpad/o_g6s2 並於結束刪除；rep_o19 不刪。',
}
R['PREREG'] = PR
R['PREREG_written_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
R['PREREG_sha256'] = hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
json.dump(R, open(P, 'w'), ensure_ascii=False, indent=1)
print(R['PREREG_sha256'], R['PREREG_written_utc'])

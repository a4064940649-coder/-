#!/usr/bin/env python3
"""事前計畫（PLAN）：在算任何報酬之前寫入 results.json。已存在就拒絕覆寫。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE / 'results.json'
res = json.loads(RES.read_text(encoding='utf-8')) if RES.exists() else {}
assert 'PLAN' not in res, 'PLAN 已存在，不覆寫'
fz = json.loads((HERE / 'data_freeze.json').read_text(encoding='utf-8'))
assert not (HERE / 'explore.json').exists() and not (HERE / 'iterations.json').exists(), '已經有報酬結果，PLAN 必須先寫'

PLAN = {
 'question': '主策略（排除金融 n10、gapfix、v2 正式）要不要排除處置股？用 2006-04～2021-08 的 TWSE 歷史處置名單當挑選期重新判定；2021-09 以後只報告。',
 'seen_periods': {
   '2021-09～2026-10 組合層': 'o_rev_survivorship 執行審查：不排除處置股年化 +5.1pp（ORIG 排除金融 n10），S4 t 0.38→0.78。已看過 → 只報告，不選版，判定上限待觀察。',
   '2006～2021-09 近似事件層': 'disposal_x_main：近似處置事件在主策略前 100 名的報酬（t_month 1.09）；disposal_bear_longhistory：近似事件策略。已看過事件層、未看過組合層。',
   '2021-10～2026 真實名單事件層': 'disposal_v2、disposal_reconcile、combo_search（處置第一天買、最後一天賣）。',
   '本題第一次看的': '2006-04～2021-08 以真實歷史名單做的組合層「排除／不排除」比較。'},
 'data_gate': {
   'list': fz['list_csv'], 'list_sha256': fz['list_sha256'], 'source_A': fz['source_A']['repo'],
   'agree_disposal_5y_2021-09_2026-10': fz['agree_disposal_5y'],
   'agree_infra_v3': fz['agree_infra_v3'],
   'gate_rule': '一致率 ≥ 90%（第 1 題名單尚未產出 → 以 2021-09～2026 與 disposal_5y 的重疊一致率代替，並在結束前再檢查第 1 題名單）；逐年事件數 ≥ 2022～2025 年均（' + str(fz['post2021_annual_mean_2022_2025']) + '）的 50%。',
   'gate_years_fail': [int(y) for y, v in fz['gate_50pct'].items() if not v['pass50']],
   'handling': '主名單＝A（官方封存）。未過逐年門檻的年份另跑「A＋近似規則補」（標明近似名單），判定必須在 A 與 A+P 兩者都成立。'
               '說明：A 每年檔的 len(data)==TWSE 回傳 total；2021 全年 336 列與 audichuang 獨立抓取一致；低事件年份是當年規則（連續五次）造成，近似規則在 2011～2014 精確率 < 0.3，補進來多半是假陽性。'},
 'universe_engine': 'main_v2/v2_core.V2(gapfix, exfin=True)、正式同分序、n10 主、n30 附報；引擎 drj_core.simulate2（純量成本時與 ia_core.simulate 逐日相同，validate.py 斷言）。'
                    '成本：一般＝買 0.0855%+0.1%、賣 0.0855%+0.3%+0.1%；壓力＝一般＋處置中交易每邊再加 0.5%（另報 v2 的全面 0.5/0.5）。',
 'disposal_status': '訊號日 d：公告日 ≤ d 且 start ≤ d ≤ end（＝現行 SOP 程式）。rem＝[e, end] 面板交易日數（e＝成交日），L＝該次處置面板交易日數，thr＝ceil(0.6L)（10 日→6、5 日→3、新制 5 日→3）。'
                    'nth：第二次處置／收足全部款券／內容含「曾發布處置」／督導會報 → 2+；其餘 1。連續處置重疊取較晚 end、較高 nth。',
 'versions': {'E1': '全部排除（現行 SOP；基準）', 'E0': '全部不排除', 'E2': '只排除第二次以上', 'E3': '只排除 rem ≥ thr',
              'E4': '不排除，但處置股買進延到處置結束後第一個交易日開盤（期間資金放 0050；0050 不計買進日與賣出日的報酬；開盤漲停或又進入已公告的新處置 → 再延；下次換股前沒進場就併入換股）',
              'E5': 'E2 ∪ E3（第二次以上或 rem ≥ thr 就排除）'},
 'neighbors_report_only': 'E3 門檻 0.5L、0.7L；n30 各版。',
 'selection_rule': '只用挑選期（訊號日 2006-04～2021-08）：E0/E2/E3/E4/E5 中，對 E1 的配對差（訊號窗月報酬差，一般成本）t 最高，且挑選期最大回撤不比 E1 深 3pp 以上者。',
 'predictions': {
   'P1': '挑選期 E0−E1 配對差 > 0，月份分群 t ≥ 1.5；否則 SOP 維持排除。',
   'P2': '剔除股（E0 入選、E1 剔除）的持有期報酬（e 開盤→下一 e 開盤，減同月其他入選股平均）：rem 越少越高（R0: rem=0、R1: 1≤rem<thr、R2: rem≥thr）。',
   'P3': '第一次 > 第二次以上；10% 漲跌幅（2015-06 起）> 7%；加權指數在 200 日均線上 > 下；營收分數前半 > 後半。另報制度期：預收期（2010-11 前，無分盤）vs 分盤期。',
   'P4': '安慰劑：E0 n10 入選、訊號日不在處置（且過去 30 個交易日沒有處置），但過去 10 日內最大 6 日還原漲幅 ≥ 25%（近似未達標組）。其事件層配對差絕對值 < 處置組的一半，否則判不可用。2021-09 後另報「注意股未處置」（只報告）。',
   'P5': '不排除多出來的報酬若 > 一半來自前 3 檔股票 → 判定不高於待觀察，並列出。'},
 'event_tests': '事件窗 3 種（full：e 開盤→下一 e 開盤；in：e 開盤→處置結束日收盤；post：處置結束後第一日開盤→下一 e 開盤）× 子群 14 種（全部、nth1、nth2+、R0、R1、R2、多頭、空頭、營收前半、營收後半、7%、10%、預收期、分盤期）；'
                'in 窗對 R0 不定義。檢定量＝事件報酬減同月同窗其他入選股平均，按換股月平均後 t。Holm 修正，總數＝實際可算的格數（上限 42）。',
 'residualize': '合併挑選期 E0 入選股（n10）：月內去均值的持有期報酬，對月內去均值的 52 週比百分位、營收百分位、log 60 日均成交值、電子業、處置 dummy 做 OLS，月份群聚標準誤。',
 'judgement': {
   '(1)機制': 'P1 挑選期配對差 > 0 且 t ≥ 1.5；P4 安慰劑不成立。任一不成立 → 不可用，SOP 維持排除。名單 A 與 A+P 都要成立。',
   '(2)選中版本': '2021-09～2026 配對差 > 0（只報告）；跨循環年化 ≥ E1 且對 0050 t 不低於 E1（V3 資料若第 1 題完成就用 V3，否則用現行面板並註明）；壓力成本下挑選期配對差 > 0；剔除貢獻最大 3 檔後挑選期配對差 > 0。',
   '(3)': '全部通過最高只判待觀察；前瞻：live_tracking 2026-11 起記錄入選股是否處置股、E1 與選中版本紙上淨值差；新制滿 12 個月（2027-08）重評，紙上差 ≥ 0 才建議改 SOP。',
   '(4)': '報告最小可偵測效果（MDE＝2.8×配對差標準誤）；受影響名額 < 60 寫明檢定力不足。',
   '(5)': '不論結果都交 SOP 文字建議與對處置股策略 20% 部位的重複曝險評估。'},
 'periods': {'selection': '訊號日 2006-04～2021-08', 'validation(seen)': '2021-09～2023-12', 'holdout(seen)': '2024-01～2026-10-12', 'cross_cycle': '2006-04～2026-10-12'},
 'max_versions': 6,
}
txt = json.dumps(PLAN, ensure_ascii=False, sort_keys=True)
res['PLAN'] = PLAN
res['PLAN_sha256'] = hashlib.sha256(txt.encode('utf-8')).hexdigest()
res['PLAN_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(res['PLAN_sha256'], res['PLAN_time_utc'])

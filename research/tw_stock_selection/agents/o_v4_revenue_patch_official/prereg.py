#!/usr/bin/env python3
"""凍結：把 V4 規格、資料 sha256、必報項目、預測與判定規則寫進 results.json 的 PREREG（含 sha256 與時間戳）。
之後 run_v4.py 斷言 PREREG 存在且 data_freeze_v4.json 未變，才計算任何 V4 報酬。"""
import sys, json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
fz = json.loads((HERE / 'data_freeze_v4.json').read_text(encoding='utf-8'))
P = {
 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
 'data_freeze_v4_sha256': sha(HERE / 'data_freeze_v4.json'),
 'R_FIX_sha256': fz['R_FIX']['sha256'], 'Y5_FIX_sha256': fz['Y5_FIX']['sha256'], 'merge_rule': fz['merge_rule_selected'],
 'V4_spec': {
  'universe': '上市日線面板（含已下市）＋P（delist_prices.parquet）；換股訊號日＝每月第一個日 ≥ 11 的面板交易日（v2：營收面板最後完整月之後與 2026-10-12 起不交易）；成交日 e＝訊號日次一面板交易日',
  'pool': '60 日均成交值（gapfix 窗）前 300 且 ≥ 1,000 萬、收盤 ≥ 10；52 週比（252 日 gapfix）有值；成交日有開盤；排除金融（v2 FINMASK ∪ 終止上市金融股：合庫、元京證、金鼎證…）；排除 KY／DR／F-／91xx；排除處置股（成交日口徑）；y3 有值',
  'disposal': '官方名單＝o_disposal_rejudge/disposal_hist_rejudge.csv（in_panel）∪ disposal_5y 獨有列；主口徑 trade：公告日 ≤ 訊號日 d 且 start ≤ IDX[e] ≤ end → 排除；敏感度 signal：公告日 ≤ d 且 start ≤ d ≤ end',
  'revenue': '訊號日 < 2021-03-01：R_FIX（yoy_long_R_FIX.parquet＝R_DEL＋本題新聞修補 45 格）；≥ 2021-03-01：Y5_FIX（＝rev_5y yoy，本題 0 格修補）；y3＝m−1、m−2、m−3 三個月年增率平均（三個都要有）',
  'score': '0.5×52 週比池內百分位＋0.5×y3 池內百分位；正式同分序（分數 → y3 高 → 代碼欄位序）；n＝10；開盤漲停（tick 價）跳過遞補',
  'exec_cost': '隔日開盤成交；一般成本：買＝手續費 0.1425%×0.6＋滑價 0.1%，賣＝同＋證交稅 0.3%；壓力：每邊 0.5%；二分法解調倉後總值',
  'X': 'merger_consideration.csv（main）：現金併購按對價、換股／轉換按最後收盤、經營惡化歸零；代碼重用規則照 o3_core',
  'period': '日頻 2006-04-01～2026-10-12；t＝訊號窗（訊號日收盤→下一訊號日收盤）對 0050 含息的月超額 t（窗起點落在區間內）',
  'AB': '只用已公告營收：v2_core.make_unavail("mc", seed) seed 0～39，40 條日報酬平均成一條淨值（同 main_v2）；另報 worst',
  'bounds': 'gap＝終止上市名單（4 碼）中非 KY／DR／F-、非金融者；訊號日 ≤ 終止上市日、營收以外都合格但 y3 缺 → 營收百分位固定 0.5（B05）或 1.0（B10；同分序視為營收最高）放回；B_Q＝改用季營收代理 y3（yoy_qproxy.parquet）放回，無代理者仍排除（估計值，不是界限）'},
 'main_decision': '以「AB、一般成本」的 V4 對 0050 t 判級：≥ 2.58（m＝10）第 1 級；2.30～2.58 第 1 級邊緣；< 2.30 正式降第 2 級，SOP 預期年化改用 V4 保守口徑（AB＋B10）。拿掉最好 1 個月後 t 若跌過一級 → 加註「端點敏感」',
 'columns_to_report': ['V4（主，一般）', 'V4 壓力', 'V4 AB（mc40）一般／壓力', 'V4 AB worst', 'V4 B05', 'V4 B10', 'V4 AB＋B05', 'V4 AB＋B10', 'V4 B_Q（估計）', 'V4 AB＋B_Q',
                       'V4 訊號日處置口徑（敏感度）', 'V4 用 R_DEL（R_FIX 的效果）', 'V4 不含 2013 年修補格（IFRS 敏感度）', 'V4 fin2 關＋不明文排除 KY（規格明文化的效果）',
                       'o3 重現（R_DEL、訊號日口徑、fin2 關）＝26.95%／2.55'],
 'stats_to_report': ['跨循環年化、最大回撤、對 0050 t、對同池等權 t、月均超額與標準差', 'S1 2006-04～2013、S2 2014～2020、S3 2021-03～2026-10、S4 2024～2026-10；SEL 2006-04～2018、VAL 2019～2023、HOLD 2024～2026-10',
                     '逐年對 0050', 'NW t（lag 3／6／12）', '12 個月區塊 bootstrap（10,000 次，單尾 p）', '端點：拿掉最後 1 個月、最好 1／3／5 個月、最好年份；日曆月 t',
                     '成本損益兩平：單邊滑價 0.10%～0.40% 格點，t＝2.3、2.0 的滑價（線性內插）；手續費全額（＋0.1% 滑價）', 'm＝10 Bonferroni p＝10×單尾常態 p',
                     'main_v2 同分序區間：隨機同分 rand:0～19 的年化與 t 範圍', '與 main_v2 正式淨值、與 o3 V3 官方名單版的逐月配對差',
                     '分年、分產業貢獻、去掉最大貢獻股', '修補股入選股月與下月超額（按月分群，對其他入選股）'],
 'predictions': {
  'P1': '修補後 2013～2018 每月合格池平均增加 4～6 檔；2006～2012 < 1 檔',
  'P2': 'n10 實際選進的修補股股月 ≤ 10，集中在 S2；S1、S3、S4 年化變動 < 0.2pp',
  'P3': 'V4 一般成本跨循環年化落在 24.27%～26.95%，點估計 25.5～26.5%，t 2.3～2.5；< 24.27% 要逐檔核對；> 26.95% 表示併購溢價通道成立',
  'P4': '處置日改成交日口徑影響 ≤ 0.2pp；AB 約少 0.3～0.5pp；損益兩平單邊滑價約 0.11～0.14%',
  'P5_falsifiable': '修補股入選股月的下月超額若顯著不同於其他入選股（按月分群 |t| > 2）＝「選股力不受偏誤影響」的反證'},
 'explore_known_before_prereg': {'R_FIX 新增池內可算 y3 股票月': 6, 'R_FIX 2013～2018 每換股日新增合格股': 0.069, '2013～2018 每換股日仍缺營收的合格股': 8.5,
                                 '2006～2012 每換股日仍缺營收的合格股': 5.27, 'B_Q 2013～2018 每換股日可代理': 5.35,
                                 '處置口徑不同的名額': 155, '其中分數前 10': 10},
 'versions_tried_count_for_multiple_testing': {'this_topic_data_rule_versions': 6, 'strategy_spec_versions': 1,
                                                'main_strategy_prior_revisions': 'main_refine、main_v2、gapfix、integrate_o1／o2、o3（≥ 6 輪）→ m＝10'},
 'replication': '獨立複製員只讀 data_freeze_v4.json 列出的檔案與本規格；V4 主欄淨值逐日最大相對差 < 1e-9，B05／B10、AB 欄也要重現',
}
canon = json.dumps(P, ensure_ascii=False, sort_keys=True).encode()
res = {'PREREG': P, 'PREREG_sha256': hashlib.sha256(canon).hexdigest()}
f = HERE / 'results.json'
assert not f.exists() or 'PREREG' not in json.loads(f.read_text(encoding='utf-8')), 'PREREG 已存在，不得覆寫'
f.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(res['PREREG_sha256'], P['created_utc'])

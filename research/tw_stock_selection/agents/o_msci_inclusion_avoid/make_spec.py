#!/usr/bin/env python3
"""寫出前瞻紙上追蹤規格 agents/live_tracking/msci_inclusion/SPEC.json（含 sha256），與空白 reviews.tsv。只能在 PREREG 之後跑。"""
import os, json, hashlib, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
LT = '/home/user/-/research/tw_stock_selection/agents/live_tracking/msci_inclusion'
RES = json.load(open(os.path.join(HERE, 'results.json')))
assert 'PREREG' in RES
spec = {
 'title': 'MSCI 新納入股：生效日後回吐（事件層）與主策略 B2「E 收盤賣出、持有 0050 到下一換股日」（策略層）前瞻紙上追蹤',
 'origin': 'agents/o_msci_inclusion_avoid（PREREG sha256 ' + RES['PREREG']['sha256'] + '）',
 'why_paper_tracking': '2019～2026 的樣本外已被 o1 看過，不算盲測；2026-11 審核起的資料是唯一乾淨樣本。',
 'start_review_eff_on_or_after': '2026-11-01',
 'first_review': '2026-11 半年度審核（預計台灣時間 2026-11 上旬～中旬清晨公布，2026-11-30 收盤生效；以實際公告為準）',
 'data_entry': {
   'file': 'agents/live_tracking/msci_inclusion/reviews.tsv',
   'when': '每次 MSCI 公告當天（2、5、8、11 月）；另外臺灣50（3、6、9、12 月）與高股息 ETF 納入當作安慰劑一併登錄',
   'columns': ['review', 'ann_tw', 'eff', 'cls', 'code', 'name', 'src'],
   'cls': {'STD': 'MSCI 全球標準指數新增（含由小型升級）', 'SC': 'MSCI 全球小型指數新增（不含由標準降級）', 'TW50': '臺灣50 新增（安慰劑）', 'HDETF': '0056/00878/00919/00713/00918/00929/00940 新增（安慰劑）'},
   'ann_tw': 'MSCI 公布當天的台灣日期（台灣清晨公布＝當天開盤前已知）',
   'eff': '台股最後一個生效交易日（收盤生效）',
   'src': '新聞網址（中央社、經濟日報、工商時報、鉅亨、MoneyDJ 至少一則）；來源視為不可信，交叉核對兩則以上'},
 'layer_i_events': {
   'universe': '上市、在面板、E−1 的 60 日均成交值 ≥ NT$1,000 萬、收盤 ≥ 10 元、E 日有成交',
   'record': ['E−10、E、E+5、E+20 收盤', 'E−10～E−1、E 日、E+1～E+5、E+1～E+20 的異常報酬加總（個股含息日報酬 − 上市等權日報酬）'],
   'primary_metric': 'STD 的 E+1～E+5 異常報酬，先按審核平均再對審核做 t',
   'expected_rate': '每年 STD 約 4～8 筆、SC 約 15～30 筆（2019～2026 平均）'},
 'layer_ii_strategy': {
   'strategy': '主策略正式版（排除金融，n10 為主、n30 為輔，main_v2 gapfix official）',
   'event': '持有期間（成交日 ≤ E ≤ 下一成交日前一日）持股遇到 STD 生效日 E',
   'paper_rule_B2': 'E 日 13:25～13:30 收盤集合競價賣出該股（收盤鎖跌停視為賣不出，延到下一日收盤）；隔日開盤買 0050；下一換股日開盤賣 0050，照原規則換股',
   'record': ['該股 E 收盤 → 下一成交日開盤報酬', '0050 同期報酬', '紙上淨值差 ≈ (1/n) × (0050 報酬 − 該股報酬 − 0.69% 額外成本)'],
   'counterfactual_also_record': 'A1（換股日已公告就跳過）只記錄，不作判定；2019～2026 回測顯示 A1／A2 年化 −0.6～−0.9pp，不建議'},
 'decision': {'min_reviews': 8, 'event_mean_le%': -2.0, 'event_t_le': -2.0, 'strategy_cum_diff_ge': 0.0, 'close_if_event_mean_gt%': -1.0,
   'text': '累積至少 8 次 MSCI 審核（約 2 年，2026-11～2028-08）：STD 事件層 E+1～E+5 分群平均 ≤ −2%、分群 t ≤ −2，且主策略層 n10 紙上累積差 ≥ 0 → 才建議寫進 SOP（仍須總監核准）。8 次後事件層平均 > −1% → 關閉此題。中間只報告，不提前判定。另：安慰劑（臺灣50、高股息 ETF）若也回吐 ≥ 1%，在報告中標註「無法區分指數資金流與近期漲多反轉」。'},
 'power_note': '主策略 n10 每年約 1 個受影響事件，8 次審核預期只有 2～4 筆，主策略層幾乎不可能達到顯著；主策略層的判定只看「累積差 ≥ 0」作為不傷害條件，主要證據來自事件層。',
 'program': 'python3 -I -B /home/user/-/research/tw_stock_selection/agents/o_msci_inclusion_avoid/live_track.py（先 --check-spec）；面板需先更新到 E+20',
 'outputs': ['events_ledger.csv', 'strategy_ledger.csv', 'status.json'],
}
body = json.dumps(spec, ensure_ascii=False, sort_keys=True)
sha = hashlib.sha256(body.encode('utf-8')).hexdigest()
doc = {'sha256': sha, 'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'spec': spec}
p = os.path.join(LT, 'SPEC.json')
if os.path.exists(p): raise SystemExit('SPEC.json 已存在，不可覆寫')
json.dump(doc, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(os.path.join(LT, 'reviews.tsv'), 'w', encoding='utf-8') as f:
    f.write('review\tann_tw\teff\tcls\tcode\tname\tsrc\n')
RES['LIVE_SPEC'] = {'path': p, 'sha256': sha, 'frozen_utc': doc['frozen_utc']}
json.dump(RES, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print(sha, doc['frozen_utc'])

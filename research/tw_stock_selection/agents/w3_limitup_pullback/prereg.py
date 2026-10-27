#!/usr/bin/env python3
"""事前設定（看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '漲停回馬槍（首板漲停後縮量回檔、收盤不破漲停日開盤）',
 'sources': ['https://www.bilibili.com/read/cv50847678', 'https://caifuhao.eastmoney.com/news/20260711143433205026560'],
 'type': '事件型：主報告＝事件報酬（扣成本、對同池等權與上市等權的異常報酬、t 按進場月份分群）；另報日曆時間組合（年化、日頻最大回撤、對 0050／同池等權月超額 t、逐年）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。先剔除「有值檔數 < 300」的日子（2018-01-13）再做任何 rolling；flag 5 價格視為缺值；Day0 與訊號日需 flag ∈ {0,1,2}（flag 3/4 = 公司行動／前日缺口，漲停判斷不可靠，不取）',
   'stocks': '4 碼數字代碼、排除 91xx（存託憑證）',
   'volume': '量 ＝ value / close_raw（股數）；停牌日量缺值',
   'price_compare': '所有跨日價格比較用還原價：還原收盤 adj，還原開盤＝open × adj / close_raw（避免除息造成假跌破）',
 },
 'limit_up': {
   'reference': '參考價 ref_t ＝ close_raw_t / (1 + ret_t)（一般日＝前收；除權息日＝除權息參考價近似）',
   'limit_price': '漲停價 ＝ ref × (1 + 漲幅)，依升降單位無條件捨去；漲幅 2015-06-01 前 7%、之後 10%；升降單位 <10:0.01、10–50:0.05、50–100:0.1、100–500:0.5、500–1000:1、≥1000:5（以 ref×(1+漲幅) 所在級距）',
   'close_limit_up': '收盤漲停 ＝ close_raw ≥ 漲停價 − 0.5 個升降單位 且 close_raw ≥ high × 0.999',
   'open_limit_up': '開盤漲停 ＝ open ≥ 漲停價 − 0.5 個升降單位（進場日此情形 → 買不到、放棄該訊號、不遞補）',
 },
 'universe': {
   'liquidity': 'Day0 與訊號日：60 日均成交值（上市後停牌日記 0，滿 60 筆）≥ NT$1,000 萬且原始收盤 ≥ 10 元',
   'disposal': '處置股：disposal_5y（2021-09 起才有官方名單）中，訊號日或進場日落在處置期間 → 排除；2021-09 前無資料無法排除（限制）',
   'full_delivery': '全額交割股：面板無此欄位，無法排除（限制；這類股票多半流動性差、股價 < 10，大多已被流動性池擋掉）',
   'attention': 'attention_5y（2021-10 起）記錄 Day0 是否為注意股，只做分組報告，不排除',
 },
 'signal': {
   'day0': 'Day0 收盤漲停；Day0 量 ≥ 前 20 日（Day-20..Day-1，至少 15 筆有值）均量 × 2；Day-10..Day-1 無其他收盤漲停（首板）',
   'pullback': 'Day1..DayW（主規格 W＝3）中第一個同時滿足：adj_k < adj_Day0；adj_k ≥ Day0 還原開盤；adj_k ≥ adj_Day0 × 0.92；vol_k ≤ Day0 量 × r（主規格 r＝0.5）。該日收盤算訊號日 s',
   'invalidation': '觀察窗內若在訊號出現前已有一天收盤 < Day0 還原開盤（已破）→ 該 Day0 作廢（「不破」逐字解讀）；觀察窗內其他不符合的日子（例如續漲、量太大）直接跳過繼續看',
   'no_low_note': '面板無最低價，「不破」只能以收盤判斷',
 },
 'execution': {
   'entry': 's+1 開盤（還原開盤）買；s+1 無開盤價或開盤漲停 → 放棄',
   'exit': '從進場日 e 起每天收盤檢查（e..e+H−1）：adj < Day0 還原開盤 → 停損；adj ≥ 進場價 × 1.10 → 停利；任一觸發 → 隔日開盤賣；都沒觸發 → e+H 開盤賣（主規格 H＝5）。同日兩者不可能同時成立',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤。面板無最低價，無法判斷開盤跌停鎖死賣不掉（限制）',
   'dedupe': '同一檔持有中（尚未到出場開盤日）的新訊號忽略',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋證交稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%（來回 1%）',
 },
 'variants': {
   'PRIMARY': 'r＝0.5、W＝3、H＝5、首板',
   'grid': 'r ∈ {0.5, 0.6} × W ∈ {3, 5} × H ∈ {3, 5, 10}（共 12 組，含主規格），全部報告',
   'BOARD2': '二板回檔版：Day-1 與 Day0 皆收盤漲停、Day-11..Day-2 無其他漲停；量條件套在 Day0（≥ Day0 前 20 日均量 × 2）；其他同主規格（r＝0.5、W＝3、H＝5）',
   'rule': '判定只看 PRIMARY；其他只作穩健性參考；不做任何事後調參',
 },
 'event_stats': {
   'raw': '出場價 / 進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_pool': 'net − 同期間（進場開盤→出場開盤）同池等權報酬；同池＝前一日符合流動性條件的上市普通股，每日等權、開盤對開盤（判定用）',
   'abnormal_listed': 'net − 同期間上市等權（所有有開盤價的上市普通股，開盤對開盤）',
   'vs_0050': 'net − 0050 含息（s 收盤→出場前一日收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件、等權每日再平衡；進場日報酬＝收/開，出場日＝出場價/前收；進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收對收）月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'src3_only': '持有期碰到 src 3（2024-02-15~2025-09-23）的事件另報', 'valid_ex_src3': '驗證期剔除碰到 src 3 的事件',
             'data2021': '處置／注意股資料只有 2021 起：另報 2021-01~2023-12 與 2024-01~2026-10'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month ≥ 2，且「至少不輸 0050」＝日曆組合淨年化 ≥ 0050 同期年化 → 可用；只有一段過 → 待觀察；都沒過 → 不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料', 'src 3 收盤為 13:20 近似，收盤漲停判斷與量都可能失真',
                  '無最低價：不破與停損都以收盤判斷', '2021-09 前無處置名單、全程無全額交割名單', '漲停價以 ref 近似（除權息日參考價四捨五入差異）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

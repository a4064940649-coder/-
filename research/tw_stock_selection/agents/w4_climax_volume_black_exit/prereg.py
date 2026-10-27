#!/usr/bin/env python3
"""事前設定（在算任何報酬之前寫入 results.json 的 PREREG；檔案已有 PREREG 時拒絕覆寫）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '高檔爆量長黑出場（疊加在主策略 52 週高點 × 月營收 M2 上的出場濾網）',
 'sources': ['https://www.stockfeel.com.tw/?p=140476', 'https://www.cmoney.tw/notes/note-detail.aspx?nid=63810',
             'https://www.cmoney.tw/notes/note-detail.aspx?nid=9932', 'https://www.pocket.tw/school/articles/5186/',
             'https://www.moneyweekly.com.tw/_Article?AID=128558'],
 'type': '出場規則（疊加型）：主報告＝組合（有加 vs 沒加），報年化、日頻最大回撤、對 0050／同池等權月超額 t、逐年、換手與成本；另報事件層（被賣出股票從賣出到下一換股日的報酬、對同池等權異常、t 按觸發月份分群）',
 'base_strategy': {
   'code': 'revenue_longhistory/lhlib.py 的 Setup(yoy_long()).selection 邏輯（= exec_rebuild/exlib M2，長期營收版），參數一律不改',
   'rule': '60 日均成交值前 300 名（>= NT$1,000 萬、收盤 >= 10 元、非處置［名單 2021-09 起］）；分數 = 0.5×52 週高點比百分位 + 0.5×近 3 月營收年增率平均百分位；前 n 檔等權；訊號 = 每月第一個 >= 11 日交易日收盤，隔日開盤成交；開盤漲停（還原開盤／前收 − 1 >= 漲跌幅 − 0.6%，2015-06-01 前 7%、之後 10%）且未持有者跳過、下一名遞補',
   'n': '主規格 n=30（研究中心主策略）；n=10 另報',
   'revenue': '長期月營收 tw_revenue.csv（2005-01 起，自算年增率）→ 第一個換股日約 2006-04；2021 起另跑 rev_5y 版（實盤版）作診斷',
   'panel_fix': '有值檔數 < 300 的日子（2018-01-13）視為不存在後再算滾動值（lhlib 的 s52_fix／v60_fix，本研究的觸發指標同樣處理）',
 },
 'trigger_at_close_t': {
   'scope': '只檢查 t 日收盤時實際持有的主策略持股（含換股成交日當天收盤）',
   'C1_high_zone': 'adj_t / min(adj_{t-60..t-1}) − 1 >= 30%（前 60 個有效交易日最低還原收盤，min_periods 48；面板無最低價，用收盤）',
   'C2_volume_spike': {'V20': 'vol_t >= 2.5 × mean(vol_{t-20..t-1})（min_periods 16）',
                       'V240': 'vol_t >= 2.5 × mean(vol_{t-240..t-1})（min_periods 192）',
                       'V5': 'vol_t >= 2.0 × mean(vol_{t-5..t-1})（min_periods 4）',
                       'vol_def': 'vol = value / close_raw（股數）；value 缺值（停牌）不計入均量'},
   'C3_black_candle': {'BLK': '教學定義：close_raw_t < open_t 且 含息日報酬 ret_t <= −1.5%',
                       'OHLC': '開高走低：還原開盤_t > 前一日還原收盤 且 close_raw_t < open_t',
                       'flag': 't 日 flag ∈ {0,1,2}（公司行動 3／資料缺口 4／資料錯誤 5 不觸發）'},
   'specs': 'C2 三版 × C3 兩版 = 6 組，全部報告；PRIMARY = V20 × BLK（教學最常見寫法：20 日均量 2.5 倍＋跌幅 >= 1.5% 長黑）',
 },
 'execution': {
   'exit': 't 日收盤觸發 → t+1 開盤以還原開盤賣出；t+1 無開盤價或開盤跌停（開盤／前收 − 1 <= −漲跌幅 + 0.6%）→ 順延到下一個可賣的開盤',
   'refill': '賣出的資金在下一個換股日才遞補：換股日把總資產（持股＋現金／0050）調回 n 檔等權；空出期間持現金（主規格）或 0050（另報）',
   'idle_0050': '0050 只有含息收盤：賣出當日收盤買進 0050（買成本 0.0855%＋滑價），換股訊號日 d 收盤賣出（賣成本 0.0855%＋ETF 稅 0.1%＋滑價）；賣出日開盤到收盤、d 收盤到成交日開盤以現金計；若賣出日的隔日就是換股成交日則不買 0050',
   'rebalance_interaction': '換股成交日前一日收盤觸發（尚未賣出）的股票，該次換股排除不買、由下一名遞補；已在月中賣掉的股票，下一換股日若再入選照主策略買回；開盤漲停「未持有不買」的判斷以實際持有為準（被賣掉者視為未持有）',
   'cost': 'exec_rebuild/exlib.cost_rates：買 = 0.1425%×0.6 + 滑價；賣 = 0.1425%×0.6 + 稅 0.3% + 滑價；滑價 0.1%（主）、0.3%；壓力 買 0.5%＋賣 0.5%',
   'engine': 'lhlib.sim_portfolio 的換股邏輯（漂移後權重調回等權、成本內生），加上日內出場與閒置部位處理；不加出場時必須與 lhlib.sim_portfolio 完全一致（程式內 assert）',
 },
 'benchmarks': {'0050': '0050 含息收盤', 'EW': '同池等權（每次換股的合格股池等權，lhlib 作法，不扣成本）', 'BASE': '主策略不加出場（同 n、同成本）'},
 'windows': {
   'select': '第一個換股日（約 2006-04）～2019-04-30（價格類挑選期；主策略需要月營收，2005 年無法開始）',
   'valid': '2019-05-01～2026-10-22',
   'valid_2024_2025': '2024-01-01～2025-12-31',
   'rev2021_select': '2021-03-01～2023-12-31（2021 起資料口徑，報告用）', 'rev2021_valid': '2024-01-01～2026-10-22',
   'src3': '2024-02-15～2025-09-23（來源 3：開盤、收盤為 5 分 K 近似，K 棒顏色可能判錯）',
   'valid_official_only': '驗證期剔除 src3 期間',
   'sub_select': '2006-04～2012-12、2013-01～2019-04（挑選期內一致性）',
 },
 'statistics': {
   'portfolio': 'lhlib.daily_stats：日頻年化、日頻最大回撤、月窗（換股訊號日收盤 → 下一訊號日收盤）超額的 t（每月一筆＝按月份分群）',
   'overlay_minus_base': '同月窗的（加出場 − 不加）月報酬差，平均與 t；另報年化差、回撤差、年換手（賣出金額／淨值）、年成本（成本／淨值）',
   'event': '每筆出場：r_stock = 下一換股成交日開盤 ／ 賣出價 − 1；同期同池（上次換股的合格股池）等權開盤買進持有報酬 r_pool；abn = r_stock − r_pool；'
            '出場淨效益（現金版）= 1/(1+r_stock) − 1，若該股下一換股日又被買回再減一次來回成本；t 一律先按觸發月份平均再做 t',
 },
 'verdict_rule': ('PRIMARY（V20×BLK、n30、滑價 0.1%、閒置持現金）。某期間「通過」= (A) 加出場後的組合扣成本對同池等權月超額 t >= 2 且淨年化 >= 0050 同期年化（任務原文判定），'
                  '且 (B) 加出場相對不加出場有增益：（加 − 不加）月報酬差 t >= 2。B 是出場濾網本身的證據，沒有 B 時 A 只是主策略的功勞。'
                  '挑選期與驗證期都通過 → 可用；只有一段通過 → 待觀察；都沒通過 → 不可用；某段 PRIMARY 出場事件 < 30 筆或事件月份 < 12 → 無法測試。另列 (A) 單獨結果與其他 5 組規格作穩健性參考，不做事後調參。'),
 'known_limits': ['2019–2023 面板只有存活者', '2013 起長期營收只剩存活公司、2013 年 IFRS 口徑斷點（影響主策略選股，不影響出場訊號）',
                  '2024-01-02～2024-02-14 無資料（滾動視窗跨缺口）', 'src 3 期間開盤、收盤、成交值都是近似',
                  '面板無最低價，高檔判斷用 60 日最低收盤', '量 = value/close_raw，減資、分割前後口徑跳動', '跌停只看開盤，不知道盤中是否打開'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written', PREREG['written_at'])

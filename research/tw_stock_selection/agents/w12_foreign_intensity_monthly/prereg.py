#!/usr/bin/env python3
"""事前設定（看任何回測結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '外資買超強度月選股（北向資金增持因子台股版）',
 'category': 'chips',
 'sources': ['https://search.napstic.cn/literature/periodical/010tzyj202410007', 'https://www.cls.cn/detail/1550608',
             'https://www.nbd.com.cn/articles/2021-04-22/1712288.html', 'https://wallstreetcn.com/articles/3596985'],
 'source_rule_verbatim': '股池：流動性池，排除處置與全額交割。每月最後一個交易日收盤算 FI20 = Σ(過去 20 交易日外資買超張數 × 1000) / Σ(過去 20 交易日成交量)。'
     '取 FI20 最高的 20 檔等權，隔日開盤換股（exlib M2，開盤漲停遞補），持有 1 個月。敏感度：60 日版 FI60；取前 10%（十分位）；兩者都 > 0 的交集（20 日與 60 日同時買超）；'
     '排除市值前 50 大，避免只是權值股 beta。對手：0050 含息、同池等權；報十分位價差。',
 'proxy_note': '原始北向資金因子用「持股比例變化」（增持股數 / 總股本）。本資料股本只有 2025-09 起，無法回推歷史持股比例，'
               '改用「外資買超股數占同期成交量比例」當代理變數（FI＝參與率型強度），兩者概念不同：分母是成交量而非股本，偏向「外資占交易的比重」。',
 'type': '選股型（月換股）：組合報告（年化、日頻最大回撤、對 0050 日曆月 t、對同池等權日曆月 t、逐年、周轉）',
 'data': {
   'inst': 'extras/inst_5y.csv.gz foreign 欄（2021-10-08~2026-10-07，單位張）；某日某股無列＝0',
   'panel': 'scratchpad/panel 上市日線 adj、close_raw、open、value、flag；有值檔數 < 300 的日子剔除（2021 起無）；只用普通股：4 碼數字、非 0 開頭、非 91 開頭',
   'calendar': '面板日曆（2024-01-02~2024-02-14 面板無資料：該段 inst 日不進 rolling 窗，20/60 日窗直接跨過缺口；1 月底無換股，12 月底持股一路持有到 2024-02-29 訊號、03-01 開盤換股）',
   'volume': '成交量（股）＝value / close_raw；外資買超股數＝foreign（張）× 1000',
   'FI_N': 'FI_N(d) = Σ_{d-N+1..d} foreign×1000 / Σ_{d-N+1..d} 成交量；只計入該股當日 value 與 close_raw 都有值的日子（分子分母同一組日子）；有效日 < 0.75N 或分母 ≤ 0 → NaN',
   'mcap_proxy': '市值＝shares_hist 各股最早一筆股數（2025-09，backfill）× d 日原始收盤；股本變動用到未來股數（前視，僅用於「排除市值前 50 大」敏感度；無股數者視為非前 50）',
 },
 'universe_d': {
   'liquidity': 'd 日 60 日均成交值（上市後停牌日記 0，min_periods=60，面板日曆）≥ NT$1,000 萬；原始收盤 ≥ 10 元；flag ∈ {0,1,2}',
   'disposal': '排除 disposal_5y 中公告日 ≤ d 且處置迄日 ≥ d 的股票（d 日處置中或 d 盤後公告、d+1 起處置）',
   'full_delivery': '無全額交割名單資料 → 無法排除（限制；這類股票多半股價 < 10 或流動性差，多數已被流動性池擋掉）',
   'tradable': 'd+1 有開盤價與還原收盤',
   'signal_valid': 'FI_N 非 NaN（主版本 N=20）',
   'min_pool': '股池 < 50 檔則該月不換股',
 },
 'execution': {
   'signal': '每月最後一個面板交易日 d 收盤（d ≤ 2026-10-07 inst 最後日；首月 FI20 有效＝2021-11-30）',
   'trade': 'd+1 開盤（exlib M2）：還原開盤＝open×adj/close_raw；開盤漲停（還原開盤/前一日還原收盤 − 1 ≥ 9.4%）且未持有 → 跳過、依名次遞補；已持有者可續抱',
   'hold': '持有到下一個換股日開盤；換股時把漂移後持股調回等權，買賣金額分別乘費率',
   'cost': 'exlib.cost_rates：base 買＝0.1425%×0.6＋滑價 0.1%，賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%；壓力 買 0.5%＋賣 0.5%；另報毛報酬',
   'suspension': '持有期間停止交易／下市以最後成交價計（exlib 規則）',
   'last': '最後訊號 2026-09-30，持有到面板最後一天 2026-10-22',
 },
 'variants': {
   'PRIMARY': 'FI20_TOP20：FI20 最高 20 檔等權',
   'sensitivity_reported_not_for_verdict': {
     'FI60_TOP20': 'FI60 最高 20 檔（FI60 有效起 2022-01）',
     'FI20_D10': 'FI20 前 10%（十分位最高組）等權',
     'FI20_POS_FI60_POS_TOP20': '股池再加 FI20 > 0 且 FI60 > 0，依 FI20 取前 20',
     'FI20_EX50_TOP20': '股池先排除市值前 50 大（mcap_proxy），依 FI20 取前 20；對手同池等權改為排除後股池',
     'DECILES': 'FI20 十分位 D1..D10 等權（毛與 base 成本）；十分位價差＝D10 − D1（月報酬差，日曆月 t）',
   },
   'defaults_not_given_by_tutorial': ['流動性池定義採 BRIEF（60 日均成交值 ≥ 1,000 萬、股價 ≥ 10 元）', '有效日門檻 0.75N', '漲停判定 9.4%（exlib）'],
 },
 'benchmarks': {'0050': '0050 含息（exlib.E50）', 'EW': '同池等權：每個換股日 d+1 開盤等權買入當期股池（零成本），漂移持有到下一換股日'},
 'stats': {'monthly_t': '日頻淨值 → 日曆月報酬；策略 − 對手 的月差做 t（btlib.tstat）', 'CAGR': '日頻、以實際天數年化', 'maxDD': '日頻'},
 'windows': {'select': ['2021-10-01', '2023-12-31'], 'valid': ['2024-01-01', '2026-10-31'], 'valid_2024_2025': ['2024-01-01', '2025-12-31'],
             'full': ['2021-10-01', '2026-10-31']},
 'price_split_2005_2019': '不適用：外資日資料 2021-10 才有',
 'verdict_rule': {
   'segment_pass': 'PRIMARY 扣成本（base）後：對同池等權月 t ≥ 2 且年化超額 > 0，且年化 ≥ 0050 同期年化',
   'grade': '挑選期與驗證期都過＝可用；一段＝待觀察；零段＝不可用；資料不足＝無法測試',
   'overall': '整體判定只看 PRIMARY；敏感度不得升級判定',
 },
 'known_limits': ['只測上市（上櫃 2025-09 前無股價）', '2019–2023 面板只有存活者（偏樂觀）', '代理變數：買超/成交量 ≠ 持股比例變化',
                  '無全額交割名單', 'src 3（2024-02-15~2025-09-23）開盤/收盤為 5 分 K 近似', '挑選期只有約 2.1 年（2021-12～2023-12，25 個月）',
                  '市值前 50 大用 2025-09 股數回推（前視）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

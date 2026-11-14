#!/usr/bin/env python3
"""事前登記（PREREG）。在計算任何報酬之前執行一次：python3 -I -B prereg.py
寫入 results.json 的 PREREG 區塊與其 sha256（PREREG_SHA256）。之後 run.py 只附加結果、並檢查 sha256 沒變。
撰寫時點：已讀 raw/vcp_alert.xs、raw/SOURCE.txt、兩份範例報告，已完成最低價／成交股數資料閘門（不含任何報酬）；
尚未計算本腳本任何訊號或報酬。已知的先前資訊（揭露）：研究中心 w3_vcp_contraction_breakout 的結論是另一種 VCP
程式化近似（zigzag、還原收盤、沒有最低價）「不可用」，我讀過它的 FINDINGS 前 60 行。"""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent

PREREG = {
    'written_before_any_return': True,
    'question': 'VCP 突破警示 XS 腳本（預設參數、忠實重做）在台股上市股，扣成本後能否顯著打敗 0050 含息',
    'universe': '研究中心上市面板（scratchpad/panel，1,249 檔含下市股，2004-01-02～2026-10-22）；上櫃不做（沒有最低價與成交股數的完整來源）',
    'bars': 'K 棒＝該股面板 close_raw 有值的交易日；所有 Average／Highest／Lowest／[n] 都以該股自己的 K 棒計數（不是日曆日）',
    'signal_default_params': {'SegLen': 25, 'MaxLastDepth': 10, 'MinFirstDepth': 10, 'VolDryRatio': 0.7, 'VolBreak': 1.5, 'En_RS': 1,
                              'RSMinDiff': 0, 'MinVol': 500, 'LowMult': 1.3, 'HighPct': 25, 'MA200UpBars': 20},
    'price_basis': {'main': '原始價（XQ 日線預設不還原）：Close/High/Low 用面板 close_raw、high 與 data/low.parquet',
                    'side_by_side': '還原價版：每日 O/H/L/C × (adj ÷ close_raw)；成交量不調整。只並列，不是候選'},
    'min_bars': '主欄 CurrentBar ≥ 260（該股在面板內第 260 根起計算；第 260 根的 _pivot[1] 為初值 0，所以最早訊號在第 261 根）。穩健性：≥ 660 根（若 XQ 的 CurrentBar 從 setbackbar(400) 之後起算；需查證）',
    'execution': {
        'entry': '訊號日 d 收盤確認 → 面板日曆下一個交易日 e=d+1 以開盤價買進',
        'not_buyable': ['e 當日該股沒有開盤價（停牌、無成交）→ 放棄，不順延',
                        'e 開盤價 ≥ 漲停價（參考價×(1+漲跌幅)，向下取到跳動點；漲跌幅 2015-06-01 前 7%、之後 10%）→ 視為買不到，放棄（另報其中全日鎖死的比例）',
                        '組合回測：e 當日在處置期間且公告日 ≤ d（官方事件表 universe_announce.csv）→ 不買；事件研究則另列'],
        'ranking_when_slots_short': '同一 e 日候選多於空位：依訊號日量比（Volume ÷ 前 50 根均量）由大到小；同分依股票代號字串由小到大',
        'no_pyramiding': '已持有的股票再出訊號 → 忽略；出場後可再進',
        'sizing': '每筆預算 = min(可用現金, 前一日收盤淨值 ÷ K)；股數向下取整（2020-10-26 前為 1,000 股整數倍，之後 1 股）；買不到 1 個單位就放棄',
        'capacity': '單筆買進金額 ≤ e 當日該股成交值 × 5%（所有資金級距都套用）',
        'costs_normal': '手續費 max(20 元, 成交金額 × 0.1425% × 0.28) 每邊；賣方證交稅 0.3%；滑價每邊半個跳動點（買加、賣減）',
        'costs_stress': '一般成本之上，每邊再加 0.3%',
        'halts_delist': '持有中停牌：以最後收盤計值，復牌後第一根才能賣；到資料終點仍無成交（下市）：以最後收盤計值（下市價未另行查核，屬限制）',
        'limit_down': '賣出日若全日鎖跌停（最高＝最低且當日報酬 ≤ −(漲跌幅−0.6%)）→ 賣不掉，順延到下一根',
        'end_of_data': '資料終點仍持有的部位以最後收盤計值，不強制出場',
    },
    'exit_family': {
        'E1_hold20': '固定持有 20 個交易日：第 20 個持有日（含進場日）收盤賣出',
        'E2_hold60': '固定持有 60 個交易日：第 60 個持有日收盤賣出',
        'E3_stop8_ma50': '停損 8%（盤中觸價：還原最低 ≤ 進場還原價×0.92 → 以 min(開盤, 停損價) 成交；進場日也檢查）＋收盤跌破還原收盤的 50 根均線 → 次一根開盤賣；最長 250 個交易日（第 250 日收盤賣）',
        'E4_stop8_tp20': '停損 8%（同上）＋停利 20%（盤中觸價：還原最高 ≥ 進場還原價×1.20 → 以 max(開盤, 停利價) 成交）；同日兩者都觸及算停損；最長 250 日',
        'E5_close_below_pivot': '還原收盤 < 樞紐（訊號日樞紐價 × 訊號日還原因子）→ 次一根開盤賣；最長 250 日',
        'E6_trail10': '移動停損 10%：還原收盤 ≤ 0.90 × max(進場還原價, 進場以來最高還原收盤) → 次一根開盤賣；最長 250 日',
    },
    'idle_cash': {'CASH': '空位放現金、利率 0（主）', 'P50': '空位放 0050 含息（近似：前一日收盤現金賺當日 0050 報酬；買股時賣 0050 收 0.17%、賣股後買 0050 收 0.07%）'},
    'portfolio_main': {'capital': 1_000_000, 'K': 10, 'cost': 'normal', 'start': '第一個可能訊號日所在月份的次月第一個交易日', 'end': '2026-10-22'},
    'candidate_family': '6 個出場規則 × 2 種空位處理（CASH、P50）＝ 12 個可投資候選，全部計入資料窺探族群',
    'N_before': 2573, 'N_added': 12, 'N_after': 2585,
    'primary': 'E3_stop8_ma50 × CASH × K=10 × 100 萬 × 一般成本 × 原始價訊號（Minervini 自己的做法最接近這個；看報酬前指定）',
    'robustness_not_candidates': ['K=20', '不限檔數、每筆固定 10 萬（可能隱含槓桿，只當對照）', '資金 30 萬／300 萬／1,000 萬', '壓力成本',
                                  '還原價訊號', '進場延後 j=1,2,3 個交易日（d+1+j 開盤）', '≥ 660 根', '處置中也買', '開盤漲停也買（回測上緣，實現不了）',
                                  '零成本（回測上緣，實現不了）',
                                  '參數鄰域 81 組：SegLen 20/25/30 × MaxLastDepth 8/10/12 × VolDryRatio 0.6/0.7/0.8 × VolBreak 1.3/1.5/2.0（只看是否翻號，不挑參數）',
                                  '消融：逐一拿掉 趨勢樣板／收縮／量縮／突破放量／相對強度／500 張（診斷，不是候選）'],
    'event_study': {
        'horizons': [5, 10, 20, 60],
        'stock_return': '還原：收盤(e+h−1，停牌以最後收盤) ÷ 開盤(e) − 1，e=d+1；毛報酬為主，另報扣一般成本（比例手續費＋稅＋半跳）',
        'bench_0050': '0050 含息：收盤(d+h) ÷ 收盤(d) − 1（沒有 0050 開盤價序列，所以多含 d→e 一個隔夜；對策略略不利）',
        'bench_pool': '同日同池：d 日成交量 ≥ 500 張、K 棒數 ≥ 260、e 有開盤價的全部面板股票，同樣「開盤(e)→收盤(e+h−1)」的等權平均',
        'inference': 't 以訊號月份分群（cluster-robust，含 G/(G−1) 校正）；另報月份等權平均的 t；有效樣本數＝訊號月份數與 N ÷ 設計效果；h=60 另做 3 個月區塊 bootstrap',
        'exclusions_reported': ['e 無開盤', 'e 開盤漲停（其中全日鎖死）', 'e 處置中（另列）'],
    },
    'periods': {'headline_first': '2019 起（2019-01-02 收盤 → 2026-10-22 收盤，沿用 V7X 口徑）', 'full': '起點到 2026-10-22',
                'three': ['2006-01-01～2012-12-31', '2013-01-01～2018-12-31', '2019-01-01～2026-10-22'],
                'split': {'selection': '2006～2014', 'validation': '2015～2020', 'holdout': '2021～2026-10'}},
    'stats': {'annualized': '日曆年化：(終值÷起值)^(365.25÷日曆天數) − 1', 'excess': '策略年化 − 0050 含息同期年化（pp）',
              't': '月報酬（月底收盤對月底收盤）超額的一般 t = 平均 ÷ (標準差 ÷ √n)', 'maxDD': '日頻收盤淨值',
              'ex_best3': '拿掉日曆年超額最大的 3 年後，剩餘月份的超額 t'},
    'gates': {
        'G1': '主規格在一般成本與壓力成本下全期超額都 > 0；進場延後 j=0..3 的全期超額中位 > 0',
        'G2': '全期月超額 t ≥ 3.0 且 2019 起 t ≥ 2.0',
        'G3': '12 個候選的 SPA（月超額、定態 bootstrap）p < 0.05，且以 N=2,585 計的 DSR ≥ 0.95',
        'G4': '三段超額都 > 0；拿掉最好 3 年 t ≥ 2.0；81 組鄰域全期超額與預設同號（不翻號）',
        'G5': '無前視、無存活者偏誤、資料為當時可得',
        'G6': '另一位研究員只看規格重做，關鍵數字一致',
        'G7': '100 萬、300 萬、1,000 萬全期超額都 > 0',
        'G8': '凍結後前瞻 12 個月超額 > 0',
        'verdict': 'G1～G8 全過才是可投資。只報最好的出場規則不算；如果選，必須只用挑選期 2006～2014 選、再看驗證期與保留期',
    },
}

if __name__ == '__main__':
    out = HERE / 'results.json'
    assert not out.exists(), 'results.json 已存在，PREREG 不得重寫'
    s = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()
    out.write_text(json.dumps({'PREREG': PREREG, 'PREREG_SHA256': h, 'PREREG_SHA256_method': 'sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode("utf-8"))'},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print(h)

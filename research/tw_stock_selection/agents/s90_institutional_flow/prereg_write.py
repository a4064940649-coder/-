#!/usr/bin/env python3
"""S90 事前登記：寫 PREREG.json＋PREREG.sha256＋code.sha256（s90lib.py、train.py、validate.py）。已存在就拒絕重寫。用法：python3 -I -B prereg_write.py"""
import sys, json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
assert not (HERE / 'PREREG.json').exists(), 'PREREG.json 已存在'
sys.path.insert(0, str(HERE))
import s90lib as L
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
st0 = json.loads((HERE / 'step0.json').read_text()); dc = json.loads((HERE / 'data_check.json').read_text())
assert st0['A0']['pass']
PR = {
 'id': 'S90', 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'question': '三大法人逐股買賣超（2015 起完整資料）能不能幫營收驚奇（SURGE）分辨好壞交易？四種用法：過濾、確認、排序加權、放寬入口。不改正式策略，只回答要不要加。',
 'engine': '正式管線（official_nav/run_official.py 的 E0｜W10_70 寫法）：EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK、REDUCE_FIX、0050 官方含息五個開關全開；s10lib.simulate_s10 原樣呼叫；只改訊號列的買進名單（sig／rank），出場 X 不動；嚴成本、空位 0050。',
 'acceptance（登記前已跑，step0.json）': st0['A0'],
 'data': {'source': 'raw/tsd_extra/inst_twse（上市三大法人逐股逐日，單位股，2015-01-05 起）；成交股數與發行股數取 raw/tsd_extra/daily。某股某日沒有法人列＝0。',
          'pit': '訊號列 k（訊號日收盤後）只用列 ≤ k 的資料；T86 每日收盤後公布；次一交易日開盤成交。換股日位移 +1～+3 的路徑用位移後那一列（含）以前的資料。',
          'check': {'build': dc['build'], 'signal_days': dc['signal_days'], 'old_source_compare': dc['old_source_compare'], 'unit_check': dc['unit_check']}},
 'features': {'fvN／tvN／avN': '近 N 個交易日（含訊號日）外資／投信／三大合計買賣超股數 ÷ 同 N 日成交股數合計',
              'fsN／tsN／asN': '同上 ÷ 訊號日發行股數（流通量代理）；本題只用它的正負號',
              'TFB 投信初買': '投信近 5 日合計買超 > 0 且之前 20 日（k−24～k−5）合計 ≤ 0',
              'streak_a': '到訊號日為止三大合計連續買超（> 0）天數',
              'p_x': 'x 在同日股池（上市普通股、排除創新板，同引擎 U）內的百分位'},
 'cells（30 格，事前固定，不得增減）': {
   '(a) 過濾 6 條 × Z／F＝12 格': {'FA_F20L': '外資近 20 日占量落在同日股池最低 20% → 不買', 'FA_F5L': '外資近 5 日占量最低 20% → 不買', 'FA_A20L': '三大合計近 20 日占量最低 20% → 不買',
                              'FA_A5L': '三大合計近 5 日占量最低 20% → 不買', 'FA_T20N': '投信近 20 日合計賣超 → 不買', 'FA_FT20N': '外資與投信近 20 日都賣超 → 不買'},
   '(b) 確認 5 條 × Z／F＝10 格': {'CB_F20P': '只買外資近 20 日合計買超的', 'CB_T20P': '只買投信近 20 日合計買超的', 'CB_A20P': '只買三大合計近 20 日買超的', 'CB_A5P': '只買三大合計近 5 日買超的', 'CB_STRK3': '只買三大合計已連買 ≥ 3 日的'},
   'Z／F': 'Z＝被擋的名額不遞補（錢照引擎放 0050）；F＝由季線上 SURGE 第 11～20 名中近 20 日漲幅 > 10% 且通過同一條件者依 SURGE 遞補。',
   '(c) 排序加權 4 格': '季線上 SURGE 前 20 名內，SURGE 名次（0～19）＋法人指標名次（由大到小 0～19；缺值最後）相加由小到大重排（同分看 SURGE），取前 10，再套近 20 日漲幅 > 10%。買進優先序照新名次。RK_A20＝三大合計 20 日占量、RK_F20＝外資 20 日、RK_T20＝投信 20 日、RK_A5＝三大合計 5 日。',
   '(d) 放寬入口 4 格': '現行名單不動，另外加：季線上 SURGE 前 10 名中近 20 日漲幅 ≤ 10%（動能沒過）但符合條件者。WD_TFB＝投信初買；WD_T20P＝投信近 20 日買超；WD_A20H＝三大合計 20 日占量在股池最高 20%；WD_TFB20＝WD_TFB 再加第 11～20 名中漲幅 > 10% 且投信初買者。加進來的照 SURGE 分數排優先序；出場規則同現行。',
   'ids': list(L.CELLS)},
 'paths': '20 條＝資金 60／80／100／150／200 萬 × 換股日位移 0～3；每格與現行同一路徑配對。',
 'train': {'window': '訊號日 2016-02～2021-12；全期連續模擬，取 2016-01-29 → 2021-12-30 的年化（同 S50／S70／S88／S89）。', 'score': '20 條路徑訓練窗年化中位數。',
           'selection（同 S25／S88／S89 標準）': '中位數最高的格為 top（同分看 ≥ +0.25 個百分點的路徑數，再看格子順序）。top 要同時滿足：(a) 中位數 − 現行中位數 ≥ +1 個百分點；(b) 訓練窗最大回撤中位數不比現行深超過 2 個百分點；(c) 2016-01～2018 與 2019～2021 兩半的配對差中位數都 > 0。滿足 → selected＝top；否則 selected＝「不改」。',
           'T0': '配對差中位數 ≥ +1 個百分點且 ≥ 14／20 條路徑 ≥ +0.25 個百分點。', 'freeze': '寫 FROZEN.json＋sha256。selected＝「不改」時仍以 top 跑驗證期，只作描述，總判定為不通過。'},
 'validate（只跑一次）': {'window': '重新起跑：只保留營收月份 ≥ 2021-12 的進場訊號（訊號日 2022-01 起），起始資金除以現行標準路徑在 2021-12-30 的淨值倍數；驗證窗 2021-12-30 → 2026-10-22。',
                    'V1': '凍結格 − 現行，驗證窗年化差 > 0 的路徑 ≥ 15／20。', 'V2': '配對差中位數 ≥ +2 個百分點。',
                    'V3': '現行與凍結格各自（標準路徑）最賺 5 筆的持有期聯集禁買後，20 條路徑配對差中位數仍 > 0。', 'V4': '2022、2024-07～2025-04 兩段空頭的段內最大跌幅配對差中位數都不深於 −2 個百分點。',
                    'overall': 'selected＝top 且 T0、V1～V4 全過才算通過（建議加）；否則不通過（維持現行）。不論結果都不改正式策略。'},
 'descriptive（不影響判定）': [
   '驗證段標準路徑：凍結格與現行的名單差異逐筆清單——現行有、凍結格沒買的（代號、進場月、現行最後報酬、是否大贏家／大輸家）與凍結格多買的（代號、進場月、最後報酬）。大贏家＝現行驗證段已平倉報酬前 5%，大輸家＝後 5%。',
   '隨機對照（S88 做法，seed 90、20 次）：過濾／確認→每月從名單隨機剔除與凍結格相同檔數（F 模式照樣遞補、不套法人條件）；重排→第二個名次改成隨機；放寬→從同一批候選隨機加同檔數。比較凍結格驗證窗年化落在隨機分佈哪裡。',
   '全部 389 筆交易（381 已平倉）逐筆加上訊號日法人特徵的分組描述：近 5／10／20 日外資、投信、三大合計（占量、正負）、投信初買、外資投信同向、連買天數 × 筆數、勝率、平均與中位數報酬、大贏家（前 5%）／大輸家（後 5%）／損益最大 10 筆落在哪。驗證期部分在 FROZEN 之後才算（validate.py 內）。',
   '凍結格全期連續模擬 2019 起年化（20 條）、成本加倍。'],
 'dedup': {'S22（2026-11-16）': '同一題，因法人資料只到 2021-10-08（60 個訊號月）判資料不足，沒寫 PREREG、沒算報酬。本題用 2015 起資料補做。',
           'w1_trust_first_buy_small': '投信首買（60 日沒買→連 3 日買）獨立策略，2022～2026：不可用（驗證期每筆 −4.77%、t −2.70）。不是加在 SURGE 上。',
           'w12_foreign_intensity_monthly': '外資 20 日買超占量月選股，2021-10 起：不可用（毛 t 0.14／0.29，十分位價差方向翻轉）。',
           'w12_foreign_streak_pullback／w1_tuyang_streak／w12_foreign_flow_bollinger_timing': '外資連買拉回、土洋同買、外資總買超擇時：都不可用。',
           'bt_institution_horizons（output/bt_institution_horizons.json）': '法人買賣超的預測力集中在當天與隔夜；第 6～20 日約 +0.9 bp（t 0.2）。',
           '沒做過': '把法人買賣超當成 SURGE 名單的過濾、確認、重排或放寬條件；2015～2021 的任何法人檢定。'},
 'step0_seen_before_prereg': {'note': '登記前只跑了現行＋驗收＋資料驗收，並只看訓練期（訊號日 2016-02～2021-12）現行已平倉 186 筆的法人分組描述；任何加了法人條件的模擬都還沒跑；驗證期逐筆的法人特徵沒看。例外（如實揭露）：格子在 s90lib.py 寫定後，為了檢查程式流程跑過一次 train.py --dry（只有位移 0、100 萬一條路徑、5 格），看到這 5 格的訓練窗年化差：FA_F20L|Z −0.01、FA_F20L|F −1.36、FA_F5L|Z −0.87、RK_A20 −0.41、WD_TFB20 +1.50；看到之後沒有增減或修改任何格子與門檻。validate.py --dry 只跑現行對現行（配對差全 0）。',
                              'train_desc': st0['train_desc']},
 'predictions（事前寫下，可被否證）': [
   '訓練期描述裡買超組沒有比較好（外資 20 日買超組中位數 −1.9%、賣超組 +3.5%），大贏家 9 筆散在各組 → 確認類（b）在訓練期多數為負，尤其 CB_F20P、CB_A20P。',
   '過濾類（a）的百分位版本只擋掉很少筆（訓練期 4～7 筆），配對差接近 0（±1 個百分點內）。',
   '重排類（c）會換掉約一半名單，訓練期多數為負。',
   '放寬類（d）加進來的是動能沒過的股票，S76 顯示動能門檻在無摩擦下是扣分的，所以 WD 類有機會是正的；但估計 < +1 個百分點。',
   '最多 1～2 格達到 selected 條件；若有，驗證期 V2（≥ +2 個百分點）不會過。'],
 'data_snooping': {'added': 30, 'note': '30 格全部計入；N 以研究中心彙總為準。'},
 'resources': '單行程、峰值記憶體 ≤ 3GB、python3 -I -B、不連外網；暫存 scratchpad/s90/（用完刪）。',
}
(HERE / 'PREREG.json').write_text(json.dumps(PR, ensure_ascii=False, indent=1, default=str))
(HERE / 'PREREG.sha256').write_text(f"{sha(HERE / 'PREREG.json')}  PREREG.json\n")
(HERE / 'code.sha256').write_text(''.join(f"{sha(HERE / f)}  {f}\n" for f in ('s90lib.py', 'train.py', 'validate.py')))
print((HERE / 'PREREG.sha256').read_text(), (HERE / 'code.sha256').read_text())

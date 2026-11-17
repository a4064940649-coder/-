#!/usr/bin/env python3
"""S43 事前登記：在任何新規則報酬計算之前寫 PREREG.json／PREREG.sha256／code_prereg.sha256。已存在就拒絕。python3 -I -B prereg_write.py"""
import json, hashlib, datetime, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent
assert not (H / 'PREREG.json').exists(), 'PREREG.json 已存在'
import s43lib
dc = json.loads((H / 'data_check.json').read_text())
sh = lambda f: hashlib.sha256((H / f).read_bytes()).hexdigest()
P = {
    'topic': 'S43 持股的下個月營收一首次公布就很差 → 公布隔天開盤提早賣出（不等月中換股日），能不能比現行（只在訊號日照 E0 出場）更好？S36（公告即進場）的出場版。',
    'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'status_at_write': '只跑過 prep.py（data_check.json）：現行重現、持股新營收提早可見比例與天數、首次值 SURGE 分布、各規則×θ 在現行持股上的標記筆數（只數筆數）；沒有算任何新規則的報酬。θ 格點在看過 prep 筆數後、看任何報酬前定為 {0.5, 1.0, 1.5, 2.0}（prep 原試 0.0 時驗證期只 1 筆）。',
    'prior_work': {
        'S36': '公告即進場：EA_MOM 2.8 T0 不過（訓練 +1.68、10/20），驗證好處集中在 1 筆 → 不通過；同一份 Tasi 首次可見時間',
        'S39': '首次值 vs 更正值：28 個月 F1 名單不變、年化約 −0.8 個百分點；實盤公布當下只看得到首次值 → 本題用首次值 SURGE（FPM）判斷',
        'S10': '出場家族（R10／R15／MA20／DMA60／移動停損等）都只在訊號日或每日價格判斷，沒有用逐檔營收公告時點',
        '結論': '沒有做過「持股新營收首次公布後即提早賣」→ 不算重做',
    },
    'data': {
        'seen': '同 S36：o_rev_event_clock_ts/data/firstseen_sources.csv.gz 的 Tasi 首次可見時間；r＝T 的台北日期（含）以前最後一個交易日；r 收盤判斷、r＋1 開盤賣；T 是上界（保守）',
        'first_print': 's39_first_print_revenue/data/first_print.csv.gz；SURGE_FP＝s39lib.surge_variant(mode="FPM")（當月 D 用首次值，μ、σ 用目前值）；首次值缺 → 用目前值',
        'months': '營收月份 2024-06～2026-09（28 個月）',
        'holdings_preview': {k: dc['holdings'][k] for k in ('n_position_months', 'held_at_r_share', 'lead_days_dist', 'surge_fp_quantiles', 'prev20_quantiles', 'e0_exit_at_k_share')},
        'trigger_preview_on_CUR_holdings': dc['trigger_preview'],
    },
    'code_sha256_at_write': {f: sh(f) for f in ('s43lib.py', 'prep.py')},
    'baseline（不改定義）': 'S10 的 E0／W10_70，cslib.EXRIGHT_FIX＝True：嚴成本、只上市、排除創新板、官方訊號列 k 次日開盤、F1 進場、E0 出場（訊號日判斷「跌出 L_MA 前 20 名或收盤 ≤ 60 日線」）；衛星 70%、每檔＝衛星 ÷ 10、名額 10；0050 停牌日用前一日收盤。引擎 s20lib→s17lib→s10lib 直接 import，不改。',
    'wrapper': '不改引擎：simulate_s10 在非訊號列也看 X（DMA60 用的同一個掛鉤）。提早出場＝把 E0 的 X 複製一份，在可見列 e＝r＋shift（只在 r＜官方 av[m]）把符合規則的格子設 True；持有中的股票被標記、e＋1 開盤賣（跌停／無開盤順延，同引擎），錢掃進 0050；名額空著到官方 k＋1 照現行 F1 補（F1 若又選到同一檔會再買回，照引擎）。進場與月中出場完全不變。規則永不觸發（沒開月份或 θ＝−∞）時與現行逐位相同。',
    'rule_form': {
        'candidates（3 條規則、6 個設定，全部計入 N）': s43lib.RULES,
        'thr_grid': list(s43lib.THR_GRID),
        'common': '對象＝可見時持有中的股票；SURGE_FP 為 NaN 不觸發；只在 r＜官方 av[m] 時判斷（官方日當天或之後才可見的照現行）。',
    },
    'windows': {
        'train': '營收月份 2024-06～2025-07（14 個月）；只在這些月份開提早出場；訓練窗淨值 2024-06-28 → 2025-08-29',
        'valid': '營收月份 2025-08～2026-09（14 個月）；只在這些月份開提早出場；驗證窗 2025-08-29 → 2026-10-22；窗起點前與現行逐位相同（assert）',
        'split': '同 S36 事前對半切',
    },
    'paths': '20 條＝執行延遲位移 0～3 天（官方日與提早出場同時位移）× 起始資金 60／80／100／150／200 萬；全期連續模擬，從窗內切',
    'selection': '訓練分數＝設定 − 現行（CUR）的訓練窗年化差，20 條路徑中位數（個百分點）。最高者凍結（同分依改善路徑數、再依標記筆數少者）。T0（沿用 S25）：分數 ≥ +1 個百分點，且差 ≥ +0.25 個百分點的路徑 ≥ 14／20。T0 不過仍凍結最高者，驗證期只描述、判定必為不通過。',
    'verdict（驗證期只跑一次，沿用 S25）': {
        'V1': '驗證窗年化差為正的路徑 ≥ 15／20',
        'V2': '配對差中位數 ≥ +2 個百分點',
        'V3': '禁買 CUR 與凍結設定各自（位移 0、100 萬主跑、驗證窗內進場）最賺 5 筆的聯集後，配對差中位數 > 0',
        'V4': '驗證窗內最大回撤的配對差中位數不深於 −2 個百分點',
        'overall': 'T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換門檻重跑',
    },
    'acceptance': {
        'A0': 'CUR 全期主跑重現 48.99％／34.54％／−32.11％（±0.025 個百分點）、381 筆已出場（prep 已過）',
        'A2': '包裝在「沒有開任何月份」與「θ＝−∞」時，交易與淨值與 CUR 逐位相同',
        'A3': '只開驗證月份時，驗證窗起點以前淨值與 CUR 逐位相同',
    },
    'describe（不參與判定）': '提早賣出筆數（引擎實際成交，原因 ma_daily）；每筆被提早賣掉的股票「賣出日開盤 → CUR 原本出場日開盤」的報酬與對 0050 超額（＝提早賣的得失，正＝賣掉後來還漲＝賣錯）；最大的幾筆賣錯；全期開 28 個月連續模擬；成本加倍。',
    'N': '資料窺探族群 N 加 6（3 條規則共 6 個設定）。',
    'predictions（事前預測）': {
        'P1': 'prep：門檻型規則（XA_THR、XA_NEG_MA20）在現行持股上標記者官方日 ≥ 95% 本來就會被 E0 賣掉 → 效果只是提早 1～7 個交易日（中位數約 3 天）賣出，量級＝這幾天的個股對 0050 超額 × 少數筆數，預期年化差 < 1 個百分點',
        'P2': '營收很差的消息當晚多半已部分反映在隔天開盤（跳空），提早賣的平均超額接近 0；XA_PREV20 會賣掉一些官方日其實留任的股票（訓練 3、驗證 7 筆），可能誤殺',
        'P3': '驗證期門檻型規則只有 1～10 筆會觸發，檢定力極低；整體通過機率約 5％',
    },
}
b = json.dumps(P, ensure_ascii=False, indent=1).encode()
(H / 'PREREG.json').write_bytes(b)
(H / 'PREREG.sha256').write_text(hashlib.sha256(b).hexdigest() + '  PREREG.json\n')
(H / 'code_prereg.sha256').write_text(''.join(f'{sh(f)}  {f}\n' for f in ('s43lib.py', 'prep.py', 'prereg_write.py')))
print(hashlib.sha256(b).hexdigest())

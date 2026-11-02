#!/usr/bin/env python3
"""分支 A3：凍結事件時鐘前瞻規格與 12／24 個月判定規則（寫入 results.json 的 PREREG、PREREG_sha256、PREREG_FORWARD）。
只能執行一次；之後 forward/event_clock_log.py 正式模式會斷言本登記存在且自身 sha256 相符。
執行：python3 -I -B prereg_forward.py"""
import sys, json, math, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_common as T

r = T.jload(T.RES_F)
assert 'PREREG' not in r, 'PREREG 已存在：只能登記一次'
G = T.jload(T.HERE / 'gate.json'); X = T.jload(T.HERE / 'explore.json'); I = T.jload(T.HERE / 'iterations.json')
assert G['branch'] == 'A' and X and I
files = ['prereg_gate.py', 'gate_prereg.json', 'ts_common.py', 'gate.py', 'gate.json', 'a1_describe.py', 'explore.json', 'iterations.json',
         'forward/event_clock_log.py', 'prereg_forward.py']
H = {f: T.sha(T.HERE / f) for f in files}
H['../../main_picks_today.py'] = T.sha(T.AG.parent / 'main_picks_today.py')
H['../o_v7_listing_exec_engine/data/model_spec7.json'] = T.sha(T.AG / 'o_v7_listing_exec_engine' / 'data' / 'model_spec7.json')
# 檢定力（事前寫死）
mde_full, n_full = 0.51, 240
sd_d = mde_full * math.sqrt(n_full) / 2.8
se24 = sd_d / math.sqrt(24)
from statistics import NormalDist
N = NormalDist()
power = {f'{mu:.2f}pp': 1 - N.cdf(1.5 - mu / se24) for mu in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)}
PRE = dict(
    topic='o_rev_event_clock_ts', branch='A（時間戳覆蓋 19.8% < 60%）', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    frozen_version=I['selected'], version_params=I['versions'][I['selected']],
    rule_text=open(T.HERE / 'forward' / 'event_clock_log.py', encoding='utf-8').read().split('"""')[1],
    start='2026-11 錨定月（2026-11-03 開盤起第一筆決策）；初始持股＝2026-10-12 官方名單更正版（live_tracking/main_forward/corrections/official_202610_corrected.csv）',
    statistic='事件時鐘（EC）減官方日主策略（OFF）的月配對淨報酬差（官方訊號日收盤到下一個官方訊號日收盤）；一般成本與壓力成本兩欄；j 固定為公告隔日（成交日＝首見後第一個開盤）；100 萬級距為主欄。',
    judge_12m='滿 12 個錨定月（2027-10）：只報告配對差平均、t、壓力成本差、最大回撤差、漏跑天數、事件延遲分布、早買命中官方名單比例、P5 的價格漂移，不做任何判定。',
    judge_24m=('滿 24 個錨定月（2028-10）：以下全部成立才可提請把主策略執行改為事件時鐘：(1) 一般成本月配對差平均 > 0 且 t ≥ 1.5（月份為單位，不重疊）；'
               '(2) 壓力成本月配對差平均 > 0；(3) EC 淨值最大回撤不比 OFF 多 5pp 以上；(4) 漏跑成交日 ≤ 10%。否則維持官方日。'
               '主策略本身的投資判定（不可投資）不因本規則改變。'),
    P5=('2026-11 起 12 個月：事件股（is_event＝1 的買進）實際開盤成交價，比官方日名單同檔成交價平均便宜 ≥ 1%（＝更早吃到漂移）；'
        '計算＝同一錨定月內 EC 早買的股票若也在官方名單，比較 EC 成交開盤價與官方成交開盤價（含息還原）；≤ 0 表示時效優勢已在公告當晚被定價完。'),
    power=dict(mde_full_pp=mde_full, n_full=n_full, implied_sd_diff_pp=sd_d, se_24m_pp=se24, mde_24m_pp=2.8 * se24,
               prob_t_ge_1p5_at_true_mean=power,
               note='以歷史 J0 對 J+1 配對差的 MDE（全期約 0.51pp／月、約 240 個月）推回月配對差標準差約 2.8pp；24 個月 MDE 約 1.6pp／月，'
                    '合理效果 ≤ 0.3pp／月時 t ≥ 1.5 的機率只有約 16%（真實效果 0 時約 7% 誤升級）。前瞻紀錄主要是在累積資料，不太可能給出肯定結論。'),
    expected_cost=dict(extra_trades_per_month=X['P3']['mean_extra_trades_per_month'], breakeven_drift=X['P3']['breakeven_drift_per_extra_trade'],
                       extra_cost_pp_per_month=X['P3']['extra_cost_pp_per_month']),
    hashes=H)
sh = T.canon_sha(PRE)
r['PREREG'] = T.conv(PRE); r['PREREG_sha256'] = sh
r['PREREG_FORWARD'] = dict(file_sha256={'forward/event_clock_log.py': H['forward/event_clock_log.py']}, prereg_sha256=sh, utc=PRE['utc'])
T.jdump(r, T.RES_F)
print(sh, PRE['utc']); print(json.dumps(T.conv(PRE['power']), ensure_ascii=False, indent=1))

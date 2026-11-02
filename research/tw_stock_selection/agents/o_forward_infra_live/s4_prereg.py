#!/usr/bin/env python3
"""凍結：把 PREREG_FORWARD（全部腳本與引擎 sha256、成本參數、各線定義、G8 規則、檢定力警告、修 bug 規則）寫進 results.json。
已存在就不覆寫（只能另記 bugfix_log）。"""
import json, hashlib, datetime, math
from pathlib import Path
import numpy as np
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
AG = HERE.parent; TW = AG.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rj = HERE / 'results.json'
res = json.loads(rj.read_text(encoding='utf-8')) if rj.exists() else {}
if 'PREREG_FORWARD' in res:
    raise SystemExit('PREREG_FORWARD 已存在，不覆寫（修 bug 請寫 bugfix_log）')
files = {
    'make_forward.py': HERE / 'make_forward.py', 'fwd_data.py': HERE / 'fwd_data.py', 'xlive.py': HERE / 'xlive.py', 'disp_sat.py': HERE / 'disp_sat.py',
    'daily_step0.sh': HERE / 'daily_step0.sh', 'fallback_manual.sh': HERE / 'fallback_manual.sh', 'calib_real.py': HERE / 'calib_real.py',
    'replay.py': HERE / 'replay.py', 'accept.py': HERE / 'accept.py', 'a3_v7x_compare.py': HERE / 'a3_v7x_compare.py',
    'main_picks_today.py': TW / 'main_picks_today.py', 'disposal_today.py': TW / 'disposal_today.py',
    'x_exec.py': AG / 'o_exec_engine_rewrite_v7x/x_exec.py', 'x_data.py（A3 對照，只讀）': AG / 'o_exec_engine_rewrite_v7x/x_data.py',
    'frozen_spec_v7x.txt': AG / 'o_exec_engine_rewrite_v7x/frozen_spec.txt', 'model_spec7.json': AG / 'o_v7_listing_exec_engine/data/model_spec7.json',
    's8_forward_calib.py': AG / 'o_exec_engine_rewrite_v7x/s8_forward_calib.py',
    'o_forward_sequential_v2/make_forward.py（舊版，只讀）': AG / 'o_forward_sequential_v2/make_forward.py',
    'frozen_spec.txt': HERE / 'frozen_spec.txt',
}
fh = {k: sha(v) for k, v in files.items()}
sigma = 6.141373712658076           # o_forward_sequential_v2 PREREG σ_design（JMED 月超額標準差，pp）
nd = NormalDist()
power = {f'alpha_{a:+.1f}pp': round(nd.cdf(a * math.sqrt(12) / sigma), 3) for a in (-1.0, -0.5, -0.4, -0.3, 0.0, 0.3, 0.5, 1.0)}
now = datetime.datetime.now(datetime.timezone.utc)
P = dict(
    frozen_utc=now.isoformat(), frozen_taipei=(now + datetime.timedelta(hours=8)).strftime('%Y-%m-%dT%H:%M:%S+08:00'),
    deadline='2026-11-09 23:59（台北）', file_sha256=fh,
    data=dict(screener='raw/screener（github stevenhsiehsheng-ctrl/tw-stock-screener 稀疏 clone，每日 fetch＋reset 到 origin/main）',
              finlab='raw/finlab_repo（github jonatw/finlab-txd 稀疏 clone）→ raw/finlab/etf_0050.csv、taiex_twii.csv',
              panel='scratchpad/panel 的 [2025-01-01, history 起日) 子集（work/panel_pre；只在 history 不足 252 日時拼接）'),
    truncation='fwd_data.py 文件字串（A1）：價格 date ≤ asof；營收 ann_eff ≤ asof（ann_date，缺值＝法定期限日）；處置公告日 ≤ asof；0050、加權指數 ≤ asof；每個資料框載入後 assert',
    costs=dict(main='V7X K5T（x_exec.py 逐字）：' + json.dumps(dict(tier=1_000_000, mode='GROW', model='K5T', f1=True, f2=True, f3=0.5, f4=True, rank_right=True,
                                                                   stress_odd_extra=True, cap=0.20, base=0.001, kappa=0.5, cash_buffer=0.02,
                                                                   fee='0.1425%×0.28（最低 20 元）', tax_sell=0.003, stress='每邊＋0.4%、手續費 6 折、零股買成本×2'), ensure_ascii=False),
               cap10='同上，但參與上限 0.10（SOP 建議，報告用）',
               disposal='處置可執行層 v1_cap20（disp_sat.py 文件字串）：每邊 手續費＋max(一跳/收盤, 0.1%)＋0.5σ√(q/Qref)；賣出 0.3%；Qref＝V×match_min/270；上限 20%×Qref；閒置 0050（買賣 0050 手續費＋賣出 0.1%）',
               mix_rebalance='窗初再平衡，漂移流量×0.5%'),
    lines={
        'MAIN_OFFICIAL': '官方日（第一個 ≥ 11 日的交易日）收盤訊號＝main_picks_today.py --ex-fin 前 10 名排序；隔日開盤 M2 遞補；V7X K5T GROW 100 萬',
        'J0..J12': 'D_m（10 日起第一個交易日）後第 j 個交易日的同一分數；m−1 月營收只用已公告（J0 缺 ann_date＝未公告；J≥1 缺＝已公告）；同引擎',
        'JMED': '每個錨定月 13 個 J 窗（訊號日收盤→下月同 j 訊號日收盤）超額（pp）的中位數；13 窗全部完成才定稿',
        'E50': '0050 含息（finlab adj），官方日窗',
        'DISP_TWSE': '處置衛星（上市），disposal_today.py 規則＋v1_cap20 可執行層，10 槽、100 萬、閒置 0050；官方日窗計報酬',
        'DISP_TWSE_FIRST': '同上，只做第一次處置（排除 30 個交易日內再處置）',
        'DISP_TPEX': '上櫃另列（同規則）',
        'MIX_C70': '0050 70%＋MAIN_OFFICIAL 30%（官方日窗初再平衡）',
        'MIX_D80': '0050 80%＋DISP_TWSE 20%（官方日窗初再平衡）'},
    snapshot_rule='名單快照在訊號日（含）以前的資料下寫入，寫入後不覆寫；frozen_ts 必須早於成交日 09:00（台北）；最近 20 個交易日的訊號日每天重算前 20 名，不同寫 list_diffs.csv',
    G8=dict(first_anchor='2026-11', n_anchors=12, last_anchor='2027-10',
            lines=['JMED', 'DISP_TWSE', 'MIX_C70', 'MIX_D80'],
            rule='每條線各自判定：2026-11～2027-10 共 12 個錨定月的「完成」窗超額（excess_pp，一般成本）加總 > 0 → 該線過 G8；JMED 用每月 13 個 J 窗的中位數（必須 13 窗都完成）。',
            decision_dates='官方日窗最後一窗結束 2027-11-11 前後；JMED 在 2027-10 錨定月的 J12 窗結束（約 2027-11-26）後定稿。',
            descriptive=['MAIN_OFFICIAL', 'DISP_TWSE_FIRST', 'DISP_TPEX', '壓力成本欄', '參與上限 10% 欄', '幾何累積超額'],
            timestamp_flaw='任何一個月的快照 ts_ok＝false（凍結晚於成交日開盤）：該月照算但標註；若有 fallback_manual 的較早存檔則改用存檔名單重算；12 個月中有 ≥2 個月時戳不合格的線，G8 寫「時戳瑕疵，不判定」。'),
    power_warning=dict(sigma_design_pp=sigma, p_pass_given_alpha=power,
                       text='o8 檢定力警告：12 個月無法在統計上區分每月 −0.3～−0.5pp 的落後。以 σ＝6.14pp／月，真實 α＝−0.3／−0.5pp 時 12 個月加總仍 > 0 的機率約 '
                            f"{power['alpha_-0.3pp']:.0%}／{power['alpha_-0.5pp']:.0%}；α＝0 時 50%。G8 通過只是必要條件，不是 α>0 的證據。"),
    bugfix_rule='凍結後只准修 bug：每次修改記錄於 results.json bugfix_log（舊新 sha256、diff 摘要、重算前後 forward_returns 差異），不得改規則、線定義、成本參數或 G8。',
    topic_verdict_rule='基礎建設「可用」＝A1～A6 全過，且 2026-11-10（J0）與 11-11（官方日＝J1）的真實運行快照 frozen_ts 都早於各自成交日 09:00（台北）；任一項沒過＝不可用，首月改用 fallback_manual.sh。',
)
res['PREREG_FORWARD'] = P
rj.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(dict(frozen=P['frozen_taipei'], power=power), ensure_ascii=False))

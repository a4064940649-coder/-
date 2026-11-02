#!/usr/bin/env python3
"""步驟五：前瞻校準協議（2026-11 起；事前寫死，PREREG 涵蓋本檔）。
讀 live_tracking/main_forward/trades.csv（第 4 題產生；欄位見 FIELDS），對 K5T 的 κ 做常態共軛貝氏更新：
  觀測：每筆非零股整股成交 i，y_i＝side×(成交價/開盤價−1)（正值＝不利）；x_i＝σ60_{t−1}·√(Q_i/OA_pred_i)，Q＝張數×1000×前收，
        OA_pred＝share_dec×V60_{t−1}（V7X 第 4 項修正的事前開盤量）。模型 y_i − 0.001 ＝ κ·x_i ＋ ε_i，ε ~ N(0, s_ε²)。
  先驗：κ ~ N(0.5, 0.17²)。0.5＝model_spec7.kappa；sd 0.17＝√(0.094²＋0.14²)：
        0.094＝o6 bootstrap 隱含衝擊 p5～p95 的相對半寬 5.6%×0.5，再除以 o_tier CV 校準斜率 0.3（o7 教訓 8：依 CV 放寬）；
        0.14＝o6 MIS 開盤量對數誤差 0.57 經平方根傳到 κ（0.5×0.285）。
  s_ε：先用 0.003（30bp），n ≥ 60 時改用殘差標準差（取兩者較大值）。
規則：
  1. 累積 ≥ 60 筆非零股成交之前，只記錄、不更新、不得改正式表（輸出 status=accumulating）。
  2. ≥ 60 筆後，後驗均值落在先驗 90% 區間 [0.22, 0.78] 之外 → status=recalibrate：用後驗均值重跑 V7X 正式表，另存新版本，
     判定仍以舊表為準直到總監核准；落在區間內 → status=consistent，正式表不動。
  3. TICK 檢查：非零股成交中「距開盤 > 1 跳」的比例 > 50% → TICK 下界在實測中不成立（寫進報告）。
  4. 每 12 個月或每累積 60 筆報告一次；零股成交另報（價差對 odd 表的偏差），不併入 κ。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd

AG = Path(__file__).resolve().parent.parent
TRADES = AG / 'live_tracking' / 'main_forward' / 'trades.csv'
FIELDS = ['date', 'code', 'side', 'order_time', 'limit_px', 'fill_px', 'open_px', 'prev_close', 'ticks_from_open',
          'shares', 'is_odd', 'sigma60_prev', 'V60_prev', 'oa_share_dec', 'share_of_open_vol', 'list']
PRIOR_MU, PRIOR_SD, S_EPS0, N_MIN = 0.5, 0.17, 0.003, 60


def update(df):
    d = df[(df['is_odd'].astype(str).str.lower().isin(['0', 'false'])) & df['fill_px'].notna() & df['open_px'].notna()].copy()
    n = len(d)
    out = dict(n_nonodd=n, prior=dict(mu=PRIOR_MU, sd=PRIOR_SD), status='accumulating')
    if n == 0: return out
    side = np.where(d['side'].astype(str).str.lower().str.startswith('b'), 1.0, -1.0)
    y = side * (d['fill_px'].astype(float) / d['open_px'].astype(float) - 1) - 0.001
    oa = d['oa_share_dec'].astype(float) * d['V60_prev'].astype(float)
    q = d['shares'].astype(float) * d['prev_close'].astype(float)
    x = d['sigma60_prev'].astype(float) * np.sqrt(q / oa)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok].to_numpy(), y[ok].to_numpy()
    out['ticks_gt1_share'] = float(np.mean(d['ticks_from_open'].astype(float) > 1))
    if n < N_MIN: return out
    s_eps = S_EPS0
    k_ols = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else float('nan')
    if np.isfinite(k_ols): s_eps = max(S_EPS0, float(np.std(y - k_ols * x, ddof=1)))
    prec = 1 / PRIOR_SD ** 2 + (x * x).sum() / s_eps ** 2
    mu = (PRIOR_MU / PRIOR_SD ** 2 + (x * y).sum() / s_eps ** 2) / prec
    lo, hi = PRIOR_MU - 1.645 * PRIOR_SD, PRIOR_MU + 1.645 * PRIOR_SD
    out.update(post=dict(mu=float(mu), sd=float(1 / math.sqrt(prec))), s_eps=s_eps, kappa_ols=k_ols,
               status='recalibrate' if not (lo <= mu <= hi) else 'consistent',
               tick_bound_fails=bool(out['ticks_gt1_share'] > 0.5))
    return out


if __name__ == '__main__':
    if not TRADES.exists():
        print(json.dumps(dict(status='no_trades_file', expected=str(TRADES), fields=FIELDS), ensure_ascii=False)); sys.exit(0)
    print(json.dumps(update(pd.read_csv(TRADES, dtype={'code': str})), ensure_ascii=False, indent=1))

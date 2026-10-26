#!/usr/bin/env python3
"""重測 r1_1：技術組結論「超賣類（KD、RSI、布林下軌）扣成本前就是負超額」的質疑。

質疑內容：
  (1) T05 挑選期每筆扣成本前對等權超額 +0.09%、每日 +0.5 bp，是正的；
  (2) KD 從訊號日收盤算起毛報酬約 0，負值來自隔日開盤進場的跳空；
  (3) 只有 RSI（挑選期每筆 −0.57%）符合說法。

做法（事前固定，全部報告）：
  * 規則與事件表完全沿用 t1_rules.build_rules + tlib.make_events + tlib.dedup（同一組交易、同一出場點 x），
    只改「進場價」：
      OPEN  ＝ 隔日開盤 AO[e]（原研究、實際可成交）
      CLOSE ＝ 訊號日收盤 AC[t]（假想；訊號要收盤後才確定，實務上不可完全成交，僅用來拆解跳空）
    出場一律為原規則的 PX[x]（出場日開盤）。
  * 扣成本前（gross）；對等權超額用兩種基準：
      EW_orig ＝ 原研究近似：上市等權收盤指數 I[x−1]/I[t]−1（收盤 t → 收盤 x−1），兩種進場都用同一基準
      EW_match＝ 時點對齊：用「半日」等權指數（開盤、收盤交錯），OPEN 對 開盤 e → 開盤 x；CLOSE 對 收盤 t → 開盤 x。
               半日指數的成分＝原 r_ew 同一池（前一日 v60≥1e7、價≥10、flag∈{0,1,2}、|ret|≤漲跌幅+3%），
               隔夜段 AO[d]/AC[d−1]−1、盤中段 AC[d]/AO[d]−1，同日兩段用同一成分池。
  * 每筆超額、每持有日 bp（除以平均 x−e 交易日，與原研究相同；CLOSE 另列除以 x−t）、跳空 AO[e]/AC[t]−1 的平均、
    原始毛報酬（不扣基準）、天真 t 與「按訊號月分群」t（每月平均超額的 t，降低同日交易重疊造成的高估）。
  * 期間：S＝2005-01～2019-04（挑選）、V_all＝2019-05～2026-10（驗證）、V_off＝V_all 剔除碰到 src 3 的交易。
輸出 recheck_r1_1.json（同資料夾）。
"""
import sys, json, math, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlib as L
import t1_rules as R1

RULES = ['T01_KD_golden_K<20', 'T03_RSI<30_buy_>70_sell', 'T05_BB_touch_lower']
WINS = ['S', 'V_all', 'V_off']
t0 = time.time()


def log(*a):
    print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)


def halfday_ew_index(D):
    """回傳長度 2T 的等權指數：位置 2d＝第 d 日開盤、2d+1＝第 d 日收盤。"""
    AC, AO, F, Rt, lim = D['AC'], D['AO'], D['flag'], D['ret'], D['lim']
    elig_prev = ((D['v60'] >= 1e7) & (D['close_raw'] >= 10)).shift(1, fill_value=False)
    pool = elig_prev & F.isin([0, 1, 2]) & (Rt.abs() <= lim.values[:, None] + 0.03)
    on = (AO / AC.shift(1) - 1)            # 隔夜：收盤 d−1 → 開盤 d（還原價，含除權息）
    intra = (AC / AO - 1)                  # 盤中：開盤 d → 收盤 d
    ok = pool & on.notna() & intra.notna()
    r_on = on.where(ok).mean(axis=1).fillna(0.0).to_numpy()
    r_in = intra.where(ok).mean(axis=1).fillna(0.0).to_numpy()
    T = len(r_on)
    steps = np.empty(2 * T); steps[0::2] = r_on; steps[1::2] = r_in
    H = np.cumprod(1 + steps)
    # 檢查：兩段相乘 vs 原 r_ew（收盤對收盤）的差距
    rr = (1 + r_on) * (1 + r_in) - 1
    chk = pd.Series(rr, index=D['idx']) - D['r_ew'].fillna(0.0)
    return H, {'halfday_vs_r_ew_daily_mean_diff_bp': float(chk.mean() * 1e4), 'abs_mean_diff_bp': float(chk.abs().mean() * 1e4),
               'corr': float(np.corrcoef(rr, D['r_ew'].fillna(0.0))[0, 1])}


def tclust(x, month_key):
    s = pd.Series(x).groupby(month_key).mean()
    return L.tstat(s), int(len(s))


def stats_block(D, ev, w, H):
    m = L.win_mask(D, w).to_numpy()
    tt = ev['t'].to_numpy(); k = m[tt]
    if w == 'V_off':
        off = D['off_day'].to_numpy()
        cs = np.concatenate([[0], np.cumsum(~off)])
        k &= (cs[ev['x'].to_numpy() + 1] - cs[tt]) == 0
    e = ev.loc[k]
    t, ee, x, j = e['t'].to_numpy(), e['e'].to_numpy(), e['x'].to_numpy(), e['j'].to_numpy()
    AC, AO, PX = D['AC'].to_numpy(), D['AO'].to_numpy(), D['PX'].to_numpy()
    pc, po, px = AC[t, j], AO[ee, j], PX[x, j]
    ok = np.isfinite(pc) & np.isfinite(po) & np.isfinite(px)
    t, ee, x, j, pc, po, px = t[ok], ee[ok], x[ok], j[ok], pc[ok], po[ok], px[ok]
    g_open = px / po - 1
    g_close = px / pc - 1
    gap = po / pc - 1
    I = D['I_ew']
    ew_orig = I[x - 1] / I[ee - 1] - 1                     # 原研究（ee−1 = t）
    ew_m_open = H[2 * x] / H[2 * ee] - 1                    # 開盤 e → 開盤 x
    ew_m_close = H[2 * x] / H[2 * t + 1] - 1                # 收盤 t → 開盤 x
    ew_gap = H[2 * ee] / H[2 * t + 1] - 1                   # 等權同一夜的隔夜報酬
    ew_on_exit = H[2 * x] / H[2 * x - 1] - 1                # 等權在出場前一夜（收盤 x−1 → 開盤 x）的隔夜報酬
    hd = (x - ee).mean(); hd_c = (x - t).mean()
    mk = pd.PeriodIndex(D['idx'][t], freq='M')
    out = {'n': int(len(t)), 'n_signal_months': int(mk.nunique()), 'avg_hold_days_x_minus_e': round(float(hd), 2),
           'gap_mean_pct': float(gap.mean() * 100), 'gap_median_pct': float(np.median(gap) * 100),
           'EW_overnight_same_night_pct': float(ew_gap.mean() * 100), 'gap_excess_vs_EW_overnight_pct': float((gap - ew_gap).mean() * 100),
           'EW_overnight_exit_night_pct': float(ew_on_exit.mean() * 100),
           'EW_overnight_unconditional_daily_pct': float(np.mean(H[2 * np.arange(1, len(H) // 2)][m[1:]] / H[2 * np.arange(1, len(H) // 2) - 1][m[1:]] - 1) * 100),
           'note_benchmark_gap': 'EW_match(OPEN) − EW_orig ≈ EW 出場夜隔夜 − EW 進場夜隔夜；原研究基準少算出場夜、多算進場夜'}
    for lab, g, ewo, ewm, hdd in [('OPEN', g_open, ew_orig, ew_m_open, hd), ('CLOSE', g_close, ew_orig, ew_m_close, hd)]:
        xo, xm = g - ewo, g - ewm
        to, no = tclust(xo, mk); tm, _ = tclust(xm, mk)
        out[lab] = {'raw_gross_mean_pct': float(g.mean() * 100), 'raw_gross_median_pct': float(np.median(g) * 100),
                    'raw_gross_win_rate': float((g > 0).mean()),
                    'xs_EW_orig_pct': float(xo.mean() * 100), 'xs_EW_orig_bp_per_day': float(xo.mean() / hd * 1e4),
                    'xs_EW_orig_t_naive': L.tstat(xo), 'xs_EW_orig_t_month_cluster': to,
                    'xs_EW_match_pct': float(xm.mean() * 100), 'xs_EW_match_bp_per_day': float(xm.mean() / hd * 1e4),
                    'xs_EW_match_t_naive': L.tstat(xm), 'xs_EW_match_t_month_cluster': tm,
                    'xs_EW_match_net_0.5pct': float((xm.mean() - L.RT) * 100)}
        if lab == 'CLOSE':
            out[lab]['xs_EW_match_bp_per_day_x_minus_t'] = float(xm.mean() / hd_c * 1e4)
    # 逐年（訊號日歸屬年份）：時點對齊超額
    yr = D['idx'][t].year
    df = pd.DataFrame({'y': yr, 'o': g_open - ew_m_open, 'c': g_close - ew_m_close, 'gap': gap - ew_gap})
    out['yearly_xs_EW_match_pct'] = {int(y): {'OPEN': round(float(g['o'].mean() * 100), 2), 'CLOSE': round(float(g['c'].mean() * 100), 2),
                                              'gap_xs': round(float(g['gap'].mean() * 100), 2), 'n': int(len(g))} for y, g in df.groupby('y')}
    return out


def main():
    D = L.load(); log('loaded')
    H, chk = halfday_ew_index(D); log('halfday EW', chk)
    rules = R1.build_rules(D); log('rules built')
    orig = json.loads((L.OUT / 'results_rules.json').read_text())['rules']
    res = {'halfday_index_check': chk, 'rules': {}}
    for name in RULES:
        sig, stg, ex, hold = rules[name]
        ev = L.dedup(L.make_events(D, sig, stg, ex, hold, name))
        res['rules'][name] = {}
        for w in WINS:
            b = stats_block(D, ev, w, H)
            o = orig[name][w]['trades_all_signals']
            b['orig_reported'] = {'n': o['n'], 'gross_excess_vs_EW_pct': o['gross_excess_vs_EW_pct'], 'gross_excess_per_day_bp': o['gross_excess_per_day_bp'],
                                  'mean_gross_pct': o['mean_gross_pct']}
            res['rules'][name][w] = b
            log(f"{name:26s} {w:5s} n={b['n']:6d} gap={b['gap_mean_pct']:+.2f}% (EWon {b['EW_overnight_same_night_pct']:+.2f}) | "
                f"OPEN raw {b['OPEN']['raw_gross_mean_pct']:+.2f} xsOrig {b['OPEN']['xs_EW_orig_pct']:+.2f} xsM {b['OPEN']['xs_EW_match_pct']:+.2f} "
                f"({b['OPEN']['xs_EW_match_bp_per_day']:+.1f}bp/d tC {b['OPEN']['xs_EW_match_t_month_cluster']}) | "
                f"CLOSE raw {b['CLOSE']['raw_gross_mean_pct']:+.2f} xsOrig {b['CLOSE']['xs_EW_orig_pct']:+.2f} xsM {b['CLOSE']['xs_EW_match_pct']:+.2f} "
                f"({b['CLOSE']['xs_EW_match_bp_per_day']:+.1f}bp/d tC {b['CLOSE']['xs_EW_match_t_month_cluster']}) | orig xs {o['gross_excess_vs_EW_pct']} | "
                f"EWon exit {b['EW_overnight_exit_night_pct']:+.2f} uncond {b['EW_overnight_unconditional_daily_pct']:+.3f}")
    L.save('recheck_r1_1.json', res)
    log('saved')


if __name__ == '__main__':
    main()

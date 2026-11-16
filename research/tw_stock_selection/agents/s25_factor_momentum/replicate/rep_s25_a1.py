#!/usr/bin/env python3
"""S25 獨立複製 A1：因子動能（只用 s24 monthly.csv）。輸出 scratchpad/s25rep/a1.json（之後併入 results_rep.json）。
依 SPEC.json 文字自寫；未讀、未 import、未執行 s25lib.py／prep.py／describe.py／train.py／validate.py／posthoc.py。
python3 -I -B rep_s25_a1.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rep_s25_stats import tstat, welch_t, ann_stats, ols_nw, block_boot_sharpe_diff

H = Path(__file__).resolve().parent
AG = H.parent.parent
REP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s25rep'); REP.mkdir(parents=True, exist_ok=True)
FACS = ['SMB', 'HML', 'EPF', 'MOM', 'RMOM']
M = pd.read_csv(AG / 's24_return_attribution' / 'monthly.csv', index_col='month')
M.index = pd.PeriodIndex(M.index, freq='M')


def trend(F, w=12):
    """月 t 的趨勢＝t−w … t−1 的報酬算術和（w 個月都要有值）。"""
    return F.shift(1).rolling(w, min_periods=w).sum()


def a1(F, TR, mask=None):
    out = {'per_factor': {}}
    for f in FACS:
        ok = TR[f].notna() & F[f].notna()
        if mask is not None:
            ok &= mask
        r = F[f][ok]; w = TR[f][ok] > 0
        a, b = r[w].to_numpy(), r[~w].to_numpy()
        out['per_factor'][f] = dict(win_mean_pct=a.mean() * 100, win_t=tstat(a), win_n=int(len(a)), lose_mean_pct=b.mean() * 100, lose_t=tstat(b),
                                    lose_n=int(len(b)), diff_pct=(a.mean() - b.mean()) * 100, diff_welch_t=welch_t(a, b),
                                    uncond_mean_pct=r.mean() * 100, first=str(r.index[0]), last=str(r.index[-1]))
    # 合併（因子×月）
    ok = TR[FACS].notna() & F[FACS].notna()
    if mask is not None:
        ok = ok & mask.to_numpy()[:, None]
    win = ok & (TR[FACS] > 0); lose = ok & (TR[FACS] <= 0)
    Fv = F[FACS]
    wv = Fv.where(win).stack().dropna(); lv = Fv.where(lose).stack().dropna()
    wm = Fv.where(win).mean(axis=1); lm = Fv.where(lose).mean(axis=1)
    d = (wm - lm).dropna()
    out['pooled'] = dict(win_mean_pct=wv.mean() * 100, win_n=int(len(wv)), lose_mean_pct=lv.mean() * 100, lose_n=int(len(lv)),
                         diff_pct=(wv.mean() - lv.mean()) * 100, diff_t_monthly_ts=tstat(d), n_months_both=int(len(d)),
                         diff_ts_mean_pct=d.mean() * 100)
    # TSFM：5 因子趨勢都有值的月份
    allok = TR[FACS].notna().all(axis=1) & F[FACS].notna().all(axis=1)
    if mask is not None:
        allok &= mask
    W = (TR[FACS] > 0)[allok]; R = F[FACS][allok]
    nw = W.sum(axis=1)
    WIN = R.where(W).mean(axis=1).fillna(0.0)
    EW = R.mean(axis=1)
    LOS = R.where(~W).mean(axis=1)
    WINm = R.where(W).mean(axis=1)
    WML_a = (WINm.fillna(0) - LOS.fillna(0))                      # 解讀 a：缺一邊時，缺的那邊當 0（只有輸家＝−輸家）
    WML_b = WINm.where(WINm.notna() & LOS.notna(), WINm.fillna(-LOS))   # 同 a（顯式）
    WML_c = WINm.where(LOS.notna(), WINm).where(WINm.notna(), LOS)      # 解讀 c：只有一邊時直接用該邊報酬（含只有輸家＝+輸家）
    WML_c = (WINm - LOS).where(WINm.notna() & LOS.notna(), WML_c)
    out['tsfm'] = dict(months=[str(R.index[0]), str(R.index[-1])], n=int(len(R)), avg_n_winners=float(nw.mean()),
                       n_all_win=int((nw == 5).sum()), n_all_lose=int((nw == 0).sum()),
                       WIN=ann_stats(WIN), EW=ann_stats(EW), WML=ann_stats(WML_a), WML_alt_onlyside_raw=ann_stats(WML_c),
                       LOSERS_only=ann_stats(LOS.dropna()),
                       WIN_minus_EW_sharpe=block_boot_sharpe_diff(WIN.to_numpy(), EW.to_numpy()),
                       WIN_on_EW=ols_nw(WIN.to_numpy(), EW.to_numpy()),
                       share_months_factor_winner={f: float(W[f].mean()) for f in FACS})
    return out, dict(WIN=WIN, EW=EW, WML=WML_a, WML_c=WML_c)


if __name__ == '__main__':
    F = M[FACS]
    TR = trend(F, 12)
    res = {}
    res['A1'], S = a1(F, TR)
    TR6 = trend(F, 6)
    yrs = pd.Series(M.index.year, index=M.index)
    res['A1_robust'] = dict(trend6=a1(F, TR6)[0], ex2020=a1(F, TR, yrs != 2020)[0], since2019=a1(F, TR, yrs >= 2019)[0])
    # 現行策略對 0050 月超額，依上月底已知的因子趨勢分組
    ex = M['R_FULL'] - M['R_0050']
    sbt = {}
    for f in ('MOM', 'RMOM'):
        ok = TR[f].notna() & ex.notna()
        w = TR[f][ok] > 0; e = ex[ok]
        a, b = e[w].to_numpy(), e[~w].to_numpy()
        sbt[f] = dict(win_ex_mean_pct=a.mean() * 100, win_n=int(len(a)), lose_ex_mean_pct=b.mean() * 100, lose_n=int(len(b)), diff_welch_t=welch_t(a, b),
                      win_t=tstat(a), lose_t=tstat(b))
    res['A1_strategy_by_trend'] = sbt
    # B 用的觸發表（月份＝訊號月份；值＝上月底已知的 12 月趨勢）
    TR[['MOM', 'RMOM']].to_csv(REP / 'trend12.csv')
    pd.DataFrame(S).to_csv(REP / 'tsfm_series.csv')
    (REP / 'a1.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(res['A1']['pooled'], ensure_ascii=False))
    for f in FACS:
        print(f, {k: round(v, 3) if isinstance(v, float) else v for k, v in res['A1']['per_factor'][f].items()})
    t = res['A1']['tsfm']
    print({k: (round(t[k]['mean_ann_pct'], 3), round(t[k]['vol_ann_pct'], 3), round(t[k]['sharpe'], 3)) for k in ('WIN', 'EW', 'WML', 'WML_alt_onlyside_raw', 'LOSERS_only')})
    print(t['WIN_minus_EW_sharpe'], t['WIN_on_EW'], t['n_all_win'], t['n_all_lose'])
    print(json.dumps(sbt, ensure_ascii=False))

#!/usr/bin/env python3
"""事後診斷（不用於判定）：(a) 基準偏誤安慰劑：隨機池內日／隨機「低位」日，同樣隔日開盤進、持有 20 日，對同池等權毛異常報酬；
(b) 不可成交的 t 收盤進場（版本 A，L120，H20）毛異常報酬；(c) 地量日特性（鎖跌停占比）；(d) 隨機抽 5 筆版本 B 事件逐條核對。"""
import json, sys, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa
from btlib import tstat
rng = np.random.default_rng(0)
out = {}
POOLP = R.POOL_PREV.to_numpy() & np.isfinite(R.VOL.to_numpy())
LOWP = POOLP & R.LOWPOS.fillna(False).to_numpy()
S = R.trough(120)
def gross_ab(mask, H=20, nsamp=40000, entry='open'):
    tt, jj = np.where(mask)
    if len(tt) > nsamp:
        k = rng.choice(len(tt), nsamp, replace=False); tt, jj = tt[k], jj[k]
    rows = []
    ew = R.ew_cc.to_numpy(); LCC = np.concatenate([[0.0], np.cumsum(np.log1p(ew))])
    for t, j in zip(tt, jj):
        e = t + 1; x = e + H
        if x >= R.T: continue
        if entry == 'open':
            p0 = R.AOn[e, j]; p1 = R.PX[x, j]
            if not np.isfinite(p0): continue
            rows.append((R.IDX[t], p1 / p0 - 1 - math.expm1(R.LEW[x] - R.LEW[e])))
        else:
            p0 = R.ACn[t, j]; p1 = R.ACn[t + H, j]
            rows.append((R.IDX[t], p1 / p0 - 1 - math.expm1(LCC[t + H + 1] - LCC[t + 1])))
    d = pd.DataFrame(rows, columns=['sig', 'ab'])
    o = {}
    for w, (a, b) in R.WIN.items():
        x = d[(d.sig >= a) & (d.sig <= b)]; g = x.groupby(x.sig.dt.to_period('M'))['ab'].mean()
        o[w] = {'n': int(len(x)), 'gross_ab_pct': round(x.ab.mean() * 100, 3), 't_month': tstat(g)}
    return o
out['placebo_random_pool_day_open_H20'] = gross_ab(POOLP)
out['placebo_random_lowpos_day_open_H20'] = gross_ab(LOWP)
out['A_L120_trough_open_H20_gross'] = gross_ab(S)
out['A_L120_trough_tclose_H20_gross_unexecutable'] = gross_ab(S, entry='close')
# 地量日特性
tt, jj = np.where(S)
rt = R.ADJn[tt, jj] / R.ADJn[tt - 1, jj] - 1
lim = np.where(R.IDX[tt] < pd.Timestamp('2015-06-01'), 0.07, 0.10)
out['trough_day_profile_L120'] = {'n': int(len(tt)), 'share_ret_le_-limit+0.6pct(跌停附近)': round(float(np.nanmean(rt <= -lim + 0.006)), 4),
                                  'median_ret_pct': round(float(np.nanmedian(rt)) * 100, 2),
                                  'share_on_2025-04-07': round(float((R.IDX[tt] == pd.Timestamp('2025-04-07')).mean()), 4)}
ev = pd.read_csv(HERE / 'events_B_L120_H20.csv.gz', parse_dates=['sig', 'entry'], dtype={'code': str})
ldn = set(zip(R.IDX[tt][rt <= -lim + 0.006], np.array(R.cols)[jj[rt <= -lim + 0.006]]))
ev['ldn'] = [(s, c) in ldn for s, c in zip(ev.sig, ev.code)]
o = {}
for w, (a, b) in R.WIN.items():
    x = ev[(ev.sig >= a) & (ev.sig <= b) & ~ev.ldn]; g = x.groupby(x.entry.dt.to_period('M'))['ab'].mean()
    o[w] = {'n': int(len(x)), 'ab_net_pct': round(x.ab.mean() * 100, 3), 't_month': tstat(g)}
out['B_L120_H20_excl_limitdown_trough'] = o
chk = []
for i in rng.choice(len(ev), 5, replace=False):
    r = ev.iloc[i]; j = R.cols.index(r.code); t = R.IDX.get_loc(r.sig); e = R.IDX.get_loc(r.entry); c = e - 1
    v = R.VOL.iloc[:, j]
    chk.append({'code': r.code, 'trough': str(r.sig.date()), 'confirm': str(R.IDX[c].date()), 'entry': str(r.entry.date()),
                'V_t/min120': round(float(v.iat[t] / v.iloc[t - 119:t + 1].min()), 3), 'adj/MA60': round(float(R.ADJn[t, j] / R.MA60.iat[t, j]), 3),
                'V60_prev_mn': round(float(R.V60P[t, j]) / 1e6, 1), 'V_c/MA5_c': round(float(v.iat[c] / R.VMA5.iat[c, j]), 2),
                'ret_c_pct': round((R.ADJn[c, j] / R.ADJn[c - 1, j] - 1) * 100, 2), 'days_t_to_c': int(c - t),
                'hold': int(r.hold), 'why': r.why, 'raw_pct': round(r.raw * 100, 2)})
out['spot_check_B'] = chk
print(json.dumps(out, ensure_ascii=False, indent=1))
f = HERE / 'results.json'; Dd = json.loads(f.read_text(encoding='utf-8')); Dd['diagnostics_posthoc'] = out
f.write_text(json.dumps(Dd, ensure_ascii=False, indent=1), encoding='utf-8')

#!/usr/bin/env python3
"""S14 第 0 步：處置名單的資料檢查（涵蓋、PIT、可買比例、事件密度、和 SURGE 名單重疊），不算任何報酬。
python3 -I -B s0_datacheck.py → data_check.json"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s14lib
from s14lib import s4lib, s2lib, stlib

H = Path(__file__).resolve().parent
t0 = time.time()
P = s14lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
U0 = stlib.universe(P, 'twse'); INNOV, _ = s4lib.innov_panel(P); U = U0 & ~INNOV
D = s14lib.disp_table()
out = dict(source=str(s14lib.DISP_CSV.relative_to(s14lib.AG)), sha256=s14lib.DISP_SHA, n_rows=int(len(D)),
           announce_range=[str(D.announce.min().date()), str(D.announce.max().date())])
out['pit'] = dict(all_announce_before_start=bool((D.announce < D.start).all()), lag_td_counts=D.lag_td.value_counts().sort_index().astype(int).to_dict())
D15 = D[D.start >= '2015-01-01']
out['since2015_kind_nth'] = {f'{k}|nth{n}': int(v) for (k, n), v in D15.groupby(['kind', 'nth']).size().items()}
inpanel = D15.code.isin(set(P.COLS))
out['since2015_in_panel_share'] = float(inpanel.mean())
D15 = D15.copy(); D15['days'] = (D15.end - D15.start).dt.days
D15['regime'] = np.where(D15.start >= '2026-08-10', 'new_5d', 'old_10d')
out['calendar_days_start_to_end'] = {r: dict(median=float(g.days.median()), min=int(g.days.min()), max=int(g.days.max())) for r, g in D15.groupby('regime')}
res = {}
for entry in ('first', 'all'):
    CH = s14lib.disp_chains(P, D, entry=entry)
    CH = CH[CH.k >= P.t0 - 1]
    ok = U[CH.k.to_numpy(), CH.j.to_numpy()]
    hold = (CH.e_row - CH.s_row + 1)
    o = P.O[CH.s_row.to_numpy(), CH.j.to_numpy()]
    lim = o >= P.LIMUP[CH.s_row.to_numpy(), CH.j.to_numpy()] - 1e-9
    yr = pd.DatetimeIndex(P.IDX[CH.s_row.to_numpy()]).year
    pit_ok = bool((CH.announce0.to_numpy() < P.IDX[CH.s_row.to_numpy()]).all())
    res[entry] = dict(n_chains_since_2015_07=int(len(CH)), n_eligible_univ=int(ok.sum()), eligible_share=float(ok.mean()),
                      n_extended=int((CH.n_ext > 0).sum()), hold_tdays_median=float(hold.median()), hold_tdays_p90=float(hold.quantile(0.9)), hold_tdays_max=int(hold.max()),
                      open_limit_up_share_eligible=float(lim[ok].mean()), no_open_share_eligible=float((~np.isfinite(o[ok])).mean()),
                      announce_before_entry_day_all=pit_ok,
                      eligible_by_year={int(y): int(v) for y, v in pd.Series(ok, index=yr).groupby(level=0).sum().items()},
                      eligible_per_month_median_2019_2023=float(pd.Series(ok, index=pd.DatetimeIndex(P.IDX[CH.s_row.to_numpy()])).loc['2019':'2023'].resample('ME').sum().median()),
                      eligible_per_month_mean_2024_=float(pd.Series(ok, index=pd.DatetimeIndex(P.IDX[CH.s_row.to_numpy()])).loc['2024':].resample('ME').sum().mean()))
    if entry == 'first':
        # 和 SURGE（S10 E0）買進名單重疊（只看名單，不看報酬）：事件股是否在訊號列之前最近一次 SURGE 買進名單或 L_MA 前 20 名
        RT = s2lib.rev_tables(P)
        SE = s14lib.s10lib.make_signal_s10(P, U, RT, shift=0)
        rows = np.array([x['k'] for x in SE['info']])
        tg = {x['k']: set(x['targets']) for x in SE['info']}
        X0 = SE['XV']['E0']
        e = CH[ok].reset_index(drop=True)
        inbuy = intop20 = 0
        for r in e.itertuples():
            m = rows[rows <= r.k]
            if not len(m):
                continue
            kk = m[-1]
            inbuy += r.code in tg[kk]
            intop20 += (not X0[kk, r.j])
        res[entry]['overlap_with_surge'] = dict(n=int(len(e)), in_latest_surge_buy_list=int(inbuy), in_latest_surge_top20=int(intop20),
                                               share_buy=float(inbuy / len(e)), share_top20=float(intop20 / len(e)))
        del RT, SE, X0
out['chains'] = res
# 上櫃：本題主檔只有上市；screener disposal_5y（2021-09 起，上市＋上櫃，無第一次／第二次欄位）只在事後描述用
d5 = pd.read_csv(s14lib.EXTRA / 'disposal_5y.csv.gz', dtype={'code': str})
mk = {c: int(P.MK[:, P.ci[c]].max()) if c in P.ci else 0 for c in d5.code.unique()}
d5['mk'] = d5.code.map(mk)
out['disposal_5y'] = dict(rows=int(len(d5)), date_range=[d5.date.min(), d5.date.max()], by_market={str(k): int(v) for k, v in d5.mk.value_counts().items()})
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

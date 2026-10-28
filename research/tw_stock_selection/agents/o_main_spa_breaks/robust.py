#!/usr/bin/env python3
"""穩健性（PREREG 後執行；判定欄的端點與起點敏感度，o3／o4 方法教訓）：
F_long × W_long × L6 的 SPA_c（主策略檢定量）與 StepM p，在下列樣本變動下重算（所有成員同步刪窗）：
起點平移 1 年、2 年；拿掉第一年；拿掉主策略最好的 1／3／5／10 個窗；拿掉主策略最好的日曆年；拿掉最後 1 窗；
另報主策略逐年對 0050 超額（年內窗加總）與日曆月 t。寫入 RESULTS_ROBUST。"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C, spa_lib as S
RES = C.check_prereg()
PR = RES['PREREG']; MAIN_ID = PR['main_id']
T0 = time.time()

def main():
    D = pd.read_parquet(C.DATA / 'family_excess.parquet'); MM = pd.read_csv(C.DATA / 'family_members.csv')
    mem = MM[MM.F_long].member_id.tolist(); f = mem.index(MAIN_ID)
    m = C.wmask(D.index, *C.WINDOWS['W_long'])
    X = D.loc[m, mem].fillna(0.0); x = X[MAIN_ID]
    order = x.sort_values(ascending=False).index
    yr = x.groupby(x.index.year).sum(); best_year = int(yr.idxmax())
    variants = {'base': X.index,
                'start+1y': X.index[X.index >= pd.Timestamp('2007-04-01')],
                'start+2y': X.index[X.index >= pd.Timestamp('2008-04-01')],
                'drop_first_year': X.index[X.index >= pd.Timestamp('2007-04-01')],
                'drop_last1': X.index[:-1]}
    for k in (1, 3, 5, 10): variants[f'drop_best{k}'] = X.index.difference(order[:k])
    variants[f'drop_best_year_{best_year}'] = X.index[X.index.year != best_year]
    out = {'variants': {}}
    for name, ix in variants.items():
        Y = X.loc[ix].to_numpy(); n = len(Y)
        Wc = S.sb_counts(n, 6, PR['spa']['B'], PR['spa']['seed_base'] + 9000 + n)
        o = S.mcp_tests(Y, Wc, 6, PR['spa']['variance'], focus=f)
        out['variants'][name] = {'n': n, 't_main': o['focus_t'], 'rank': o['focus_rank'], 'p_SPA_c_focus': o['p_SPA_c_focus'],
                                 'p_StepM_focus': o['p_StepM_focus'], 'p_SPA_c_familymax': o['p_SPA_c'], 'max_t': o['max_t']}
        print(f'[{time.time() - T0:.0f}s]', name, out['variants'][name], flush=True)
    out['yearly_excess_pp'] = {int(k): float(v * 100) for k, v in yr.items()}
    out['best_year'] = best_year
    # 日曆月 t（主策略日頻淨值，月底取值）
    nav = pd.read_parquet(C.V4DIR / 'navs_v4.parquet')['AB_V4|normal']; e50 = C.e50_nav()
    me = nav.groupby(nav.index.to_period('M')).last(); be = e50.reindex(nav.index.union(e50.index)).ffill().reindex(nav.index).groupby(nav.index.to_period('M')).last()
    r = (me.pct_change() - be.pct_change()).dropna()
    out['calendar_month_t'] = C.tstat(r.to_numpy()); out['calendar_month_n'] = int(len(r))
    C.save_key('RESULTS_ROBUST', out)

if __name__ == '__main__':
    main()

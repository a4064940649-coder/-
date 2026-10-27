#!/usr/bin/env python3
"""診斷（不影響判定）：只上櫃版與合併版的極端持股期報酬、公司行動 flag、缺口來源、去極值後的 t。輸出 results_diag.json。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tplib as T  # noqa: E402
from btlib import tstat  # noqa: E402
L = T.L


def main():
    S = L.Setup(L.yoy_long(), start='2005-01-01'); dates = S.dates
    tw = T.twse_panel(); tp = T.load_tpex(); cb = T.combine(tw, tp)
    cnt = tp['adj'].notna().sum(axis=1); med = cnt.rolling(21, center=True, min_periods=5).median()
    g_tp = (cnt >= 0.85 * med) & L.GOOD
    tpflag = pd.read_parquet(T.TPP / 'flag.parquet').reindex(T.IDX)
    tpsrc = pd.read_parquet(T.TPP / 'srcname.parquet').reindex(T.IDX)
    twflag = pd.read_parquet(L.P / 'flag.parquet').reindex(T.IDX)
    out = {}
    for name, panel, k in (('TPEX_top300', tp, 300), ('COMB_top400', cb, 400), ('TWSE_top300', tw, 300)):
        U = T.Universe(panel, k, name, good=(L.GOOD if name.startswith('TWSE') else g_tp))
        U.setup(T.yoy_long(U.COLS), dates)
        ev, pools = U.selection(30)
        exs = sorted(ev); rows = []
        for a, b in zip(exs[:-1], exs[1:]):
            for j in ev[a][1]:
                r = U.AC[b, j] / U.AO[a, j] - 1 if np.isfinite(U.AO[a, j]) else np.nan
                c = U.COLS[j]
                fl = tpflag[c].iloc[a:b + 1] if c in tpflag.columns and c not in tw['adj'].columns else twflag[c].iloc[a:b + 1] if c in twflag.columns else pd.Series(dtype=float)
                rows.append((str(T.IDX[a].date()), c, r, int((fl.isin([3, 4, 5])).sum())))
            pr = [U.AC[b, j] / U.AO[a, j] - 1 for j in pools[a][1] if np.isfinite(U.AO[a, j])]
            rows.append((str(T.IDX[a].date()), '__EW__', float(np.nanmean(pr)), 0))
        df = pd.DataFrame(rows, columns=['ex', 'code', 'r', 'n_flag345'])
        picks = df[df.code != '__EW__']; ew = df[df.code == '__EW__'].set_index('ex')['r']
        top = picks.sort_values('r', ascending=False).head(12)
        port = picks.groupby('ex')['r'].mean()
        win = picks.assign(r=picks['r'].clip(-0.5, 0.6)).groupby('ex')['r'].mean()
        noflag = picks[picks.n_flag345 == 0].groupby('ex')['r'].mean()
        def tt(x, a, b):
            x = x[(x.index >= a) & (x.index <= b)]; e = ew.reindex(x.index)
            return {'mean_excess_vs_EW_pct': round(float((x - e).mean() * 100), 2), 't': tstat(x - e), 'n': int(len(x))}
        o = {'top12_pick_period_returns': top.round(4).values.tolist(),
             'share_pick_periods_with_flag345': round(float((picks.n_flag345 > 0).mean()), 4),
             'pick_periods_ret_gt_100pct': int((picks.r > 1).sum()), 'pick_periods_total': int(len(picks))}
        for lab, (a, b) in {'S1_2008-05_2020-12': ('2008-05-01', '2020-12-31'), 'S2a_2021_2022': ('2021-01-01', '2022-12-31')}.items():
            o[lab] = {'gross_vs_EW': tt(port, a, b), 'winsor_-50_+60_vs_EW': tt(win, a, b), 'drop_flag345_picks_vs_EW': tt(noflag, a, b)}
        if name != 'TWSE_top300':
            o['tpex_pick_source_share'] = tpsrc.reindex(columns=[c for c in tpsrc.columns]).stack().value_counts(normalize=True).round(3).to_dict() if False else None
        out[name] = o
        print(name, json.dumps({k: v for k, v in o.items() if k.startswith('S') or k.startswith('share') or k.startswith('pick')}, ensure_ascii=False), flush=True)
        print(top.to_string(), flush=True)
    (HERE / 'results_diag.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""strict vs lenient 的差異拆解：哪些期、哪些股票（窗內有缺值的股票）造成年化差距。"""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ia_core as K
nv = pd.read_parquet(K.HERE / 'navs.parquet'); pk = json.loads((K.HERE / 'picks.json').read_text())
S52s, V60s = K.rolling_signals('strict')
out = {}
for P, a, b in (('P2_rev5y', '2021-03-01', '2026-10-31'), ('P1_long', '2006-04-01', '2020-12-31')):
    for n in (10, 30):
        s = nv[f'{P}|n{n}|0.001'].dropna(); l = nv[f'{P}_lenient|n{n}|0.001'].dropna()
        ks = sorted(pk[f'{P}|n{n}']); ks = [k for k in ks if a <= k <= b]
        rows = []
        for i, k in enumerate(ks[:-1]):
            e0, e1 = pd.Timestamp(k), pd.Timestamp(ks[i + 1])
            rs = s.loc[e1] / s.loc[s.index[s.index.get_loc(e0) - 1]] - 1; rl = l.loc[e1] / l.loc[l.index[l.index.get_loc(e0) - 1]] - 1
            A, B = set(pk[f'{P}|n{n}'][k]), set(pk[f'{P}_lenient|n{n}'][k])
            t = K.IDX.get_loc(e0) - 1
            newc = [c for c in B - A if not np.isfinite(S52s[t, K.CI[c]])]
            # 期間報酬（開盤到下一換股開盤）
            def hp(c):
                j = K.CI[c]; i0 = K.IDX.get_loc(e0); i1 = K.IDX.get_loc(e1)
                g = K.Cn[i0, j] / K.On[i0, j] * np.prod(K.GROW[i0 + 1:i1 + 1, j]) * K.On[i1, j] / K.Cn[i1, j] if np.isfinite(K.On[i1, j]) else np.nan
                return round(float(g - 1), 3)
            rows.append({'exec': k, 'diff_pp': round((rl - rs) * 100, 2), 'lenient_only': [(c, hp(c)) for c in B - A], 'strict_only': [(c, hp(c)) for c in A - B], 'lenient_only_missing_in_strict': newc})
        df = pd.DataFrame(rows)
        df['year'] = df['exec'].str[:4]
        out[f'{P}|n{n}'] = {'by_year_sum_diff_pp': df.groupby('year')['diff_pp'].sum().round(2).to_dict(),
                            'top_neg': df.nsmallest(6, 'diff_pp').to_dict('records'), 'top_pos': df.nlargest(6, 'diff_pp').to_dict('records')}
        print(P, n, out[f'{P}|n{n}']['by_year_sum_diff_pp'])
        for r in out[f'{P}|n{n}']['top_neg'][:4] + out[f'{P}|n{n}']['top_pos'][:4]: print('   ', r)
(K.HERE / 'results_lenient_diag.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

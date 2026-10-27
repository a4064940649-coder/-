#!/usr/bin/env python3
"""診斷：A_exFIN／BASE n10 在舊 value 下重跑，解釋與 rev_announce 的 >1pp 差距（評價終點統一 2026-10-12）。"""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ia_core as K
sl = pd.read_csv(K.X.parent / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
FIN = set(sl.index[sl['industry'] == '金融保險業']) | {c for c in K.COLS if c[:2] == '28' or c in ('5876', '5880')}
SEG = {'P1': ('2006-04-01', '2020-12-31'), 'P2': ('2021-01-01', '2026-10-12'), 'V24': ('2024-01-01', '2026-10-12')}
b, s = K.rates(0.001); out = {}
for vf in ('value.parquet', 'value_orig_dupbug.parquet'):
    K.V = pd.read_parquet(K.P / vf); K._RS.clear()
    for v, kw in (('BASE', {}), ('A_exFIN', {'fin_exclude': FIN})):
        for tb in ('stable', 'reverse'):
            S = K.Strategy(K.yoy_from_long(), '2005-01-01', tiebreak=tb, **kw)
            t0 = K.IDX.get_loc(S.sig_days[0]); t1 = K.IDX.get_loc(S.end_day); sd = S.sig_days + [S.end_day]
            pl = S.plan(10); nav, _ = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1)
            o = {k: round(K.stats(nav, *w, sd, K.E50)['CAGR'] * 100, 2) for k, w in SEG.items()}
            out[f'{vf}|{v}|{tb}'] = o; print(vf, v, tb, o, flush=True)
(K.HERE / 'results_ra_oldvalue.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

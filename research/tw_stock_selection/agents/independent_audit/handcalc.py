#!/usr/bin/env python3
"""抽 10 個換股日，用純 Python 迴圈手算前 10 名的 52 週高點比、營收 3 月年增率、百分位、分數與隔日開盤成交價，並與引擎結果核對。
輸出 results_handcalc.json。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ia_core as K

DATES = {'long': ['2006-04-11', '2008-10-13', '2011-08-11', '2013-02-18', '2015-06-11', '2018-04-11', '2020-03-11'],
         'rev5y': ['2022-01-11', '2024-02-15', '2026-04-13']}
Ys = {'long': K.yoy_from_long(), 'rev5y': K.yoy_from_rev5y()}
starts = {'long': '2005-01-01', 'rev5y': '2021-01-01'}
good_days = [i for i in range(len(K.IDX)) if K.GOOD[i]]
out = {}; allok = True

def pct_rank(vals):
    """平均名次百分位（同 pandas rank(pct=True)）：純迴圈。"""
    items = sorted(vals.items(), key=lambda kv: kv[1])
    n = len(items); res = {}; i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]: j += 1
        r = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1): res[items[k][0]] = r / n
        i = j + 1
    return res

for src, dl in DATES.items():
    Y = Ys[src]; S = K.Strategy(Y, starts[src]); held = set()
    plan = {p[3]: p for p in S.plan(10)}
    sdl = list(S.sig_days)
    for ds in dl:
        d = [x for x in sdl if x >= pd.Timestamp(ds)][0]; ds = str(d.date()); t = K.IDX.get_loc(d); e = t + 1
        k = sdl.index(d)
        held = set(K.COLS[j] for j in plan[sdl[k - 1]][1]) if k > 0 and sdl[k - 1] in plan else set()
        gpos = good_days.index(t)
        win = good_days[gpos - 251:gpos + 1]; win60 = good_days[gpos - 59:gpos + 1]
        # 股池：60 日均值（迴圈）
        v60 = {}; s52 = {}; y3 = {}
        m = pd.Period(d, 'M'); months = [m - 1, m - 2, m - 3]
        dsp = K.disposal_mask(d)
        for j, c in enumerate(K.COLS):
            vs = [K.V.iat[i, j] for i in win60]
            if any(not np.isfinite(x) for x in vs): continue
            v = sum(vs) / 60.0; px = K.Cn[t, j]
            if not (v >= 1e7 and np.isfinite(px) and px >= 10): continue
            v60[c] = v
        top = sorted(v60, key=lambda c: -v60[c])[:300]
        for c in top:
            j = K.CI[c]
            a = [K.MYADJ.iat[i, j] for i in win]
            if any(not np.isfinite(x) for x in a): continue
            yy = [Y.at[mm, c] if (mm in Y.index and c in Y.columns) else np.nan for mm in months]
            if any(not np.isfinite(x) for x in yy): continue
            if dsp[j]: continue
            if not (np.isfinite(K.On[e, j]) and np.isfinite(K.Cn[e, j])): continue
            s52[c] = a[-1] / max(a); y3[c] = sum(yy) / 3.0
        r1 = pct_rank(s52); r2 = pct_rank(y3)
        sc = {c: 0.5 * r1[c] + 0.5 * r2[c] for c in s52}
        order = sorted(sc, key=lambda c: (-round(sc[c], 10), K.CI[c]))
        rows = []; picks = []
        prev_held = held
        for c in order:
            j = K.CI[c]
            ref = K.ref_price(e)[j]; lim = 0.07 if K.IDX[e] < pd.Timestamp('2015-06-01') else 0.10
            lu = float(K.limit_up_price(np.array([ref]), lim)[0])
            blocked = (K.On[e, j] >= lu - 1e-6) and (c not in prev_held)
            row = {'code': c, 'close_d': K.Cn[t, j], 's52': round(s52[c], 4), 'pct52': round(r1[c], 4),
                   'yoy_m-1,m-2,m-3': [round(float(Y.at[mm, c]), 2) for mm in months], 'yoy3': round(y3[c], 2), 'pctrev': round(r2[c], 4),
                   'score': round(sc[c], 4), 'exec_day': str(K.IDX[e].date()), 'open_raw': K.On[e, j], 'ref': round(float(ref), 2),
                   'limit_up': round(lu, 2), 'open_vs_ref_pct': round((K.On[e, j] / ref - 1) * 100, 2), 'flag_exec': int(K.Fn[e, j]) if np.isfinite(K.Fn[e, j]) else None,
                   'blocked': bool(blocked)}
            rows.append(row)
            if not blocked: picks.append(c)
            if len(picks) == 10: break
        eng = [K.COLS[j] for j in plan[d][1]] if d in plan else None
        ok = eng is not None and set(eng) == set(picks)
        allok &= ok
        out[ds] = {'source': src, 'pool_size': len(sc), 'months_used': [str(x) for x in months], 'rows': rows, 'hand_picks': picks, 'engine_picks': eng, 'match': ok}
        print(ds, src, 'pool', len(sc), 'match', ok)
        for r in rows: print('   ', r['code'], r['close_d'], r['s52'], r['pct52'], r['yoy_m-1,m-2,m-3'], r['yoy3'], r['pctrev'], r['score'], r['exec_day'], r['open_raw'], r['ref'], r['limit_up'], r['blocked'])
out['all_match'] = bool(allok)
(K.HERE / 'results_handcalc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')

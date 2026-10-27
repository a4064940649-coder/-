"""穩健性檢查（事前固定）：
1. 動能配對對照：每筆事件，在同一時點挑「流動性合格、60 日均成交值前 150 名（只用於配對大小，不當市值用）、
   非本期事件股」中，過去 60 日報酬（close(t−60)→close(t)）最接近的 10 檔當對照組，同一窗口算對照組平均報酬；
   超額＝事件股 − 對照組。R1（納入）用 t＝A；R2（刪除）用 t＝E。
2. 去極端：排除 2021、2026 兩年；截尾 10%（上下各 5%）平均。
結果寫回 results.json 的 robust 欄。
"""
import json, math
from pathlib import Path
import sys
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # -I 模式不含腳本目錄
import study as S  # noqa: E402


def main():
    X, D, K = S.run()
    idx = D['idx']; adj = D['adj']; V60 = D['value'].rolling(60, min_periods=40).mean()
    LA = D['LA']
    ev_codes = X.groupby(['etf', 'review'])['code'].apply(set).to_dict()
    rows = []
    for _, r in X[(X.in_panel == True) & (X.liq_ok_A == True)].iterrows():
        iA, iE = K.pos(r.ann), K.pos(r.eff)
        for rule, t0, (b, s) in [('R1', iA, (iA + 1, iE - 1)), ('R2', iE, (iE, iE + 20))]:
            if s >= len(idx) or t0 < 61: continue
            liq = D['liq'].iloc[t0]; v = V60.iloc[t0].where(liq).dropna().sort_values(ascending=False)
            pool = [c for c in v.index[:150] if c not in ev_codes.get((r.etf, r.review), set())]
            mom = (LA.iloc[t0][pool] - LA.iloc[t0 - 60][pool]).dropna()
            if r.code not in LA.columns: continue
            me = LA[r.code].iat[t0] - LA[r.code].iat[t0 - 60]
            if not np.isfinite(me) or len(mom) < 20: continue
            ctrl = (mom - me).abs().sort_values().index[:10]
            if rule == 'R1':
                ev = K.oc_trade(r.code, b, s)[0]; cr = [K.oc_trade(c, b, s)[0] for c in ctrl]
            else:
                ev = K.cc(r.code, b, s)[0]; cr = [K.cc(c, b, s)[0] for c in ctrl]
            cm = np.nanmean(cr)
            rows.append(dict(etf=r.etf, review=r.review, code=r.code, action=r.action, rule=rule, year=int(str(r.eff)[:4]), ev=ev, ctrl=cm, ex=ev - cm,
                             ab=r[f'{rule}_ab']))
    R = pd.DataFrame(rows)
    out = {}
    for (etf, act, rule), g in R.groupby(['etf', 'action', 'rule']):
        k = f'{etf}_{act}_{rule}'
        g = g.dropna(subset=['ex'])
        lo, hi = g.ab.quantile([0.05, 0.95])
        out[k] = {'vs_momentum_control_net05': S.stats(g, 'ex', 0.005),
                  'vs_EW_net05_same_sample': S.stats(g, 'ab', 0.005),
                  'excl_2021_2026_vs_EW_net05': S.stats(g[~g.year.isin([2021, 2026])], 'ab', 0.005),
                  'excl_2021_2026_vs_ctrl_net05': S.stats(g[~g.year.isin([2021, 2026])], 'ex', 0.005),
                  'trim10_vs_EW_net05': S.stats(g[(g.ab >= lo) & (g.ab <= hi)], 'ab', 0.005)}
    R.to_csv(HERE / 'robust_rows.csv', index=False)
    res = json.loads((HERE / 'results.json').read_text())
    res['robust'] = out
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    for k, v in out.items():
        if k.startswith('TW50') or k.endswith('add_R1') or k.endswith('del_R2'):
            print(k, {kk: (vv['n'], vv.get('mean%'), vv.get('median%'), vv.get('win%'), vv.get('t_review')) for kk, vv in v.items()})


if __name__ == '__main__':
    main()

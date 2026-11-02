#!/usr/bin/env python3
"""步驟六：把 V7X 正式可執行引擎（凍結版）套到其他題目的名單。
輸入：任一 CSV（欄位 signal, exec, picks[, seed][, j]；picks 以逗號分隔代碼），等權 k 檔（k＝該期 picks 數）。
輸出：4 級距 × FIX／GROW × K5T（正式）／TICK（敏感度）× 一般／壓力 的年化、t、2019 年起；四項修正全開（正式）與原設定兩版。
權重非等權的組合（例如 capmatched F1 的 0050 骨架權重）本引擎不支援：只能報「無法套用」，不得改用等權近似當正式數字。
用法：python3 -I -B s7_apply.py <名單.csv> <標籤>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_data as D, x_exec as X, x_stats as S, x_run as XR

HERE = Path(__file__).resolve().parent
FINAL = json.loads((HERE / 'results.json').read_text())['PREREG']['final_cfg'] if (HERE / 'results.json').exists() else None


def load(p):
    d = pd.read_csv(p, dtype={'picks': str})
    if 'seed' not in d: d['seed'] = 0
    if 'j' not in d: d['j'] = 'X'
    out = {}
    for (j, s), g in d.groupby(['j', 'seed']):
        L = []
        for sig, e, pk in zip(g.signal, g.exec, g.picks):
            ei = D.TPOS[pd.Timestamp(e)]; si = D.TPOS[pd.Timestamp(sig)]
            L.append((ei, np.array([D.CI[c] for c in str(pk).split(',') if c in D.CI], int), si))
        out[(j, s)] = sorted(L)
    return out


if __name__ == '__main__':
    assert FINAL is not None, 'PREREG 不存在：先凍結 V7X'
    P = load(sys.argv[1]); tag = sys.argv[2]
    res = {}
    for (j, s), pl in P.items():
        for (m, c, t, v) in XR.cells():
            for ver, extra in (('V7X', FINAL), ('V7orig', {})):
                cfg = dict(tier=XR.TIERS[t], mode=v, model=m, cost=c); cfg.update(extra)
                nav, acc = X.run_one(pl, cfg)
                sd = pd.DatetimeIndex([D.IDX[x[2]] for x in pl]); e = nav.index[-1]
                xw = ((nav.reindex(sd.append(pd.DatetimeIndex([e]))).pct_change()) - (S.E50.reindex(sd.append(pd.DatetimeIndex([e]))).pct_change())).iloc[1:]
                yrs = (e - nav.index[0]).days / 365.25
                res[f'{ver}|{j}|{s}|{m}|{c}|{t}|{v}'] = dict(CAGR=float((nav.iloc[-1]) ** (1 / yrs) - 1) * 100, t_0050=S.tstat(xw),
                                                             t_Y19on=S.tstat(xw[xw.index >= '2019-01-01']))
    (HERE / 'out' / f'apply_{tag}.json').write_text(json.dumps(res, ensure_ascii=False, indent=0))
    print('done', len(res))

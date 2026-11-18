#!/usr/bin/env python3
"""S71 第 0 步之三（純描述，只讀月營收檔；不載入價格、不算任何策略報酬）：B2 估計量用 1 個月（2012-12→2013-01）還是累計到 3 個月
（2013 年前 n 個月合計 ÷ 2012 年最後 n 個月合計，n＝1、2、3；各自減去 2008～2012 同一算法的中位數）。
指標用 2013-04～2013-12（9 個月，n ≤ 3 的估計都沒用到）：持續偏高＝≥ 7 個月校正後年增率 > +30%；持續偏低＝≥ 7 個月 < −30%；
參考＝2012、2014 同樣 9 個月（未校正）平均。k＝max(1, exp(a_n))（ZERO，step0b 選出）。python3 -I -B step0c.py → step0c.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s71lib as S
from s71lib import stlib

raw = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
sii = sorted(raw.loc[raw['market'] == 'sii', 'code'].unique())
Rv = S.rev_table(None)[sii]
P_ = lambda s: pd.Period(s, 'M')
Y = Rv / Rv.shift(12) - 1


def jump_n(y, n):
    """ln(y 年前 n 個月合計 ÷ y−1 年最後 n 個月合計)；n 個月都要有值。"""
    a = Rv.loc[P_(f'{y}-01'):P_(f'{y}-{n:02d}')]; b = Rv.loc[P_(f'{y - 1}-{13 - n:02d}'):P_(f'{y - 1}-12')]
    return np.log(a.sum(min_count=n) / b.sum(min_count=n))


def abn(y, n, prior):
    pri = pd.concat([jump_n(v, n) for v in prior], axis=1)
    a = jump_n(y, n) - pri.median(axis=1, skipna=True)
    a[pri.notna().sum(axis=1) < S.MIN_PRIOR] = np.nan
    return a


def pers(Yy, y, cols=None):
    yy = Yy.loc[P_(f'{y}-04'):P_(f'{y}-12')]
    if cols is not None:
        yy = yy.loc[:, cols]
    ok = yy.notna().sum() == 9
    return dict(n=int(ok.sum()), hi=float(((yy > .3).sum() >= 7)[ok].mean()), lo=float(((yy < -.3).sum() >= 7)[ok].mean()),
                median_pct=float(yy.loc[:, ok].median(axis=1).mean() * 100), share_gt50=float((yy.loc[:, ok] > .5).mean(axis=1).mean()),
                share_lt_m30=float((yy.loc[:, ok] < -.3).mean(axis=1).mean()))


ref = {str(y): pers(Y, y) for y in (2012, 2014)}
rhi = (ref['2012']['hi'] + ref['2014']['hi']) / 2; rlo = (ref['2012']['lo'] + ref['2014']['lo']) / 2
out = dict(reference_years=ref, reference=dict(hi=rhi, lo=rlo), variants={}, placebo_sd={})
base = abn(2013, 1, S.PRIOR_YEARS).notna() & abn(2013, 3, S.PRIOR_YEARS).notna()
out['raw_2013'] = pers(Y, 2013, base[base].index)
for n in (1, 2, 3):
    a = abn(2013, n, S.PRIOR_YEARS)
    k = np.exp(a.clip(lower=0))
    A = Rv.to_numpy(float).copy(); pre = np.asarray(Rv.index < S.BREAK)
    A[pre] = A[pre] * k.reindex(Rv.columns).to_numpy()[None, :]
    Ra = pd.DataFrame(A, index=Rv.index, columns=Rv.columns)
    v = pers(Ra / Ra.shift(12) - 1, 2013, base[base].index)
    v['distance'] = abs(v['hi'] - rhi) + abs(v['lo'] - rlo)
    out['variants'][f'n{n}'] = v
    # 安慰劑年（沒有口徑變動）異常跳動的離散度＝估計雜訊
    pl = pd.concat([abn(2011, n, range(2006, 2011)), abn(2012, n, range(2007, 2012))]).dropna()
    out['placebo_sd'][f'n{n}'] = dict(n=int(len(pl)), mad_sd=float(1.4826 * (pl - pl.median()).abs().median()), p10=float(pl.quantile(.1)), p90=float(pl.quantile(.9)), median=float(pl.median()))
(S.HERE / 'step0c.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False))

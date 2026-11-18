#!/usr/bin/env python3
"""S71 第 0 步之二（純描述，只讀月營收檔；不載入價格、不算任何策略報酬）：B2 的「不校正帶」（deadband）怎麼定。
step0.py 顯示以安慰劑 P90 當不校正帶時，校正後 2013-02～12 的年增率中位數偏低（−7.3%），所以在寫 PREREG 前，
用營收資料本身比較 3 個事前想得到的帶寬，選「整年持續失真」公司最接近正常年的那個：
  NONE＝不設帶（k＝exp(a)，可小於 1）；ZERO＝k＝max(1, exp(a))（合併營收不小於個體）；P90＝a > 安慰劑年 P90 才校正。
指標（2013-02～2013-12 共 11 個月，k 只用到 2013-01，所以這 11 個月沒有參與估計）：
  持續偏高＝11 個月中 ≥ 9 個月校正後年增率 > +30% 的公司比例；持續偏低＝≥ 9 個月 < −30%；
  參考值＝2012、2014 同樣 11 個月（未校正）的平均。選擇準則：|持續偏高 − 參考| ＋ |持續偏低 − 參考| 最小者。
python3 -I -B step0b.py → step0b.json"""
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
p90 = json.load(open(S.HERE / 'step0.json'))['abnormal_jump']['placebo_pooled_2011_2012']['p90']


def pers(Yy, y):
    yy = Yy.loc[P_(f'{y}-02'):P_(f'{y}-12')]
    ok = yy.notna().sum() == 11
    return dict(n=int(ok.sum()), hi=float(((yy > .3).sum() >= 9)[ok].mean()), lo=float(((yy < -.3).sum() >= 9)[ok].mean()),
                median_pct=float(yy.loc[:, ok].median(axis=1).mean() * 100), share_gt50=float((yy.loc[:, ok] > .5).mean(axis=1).mean()),
                share_lt_m30=float((yy.loc[:, ok] < -.3).mean(axis=1).mean()))


ref = {str(y): pers(Y, y) for y in (2011, 2012, 2014, 2015)}
rhi = (ref['2012']['hi'] + ref['2014']['hi']) / 2; rlo = (ref['2012']['lo'] + ref['2014']['lo']) / 2
a = S.abnormal_jump(Rv, 2013, S.PRIOR_YEARS)
out = dict(reference_years=ref, reference=dict(hi=rhi, lo=rlo), raw_2013=pers(Y.loc[:, a.notna()], 2013), placebo_p90=p90, variants={})
for name, k in (('NONE', np.exp(a)), ('ZERO', np.exp(a.clip(lower=0))), ('P90', pd.Series(np.where(a > p90, np.exp(a), 1.0), index=a.index).where(a.notna()))):
    A = Rv.to_numpy(float).copy(); pre = np.asarray(Rv.index < S.BREAK)
    A[pre] = A[pre] * k.reindex(Rv.columns).to_numpy()[None, :]
    Ra = pd.DataFrame(A, index=Rv.index, columns=Rv.columns)
    v = pers(Ra / Ra.shift(12) - 1, 2013)
    v['distance'] = abs(v['hi'] - rhi) + abs(v['lo'] - rlo)
    v['share_k_gt1'] = float((k.dropna() > 1).mean()); v['share_k_lt1'] = float((k.dropna() < 1).mean())
    out['variants'][name] = v
out['selected'] = min(out['variants'], key=lambda n: out['variants'][n]['distance'])
(S.HERE / 'step0b.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))

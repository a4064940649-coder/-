#!/usr/bin/env python3
"""S83 第 4 步（事後、僅描述，不在 PREREG 內）：①三個溫度計合成讀數（冷／溫／熱）的歷史與現況；②同步關係的穩健性（分小段、去掉 2020、去掉最熱月份）；
③制度時點前後 24 個月的指標比較（只是「同時發生」，不是因果）；④逐年上市成交值。python3 -I -B posthoc.py → results_posthoc.json、thermometer_monthly.csv"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s83lib as E
HERE = Path(__file__).resolve().parent
PRE = json.load(open(HERE / 'PREREG.json', encoding='utf-8')); thr = PRE['thresholds']
I = pd.read_csv(HERE / 'indicators_monthly.csv', index_col=0); O = pd.read_csv(HERE / 'outcomes_monthly.csv', index_col=0)
mon = pd.PeriodIndex(I.index, freq='M'); OUT = {}
TH = ['SPURT3', 'BIGUP21', 'TAIL21']
hi = sum((I[i] > thr[i]['hi']).astype(int) for i in TH); lo = sum((I[i] < thr[i]['lo']).astype(int) for i in TH)
state = np.where(hi >= 2, '熱', np.where(lo >= 2, '冷', '溫'))
Tm = pd.DataFrame({'date': I['date'], **{i: I[i] for i in TH}, 'n_high': hi, 'n_low': lo, 'state': state, 'ex_month_pct': O['sync1'] * 100, 'ex_next_month_pct': O['lead1'] * 100, 'ex_next3_pct': O['lead3'] * 100})
Tm.to_csv(HERE / 'thermometer_monthly.csv')
OUT['thermometer_def'] = dict(indicators=TH, thresholds={i: thr[i] for i in TH}, rule='三項中 ≥2 項高於訓練段上三分位＝熱；≥2 項低於訓練段下三分位＝冷；其餘＝溫（合成規則為事後訂定）')
ERAS = [('2008–2012', '2008-01', '2012-12'), ('2013–2015', '2013-01', '2015-12'), ('2016–2018', '2016-01', '2018-12'), ('2019–2023', '2019-01', '2023-12'), ('2024–2026-10', '2024-01', '2026-10'),
        ('訓練 2008–2015', '2008-01', '2015-12'), ('驗證 2016–2026', '2016-01', '2026-10'), ('全期', '2008-01', '2026-10')]
tab = {}
for nm, a, b in ERAS:
    m = np.asarray((mon >= pd.Period(a, 'M')) & (mon <= pd.Period(b, 'M'))); x = Tm[m]
    tab[nm] = {s: dict(months=int((x.state == s).sum()), share=float((x.state == s).mean()),
                       ex_same_month_ann_pp=float(x.loc[x.state == s, 'ex_month_pct'].mean() * 12) if (x.state == s).any() else None,
                       ex_next_month_ann_pp=float(x.loc[x.state == s, 'ex_next_month_pct'].mean() * 12) if (x.state == s).any() else None,
                       ex_next3_ann_pp=float(x.loc[x.state == s, 'ex_next3_pct'].mean() * 4) if (x.state == s).any() else None) for s in ('冷', '溫', '熱')}
OUT['thermometer_by_era'] = tab
OUT['thermometer_current'] = Tm.tail(6).reset_index().to_dict('records')
OUT['thermometer_last12_counts'] = Tm.tail(12)['state'].value_counts().to_dict()
# 逐年
OUT['thermometer_yearly'] = {int(y): dict(hot=int((g.state == '熱').sum()), warm=int((g.state == '溫').sum()), cold=int((g.state == '冷').sum()), ex_ann_pp=float(g.ex_month_pct.mean() * 12)) for y, g in Tm.groupby(mon.year)}
yy = pd.DataFrame(OUT['thermometer_yearly']).T
OUT['yearly_hot_months_vs_excess_spearman'] = float(yy['hot'].rank().corr(yy['ex_ann_pp'].rank()))

# 穩健性：通過的同步配對
rob = {}
PAIRS = [('SPURT3', 'sync3'), ('BIGUP21', 'sync3'), ('TAIL21', 'sync3'), ('TAIL21', 'sync1'), ('XSD21', 'sync3'), ('WML3', 'sync3')]
def msk(a, b, drop_year=None):
    m = np.asarray((mon >= pd.Period(a, 'M')) & (mon <= pd.Period(b, 'M')))
    if drop_year:
        m &= np.asarray(mon.year != drop_year)
    return m
for i, o in PAIRS:
    r = {}
    for nm, m in (('2008–2015', msk('2008-01', '2015-12')), ('2016–2018', msk('2016-01', '2018-12')), ('2019–2026', msk('2019-01', '2026-10')), ('2016–2026', msk('2016-01', '2026-10')),
                  ('2016–2026 去掉 2020', msk('2016-01', '2026-10', 2020)), ('2016–2026 去掉 2020、2021', msk('2016-01', '2026-10', 2020) & np.asarray(mon.year != 2021))):
        r[nm] = E.spearman(I.loc[m, i].to_numpy(), O.loc[m, o].to_numpy())
    # 控制大盤當月（當期）報酬後：指標與超額各自對 0050 同期報酬做等級殘差
    m = msk('2016-01', '2026-10'); x = I.loc[m, i].rank(); y = O.loc[m, o].rank(); z = (O.loc[m, 'e0050'] if o == 'sync1' else (1 + O['e0050']).rolling(3).apply(np.prod).loc[m] - 1).rank()
    ok = x.notna() & y.notna() & z.notna()
    rx = x[ok] - np.polyval(np.polyfit(z[ok], x[ok], 1), z[ok]); ry = y[ok] - np.polyval(np.polyfit(z[ok], y[ok], 1), z[ok])
    r['2016–2026 控制 0050 同期報酬（偏等級相關）'] = float(np.corrcoef(rx, ry)[0, 1])
    rob[f'{i}|{o}'] = r
OUT['robust_sync'] = rob
# 反例：飆股密度很高但超額不高的年份
yr = pd.read_csv(HERE / 'yearly_means.csv', index_col=0)
OUT['counterexample_years'] = yr[['SPURT3', 'SPURT6', 'TAIL21', 'BIGUP21', 'ex_ann_pp_simple']].round(4).to_dict('index')
# 飆股密度的前瞻版（之後 63 日漲超過 50%）與回顧版的關係：指標在 k 的「未來」＝ k+63 的 SPURT3 → 用月資料 shift(-3)
fwd = I['SPURT3'].shift(-3)
OUT['spurt_persistence'] = dict(spearman_now_vs_next3m_full=E.spearman(I['SPURT3'].to_numpy(), fwd.to_numpy()),
                                spearman_train=E.spearman(I['SPURT3'].to_numpy()[np.asarray(mon <= pd.Period('2015-09', 'M'))], fwd.to_numpy()[np.asarray(mon <= pd.Period('2015-09', 'M'))]),
                                spearman_valid=E.spearman(I['SPURT3'].to_numpy()[np.asarray(mon >= pd.Period('2016-01', 'M'))], fwd.to_numpy()[np.asarray(mon >= pd.Period('2016-01', 'M'))]))
# 制度時點前後 24 個月
ev = {}
for nm, d in (('2017-04 當沖降稅', '2017-04'), ('2020-10 盤中零股', '2020-10')):
    p = pd.Period(d, 'M'); b = np.asarray((mon >= p - 24) & (mon < p)); a = np.asarray((mon >= p) & (mon < p + 24))
    ev[nm] = {i: dict(before24=float(I.loc[b, i].mean()), after24=float(I.loc[a, i].mean())) for i in ['SPURT3', 'SPURT6', 'TAIL21', 'BIGUP21', 'XSD21', 'TURN20_250']}
    ev[nm]['ex_ann_pp'] = dict(before24=float(O.loc[b, 'sync1'].mean() * 1200), after24=float(O.loc[a, 'sync1'].mean() * 1200))
OUT['event_windows(只是同時發生，不是因果)'] = ev
json.dump(OUT, open(HERE / 'results_posthoc.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k not in ('counterexample_years',)}, ensure_ascii=False, indent=1))

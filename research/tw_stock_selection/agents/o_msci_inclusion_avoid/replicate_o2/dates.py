import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
import json, pandas as pd, numpy as np
import rlib as L
rd = pd.read_csv(L.RD, dtype={'review': str})
import itertools
# 訊號日＝生效月（審核月）第一個 >=11 日的交易日；成交日＝下一交易日
out = []
for _, r in rd.iterrows():
    y, m = map(int, r['review'].split('-'))
    ii = L.IDX[(L.IDX.year == y) & (L.IDX.month == m) & (L.IDX.day >= 11)]
    s = ii[0]; e = L.IDX[L.IDX.get_loc(s) + 1]
    a = L.IDX[L.tday_ge(r['ann_tw'])]
    E = L.IDX[L.tday_le(r['eff'])]
    cat = 'signal_close' if a <= s else ('exec_open' if a == e else 'after_exec')
    out.append(dict(review=r['review'], ann=str(a.date()), sig=str(s.date()), ex=str(e.date()), E=str(E.date()), cat=cat,
                    period='select' if E <= pd.Timestamp('2018-12-31') else 'oos',
                    chk=(str(s.date()) == r['signal'] and str(e.date()) == r['exec'] and str(a.date()) == r['ann_trade_day'])))
d = pd.DataFrame(out)
print(d.to_string())
print('all', d.cat.value_counts().to_dict())
print('select', d[d.period == 'select'].cat.value_counts().to_dict(), len(d[d.period=='select']))
may = d[d.review.str.endswith('-05')]
print('may', len(may), may.cat.value_counts().to_dict())
print('match csv', d.chk.all())
json.dump(d.to_dict('records'), open(L.HERE / 'dates_rep.json', 'w'), indent=1)

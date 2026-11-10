# 煙霧測試用假 K 檔（只用自建名單前 6 個月；不是原名單）
import pickle, xlists, xdata, pandas as pd
L = xlists.Lists(); d = L.dates
days = [x for x in L.jdays(0)][:8]
p = L.plan(days, seed=0)
f = lambda p: [(d[s].strftime('%Y-%m-%d'), d[e].strftime('%Y-%m-%d'), [L.codes[c] for c in pk]) for s, e, pk in p]
KD = {'variants': {v: {(0, 0): f(p)} for v in ['orig', 'REAL_opt']}, 'paper': {'C0': f(p), 'ALLPIT_REAL_opt': f(p)}}
pickle.dump(KD, open(xdata.SCR + 'fake_k.pkl', 'wb')); print('ok')

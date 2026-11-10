"""取主策略（o18 C0）訊號日 d 與成交日 e：只讀 import o18lib。"""
import sys, json, time
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_delist_rev_backfill_allpit')
import o18lib as O
t = time.time()
b = O.build(('base', '2006-01-01', None, True, None))
pl = b['plan']
print(type(pl), str(pl[0])[:300], b['t0'], b['t1'], str(b['sd'][:3]))
rows = [(str(d)[:10], str(O.IDX[e])[:10] if isinstance(e, (int,)) else str(e)[:10]) for e, pk, el, d in pl]
print(len(rows), rows[:3], rows[-3:], time.time() - t)
json.dump(dict(pairs=rows, sd=[str(x)[:10] for x in b['sd']]), open('r2b_sigdays.json', 'w'))

# K：o18 紙上名單（C0、ALLPIT_REAL_opt／p90／carry）＋原紙上引擎淨值（密封）
import sys, json, time, pickle
O18 = '/home/user/-/research/tw_stock_selection/agents/o_delist_rev_backfill_allpit'
OUT = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
sys.path.insert(0, O18); import o18lib as O; sys.path.pop(0)
KINDS = {'C0': ('base', None, None, False, None), 'ALLPIT_REAL_opt': ('opt', '2006-01-01', 'UB', True, 'B1'),
         'ALLPIT_REAL_p90': ('p90', '2006-01-01', 'UB', True, 'B1'), 'ALLPIT_REAL_carry': ('carry', '2006-01-01', 'UB', True, 'B1')}
t0 = time.time(); paper = {}; navs = {}
for k, kind in KINDS.items():
    b = O.build(kind)
    paper[k] = [(str(O.K.IDX[p[0]].date() if hasattr(O.K.IDX[p[0]], 'date') else O.K.IDX[p[0]]), None, None) for p in b['plan'][:1]]
    paper[k] = [(str(p[3].date()), str(O.K.IDX[p[0]].date()), [O.COLS[j] for j in p[1]]) for p in b['plan']]
    for cost in ['normal', 'stress']:
        for eng in (['freeze', 'cons_main'] if k == 'C0' else ['cons_main']):
            navs[(k, cost, eng)] = O.sim(b, cost, eng)
    print(k, len(paper[k]), round(time.time() - t0), flush=True)
pickle.dump(paper, open(OUT + 'g6K_lists/k_paper_lists.pkl', 'wb'))
pickle.dump({'navs': navs}, open(OUT + 'g6_sealed/k_paper_navs.pkl', 'wb'))
print('done')

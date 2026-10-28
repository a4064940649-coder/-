import sys, pickle; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
SP = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
R = pickle.load(open(SP + 'rep_o4_batch.pkl', 'rb'))
pl = R['B10'][0][0]
orig = K.x_action
def mk(incl_cash, incl_del, incl_unk):
    def xa(j, ie):
        di = K.dead_info(j, ie)
        if di is None: return None
        lt, dd = di; e = K.IDX[ie]
        recs = [r for r in K.XINFO.get(j, []) if abs((pd.Timestamp(r.last_trade) - K.IDX[lt]).days) <= 40]
        cmp = lambda a, b, inc: (a >= b) if inc else (a > b)
        if recs:
            r = recs[0]; typ = r.type; ddate = pd.Timestamp(r.delist_date)
            if typ == 'cash' and isinstance(r.cash_over_last_close, str) and r.cash_over_last_close:
                pay = pd.Timestamp(r.pay_date) if isinstance(r.pay_date, str) and r.pay_date else ddate + pd.Timedelta(days=14)
                return ('cash', float(r.cash_over_last_close)) if cmp(e, pay, incl_cash) else ('hold', 1.0)
            if typ in ('swap', 'cash'): return ('cash', 1.0) if cmp(e, ddate, incl_del) else ('hold', 1.0)
            if typ == 'failure': return ('zero', 0.0)
        return ('cash', 1.0) if cmp(e, K.IDX[lt] + pd.Timedelta(days=14), incl_unk) else ('hold', 1.0)
    return xa
b, s = K.rates()
for v in [(0,0,0),(1,0,0),(1,1,1),(0,1,1)]:
    K.x_action = mk(*v)
    nav, st = K.run(pl, b, s); o = K.summary(nav, pl); print(v, o['cagr'], o['t'], st)
K.x_action = lambda j, ie: None
nav, st = K.run(pl, b, s); o = K.summary(nav, pl); print('noX', o['cagr'], o['t'], st)

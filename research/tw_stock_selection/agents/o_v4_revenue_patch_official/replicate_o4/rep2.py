import sys, pickle; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
from ab import avg_nav
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
R = pickle.load(open(SP + 'rep_o4_batch.pkl', 'rb'))
N = lambda n: R[n][1]['normal'][0]
PL = R['V4'][0][0]
o = K.summary(N('O3REP'), PL); print('O3REP %.4f%% %.4f' % (o['cagr'] * 100, o['t']))
# 配對差
def wr(nav): return K.win_ret(nav, PL)
mv = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/main_v2/navs_gapfix.parquet')['EXFIN|n10|normal']
mv = mv.reindex(K.IDX).ffill()
def pair(a, b, lab):
    d = (wr(a) - wr(b)) * 100
    out = [lab, f'all {d.mean():+.3f} ({K.tstat(d):.2f}) nz {(d.abs() > 1e-9).sum()}']
    for k in ['SEL', 'VAL', 'S4']:
        x, y = K.SEGS[k]; dd = d[(d.index >= x) & (d.index <= y)]
        out.append(f'{k} {dd.mean():+.3f} ({K.tstat(dd):.2f})')
    print(' | '.join(out))
pair(N('V4'), mv, 'V4-main_v2')
pair(N('V4'), N('O3REP'), 'V4-o3')
pair(N('V4'), N('RDEL'), 'R_FIX effect')
pair(N('V4'), N('SIG'), 'trade-sig')
for n in ['AB', 'B05', 'B10', 'BQ']: pair(N(n), N('V4'), n + '-V4')
# 隨機同分
def rj(sd):
    pl = K.build_plan(tie='rand', seed=sd); nav, _ = K.run(pl, *K.rates()); o = K.summary(nav, pl); return o['cagr'], o['t']
with Pool(4) as P: rr = P.map(rj, range(20))
c = [x[0] for x in rr]; t = [x[1] for x in rr]
print('rand cagr %.2f~%.2f t %.3f~%.3f' % (min(c) * 100, max(c) * 100, min(t), max(t)))
# 損益兩平
def tsl(args):
    name, slip = args
    pls = R[name][0]
    navs = [K.run(pl, *K.rates('normal', slip=slip))[0] for pl in pls]
    nav = navs[0] if len(navs) == 1 else avg_nav(navs)
    return K.summary(nav, pls[0])['t']
def solve(name, target):
    lo, hi = 0.001, 0.006
    for _ in range(14):
        mid = (lo + hi) / 2
        if tsl((name, mid)) > target: lo = mid
        else: hi = mid
    return (lo + hi) / 2
with Pool(4) as P:
    be = P.starmap(solve, [('V4', 2.3), ('V4', 2.0), ('AB', 2.3), ('AB', 2.0)])
print('breakeven V4 %.3f%% %.3f%%  AB %.3f%% %.3f%%' % tuple(x * 100 for x in be))

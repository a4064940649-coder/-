"""R2 面板 B（V7 面板＋V4 R_FIX／Y5_FIX＋官方處置 trade＋排除 KY/DR 與 FIN2）。先斷言重現 plan_V7.csv。"""
import sys
AGV7 = '/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine'
sys.path.insert(0, AGV7)
import v7env
L = v7env.load('v7')
sys.path.pop(0)
import numpy as np, pandas as pd
K, V = L.K, L.V
IDX, COLS, N = K.IDX, K.COLS, K.N
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade')
S = L.V4(YFX, Y5F, DMT)
pl = S.plan(10, 'official')
ref = pd.read_csv(AGV7 + '/plan_V7.csv', dtype=str)
mine = [(d.date().isoformat(), ','.join(COLS[j] for j in p)) for e, p, el, d in pl]
assert len(mine) == len(ref) and all(a == b for a, b in zip(mine, zip(ref.signal, ref.picks))), 'plan_V7 不符'
print('plan_V7 reproduced', len(mine))

# 原因分類：沿用 r2.py 的定義（以面板 B 的 COLS／交易日判 TRUNC）
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:]]
import importlib.util
src = open('/home/user/-/research/tw_stock_selection/agents/o_rev_avail_ds_pit_count/replicate_o17/r2.py', encoding='utf-8').read()
a = src.index('# ---- 營收水準'); b = src.index('GOV = ')
ns = dict(np=np, pd=pd, K=K, V=V, IDX=IDX, COLS=COLS, N=N, SWITCH=V.SWITCH, ARG={}, NAMES=['rev', 'sl', 'snap'], Y5=Y5F)
exec(src[a:b], ns)
cause = ns['cause']
GOV = {'TRUNC', 'IFRS13', 'POST13', 'POST13_R5', 'PRE12', 'HOLE', 'NEVER_DOM'}
START, END = pd.Timestamp('2013-01-01'), pd.Timestamp('2026-09-11')

def run():
    held = set(); recs = []
    for d in S.sig_days:
        cs = S.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        A = np.zeros(N, bool); causes = {}
        if START <= d <= END:
            miss = cs['ok_other'] & ~np.isfinite(cs['y3'])
            for j in np.where(miss)[0]:
                c = cause(COLS[j], d); causes[j] = c
                if c in GOV: A[j] = True
        if A.any():
            elp = el | A; n = elp.sum()
            p52 = pd.Series(np.where(elp, cs['s52'], np.nan)).rank(pct=True).to_numpy()
            rr = pd.Series(np.where(elp & ~A, cs['y3'], np.nan)).rank().to_numpy() / n
            prv = np.where(A, 1.0, rr); sc = 0.5 * p52 + 0.5 * prv; sc[~elp] = np.nan
            csx = dict(t=cs['t'], e=cs['e'], el=elp, sc=sc, y3=np.where(A, 1e300, cs['y3']))
        else:
            csx = cs
        e = cs['e']; gu = K.gap_up_open(e)
        cand = V.V2.order(csx, 'official')[:int(csx['el'].sum())]
        pick = []
        for j in cand:
            if gu[j] and j not in held: continue
            pick.append(j)
            if len(pick) == 10: break
        for r, j in enumerate(pick):
            recs.append(dict(sig=d.date().isoformat(), rank=r + 1, code=COLS[j], put=bool(A[j]), cause=causes.get(j, '')))
        held = set(pick)
    return pd.DataFrame(recs)
R = run(); R.to_csv('rep_r2_B_opt_gov.csv', index=False)
P = R[R.put]
for nm, a, b in (('2013_2026', '2013', '2026-12'), ('2013_2018', '2013', '2018-12-31'), ('2019_2026', '2019', '2026-12')):
    den = R[(R.sig >= a) & (R.sig <= b)].sig.nunique() * 10
    s = P[(P.sig >= a) & (P.sig <= b)]; top = s.code.value_counts()
    print(nm, f'{len(s)}/{den}={len(s)/den:.4%} days={s.sig.nunique()} iss={s.code.nunique()}', top.head(5).to_dict(), s.cause.value_counts().to_dict())

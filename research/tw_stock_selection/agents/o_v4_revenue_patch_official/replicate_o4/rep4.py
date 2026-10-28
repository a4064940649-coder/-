import sys, pickle, math; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
SP = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
R = pickle.load(open(SP + 'rep_o4_batch.pkl', 'rb'))
PL = R['V4'][0][0]
cnt = {}
for p in PL:
    i_d, i_e = p['i_d'], p['i_e']; d, e = K.IDX[i_d], K.IDX[i_e]
    v = K.V60[i_d]; c = K.Cn[i_d]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(c) & (c >= 10)
    bi = np.where(base)[0]; bi = bi[np.argsort(-v[bi], kind='stable')][:300]
    top = np.zeros(K.N, bool); top[bi] = True
    a = K.disposal_mask(d, e); b = K.disposal_mask(d, d)
    for nm, m in [('all', np.ones(K.N, bool)), ('base', base), ('top300', top), ('top300_exfin', top & ~K.FINMASK & ~K.FIN2MASK & ~K.FORMASK)]:
        x = cnt.setdefault(nm, [0, 0, 0])
        x[0] += int((m & a & ~b).sum()); x[1] += int((m & b & ~a).sum()); x[2] += int((m & (a != b)).sum())
print('slots (trade-only, sig-only(rem0), total)', cnt)
# bootstrap 變體
def tst(x): return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
for name in ['V4', 'AB']:
    w = K.win_ret(R[name][1]['normal'][0], PL).values; n = len(w); t0 = tst(w)
    rng = np.random.default_rng(0); B = 20000; blk = 12; nb = math.ceil(n / blk)
    res = {'circ_t': 0, 'circ_mean_le0': 0, 'noncirc_t': 0, 'noncirc_mean_le0': 0}
    z = w - w.mean()
    for _ in range(B):
        st = rng.integers(0, n, nb); ii = ((st[:, None] + np.arange(blk)).ravel()[:n]) % n
        res['circ_t'] += tst(z[ii]) >= t0; res['circ_mean_le0'] += w[ii].mean() <= 0
        st = rng.integers(0, n - blk + 1, nb); ii = (st[:, None] + np.arange(blk)).ravel()[:n]
        res['noncirc_t'] += tst(z[ii]) >= t0; res['noncirc_mean_le0'] += w[ii].mean() <= 0
    print(name, {k: v / B for k, v in res.items()})

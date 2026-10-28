import sys, json; sys.path.insert(0, '.')
from breaks import *
v4 = pd.read_parquet(AG / 'o_v4_revenue_patch_official/navs_v4.parquet')
b50 = win_ret(e0050()); rm = win_ret(v4['AB_V4|normal']); ew = win_ret(v4['AB_V4|EW'])
ser = {'x50_net': rm - b50, 'xEW_net': rm - ew}
try:
    q = pd.read_parquet(OUT / 'q_navs.parquet')
    ser['Q5mQ1_gross'] = win_ret(q['Q5|zero']) - win_ret(q['Q1|zero'])
    ser['Q5mQ1_net'] = win_ret(q['Q5|normal']) - win_ret(q['Q1|normal'])
except Exception as e: print('no Q', e)
n = NW
nullH = null_supF(n, True); nullI = null_supF(n, False)
rng = np.random.default_rng(12345)
null_rc = np.array([rec_cusum_stat(rng.standard_normal(n)) for _ in range(5000)])
res = {}
for nm, x in ser.items():
    x = np.asarray(x, float); assert np.isfinite(x).all(), nm
    F, k = supF(x, True); Fi, ki = supF(x, False)
    oc, ocp = ols_cusum(x)
    rc = rec_cusum_stat(x)
    bp = bp_select(x)
    o = dict(mean_pp=x.mean() * 100, supF=F, supF_p=float((nullH >= F).mean()), supF_date=str(G[k].date()),
             supF_iid=Fi, supF_iid_p=float((nullI >= Fi).mean()), ols_cusum=oc, ols_cusum_p=ocp,
             rec_cusum=rc, rec_cross=bool(rc > 0.948), rec_p=float((null_rc >= rc).mean()),
             BP_BIC=bp['BIC'], BP_LWZ=bp['LWZ'], BP_seq=int(F > np.quantile(nullH, 0.95)), BP_breaks={str(m): [str(G[b].date()) for b in v] for m, v in bp['breaks'].items()},
             chow2019=chow(x, 157), chow2024=chow(x, 213))
    res[nm] = o
    print(nm, json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in o.items() if not isinstance(v, dict)}, ensure_ascii=False))
    print('   chow2019', {k2: round(v, 3) for k2, v in o['chow2019'].items()}, ' chow2024', {k2: round(v, 3) for k2, v in o['chow2024'].items()})
# 檢定力：SEL 區塊重抽（L6），2019-05 起均值下降
x = np.asarray(ser['x50_net'], float); sel = x[:157]; mu = sel.mean(); crit = np.quantile(nullH, 0.95)
pw = {}
for drop in (0.25, 0.5, 1.0):
    rng = np.random.default_rng(4242)
    idx = sb_indices(157, 6, 400, 4242)
    rs = rng.integers(0, 157, (400,))
    hitF = hitC = 0
    for b in range(400):
        # 從 SEL 抽 245 長度的平穩 bootstrap 路徑
        ii = sb_indices(n, 6, 1, 4242 + b)[0] % 157
        y = sel[ii].copy(); y[157:] -= drop * mu
        hitF += supF(y, True)[0] > crit
        hitC += chow(y, 157)['p'] < 0.05
    pw[str(drop)] = dict(supF=hitF / 400, chow=hitC / 400)
    print('power', drop, pw[str(drop)], flush=True)
res['power'] = pw
json.dump(res, open(OUT / 'breaks.json', 'w'), ensure_ascii=False, indent=1, default=float)

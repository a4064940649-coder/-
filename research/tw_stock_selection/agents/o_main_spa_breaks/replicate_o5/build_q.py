"""五分位價差：V4 合格池 el 內依 sc 分五組等權，AB 最壞情境，隔日開盤成交、無漲停遞補。"""
import sys; sys.path.insert(0, '.')
from rlib import *
sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official'))
import v4lib as L
K, V = L.K, L.V
L.check_freeze()
XR = L.setup_px()
YFX, Y5F = L.load_rfix()
S = L.V4(YFX, Y5F, L.disp_mask_fn('trade'), unavail=V.make_unavail('worst'))
t0, t1, sd = S.bounds()
plans = {q: [] for q in range(5)}
for d in S.sig_days:
    cs = S.cross_section(d); el = cs['el']
    if el.sum() < 50: continue
    j = np.where(el)[0]; sc = cs['sc'][j]
    rk = pd.Series(sc).rank(method='first').to_numpy()
    q = np.floor((rk - 1) * 5 / len(j)).astype(int)
    for k in range(5): plans[k].append((cs['e'], j[q == k]))
out = {}
for k in (0, 4):
    for c in ('zero', 'normal'):
        nav, _ = L.sim_plan(plans[k], t0, t1, c, XR)
        out[f'Q{k+1}|{c}'] = nav
        print(k, c, float(nav.iloc[-1]), flush=True)
pd.DataFrame(out).to_parquet(OUT / 'q_navs.parquet')
print('sig days', len(S.sig_days), S.sig_days[0], S.sig_days[-1])

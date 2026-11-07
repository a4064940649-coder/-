import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, rlib as R
def acf(x, K=12):
    x = np.asarray(x, float) - np.mean(x); v = (x*x).sum()
    return [float((x[k:]*x[:-k]).sum()/v) for k in range(1, K+1)]
def vr(r, H=24, trunc=11):
    return 1 + 2*sum((1-k/H)*r[k-1] for k in range(1, trunc+1))
out = {}
for nm, x in [('sel_D5', R.P_['data']['resid_sel_D5']), ('post_protocol_resid', R.P_['data']['resid_post2019'])]:
    a = acf(x, 23)
    out[nm] = dict(n=len(x), acf1=round(a[0],4), acf6=round(a[5],4), VR24_t11=round(vr(a),3), VR24_t23=round(vr(a,24,23),3))
print(json.dumps(out))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),'rep_e1.json'),'w'))

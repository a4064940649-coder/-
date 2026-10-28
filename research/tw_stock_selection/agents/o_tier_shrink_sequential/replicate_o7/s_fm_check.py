import json, numpy as np, pandas as pd
B='/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/'
Z=np.load(B+'data/forward_priors.npz'); th=Z['theta']
REG=json.load(open(B+'prereg_forward.json'))['registration']
fo=json.load(open(B+'replicate_o7/out/fm_0.json'))['lines']
ff=pd.read_csv(B+'data/fake_forward.csv')
spec=dict(REG['lines']); L1=REG['lines_L1_variants']['L1_MAIN_B10_EXEC@100萬|FIX']
for line,g in ff.groupby('line'):
    key=[k for k in fo if k==line or k.split('_')[0]==line]
    if not key: print('no',line); continue
    key=key[0]
    s=L1 if line=='L1' else spec[key]
    p=Z[s['prior_key']]; p=p/p.sum(); sig=s['sigma_pp']; am=s['alpha_min']
    x=(g.ret_line-g.ret_0050).values*100; n=len(x); S=x.sum()
    llr=(S*th-n*th**2/2)/sig**2
    pos=th>0; neg=th<am
    Eup=(np.exp(llr)*p*pos).sum()/(p*pos).sum()
    Edn=(np.exp(llr-(S*am-n*am**2/2)/sig**2)*p*neg).sum()/(p*neg).sum()
    po=np.exp(llr-llr.max())*p; po/=po.sum(); P=po[pos].sum(); P2=P+0.5*po[th==0].sum()
    W=np.prod(1+g.ret_line)/np.prod(1+g.ret_0050)
    o=fo[key]
    print(key,n,'E_up %.6g/%.6g'%(Eup,o['E_up']),'E_dn %.6g/%.6g'%(Edn,o['E_down']),'P %.6g(half0 %.6g)/%.6g'%(P,P2,o['post_P_alpha_gt_0']),'W %.6g/%.6g'%(W,o['wealth_ratio_vs_0050']))

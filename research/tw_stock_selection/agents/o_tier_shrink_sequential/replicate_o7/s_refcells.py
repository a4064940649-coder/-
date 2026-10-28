import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7')
from rlib import *
import pickle
S=setup_g2(); sw=sigma_w(S['X'],S['cl'])
fw=pd.read_parquet(RAW+'o6shrink/focus_windows.parquet'); F=(fw.sub(fw.r0050,axis=0)*100)
post={}
for f in ['AB','B10']:
    gr,p,_=v5_post(S,F[f].values,sw=sw); post[f]=(gr,p)
jAB=jshifts('AB_V6','normal'); jB=jshifts('AB_V6_B10','normal'); jBs=jshifts('AB_V6_B10','stress')
res={}
res['AB']=summ2(*post['AB']); res['B10']=summ2(*post['B10'])
res['AB_J']=summ2(*mix_shift(*post['AB'],jAB)); res['B10_J']=summ2(*mix_shift(*post['B10'],jB))
res['B10_J']['ratio']=res['B10_J']['mean']/(F.B10.mean()+jB.mean())
res['B10_stress_J']=summ2(*mix_shift(*post['B10'],(F.B10_stress-F.B10).mean()+jBs))
for k,v in res.items(): print(k,{a:round(b,4) for a,b in v.items()})
pickle.dump(dict(post=post,jAB=jAB,jB=jB,jBs=jBs,sw=sw,vP=S['vP']),open('/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7/out/post.pkl','wb'))

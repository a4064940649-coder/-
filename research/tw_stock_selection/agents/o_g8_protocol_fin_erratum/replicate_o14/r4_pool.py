import json,glob,numpy as np
fs=sorted(glob.glob('oc_seed*.json')); ms=[json.load(open(f)) for f in fs]
d=json.load(open('/home/user/-/research/tw_stock_selection/agents/o_g8_protocol_fin_erratum/G8_PROTOCOL.json'))['oc']
T={'primary':d['summary_primary'],'sens':d['summary_sensitivity'],'stress':d['summary_stress']}
Pn=40000*len(ms); out={}; zs=[]
for grp in ('primary','sens','stress'):
  for g in ms[0][grp]:
    for k in ms[0][grp][g]:
      if k.endswith('_se') : continue
      a=np.mean([m[grp][g][k] for m in ms]); b=T[grp][g].get(k)
      if b is None: out[f'{grp}.{g}.{k}']=round(a,4); continue
      se=np.sqrt(a*(1-a)/Pn+b*(1-b)/40000); z=(a-b)/se if se>0 else 0
      out[f'{grp}.{g}.{k}']=(round(a,4),b,round(z,2)); zs.append((abs(z),grp,g,k,round(a,4),b))
zs.sort(reverse=True)
print('n seeds',len(ms),'cells',len(zs),'|z|>2:',sum(z[0]>2 for z in zs),'|z|>3:',sum(z[0]>3 for z in zs)); print(zs[:10])
for k in ['primary.D1.false_up','primary.D2.false_up','primary.D3.false_up','primary.D1.down_power','primary.D2.down_power','primary.D3.down_power','primary.D1.false_down','primary.D2.false_down','primary.D3.false_down','primary.D1.combined_fail_a0','primary.D2.combined_fail_a0','primary.D3.combined_fail_a0','primary.D1.combined_up_a0','primary.D3.combined_up_a0','stress.D1.literal_g8_a0']: print(k,out[k])
json.dump(out,open('oc_pooled.json','w'),indent=1,ensure_ascii=False)

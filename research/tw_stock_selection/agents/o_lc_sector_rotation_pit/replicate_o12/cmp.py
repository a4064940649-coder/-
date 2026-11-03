import pandas as pd, pickle, numpy as np, sys
o=pd.read_parquet('../data/oos_windows.parquet')
W=pickle.load(open(f'win_{sys.argv[1]}.pkl','rb'))
out=[]
for j in range(13):
  x=W[j].set_index('d')
  c=pd.DataFrame({'myB':x.B_normal,'thB':o[f'B|normal|J{j}'],'my50':x.e0050,'th50':o[f'0050|J{j}'],'myS':x.normal,'thS':o[f'H52|k3|A15|M|normal|J{j}']}).loc['2014':]
  cc=c.dropna()
  out.append(dict(j=j,nmy=c.myS.notna().sum(),nth=c.thS.notna().sum(),dB=(cc.thB-cc.myB).mean()*100,d50=(cc.th50-cc.my50).mean()*100,dS=(cc.thS-cc.myS).mean()*100,corrA=np.corrcoef(cc.thS-cc.thB,cc.myS-cc.myB)[0,1],madA=((cc.thS-cc.thB)-(cc.myS-cc.myB)).abs().median()*100))
print(pd.DataFrame(out).round(4).to_string())

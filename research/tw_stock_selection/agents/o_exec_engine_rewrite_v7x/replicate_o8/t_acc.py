import sys, json; sys.path.insert(0,'.')
import numpy as np, pandas as pd
import e8, s8, g8_zero as Z, d8
from multiprocessing import Pool
PJ={k:v for k,v in Z.load_plans('j').items() if k[0]=='AB_V7_B10' and k[1]==int(sys.argv[1])}
cell=sys.argv[2].split('|')
def job(k):
    plan=PJ[k]; t1=e8.TPOS[pd.Timestamp(s8.ENDS[f'AB_V7_B10|J{k[1]}'])]
    S=e8.Sim(plan,t1,dict(model=cell[0],cost=cell[1],tier=cell[2],acct=cell[3])); S.run(); A=dict(S.acc); A.pop('qoa'); return A
with Pool(4) as P: R=P.map(job,list(PJ))
T=json.load(open(d8.O7/'results.json'))['RESULTS_EXEC']['table'][f'B10|J{sys.argv[1]}|'+sys.argv[2]]['acc']
for k in R[0]: print(k, round(np.mean([r[k] for r in R]),1), T.get(k))

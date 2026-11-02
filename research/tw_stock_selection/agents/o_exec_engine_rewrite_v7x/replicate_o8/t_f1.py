import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
import e8, s8, g8_zero as Z
PJ=Z.load_plans('j'); k=('AB_V7_B10',3,0); plan=PJ[k]; t1=e8.TPOS[pd.Timestamp(s8.ENDS['AB_V7_B10|J3'])]
for F1 in (False,True):
    S=e8.Sim(plan,t1,dict(model='K5T',cost='normal',tier='30萬',acct='FIX',F1=F1)); r=S.run()
    nav=(1+r).cumprod(); yrs=(nav.index[-1]-nav.index[0]).days/365.25
    print(F1, round((nav.iloc[-1]**(1/yrs)-1)*100,3), {a:round(b) for a,b in S.acc.items() if a!='qoa'})

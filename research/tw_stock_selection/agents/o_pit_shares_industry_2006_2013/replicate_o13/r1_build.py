import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib, numpy as np, pandas as pd
t=time.time()
P=rlib.load_panel(); print('panel',P['close_raw'].shape, time.time()-t)
S=rlib.survivors(); print('surv', S.loc['2010-06-30'].sum())
A=rlib.anchors(); print(A.src.value_counts()); print(rlib.anchors_raw().src.value_counts())
print(A[A.date.between('2006-01-01','2013-12-31')].groupby('src').code.nunique())
E=rlib.events(); print(E.flag.value_counts()); print((E.m!=1).groupby(E.flag).sum())
print(E[(E.code=='2330')&(E.m!=1)].head(8))
F=rlib.full_build('V2'); print(F.loc['2010-06-30'].notna().sum(), time.time()-t)

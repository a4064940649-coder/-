import sys, time
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/s75_continuous_2008_2026')
t0=time.time()
import s75lib as K
import numpy as np, pandas as pd
P,U=K.load_cont()
print('load',time.time()-t0, P.T,P.N,P.t0, sorted(a for a in vars(P) if not a.startswith('_')))
F,RT=K.surge(P,'B2')
print(F.keys(), type(RT['S']), RT['S'].shape, RT['S'].index[:3], list(RT['av'].items())[:3] if hasattr(RT['av'],'items') else type(RT['av']))
sg=K.signal(P,U,RT,0)
t1=time.time(); nav,lg,st=K.sim(P,sg,U,1e6); print('sim',time.time()-t1, nav.iloc[-1], len(lg))
print(K.cslib.ODD_DATE, K.s20lib.COSTS if hasattr(K.s20lib,'COSTS') else '')

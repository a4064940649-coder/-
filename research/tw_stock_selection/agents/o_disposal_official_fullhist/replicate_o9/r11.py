import sys; sys.path.insert(0,'.')
from ev import *
o=pd.read_csv('my_oos_feat.csv.gz',dtype={'code':str},parse_dates=['s'])
def cum(j,a,b): return np.expm1(CS[b+1,j]-CS[a+1,j])-np.expm1(CE[b+1]-CE[a+1])
for nm,f in [('late_to_e',lambda r,j,h:(cum(j,r.si,h),cum(j,h,r.ei))),
             ('N_x',lambda r,j,h:(cum(j,r.si,r.si+(r.xi-r.si)//2),cum(j,r.si+(r.xi-r.si)//2,r.xi))),
             ('arith_late',lambda r,j,h:(cum(j,r.si,h),r.gross-cum(j,r.si,h))),
             ('ceil',lambda r,j,h:(cum(j,r.si,r.si-(-(r.ei-r.si)//2)),cum(j,r.si-(-(r.ei-r.si)//2),r.xi))),
             ('ceil_e',lambda r,j,h:(cum(j,r.si,r.si-(-(r.ei-r.si)//2)),cum(j,r.si-(-(r.ei-r.si)//2),r.ei))),
             ]:
    el=[f(r,cix[r.code],r.si+(r.ei-r.si)//2) for r in o.itertuples()]
    q=o.assign(early=[a for a,b in el],late=[b for a,b in el]); q['lme']=q.late-q.early
    print(nm, summ(q,'lme'), round(q.early.mean()*100,3), round(q.late.mean()*100,3))

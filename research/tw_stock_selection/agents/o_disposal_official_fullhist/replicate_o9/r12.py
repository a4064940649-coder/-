import sys; sys.path.insert(0,'.')
from ev import *
o=pd.read_csv('my_oos_feat.csv.gz',dtype={'code':str},parse_dates=['s'])
def sm(j,a,b): return (R0[a+1:b+1,j]-EW[a+1:b+1]).sum()
def cp(j,a,b): return np.prod(1+R0[a+1:b+1,j]-EW[a+1:b+1])-1
def lr(j,a,b): return (CS[b+1,j]-CS[a+1,j])-(CE[b+1]-CE[a+1])
for fn in [sm,cp,lr]:
  for end in ['x','e']:
    el=[]
    for r in o.itertuples():
        j=cix[r.code]; h=r.si+(r.ei-r.si)//2; b=r.xi if end=='x' else r.ei
        el.append((fn(j,r.si,h),fn(j,h,b)))
    q=o.assign(early=[a for a,b in el],late=[b for a,b in el]); q['lme']=q.late-q.early
    print(fn.__name__,end, summ(q,'lme'), round(q.early.mean()*100,3), round(q.late.mean()*100,3))

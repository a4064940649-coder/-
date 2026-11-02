import sys; sys.path.insert(0,'.')
from ev import *
o=pd.read_csv('my_oos_feat.csv.gz',dtype={'code':str},parse_dates=['s'])
def cum(j,a,b): return np.expm1(CS[b+1,j]-CS[a+1,j])-np.expm1(CE[b+1]-CE[a+1])
for Nadd in [0,1]:
  for off in [-1,0,1]:
    for end in ['x','e','x+1']:
      el=[]
      for r in o.itertuples():
        j=cix[r.code]; N=r.ei-r.si+Nadd; h=r.si+N//2+off
        b={'x':r.xi,'e':r.ei,'x+1':min(r.xi+1,T-1)}[end]
        el.append((cum(j,r.si,h),cum(j,h,b)))
      q=o.assign(early=[a for a,b in el],late=[b for a,b in el]); q['lme']=q.late-q.early
      s=summ(q,'lme'); print(Nadd,off,end,s['mean'],s['t'], round(q.early.mean()*100,3), round(q.late.mean()*100,3))

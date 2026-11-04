# 獨立複製 o14：由凍結協定的 data 欄重算 c、b、c_d
import json, hashlib, numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq
P='/home/user/-/research/tw_stock_selection/agents/o_g8_protocol_fin_erratum/G8_PROTOCOL.json'
raw=open(P,'rb').read(); print('sha256',hashlib.sha256(raw).hexdigest())
d=json.loads(raw); r=np.array(d['data']['resid_post2019']); r24=np.array(d['data']['resid_2024'])
sd=6.141373712658076
print('n',len(r),'mean',r.mean(),'n24',len(r24),'mean24',r24.mean())
def lrv(x,L):
    x=x-x.mean(); n=len(x); g0=(x@x)/n; v=g0
    for k in range(1,L+1): v+=2*(1-k/(L+1))*(x[k:]@x[:-k])/n
    return v
res={}
for dd in (0,1):
    s2=r.var(ddof=dd); res[f's2_ddof{dd}']=s2/sd**2
for L in (3,6): res[f'LRV_L{L}']=lrv(r,L)/sd**2
print(res)
for dd in (0,1):
    c=np.sqrt(max(1,res[f's2_ddof{dd}'],res['LRV_L3'],res['LRV_L6'])); print('c ddof',dd,c)
# b: P(Z12>=b*sqrt2 or Z24>=b)=0.05, corr sqrt(1/2)
rho=np.sqrt(0.5)
mvn=multivariate_normal([0,0],[[1,rho],[rho,1]])
def cross(b): return 1-mvn.cdf([b*np.sqrt(2),b])
b=brentq(lambda b:cross(b)-0.05,1,3); print('b',b,'b*sqrt2',b*np.sqrt(2))
print('cross(1.6787026)=',cross(1.6787025976148637))
# c_d：D1、α=-1、P=400k、looks 12/18/24 的 z 最小值分位數（不含 refit，近似）
c=1.0596378779547428
rng=np.random.default_rng(777)
x=-1+sd*rng.standard_normal((400000,24)); S=x.cumsum(1)
z=np.stack([S[:,n-1]/(c*sd*np.sqrt(n)) for n in (12,18,24)],1)
mz=z.min(1); cd=-np.quantile(mz,0.55); print('c_d approx (no refit)',cd)
from scipy.stats import multivariate_normal as MV
# 解析：三維常態
mu=np.array([-n/(c*sd*np.sqrt(n)) for n in (12,18,24)])
ns=np.array([12,18,24]); C=np.sqrt(np.minimum.outer(ns,ns)/np.maximum.outer(ns,ns))/c**2
# z_n 的變異 = 1/c^2；z 相關 sqrt(min/max)
mv3=MV(-mu, C)
f=lambda t: 1-mv3.cdf(np.full(3,t))  # P(any z<=-t)= P(any -z>=t)
f=lambda t: 1-MV(-mu,C).cdf(np.full(3,t))
cda=brentq(lambda t:f(t)-0.55,0.5,1.5); print('c_d analytic (no refit, no up-absorb)',cda)

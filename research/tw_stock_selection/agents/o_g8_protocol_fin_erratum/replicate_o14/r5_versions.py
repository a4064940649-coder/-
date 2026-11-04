# R-E / R-E+ / R-OBF 粗略重做（D1），檢查「R-E 升級檢定力＝0」與 R-OBF 合格
import json,numpy as np
from scipy.stats import multivariate_normal as MV
from scipy.optimize import brentq
r=json.load(open('/home/user/-/research/tw_stock_selection/agents/o_g8_protocol_fin_erratum/results.json'))
mp=r['PREREG_DESIGN']['mixture_prior']; th=np.array(mp['theta']); w=np.array(mp['w']); w=w/w.sum()
SD=6.141373712658076; C=1.0596378779547428
rng=np.random.default_rng(5)
for a in (0,1,2,3):
    x=a+SD*rng.standard_normal((40000,24)); S=x.cumsum(1)
    for lab,s in (('R-E',SD),('R-E+',C*SD)):
        hit=np.zeros(40000,bool)
        for n in range(3,25,3):
            E=(w[None,:]*np.exp(np.outer(S[:,n-1],th)/s**2-n*th[None,:]**2/(2*s**2))).sum(1); hit|=E>=20
        print(a,lab,'P(E>=20 within 24)',hit.mean())
# R-OBF：looks {6,12,18,24} 的 b
ns=np.array([6,12,18,24]); Cm=np.sqrt(np.minimum.outer(ns,ns)/np.maximum.outer(ns,ns))
b=brentq(lambda b:1-MV(np.zeros(4),Cm).cdf(b*np.sqrt(24/ns))-0.05,1,3); print('R-OBF b',b)

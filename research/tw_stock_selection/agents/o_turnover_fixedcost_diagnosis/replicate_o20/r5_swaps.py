import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from swaps import *
ex,en,t,rb=build('100n')
ex.to_pickle('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/ex100.pkl'); en.to_pickle('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/en100.pkl')
y=ex[ex.signal>='2019-01-01']; s=ex[ex.signal<'2019-01-01']
print('n Y19',len(y),'with entry',y.n_en.notna().sum(),'S0618',len(s),s.n_en.notna().sum())
# compare to researcher swaps table
R=pd.read_csv(D+'swaps_exitlevel_100n_REAL_opt_seed0.csv.gz',dtype={'code':str})
m=ex.merge(R[['j','k','code','r','edge_B','rtc_B','rank','rb','price','sell_rate']],on=['j','k','code'],how='outer',suffixes=('','_R'),indicator=True)
print(m._merge.value_counts())
b=m[m._merge=='both']
for a,c in [('r','r_R'),('edge','edge_B'),('rtc','rtc_B'),('rate','sell_rate'),('price','price_R')]:
    d=(b[a]-b[c]).abs(); print(a,'maxdiff',d.max(),'n>1e-6',(d>1e-6).sum())
print('rb mismatch',(b.rb!=b.rb_R).sum()); print(b[b.rb!=b.rb_R][['j','k','code','rb','rb_R','rank','rank_R','why']].head())

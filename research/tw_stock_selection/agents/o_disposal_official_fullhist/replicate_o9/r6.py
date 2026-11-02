import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from ev import *
d=load_events()
w=d[(d.start>='2006-03-25')&(d.start<='2021-10-05')]
print('not in panel:',(~w.code.isin(cix)).sum(), w[~w.code.isin(cix)][['code','name','start','in_panel']].to_string()[:3000])
print('invalid e<=s:',((w.ei<=w.si)&w.code.isin(cix)).sum())
print(w[(w.ei<=w.si)&w.code.isin(cix)][['code','announce','start','end','si','ei','adopt','n_tdays']].to_string())

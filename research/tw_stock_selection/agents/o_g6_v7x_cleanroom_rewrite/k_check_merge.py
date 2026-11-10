# K：核對 orig 名單＝plans_j7 AB_V7_B10（seeds 0～19），並把紙上名單併入名單 pickle
import pickle, pandas as pd
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/g6K_lists/'
KD = pickle.load(open(SCR + 'k_v7x_plans.pkl', 'rb')); PP = pickle.load(open(SCR + 'k_paper_lists.pkl', 'rb'))
ref = pd.read_csv(open('/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine/data/plans_j7.csv'), dtype={'picks': str})
ref = ref[(ref.strat == 'AB_V7_B10') & (ref.seed < 20)]
bad = 0; n = 0
for (j, s), g in ref.groupby(['j', 'seed']):
    mine = KD['variants']['orig'][(j, s)]
    rr = list(zip(g.sort_values('signal').signal, g.sort_values('signal').picks))
    n += 1
    if [(a, ','.join(c)) for a, b, c in mine] != [(a, b) for a, b in rr]: bad += 1
print('orig plans vs plans_j7: plans', n, 'mismatched', bad)
KD['paper'] = PP
pickle.dump(KD, open(SCR + 'k_all_lists.pkl', 'wb'))

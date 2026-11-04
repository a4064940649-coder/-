import json,sys,numpy as np
d=json.load(open('/home/user/-/research/tw_stock_selection/agents/o_g8_protocol_fin_erratum/G8_PROTOCOL.json'))['oc']
m=json.load(open(sys.argv[1])); P=40000
keys=['false_up','down_power','false_down','up_p05','up_p1','up_p2','literal_g8_a0','literal_g8_p05','literal_g8_m1','combined_up_p1','combined_fail_a0','down_a0_48','down_m1_48']
worst=[]
for grp,theirs in (('primary',d['summary_primary']),('sens',d['summary_sensitivity']),('stress',d['summary_stress'])):
    for g,v in m[grp].items():
        t=theirs[g]; row=[]
        for k in keys:
            a,b=v[k],t[k]; se=np.sqrt((a*(1-a)+b*(1-b))/P) or 1e-9; z=(a-b)/se; row.append(f'{k}:{a:.4f}/{b:.4f}({z:+.1f})'); worst.append((abs(z),grp,g,k,a,b))
        print(grp,g,'\n  '+'\n  '.join(row))
worst.sort(reverse=True); print('worst',worst[:8])

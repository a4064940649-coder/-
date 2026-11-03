import sys,json; d=json.load(open(sys.argv[1]))
keys=['n','ann','ann0050','annB','v0050_mean_pp','v0050_t','v0050_nwt','v0050_IR','v0050_TE','vB_mean_pp','vB_t','vB_IR','vB_TE','zvB_mean_pp','zvB_t','sv0050_mean_pp','sv0050_t','svB_mean_pp','Bv0050_mean_pp','Bv0050_t','turn','Bturn','cost_pp']
for p in ['VAL','FORMAL','SEL','S1418','S19']:
  print(p,d[p]['nj'],{k:d[p]['med'][k] for k in keys}, d[p]['p10'], d[p].get('drop3_t_med'), d[p].get('drop3_years'))
print(d['B_v0050_2024_pp_med'])

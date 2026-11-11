import json
S=dict(
 engine_from_seedavg_navs=dict(note='原引擎淨值未保存；diff=複製員自算 rep 中位數 − 研究員報的 orig_med',
  N1000=dict(V0=-3.076,V1=-3.075,V2=-3.042,V3=-2.971,orig=-2.952,y19_V0=-10.116,y19_V3=-9.844,med_j=8,t_V3_minus_orig=-0.001),
  S100=dict(V0=-5.988,V1=-6.000,V2=-6.011,V3=-5.877,orig=-5.877,y19_V0=-14.546,med_j=6),
  O30=dict(V0=2.356,V1=2.370,V2=2.370,V3=2.515,orig=2.508,y19_V0=-4.441,y19_V3=-4.103),
  t_N1000_V0=-0.321,t_S100_V0=-0.916,t_O30_V0=0.697,
  y2026_flat_vs_full=dict(N1000=1.063,S100=1.022,O30=1.087)),
 paper=dict(M0=[22.219,1.639,23.887,25.599],M1=[22.057,1.589,23.562,25.384],M2=dict(opt=[22.205,1.600],p90=[23.712,1.866],carry=[25.540,2.186],cal=22.10,exc_cal=7.16,seg=[8.72,0.62,11.87],t_x_best3=0.704,stress=[19.268,1.075]),
  breakeven=dict(M0=0.4194,M1=0.4102,M2=0.4186),M1_M2_opt_slot_diff=6,windows=245),
 listA=dict(C0_match='928/930',C0_mean_jaccard=0.99609,C0_identical_to_A=True,U_strict=dict(opt=181,p90=118,carry=18,jacc=0.8846,extra={'3697':12,'3638':5,'6286':1,'4144':1,'2311':1},missing={'3561':1}),U_posthoc=dict(opt=162,p90=105,carry=18,identical_to_official=True)),
 events=dict(disposal_rows=2218,announce_ge_20261006=15),
 cases=dict(c3007=dict(official_term='2007-04-13',last_trade='2007-04-12',close=107.5,value_761=81808),c3057=dict(susp='2013-09-02~09-30 NaN',open_1022=33.35,proceeds_6000=200100),c6414=dict(close_0323=475,val_907=430825)),
 not_verifiable=['N100/N300 兩格（npz 無淨值）','原引擎逐 j 淨值（未保存）','vrank 10.8%/306,097 格、抽樣 115/0、V60 6,389 NaN（兩引擎輸入陣列已刪）','逐筆成交細節（零股上限 74 股、整股上限 4 張、3981/4013 股）'])
json.dump(S,open('r9_summary.json','w'),ensure_ascii=False,indent=1)

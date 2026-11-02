import sys, pickle, json; sys.path.insert(0,'.')
import d8
T=json.load(open(d8.O7/'results.json'))['RESULTS_EXEC']['table']
d=pickle.load(open(sys.argv[1],'rb'))['stats']
for k,v in d.items():
    j,m,c,ti,a=k.split('|'); r=T[f'B10|{j}|{m}|{c}|{ti}|{a}']
    print(k, f"{v['CAGR']:.2f} vs {r['CAGR']:.2f}  d={v['CAGR']-r['CAGR']:+.3f}  SEL d={v['CAGR_SEL']-r['CAGR_SEL']:+.3f}")

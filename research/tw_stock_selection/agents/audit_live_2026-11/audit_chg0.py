#!/usr/bin/env python3
"""資料品質補充：官方日 K「漲跌價差」＝0 但收盤有變動的（股票×日）筆數（cslib 會把當日含息報酬算成 0）。輸出 audit_chg0.json。python3 -I -B audit_chg0.py"""
import subprocess,io,pandas as pd,numpy as np,json,sys
R='/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data'
ls=subprocess.run(['git','-C',R,'ls-tree','HEAD','data/universe/daily/'],capture_output=True,text=True).stdout.split('\n')
items=sorted((l.split('\t')[1].rsplit('/',1)[1][:10],l.split()[2]) for l in ls if l)
p=subprocess.Popen(['git','-C',R,'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
prev={}; rows=[]; bad=[]
for d,sha in items:
    p.stdin.write((sha+'\n').encode());p.stdin.flush();h=p.stdout.readline().split();b=p.stdout.read(int(h[2]));p.stdout.read(1)
    df=pd.read_csv(io.BytesIO(b),usecols=['stock_id','market','close','change'],dtype={'stock_id':str})
    df=df[df.stock_id.str.fullmatch(r'[1-9]\d{3}|0050')&df.market.isin(['twse','tpex'])].drop_duplicates('stock_id',keep='last')
    df=df[df.close>0]
    pc=df.stock_id.map(prev)
    ok=pc.notna()
    ref=df.close-df.change
    z=ok&(df.change.fillna(0)==0)&((df.close-pc).abs()>1e-9)
    mism=ok&df.change.notna()&((ref-pc).abs()/pc>0.003)
    if d>='2026-08-14': [bad.append((d,c)) for c in df.stock_id[z]]
    rows.append((d,int(ok.sum()),int(z.sum()),int(mism.sum()),int(df.change.isna().sum())))
    prev.update(dict(zip(df.stock_id,df.close)))
r=pd.DataFrame(rows,columns=['date','n','chg0_moved','ref_ne_prev','chg_nan'])
r['frac0']=r.chg0_moved/r.n.clip(lower=1)
r["m"] = r.date.str[5:7]
out = dict(total_cells=int(r.chg0_moved.sum()), by_month=r.groupby("m").chg0_moved.sum().astype(int).to_dict(), max_day_frac=float(r.frac0.max()),
           cells_2026_08_14_to_11_11=int(r[(r.date >= "2026-08-14") & (r.date <= "2026-11-11")].chg0_moved.sum()), recent=bad)
from pathlib import Path
json.dump(out, open(Path(__file__).resolve().parent / "audit_chg0.json", "w"), ensure_ascii=False, indent=0)
print(out["total_cells"], out["by_month"])

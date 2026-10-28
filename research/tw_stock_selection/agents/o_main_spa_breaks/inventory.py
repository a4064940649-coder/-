# 盤點 agents/ 下所有 parquet（淨值候選）
import os, sys, json, glob
import pandas as pd
ROOT='/home/user/-/research/tw_stock_selection/agents'
rows=[]
for p in sorted(glob.glob(ROOT+'/**/*.parquet', recursive=True)):
    rel=os.path.relpath(p,ROOT)
    if rel.startswith('o_main_spa_breaks'): continue
    try:
        d=pd.read_parquet(p)
    except Exception as e:
        rows.append(dict(path=rel,err=str(e)[:80])); continue
    idx=d.index
    info=dict(path=rel, shape=list(d.shape), idx_type=str(type(idx).__name__), idx_dtype=str(idx.dtype))
    try:
        info['first']=str(idx[0])[:10]; info['last']=str(idx[-1])[:10]
    except Exception: pass
    info['cols']=[str(c) for c in d.columns[:12]]
    info['ncols']=d.shape[1]
    rows.append(info)
json.dump(rows, open(ROOT+'/o_main_spa_breaks/data/inventory_parquet.json','w'), ensure_ascii=False, indent=0)
for r in rows:
    print(r.get('path'), r.get('shape'), r.get('first'), r.get('last'), r.get('cols','')[:5] if 'cols' in r else r.get('err'))

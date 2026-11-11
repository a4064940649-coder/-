# -*- coding: utf-8 -*-
"""A 事後修正版（揭盲後，與嚴格版並列；嚴格版為主欄）：FOREIGN 名稱來源加入 delist_official 名稱（R07 規格缺口）。
python3 -I -B a_build_u_posthoc.py"""
import sys, os, json, runpy
import pandas as pd, numpy as np
sys.argv = [sys.argv[0], 'v1']; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_build as B
dl = pd.read_csv('/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/delist_official.csv', dtype=str)
for c, n in zip(dl.code, dl.name): B.names.setdefault(c, set()).add(str(n))
B.FOR[:] = np.array([B.is_foreign(c) for c in B.cols])
src = open(os.path.join(B.HERE, 'a_build_u.py')).read().replace("import a_build as B   # 本研究員自寫", "pass").replace("out/a_lists_2006_2013.csv.gz", "out/a_lists_2006_2013_posthoc.csv.gz").replace("out/a_U_2006_2013.json", "out/a_U_2006_2013_posthoc.json")
g = dict(B=B, __name__='__main__', __file__=os.path.join(B.HERE, 'a_build_u.py')); exec(compile(src, 'a_build_u_posthoc', 'exec'), g)

#!/usr/bin/env python3
"""凍結後修正（只修核對程式，不重算名單）：v7x_plans.py 的 orig 核對把 plans_j7.csv 的 40 個 seeds 全拿來比（本題只產生 seeds 0～19），
使逐筆比較必然不等。改成只比 seeds 0～19；名單檔（scratch/plans_v7x.csv，sha 已記在 RESULTS_V7X_PLANS）不動。記入 POSTFREEZE_FIXES.json。"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
X = R['RESULTS_V7X_PLANS']; f = Path(X['file'])
assert hashlib.sha256(f.read_bytes()).hexdigest() == X['sha256']
D = pd.read_csv(f, dtype={'picks': str})
ref = pd.read_csv(AG / 'o_v7_listing_exec_engine/data/plans_j7.csv', dtype={'picks': str})
ref = ref[(ref.strat == 'AB_V7_B10') & (ref.seed < 20)].sort_values(['j', 'seed', 'signal'])
mine = D[D.variant == 'orig'].sort_values(['j', 'seed', 'signal'])
same = list(zip(mine.j, mine.seed, mine.signal, mine.picks.astype(str), mine.exec_idx)) == list(zip(ref.j, ref.seed, ref.signal, ref.picks.astype(str), ref.exec_idx))
X['orig_reproduces_plans_j7_original_check'] = X['orig_reproduces_plans_j7']
X['orig_reproduces_plans_j7'] = bool(same); X['n_orig_rows'] = int(len(mine)); X['n_ref_rows_seed_lt20'] = int(len(ref))
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
fx = HERE / 'POSTFREEZE_FIXES.json'
L = json.loads(fx.read_text(encoding='utf-8')) if fx.exists() else []
L.append(dict(utc=pd.Timestamp.now('UTC').isoformat(), file='v7x_plans.py（核對段）', bug='plans_j7.csv 有 40 seeds，本題只產生 20 seeds；原核對未限制 seed < 20，必然 False',
              fix='fix_v7x_plans_check.py 以已存名單檔（sha 不變）重新核對 seeds 0～19（含 exec_idx）', result=bool(same), recomputed_lists=False, saw_returns=False))
fx.write_text(json.dumps(L, ensure_ascii=False, indent=1), encoding='utf-8')
print(same, len(mine), len(ref))

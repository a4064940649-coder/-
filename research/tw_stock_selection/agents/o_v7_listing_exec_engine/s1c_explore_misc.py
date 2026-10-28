#!/usr/bin/env python3
"""探索補充（只看資料）：(c) 2024-01 前視條件影響的代碼、颱風日事件清單、迭代選版理由 → explore.json／iterations.json。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7common as C
P4 = pd.read_parquet(C.PANEL4 / 'close_raw.parquet')
pre = P4.loc['2023-12-21':'2023-12-29'].notna().any(); post = P4.loc['2024-02-15':'2024-02-22'].notna().any()
LM = pd.read_parquet(C.DATA / 'listed_twse.parquet')
gap = LM.loc['2024-01-02':'2024-02-14']
look = sorted(pre[pre & ~post].index)
ex = C.jload(C.HERE / 'explore.json')
ex['lookahead_2024_01'] = {'codes_pre_but_not_post(V6 排除)': look,
                           'listed_in_gap_per_master': {c: int(gap[c].sum()) if c in gap else None for c in look},
                           'codes_post_but_not_pre(V6 排除)': sorted(post[post & ~pre].index),
                           'note': 'V6 只收「2023-12-21～29 與 2024-02-15～22 都有值」者；V7 改用逐日上市矩陣（只用上市身分，不看 2024-02-15 之後是否還有價格）。'}
meta = C.jload(C.DATA / 'panel_v7_meta.json')
ty = [x for x in meta['meta']['stats']['typhoon_events_list'] if x[0] in [str(d.date()) for d in C.TYPHOON]]
ex['typhoon_events_in_window'] = {'n': len(ty), 'list': ty}
C.jdump(ex, C.HERE / 'explore.json')
it = C.jload(C.HERE / 'iterations.json')
it['selection'] = {'chosen': 'L1', 'n_versions_tried': len(it['data_versions']),
                   'rule': '先排除不符題目修補內容的版本（L3／V6REPRO＝舊上市過濾，L4＝不修颱風），在剩下的 L1／L2／L5 中選「全市場上市母體保守覆蓋率」最高、颱風事件參考價相符 100%、除權息前收相符 100% 者；L2（排除創新板）與 L5（只用強證據）列為敏感度。不看任何策略報酬。',
                   'coverage_conservative': {v: r['metrics']['pop_coverage_conservative']['official_share'] for v, r in it['data_versions'].items()}}
C.jdump(it, C.HERE / 'iterations.json')
print(look, ex['lookahead_2024_01']['listed_in_gap_per_master'], len(ty))

import json; from pathlib import Path; O = Path(__file__).resolve().parent / 'out'
S = {k: json.load(open(O / f'{k}.json')) for k in ('stats', 'stats2', 'gate', 'jstats', 'ideal', 'ideal_j', 'dispsell', 'ab_b')}
ab = S['ab_b']['t']; import numpy as np
seed = {k: dict(median=float(np.median([ab[f'{k}|s{s}']['normal'] for s in range(40)])), min=min(ab[f'{k}|s{s}']['normal'] for s in range(40)),
                max=max(ab[f'{k}|s{s}']['normal'] for s in range(40)), lt2=sum(ab[f'{k}|s{s}']['normal'] < 2 for s in range(40)),
                lt23=sum(ab[f'{k}|s{s}']['normal'] < 2.3 for s in range(40))) for k in ('V7', 'V7_B10')}
spa = [l for l in open(O / 'ra_spa.log') if l.startswith('[')]
out = dict(seed_t=seed, spa_log=spa, gate={k: v for k, v in S['gate'].items() if k != 'holes_in_top300'}, holes={k: len(v) for k, v in S['gate']['holes_in_top300'].items()},
           stats=S['stats'], stats2=S['stats2'], jstats={k: {kk: vv for kk, vv in v.items()} for k, v in S['jstats'].items()},
           ideal=S['ideal'], ideal_j=S['ideal_j'], dispsell=S['dispsell'],
           not_replicated=['可執行表（ecore7：整股＋零股、OA 參與上限、K5T／TICK、FIX／GROW、級距）全部格子', 'V7 面板自原始來源重建（用研究員凍結面板＋sha256 核對＋閘門性質檢查）'])
json.dump(out, open(Path(__file__).resolve().parent / 'rep_summary.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('ok')

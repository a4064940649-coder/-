#!/usr/bin/env python3
"""把 explore／iterations／oos 的關鍵數字與判定合併進 results.json 的 SUMMARY（不重算任何報酬）。"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG_sha256' in res
ex = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
oo = json.loads((HERE / 'oos.json').read_text(encoding='utf-8'))
S = oo['selected']
def blk(k, p): x = oo['stats'][k][p]; return dict(CAGR=x['CAGR'], maxDD=x['maxDD'], t0050=x['vs0050']['t'], ex0050_pp=x['vs0050']['mean_pp'],
                                                  paired=(dict(mean_pp=x['paired']['mean_pp'], t=x['paired']['t'], n=x['paired']['n']) if 'paired' in x else None))
res['SUMMARY'] = {
 'verdict': '不可用', 'mechanism': '不支持',
 'selected_version': S, 'selected_spec': oo['spec'],
 'n_versions': it['n_versions_total'], 'n_iterations': len(it['iterations']),
 'explore_key': {'turnover_decomp_normal': ex['turnover_decomp_normal'], 'turnover_noreweight_normal': ex['turnover_noreweight_normal'],
                 'P2a': ex['bucket_vs_new']['P2a_rank11-30_h>=0.90'], 'P2b': ex['bucket_vs_new']['P2b_all_old_h<0.85'],
                 'rank11-30_h>=0.95': ex['bucket_vs_new']['rank11-30_h>=0.95'], 'cont_vs_new': ex['bucket_vs_new']['cont_vs_new'],
                 'by_holding_age': ex['by_holding_age'], 'capacity_10m': ex['capacity_10m']},
 'selection_table': it['selection']['table'], 'selection_chain': it['selection']['chain'],
 'thresholds': oo['thresholds'][S],
 'oos': {f'{k}|{p}': blk(k, p) for k in ('BASE|normal', f'{S}|normal', 'BASE|stress', f'{S}|stress', 'BASE|gross', f'{S}|gross') for p in ('SEL', 'VAL', 'HOLD', 'POST', 'ALL')},
 'turnover': oo['turnover'], 'breakeven_ALL': oo['breakeven_ALL'], 'endpoint': oo['endpoint_ALL_vs0050'], 'bonferroni': oo['bonferroni'],
 'AB_ALL': {nm: dict(CAGR=d['ALL']['CAGR'], t0050=d['ALL']['vs0050']['t'], maxDD=d['ALL']['maxDD']) for nm, d in oo['AB'].items()},
 'AB_HOLD': {nm: dict(CAGR=d['HOLD']['CAGR'], t0050=d['HOLD']['vs0050']['t']) for nm, d in oo['AB'].items()},
}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print('ok')

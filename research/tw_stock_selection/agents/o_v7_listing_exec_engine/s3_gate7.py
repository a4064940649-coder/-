#!/usr/bin/env python3
"""步驟 3：閘門（只看資料覆蓋、一致率、颱風日事件；不看報酬）→ results.json 的 GATE；通過才可凍結。
(1) 以全市場上市身分為母體（gatepop.py，2020-01 起每個訊號日的外部官方 V60 前 300 ＋ 保守代入），池內官方 OHLC 覆蓋 ≥ 99.5%；
(2) 2024-01-02～02-14 覆蓋 100%；(3) 颱風停市日除權息事件：順延後參考價相符率 100%；
另報：官方對官方收盤一致率、除權息前收相符率、煙霧測試（2006～2023 與 V6 逐位相同）、16 檔上市空洞補回後進股池／進名單次數（預測 1）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7common as C, gatepop as GP
POP = pd.read_parquet(C.DATA / 'gate_population.parquet')
pop, cv = GP.coverage(POP, C.DATA, conservative=True); _, cvs = GP.coverage(POP, C.DATA, conservative=False)
g24 = POP[(POP.date >= C.GAP_A) & (POP.date <= C.GAP_B)]
_, c24 = GP.coverage(g24, C.DATA, conservative=True)
it = C.jload(C.HERE / 'iterations.json'); meta = C.jload(C.DATA / 'panel_v7_meta.json')
mt = it['data_versions'][meta['version']]['metrics']
smoke = C.jload(C.HERE / 'smoke.json')
gate = {'population': GP.__doc__.strip(), 'coverage_conservative': cv, 'coverage_strict': cvs, 'coverage_2024_01_02_to_02_14': c24}
bysub = {}
for lab, a, b in (('2020-01~2023-12', '2020-01-01', '2023-12-31'), ('2024-01-02~2025-09-23(窗)', '2024-01-02', '2025-09-23'), ('2025-09-24~2026-10', '2025-09-24', '2026-10-31')):
    x = pop[(pop.date >= a) & (pop.date <= b)]
    bysub[lab] = {'n': int(len(x)), 'official_share': float(x.official.mean()) if len(x) else None, 'n_not_official': int((~x.official).sum())}
gate['coverage_by_period'] = bysub
gate['not_official_list'] = pop[~pop.official].groupby('code').agg(n=('date', 'size'), first=('date', 'min'), last=('date', 'max'), proxy=('proxy', 'max')).reset_index().astype(str).to_dict('records')
gate['proxy_imputed'] = pd.read_csv(C.DATA / 'gate_population_proxy.csv', dtype=str).to_dict('records')
gate['typhoon'] = mt['typhoon_check']
gate['official_vs_official_close_agree'] = mt['chosen_close_agree_all']; gate['exdiv_prevclose_match'] = mt['exdiv_prevclose_match_rate']
gate['smoke_2006_2023_identical_to_V6'] = smoke.get('all_identical')
gate['checks'] = {'pool_official_ohlc>=99.5%(全市場上市母體、保守)': cv['official_share'] >= 0.995,
                  '2024-01-02~02-14 100%': c24['official_share'] >= 1 - 1e-12,
                  'typhoon_exdiv_ref_match=100%': mt['typhoon_check']['ref_applied_rate'] >= 1 - 1e-12 and mt['typhoon_check']['prevclose_match_rate'] >= 1 - 1e-12,
                  'smoke_identical': bool(smoke.get('all_identical'))}
gate['PASS'] = bool(all(gate['checks'].values()))
# 16 檔空洞（統計審查列名者）與 V6 股池差異
pv7 = C.jload(C.DATA / 'pools_v7.json'); pv6 = C.jload(C.V6DATA / 'pools_v6.json')
holes = ['2254', '2258', '2432', '6534', '6645', '6757', '6854', '6869', '6873', '6902', '6890', '6923', '6957', '6958', '6919', '2646']
hstat = {}
sig = sorted(k for k in pv7 if not k.startswith('_'))
for d in sig:
    a = set(pv7[d]['top300']); b = set(pv6.get(d, {}).get('top300', []))
    hstat[d] = {'n_v7_only': len(a - b), 'n_v6_only': len(b - a), 'v7_only': sorted(a - b), 'v6_only': sorted(b - a),
                'picks_v7_only': sorted(set(pv7['_picks'].get(d, [])) - set(pv6['_picks'].get(d, []))),
                'picks_v6_only': sorted(set(pv6['_picks'].get(d, [])) - set(pv7['_picks'].get(d, [])))}
gate['pool_diff_v7_vs_v6'] = hstat
win = [d for d in sig if '2024-07-01' <= d <= '2025-03-31']
gate['prediction1'] = {'n_v7_only_in_top300_by_signal_2024_07_to_2025_03': {d: hstat[d]['n_v7_only'] for d in win},
                       'range': [min(hstat[d]['n_v7_only'] for d in win), max(hstat[d]['n_v7_only'] for d in win)],
                       'n_pick_changes_total_2024_2025': int(sum(len(hstat[d]['picks_v7_only']) for d in sig if d <= '2025-12-31')),
                       'n_hole_codes_picked_total': int(sum(len(set(pv7['_picks'].get(d, [])) & set(hstat[d]['v7_only'])) for d in sig))}
res = C.load_res(); res['GATE'] = gate; C.jdump(res, C.RES_F)
print(json.dumps(C.conv({k: gate[k] for k in ('coverage_conservative', 'coverage_strict', 'coverage_2024_01_02_to_02_14', 'coverage_by_period', 'checks', 'PASS', 'prediction1')}), ensure_ascii=False, indent=0))

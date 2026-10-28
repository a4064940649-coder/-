#!/usr/bin/env python3
"""可執行引擎 V7 的成本模型參數（PREREG 前寫定；只取既有校準結果，不看策略報酬）→ data/model_spec7.json。
  OA（開盤集合競價成交值）＝share(V60 名次十分位) × 當日成交值；share 取 o_open_impact_calib 的 OA_b 驗證期十分位中位數（2.2～3.8%，
  MIS 真值中位數 3.4%；dec＝ceil(前一日 V60 名次 / 30)，名次 > 300 或缺值用第 10 十分位）。取代 o5 的 OA_hat（中位數占全日 9.0～9.1%）。
  上界模型 K5T：每邊 max(0.1%＋0.5·σ60·√(Q/OA), 一個跳動單位/價格)（o5 κ＝0.5 式，加跳動單位下限使上界 ≥ 下界）；
  原式 K5：0.1%＋0.5·σ60·√(Q/OA)（不加下限，只當敏感度）；
  下界模型 TICK：每邊一個跳動單位/價格（κ 未辨識時可確定的最小開盤摩擦）。
  兩者都有 20% 參與上限（OA 的 20%），超過的張數順延次一交易日開盤。零股：9:10 漂移、零股價差、零股量上限（同 o6）。"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
RI = HERE / 'raw' / 'exec_inputs'
mc = json.loads((RI / 'model_calib.json').read_text(encoding='utf-8'))
it = json.loads((RI / 'o_open_iterations.json').read_text(encoding='utf-8'))
ex = json.loads((RI / 'o_open_explore.json').read_text(encoding='utf-8'))
share = it['rounds'][0]['oa_check']['OA_b_share_by_dec_VAL']
assert all(it['rounds'][k]['oa_check']['OA_b_share_by_dec_VAL'] == share for k in range(len(it['rounds'])))
spec = {'version': 'v7_exec_1',
        'oa_share_by_dec': {int(k): float(v) for k, v in share.items()}, 'oa_share_default_dec': 10,
        'mis_truth_median': ex['mis_truth']['oa_share']['median'], 'mis_n_stockdays': ex['mis_truth']['n_stockdays'],
        'oa_hat_share_by_dec_VAL_o5': it['rounds'][0]['oa_check']['OA_hat_share_by_dec_VAL'],
        'cap': 0.20, 'base': 0.001, 'kappa': 0.5,
        'drift': mc['drift'], 'odd': mc['odd'], 'zcap': mc['zcap'], 'zpart': 1.0,
        'frictions': {'lim': True, 'odd0': True, 'cash': 0.02, 'feemix': True},
        'stress': {'extra_per_side': 0.004, 'fee': 'B', 'odd_stress': True, 'feemix': False},
        'disposal_sell': '處置期間（V4 凍結官方處置名單 start ≤ t ≤ end）的賣出：預收券、分盤撮合；以當日收盤減一個跳動單位成交（首盤價資料不足時的事前規則），無開盤衝擊、無參與上限',
        'merger_delay': '現金對價併購下市（O3 事件 type=cash）：下市日起現金對價以 0050 含息報酬計值，撥款日＝下市日＋20 個交易日（找不到撥款日一律用此保守值），撥款後第一個換股日才再投入',
        'source': {'model_calib': 'o_open_impact_calib/data/model_calib.json', 'oa_share': 'o_open_impact_calib/iterations.json rounds[*].oa_check.OA_b_share_by_dec_VAL',
                   'mis': 'o_open_impact_calib/explore.json mis_truth'}}
(HERE / 'data').mkdir(exist_ok=True)
(HERE / 'data' / 'model_spec7.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
print(spec['oa_share_by_dec'], spec['mis_truth_median'])

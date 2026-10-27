#!/usr/bin/env python3
"""事後診斷（看完 run.py 結果後才加，不影響判定）：事件平均報酬「每筆等權」與「依投入資金加權」的差異。
讀 events_*.csv.gz（run.py 產出），寫入 results.json → posthoc_diagnostics。執行：python3 -I diag_capital.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
from btlib import tstat  # noqa: E402
FEE = 0.001425 * 0.6; B, S = FEE + 0.001, FEE + 0.003 + 0.001   # 與 exlib.cost_rates(0.001) 相同
W = {'select': ('2005-01-01', '2019-04-30'), 'valid': ('2019-05-01', '2026-10-31'), 'y2024_2025': ('2024-01-01', '2025-12-31')}
out = {}
for vn in ('P_F_T20', 'D_F_T20'):
    df = pd.read_csv(HERE / f'events_{vn}.csv.gz', parse_dates=['sig', 'entry'], dtype={'code': str})
    df['net'] = df.U * df.px * (1 - S) / (1 + B) / df.Cf - 1; df['ab'] = df.net - df.ew; df['abg'] = df.raw - df.ew
    o = {}
    for k, (a, b) in W.items():
        x = df[(df.sig >= a) & (df.sig <= b)].copy(); em = x.entry.dt.to_period('M')
        x['w'] = x.Cf; x['wab'] = x.ab * x.Cf; x['wabg'] = x.abg * x.Cf
        g = x.groupby(em)
        cw = g.wab.sum() / g.w.sum(); cwg = g.wabg.sum() / g.w.sum()
        # 損益以預算為單位（每筆預算=1）：每月總淨異常損益 / 每月總投入
        o[k] = {'n': int(len(x)), 'equal_weight_ab_net_pct': round(float(x.ab.mean() * 100), 3), 't_month_equal': tstat(x.ab.groupby(em).mean()),
                'capital_weighted_ab_net_pct': round(float(x.wab.sum() / x.w.sum() * 100), 3), 't_month_capital_weighted': tstat(cw),
                'capital_weighted_ab_gross_pct': round(float(x.wabg.sum() / x.w.sum() * 100), 3), 't_month_capital_weighted_gross': tstat(cwg),
                'share_capital_in_4tranche_trades': round(float(x.w[x.ntr == 4].sum() / x.w.sum()), 3),
                'by_ntr_capital_share_and_ab_net_pct': {int(n): [round(float(gg.w.sum() / x.w.sum()), 3), round(float(gg.ab.mean() * 100), 3)] for n, gg in x.groupby('ntr')}}
    out[vn] = o
r = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
r['posthoc_diagnostics'] = {'note': '看完結果後加入，只解釋事件層與組合層的落差，不改判定。', 'capital_weighting': out,
                            'stuck_position_note': '2432 於 2008-08-19 進場後 2008-08-26 起長期停牌（面板至 2023 年才恢復成交），組合中占住 1 槽約 15 年（exit_delayed 3,653 日）；事件層該筆 raw −44.8%。'}
(HERE / 'results.json').write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))

#!/usr/bin/env python3
"""事後描述（未事前登記、不得推翻判定）：R2 去掉漏判 KY 的 NEVER_DOM、2019 起 D=S 日逐日狀態表。寫入 results.json 的 POSTHOC。python3 -I -B s5_posthoc.py"""
import json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
E = R['ERRATUM']
KY = {'6452': '康友-KY', '1262': '綠悅-KY', '6422': '君耀-KY'}
out = {'label': '事後描述（未事前登記，不得推翻判定）', 'utc': pd.Timestamp.now('UTC').isoformat()}
r2 = {}
for k, dn in (('2013_2026', 1640), ('2013_2018', 720), ('2019_2026', 920)):
    b = E['R2']['panelA_C0'][k]['by_issuer']; ky = sum(v for c, v in b.items() if c in KY)
    r2[k] = {'U_opt': E['R2']['panelA_C0'][k]['U_opt'], 'KY_slots': ky, 'U_opt_ex_KY': E['R2']['panelA_C0'][k]['U_opt'] - ky,
             'share_ex_KY': (E['R2']['panelA_C0'][k]['U_opt'] - ky) / dn}
out['R2_panelA_ex_KY_misclassified'] = {'KY_codes': KY, 'note': '這 3 檔下市 KY 股在名稱來源中沒有名稱，被判成 NEVER_DOM 而不是 FOREIGN（SOP 排除 KY）。扣掉後仍 > 5%。', 'by_seg': r2}
SA = pd.read_csv(HERE / 'data/slot_avail.csv', dtype={'code': str, 'sig': str})
x = SA[SA.sig >= '2019-01-01']
tab = x.groupby(['sig', 'case']).state.value_counts().unstack(fill_value=0).reset_index()
out['daily_2019_on'] = tab.to_dict('records')
R['POSTHOC'] = out
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out['R2_panelA_ex_KY_misclassified'], ensure_ascii=False, indent=1))
print(tab.to_string())

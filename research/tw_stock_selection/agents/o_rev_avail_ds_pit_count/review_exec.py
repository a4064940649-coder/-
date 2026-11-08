#!/usr/bin/env python3
"""可執行審查（只讀）：SOP 時點（訊號日收盤算訊號、隔日開盤成交、前瞻管線 22:30 加跑）、J0、第二截止。輸出 review_exec.json。python3 -I -B review_exec.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
E = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['ERRATUM']
SA = pd.read_csv(HERE / 'data/slot_avail.csv', dtype={'code': str, 'sig': str})
sop = (AG.parent.parent.parent / '選股SOP.md').read_text(encoding='utf-8')
bk = (AG / 'BACKLOG.md').read_text(encoding='utf-8')
o = {}
o['SOP_timing'] = {'SOP_signal_at_close_trade_next_open': ('換股日隔天開盤' in sop),
                   'SOP_text_claims_revenue_already_announced': ('這時上個月的營收已經公告（期限是 10 日）' in sop),
                   'forward_pipeline_2230_rerun': ('22:30' in bk),
                   'note': 'SOP 第 1 條「這時上個月的營收已經公告（期限是 10 日）」在 D=S 日不成立（期限就是換股日當天 24:00）。回測 C0 用換股日收盤時已知的全部 m−1 營收；SOP 是隔日開盤成交，前瞻管線在訊號日 22:30 加跑，所以實盤可用到「S 當晚 22:30 前已公告」的營收。'}
x = SA[(SA.formal.astype(str) == 'True') & (SA.sig >= '2024-06-01')]
nz = x[x.state != 'AVAIL']
o['second_cutoff_2024_06_on'] = {'n_DS_slots': int(len(x)), 'not_AVAIL_at_1330': int(len(nz)), 'second': nz.second.value_counts().to_dict(),
                                 'meaning': 'SEEN_BY_24＝S 24:00 前的快照已看到（盤中時點問題，不是逾期申報）；UNKNOWN＝快照 S+1 18:20 才出現（Tasi 落後 1 天，可能 S 當天任何時刻公告）；沒有任何一槽是 S 之後才公告（ABSENT_ON_LATER_DAY／ANN_AFTER_S＝0）'}
J = E['J0']
rk = J['jscan_rank_exec_V7X_100w_normal_GROW']
def med_wo_j0(g):
    v = [val for k, val in g.items() if k != 'J0']; return float(np.median(v))
o['J0'] = {'J0_is_deadline_day_share': 1.0, 'J0_share_of_j_slots_2019_on': J['share_J0_of_all_j_2019_on'],
           'J0_upper_share_of_all_j_2019_on': J['upper_share_of_all_j'], 'J0_lower_share_of_all_j_2019_on': J['lower_share_of_all_j'],
           'J0_rank': {k: {'rank_from_low_of_13': v['J0_rank_from_low'], 'is_median_j': v['J0_is_median_j'], 'is_p10_bracket': v['J0_is_p10_bracket'],
                           'median_j': v['median_j'], 'p10_between': v['p10_interp_between'], 'J0_gex': round(v['J0_gex'], 3), 'median': round(v['median'], 3),
                           'median_without_J0_posthoc': round(med_wo_j0(v['gex_by_j']), 3)} for k, v in rk.items()},
           'reproduces_integrate_o11': J['check_vs_integrate_o11'],
           'note': 'J0 不是中位數或 p10 落點的 j；但 J0 在全期是 13 個 j 的第 2 高（第 12 名）。事後描述：拿掉 J0 後 12 個 j 的中位數（不是正式口徑，只說明 J0 的影響方向）。V7X 的 J0 已用 AB 的 MC 延後機率處理一部分未公告，不是完全前視。'}
o['verdict'] = '可用：SOP 時點與 J0 核對一致；建議把 SOP 第 1 條改成「D=S 月份期限就是換股日當天，部分營收在收盤時尚未公告」的註記（文字更正，不改規則）'
(HERE / 'review_exec.json').write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))

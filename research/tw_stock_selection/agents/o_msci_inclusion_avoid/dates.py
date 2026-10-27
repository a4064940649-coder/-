"""(a) 日期對齊表：44 次 MSCI 審核的台灣公布日、換股訊號日、成交日、生效日。只用日曆，不碰報酬。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import oa_lib as O
RT = O.review_table()
RT.to_csv(O.HERE / 'review_dates.csv', index=False)
def cnt(g):
    return {'n_reviews': int(len(g)), 'known_at_signal_close': int(g.known_at_signal.sum()), 'known_only_at_exec_open': int((g.known_at_exec & ~g.known_at_signal).sum()),
            'after_exec': int((~g.known_at_exec).sum()), 'median_tdays_ann_to_E': float(g.tdays_ann_to_E.median()), 'median_tdays_exec_to_E': float(g.tdays_exec_to_E.median())}
out = {'all': cnt(RT), 'select_2010_2018': cnt(RT[RT.period == 'select']), 'oos_2019_2026': cnt(RT[RT.period == 'oos']),
       'by_month': {m: cnt(g) for m, g in RT.groupby(RT.review.str[-2:])}, 'notes': O.ANN_NOTE}
json.dump(out, open(O.HERE / 'dates.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(RT[['review', 'ann_tw', 'signal', 'exec', 'eff', 'known_at_signal', 'known_at_exec', 'tdays_ann_to_E', 'tdays_exec_to_E', 'src3']].to_string())
print(json.dumps({k: out[k] for k in ['all', 'select_2010_2018', 'oos_2019_2026', 'by_month']}, indent=1))

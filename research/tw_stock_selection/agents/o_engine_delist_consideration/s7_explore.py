#!/usr/bin/env python3
"""探索（凍結前，只有描述統計，不看任何策略報酬）：普查摘要、登記集合的被持有事件、價格面存活者缺口、事件層級描述。
python3 -I -B s7_explore.py → explore.json"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cz_lib as Z
C = json.loads((HERE / 'census.json').read_text(encoding='utf-8'))
E = pd.read_csv(HERE / 'events_v2.csv', dtype={'code': str, 'successor': str})
rows = pd.read_parquet(Z.WORK / 'census_rows.parquet')
per = C['per_result']
def grp(prefix):
    ks = [k for k in per if k.startswith(prefix)]
    ev = sorted({tuple(e) for k in ks for e in per[k]['perm_events_U']}); evi = sorted({tuple(e) for k in ks for e in per[k]['perm_events_IA']})
    return dict(n_results=len(ks), perm_slots_U=int(sum(per[k]['perm_slots_U'] for k in ks)), perm_slots_IA=int(sum(per[k]['perm_slots_IA'] for k in ks)),
                events_U=ev, events_IA=evi, n_events_U=len(ev), n_issuers_U=len({e[0] for e in ev}),
                frozen_share_max=max([per[k].get('frozen_share_max_U') or 0 for k in ks] or [0]))
REG = {'C0（FM_COUNTS C0|C0 2006～2013＋o14 lists_C0）': grp('o_g8_protocol_fin_erratum/ERRATUM.json/lists_C0'),
       'U_opt（FM_COUNTS C0|U_opt，只含 2006～2013）': grp('o_finmask_delisted_fin_erratum/FM_COUNTS.json/lists/C0|U_opt'),
       'U_opt_F2（FM_COUNTS F2|U_opt）': grp('o_finmask_delisted_fin_erratum/FM_COUNTS.json/lists/F2|U_opt'),
       'V7X 名單（o_v7_listing_exec_engine/data/plans_j7.csv AB_V7_B10，13 j×20 seeds）': grp('o_v7_listing_exec_engine/data/plans_j7.csv|AB_V7_B10'),
       'V7 AB（plans_j7.csv AB_V7）': grp('o_v7_listing_exec_engine/data/plans_j7.csv|AB_V7|'),
       'V7 官方日 B10（plan_V7_B10.csv）': grp('o_v7_listing_exec_engine/plan_V7_B10.csv'),
       'main_v2 主策略（independent_audit picks、o_rev_survivorship ORIG）': grp('o_rev_survivorship/plans_ORIG.json')}
# 價格面存活者缺口：delist_prices 有、ia 面板沒有
dp = pd.read_parquet(Z.O3 / 'delist_prices.parquet')
gap = dp.groupby('code').agg(first=('date', 'min'), last=('date', 'max'), n=('date', 'size')).reset_index()
gap['first'] = gap['first'].dt.date.astype(str); gap['last'] = gap['last'].dt.date.astype(str)
# 事件描述
H = E[E.held]
H2 = H.assign(year=pd.to_datetime(H.last_trade).dt.year)
seg = lambda y: '2006-2012' if y <= 2012 else ('2013-2018' if y <= 2018 else '2019-')
H2['seg'] = H2.year.map(seg)
win = []
for r in H.itertuples():
    lt = pd.Timestamp(r.last_trade)
    s = pd.Timestamp(r.succ_first_trade) if isinstance(r.succ_first_trade, str) and r.succ_first_trade else None
    term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) and r.terminate_date else None
    win.append(dict(code=r.code, type=r.type, lt=str(lt.date()), td_lt_to_s=(int(Z.idx_of(s) - Z.idx_of(lt)) if s is not None else None),
                    td_lt_to_term=(int(Z.idx_of(term) - Z.idx_of(lt)) if term is not None else None)))
# 被持有槽位（V7X 名單）依事件
v7 = rows[rows.result.str.startswith('o_v7_listing_exec_engine/data/plans_j7.csv|AB_V7_B10') & rows.perm_U]
slot_by_event = v7.groupby('code').size().sort_values(ascending=False).to_dict()
P = pickle.loads((Z.WORK / 'plans_ia.pkl').read_bytes())
out = dict(utc=pd.Timestamp.now('UTC').isoformat(), no_strategy_returns=True,
           census=dict(n_files_scanned=C['n_files_scanned'], n_files_with_lists=C['n_files_with_lists'], n_results=C['n_results'],
                       n_distinct_perm_events_U=C['n_distinct_perm_events_U'], n_issuers_U=C['n_issuers_U'],
                       price_gap_only_IA_events=C['price_gap_only_IA_events'], files_skipped_large=C['files_skipped_large'], files_error=C['files_error']),
           registered_sets=REG,
           v7x_slots_by_event=slot_by_event,
           price_survivor_gap_ia_panel=dict(n_codes=len(gap), rows=gap.to_dict('records'),
                                            note='V7 面板（x_data 以 delist_prices 補值）有、ia_core 用的 scratchpad/panel 沒有：2019-05-17～2023 下市的 32 檔，本題只列、不修'),
           events_held=dict(n=len(H), by_type=H.type.value_counts().to_dict(), by_verified=H.verified.value_counts().to_dict(),
                            by_segment=H2.groupby('seg').size().to_dict(), by_segment_type=H2.groupby(['seg', 'type']).size().rename('n').reset_index().to_dict('records'),
                            cash_premium_verified={r.code: float(r.dev_vs_lastclose_lt) for r in H[H.type == 'cash'].itertuples()},
                            stock_dev_lt_verified={r.code: float(r.dev_vs_lastclose_lt) for r in H[H.type.isin(['mixed', 'conversion', 'swap'])].itertuples() if np.isfinite(r.dev_vs_lastclose_lt)},
                            stock_dev_s_verified={r.code: float(r.dev_vs_lastclose_s) for r in H[H.type.isin(['mixed', 'conversion', 'swap'])].itertuples() if np.isfinite(r.dev_vs_lastclose_s)},
                            windows=win),
           mechanism_note='主策略 52 週腿會買到已鎖價的現金併購標的（o3：52 週比 0.998～1.000）；本題已核實的 5 筆現金對價相對最後收盤 +0.39%～+5.46%，'
                          '是套利價差，不是動能。換股類在 lt→s 空窗期承擔存續股價格風險（2311→3711 −9.8%、2448→3714 +9.9%、3698 +12.5%）。')
(HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: (v if k != 'rows' else None) for k, v in out['registered_sets'].items()}, ensure_ascii=False, default=str)[:3000])
print(out['v7x_slots_by_event']); print(out['events_held']['by_segment_type'])

#!/usr/bin/env python3
"""第 1 步：分母普查（不看報酬、不看任何可得性判定）。
1) 重建 C0（正式 245 日，另 latest 版含 2026-10-12），2006～2013 名單必須與 o_g8_protocol_fin_erratum/ERRATUM.json lists_C0 逐日逐檔相同（斷言）。
2) 每個官方訊號日分類 D<S／D=S／D>S（v2_core.deadline_case），逐日列出 D=S、D>S 日；2006～2013 必須是 33 日／330 槽（斷言）。
3) 可執行 j-scan：plans_j7.csv（AB_V7）逐月 J0 槽位數與 13 個 j 的槽位數。
4) 煙霧測試：pclib.VU 在 2006～2013 重現 ER2 的 U_opt＝162 槽（只是港口正確性檢查，數字登記前已知）。
輸出：census.json、data/c0_slots.csv、data/census_days.csv、scratchpad/o17_revavail/c0_pool.csv.gz。python3 -I -B s1_census.py"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pclib as P
import numpy as np, pandas as pd
V, K = P.V, P.K
T0 = time.time()
(HERE / 'data').mkdir(exist_ok=True)

rows = P.c0_cached(False)
rowsL = P.c0_cached(True)
ER = P.jload(P.AG / 'o_g8_protocol_fin_erratum/ERRATUM.json')['lists_C0']
mine = {str(r['sig'].date()): r['picks'] for r in rows if r['sig'] <= pd.Timestamp('2013-12-31')}
assert len(mine) == 93 and mine == ER, 'C0 2006～2013 名單與 lists_C0 不符'
# latest 版與正式版在 2026-09-11 以前逐日相同（只差 2026-10-12 多一日）
fm = {str(r['sig'].date()): r['picks'] for r in rows}
lt = {str(r['sig'].date()): r['picks'] for r in rowsL}
diff_latest = [d for d in fm if fm[d] != lt.get(d)]
extra_latest = [d for d in lt if d not in fm]

days = []
for r in rowsL:
    d = r['sig']; m = str(pd.Period(d, 'M'))
    case = V.deadline_case(d)
    De = P.deadline_engine(d); Dl = P.legal_deadline(d)
    days.append(dict(sig=str(d.date()), exec=str(r['exec'].date()), ym=m, rev_month=str(pd.Period(d, 'M') - 1), case=case,
                     D_engine=str(De.date()), D_legal=str(Dl.date()), legal_vs_S=('D<S' if Dl < d else 'D=S' if Dl == d else 'D>S'),
                     n_slots=len(r['picks']), n_pool=len(r['pool']), seg4=P.seg4(d), formal=str(d.date()) in fm,
                     note=P.SPECIAL_NOTE.get(m, '')))
DD = pd.DataFrame(days)
DD.to_csv(HERE / 'data/census_days.csv', index=False)
F = DD[DD.formal]
def cnt(x): return {'days': int(len(x)), 'slots': int(x.n_slots.sum()), 'DS_days': int((x.case == 'D=S').sum()), 'DS_slots': int(x[x.case == 'D=S'].n_slots.sum()),
                    'DgtS_days': int((x.case == 'D>S').sum()), 'DgtS_slots': int(x[x.case == 'D>S'].n_slots.sum()),
                    'DltS_days': int((x.case == 'D<S').sum()), 'legal_DgtS_days': int((x.legal_vs_S == 'D>S').sum())}
seg = {'2006_2013': F[F.sig <= '2013-12-31'], '2013_2018': F[(F.sig >= '2013-01-01') & (F.sig <= '2018-12-31')],
       '2013_2026': F[F.sig >= '2013-01-01'], '2014_2018': F[(F.sig >= '2014-01-01') & (F.sig <= '2018-12-31')],
       '2019_2026': F[F.sig >= '2019-01-01'], '2006_2012': F[F.sig <= '2012-12-31'], 'ALL': F,
       '2019_2024-05': F[(F.sig >= '2019-01-01') & (F.sig <= '2024-05-31')], '2024-06_': F[F.sig >= '2024-06-01'],
       '2014_2021-09-23': F[(F.sig >= '2014-01-01') & (F.sig <= '2021-09-23')]}
C = {k: cnt(v) for k, v in seg.items()}
C['latest_extra_2026-10-12'] = cnt(DD[~DD.formal])
assert C['2006_2013']['DS_days'] == 33 and C['2006_2013']['DS_slots'] == 330 and C['2006_2013']['slots'] == 930, C['2006_2013']
lst = DD[DD.case.isin(['D=S', 'D>S'])][['sig', 'case', 'D_engine', 'D_legal', 'legal_vs_S', 'n_slots', 'formal', 'note']].to_dict('records')
post19 = [x for x in lst if x['sig'] >= '2019-01-01']

# ---------- 槽位表 ----------
sl = []
for r in rowsL:
    d = r['sig']
    for k, c in enumerate(r['picks']):
        sl.append(dict(sig=str(d.date()), exec=str(r['exec'].date()), rank=k + 1, code=c, pr_rev=r['pr_rev'].get(c), pr_52=r['pr_52'].get(c),
                       formal=str(d.date()) in fm))
SL = pd.DataFrame(sl).merge(DD[['sig', 'case', 'legal_vs_S', 'seg4']], on='sig')
SL.to_csv(HERE / 'data/c0_slots.csv', index=False)
pool = []
for r in rowsL:
    for c in r['pool']:
        pool.append((str(r['sig'].date()), c, r['pr_rev'][c], r['pr_52'][c]))
PL = pd.DataFrame(pool, columns=['sig', 'code', 'pr_rev', 'pr_52'])
PL.to_csv(P.SCR / 'c0_pool.csv.gz', index=False)

# ---------- j-scan J0 ----------
PJ = pd.read_csv(P.AG / 'o_v7_listing_exec_engine/data/plans_j7.csv', dtype=str)
PJ = PJ[PJ.strat == 'AB_V7']
PJ['n'] = PJ.picks.fillna('').map(lambda s: len([x for x in s.split(',') if x]))
PJ['anchor'] = PJ.signal.str[:7]
j0 = PJ[PJ.seed == '0']
byj = j0.groupby(['anchor', 'j']).n.sum().unstack(fill_value=0)
byj.columns = [int(c) for c in byj.columns]; byj = byj[sorted(byj.columns)]
byj['all'] = byj.sum(axis=1)
byj.to_csv(HERE / 'data/jscan_slots_by_month_seed0.csv')
def jshare(a, b=None):
    x = byj[(byj.index >= a) & ((byj.index <= b) if b else True)]
    return {'J0_slots': int(x[0].sum()), 'all_j_slots': int(x['all'].sum()), 'share': float(x[0].sum() / x['all'].sum()), 'n_months': int(len(x))}
J = {'ALL': jshare('2006-01'), '2019_': jshare('2019-01'), '2006_2013': jshare('2006-01', '2013-12'), '2019_2024-05': jshare('2019-01', '2024-05'),
     '2024-06_': jshare('2024-06')}
# J0 訊號日＝期限日本身（逐月檢查）
j0d = j0[j0.j == '0'][['anchor', 'signal']].drop_duplicates()
from_dl = []
for a, s in zip(j0d.anchor, j0d.signal):
    m = pd.Period(a, 'M')
    import importlib
    D = P.legal_deadline(pd.Timestamp(m.year, m.month, 11)) if a not in ('2020-02',) else P.deadline_engine(pd.Timestamp(m.year, m.month, 11))
    from_dl.append(dict(anchor=a, j0_signal=s, deadline=str(D.date()), equal=(s == str(D.date()))))
J['J0_equals_deadline_share'] = float(np.mean([x['equal'] for x in from_dl]))
J['J0_not_equal'] = [x for x in from_dl if not x['equal']]

# ---------- 煙霧測試：ER2 港口 ----------
S = P.VU('opt', lo='2006-01-01', hi='2013-12-31')
UL = P.u_lists(S)
u_ins = sum(len(set(v['picks']) & set(v['aff'])) for k, v in UL.items() if k <= '2013-12-31')
er2 = P.jload(P.AG / 'o_rev_trunc_2009_erratum/ERRATUM2.json')
er2_list = sorted(map(tuple, er2['U_opt_slot_list']))
mine_list = sorted((k, c) for k, v in UL.items() if k <= '2013-12-31' for c in v['picks'] if c in set(v['aff']))
smoke = {'U_opt_2006_2013_mine': u_ins, 'ER2': er2['counts']['U_opt']['U_2006_2013'], 'slot_list_equal': mine_list == er2_list}
assert smoke['slot_list_equal'], smoke

out = {'utc': pd.Timestamp.utcnow().isoformat(), 'no_returns': '本程式未計算任何報酬、淨值或 t 值；未計算任何可得性判定比例',
       'engine_sha': P.ENGINE_SHA, 'n_formal_days': int(len(F)), 'n_latest_days': int(len(DD)), 'first': F.sig.min(), 'last_formal': F.sig.max(),
       'latest_diff_days_vs_formal': diff_latest, 'latest_extra_days': extra_latest,
       'counts': C, 'DS_DgtS_days_all': lst, 'DS_DgtS_days_2019_on': post19, 'n_DS_DgtS_2019_on': len(post19),
       'jscan': J, 'smoke_ER2_port': smoke, 'runtime_s': round(time.time() - T0, 1)}
P.jdump(out, HERE / 'census.json')
print(json.dumps({k: out[k] for k in ('n_formal_days', 'counts', 'n_DS_DgtS_2019_on', 'jscan', 'smoke_ER2_port', 'latest_diff_days_vs_formal', 'latest_extra_days')}, ensure_ascii=False, indent=0, default=str)[:6000])

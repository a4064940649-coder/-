#!/usr/bin/env python3
"""探索（後備題；只用挑選期：訊號日 ≤ 2018-12-31；不看任何報酬）。
(1) 換股日日曆：官方日是 J0 還是 J+1、各候選的訊號日與成交日。
(2) 公告時點（鉅亨網新聞，營收月 2012-12～2018-11）：首次新聞日相對期限 D 的交易日數分布、早公告比例逐年趨勢。
(3) 預測 2（描述）：早公告股的近 3 月年增率池內百分位 vs 期限日公告股。
(4) J0N 的 q：2026-09 期限日（10-12）當晚、次一交易日開盤前仍未公告的比例（非金融、非 INS15）。
(5) 前視暴露（名單層，不看報酬）：各候選非 AB 名單中「訊號時點還沒公告」的期望名額、AB 與非 AB 名單差異名額。
(6) 名單變動數（新進名額／月）。
輸出 explore.json。
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import fx_core as F
L, K, V = F.L, F.K, F.V
st = F.setup()
SEL_END = pd.Timestamp('2018-12-31')
out = {}

# (1) 日曆
off, off_end = F.official_days()
offs = [(m, d) for m, d in off if d <= SEL_END]
cnt_all = F.jcase_official()
cnt_sel = {}
for m, d in offs:
    j = F.TCAL.get_loc(d) - F.TCAL.get_loc(F.deadline(m)); cnt_sel[j] = cnt_sel.get(j, 0) + 1
J0, _ = F.version_days(0); J1, _ = F.version_days(1)
out['calendar'] = {'official_jcase_all': {str(k): v for k, v in cnt_all.items()}, 'official_jcase_SEL': {str(k): v for k, v in cnt_sel.items()},
                   'n_months_all': {'official': len(off), 'J0': len(J0), 'J+1': len(J1)},
                   'n_months_SEL': {'official': len(offs), 'J0': sum(d <= SEL_END for _, d in J0), 'J+1': sum(d <= SEL_END for _, d in J1)},
                   'missing_J0_months': [str(m) for m in st['anchors'] if m not in {x for x, _ in J0}],
                   'missing_J1_months': [str(m) for m in st['anchors'] if m not in {x for x, _ in J1}],
                   'J0_equals_official_share_all': float(np.mean([dict(J0).get(m) == d for m, d in off]))}

# (2)(3) 公告時點：新聞
A = pd.read_csv(F.DATA / 'ann_news.csv', dtype={'code': str, 'rev_month': str}, parse_dates=['first_news', 'first_strict'])
A = A[A.rev_month <= '2018-11']
S0 = L.V4(st['YFX'], st['Y5F'], st['DMT'])
rows = []
for m, d in offs:
    if d < pd.Timestamp('2013-01-01'): continue
    cs = S0.cross_section(d); pool = cs['top'] & ~L.FIN2MASK & ~L.FORMASK & np.isfinite(cs['y3'])
    y3 = np.where(pool, cs['y3'], np.nan); pr = pd.Series(y3).rank(pct=True).to_numpy()
    D = F.deadline(m); rm = str(m - 1)
    sub = A[A.rev_month == rm].set_index('code')
    for j in np.where(pool)[0]:
        c = K.COLS[j]
        if c not in sub.index: continue
        fn = sub.at[c, 'first_news']; fs_ = sub.at[c, 'first_strict']
        lag = F.TCAL.searchsorted(fn) - F.TCAL.get_loc(D)       # 新聞日相對 D 的交易日數（負＝早於期限）
        lag_s = (F.TCAL.searchsorted(fs_) - F.TCAL.get_loc(D)) if pd.notna(fs_) else np.nan
        rows.append((str(m), int(d.year), c, int(lag), lag_s, float(pr[j]), float(cs['y3'][j])))
E = pd.DataFrame(rows, columns=['m', 'year', 'code', 'lag', 'lag_strict', 'y3_pct', 'y3'])
def bucket(x):
    if not np.isfinite(x): return None
    if x <= -5: return 'early_ge5'
    if x <= -1: return 'early_1_4'
    if x == 0: return 'on_D'
    return 'after_D'
E['b'] = E.lag.map(bucket); E['bs'] = E.lag_strict.map(bucket)
dist = E.lag.clip(-12, 5).value_counts().sort_index()
out['ann_timing_news'] = {
    'n_pool_stock_months_with_news': int(len(E)), 'n_with_strict': int(E.lag_strict.notna().sum()),
    'lag_distribution_trading_days_vs_D(clip -12..5)': {int(k): int(v) for k, v in dist.items()},
    'bucket_share_loose': E.b.value_counts(normalize=True).round(4).to_dict(),
    'bucket_share_strict': E.bs.value_counts(normalize=True).round(4).to_dict(),
    'early_ge5_share_by_year_loose': E.groupby('year').apply(lambda g: float((g.lag <= -5).mean())).round(4).to_dict(),
    'before_D_share_by_year_loose': E.groupby('year').apply(lambda g: float((g.lag <= -1).mean())).round(4).to_dict(),
    'on_or_after_D_share_loose': float((E.lag >= 0).mean()),
    'note': '新聞日期是公告時間的上限：早公告比例是下限；新聞涵蓋偏向大型股與好消息（池內股月只涵蓋約 12%）'}
P2 = {}
for col in ('b', 'bs'):
    g = E.dropna(subset=[col]).groupby(col)['y3_pct']
    P2[col] = {k: {'mean_pct': float(v.mean()), 'n': int(v.size)} for k, v in g}
    # 按月分群：早公告（≥5 日）減期限日的月內差
    dd = []
    for m_, x in E.dropna(subset=[col]).groupby('m'):
        a = x[x[col] == 'early_ge5'].y3_pct; b = x[x[col] == 'on_D'].y3_pct
        if len(a) and len(b): dd.append(a.mean() - b.mean())
    P2[col]['early_ge5_minus_onD_by_month'] = {'mean': float(np.mean(dd)) if dd else None, 't': F.tstat(dd), 'n_months': len(dd)}
out['prediction2_descriptive'] = P2

# (4) q：2026-09 期限日當晚
FS = pd.read_csv(F.AG / 'rev_announce' / 'firstseen_2026-09.csv', dtype=str, index_col=0)
assert F.sha(F.AG / 'rev_announce' / 'firstseen_2026-09.csv') == st['fz']['other']['rev_announce_firstseen']['sha256']
FS['t'] = pd.to_datetime(FS.first_seen.str[:19])
nonfin = (FS.fin != 'True') & (FS.ins15 != 'True')
cut_open = pd.Timestamp('2026-10-13 09:00')
late = FS.t >= cut_open
q = float(late[nonfin].mean())
out['q_night'] = {'D': '2026-10-12', 'cutoff': str(cut_open), 'n_nonfin_nonins15': int(nonfin.sum()), 'n_late': int((late & nonfin).sum()),
                  'q': q, 'late_codes': sorted(FS.index[late & nonfin].tolist()),
                  'completion_rate_D_night_nonfin': 1 - q,
                  'completion_rate_D_close_nonfin(A_before)': float((FS.cat[nonfin] == 'A_before').mean()),
                  'rule': '完成率 ≥ 99% 的最早日＝期限日 D 當晚（非金融）→ J0N；q＝D 當晚仍未公告機率，全產業共用'}
Q = max(q, 1e-4)
(F.DATA / 'q_night.json').write_text(json.dumps({'q': Q, 'source': 'firstseen_2026-09.csv', 'n_late': out['q_night']['n_late'], 'n': out['q_night']['n_nonfin_nonins15']}, ensure_ascii=False), encoding='utf-8')

# (5)(6) 前視暴露與名單變動（只看名單，不看報酬）
def plan_of(days, end, U):
    S, pl, t0, t1, sd = F.build(days, end, U)
    return S, [p for p in pl if p[3] <= SEL_END]
cands = {'J+1': (J1, 'day'), 'J0': (J0, 'day'), 'J0N': (J0, 'night'), 'OFF': (off, 'day')}
EXP = {}
for name, (days, mode) in cands.items():
    end = F.version_days(1 if name == 'J+1' else 0)[1] if name != 'OFF' else off_end
    S_n, pn = plan_of(days, end, None)
    S_a, pa = plan_of(days, end, F.make_unavail(mode, 0, q=Q) if name != 'OFF' else V.make_unavail('mc', 0))
    exp_un = []; diff = []; newn = []; prev = set()
    for (e, pk, pool, d), (_, pk2, _, _) in zip(pn, pa):
        m = pd.Period(d, 'M'); D = F.deadline(m)
        if d < D: pu = np.ones(len(pk))
        elif d == D: pu = (np.full(len(pk), Q) if mode == 'night' else F.PCOL[pk])
        else: pu = np.zeros(len(pk))
        if str(m) == '2020-02' and D < d <= pd.Timestamp('2020-02-14'): pu = F.PCOL[pk]
        exp_un.append(pu.sum()); diff.append(len(set(pk) - set(pk2))); newn.append(len(set(pk) - prev)); prev = set(pk)
    EXP[name] = {'n_months': len(pn), 'expected_unannounced_slots_per_month(nonAB_list)': float(np.mean(exp_un)),
                 'share_months_with_exposure': float(np.mean(np.array(exp_un) > 1e-9)),
                 'slots_diff_AB_seed0_vs_nonAB_per_month': float(np.mean(diff)), 'months_with_slot_diff': int(np.sum(np.array(diff) > 0)),
                 'new_names_per_month': float(np.mean(newn[1:]))}
out['lookahead_exposure_SEL'] = EXP
(HERE / 'explore.json').write_text(json.dumps(F.jsonable(out), ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(F.jsonable({k: out[k] for k in ('calendar', 'q_night', 'lookahead_exposure_SEL', 'prediction2_descriptive')}), ensure_ascii=False, indent=1))
print(json.dumps(F.jsonable(out['ann_timing_news']), ensure_ascii=False, indent=1))

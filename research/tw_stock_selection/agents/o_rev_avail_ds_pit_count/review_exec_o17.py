#!/usr/bin/env python3
"""o17 題 1 反方可執行審查（成本與可成交、資料取得實際時點）。只讀研究員的檔案，不改任何檔案、不算報酬。
python3 -I -B review_exec_o17.py"""
import json, sys
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent
AG = HERE.parent
out = {}

# ---------- 1. 期限日 D=S／D>S 獨立重算（用 v2_core.TCAL，只讀） ----------
sys.path.insert(0, str(AG / 'main_v2'))
import v2_core as V
sys.path.pop(0)
TCAL = V.TCAL
CD = pd.read_csv(HERE / 'data/census_days.csv', dtype=str)
SPECIAL = {'2019-02': '2019-02-15', '2023-04': '2023-04-12', '2020-02': '2020-02-17'}
mis = []
for r in CD.itertuples():
    s = pd.Timestamp(r.sig); m = r.ym
    base = pd.Timestamp(s.year, s.month, 10)
    D = TCAL[TCAL >= (pd.Timestamp(SPECIAL[m]) if m in SPECIAL else base)][0]
    lab = 'D<S' if D < s else ('D=S' if D == s else 'D>S')
    if lab != r.legal_vs_S: mis.append((r.sig, lab, r.legal_vs_S))
F = CD[CD.formal == 'True']
post19 = F[(F.sig >= '2019-01-01') & (F.legal_vs_S != 'D<S')]
out['deadline_recheck'] = {'mismatch_legal': mis, 'n_formal': int(len(F)), 'n_slots_formal': int(F.n_slots.astype(int).sum()),
                           'formal_2019_on_days': int((F.sig >= '2019-01-01').sum()),
                           'formal_2019_on_DS_or_DgtS_days_legal': int(len(post19)),
                           'formal_all_DS_or_DgtS_days_legal': int((F.legal_vs_S != 'D<S').sum()),
                           'months_2019_on_missing': sorted(set(str(p) for p in pd.period_range('2019-01', '2026-09', freq='M')) - set(F[F.sig >= '2019-01-01'].ym))}

# ---------- 2. R1 計數由 slot_avail 重推 ----------
SA = pd.read_csv(HERE / 'data/slot_avail.csv', dtype={'code': str})
SAf = SA[SA.formal.astype(str) == 'True']
def up(x): return int((x.state != 'AVAIL').sum())
r1 = {'r1a': (up(SAf[SAf.sig <= '2013-12-31']), 930), 'r1b': (up(SAf), 2450), 'r1c': (up(SAf[SAf.sig >= '2019-01-01']), 920),
      'lower_all': int((SAf.state == 'UNAVAIL').sum())}
out['R1_recount'] = {k: (v if isinstance(v, int) else {'n': v[0], 'den': v[1], 'share': v[0] / v[1]}) for k, v in r1.items()}

# ---------- 3. 2024-06 起：可得時點相對於實盤的三個時鐘 ----------
EV = pd.read_csv(HERE / 'data/slot_avail_evidence.csv', dtype={'code': str})
M = SA.merge(EV[['sig', 'code', 'evidence']], on=['sig', 'code'])
rows = []
for r in M[M.sig >= '2024-06-01'].itertuples():
    ev = json.loads(r.evidence); s = pd.Timestamp(r.sig)
    t = ev.get('tasi'); u = ev.get('upstream')
    tasi_T = pd.Timestamp(t[0]) if t else pd.NaT
    up_real = u is not None and u[2] != 'file_add_no_prev'
    rows.append(dict(sig=r.sig, code=r.code, state=r.state, formal=r.formal, tasi_first=tasi_T,
                     tasi_by_1330=bool(pd.notna(tasi_T) and tasi_T <= s + pd.Timedelta(hours=13, minutes=30)),
                     tasi_by_2230=bool(pd.notna(tasi_T) and tasi_T <= s + pd.Timedelta(hours=22, minutes=30)),
                     tasi_by_next_0900=bool(pd.notna(tasi_T) and tasi_T <= s + pd.Timedelta(days=1, hours=9)),
                     upstream_realtime=up_real))
X = pd.DataFrame(rows)
Xf = X[X.formal.astype(str) == 'True']
out['clock_2024_06_formal'] = {'n': int(len(Xf)), 'tasi_by_S1330': int(Xf.tasi_by_1330.sum()), 'tasi_by_S2230': int(Xf.tasi_by_2230.sum()),
                               'tasi_by_S+1_0900': int(Xf.tasi_by_next_0900.sum()),
                               'upstream_has_realtime_timestamp': int(Xf.upstream_realtime.sum()),
                               'not_seen_by_S+1_0900': [f'{a} {b}' for a, b in zip(Xf[~Xf.tasi_by_next_0900].sig, Xf[~Xf.tasi_by_next_0900].code)]}
FS = pd.read_csv(HERE / 'raw/ts_tasi_upstream_firstseen.csv.gz', dtype=str)
FS['ymd'] = FS.ym
u24 = FS[(FS.source == 'upstream') & (FS.ym >= '2024-05') & (FS.ym <= '2026-08')]
out['upstream_2024_05_to_2026_08'] = {'n': int(len(u24)), 'reason': u24.reason.value_counts().to_dict(),
                                      'first_seen_min': str(pd.to_datetime(u24['T'], utc=True).min())}

# ---------- 4. R3 處置重算（官方全歷史，正式槽）＋ 2026-10-12 latest ----------
DU = pd.read_csv(HERE / 'raw/disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
g = {c: x for c, x in DU.groupby('code')}
def ind(c, d):
    x = g.get(c)
    return False if x is None else bool(((x.start <= d) & (x.end >= d)).any())
SL = pd.read_csv(HERE / 'data/c0_slots.csv', dtype={'code': str})
SL['dsig'] = [ind(c, pd.Timestamp(s)) for c, s in zip(SL.code, SL.sig)]
SL['dtrd'] = [ind(c, pd.Timestamp(e)) for c, e in zip(SL.code, SL.exec)]
SLf = SL[SL.formal.astype(str) == 'True']
def seg(a, b):
    x = SLf[(SLf.sig >= a) & (SLf.sig <= b)]
    return {'slots': int(len(x)), 'sig': int(x.dsig.sum()), 'trd': int(x.dtrd.sum()), 'max_share': float(max(x.dsig.mean(), x.dtrd.mean())),
            'trd_list': [f'{s} {c}' for s, c in zip(x[x.dtrd].sig, x[x.dtrd].code)]}
out['R3_recount'] = {'2006_2013': seg('2006-01-01', '2013-12-31'), '2014_2021-09-23': seg('2014-01-01', '2021-09-23'),
                     '2021-09-24_': seg('2021-09-24', '2026-12-31'), 'ALL_formal': seg('2006-01-01', '2026-12-31')}
L = SL[SL.sig == '2026-10-12']
out['latest_2026_10_12'] = {'picks': list(L.code), 'in_disposal_on_S': list(L[L.dsig].code), 'in_disposal_on_exec': list(L[L.dtrd].code),
                            'note': 'C0 latest 名單用 disposal_5y（2026-10-05 起停更）；live_tracking C-1 已刪 2466、6213'}

# ---------- 5. MOPS ann_date（2026-09 營收）晚於法定期限 ----------
AN = pd.read_csv(HERE / 'raw/ts_mops_ann_date.csv', dtype=str)
a9 = AN[AN.ym == '2026-09']
out['mops_2026_09_after_deadline_1012'] = {'n': int(len(a9)), 'after': int((a9.ann_date > '2026-10-12').sum()),
                                          'share': float((a9.ann_date > '2026-10-12').mean()),
                                          'on_deadline': int((a9.ann_date == '2026-10-12').sum())}
# ---------- 6. 22:30 實盤時點 vs FINDINGS §11 ----------
CORR = (AG / 'live_tracking/main_forward/CORRECTIONS.md').read_text(encoding='utf-8')
out['forward_rule_conflict'] = {'CORRECTIONS_says_2230_list_wrong': ('3231、6691、6196' in CORR) and ('03:48' in CORR),
                                'FINDINGS_s11_says_use_2230': True}
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

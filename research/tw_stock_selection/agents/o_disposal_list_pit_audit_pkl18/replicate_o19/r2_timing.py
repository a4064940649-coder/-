"""獨立複製 o19 S1 時點：領先交易日數、與月末換股日 d／成交日 e 的關係。"""
import json
import numpy as np, pandas as pd

BASE = '/home/user/-/research/tw_stock_selection/agents/o_disposal_list_pit_audit_pkl18/'
OUT = BASE + 'replicate_o19/'
PANEL = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'

C = pd.read_csv(OUT + 'r1_universe.csv', dtype={'code': str})
adj = pd.read_parquet(PANEL + 'adj.parquet')
cnt = adj.notna().sum(axis=1)
cal = adj.index[cnt >= 300]
cal_raw = adj.index
print('panel', cal_raw[0].date(), cal_raw[-1].date(), len(cal_raw), 'cal>=300', len(cal))

ann = pd.to_datetime(C.announce); st = pd.to_datetime(C.start)
pos = {d: i for i, d in enumerate(cal)}
lag = []
for a, s in zip(ann, st):
    if a in pos and s in pos:
        lag.append(pos[s] - pos[a])
    else:
        lag.append(np.nan)
C['lag_td'] = lag
vc = C.lag_td.value_counts(dropna=False).to_dict()
out_cal = C[C.lag_td.isna()]
oc_years = out_cal.announce.str[:4].value_counts().sort_index().to_dict()
lag2 = C[C.lag_td == 2][['code', 'name', 'announce', 'start']].to_dict('records')
# 使用原始面板日曆（不剔除 <300）也算一次
pos2 = {d: i for i, d in enumerate(cal_raw)}
lagr = [pos2[s] - pos2[a] if (a in pos2 and s in pos2) else np.nan for a, s in zip(ann, st)]
vc_raw = pd.Series(lagr).value_counts(dropna=False).to_dict()

# 月末訊號日：2006-01～2026-09
me = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).max()
me = [d for d in me if pd.Timestamp('2006-01-01') <= d <= pd.Timestamp('2026-09-30')]
print('signal days', len(me), me[0].date(), me[-1].date())
E = []
for d in me:
    i = pos[d]
    E.append(cal[i + 1] if i + 1 < len(cal) else None)
en = pd.to_datetime(C.end)
hit = set()
rel = {'lt': 0, 'eq': 0, 'gt': 0, 'eq_start_e': 0}
pairs = []
for d, e in zip(me, E):
    m = ((st <= d) & (en >= d)) | ((st <= e) & (en >= e))
    for i in np.where(m)[0]:
        pairs.append((i, d, e))
        a = ann.iloc[i]
        if a < d: rel['lt'] += 1
        elif a == d:
            rel['eq'] += 1
            rel['eq_start_e'] += int(st.iloc[i] == e)
        else: rel['gt'] += 1
res = dict(lag_counts={str(k): int(v) for k, v in vc.items()}, out_cal_years=oc_years, lag2=lag2,
           lag_counts_rawcal={str(k): int(v) for k, v in vc_raw.items()},
           n_signal_days=len(me), n_pairs=len(pairs), n_rows=len(set(p[0] for p in pairs)), rel=rel)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
json.dump(res, open(OUT + 'r2_timing.json', 'w'), ensure_ascii=False, indent=1, default=str)
# 口徑二：start 落在面板缺日（颱風休市日等）時取下一個面板交易日（searchsorted）
cal2 = cal; pos2 = pos  # 剔除有值檔數 < 300 的日子（2018-01-13 週六）
lag3 = []
for a, s in zip(ann, st):
    if a in pos2 and cal2[0] <= s <= cal2[-1] and not (pd.Timestamp('2024-01-01') <= s <= pd.Timestamp('2024-02-14')):
        lag3.append(int(cal2.searchsorted(s)) - pos2[a])
    else:
        lag3.append(np.nan)
C['lag_ss'] = lag3
vc3 = C.lag_ss.value_counts(dropna=False).to_dict()
print('searchsorted lag:', vc3, C[C.lag_ss.isna()].announce.str[:4].value_counts().to_dict())
res['lag_counts_searchsorted'] = {str(k): int(v) for k, v in vc3.items()}
json.dump(res, open(OUT + 'r2_timing.json', 'w'), ensure_ascii=False, indent=1, default=str)

"""用主策略實際訊號日（o18 C0 plan／sd）重算 d／e 涵蓋關係。"""
import json, numpy as np, pandas as pd
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
adj = pd.read_parquet(P + 'adj.parquet'); cal = adj.index
J = json.load(open('r2b_sigdays.json'))
C = pd.read_csv('r1_universe.csv', dtype={'code': str})
ann = pd.to_datetime(C.announce); st = pd.to_datetime(C.start); en = pd.to_datetime(C.end)
out = {}
def run(days):
    lt = eq = gt = eqe = n = 0; rows = set()
    for d in days:
        d = pd.Timestamp(d); i = cal.get_loc(d); e = cal[i + 1] if i + 1 < len(cal) else d
        m = ((st <= d) & (en >= d)) | ((st <= e) & (en >= e))
        for k in np.where(m)[0]:
            n += 1; rows.add(k); a = ann.iloc[k]
            if a < d: lt += 1
            elif a == d: eq += 1; eqe += int(st.iloc[k] == e)
            else: gt += 1
    return dict(ndays=len(days), pairs=n, rows=len(rows), lt=lt, eq=eq, eq_start_e=eqe, gt=gt)
out['sd'] = run(J['sd']); out['plan_d'] = run([p[0] for p in J['pairs']])
out['sd_first_last'] = [J['sd'][0], J['sd'][-1]]
print(json.dumps(out, indent=1))
json.dump(out, open('r2c_rel.json', 'w'), indent=1)
# 補 2006-01～03 三個訊號日（規則：每月 10 日之後第一個交易日，與 sd 其餘日期一致）
chk = sum(1 for s in J['sd'] if pd.Timestamp(s) == cal[cal > pd.Timestamp(s[:8] + '10')][0])
extra = [str(cal[cal > pd.Timestamp('2006-%02d-10' % m)][0].date()) for m in (1, 2, 3)]
out['rule_match_sd'] = chk
out['sd249'] = run(extra + J['sd']); out['extra'] = extra
print(json.dumps({k: out[k] for k in ('rule_match_sd', 'sd249', 'extra')}, indent=1))
json.dump(out, open('r2c_rel.json', 'w'), indent=1)

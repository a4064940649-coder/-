"""獨立複製 o19 S1：年檔溯源、封存一致性、普通股 universe、接回 announce。"""
import json, glob, re, collections, sys
import pandas as pd

BASE = '/home/user/-/research/tw_stock_selection/agents/o_disposal_list_pit_audit_pkl18/'
OUT = BASE + 'replicate_o19/'


def roc(s):
    y, m, d = s.strip().split('/')
    return '%04d-%s-%s' % (int(y) + 1911, m, d)


rows = []
cons = {}
for f in sorted(glob.glob(BASE + 'raw/twse_punish/twse_*.json')):
    fy = int(re.findall(r'(\d{4})\.json', f)[0])
    d = json.load(open(f))
    D = d['data']
    c = collections.Counter(r[2] for r in D)
    cum_bad = sum(1 for r in D if int(r[4]) != c[r[2]])
    cons[fy] = dict(total=d['total'], n=len(D), count=d['count'], maxno=max(r[0] for r in D),
                    total_eq=d['total'] == len(D), count_eq=d['count'] == max(r[0] for r in D),
                    cum_bad=cum_bad)
    for r in D:
        a, b = r[6].split('～')
        rows.append(dict(fy=fy, no=r[0], announce=roc(r[1]), code=r[2].strip(), name=r[3].strip(),
                         cum=int(r[4]), cond=r[5], start=roc(a), end=roc(b), measure=r[7]))
A = pd.DataFrame(rows)
key = ['code', 'announce', 'start', 'end']
nA = len(A)
B = A.drop_duplicates(key)
overlap = nA - len(B)
# 重疊列是否同時出現在相鄰兩年
dup = A[A.duplicated(key, keep=False)]
ov_years = dup.groupby(key).fy.apply(lambda s: tuple(sorted(s))).value_counts().head(5).to_dict()


def is_common(code, name):
    return len(code) == 4 and code.isdigit() and code[0] != '0' and 'DR' not in str(name).upper()

B = B.copy()
B['common'] = [is_common(c, n) for c, n in zip(B.code, B.name)]
C = B[B.common].copy()
C['sy'] = C.start.str[:4].astype(int)
by_sy = C.sy.value_counts().sort_index()

# 2019 年檔濾掉
f19 = A[A.fy == 2019]
etf = f19[~f19.code.str.fullmatch(r'[1-9]\d{3}')]
dr = f19[f19.code.str.fullmatch(r'[1-9]\d{3}') & f19.name.str.upper().str.contains('DR')]

# 跨年重疊缺漏：一個證券在 N 年檔 end 跨入 N+1，是否在 N+1 年檔也出現
miss = 0
for _, r in A[A.end.str[:4].astype(int) > A.fy].iterrows():
    nxt = A[(A.fy == r.fy + 1)]
    if not ((nxt.code == r.code) & (nxt.start == r.start)).any():
        miss += 1

# universe 對照
U = pd.read_csv(BASE + 'raw/lineage/disposal_official_universe.csv', dtype={'code': str})
H = pd.read_csv(BASE + 'raw/lineage/disposal_hist_rejudge.csv', dtype={'code': str, 'name': str}, keep_default_na=False)
setC = set(zip(C.code, C.start))
setU = set(zip(U.code, U.start))
H['common'] = [is_common(c, n) for c, n in zip(H.code, H.name)]
HB = set(zip(H.code, H.announce, H.start, H.end))
setB = set(map(tuple, B[key].values))
# 接回 announce：(code,start) 一對一
cs = C.groupby(['code', 'start']).size()
UJ = U.merge(C[['code', 'start', 'announce', 'name']], on=['code', 'start'], how='left')
ok = UJ.announce.notna().sum()
lt = (UJ.announce < UJ.start).sum()
ge = (UJ.announce >= UJ.start).sum()
# 以 ordr 接回 rejudge 的同排序
Hc = H[H.common].reset_index(drop=True)
kind = H[H.common].kind.value_counts().to_dict()
inpanel = int(H[H.common].in_panel.sum())

res = dict(
    nA=nA, overlap=overlap, nB=len(B), nC=len(C), nU=len(U), C_eq_U=setC == setU,
    B_eq_hist_rejudge_keys=setB == HB, n_hist=len(H), ov_years=str(ov_years),
    by_start_year={int(k): int(v) for k, v in by_sy.items()},
    f19_etf_warrant=len(etf), f19_dr=len(dr), f19_total=len(f19),
    cons_all_ok=all(v['total_eq'] and v['count_eq'] and v['cum_bad'] == 0 for v in cons.values()),
    cum_bad_total=sum(v['cum_bad'] for v in cons.values()), cons_2019=cons[2019],
    overlap_missing=miss, cs_one_to_one=bool((cs == 1).all()),
    join_ok=int(ok), announce_lt_start=int(lt), start_le_announce=int(ge),
    kind=kind, in_panel=inpanel,
)
json.dump(res, open(OUT + 'r1_lineage.json', 'w'), ensure_ascii=False, indent=1)
C.to_csv(OUT + 'r1_universe.csv', index=False)
print(json.dumps(res, ensure_ascii=False, indent=1))

# 口徑二：去重後 start 年＝2019，先判 DR（名稱含 DR），其餘非 4 碼普通股＝ETF／權證
B19 = B[B.start.str[:4] == '2019']
isdr = B19.name.str.upper().str.contains('DR')
dr2 = int(isdr.sum())
etf2 = int((~isdr & ~B19.common).sum())
print('start2019 dedup: total', len(B19), 'common', int(B19.common.sum()), 'DR', dr2, 'ETF/warrant', etf2)
res.update(s19_dedup_total=len(B19), s19_dr=dr2, s19_etf_warrant=etf2)
json.dump(res, open(OUT + 'r1_lineage.json', 'w'), ensure_ascii=False, indent=1)

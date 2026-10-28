"""獨立建族群：掃描 agents/ 下日頻淨值 parquet → 窗報酬。"""
import sys, re, json; sys.path.insert(0, '.')
from rlib import *
EXCL_FOLDERS = {'o_main_spa_breaks', 'o_v5_pit_revenue', 'o_rev_event_trigger', 'value_fix'}
BENCH_RE = re.compile(r'^(EW.*|.*EW|0050|E0050|E50|A2330|r0050|npos|nav_?0050|TAIEX|.*POOL_EW.*|BM_.*)$', re.I)
SEED_RE = re.compile(r'\|s\d{2}$')
EXCL_RE = re.compile(r'DIAG|placebo|shadow|short|gross|zero|nocost|no_cost|stress|nodisc|fullfee|cap|flat|cost\d|npos|r0050', re.I)
SLIP_RE = re.compile(r'(?<![\d.])0\.00\d+')
PRIO = ['o_v4_revenue_patch_official', 'main_v2', 'o_infra_v3_rebuild', 'main_refine', 'revenue_longhistory', 'momentum_family']

def is_panel(cols):
    c = [str(x) for x in cols[:20]]
    return sum(bool(re.fullmatch(r'\d{4,6}[A-Z]?', x)) for x in c) >= len(c) * 0.5

def bad_slip(name):
    for m in SLIP_RE.findall(name):
        if abs(float(m) - 0.001) > 1e-12: return True
    return False

files = []
for f in sorted(AG.glob('**/*.parquet')):
    rel = f.relative_to(AG); top = rel.parts[0]
    if top in EXCL_FOLDERS: continue
    files.append(rel)
# 哪些資料夾在 replicate 之外就有日頻淨值
def daily(rel):
    try:
        import pyarrow.parquet as pq
        sch = pq.read_schema(AG / rel)
        if sch.metadata is None: pass
        d = pd.read_parquet(AG / rel)
    except Exception: return None
    if not isinstance(d.index, pd.DatetimeIndex) or len(d) < 30: return None
    gap = np.median(np.diff(d.index.values).astype('timedelta64[D]').astype(float))
    if gap > 1.5: return None
    if is_panel(d.columns): return None
    if 'search_monthly' in str(rel): return None
    return d
cache = {}
for rel in files:
    d = daily(rel)
    if d is not None: cache[rel] = d
has_own = {rel.parts[0] for rel in cache if not any(p.startswith('replicate_') for p in rel.parts)}
rows = []; R = {}
ncols = 0; folders = set()
order = sorted(cache, key=lambda r: (PRIO.index(r.parts[0]) if r.parts[0] in PRIO else len(PRIO), str(r)))
for rel in order:
    rep = any(p.startswith('replicate_') for p in rel.parts)
    if rep and rel.parts[0] in has_own: continue
    d = cache[rel]; folders.add(rel.parts[0]); ncols += d.shape[1]
    for c in d.columns:
        name = str(c); key = f'{rel}::{name}'
        why = None
        if BENCH_RE.match(name): why = 'bench'
        elif EXCL_RE.search(name) or EXCL_RE.search(rel.stem): why = 'excl'
        elif bad_slip(name): why = 'slip'
        elif SEED_RE.search(name): why = 'seed'
        if why: rows.append(dict(key=key, why=why)); continue
        s = pd.to_numeric(d[c], errors='coerce').dropna()
        if len(s) < 30: rows.append(dict(key=key, why='short_series')); continue
        if s.min() < 0 or (s.abs().median() < 0.05 and s.max() < 1.0):
            s = (1 + s).cumprod(); kind = 'ret'
        else: kind = 'nav'
        r = win_ret(s)
        if np.isfinite(r).sum() < 3: rows.append(dict(key=key, why='no_windows')); continue
        R[key] = r; rows.append(dict(key=key, why='member', kind=kind))
print('folders', len(folders), 'cols', ncols, 'members_pre', len(R))
M = pd.DataFrame(R, index=G[:-1])
# 去重：窗報酬完全相同（含缺值位置）
sig = M.round(12).T.apply(lambda r: hash(tuple(np.where(np.isnan(r.values), -999, r.values))), axis=1)
keep = ~sig.duplicated(keep='first')
M = M.loc[:, keep.values]
print('members_dedup', M.shape[1], 'folders', len({k.split('/')[0].split('::')[0] for k in M.columns}))
M.to_parquet(OUT / 'scan_ret.parquet')
pd.DataFrame(rows).to_csv(OUT / 'scan_rows.csv', index=False)

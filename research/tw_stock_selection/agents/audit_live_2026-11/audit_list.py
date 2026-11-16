#!/usr/bin/env python3
"""稽核第 1～3 項：2026-11 名單與網站資料的獨立重算（不 import、不執行 make_list.py 或任何研究程式庫）。

規則依 s20_persistent_growth/FINDINGS.md「條件」（S10 E0／W10_70）與 s4lib.make_signal／cslib.Panel／stlib.universe／s17lib.zscore 的文字定義自寫：
- 訊號日（A10）：訊號月份 10 日（含）之後第一個交易日的下一個交易日；成交日＝訊號日次一交易日開盤。
- 股池（訊號日 t）：當天日 K market＝twse、代號 [1-9]ddd、不是 91 開頭、當天有成交（收盤 > 0）、原始收盤 ≥ 10、
  60 日平均成交值 ≥ 1,000 萬（近 60 個交易日每天都要在日 K 內〔上市或上櫃〕，在日 K 內但無成交算 0）、當天股名不符合 -(KY)?創$。
- 還原收盤 G：含息日報酬＝收盤 ÷（收盤 − 漲跌價差）− 1；|報酬| > 50% 或參考價 ≤ 0 改用收盤 ÷ 前收（無成交日 0）；連乘。
  季線：G > G 的 60 日簡單平均（嚴格大於）；RET20＝G[t] ÷ G[t−20] − 1；距季線＝G ÷ 60 日均 − 1。
- SURGE（營收月份 m＝訊號月份 − 1）：D＝營收(m) − 營收(m−12)；SURGE＝(D(m) − 平均 D(m−24…m−1)) ÷ 樣本標準差（ddof 1）；
  24 個月全部有值、D(m) 有值、σ > 1e-9 ×（窗內 |D| 平均 ＋ 1e-12）才給值。營收：同代號優先上市（sii）、營收 ≤ 0 當缺值、千元。
  營收 ≤ 2026-09 用 tw_revenue.csv；2026-10 用官方日 K 倉庫 data/mops/revenue_hist/2026-10_twse.csv，
  並以 data/meta/revenue_seen/2026-10/ 截止 2026-11-11T14:00:00Z 前已看到的公司為準（另跑不截止版比較）。
- L_MA：股池內、有 SURGE、在季線上，SURGE 由大到小（同分依代號）。F1：L_MA 前 10 名中 RET20 > 10%。E0：持股不在 L_MA 前 20 名就賣。
- 有訊號：股池內有 SURGE ≥ 50 檔。
輸出：audit_list.json（本資料夾）。暫存：scratchpad/audit/。python3 -I -B audit_list.py"""
import io, json, re, subprocess, math, gzip, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = '/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data'
TSD = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsd'
REV = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
SITE = Path('/home/user/-/xq_scripts/site')
LISTS = AG / 's18_live_list' / 'lists'
README = Path('/home/user/-/xq_scripts/README.md')
OPMAN = Path('/home/user/-/xq_scripts/底倉衛星_操作說明.md')
NAVS = AG / 's19_xq_threshold' / 'navs.csv.gz'
S19TR = AG / 's19_xq_threshold' / 'trades.csv.gz'
MONTH = pd.Period('2026-11', 'M'); RM = MONTH - 1
CUTOFF = '2026-11-11T14:00:00Z'
WIN = 160
INNOV = re.compile(r'-(KY)?創$'); CODE = re.compile(r'^[1-9]\d{3}$')
TOL3 = 5e-4


def git(repo, *a, inp=None):
    return subprocess.run(['git', '-C', repo, *a], input=inp, capture_output=True, check=True).stdout


def ls(repo, sub):
    out = {}
    for line in git(repo, 'ls-tree', 'HEAD', sub.rstrip('/') + '/').decode().strip().split('\n'):
        if line:
            meta, path = line.split('\t', 1); out[path.rsplit('/', 1)[1]] = meta.split()[2]
    return out


def cat(repo, shas):
    if not shas:
        return {}
    data = git(repo, 'cat-file', '--batch', inp=('\n'.join(shas) + '\n').encode())
    out = {}; i = 0
    while i < len(data):
        nl = data.index(b'\n', i); hdr = data[i:nl].split()
        sha, size = hdr[0].decode(), int(hdr[2]); out[sha] = data[nl + 1:nl + 1 + size]; i = nl + 1 + size + 1
    return out


def r3(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 3)


res = {'checks': [], 'notes': []}


def chk(item, site, mine, ok, src='', note='', warn=False):
    d = None
    try:
        if site is not None and mine is not None and not isinstance(site, (str, bool)) and not isinstance(mine, (str, bool)):
            d = float(mine) - float(site)
    except Exception:
        d = None
    res['checks'].append(dict(item=item, site=site, recomputed=mine, diff=d, verdict='通過' if ok else ('注意' if warn else '不通過'), source=src, note=note))
    return ok


# ---------------------------------------------------------------- 1. 交易日與日 K
raw_ls = ls(RAW, 'data/universe/daily'); tsd_ls = ls(TSD, 'data/universe/daily')
days_raw = sorted(k[:10] for k in raw_ls); days_tsd = sorted(k[:10] for k in tsd_ls)
overlap = sorted(set(days_raw) & set(days_tsd))
blob_diff = [d for d in overlap if raw_ls[d + '.csv'] != tsd_ls[d + '.csv']]
only_raw = sorted(set(days_raw) - set(days_tsd)); only_tsd = sorted(set(days_tsd) - set(days_raw))
DAYS = sorted(set(days_raw) | set(days_tsd))
res['data'] = dict(raw_head=git(RAW, 'rev-parse', 'HEAD').decode().strip(), tsd_head=git(TSD, 'rev-parse', 'HEAD').decode().strip(),
                   raw_last=days_raw[-1], tsd_last=days_tsd[-1], n_days_tsd=len(days_tsd), overlap=len(overlap), overlap_blob_diff=blob_diff,
                   only_in_raw=only_raw[-5:], n_only_in_raw=len(only_raw), only_in_tsd=only_tsd)
# 缺日：2025-07 之後的平日沒有日 K 檔
wk = pd.bdate_range('2025-07-01', DAYS[-1])
miss = [str(d.date()) for d in wk if str(d.date()) not in set(DAYS)]
res['data']['weekday_without_file_since_2025_07'] = miss

# 訊號日
def first_on_after(d):
    return next(x for x in DAYS if x >= d)
d10 = first_on_after(f'{MONTH}-10'); i10 = DAYS.index(d10)
SIG = DAYS[i10 + 1]; EXE = DAYS[i10 + 2]
res['signal'] = dict(deadline_day=d10, sig_date=SIG, exec_date=EXE)
iS = DAYS.index(SIG)
use = DAYS[max(0, iS - WIN + 1):]
blobs_raw = cat(RAW, [raw_ls[d + '.csv'] for d in use if d + '.csv' in raw_ls and d not in only_tsd])
need_tsd = [d for d in use if d + '.csv' not in raw_ls]
blobs_tsd = cat(TSD, [tsd_ls[d + '.csv'] for d in need_tsd])
frames = {}; rowcount = {}
for d in use:
    b = blobs_raw.get(raw_ls.get(d + '.csv')) if d + '.csv' in raw_ls else blobs_tsd[tsd_ls[d + '.csv']]
    df = pd.read_csv(io.BytesIO(b), dtype={'stock_id': str, 'market': str, 'name': str})
    rowcount[d] = int(len(df))
    frames[d] = df
res['data']['rows_per_day_last10'] = {d: rowcount[d] for d in use[-10:]}
rc = pd.Series(rowcount); res['data']['rows_per_day_min_max_window'] = [int(rc.min()), int(rc.max())]

# 面板（只用個股 [1-9]ddd、上市或上櫃；同代號留最後一筆）
recs = []
for t, d in enumerate(use):
    df = frames[d]
    df = df[df['stock_id'].str.fullmatch(r'[1-9]\d{3}') & df['market'].isin(['twse', 'tpex'])].drop_duplicates('stock_id', keep='last')
    df = df.assign(t=t)
    recs.append(df[['t', 'stock_id', 'name', 'market', 'open', 'close', 'amount', 'change']])
L = pd.concat(recs)
for c in ('open', 'close', 'amount', 'change'):
    L[c] = pd.to_numeric(L[c], errors='coerce')
codes = np.array(sorted(L['stock_id'].unique())); ci = {c: j for j, c in enumerate(codes)}
T, N = len(use), len(codes)
def mat(col, fill=np.nan, dtype=float):
    A = np.full((T, N), fill, dtype=dtype); A[L['t'].to_numpy(), L['stock_id'].map(ci).to_numpy()] = L[col].to_numpy(); return A
O, C, AMT, CHG = mat('open'), mat('close'), mat('amount'), mat('change')
MKT = np.zeros((T, N), np.int8); MKT[L['t'].to_numpy(), L['stock_id'].map(ci).to_numpy()] = np.where(L['market'].to_numpy() == 'twse', 1, 2)
NAME = {}
last = L[L['t'] == T - 1].set_index('stock_id')['name'].to_dict()
names_sig = L[L['t'] == iS - (len(DAYS) - len(use)) - 0].set_index('stock_id')['name'].to_dict() if False else None
tS = use.index(SIG); tE = use.index(EXE) if EXE in use else None
names_sig = L[L['t'] == tS].set_index('stock_id')['name'].to_dict()
has = np.isfinite(C) & (C > 0)
ref = C - CHG; ref[~(ref > 0)] = np.nan
R = C / ref - 1
bad = np.abs(R) > 0.5
R[bad] = np.nan
CF = pd.DataFrame(C).ffill().to_numpy(); prevC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
R = np.where(np.isfinite(R), R, np.where(has, C / prevC - 1, 0.0)); R = np.where(np.isfinite(R), R, 0.0)
G = np.cumprod(1 + R, axis=0)
MA60 = pd.DataFrame(G).rolling(60, min_periods=60).mean().to_numpy()
RET20 = G / np.vstack([np.full((20, N), np.nan), G[:-20]]) - 1
listed = MKT > 0
amt = np.where(listed, np.nan_to_num(AMT), np.nan)
V60 = pd.DataFrame(amt).rolling(60, min_periods=60).mean().to_numpy()
not91 = np.array([not c.startswith('91') for c in codes])
innov = np.array([bool(INNOV.search(str(names_sig.get(c, '')).strip())) for c in codes])
t = tS
univ = (MKT[t] == 1) & has[t] & (C[t] >= 10) & (V60[t] >= 1e7) & not91 & ~innov
univ_noinnov = (MKT[t] == 1) & has[t] & (C[t] >= 10) & (V60[t] >= 1e7) & not91
above = G[t] > MA60[t]
dist = G[t] / MA60[t] - 1

# ---------------------------------------------------------------- 2. 營收與 SURGE
rv = pd.read_csv(REV, dtype={'code': str}, parse_dates=['month'])
rv['pri'] = (rv['market'] != 'sii').astype(int)
rv = rv.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
Rv = rv.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
Rv.index = pd.PeriodIndex(Rv.index, freq='M')
rev_last = str(Rv.index.max())
tsd_rev = ls(TSD, 'data/mops/revenue_hist')
mops_b = cat(TSD, [tsd_rev[f'{RM}_twse.csv'], tsd_rev[f'{RM - 1}_twse.csv']])
mops = pd.read_csv(io.BytesIO(mops_b[tsd_rev[f'{RM}_twse.csv']]), dtype={'stock_id': str})
mops_prev = pd.read_csv(io.BytesIO(mops_b[tsd_rev[f'{RM - 1}_twse.csv']]), dtype={'stock_id': str})
seen_ls = ls(TSD, f'data/meta/revenue_seen/{RM}')
seen_b = cat(TSD, list(set(seen_ls.values())))
seen_first = {}; seen_rev = {}
for fn in sorted(seen_ls):
    ts = pd.Timestamp(fn[:17].replace('T', ' ')[:11] + fn[11:13] + ':' + fn[13:15] + ':' + fn[15:17], tz='UTC')
    s = pd.read_csv(io.BytesIO(seen_b[seen_ls[fn]]), dtype={'stock_id': str})
    for _, r in s.iterrows():
        if r['market'] == 'sii':
            seen_first.setdefault(r['stock_id'], ts); seen_rev.setdefault(r['stock_id'], r['revenue'])
cut = pd.Timestamp(CUTOFF)
mops['rev'] = pd.to_numeric(mops['當月營收'], errors='coerce')
mops = mops.drop_duplicates('stock_id', keep='last')
seen_ok = {c for c, ts in seen_first.items() if ts < cut}
after_cut = sorted(c for c in mops['stock_id'] if c in seen_first and seen_first[c] >= cut)
never_seen = sorted(c for c in mops['stock_id'] if c not in seen_first)
seen_vs_head = sorted(c for c in mops['stock_id'] if c in seen_rev and abs(float(seen_rev[c]) - float(mops.set_index('stock_id').loc[c, 'rev'])) > 0.5)
# 單位與重疊月份一致性：2026-09 彙總表 vs tw_revenue
mp = mops_prev.drop_duplicates('stock_id', keep='last').set_index('stock_id')['當月營收'].astype(float)
tw9 = rv[(rv['month'] == (RM - 1).to_timestamp()) & (rv['market'] == 'sii')].set_index('code')['revenue_k_twd']
both = mp.index.intersection(tw9.index)
res['revenue'] = dict(tw_revenue_last_month=rev_last, mops_2026_10_rows=int(len(mops)), mops_2026_10_blob=tsd_rev[f'{RM}_twse.csv'],
                      tw_revenue_2026_09_sii=int(len(tw9)), coverage_vs_prev_sii=round(len(mops) / len(tw9), 4),
                      used_with_cutoff=int(len([c for c in mops['stock_id'] if c in seen_ok])),
                      prev_sii_codes_present=int(len(set(tw9.index) & set(mops['stock_id']))),
                      prev_sii_codes_present_cutoff=int(len(set(tw9.index) & set(mops['stock_id']) & seen_ok)),
                      coverage_intersect=round(len(set(tw9.index) & set(mops['stock_id'])) / len(tw9), 4),
                      coverage_intersect_cutoff=round(len(set(tw9.index) & set(mops['stock_id']) & seen_ok) / len(tw9), 4),
                      prev_sii_missing_cutoff=sorted(set(tw9.index) - (set(mops['stock_id']) & seen_ok)),
                      coverage_cutoff_vs_prev_sii=round(len([c for c in mops['stock_id'] if c in seen_ok]) / len(tw9), 4),
                      excluded_after_cutoff=after_cut, never_seen_in_meta=never_seen, seen_value_differs_from_head=seen_vs_head,
                      overlap_2026_09=dict(both=int(len(both)), equal=int((np.abs(mp.loc[both] - tw9.loc[both]) < 0.5).sum()),
                                           diff_codes=sorted(both[np.abs(mp.loc[both] - tw9.loc[both]) >= 0.5].tolist())))


def surge_for(rev_supp):
    R2 = Rv.copy()
    R2.loc[RM] = np.nan
    R2 = R2.reindex(pd.period_range(R2.index.min(), RM, freq='M'))
    s = pd.Series(rev_supp, dtype=float)
    R2.loc[RM, s.index.intersection(R2.columns)] = s[s.index.intersection(R2.columns)]
    extra = s.index.difference(R2.columns)
    R2 = R2.where(R2 > 0)
    D = (R2 - R2.shift(12)).loc[:RM]
    W = D.loc[RM - 24:RM - 1]
    assert len(W) == 24
    A = W.to_numpy(float); x = D.loc[RM].to_numpy(float)
    full = np.isfinite(A).all(axis=0)
    mu = A.mean(axis=0); sd = np.sqrt(((A - mu) ** 2).sum(axis=0) / 23); sc = np.abs(A).mean(axis=0)
    pos = full & np.isfinite(sd) & (sd > 1e-9 * (sc + 1e-12))
    z = np.where(pos & np.isfinite(x), (x - mu) / np.where(pos, sd, 1.0), np.nan)
    out = pd.Series(z, index=D.columns)
    n_sigma0 = int((full & np.isfinite(x) & ~pos).sum())
    return out, n_sigma0, list(extra)

supp_cut = mops[mops['stock_id'].isin(seen_ok)].set_index('stock_id')['rev']
supp_all = mops.set_index('stock_id')['rev']
SU_cut, n0_cut, extra = surge_for(supp_cut)
SU_all, n0_all, _ = surge_for(supp_all)
res['revenue']['n_sigma0_cut'] = n0_cut
res['revenue']['mops_codes_without_history'] = len(extra)


def build(SU):
    s = np.array([SU.get(c, np.nan) for c in codes], float)
    okS = univ & np.isfinite(s)
    ma = okS & above
    idx = np.flatnonzero(ma)
    order = sorted(idx, key=lambda j: (-s[j], codes[j]))
    rank = {codes[j]: i + 1 for i, j in enumerate(order)}
    top20 = [codes[j] for j in order[:20]]
    f1 = [codes[j] for j in order[:10] if RET20[t, j] > 0.10]
    return dict(s=s, n_univ=int(univ.sum()), n_univ_surge=int(okS.sum()), n_ma=int(ma.sum()), rank=rank, top20=top20, f1=f1, order=order)

B = build(SU_cut); BA = build(SU_all)
res['list'] = dict(n_univ=B['n_univ'], n_univ_surge=B['n_univ_surge'], n_ma=B['n_ma'], has_signal=B['n_univ_surge'] >= 50,
                   top20=B['top20'], f1=B['f1'], surge_10th=r3(B['s'][B['order'][9]]), surge_20th=r3(B['s'][B['order'][19]]),
                   nocutoff=dict(n_univ_surge=BA['n_univ_surge'], n_ma=BA['n_ma'], top20_same=BA['top20'] == B['top20'], f1_same=BA['f1'] == B['f1']),
                   n_innov_in_pool_excluded=int((univ_noinnov & innov).sum()))

# ---------------------------------------------------------------- 3. 對名單檔
LJ = json.load(open(LISTS / '2026-11.json', encoding='utf-8'))
chk('名單 訊號日', LJ['sig_date'], SIG, LJ['sig_date'] == SIG, 'lists/2026-11.json')
chk('名單 成交日', LJ['exec_date'], EXE, LJ['exec_date'] == EXE, 'lists/2026-11.json')
chk('名單 營收月份', LJ['rev_month'], str(RM), LJ['rev_month'] == str(RM), 'lists/2026-11.json')
chk('股池檔數', LJ['n_univ'], B['n_univ'], LJ['n_univ'] == B['n_univ'], 'lists/2026-11.json')
chk('股池有 SURGE 檔數（截止版）', LJ['n_univ_surge'], B['n_univ_surge'], LJ['n_univ_surge'] == B['n_univ_surge'], 'lists/2026-11.json')
chk('L_MA 檔數', LJ['n_ma'], B['n_ma'], LJ['n_ma'] == B['n_ma'], 'lists/2026-11.json')
chk('前 20 名代號與順序', ','.join(x['code'] for x in LJ['top20']), ','.join(B['top20']), [x['code'] for x in LJ['top20']] == B['top20'], 'lists/2026-11.json')
f1_site = [x['code'] if isinstance(x, dict) else str(x) for x in LJ['f1']]
chk('F1 名單', ','.join(f1_site), ','.join(B['f1']), f1_site == B['f1'], 'lists/2026-11.json')
maxd = {'surge': 0, 'ret20': 0, 'dist_ma60': 0, 'v60_rel': 0, 'close': 0}
rows20 = []
for x in LJ['top20']:
    j = ci[x['code']]
    mine = dict(surge=B['s'][j], ret20=RET20[t, j], dist_ma60=dist[j], v60=V60[t, j], close=C[t, j], rank=B['rank'].get(x['code']))
    rows20.append(dict(code=x['code'], name=x['name'], rank_site=x['rank_ma'], rank=mine['rank'], surge_site=x['surge'], surge=mine['surge'],
                       ret20_site=x['ret20'], ret20=mine['ret20'], dist_site=x['dist_ma60'], dist=mine['dist_ma60'], v60_site=x['v60'], v60=mine['v60'],
                       close_site=x['close'], close=mine['close'], above=bool(above[j]), name_now=names_sig.get(x['code'])))
    maxd['surge'] = max(maxd['surge'], abs(mine['surge'] - x['surge'])); maxd['ret20'] = max(maxd['ret20'], abs(mine['ret20'] - x['ret20']))
    maxd['dist_ma60'] = max(maxd['dist_ma60'], abs(mine['dist_ma60'] - x['dist_ma60'])); maxd['v60_rel'] = max(maxd['v60_rel'], abs(mine['v60'] / x['v60'] - 1))
    maxd['close'] = max(maxd['close'], abs(mine['close'] - x['close']))
res['top20_rows'] = rows20
for k, v in maxd.items():
    chk(f'前 20 名 {k} 最大差', 0.0, float(v), v < (1e-6 if k == 'v60_rel' else TOL3), 'lists/2026-11.json', '逐檔對到小數點後 3 位')
chk('2026-11 有訊號（股池有 SURGE ≥ 50）', LJ['status'], '有訊號' if B['n_univ_surge'] >= 50 else '資料不足', LJ['status'] == '有訊號' and B['n_univ_surge'] >= 50, 'lists/2026-11.json')
chk('不截止版 前 20 名與 F1 相同', None, f"前20相同={BA['top20'] == B['top20']}，F1相同={BA['f1'] == B['f1']}", BA['top20'] == B['top20'] and BA['f1'] == B['f1'], '自算', '時點檢查')
chk('2026-10 營收涵蓋率（截止版：2026-09 上市營收公司中 2026-10 已有者）', 0.9839, res['revenue']['coverage_intersect_cutoff'],
    res['revenue']['coverage_intersect_cutoff'] >= 0.95 and abs(res['revenue']['coverage_intersect_cutoff'] - 0.9839) < 0.01, 'lists/2026-11.json meta.rev', '門檻 95%；網站值為 s18 用的較早版本彙總表（1,084 筆）')
# 敏感度：2026-09 營收兩個來源不同的公司改用彙總表數字
alt = Rv.copy()
for c in res['revenue']['overlap_2026_09']['diff_codes']:
    if c in alt.columns:
        alt.loc[RM - 1, c] = mp.loc[c]
_Rv0 = Rv; Rv = alt
SU_alt, _, _ = surge_for(supp_cut); BB = build(SU_alt); Rv = _Rv0
res['list']['alt_2026_09_mops'] = dict(top20_same=BB['top20'] == B['top20'], f1_same=BB['f1'] == B['f1'],
    max_surge_change_top20=float(max(abs(SU_alt.get(c) - SU_cut.get(c)) for c in B['top20'])))
chk('敏感度：2026-09 營收 8 家不同改用彙總表', None, f"前20相同={BB['top20'] == B['top20']}，F1相同={BB['f1'] == B['f1']}", BB['top20'] == B['top20'] and BB['f1'] == B['f1'], '自算')

# E0：上月持股（回測持股縮放）
BH = pd.read_csv(LISTS / '2026-11_backtest_holdings.csv', dtype={'代號': str}, encoding='utf-8-sig')
held = BH[BH['區塊'].str.startswith('持股_')]
e0 = []
for _, r in held.iterrows():
    c = r['代號']; j = ci.get(c)
    rk = B['rank'].get(c)
    if j is None or not univ[j]:
        why = '不在股池'
    elif not np.isfinite(B['s'][j]):
        why = '沒有 SURGE'
    elif not above[j]:
        why = '跌破季線'
    elif rk is None or rk > 20:
        why = f'掉出前 20 名（第 {rk} 名）'
    else:
        why = f'續抱（第 {rk} 名）'
    act = '續抱' if why.startswith('續抱') else '賣出'
    site_act = '續抱' if r['區塊'].startswith('持股_續抱') else '賣出'
    e0.append(dict(code=c, site=site_act, mine=act, why=why, surge=r3(B['s'][j]), surge_site=r['SURGE'], ret20=r3(RET20[t, j] * 100), ret20_site=r['RET20pct'],
                   close=float(C[t, j]), close_site=float(r['收盤']), rank=rk, rank_site=None if pd.isna(r['名次']) else int(r['名次'])))
    chk(f'E0 {c} 動作', site_act, act, site_act == act, 'lists/2026-11_backtest_holdings.csv', why)
res['e0'] = e0
e0d = max(abs(x['surge'] - x['surge_site']) for x in e0)
chk('E0 持股 SURGE 最大差', 0.0, e0d, e0d < TOL3, 'lists/2026-11_backtest_holdings.csv')

# ---------------------------------------------------------------- 4. 網站資料
def site(n):
    return json.load(open(SITE / 'data' / f'{n}.json', encoding='utf-8'))['rows']
T20 = site('top20')
ok = all(a['代號'] == b and a['名次'] == i + 1 for i, (a, b) in enumerate(zip(T20, B['top20']))) and len(T20) == 20
chk('網站 top20 代號與名次', ','.join(a['代號'] for a in T20), ','.join(B['top20']), ok, 'site/data/top20.json')
md = dict(surge=0, ret=0, dist=0, close=0); sel_ok = True
for a in T20:
    j = ci[a['代號']]
    md['surge'] = max(md['surge'], abs(round(B['s'][j], 3) - a['SURGE']))
    md['ret'] = max(md['ret'], abs(round(RET20[t, j] * 100, 1) - a['前20日漲幅%']))
    md['dist'] = max(md['dist'], abs(round(dist[j] * 100, 1) - a['距季線%']))
    md['close'] = max(md['close'], abs(C[t, j] - a['收盤']))
    sel_ok &= (a['入選'] == '是') == (a['代號'] in B['f1'])
for k, v in md.items():
    chk(f'網站 top20 {k} 最大差（依網站位數四捨五入）', 0.0, float(v), v < 1e-9, 'site/data/top20.json')
chk('網站 top20 入選欄＝F1', None, ','.join(B['f1']), sel_ok, 'site/data/top20.json')

# 價格（11/12 開盤、11/13 收盤）
def px(d):
    tt = use.index(d); return {c: (O[tt, ci[c]], C[tt, ci[c]]) for c in codes}, tt
e50 = {}
for d in use[-12:]:
    df = frames[d]; z = df[df['stock_id'] == '0050']
    e50[d] = (float(z['open'].iloc[0]), float(z['close'].iloc[0]), float(z['change'].iloc[0])) if len(z) else None
LAST = use[-1]
pE, tE = px(EXE); pL, tL = px(LAST)
A = site('actions')
for a in A:
    c = a['代號']; j = ci[c]
    exp_ret = round((C[tL, j] / C[t, j] - 1) * 100, 2)
    chk(f'網站 actions {c} 訊號日收盤／最新收盤', f"{a['訊號日收盤']}／{a['最新收盤']}", f'{C[t, j]}／{C[tL, j]}', a['訊號日收盤'] == C[t, j] and a['最新收盤'] == C[tL, j], 'site/data/actions.json')
    chk(f'網站 actions {c} SURGE', a['SURGE'], round(B['s'][j], 3) if np.isfinite(B['s'][j]) else None, abs(a['SURGE'] - round(B['s'][j], 3)) < 1e-9, 'site/data/actions.json')
    chk(f'網站 actions {c} 本月以來%（對訊號日收盤）', a['本月以來%'], exp_ret, abs(a['本月以來%'] - exp_ret) < 0.005, 'site/data/actions.json')
H = site('holdings')
state = pd.read_csv(SITE / 'state' / 'holdings_after_2026-11.csv', dtype={'代號': str}).set_index('代號')['股數'].astype(float).to_dict()
for h in H:
    c = h['代號']
    if c == '0050':
        o, cl = e50[EXE][0], e50[LAST][1]; sc = e50[SIG][1] if SIG in e50 else None
    else:
        j = ci[c]; o, cl, sc = O[tE, j], C[tL, j], C[t, j]
    chk(f'網站 holdings {c} 股數', h['股數'], state.get(c), h['股數'] == state.get(c), 'site/state/holdings_after_2026-11.csv')
    chk(f'網站 holdings {c} 起始價（{EXE} 開盤）', h['起始價'], o, h['起始價'] == o, '官方日 K')
    chk(f'網站 holdings {c} 最新收盤（{LAST}）', h['最新收盤'], cl, h['最新收盤'] == cl, '官方日 K')
    since_start = round((cl / o - 1) * 100, 2)
    since_sig = round((cl / sc - 1) * 100, 2)
    chk(f'網站 holdings {c} 本月以來%', h['本月以來%'], since_start, abs(h['本月以來%'] - since_start) < 0.005, '自算（對起始價＝成交日開盤）',
        f'網站以訊號日收盤為基準（{since_sig}%，計算本身正確），和「起始價」欄與 live.json 的起點不一致', warn=abs(h['本月以來%'] - since_sig) < 0.005)
res['holdings_since_start'] = {h['代號']: dict(site=h['本月以來%'], vs_open=round(((e50[LAST][1] / e50[EXE][0]) if h['代號'] == '0050' else (C[tL, ci[h['代號']]] / O[tE, ci[h['代號']]])) * 100 - 100, 2)) for h in H}

# 換股重算（回測規則、11/12 開盤）：賣出得款、台玻股數、0050 股數、剩餘現金
FEE, TAX, ETAX, SLIP, MINF = 0.001425 * 0.28, 0.003, 0.001, 0.003, 20.0
fee = lambda v: max(MINF, v * FEE)
bh = {r['代號']: int(r['建議股數']) for _, r in BH.iterrows() if str(r['區塊']).startswith('持股_')}
e50_before = 3053; base_sh = 1995
o50 = e50[EXE][0]; c50s = e50[SIG][1]
sells = [x['code'] for x in e0 if x['mine'] == '賣出']; keeps = [x['code'] for x in e0 if x['mine'] == '續抱']
proceeds = 0.0
for c in sells:
    v = bh[c] * O[tE, ci[c]] * (1 - SLIP); proceeds += v - fee(v) - v * TAX
nav_open = proceeds + sum(bh[c] * O[tE, ci[c]] for c in keeps) + e50_before * o50
sv = nav_open - base_sh * o50
tgt = sv / 10
p1802 = O[tE, ci['1802']]
sh1802 = int(tgt // (p1802 * (1 + SLIP)))
sh1802_simple = int(tgt // p1802)
cost1802 = sh1802 * p1802 * (1 + SLIP); cost1802 += fee(cost1802)
cash = proceeds - cost1802
sh50 = int(cash // (o50 * (1 + SLIP) * (1 + FEE))) if cash >= 20000 else 0
c50 = sh50 * o50 * (1 + SLIP); c50 += fee(c50)
cash_left = cash - c50
st_cost1802 = state['1802'] * p1802 * (1 + SLIP); st_cost1802 += fee(st_cost1802)
st_c50 = (state['0050'] - e50_before) * o50 * (1 + SLIP); st_c50 += fee(st_c50)
st_cash = proceeds - st_cost1802 - st_c50
res['rebalance_open'] = dict(proceeds_sells_open=round(proceeds), nav_open_est=round(nav_open), sat_value=round(sv), target_per_slot=round(tgt),
                             shares_1802_with_slip=sh1802, shares_1802_open_only=sh1802_simple, shares_1802_state=state['1802'],
                             e50_buy_shares=sh50, e50_total=e50_before + sh50, e50_total_state=state['0050'], cash_left_mine=round(cash_left),
                             cash_left_implied_by_state=round(st_cash),
                             nav_open_state_holdings=round(sum(state[c] * (o50 if c == '0050' else O[tE, ci[c]]) for c in state)))
chk('台玻股數（目標 ÷ 開盤，回測規則）', state['1802'], sh1802_simple, abs(state['1802'] - sh1802_simple) <= 3, '自算', warn=True, note=
    f'計入 0.3% 滑價為 {sh1802} 股；網站用 make_list 收盤估的目標 69,582 ÷ 開盤 80.7；回測以開盤淨值算目標')
chk('0050 總股數（11/12 開盤重算）', state['0050'], e50_before + sh50, abs(state['0050'] - (e50_before + sh50)) <= 30, '自算', warn=True, note=
    f'網站沿用訊號日收盤估的 2,912 股；以開盤價與實際賣出價重算為 {sh50} 股；網站股數隱含剩餘現金 {round(st_cash):,} 元（live.json 未計入現金）')

# live.json
LV = site('live')
sh = state
base = sum(sh[c] * (o50 if c == '0050' else O[tE, ci[c]]) for c in sh)
live_mine = []
for d in use[use.index(EXE):]:
    tt = use.index(d)
    nav = sum(sh[c] * (e50[d][1] if c == '0050' else C[tt, ci[c]]) for c in sh)
    live_mine.append(dict(date=d, strat=round(nav / base * 1e6), e50=round(e50[d][1] / o50 * 1e6),
                          strat_with_cash=round((nav + st_cash) / (base + st_cash) * 1e6)))
for a, b in zip(LV, live_mine):
    chk(f"網站 live {a['日期']} 策略淨值", a['策略'], b['strat'], a['日期'] == b['date'] and abs(a['策略'] - b['strat']) <= 1, '自算',
        f"含剩餘現金為 {b['strat_with_cash']:,}")
    chk(f"網站 live {a['日期']} 0050", a['0050'], b['e50'], abs(a['0050'] - b['e50']) <= 1, '自算')
chk('網站 live 天數', len(LV), len(live_mine), len(LV) == len(live_mine), '自算')
res['live'] = live_mine
res['live_base_value_at_open'] = round(base)

# status
STA = site('status')[0]
nxt10 = first_on_after('2026-12-10') if DAYS[-1] >= '2026-12-10' else None
res['status_note'] = '2026-12 交易日尚無日 K；下次訊號日以「12/10（四）為交易日」推算＝12/11（五），執行 12/14（一）'
for k, v in (('本月訊號日', SIG), ('本月執行日', EXE), ('營收月份', str(RM)), ('資料最後一天', LAST), ('價格日期', LAST), ('0050最新收盤', e50[LAST][1]),
             ('下次訊號日', '2026-12-11'), ('下次執行日', '2026-12-14')):
    chk(f'網站 status {k}', STA[k], v, STA[k] == v, 'site/data/status.json', '下次日期為推算（未知 12 月假日）' if k.startswith('下次') else '')

# ---------------------------------------------------------------- 5. 資料品質
dq = {}
focus = sorted(set(B['top20']) | set(bh) | {'1802'})
exr = []
for c in focus:
    j = ci[c]
    for tt in range(T - 60, T):
        if has[tt, j] and np.isfinite(CF[tt - 1, j]) and np.isfinite(ref[tt, j]) and abs(ref[tt, j] / CF[tt - 1, j] - 1) > 0.002:
            exr.append(dict(code=c, date=use[tt], prev_close=float(CF[tt - 1, j]), ref=float(ref[tt, j]), adj=round(float(ref[tt, j] / CF[tt - 1, j] - 1), 4)))
dq['ex_rights_last60_focus'] = exr
dq['bad_ret_window_all'] = int(bad.sum())
dq['bad_ret_focus'] = [dict(code=codes[j], date=use[tt]) for tt, j in zip(*np.nonzero(bad)) if codes[j] in focus]
nt = []
for c in focus:
    j = ci[c]; z = (listed[T - 60:, j] & ~has[T - 60:, j]).sum(); nl = (~listed[T - 60:, j]).sum()
    if z or nl:
        nt.append(dict(code=c, listed_no_trade_days=int(z), not_in_file_days=int(nl)))
dq['suspension_focus_last60'] = nt
dq['e50_last12'] = e50
dq['held_ex_rights_after_signal'] = [x for x in exr if x['date'] > SIG and x['code'] in set(state)]
# 0050 分割（2025-06）：用官方日 K 檢查
sp = {}
for d in ['2025-06-10', '2025-06-11', '2025-06-17', '2025-06-18']:
    b = cat(RAW, [raw_ls[d + '.csv']])[raw_ls[d + '.csv']]
    df = pd.read_csv(io.BytesIO(b), dtype={'stock_id': str}); z = df[df['stock_id'] == '0050']
    sp[d] = None if not len(z) else dict(open=float(z['open'].iloc[0]), close=float(z['close'].iloc[0]), change=float(z['change'].iloc[0]))
dq['e50_split_2025_06'] = sp
if sp['2025-06-18'] and sp['2025-06-10']:
    refp = sp['2025-06-18']['close'] - sp['2025-06-18']['change']
    dq['e50_split_ratio'] = round(sp['2025-06-10']['close'] / refp, 4)
res['data_quality'] = dq
chk('0050 2025-06 分割（官方日 K 參考價）', '1:4', dq.get('e50_split_ratio'), abs(dq.get('e50_split_ratio', 0) - 4) < 0.01, '官方日 K',
    '6/11～6/17 無 0050 日 K（停牌），6/18 參考價＝6/10 收盤 ÷ 4；以漲跌價差算日報酬不受分割影響')
chk('日 K 重疊日 blob 相同（raw 封存 vs 新 clone）', 0, len(blob_diff), not blob_diff, 'git ls-tree')
chk('2025-07 起平日缺檔（應全為國定假日）', None, ','.join(miss), True, 'git ls-tree', '人工判讀見報告')
chk('持股成交日後除權息（影響 live 未還原收盤）', 0, len(dq['held_ex_rights_after_signal']), not dq['held_ex_rights_after_signal'], '官方日 K')

# ---------------------------------------------------------------- 6. yearly／risk／nav
nv = pd.read_csv(NAVS, parse_dates=['date']).set_index('date')
YR = site('yearly')
cols = {'現行策略': 'E0|W10_70', 'XQ版': 'XT|W10_70', '0050': '0050'}
def yret(s):
    out = {}
    ye = s.groupby(s.index.year).last()
    for y in ye.index:
        prev = s.iloc[0] if y == s.index[0].year else ye.loc[y - 1]
        out[int(y)] = round((ye.loc[y] / prev - 1) * 100, 1)
    return out
ymine = {k: yret(nv[v]) for k, v in cols.items()}
BT = json.load(open(HERE / 'audit_backtest.json', encoding='utf-8'))
yb = {int(k): v for k, v in BT['yearly_0050'].items()}; ys = {int(k): v for k, v in BT['yearly_strategy'].items()}
for r in YR:
    y = r['年']
    chk(f"網站 yearly {y} 現行策略", r['現行策略'], ys[y], abs(r['現行策略'] - ys[y]) < 0.051 and abs(r['現行策略'] - ymine['現行策略'][y]) < 0.051, 's19 navs.csv.gz E0|W10_70＋本稽核重跑引擎')
    chk(f"網站 yearly {y} XQ版", r['XQ版'], ymine['XQ版'][y], abs(r['XQ版'] - ymine['XQ版'][y]) < 0.051, 's19 navs.csv.gz XT|W10_70')
    d = abs(r['0050'] - yb[y])
    chk(f"網站 yearly {y} 0050", r['0050'], yb[y], d < 0.051, '引擎 0050 含息還原價（精確）', f"網站值＝navs.csv.gz 0050 欄（整數四捨五入）算出的 {ymine['0050'][y]}", warn=d < 0.15)
# nav.json 對 navs
NV = site('nav')
nvd = nv.copy()
mx = 0.0; bad_order = []; holiday_lbl = 0; future = []
prev = ''
for r in NV:
    if r['日期'] < prev:
        bad_order.append(f"{prev}→{r['日期']}")
    prev = r['日期']
    d = pd.Timestamp(r['日期'])
    if d > nvd.index[-1]:
        future.append(r['日期'])
    dd = nvd.index[nvd.index <= d][-1]
    if dd != d:
        holiday_lbl += 1
    mx = max(mx, abs(r['現行策略'] - nvd.loc[dd, 'E0|W10_70']), abs(r['XQ版'] - nvd.loc[dd, 'XT|W10_70']))
res['nav_json'] = dict(n=len(NV), holiday_friday_labels=holiday_lbl, after_backtest_end=future, order_breaks=bad_order)
chk('網站 nav 週淨值對 navs.csv.gz 最大差（元；假日週五標籤對到前一交易日）', 0, mx, mx < 1, 's19 navs.csv.gz')
chk('網站 nav 日期遞增、不超過回測最後一天', '', '；'.join(future + bad_order), not future and not bad_order, 'site/data/nav.json',
    f'另有 {holiday_lbl} 個週標籤是休市的週五（值＝前一交易日，不算錯）', warn=True)
# risk
RK = site('risk')
txt = README.read_text(encoding='utf-8'); man = OPMAN.read_text(encoding='utf-8')
def section(doc, n):
    m = re.search(rf'^## {n}\. .*?(?=^## \d+\. |\Z)', doc, re.S | re.M); return m.group(0) if m else ''
s = nv['E0|W10_70']; e = nv['0050']
d19 = s.index[s.index < '2019-01-01'][-1]
yrs = (s.index[-1] - d19).days / 365.25
c19 = (s.iloc[-1] / s.loc[d19]) ** (1 / yrs) - 1
mdd = (s / s.cummax() - 1).min(); mdd_t = (s / s.cummax() - 1).idxmin()
emdd = (e / e.cummax() - 1).min(); emdd_t = (e / e.cummax() - 1).idxmin()
r12 = (s / s.shift(252) - 1) - (e / e.shift(252) - 1)
r12 = r12.dropna()
lag_max = -r12.min() * 100; lag_frac = (r12 < 0).mean() * 100
tr = pd.read_csv(S19TR, dtype={'code': str}); tr = tr[(tr['run'] == 'E0|W10_70') & (tr['why'] != 'open_end')]
top5 = tr['pnl'].nlargest(5).sum() / tr['pnl'].sum() * 100
win = (tr['ret'] > 0).mean() * 100
calc = {'回撤_單一路徑': round(mdd * 100, 1), '0050_回撤': round(BT['bench_maxdd_precise'] * 100, 1), '年化_全部': round(c19 * 100, 2), '落後0050最多': round(BT['lag12m_252d_max_pp'], 1),
        '落後時間比例': round(BT['lag12m_252d_frac'] * 100), '最賺5筆占比': round(top5), '勝率': round(win, 1), '賺的平均': round(tr.loc[tr['ret'] > 0, 'ret'].mean() * 100, 2),
        '賠的平均': round(tr.loc[tr['ret'] <= 0, 'ret'].mean() * 100, 2)}
res['risk_calc'] = dict(calc, n_closed_trades_s19=int(len(tr)), mdd_trough=str(mdd_t.date()), e50_mdd_trough=str(emdd_t.date()))
for r in RK:
    src = r['出處']; m = re.search(r'第 (\d+) 節', src)
    doc = man if '操作說明' in src else txt
    sec = section(doc, m.group(1)) if m else ''
    v = r['數值']
    vs = [f'{abs(v):.2f}', f'{abs(v):.1f}', f'{abs(v):g}', f'{abs(v):.0f}']
    found = any(x in sec for x in vs)
    c = calc.get(r['項目'])
    okc = True if c is None else abs(c - v) <= (0.051 if isinstance(v, float) and r['單位'] != '%' or r['項目'] not in ('落後時間比例', '最賺5筆占比') else 1.0) + 1e-9
    if c is not None and r['項目'] in ('落後時間比例', '最賺5筆占比'):
        okc = abs(c - v) <= 1.0
    elif c is not None:
        okc = abs(c - v) <= 0.051 + 1e-9
    chk(f"網站 risk {r['項目']}", v, c if c is not None else ('出處段落有此數' if found else '出處段落找不到此數'), found and okc, src,
        '' if c is None else '另由 s19 navs.csv.gz／trades.csv.gz 重算')
json.dump(res, open(HERE / 'audit_list.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
nf = [c for c in res['checks'] if c['verdict'] != '通過']
print('檢查', len(res['checks']), '不通過', len(nf))
for c in nf:
    print(' ✗', c['item'], '| 網站', c['site'], '| 重算', c['recomputed'], '|', c['note'])

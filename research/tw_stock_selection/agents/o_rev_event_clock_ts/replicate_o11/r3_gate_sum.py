import pandas as pd, numpy as np, json, os, subprocess, gzip, io
HERE = os.path.dirname(os.path.abspath(__file__))
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
SCR = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/'
cr = pd.read_parquet(P + 'close_raw.parquet')
days = list(cr.index[cr.notna().sum(axis=1) > 0])
hol = set(json.load(open(SCR + 'extras/tw_holidays.json'))['years']['2026'])
d = days[-1] + pd.Timedelta(days=1)
while d <= pd.Timestamp('2027-01-31'):
    if d.weekday() < 5 and str(d.date()) not in hol: days.append(d)
    d += pd.Timedelta(days=1)
cal = pd.DatetimeIndex(days); pos = {x: i for i, x in enumerate(cal)}
def aday(T):
    if pd.isna(T): return pd.NaT
    T = pd.Timestamp(T).tz_convert('Asia/Taipei')
    dd = pd.Timestamp(T.date()) - pd.Timedelta(days=1 if T.hour < 9 else 0)
    return cal[cal.searchsorted(dd)]
def load(fn):
    x = pd.read_csv(os.path.join(HERE, fn), dtype={'code': str})
    for c in ['T', 'Tprev']: x[c] = pd.to_datetime(x[c], utc=True).dt.tz_convert('Asia/Taipei')
    return x
tasi, up = load('fs_tasi.csv.gz'), load('fs_up.csv.gz')
den0 = pd.read_csv(os.path.join(HERE, 'denominator.csv'), dtype=str)
res_h = pd.read_csv('../data/firstseen_hist.csv', dtype=str)
out = {}
def run(den, tag):
    D = den.merge(tasi.add_prefix('t_').rename(columns={'t_code': 'code', 't_ym': 'rev_month'}), on=['code', 'rev_month'], how='left')
    D = D.merge(up.add_prefix('u_').rename(columns={'u_code': 'code', 'u_ym': 'rev_month'}), on=['code', 'rev_month'], how='left')
    N = len(D); r = dict(N=N, anchors=D.anchor.nunique())
    D['t_seen_all'] = D.t_reason.notna()
    D['t_old'] = D.t_file.fillna('').str.extract(r'(\d{6})')[0].fillna('999999') < '202404'
    D['t_seen'] = D.t_seen_all & ~D.t_old
    D['t_valid'] = D.t_valid.fillna(False).astype(bool); D['u_valid'] = D.u_valid.fillna(False).astype(bool)
    r['tasi_seen_pct_excl_old39'] = D.t_seen.mean() * 100; r['tasi_seen_pct_all'] = D.t_seen_all.mean() * 100
    r['tasi_valid_pct'] = D.t_valid.mean() * 100
    r['tasi_reasons_excl_old'] = D[D.t_seen].t_reason.value_counts().to_dict()
    r['tasi_reasons_all'] = D[D.t_seen_all].t_reason.value_counts().to_dict()
    win = (D.anchor >= '2024-06')
    r['tasi_window_valid_pct'] = D[win].t_valid.mean() * 100; r['window_share_pct'] = win.mean() * 100
    r['up_seen_pct'] = D.u_reason.notna().mean() * 100
    r['up_reason_by_file'] = {f'{a}|{b}': int(v) for (a, b), v in D[D.u_reason.notna()].groupby(['u_file', 'u_reason']).size().items()}
    r['up_valid_n'] = int(D.u_valid.sum()); r['up_valid_pct'] = D.u_valid.mean() * 100
    D['any'] = D.t_valid | D.u_valid
    r['union_pct'] = D['any'].mean() * 100; r['tasi_only_pct'] = D.t_valid.mean() * 100
    r['union_minus_tasi_pp'] = r['union_pct'] - r['tasi_only_pct']
    D['yr'] = D.anchor.str[:4]
    r['by_year_tasi'] = (D.groupby('yr').t_valid.mean() * 100).round(1).to_dict()
    r['by_year_union'] = (D.groupby('yr')['any'].mean() * 100).round(1).to_dict()
    # 交叉核對
    B = D[D.t_valid & D.u_valid].copy(); r['overlap_n'] = len(B)
    inter = (B.t_T > B.u_Tprev) & (B.u_T > B.t_Tprev)
    r['overlap_interval_agree_pct'] = inter.mean() * 100
    r['overlap_same_aday_pct'] = (B.t_T.map(aday) == B.u_T.map(aday)).mean() * 100
    # 只算兩者都有 T（seen）的重疊另報
    B2 = D[D.t_reason.notna() & D.u_reason.notna() & D.t_valid & D.u_valid]
    out_x = D
    return r, D
# MOPS ann_date：取 rev_hist HEAD 版
blob = subprocess.run(['git', '-C', os.path.join(HERE, 'raw/screener'), 'show', 'HEAD:data/extras/rev_hist.csv.gz'], capture_output=True).stdout
rh = pd.read_csv(io.BytesIO(gzip.decompress(blob)), dtype={'code': str})
ann = rh.dropna(subset=['ann_date']).set_index(['code', 'ym']).ann_date
def mops(D, p):
    V = D[D[p + 'valid']].copy()
    V['ann'] = [ann.get((c, y)) for c, y in zip(V.code, V.rev_month)]
    V = V[V.ann.notna()]
    lo = V[p + 'Tprev'].map(aday); hi = V[p + 'T'].map(aday); a = pd.to_datetime(V.ann)
    ok = (a >= lo) & (a <= hi)
    return dict(n=len(V), agree=int(ok.sum()), pct=float(ok.mean() * 100) if len(V) else None)
# 主口徑：o8 + 研究員 2026-10 代碼；敏感度：o8 + 自算 2026-10 股池
den_res = pd.concat([den0[den0.anchor < '2026-10'], den0[den0.anchor == '2026-10'].iloc[:0],
                     res_h[res_h.anchor == '2026-10'][['anchor', 'rev_month', 'code', 'sig_day', 'deadline']]], ignore_index=True)
for tag, den in [('main_res202610', den_res), ('alt_my202610', den0)]:
    r, D = run(den, tag)
    r['mops_tasi'] = mops(D, 't_'); r['mops_up'] = mops(D, 'u_')
    out[tag] = r
    if tag == 'main_res202610': D.to_pickle(os.path.join(HERE, 'gateD.pkl'))
json.dump(out, open(os.path.join(HERE, 'gate_rep.json'), 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))

#!/usr/bin/env python3
"""S36 第 0 步（不算任何新規則的報酬）：營收首次可見時間的資料盤點 → data_check.json。
(1) 官方日 K 倉庫 tsd 的 data/meta/revenue_seen/（git show，只讀）；(2) Tasi 每日快照（o_rev_event_clock_ts 已整理）；
(3) 涵蓋月份、F1／L_MA 前 20 名覆蓋率、比官方訊號日早幾個交易日、各規則×θ 的早進場候選檔數；
(4) 現行（CUR，已知結果 48.99％）持股在早進場窗內的空名額（只數名額，不看報酬）。python3 -I -B prep.py"""
import sys, json, time, resource, subprocess
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s36lib
from s36lib import s10lib, s4lib, stlib, s17lib, COSTS

H = Path(__file__).resolve().parent
TSD = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsd')
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
chk = {}

# ---------------------------------------------------------------- (1) tsd revenue_seen
g = lambda *a: subprocess.run(['git', '-C', str(TSD), *a], capture_output=True, text=True).stdout
files = [f for f in g('ls-tree', '-r', '--name-only', 'HEAD', 'data/meta/revenue_seen/').split('\n') if f]
tsd = {}
for f in files:
    ym = f.split('/')[3]; ts = f.split('/')[4].replace('.csv', '')
    n = max(len(g('show', f'HEAD:{f}').strip().split('\n')) - 1, 0)
    tsd.setdefault(ym, []).append(dict(snapshot_utc=ts, n_new=n))
chk['source_tsd_revenue_seen'] = dict(
    head=g('log', '-1', '--format=%h %cI').strip(), n_files=len(files), months=sorted(tsd), n_months=len(tsd),
    per_month={k: dict(n_snapshots=len(v), n_codes=sum(x['n_new'] for x in v), first_snapshot=v[0]['snapshot_utc'], snapshots=v) for k, v in tsd.items()},
    precision='快照時間（UTC，檔名）；每次 make_list／排程執行才寫一次，間隔不固定（數小時到數天）。2026-09 第一個快照在 2026-10-09T19:03Z（_first），1,605 家全擠在第一檔，看不出 1～9 日誰先公告；2026-10 起較密（11-03～11-15 共 14 次）。上市（sii）＋上櫃（otc）都有，欄位 stock_id, market, part, revenue。',
    usable_months_for_backtest=1)

# ---------------------------------------------------------------- (2) 其他來源（既有研究盤點，引用）
chk['source_other'] = {
    'screener_rev_hist_ann_date': '只有 2026-10 一個月的實際公告日（BACKLOG 2026-11-15：6 日 228、7 日 291、8 日 902、12 日 239 家）',
    'o_rev_event_clock_ts_upstream(tw-stock-screener)': 'git 歷史從 2026-09-29 才開始；2024-04～2026-07 全是 rev_5y 首次加入（不能當公告時間），可用只有 2026-08／09',
    'o_rev_event_clock_ts_tasi(shohokuno10/Tasi 營收/YYYYMM.csv)': 'MOPS 上市營收表每日快照（git commit，約 18:16 台北）；營收月份 2024-04～2026-09；2024-04／05 為 2024-06-14 補建檔（排除）→ 可用 2024-06～2026-09 共 28 個月；只有上市',
    'o_rev_event_clock_ts_forward': '前瞻紀錄器 forward/ec_firstseen.csv 已於 2026-11-03 06:51 啟動（每日開盤前記錄）',
    'rev_announce': '只用 deadline_calendar（法定期限）與 2026-09 一個月 firstseen 做最壞情境，不是逐檔歷史公告日',
    'GitHub 搜尋（o11 raw/search_log.txt，11 組關鍵字）': '沒有 2019～2024-05 的逐次 commit 營收封存',
}

# ---------------------------------------------------------------- (3) Tasi 精度
P = s36lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
d, seen = s36lib.load_seen(P)
raw = pd.read_csv(s36lib.SEEN_CSV, dtype={'code': str})
raw = raw[raw['source'] == 'tasi'].copy()
raw['Tt'] = pd.to_datetime(raw['T'].str[:19]); raw['Tp'] = pd.to_datetime(raw['T_prev'].str[:19])
gap = (raw['Tt'] - raw['Tp']).dt.total_seconds() / 3600
chk['tasi_precision'] = dict(
    months_all=sorted(raw['ym'].unique().tolist()), n_months_all=int(raw['ym'].nunique()),
    months_usable=[str(m) for m in pd.period_range(s36lib.FIRST_M, s36lib.LAST_M, freq='M')], n_months_usable=int(len(pd.period_range(s36lib.FIRST_M, s36lib.LAST_M, freq='M'))),
    commit_hour_local={int(k): int(v) for k, v in raw['Tt'].dt.hour.value_counts().items()},
    commit_weekday={int(k): int(v) for k, v in raw['Tt'].dt.dayofweek.value_counts().items()},
    gap_prev_commit_hours_quantiles={str(q): float(gap.quantile(q)) for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)},
    reason_counts={k: int(v) for k, v in raw['reason'].value_counts().items()},
    share_day_exact_ok=float((raw['reason'] == 'ok').mean()),
    codes_per_month={k: int(v) for k, v in raw.groupby('ym')['code'].nunique().items()},
    note='T＝第一個包含該檔該月營收的快照 commit 時間＝「最晚何時已公開」的上界。間隔 ≤ 36 小時（ok）時可精確到日；週末／連假（約 72～98 小時）只知道落在區間內。回測一律用 T（保守：真正公告只會更早），所以全部 28 個月都能用，只是週末前後公告的會晚 1～2 天進場。')

# ---------------------------------------------------------------- (4) 覆蓋率（策略股池）
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s36lib.surge(P)
s17lib.e50_fix_suspension(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
months = s36lib.TRAIN_MONTHS + s36lib.VALID_MONTHS
cov = []
for m in months:
    k0 = RT['av'][m]
    if k0 >= P.T:
        continue
    s = RT['S'].loc[m].to_numpy()
    o = s36lib.lma_order(P, U, s, k0)
    univ = np.flatnonzero(U[k0] & np.isfinite(s))
    tg = [x for x in S0['info'] if x['k'] == k0][0]['targets']
    def stat(js):
        has = [(m, P.COLS[j]) in seen for j in js]
        early = [seen.get((m, P.COLS[j]), 10 ** 9) < k0 for j in js]
        lead = [k0 - seen[(m, P.COLS[j])] for j in js if (m, P.COLS[j]) in seen]
        return dict(n=len(js), cov=float(np.mean(has)) if js else np.nan, early=float(np.mean(early)) if js else np.nan, lead=lead)
    a, b, c = stat(list(univ)), stat(list(o[:20])), stat([P.ci[x] for x in tg])
    cov.append(dict(month=str(m), period='train' if m in s36lib.TRAIN_MONTHS else 'valid', sig_date=str(P.IDX[k0].date()),
                    univ_n=a['n'], univ_cov=a['cov'], univ_early=a['early'], top20_cov=b['cov'], top20_early=b['early'],
                    f1_n=c['n'], f1_cov=c['cov'], f1_early=c['early'], f1_lead=c['lead'], top20_lead=b['lead']))
cv = pd.DataFrame(cov)
leadF = np.concatenate([np.array(x) for x in cv['f1_lead']]) if len(cv) else np.array([])
leadT = np.concatenate([np.array(x) for x in cv['top20_lead']])
chk['coverage'] = dict(
    n_months=int(len(cv)), per_month=cv.drop(columns=['f1_lead', 'top20_lead']).to_dict('records'),
    univ_cov_mean=float(cv['univ_cov'].mean()), top20_cov_mean=float(cv['top20_cov'].mean()), f1_cov_mean=float(cv['f1_cov'].mean()),
    univ_early_mean=float(cv['univ_early'].mean()), f1_early_mean=float(cv['f1_early'].mean()),
    f1_lead_days_dist={int(k): int(v) for k, v in pd.Series(leadF).value_counts().sort_index().items()},
    top20_lead_days_dist={int(k): int(v) for k, v in pd.Series(leadT).value_counts().sort_index().items()},
    note='lead＝官方訊號列 k（第一個 > 10 日期限的交易日；現行在 k＋1 開盤買）− 可見列 r。早進場在 r＋1 開盤買 → 比現行早 lead 個交易日。lead ≤ 0＝不會早。')

# ---------------------------------------------------------------- (5) 規則×θ 候選檔數（不看報酬）
ev = s36lib.early_events(P, U, RT, seen, months, 0)
pth = s36lib.prev10_thr(P, U, RT, months, 0)
ev['per'] = np.where(ev['month'].map(lambda x: pd.Period(x, 'M') in s36lib.TRAIN_MONTHS), 'train', 'valid')
cand = {}
f1set = {(x['month'], c) for x in S0['info'] for c in x['targets']}
t20 = {}
for m in months:
    k0 = RT['av'][m]
    if k0 < P.T:
        s = RT['S'].loc[m].to_numpy(); o = s36lib.lma_order(P, U, s, k0)
        for j in o[:20]:
            t20[(str(m), P.COLS[j])] = True
for r in s36lib.RULES:
    ths = s36lib.THR_GRID if r['thr'] == 'grid' else ['prev10']
    for th in ths:
        _, added = s36lib.build_early(P, U, RT, S0, r, pth if th == 'prev10' else th, months, seen, 0, ev)
        A = pd.DataFrame(added, columns=['month', 'code', 'e'])
        o = {}
        for per, ml in (('train', s36lib.TRAIN_MONTHS), ('valid', s36lib.VALID_MONTHS)):
            a = A[A['month'].map(lambda x: pd.Period(x, 'M') in ml)] if len(A) else A
            o[per] = dict(n_signals=int(len(a)), n_months_with=int(a['month'].nunique()) if len(a) else 0,
                          in_official_F1=int(sum((x, c) in f1set for x, c in zip(a['month'], a['code']))),
                          in_official_top20=int(sum((x, c) in t20 for x, c in zip(a['month'], a['code']))))
        cand[f"{r['id']}|{th}"] = o
chk['candidates_preview'] = cand
chk['prev10_thr'] = {str(k): v for k, v in pth.items()}

# ---------------------------------------------------------------- (6) 現行持股的空名額（早進場窗內）
S2, ekw = s10lib.variant_S(S0, 'E0')
nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, sat_w=0.7, sizing='W10', lock=False, **ekw)
a0 = dict(since2019=stlib.cagr(nav, P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1], P.IDX[-1]), full=stlib.cagr(nav, P.IDX[P.t0], P.IDX[-1]),
          maxdd=stlib.maxdd(nav, P.IDX[P.t0], P.IDX[-1]), n_trades=int((lg['why'] != 'open_end').sum()))
a0['ok'] = bool(abs(a0['since2019'] - 0.4899) < 0.00025 and abs(a0['full'] - 0.3454) < 0.00025 and abs(a0['maxdd'] + 0.3211) < 0.00025 and a0['n_trades'] == 381)
chk['A0_current_reproduced'] = a0
npos = np.zeros(P.T, int)
for e, x, w in zip(lg['ent_t'], lg['exit_t'], lg['why']):
    npos[e:(x + 1 if w == 'open_end' else x)] += 1
fs = []
for m in months:
    k0 = RT['av'][m]
    if k0 >= P.T:
        continue
    d1 = int(np.searchsorted(P.IDX.values, np.datetime64((m + 1).to_timestamp())))
    fs.append(dict(month=str(m), free_slots_first_day=int(10 - npos[d1]), free_slots_at_k=int(10 - npos[k0])))
fsd = pd.DataFrame(fs)
chk['free_slots'] = dict(per_month=fs, mean_first_day=float(fsd['free_slots_first_day'].mean()), mean_at_k=float(fsd['free_slots_at_k'].mean()),
                         months_zero_free=int((fsd['free_slots_first_day'] == 0).sum()),
                         note='現行持股在次月 1 日與官方訊號列 k 的空名額（10 − 持股數）。早進場只能用空名額，因為出場仍在 k＋1。')

nm = chk['tasi_precision']['n_months_usable']
chk['data_sufficiency'] = dict(
    rule='涵蓋月份 < 24 → 資料不足以做訓練／驗證，改做描述＋前瞻紙上追蹤',
    tsd_revenue_seen_months=len(tsd), tasi_usable_months=nm, f1_coverage=chk['coverage']['f1_cov_mean'],
    ok=bool(nm >= 24 and chk['coverage']['f1_cov_mean'] >= 0.8),
    verdict=('足夠（以 Tasi 28 個月為主資料；tsd revenue_seen 只有 2 個月，只能當前瞻／交叉核對）' if nm >= 24 else '不足'),
    caveats=['只有上市（現行股池本來就只上市，無妨）', '時間是快照上界，週末前後公告的會晚 1～2 天', '28 個月切兩段各 14 個月，檢定力很低',
             'SURGE 用目前的營收檔（含事後更正），可見時的數字可能略有不同；o11 對 MOPS ann_date 交叉核對 100%（187／210 筆）',
             '第三方 repo 的 commit 時間理論上可被改寫；o11 已與 MOPS 公告日交叉核對'])
chk['runtime_s'] = round(time.time() - t0, 1); chk['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
pr = {k: v for k, v in chk.items() if k not in ('coverage', 'free_slots', 'source_tsd_revenue_seen', 'prev10_thr')}
print(json.dumps(pr, ensure_ascii=False, indent=1, default=str)[:6000])
print(json.dumps({k: v for k, v in chk['coverage'].items() if k != 'per_month'}, ensure_ascii=False, default=str))
print(json.dumps({k: v for k, v in chk['free_slots'].items() if k != 'per_month'}, ensure_ascii=False))

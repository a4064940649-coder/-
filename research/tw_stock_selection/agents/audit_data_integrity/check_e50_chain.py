#!/usr/bin/env python3
"""B7 0050、B8 每日更新鏈。python3 -I -B check_e50_chain.py → parts/e50_chain.json。只讀（含 o_forward_infra_live 的紀錄檔）。"""
import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent))
import io, json, time, resource, re, gzip
import numpy as np, pandas as pd
from common import *                                    # noqa: E402

t0 = time.time(); R = {}
z = np.load(TMP / 'off.npz', allow_pickle=False); IDX = pd.DatetimeIndex(z['IDX'])
pre_dates = pd.DatetimeIndex(sorted(set(pd.read_csv(PRE, usecols=['date'])['date']))); pre_dates = pd.to_datetime(pre_dates)
cal = pre_dates[pre_dates < IDX[0]].append(IDX)

# ============================================================ B7 0050
e = pd.read_csv(FWD / 'raw/finlab/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
e_eng = pd.read_csv(E50_CSV, parse_dates=['date']).set_index('date')['adj']
ec = e[(e.index >= cal[0])]
miss = cal.difference(ec.index); extra = ec.index.difference(cal)
ret = ec.reindex(cal).ffill().pct_change()
raw = pd.read_csv(TMP / 'e50_raw.csv', parse_dates=['date']).set_index('date')
b7 = dict(engine_file=dict(path=str(E50_CSV), sha256=sha_file(E50_CSV), first=str(e_eng.index[0].date()), last=str(e_eng.index[-1].date()), n=len(e_eng), columns=['date', 'adj'],
                           note='只有一欄還原價（含息），沒有原始價、沒有配息明細；檔內沒有記載來源網址'),
          forward_copy=dict(path=str(FWD / 'raw/finlab/etf_0050.csv'), last=str(e.index[-1].date()), n=len(e)),
          n_nan=int(ec.isna().sum()), n_le0=int((ec <= 0).sum()),
          missing_trading_days=[str(x.date()) for x in miss], extra_non_trading_rows=[dict(date=str(x.date()), weekday=x.day_name(), adj=float(ec[x]), prev=float(ec[:x].iloc[-2]), ret_vs_prev_pct=round(float(ec[x] / ec[:x].iloc[-2] - 1) * 100, 2)) for x in extra],
          max_abs_daily_ret_pct=round(float(ret.abs().max() * 100), 2), n_abs_ret_gt_7pct_before_2015_06=int((ret[:'2015-05-31'].abs() > 0.0701).sum()), n_abs_ret_gt_10pct_after=int((ret['2015-06-01':].abs() > 0.1001).sum()),
          big_moves=[dict(date=str(k.date()), ret_pct=round(float(v) * 100, 2)) for k, v in ret[ret.abs() > 0.0701].items()])
# 2015+：原始收盤 vs 還原價 → 推回配息與分割
j = pd.concat([raw['close'].rename('raw'), ec.rename('adj')], axis=1).reindex(IDX)
b7['raw_0050_from_official_k'] = dict(first=str(raw.index[0].date()), last=str(raw.index[-1].date()), n=len(raw), n_missing_days=int(j['raw'].isna().sum()),
                                      missing_days=[str(x.date()) for x in j.index[j['raw'].isna()]], n_open_missing=int(raw['open'].reindex(IDX).isna().sum()))
jj = j.dropna()
rr = jj['raw'].pct_change(); ra = jj['adj'].pct_change(); dlt = (1 + ra) / (1 + rr) - 1
evd = dlt[dlt.abs() > 0.002]
impl = []
for d, v in evd.items():
    p = jj['raw'].shift(1)[d]
    kind = '分割' if (1 + rr[d]) < 0.6 or (1 + rr[d]) > 1.6 else '配息'
    impl.append(dict(date=str(d.date()), kind=kind, raw_prev=float(p), raw=float(jj['raw'][d]), raw_ret_pct=round(float(rr[d]) * 100, 2), adj_ret_pct=round(float(ra[d]) * 100, 2),
                     implied_div=round(float(p * (v) / (1 + v) * (1 + 0)), 3) if kind == '配息' else None, factor=round(float((1 + ra[d]) / (1 + rr[d])), 4)))
b7['implied_events_2015_plus'] = impl
b7['implied_dividend_count_by_year'] = {int(y): int(n) for y, n in pd.Series([int(x['date'][:4]) for x in impl if x['kind'] == '配息']).value_counts().sort_index().items()}
# 事件檔中的 0050
E = pd.read_csv(TMP / 'events.csv', dtype=str, low_memory=False)
e50ev = E[(E['stock_id'] == '0050') & (E['sub'] != 'stophalt')]
tw = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in sorted(TWT49U.glob('*/*.csv.gz'))], ignore_index=True)
t50 = tw[tw['code'] == '0050']
b7['events_0050_twt49u'] = [dict(date=str(r.date)[:10], prev_close=r.prev_close, ref_price=r.ref_price, value=r.value, type=r.type) for r in t50.itertuples()]
b7['events_0050_repo'] = [dict(date=r.file_date, sub=r.sub, kind=getattr(r, 'kind', None), value=getattr(r, 'value', None), ref_price=getattr(r, 'ref_price', None)) for r in e50ev.itertuples()]
b7['twt49u_0050_count_by_year'] = {int(k): int(v) for k, v in t50.groupby(t50['date'].astype(str).str[:4].astype(int)).size().items()}
# 比對：TWT49U 的 0050 事件日 vs 還原價推回的事件日；息值 vs 推回息值
imp = {x['date']: x for x in impl if x['kind'] == '配息'}
cmp_rows = []
for r in t50[t50['date'].astype(str) >= '2015-01-05'].itertuples():
    d = str(r.date)[:10]; x = imp.get(d)
    cmp_rows.append(dict(date=d, twt49u_value=r.value, implied_div=x['implied_div'] if x else None, match=bool(x and abs(x['implied_div'] - r.value) <= max(0.02, 0.02 * r.value))))
b7['twt49u_vs_implied'] = dict(n_twt49u_2015_plus=len(cmp_rows), n_matched=sum(1 for x in cmp_rows if x['match']), n_implied=len(imp), implied_not_in_twt49u=sorted(set(imp) - set(x['date'] for x in cmp_rows)), rows=cmp_rows)
sp = [x for x in impl if x['kind'] == '分割']
b7['split'] = dict(detected=sp, suspension_days_missing_in_adj=[str(x.date()) for x in miss], engine_handling='cslib.Panel 對 0050 還原價先 ffill（停牌日沿用前值），run_official.py 另呼叫 s19lib.e50_fix_suspension（停牌日開盤＝前一成交收盤）',
                   adj_continuous_over_split=bool(all(abs(x['adj_ret_pct']) < 10 for x in sp)))
b7['pre2015'] = dict(dividend_list='無法驗證：本地沒有 2015 以前的 0050 原始價與配息表（TWT49U 封存只有 2014 的 1 年；還原價無法單獨推回配息）',
                     twt49u_2014=[x for x in b7['events_0050_twt49u'] if x['date'] < '2015'])
# 網站資料有沒有用到非交易日那幾列
try:
    nav = json.loads((SITE / 'data/nav.json').read_text()); rows = nav['rows'] if isinstance(nav, dict) else nav
    key = [k for k in rows[0] if '日' in k or 'date' in k.lower()][0]
    ds = pd.to_datetime([x[key] for x in rows])
    b7['site_nav_json'] = dict(n=len(rows), key=key, first=str(ds.min().date()), last=str(ds.max().date()), dates_not_trading_days=[str(x.date()) for x in ds if x not in set(cal)][:20])
except Exception as ex:
    b7['site_nav_json'] = repr(ex)
try:
    nv = pd.read_csv(SITE / 'navs_official.csv.gz', index_col=0, parse_dates=True)
    b7['navs_official'] = dict(sha256=sha_file(SITE / 'navs_official.csv.gz'), first=str(nv.index[0].date()), last=str(nv.index[-1].date()), n=len(nv), columns=list(nv.columns)[:12],
                               dates_not_in_calendar=[str(x.date()) for x in nv.index.difference(cal)], calendar_days_missing=[str(x.date()) for x in cal[(cal >= nv.index[0]) & (cal <= nv.index[-1])].difference(nv.index)],
                               n_nan=int(nv.isna().sum().sum()))
except Exception as ex:
    b7['navs_official'] = repr(ex)
R['B7_0050'] = b7

# ============================================================ B8 每日更新鏈
meta = json.loads((TMP / 'off_meta.json').read_text())
now = pd.Timestamp.now(tz='Asia/Taipei')
tw_last = pd.read_csv(FWD / 'raw/finlab/taiex_twii.csv', parse_dates=['date'])['date'].max()
# 推算「應有的最後交易日」：台北 22:00 後＝今天（平日），否則前一個平日（只扣週末；國定假日由 TWII 檔交叉看）
d = now.normalize().tz_localize(None)
if now.hour < 22:
    d -= pd.Timedelta(days=1)
while d.dayofweek >= 5:
    d -= pd.Timedelta(days=1)
st = {}
for f in ('status', 'live', 'holdings', 'nav', 'top20', 'actions'):
    try:
        jx = json.loads((SITE / f'data/{f}.json').read_text()); m = {x['label']: x['value'] for x in jx.get('meta', []) if isinstance(x, dict) and 'label' in x}
        st[f] = dict(data_last_day=m.get('資料最後一天'), exright_cover=next((x.get('covered_through') for x in jx.get('meta', []) if x.get('label') == '除權息資料涵蓋'), None),
                     exright_warning=next((x.get('warning') for x in jx.get('meta', []) if x.get('label') == '除權息警告'), None), mtime=str(pd.Timestamp((SITE / f'data/{f}.json').stat().st_mtime, unit='s', tz='Asia/Taipei'))[:19])
    except Exception as ex:
        st[f] = repr(ex)
b8 = dict(now_taipei=str(now)[:19], expected_last_trading_day_weekday_rule=str(d.date()), twii_forward_copy_last=str(tw_last.date()),
          tsd=dict(path=str(TSD), head=meta['heads']['tsd'], daily_last=meta['daily']['tsd_last'], exright_last_file=meta['events']['exright']['tsd']['last'],
                   exright_last_file_blob_local=meta['events']['exright']['tsd']['last'] not in meta['events']['exright']['unreadable_days']),
          raw_archive=dict(head=meta['heads']['raw'], daily_last=meta['daily']['raw_last'], days_behind_tsd=len(meta['daily']['only_in_tsd']), exright_blobs_local=0),
          site=st, twt49u_archive_last='2026-10-29')
b8['price_day_is_latest_trading_day'] = bool(meta['daily']['tsd_last'] == str(d.date()) == str(tw_last.date()))
b8['exright_covers_latest_price_day'] = bool(meta['events']['exright']['tsd']['last'] >= meta['daily']['tsd_last'])
# o_forward_infra_live 紀錄檔（只讀）
logs = []
for f in sorted((FWD / 'logs').glob('daily_*.log')):
    tx = f.read_text(errors='replace')
    rc = re.findall(r'make_forward rc=(\d+)', tx)
    sc = re.findall(r'^screener (\w+) (\S+)', tx, re.M); fl = re.findall(r'^finlab\s+(\w+) (\S+)', tx, re.M)
    logs.append(dict(log=f.name, rc=int(rc[-1]) if rc else None, screener_commit_time=sc[-1][1] if sc else None, finlab_commit_time=fl[-1][1] if fl else None,
                     data_fetch_failed='DATA_FETCH_FAILED' in tx, official_fetch_attempted=('官方月營收' in tx or 'OFFICIAL_REV' in tx), official_fetch_failed=('官方月營收抓取失敗' in tx or 'OFFICIAL_REV_FETCH_FAILED' in tx),
                     alerts=[l[:160] for l in tx.split('\n') if l.startswith('【')][:4]))
b8['forward_logs'] = logs
days_logged = sorted(set(l['log'][6:14] for l in logs))
alld = [x.strftime('%Y%m%d') for x in pd.date_range(pd.Timestamp(days_logged[0]), pd.Timestamp(days_logged[-1]))]
b8['forward_log_days_missing'] = [x for x in alld if x not in days_logged]
fr = []
for dd in sorted((FWD / 'raw/official_rev').iterdir()):
    fj = dd / 'FETCH.json'
    if not fj.exists():
        fr.append(dict(dir=dd.name, fetch_json=False)); continue
    x = json.loads(fj.read_text())
    errs = sorted(set(re.sub(r'\(agent proxy.*', '', str(q.get('error'))) for q in x.get('requests', []) if q.get('error')))
    fr.append(dict(dir=dd.name, ok_markets=x.get('ok_markets'), n_requests=len(x.get('requests', [])), n_ok=sum(1 for q in x.get('requests', []) if q.get('status') == 200),
                   hosts=sorted(set(re.sub(r'https?://([^/]+)/.*', r'\1', q.get('url', '')) for q in x.get('requests', []))), errors=errs[:3], n_files=len(list(dd.iterdir()))))
b8['official_rev_fetch'] = fr
ok_any = [x for x in fr if x.get('ok_markets') and any(x['ok_markets'].values())]
fail_days = sorted(set(x['dir'][:8] for x in fr if not (x.get('ok_markets') and any(x['ok_markets'].values()))))
b8['official_rev_fetch_summary'] = dict(n_attempts=len(fr), n_success=len(ok_any), first_attempt=fr[0]['dir'] if fr else None, last_attempt=fr[-1]['dir'] if fr else None,
    n_distinct_days_failed=len(fail_days), failed_days=fail_days, ever_succeeded=bool(ok_any),
    consecutive_failed_attempts_to_latest=next((i for i, x in enumerate(reversed(fr)) if x.get('ok_markets') and any(x['ok_markets'].values())), len(fr)))
# 上游月營收彙總表（tsd）的更新歷史：2026-10 上市檔的提交時間（本機 git 紀錄，不連網）
lg = git(TSD, 'log', '--format=%cI', '--', 'data/mops/revenue_hist/2026-10_twse.csv').stdout.split()
b8['upstream_revenue_hist_2026_10_twse_commits'] = dict(n=len(lg), first=lg[-1] if lg else None, last=lg[0] if lg else None, all=lg[::-1][:40])
lg2 = git(TSD, 'log', '--format=%cI', '-n', '400', '--', 'data/universe/daily').stdout.split()
b8['upstream_daily_commit_times_recent'] = lg2[:25]
seen = [p for s, p in ls_tree(TSD, 'data/meta') if 'revenue' in p][:10]
b8['revenue_seen_dirs'] = git(TSD, 'ls-tree', '--name-only', 'HEAD', 'data/meta/revenue_seen/').stdout.split()[-6:]
R['B8_daily_chain'] = b8
R['runtime_s'] = round(time.time() - t0, 1); R['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
jdump(R, HERE / 'parts/e50_chain.json')
for sec in ('B7_0050', 'B8_daily_chain'):
    for k, v in R[sec].items():
        print('##', k, json.dumps(v, ensure_ascii=False, default=str)[:2200])

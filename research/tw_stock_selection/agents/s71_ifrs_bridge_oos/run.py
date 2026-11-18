#!/usr/bin/env python3
"""S71 正式執行（只跑一次；results.json 已存在就拒絕）：凍結現行規則＋跨 IFRS 斷點規則，在自建 2007～2015H1 面板上
自 2008-01-02 連續跑到 2015-06-30。主規則與判定見 PREREG.json。先核對 SPEC／PREREG／程式的 sha256。
python3 -I -B run.py → results.json、oos_monthly.csv、oos_nav.csv.gz、oos_trades.csv.gz、oos_signals.csv"""
import sys, json, time, resource, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent
assert not (HERE / 'results.json').exists(), 'results.json 已存在：本題只跑一次'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f'{f} 與登記的 sha256 不符'
import numpy as np, pandas as pd
import s71lib as S
from s71lib import R

t0 = time.time()
PRE = json.load(open(HERE / 'PREREG.json'))
PRIMARY, RULES = PRE['primary_rule'], PRE['run_rules']
acc = json.load(open(HERE / 'accept.json'))
R.point_cache(S.TMP / 'cache_pre')
mask = pd.read_csv(S.TMP / 'mask_nav.csv', parse_dates=[0], index_col=0)
mrs, mrb = R.monthly_returns(mask['nav'], mask['e50'])
out = dict(task='S71 跨 IFRS 斷點規則延伸樣本外 2013-02～2015-06', primary_rule=PRIMARY, sha=dict(SPEC=sha(HERE / 'SPEC.json'), PREREG=sha(HERE / 'PREREG.json')), rules={})
monthly = {}; navs = {}; trades = []; sigs = []
for rule in RULES:
    P, U, F, sg, nav, lg, st = S.run(rule)
    assert str(P.IDX[P.t0].date()) == '2008-01-02' and str(P.IDX[-1].date()) == '2015-06-30'
    bench = pd.Series(P.E50, index=P.IDX).reindex(nav.index)
    rs, rb = R.monthly_returns(nav, bench)
    win = lambda a, b, x=rs, y=rb: (x[(x.index >= pd.Period(a, 'M')) & (x.index <= pd.Period(b, 'M'))], y[(y.index >= pd.Period(a, 'M')) & (y.index <= pd.Period(b, 'M'))])
    W = {k: R.stats_block(*win(a, b)) for k, (a, b) in dict(ext_2013_02_2015_06=('2013-02', '2015-06'), merged_2008_01_2015_06=('2008-01', '2015-06'),
                                                              old_primary_2008_01_2013_01=('2008-01', '2013-01')).items()}
    W['ext_2013_02_2015_06']['maxdd'] = R.maxdd(nav.loc['2013-01-31':]); W['ext_2013_02_2015_06']['maxdd_0050'] = R.maxdd(bench.loc['2013-01-31':])
    W['merged_2008_01_2015_06']['maxdd'] = R.maxdd(nav); W['merged_2008_01_2015_06']['maxdd_0050'] = R.maxdd(bench)
    info = sg['info']
    e = W['ext_2013_02_2015_06']
    dec = '支持' if (e['ex_ann_pp'] > 0 and e['t_nw6'] > 1.65) else ('方向一致但不顯著' if e['ex_ann_pp'] > 0 else '不支持')
    # 與現行（MASK）在 2008-01～2013-01 的月報酬是否相同（規則只改 2013-01 以後的營收月份）
    p1 = win('2008-01', '2013-01')[0]; pm = mrs[(mrs.index <= pd.Period('2013-01', 'M'))]
    same_pre = float(np.abs(p1.to_numpy() - pm.to_numpy()).max())
    # 對現行卡住期的配對差（描述）
    d = (win('2013-02', '2015-06')[0] - mrs[(mrs.index >= pd.Period('2013-02', 'M'))]).to_numpy()
    x = lg[lg['why'] != 'open_end']
    ext_info = [i for i in info if i['month'] >= '2013-01']
    lg2 = lg[lg['ent_date'] > pd.Timestamp('2013-01-31')]
    k3 = F['info']['k'][3] if rule == 'B2' else S.scope_ratio(F['Rv'], 3)[0]
    kk = k3.reindex(lg2['code']).to_numpy() if 'code' in lg2 else np.array([])
    o = dict(windows=W, decision_ext=dec, yearly=R.yearly_excess(rs, rb),
             pre2013_02_monthly_max_abs_diff_vs_MASK=same_pre, pre2013_02_identical_to_accept=bool(same_pre < 1e-12),
             old_primary_vs_accept=dict(ex_ann_pp=W['old_primary_2008_01_2013_01']['ex_ann_pp'], accept=acc['A2_R55_main']['primary']['ex_ann_pp']),
             vs_MASK_stuck_paired=dict(diff_ann_pp=float(d.mean() * 12 * 100), t_nw6=R.nw_t(d, 6), mask_ex_ann_pp=acc['A2_R55_main']['stuck']['ex_ann_pp']),
             signal=dict(n_signal_months=len(info), first=info[0]['sig_date'], last=info[-1]['sig_date'], n_ext_signal_months=len(ext_info),
                         ext_first=ext_info[0]['sig_date'] if ext_info else None, ext_mean_targets=float(np.mean([i['n_tgt'] for i in ext_info])) if ext_info else None,
                         ext_months_zero_target=[i['month'] for i in ext_info if i['n_tgt'] == 0]),
             trades=dict(n_buy=int(st['n_buy']), n_closed=int(len(x)), n_open_end=int((lg['why'] == 'open_end').sum()), n_entry_ext=int(len(lg2)),
                         win_rate_closed_ext=float((lg2[lg2['why'] != 'open_end']['ret'] > 0).mean()) if len(lg2) else None,
                         mean_ret_closed_ext=float(lg2[lg2['why'] != 'open_end']['ret'].mean()) if len(lg2) else None,
                         share_entries_ext_k3_gt1=float(np.nanmean(kk > 1)) if len(kk) else None, share_entries_ext_k3_missing=float(np.mean(~np.isfinite(kk))) if len(kk) else None,
                         avg_npos=float(st['avg_npos']), avg_idle_frac=float(st['avg_idle_frac']), n_delist=int(st['n_delist'])))
    out['rules'][rule] = o
    monthly[f'{rule}_strat'] = rs; monthly['e0050'] = rb; navs[f'{rule}_nav'] = nav; navs['e50'] = bench
    trades.append(lg.drop(columns=['j']).assign(rule=rule))
    sigs.append(pd.DataFrame(info).drop(columns=['targets']).assign(targets=[' '.join(i['targets']) for i in info], rule=rule))
monthly['MASK_strat'] = mrs
p = out['rules'][PRIMARY]
out['verdict'] = dict(rule=PRIMARY, window='2013-02～2015-06', ex_ann_pp=p['windows']['ext_2013_02_2015_06']['ex_ann_pp'], t_nw6=p['windows']['ext_2013_02_2015_06']['t_nw6'], decision=p['decision_ext'])
out['accept'] = dict(A1_bitwise=acc['A1_official']['bitwise_vs_ACC'], A2_pass=acc['A2_R55_main']['pass'], A3_bitwise=acc['A3_mask_equals_engine']['bitwise'])
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(HERE / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(monthly).to_csv(HERE / 'oos_monthly.csv')
pd.DataFrame(navs).to_csv(HERE / 'oos_nav.csv.gz')
pd.concat(trades).to_csv(HERE / 'oos_trades.csv.gz', index=False)
pd.concat(sigs).to_csv(HERE / 'oos_signals.csv', index=False)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

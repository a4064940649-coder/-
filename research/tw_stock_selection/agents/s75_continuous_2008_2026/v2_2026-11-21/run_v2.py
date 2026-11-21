#!/usr/bin/env python3
"""S75 連續帳戶（2008-01-02～2026-10-22）用 2026-11-21 正式口徑重跑（工程 eng_switch_v2_2026-11-21；不覆蓋 S75 原檔，只寫本資料夾）。
用法：python3 -I -B run_v2.py base   → 開關與 S75 原本相同，核對與 ../nav_2008_2026.csv.gz 逐日相同（確認這支包裝沒有改到東西）
      python3 -I -B run_v2.py v2     → 多開兩項：
        A. 0050 含息序列：2015-01-05 起用官方自建檔（raw/e50_tr_official/etf_0050_tr.csv）；2015-01-05 以前沒有官方日 K 與事件檔，
           沿用舊來源（cslib.E50_CSV）並在 2015-01-05 等比例銜接（舊序列 × 官方[2015-01-05] ÷ 舊[2015-01-05]）→ 本資料夾 e50_spliced.csv。
           2015-01-05 以前的日報酬因此與舊來源完全相同。
        B. cslib.REDUCE_FIX＝True：減資／面額變更恢復買賣日用事件參考價。事件封存從 2015-08-24 起，所以只有 2015 年後會套用；2008～2015-07 沒有來源，維持原樣。
引擎、訊號、面板全部沿用 ../s75lib.py（沒有改任何既有檔）。標準路徑＝100 萬、位移 0；另跑 20 條路徑（5 種本金 × 位移 0～3）取全期年化中位數。
輸出：base_check.json；results_v2.json、nav_2008_2026_v2.csv.gz、yearly_v2.csv、trades_v2.csv.gz、compare_old_new.json。"""
import sys, json, time, resource, gc
from pathlib import Path
HERE = Path(__file__).resolve().parent
S75 = HERE.parent
sys.path.insert(0, str(S75))
import numpy as np, pandas as pd
import s75lib as K
from s75lib import R, cslib
MODE = sys.argv[1]
assert MODE in ('base', 'v2')
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
info = {}
if MODE == 'v2':
    old = pd.read_csv(cslib.E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
    new = pd.read_csv(cslib.E50_OFFICIAL_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
    seam = new.index[0]
    assert str(seam.date()) == '2015-01-05' and seam in old.index
    k = float(new.loc[seam] / old.loc[seam])
    sp = pd.concat([old.loc[:seam].iloc[:-1] * k, new])
    assert sp.index.is_monotonic_increasing and sp.index.is_unique
    sp.rename('adj').rename_axis('date').to_csv(HERE / 'e50_spliced.csv', float_format='%.10f')
    ov = pd.concat([old, new], axis=1, keys=['old', 'new']).dropna()
    rr = (ov['new'] / ov['new'].shift(1)) / (ov['old'] / ov['old'].shift(1)) - 1
    info['e50_splice'] = dict(seam=str(seam.date()), scale=k, n_pre=int((sp.index < seam).sum()), n_official=int(len(new)),
                              first_day_daily_ret_differs_over_1e5=str(rr[rr.abs() > 1e-5].index[0].date()) if (rr.abs() > 1e-5).any() else None,   # 兩個檔的小數位數不同，1e-5 以下是四捨五入
                              n_days_daily_ret_differs_over_1e5=int((rr.abs() > 1e-5).sum()), max_abs_daily_ret_diff_before_2025_07_21=float(rr.loc[:'2025-07-18'].abs().max()))
    cslib.E50_SOURCE = HERE / 'e50_spliced.csv'
    cslib.REDUCE_FIX = True
P, U = K.load_cont()
info.update(reduce_fix=bool(getattr(P, 'reduce_fix', False)), reduce_info=getattr(P, 'reduce_info', None), e50_source=getattr(P, 'e50_source', None), e50_missing=int(getattr(P, 'e50_missing', -1)))
D0, DE = P.IDX[P.t0], P.IDX[-1]
assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')
D19 = K.last_before(P.IDX, '2019-01-01'); D157 = pd.Timestamp('2015-07-01')
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]; bench = bench / bench.iloc[0] * 1e6
out = dict(mode=MODE, info=info, versions={}); NAV = {}; TR = []
for v, rule in K.VERSIONS.items():
    F, RT = K.surge(P, rule)
    paths = {}
    for s in K.SHIFTS:
        sg = K.signal(P, U, RT, s)
        for cap in K.CAPS:
            nav, lg, st = K.sim(P, sg, U, cap)
            paths[f'{int(cap / 1e4)}萬|+{s}'] = dict(cagr_full=K.cagr(nav, D0, DE), maxdd=R.maxdd(nav), cagr_since2019=K.cagr(nav, D19, DE))
            if s == 0 and cap == 1e6:
                std = (nav, lg, [dict(month=i['month'], sig_date=i['sig_date'], targets=' '.join(i['targets'])) for i in sg['info']])
            del nav, lg, st
        del sg; gc.collect()
    nav, lg, sig = std; NAV[v] = nav; b = bench.reindex(nav.index)
    x = lg[lg['why'] != 'open_end']
    out['versions'][v] = dict(final_nav=float(nav.iloc[-1]), cagr=K.cagr(nav, D0, DE), maxdd=K.dd_info(nav), cagr_since2019=K.cagr(nav, D19, DE), cagr_since2015_07=K.cagr(nav, D157, DE),
                              cagr_2008_2015H1=K.cagr(nav, D0, K.last_before(P.IDX, '2015-07-01')),
                              n_closed=int(len(x)), n_all=int(len(lg)), win_rate=float((x['ret'] > 0).mean()),
                              final_nav_0050=float(b.iloc[-1]), cagr_0050=K.cagr(b, D0, DE), maxdd_0050=K.dd_info(b), cagr_0050_since2019=K.cagr(b, D19, DE),
                              cagr_0050_2008_2015H1=K.cagr(b, D0, K.last_before(P.IDX, '2015-07-01')),
                              yearly=K.yearly_table(nav, b), signals=sig,
                              paths20=dict(cagr_full_median=float(np.median([p['cagr_full'] for p in paths.values()])), cagr_full_min=float(min(p['cagr_full'] for p in paths.values())),
                                           cagr_full_max=float(max(p['cagr_full'] for p in paths.values())), maxdd_median=float(np.median([p['maxdd'] for p in paths.values()])),
                                           maxdd_worst=float(min(p['maxdd'] for p in paths.values()))))
    TR.append(lg.drop(columns=[c for c in ('j',) if c in lg.columns]).assign(version=v))
    print(v, {k: (round(z, 5) if isinstance(z, float) else z) for k, z in out['versions'][v].items() if k not in ('yearly', 'signals')}, f'{time.time() - t0:.0f}s', rss(), 'MB', flush=True)
    del std, F, RT; gc.collect()
df = pd.DataFrame({'V_MASK': NAV['V_MASK'], 'V_BRIDGE': NAV['V_BRIDGE'], '0050': bench.reindex(NAV['V_MASK'].index)}).round(2); df.index.name = 'date'
ref = pd.read_csv(S75 / 'nav_2008_2026.csv.gz', index_col=0, parse_dates=True)
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
if MODE == 'base':
    same = df.shape == ref.shape and list(df.columns) == list(ref.columns) and df.index.equals(ref.index)
    chk = dict(same_shape=bool(same), max_abs_diff_vs_S75_nav=float((df - ref).abs().max().max()) if same else None,
               sig_same_as_S75=None, n_closed={v: out['versions'][v]['n_closed'] for v in out['versions']}, final={v: out['versions'][v]['final_nav'] for v in out['versions']},
               runtime_s=out['runtime_s'], maxrss_mb=out['maxrss_mb'])
    chk['pass'] = bool(same and chk['max_abs_diff_vs_S75_nav'] == 0)
    for v in out['versions']: out['versions'][v].pop('signals')
    chk['base_results'] = out['versions']
    (HERE / 'base_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
    print('base check', {k: chk[k] for k in ('same_shape', 'max_abs_diff_vs_S75_nav', 'pass')}); sys.exit(0 if chk['pass'] else 2)
old_sig = pd.read_csv(S75 / 'signals.csv', dtype=str).fillna('')
for v in out['versions']:
    o = old_sig[old_sig['version'] == v]; n = pd.DataFrame(out['versions'][v].pop('signals'))
    same_n = len(o) == len(n)
    out['versions'][v]['signals_vs_S75'] = dict(n_months=int(len(n)), n_months_old=int(len(o)), sig_dates_same=bool(same_n and list(o['sig_date']) == list(n['sig_date'])),
                                                months_targets_differ=[m for m, a, b_ in zip(n['month'], o['targets'], n['targets']) if a.split() != b_.split()] if same_n else None)
(HERE / 'results_v2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
df.to_csv(HERE / 'nav_2008_2026_v2.csv.gz')
ym, yb = out['versions']['V_MASK']['yearly'], out['versions']['V_BRIDGE']['yearly']
pd.DataFrame([dict(year=m['year'], V_MASK=m['strat'], V_BRIDGE=q['strat'], e0050=m['e0050']) for m, q in zip(ym, yb)]).to_csv(HERE / 'yearly_v2.csv', index=False)
pd.concat(TR).to_csv(HERE / 'trades_v2.csv.gz', index=False)
base = json.loads((HERE / 'base_check.json').read_text())['base_results']
cmp = dict(說明='S75 連續帳戶 新（2026-11-21 正式口徑：0050 官方含息序列 2015-01-05 起銜接＋減資復牌修正 2015 年後）對 舊（S75 原結果，本資料夾 base 模式逐日重現）', info=info, versions={})
for v in out['versions']:
    a, n = base[v], out['versions'][v]
    cmp['versions'][v] = {k: dict(舊=a[k], 新=n[k]) for k in ('final_nav', 'cagr', 'cagr_since2019', 'cagr_since2015_07', 'cagr_2008_2015H1', 'n_closed', 'win_rate', 'final_nav_0050', 'cagr_0050', 'cagr_0050_since2019', 'cagr_0050_2008_2015H1')}
    cmp['versions'][v]['maxdd'] = dict(舊=a['maxdd'], 新=n['maxdd']); cmp['versions'][v]['maxdd_0050'] = dict(舊=a['maxdd_0050'], 新=n['maxdd_0050'])
    cmp['versions'][v]['paths20'] = dict(舊=a['paths20'], 新=n['paths20'])
    cmp['versions'][v]['yearly_changed'] = [dict(year=x['year'], 舊=round(x['strat'] * 100, 2), 新=round(y['strat'] * 100, 2), 舊_0050=round(x['e0050'] * 100, 2), 新_0050=round(y['e0050'] * 100, 2))
                                            for x, y in zip(a['yearly'], n['yearly']) if round(x['strat'], 4) != round(y['strat'], 4) or round(x['e0050'], 4) != round(y['e0050'], 4)]
    cmp['versions'][v]['signals_vs_S75'] = n['signals_vs_S75']
    cmp['versions'][v]['first_day_nav_differs_over_1e5_rel'] = str(df.index[((df[v] / ref[v] - 1).abs() > 1e-5).to_numpy().argmax()].date()) if ((df[v] / ref[v] - 1).abs() > 1e-5).any() else None
cmp['first_day_0050_differs_over_1e5_rel'] = str(df.index[((df['0050'] / ref['0050'] - 1).abs() > 1e-5).to_numpy().argmax()].date()) if ((df['0050'] / ref['0050'] - 1).abs() > 1e-5).any() else None
(HERE / 'compare_old_new.json').write_text(json.dumps(cmp, ensure_ascii=False, indent=1, default=str))
print('done', out['runtime_s'], 's', out['maxrss_mb'], 'MB')

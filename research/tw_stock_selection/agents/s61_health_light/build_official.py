#!/usr/bin/env python3
"""S61 第一步：用正式管線（照 agents/official_nav/run_official.py，檔案不改）重跑 E0｜W10_70 的 20 條路徑，
存健康度燈號需要的逐月資料與標準路徑的交易／名單／每日 W_ACT。共用引擎只在執行期開三個正式開關、改快取目錄。
輸出（本資料夾）：official_trades.csv.gz、official_picks.csv、official_weights.csv.gz、paths_monthly.csv.gz、build_meta.json。
用法：python3 -I -B build_official.py"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
SITE = Path('/home/user/-/xq_scripts/site')
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s61/cache')
OFF_SHA = '434f64222b8fd1f10e1e25c9ac4553e7e6f6d11305896a3ceabb3b212650dfab'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'SPEC.json') == (HERE / 'SPEC.sha256').read_text().split()[0], 'SPEC 與 sha256 不符'
assert sha(TMP / 'panel_cs.npz') == OFF_SHA, '面板快取不是正式版本'
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)
assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True

P = s19lib.load()
assert P.exright_fix
s19lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
S, _ = s19lib.rev_std(P)
months = list(S.index)
avA = s19lib.a10_rows(P, months)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]
ME = bench.groupby(bench.index.to_period('M')).tail(1).index       # 每月最後一個交易日
rows = []
meta = dict(start=str(D0.date()), end=str(DE.date()), paths={})
for s in range(4):
    Sg = s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=s)
    picks = pd.Series({pd.Timestamp(x['sig_date']).to_period('M'): x['n_tgt'] for x in Sg['info']})
    assert not picks.index.duplicated().any()
    for cap in (6e5, 8e5, 1e6, 1.5e6, 2e6):
        nav, lg, st = s10lib.simulate_s10(P, Sg, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap)
        key = f'{int(cap/1e4)}萬|+{s}'
        nav = nav.loc[D0:DE]
        w = st['stockfrac'].loc[D0:DE]
        x = lg[lg['why'] != 'open_end'].copy()
        x['xm'] = pd.DatetimeIndex(P.IDX[x['exit_t'].to_numpy()]).to_period('M')
        nc = x.groupby('xm').size(); nw = x[x['ret'] > 0].groupby('xm').size()
        mi = ME.to_period('M')
        df = pd.DataFrame(dict(path=key, month=mi.astype(str), nav=nav.reindex(ME).to_numpy(), b0050=bench.reindex(ME).to_numpy(),
                               w_act=w.groupby(w.index.to_period('M')).mean().reindex(mi).to_numpy(),
                               n_closed=nc.reindex(mi, fill_value=0).to_numpy(), n_win=nw.reindex(mi, fill_value=0).to_numpy(),
                               n_picks=picks.reindex(mi).to_numpy()))
        rows.append(df)
        meta['paths'][key] = dict(cagr19=stlib.cagr(nav, D19, DE), n_closed=int(len(x)), first_sig=str(picks.index.min()))
        if key == '100萬|+0':
            ref = pd.read_csv(SITE / 'navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
            rel = float(np.nanmax(np.abs(nav.reindex(ref.index).to_numpy() / ref.to_numpy() - 1)))
            ro = json.loads((AG / 'official_nav' / 'out' / 'res_official.json').read_text())['strats']['E0|W10_70']['std']
            meta['acceptance'] = dict(nav_vs_site_max_rel=rel, cagr19=stlib.cagr(nav, D19, DE), cagr19_ref=ro['cagr_since2019'],
                                      n_closed=int(len(x)), n_closed_ref=ro['n_trades_closed'])
            meta['acceptance']['pass'] = bool(rel < 1e-6 and abs(meta['acceptance']['cagr19'] - ro['cagr_since2019']) < 1e-9 and len(x) == 381)
            assert meta['acceptance']['pass'], meta['acceptance']
            T = lg.copy()
            T['ent_date'] = P.IDX[T['ent_t'].to_numpy()].strftime('%Y-%m-%d') if 'ent_t' in T else None
            T['exit_date'] = P.IDX[T['exit_t'].to_numpy()].strftime('%Y-%m-%d')
            T[['code', 'ent_date', 'exit_date', 'why', 'shares', 'cost', 'proceeds', 'ret']].to_csv(HERE / 'official_trades.csv.gz', index=False)
            pk = [dict(month=str(pd.Timestamp(i['sig_date']).to_period('M')), sig_date=i['sig_date'], code=c) for i in Sg['info'] for c in i['targets']]
            emp = [dict(month=str(pd.Timestamp(i['sig_date']).to_period('M')), sig_date=i['sig_date'], code='') for i in Sg['info'] if not i['targets']]
            pd.DataFrame(pk + emp).sort_values(['month', 'code']).to_csv(HERE / 'official_picks.csv', index=False)
            pd.DataFrame({'date': w.index.strftime('%Y-%m-%d'), 'w_act': w.to_numpy().round(6)}).to_csv(HERE / 'official_weights.csv.gz', index=False)
        del nav, lg, st
    del Sg; gc.collect()
    print('shift', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
pd.concat(rows).to_csv(HERE / 'paths_monthly.csv.gz', index=False)
meta['peak_mb'] = rss(); meta['secs'] = round(time.time() - t0)
meta['lg_columns'] = 'official_trades: code, ent_date, exit_date, why, shares, cost, proceeds, ret（ret＝proceeds÷cost−1）'
(HERE / 'build_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
print(json.dumps(meta['acceptance'], ensure_ascii=False), meta['peak_mb'], 'MB', meta['secs'], 's')

#!/usr/bin/env python3
"""S1 第二步：三個候選 × {只上市, 上市＋上櫃} × {(a) 空位放現金, (b) 空位放 0050} × {基準, 成本加倍, 拿掉最賺 5 筆}。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。必須先有 PREREG.json（sha256 會寫進 results.json）。"""
import sys, json, hashlib, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))      # -I 不會自動加入腳本所在資料夾
import stlib

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
t_all = time.time()
P = stlib.load()
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]            # 2018 年最後一個交易日
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'half1': (D0, DM), 'half2': (DM, DE)}
res = {'PREREG_sha256': sha, 'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), end=str(DE.date()),
                                         bad_ret=P.bad_ret, e50_missing=P.e50_missing, e50_open_missing=P.e50_open_missing),
       'bench_0050': {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b), sharpe=stlib.sharpe(bench, a, b)) for k, (a, b) in WIN.items()},
       'runs': {}, 'classification': {}}
navs = {'0050': bench}; trades = []
for cid, fn in stlib.CANDS.items():
    for uni in ('twse', 'all'):
        U = stlib.universe(P, uni)
        S = fn(P, U)
        n_sig = int(S['sig'][P.t0:].sum())
        for idle in ('cash', 'e50'):
            key = f'{cid}|{uni}|{idle}'
            nav, lg, st = stlib.simulate(P, S, U, idle=idle)
            closed = lg[lg['why'] != 'open_end']
            top5 = closed.sort_values('pnl', ascending=False).head(5)
            block = frozenset(zip(top5['sig_t'].astype(int), top5['j'].astype(int)))
            nav2, lg2, st2 = stlib.simulate(P, S, U, idle=idle, cmult=2.0)
            nav5, lg5, st5 = stlib.simulate(P, S, U, idle=idle, block=block)
            r = dict(candidate=cid, universe=uni, idle=idle, n_signals_raw=n_sig,
                     windows={k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()},
                     cost_x2={k: stlib.window(nav2, bench, a, b) for k, (a, b) in WIN.items()},
                     drop_top5={k: stlib.window(nav5, bench, a, b) for k, (a, b) in WIN.items()},
                     top5_trades=[dict(code=x.code, ent=str(x.ent_date.date()), exit=str(x.exit_date.date()), ret=float(x.ret), pnl=float(x.pnl)) for x in top5.itertuples()],
                     yearly=stlib.yearly(nav, bench), trades_full=stlib.trade_stats(lg), trades_since2019=stlib.trade_stats(lg, '2019-01-01'),
                     trades_cost_x2=stlib.trade_stats(lg2),
                     engine={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in st.items()},
                     final_nav=float(nav.iloc[-1]), cost_total_pct_of_start=float((st['fee'] + st['tax'] + st['slip'] + st['e50_cost']) / 1e6))
            w, c2, d5 = r['windows'], r['cost_x2'], r['drop_top5']
            chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0,
                       costx2_since2019=c2['since2019']['diff'] > 0, costx2_full=c2['full']['diff'] > 0,
                       drop5_since2019=d5['since2019']['diff'] > 0, drop5_full=d5['full']['diff'] > 0,
                       half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
            if all(chk.values()):
                cls = '值得上XQ測'
            elif chk['net_since2019'] or chk['net_full']:
                cls = '有條件'
            else:
                cls = '不值得'
            r['checks'] = chk; r['class'] = cls; r['fails'] = [k for k, v in chk.items() if not v]
            res['runs'][key] = r
            navs[key] = nav; navs[key + '|cost_x2'] = nav2; navs[key + '|drop5'] = nav5
            lg = lg.assign(run=key); trades.append(lg)
            print(key, cls, 'since2019 %.1f%% vs %.1f%% | full %.1f%% vs %.1f%% | mdd %.1f%% | n %d win %.0f%% hold %.1f | x2 %.1f%% | drop5 %.1f%% | fails %s' % (
                w['since2019']['cagr'] * 100, w['since2019']['cagr_0050'] * 100, w['full']['cagr'] * 100, w['full']['cagr_0050'] * 100, w['full']['maxdd'] * 100,
                r['trades_full'].get('n', 0), r['trades_full'].get('win_rate', 0) * 100, r['trades_full'].get('avg_hold_days', 0),
                c2['since2019']['cagr'] * 100, d5['since2019']['cagr'] * 100, r['fails']), flush=True)
    res['classification'][cid] = dict(basis='twse|cash', main=res['runs'][f'{cid}|twse|cash']['class'], main_fails=res['runs'][f'{cid}|twse|cash']['fails'],
                                      idle_0050=res['runs'][f'{cid}|twse|e50']['class'], idle_0050_fails=res['runs'][f'{cid}|twse|e50']['fails'],
                                      all_cash=res['runs'][f'{cid}|all|cash']['class'], all_0050=res['runs'][f'{cid}|all|e50']['class'])
res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(2).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
tr[['run', 'code', 'market', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why']].round(6).to_csv(H / 'trades.csv.gz', index=False)
print(json.dumps(res['classification'], ensure_ascii=False, indent=1)); print(res['meta'])

#!/usr/bin/env python3
"""彙總四個版本（＋穩健性 Bp、ABp）與工程試算對照 → results.json、compare_versions.csv、yearly.csv、held_events.csv、trade_diff.csv"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent; ENG = AG / 'eng_audit_fixes_2026-11'
MODES = ['base', 'A', 'B', 'AB', 'Bp', 'ABp']
R = {m: json.loads((HERE / 'out' / m / 'res.json').read_text()) for m in MODES}
NAV = {m: pd.read_csv(HERE / 'out' / m / 'navs.csv.gz', index_col=0, parse_dates=True) for m in MODES}
out = dict(baseline_check=json.loads((HERE / 'out' / 'base' / 'baseline_check.json').read_text()), versions={}, e50_build=None, reduce_build=None)
rows = []; yr = {}
for m in MODES:
    v = {}
    for col in ('E0|W10_70', 'XT|W10_70', 'XT_B11|W10_70'):
        s = R[m]['strats'][col]; b = R['base']['strats'][col]; st = s['std']
        tg_same = s['targets'] == b['targets']; sd_same = s['sig_dates'] == b['sig_dates']
        chg_m = [k for k in b['targets'] if s['targets'].get(k) != b['targets'][k]]
        v[col] = dict(cagr19=st['cagr_since2019'], cagr_full=st['cagr_full'], cagr24=st['cagr_since2024'], maxdd=st['maxdd'], dd_peak=st['dd_peak'], dd_trough=st['dd_trough'],
                      final_nav=st['final_nav'], n_closed=st['n_trades_closed'], n_all=st['n_trades_all'], open_end=st['open_end'], yearly=st['yearly'],
                      paths=s.get('paths_summary'), n_signal_months=s['n_signal_months'], targets_same_as_base=tg_same, sig_dates_same_as_base=sd_same, months_changed=chg_m)
        rows.append(dict(version=m, col=col, cagr19_pct=round(st['cagr_since2019'] * 100, 4), cagr_full_pct=round(st['cagr_full'] * 100, 4), cagr24_pct=round(st['cagr_since2024'] * 100, 4),
                         maxdd_pct=round(st['maxdd'] * 100, 4), dd_peak=st['dd_peak'], dd_trough=st['dd_trough'], final_nav=round(st['final_nav']), n_closed=st['n_trades_closed'],
                         paths_median_pct=round(s['paths_summary']['cagr19_median'] * 100, 4), paths_min_pct=round(s['paths_summary']['cagr19_min'] * 100, 4), paths_max_pct=round(s['paths_summary']['cagr19_max'] * 100, 4),
                         paths_dd_median_pct=round(s['paths_summary']['maxdd_median'] * 100, 4), paths_dd_worst_pct=round(s['paths_summary']['maxdd_worst'] * 100, 4),
                         n_signal_months=s['n_signal_months'], targets_same=tg_same, sig_dates_same=sd_same))
        yr[(m, col)] = {int(k): round(x * 100, 4) for k, x in st['yearly'].items()}
    ex = R[m]['strats']['E0|W10_70']['extras']
    v['0050'] = dict(cagr19=ex['e50_since2019'], cagr_full=ex['e50_full'], cagr24=ex['e50_since2024'], maxdd=ex['e50_maxdd'], yearly=ex['yearly_0050'], final=float(NAV[m]['0050'].iloc[-1]))
    v['extras_E0'] = {k: ex[k] for k in ex if k not in ('yearly_0050',)}
    v['rfix'] = R[m].get('rfix'); v['exright_info'] = R[m]['exright_info']; v['acceptance'] = R[m]['acceptance']
    rows.append(dict(version=m, col='0050', cagr19_pct=round(ex['e50_since2019'] * 100, 4), cagr_full_pct=round(ex['e50_full'] * 100, 4), cagr24_pct=round(ex['e50_since2024'] * 100, 4), maxdd_pct=round(ex['e50_maxdd'] * 100, 4), final_nav=round(v['0050']['final'])))
    yr[(m, '0050')] = {int(k): round(x * 100, 4) for k, x in ex['yearly_0050'].items()}
    out['versions'][m] = v
pd.DataFrame(rows).to_csv(HERE / 'compare_versions.csv', index=False)
Y = pd.DataFrame(yr); Y.columns = [f'{a}|{b}' for a, b in Y.columns]; Y.index.name = 'year'; Y.to_csv(HERE / 'yearly.csv')
# 加法性
b, A, B, AB = (out['versions'][m]['E0|W10_70'] for m in ('base', 'A', 'B', 'AB'))
out['additivity_E0_pp'] = {k: dict(A=(A[k] - b[k]) * 100, B=(B[k] - b[k]) * 100, AB=(AB[k] - b[k]) * 100, sum=(A[k] + B[k] - 2 * b[k]) * 100) for k in ('cagr19', 'cagr_full', 'cagr24', 'maxdd')}
# 與工程試算對照
e5 = json.loads((ENG / 'e50' / 'res_official_e50fix.json').read_text()); er = json.loads((ENG / 'reduce' / 'res_official_reducefix.json').read_text())
cmpv = {}
for m, e, f in (('A', e5, ENG / 'e50' / 'navs_official_e50fix.csv.gz'), ('B', er, ENG / 'reduce' / 'navs_official_reducefix.csv.gz')):
    d = {}
    for col in ('E0|W10_70', 'XT|W10_70', 'XT_B11|W10_70'):
        s = e['strats'][col]; mine = out['versions'][m][col]
        d[col] = dict(eng_cagr19=s['std']['cagr_since2019'], mine_cagr19=mine['cagr19'], diff_pp=(mine['cagr19'] - s['std']['cagr_since2019']) * 100,
                      diff_full_pp=(mine['cagr_full'] - s['std']['cagr_full']) * 100, diff_maxdd_pp=(mine['maxdd'] - s['std']['maxdd']) * 100,
                      diff_paths_median_pp=(mine['paths']['cagr19_median'] - s['paths_summary']['cagr19_median']) * 100, n_closed=(s['std']['n_trades_closed'], mine['n_closed']),
                      targets_same=bool(s['targets'] == R[m]['strats'][col]['targets']))
    en = pd.read_csv(f, index_col=0, parse_dates=True)
    d['nav_max_abs_diff_by_col'] = {c: float((en[c] - NAV[m][c]).abs().max()) for c in en.columns}
    d['eng_0050_cagr19'] = e['strats']['E0|W10_70']['extras']['e50_since2019']; d['mine_0050_cagr19'] = out['versions'][m]['0050']['cagr19']
    cmpv[m] = d
out['vs_engineer'] = cmpv
# 持股期間碰到的事件（E0、XA）
ev = pd.read_csv(HERE / 'data' / 'reduce_events_mine.csv', dtype={'code': str}); ev['date'] = pd.to_datetime(ev['date'])
held = []; tdiff = []
for sid, col in (('E0', 'E0|W10_70'), ('XA', 'XT|W10_70'), ('XT', 'XT_B11|W10_70')):
    lb = pd.read_csv(HERE / 'out' / 'base' / f'trades_{sid}.csv', dtype={'code': str}); lB = pd.read_csv(HERE / 'out' / 'B' / f'trades_{sid}.csv', dtype={'code': str})
    out.setdefault('trade_cols', list(lb.columns))
    dcols = [c for c in lb.columns if 'date' in c or c in ('ent', 'exit')]
    ec = [c for c in lb.columns if c.startswith('ent') and 'date' in c] or [c for c in lb.columns if 'ent' in c]
    xc = [c for c in lb.columns if (c.startswith('exit') or c.startswith('ex')) and 'date' in c] or [c for c in lb.columns if 'exit' in c]
    ec, xc = ec[0], xc[0]
    for _, t in lb.iterrows():
        e = ev[(ev['code'] == t['code']) & (ev['date'] > pd.Timestamp(t[ec])) & (ev['date'] <= pd.Timestamp(t[xc]))]
        for _, r in e.iterrows():
            held.append(dict(strat=col, code=t['code'], ent=t[ec], exit=t[xc], why=t['why'], evt_date=str(r['date'].date()), reason=r['reason'], kind=r['kind'], evt_pre_close=r['evt_pre_close'], evt_ref=r['evt_ref'],
                             close=r['close'], open=r['open'], engine_R_pct=round(r['engine_R'] * 100, 3), true_R_pct=round(r['true_R'] * 100, 3), exact_holder_R_pct=round(r['exact_holder_R'] * 100, 3) if np.isfinite(r['exact_holder_R']) else None,
                             sold_at_open_that_day=bool(str(r['date'].date()) == str(t[xc])[:10]), base_ret=t['ret'], base_pnl=t['pnl']))
    kb = lb.assign(k=lb['code'] + '|' + lb[ec].astype(str)).set_index('k'); kB = lB.assign(k=lB['code'] + '|' + lB[ec].astype(str)).set_index('k')
    for k in sorted(set(kb.index) | set(kB.index)):
        a = kb.loc[k] if k in kb.index else None; c = kB.loc[k] if k in kB.index else None
        if a is None or c is None:
            tdiff.append(dict(strat=col, key=k, note='只在現行' if c is None else '只在 B')); continue
        if isinstance(a, pd.DataFrame) or isinstance(c, pd.DataFrame):
            tdiff.append(dict(strat=col, key=k, note='重複鍵')); continue
        if str(a[xc]) != str(c[xc]) or abs(a['ret'] - c['ret']) > 1e-4:
            tdiff.append(dict(strat=col, key=k, exit_base=a[xc], exit_B=c[xc], ret_base=round(a['ret'], 5), ret_B=round(c['ret'], 5), pnl_base=round(a['pnl']), pnl_B=round(c['pnl']), why_base=a['why'], why_B=c['why']))
pd.DataFrame(held).to_csv(HERE / 'held_events.csv', index=False); pd.DataFrame(tdiff).to_csv(HERE / 'trade_diff_base_vs_B.csv', index=False)
out['held_events'] = held; out['n_trade_diffs_base_vs_B'] = {c: sum(1 for x in tdiff if x['strat'] == c) for c in ('E0|W10_70', 'XT|W10_70', 'XT_B11|W10_70')}
out['e50_build'] = json.loads((HERE / 'data' / 'e50_build.json').read_text()); out['reduce_build'] = json.loads((HERE / 'data' / 'reduce_build.json').read_text())
json.dump(out, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=str)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(pd.DataFrame(rows).to_string()); print(Y.to_string())
print(json.dumps(out['additivity_E0_pp'], indent=1)); print(json.dumps(cmpv, ensure_ascii=False, indent=1))
print(pd.DataFrame(held).to_string()); print(pd.DataFrame(tdiff).to_string())
for m in MODES: print(m, out['versions'][m]['rfix'], [ (c, out['versions'][m][c]['months_changed']) for c in ('E0|W10_70','XT|W10_70','XT_B11|W10_70')], out['versions'][m]['E0|W10_70']['open_end'])

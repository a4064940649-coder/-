#!/usr/bin/env python3
"""對帳主程式：重現四組數字 → 開關逐一切換 → 正式版（上市）。輸出 results_recon.json。
開關與正式版定義事前登記在 FINDINGS.md 第 0 節。"""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rlib as L

KEYS = ('n', 'mean_pct', 'median_pct', 't_month', 't_month_gross', 't_date', 'gross_mean_pct', 'ex_top5pct_mean_pct')
SP = ('sel_2021-10_2023', 'val_2024_2026-10')


def run(cfg):
    T, st = L.trades(cfg, with_stat=True)
    b = L.by_per(T)
    return T, {'cfg': cfg, 'counts': st, **{p: {k: b[p].get(k) for k in KEYS} for p in (*SP, 'all_2021-10_2026-10')}}


res = {}
# ---------- 1. 重現 ----------
REPRO = {'events_base(ev5)': L.CFG_EV_BASE, 'events_strict(A+跌停延後)': L.CFG_EV_STRICT, 'events_B_merge': L.CFG_EV_B,
         'attention_R0(chain)': L.CFG_AT_R0, 'bear_longhistory_real': L.CFG_BEAR, 'exec_rebuild_events': L.CFG_EXEC}
res['reproduce'] = {k: run(c)[1] for k, c in REPRO.items()}
res['reproduce_targets'] = {
    'events_base(ev5)': 'sel n156 +3.11% med 1.60 t_month 1.58 t_date 2.79；val n565 +3.76% t_month 5.97（毛額）',
    'events_strict(A+跌停延後)': 'sel n150 +3.02% med 1.21 t_month 1.54；val n516 +2.87% t_month 5.22（毛額）',
    'events_B_merge': 'sel n150 +3.35% t_month 1.67；val n511 +3.67% t_month 5.80（毛額，未含跌停延後）',
    'attention_R0(chain)': 'sel n146 淨 +2.04% med 0.60 t_month 0.79；val n473 淨 +2.92% t_month 4.22',
    'bear_longhistory_real': '2021-10 起 n660 淨 +2.07% 中位 +1.51% t_month 2.72',
    'exec_rebuild_events': '660 筆合格事件；10 槽實際成交 504 筆、每筆淨 +3.17%、逐筆 t 5.0（原始報酬，不扣基準）'}

# exec_rebuild 的 10 槽原樣重現（匯入其 dlib，不改原檔）
try:
    sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
    import dlib as DL
    nav, st = DL.run_standalone(1e6, idle='cash')
    res['reproduce']['exec_rebuild_sleeve_dlib'] = {k: st.get(k) for k in ('kept', 'entered', 'skipped_no_slot', 'closed_trades', 'trade_net_mean_pct', 'trade_net_median_pct', 'trade_t', 'trade_win_rate', 'mean_held')}
except Exception as e:  # noqa
    res['reproduce']['exec_rebuild_sleeve_dlib'] = {'error': repr(e)}

# ---------- 2. 開關 ----------
ALT = {'gap': ['keep', 'drop'], 'overlap': ['none', 'first_undelayed', 'first', 'merge_all', 'merge_known', 'chain'],
       'entry': ['none', 'lock', 'ret9'], 'exit': ['none', 'lock', 'ret9', 'ret9_max20'],
       'cost': ['0', '0.5', '0.8', '0.8/1.2', '1.2'], 'bench': ['now', 'lag', 'raw', '0050'], 'liq': ['none', 'lag']}


def one_at_a_time(base, name):
    out = {}
    for k, vals in ALT.items():
        for v in vals:
            if base[k] == v: continue
            c = dict(base, **{k: v}); out[f'{k}={v}'] = run(c)[1]
    out['_base'] = run(base)[1]
    return out


res['switch_from_events_base'] = one_at_a_time(L.CFG_EV_BASE, 'ev')
res['switch_from_attention_R0'] = one_at_a_time(L.CFG_AT_R0, 'at')
res['switch_from_official'] = one_at_a_time(L.CFG_OFFICIAL, 'off')
# 累加路徑 events → attention（兩種順序）
def path(order):
    c = dict(L.CFG_EV_BASE); out = [('events_base', run(c)[1])]
    for k in order:
        c = dict(c, **{k: L.CFG_AT_R0[k]}); out.append((f'+{k}={L.CFG_AT_R0[k]}', run(c)[1]))
    return [{'step': s, **{p: r[p] for p in SP}} for s, r in out]
res['path_events_to_attention_A'] = path(['cost', 'overlap', 'entry', 'exit', 'bench', 'gap'])
res['path_events_to_attention_B'] = path(['gap', 'bench', 'exit', 'entry', 'overlap', 'cost'])
# 事件層面：chain 與 first/none 的樣本差在哪裡
TA, _ = L.trades(L.CFG_AT_R0, with_stat=True); TE, _ = L.trades(L.CFG_EV_BASE, with_stat=True)
ka = set(zip(TA['code'], TA['i0'])); ke = set(zip(TE['code'], TE['i0']))
sel = lambda d: d[(d['date'] >= L.SEL[0]) & (d['date'] <= L.SEL[1])]
TEs, TAs = sel(TE), sel(TA)
only_e = TEs[[(c, i) not in ka for c, i in zip(TEs['code'], TEs['i0'])]]
only_a = TAs[[(c, i) not in ke for c, i in zip(TAs['code'], TAs['i0'])]]
both_a = TAs[[(c, i) in ke for c, i in zip(TAs['code'], TAs['i0'])]]
both_e = TEs[[(c, i) in ka for c, i in zip(TEs['code'], TEs['i0'])]]
res['sample_diff_sel_events_vs_attention'] = {
    'only_in_events': {'n': int(len(only_e)), 'mean_gross_pct': round(only_e['gross'].mean() * 100, 2) if len(only_e) else None},
    'only_in_attention': {'n': int(len(only_a)), 'mean_gross_pct': round(only_a['gross'].mean() * 100, 2) if len(only_a) else None},
    'common_events_def_gross_pct': round(both_e['gross'].mean() * 100, 2), 'common_attention_def_gross_pct': round(both_a['gross'].mean() * 100, 2),
    'common_n': int(len(both_a)), 'common_attention_mean_hold': round(float(both_a['hold'].mean()), 2), 'common_events_mean_hold': round(float(both_e['hold'].mean()), 2)}

# ---------- 3. 正式版（上市） ----------
TO, sto = L.trades(L.CFG_OFFICIAL, with_stat=True)
off = {'counts': sto, 'by_period': L.by_per(TO), 'yearly': L.yearly(TO)}
off['by_period_raw_net'] = L.by_per(TO.assign(net=TO['raw_net']))
off['by_period_vs0050_net'] = L.by_per(TO.assign(net=TO['ab50'] - TO['cost']))
for c in ('0.5', '1.2'):
    T2 = L.trades(dict(L.CFG_OFFICIAL, cost=c)); off[f'cost_{c}pct'] = {p: L.by_per(T2)[p] for p in SP}
off['cost_1.5pct'] = {p: L.by_per(TO.assign(net=TO['gross'] - 0.015))[p] for p in SP}
off['excluding_src3_entry_or_exit'] = {p: L.by_per(TO[~TO['src3']])[p] for p in (*SP, 'val_b_2026')}
off['only_src3'] = L.stats(TO[TO['src3']])
# 公司行動旗標（3/4/5）在持有期內
cf = np.array([bool(np.isin(L.FN[r.i0 + 1:r.i1 + 1, r.j], [3, 4, 5]).any()) for r in TO.itertuples()])
off['corp_flag_in_hold'] = {'n': int(cf.sum()), 'ex_flag': {p: L.by_per(TO[~cf])[p] for p in SP}}
off['extended_positions'] = {'n': int((TO['n_links'] > 1).sum()), 'extended_only': L.stats(TO[TO['n_links'] > 1]), 'single_only': L.stats(TO[TO['n_links'] == 1])}
off['second_disposal'] = {'second': L.stats(TO[TO['second']]), 'first': L.stats(TO[~TO['second']])}
off['delayed_exit_n'] = int(TO['delayed'].sum())
top = TO.sort_values('net', ascending=False).head(10)
off['top10'] = [{'code': r.code, 'date': str(r.date.date()), 'exit': str(r.exit.date()), 'raw_pct': round(r.raw * 100, 1), 'net_pct': round(r.net * 100, 1)} for r in top.itertuples()]
bot = TO.sort_values('net').head(5)
off['bottom5'] = [{'code': r.code, 'date': str(r.date.date()), 'exit': str(r.exit.date()), 'raw_pct': round(r.raw * 100, 1), 'net_pct': round(r.net * 100, 1)} for r in bot.itertuples()]
off['monthly_mean_net_pct'] = {str(k): round(v * 100, 2) for k, v in TO.groupby(TO['date'].dt.to_period('M'))['net'].mean().items()}
off['monthly_n'] = {str(k): int(v) for k, v in TO.groupby(TO['date'].dt.to_period('M')).size().items()}
res['official_TWSE'] = off
TO.assign(date=TO['date'].dt.strftime('%Y-%m-%d'), exit=TO['exit'].dt.strftime('%Y-%m-%d'), ann_date=TO['ann_date'].dt.strftime('%Y-%m-%d')).drop(columns=['j']).to_csv(HERE / 'official_trades_twse.csv', index=False)


def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, pd.Timestamp): return str(o.date())
    return str(o)


(HERE / 'results_recon.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')

# ---------- 列印 ----------
def line(name, r):
    s = r['sel_2021-10_2023']; v = r['val_2024_2026-10']
    print(f"{name:34s} sel n{s.get('n'):>4} net{s.get('mean_pct'):>6} med{s.get('median_pct'):>6} tm{s.get('t_month')!s:>6} tmg{s.get('t_month_gross')!s:>6} td{s.get('t_date')!s:>6} | "
          f"val n{v.get('n'):>4} net{v.get('mean_pct'):>6} med{v.get('median_pct'):>6} tm{v.get('t_month')!s:>6} tmg{v.get('t_month_gross')!s:>6}")
print('== reproduce'); [line(k, v) for k, v in res['reproduce'].items() if 'sel_2021-10_2023' in v]
print(res['reproduce'].get('exec_rebuild_sleeve_dlib'))
for sec in ('switch_from_events_base', 'switch_from_attention_R0', 'switch_from_official'):
    print('==', sec); [line(k, v) for k, v in res[sec].items()]
for sec in ('path_events_to_attention_A', 'path_events_to_attention_B'):
    print('==', sec); [line(x['step'], x) for x in res[sec]]
print(res['sample_diff_sel_events_vs_attention'])
print('== official', sto)
for p, x in off['by_period'].items(): print(p, x)
for y, x in off['yearly'].items(): print(y, {k: x.get(k) for k in ('n', 'mean_pct', 'median_pct', 't_month', 'ex_top5pct_mean_pct', 'hit')})
for k in ('cost_0.5pct', 'cost_1.2pct', 'cost_1.5pct', 'excluding_src3_entry_or_exit', 'by_period_raw_net', 'by_period_vs0050_net'):
    print(k, {p: {q: off[k][p].get(q) for q in ('n', 'mean_pct', 'median_pct', 't_month')} for p in SP})
print('src3 only', {q: off['only_src3'].get(q) for q in ('n', 'mean_pct', 't_month')}, 'corp', off['corp_flag_in_hold']['n'])
print('ext', {q: off['extended_positions']['extended_only'].get(q) for q in ('n', 'mean_pct', 'median_pct')}, 'second', {q: off['second_disposal']['second'].get(q) for q in ('n', 'mean_pct')})
print(off['top10']); print(off['bottom5'])

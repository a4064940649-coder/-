#!/usr/bin/env python3
"""依 PREREG.decision 機械套用判定（不改任何數字）：讀 RESULTS_PAPER、RESULTS_V7X，寫 results.json 的 DECISION。python3 -I -B s9_decide.py"""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
PR, RP, RV = R['PREREG'], R['RESULTS_PAPER'], R['RESULTS_V7X']
ST, TB = RP['stats'], RV['table']
cov = PR['coverage_known_at_freeze']
c13, c1318 = cov['slots_2013_2026'], cov['slots_2013_2018']
ge80 = c13['share'] >= 0.8 and c1318['share'] >= 0.8
def P(k, c='normal', e='cons_main'): return ST[f'{k}|{c}|{e}|vs_C0']
main, c0 = P('ALLPIT_REAL_opt'), P('C0', 'normal', 'freeze')
rows = []
for nm, f in (('全期年化', lambda s: s['ALL']['ann']), ('t', lambda s: s['ALL']['t_vs_0050']), ('2006～2012 超額', lambda s: s['SEG1']['gex']),
              ('2013～2018 超額', lambda s: s['SEG2']['gex']), ('2019 起超額', lambda s: s['SEG3']['gex'])):
    a, b = f(c0), f(main); lim = 0.05 if nm == 't' else 0.3
    rows.append(dict(part='C0 紙上（一般成本）', field=nm, old=a, new=b, delta=b - a, limit=lim, exceed=abs(b - a) > lim))
for c in ('stress',):
    a, b = P('C0', c, 'freeze'), P('ALLPIT_REAL_opt', c)
    for nm, k in (('全期年化（壓力）', 'ann'), ('t（壓力）', 't_vs_0050')):
        rows.append(dict(part='C0 紙上（壓力成本，描述）', field=nm, old=a['ALL'][k], new=b['ALL'][k], delta=b['ALL'][k] - a['ALL'][k], limit=0.05 if 't' in nm else 0.3, exceed=None))
def V(v, e, c, t, f): return TB[f'{v}|{e}|{c}|{t}'][f]
for nm, (c, t, f) in {'全期超額 j 中位數（100 萬一般）': ('normal', '100萬', 'exc_med'), 'j p10（100 萬一般）': ('normal', '100萬', 'exc_p10'),
                      '2019 起超額 j 中位數（100 萬一般）': ('normal', '100萬', 'exc_Y19on_med'), '全期超額 j 中位數（100 萬壓力）': ('stress', '100萬', 'exc_med'),
                      '全期超額 j 中位數（300 萬）': ('normal', '300萬', 'exc_med'), '全期超額 j 中位數（1,000 萬）': ('normal', '1000萬', 'exc_med')}.items():
    a, b = V('orig', 'orig', c, t, f), V('REAL_opt', 'cons_main', c, t, f)
    rows.append(dict(part='V7X 可執行（對現行正式 orig×orig）', field=nm, old=a, new=b, delta=b - a, limit=0.3, exceed=abs(b - a) > 0.3))
replace = any(r['exceed'] for r in rows if r['exceed'] is not None)
def ivl(key, f):
    xs = [f(P(f'ALLPIT_REAL_{x}')) for x in ('opt', 'p90', 'carry')]
    return dict(opt=xs[0], p90=xs[1], carry=xs[2], lo=min(xs), hi=max(xs))
IV = {'ann': ivl('ann', lambda s: s['ALL']['ann']), 't': ivl('t', lambda s: s['ALL']['t_vs_0050']),
      'seg1': ivl('s1', lambda s: s['SEG1']['gex']), 'seg2': ivl('s2', lambda s: s['SEG2']['gex']), 'seg3': ivl('s3', lambda s: s['SEG3']['gex'])}
IVS = {k: dict(zip(('opt', 'p90', 'carry'), [P(f'ALLPIT_REAL_{x}', 'stress')[s]['gex' if s != 'ALL' else 'ann'] for x in ('opt', 'p90', 'carry')])) for k, s in (('ann', 'ALL'), ('seg2', 'SEG2'))}
IVV = {f: {x: V(f'REAL_{x}', 'cons_main', 'normal', '100萬', f) for x in ('opt', 'p90', 'carry')} for f in ('exc_med', 'exc_p10', 'exc_Y19on_med', 't_0050_med')}
same_sign2 = (IV['seg2']['lo'] > 0) == (IV['seg2']['hi'] > 0)
G4txt = (f"已用實際營收排除（覆蓋 {c1318['share']:.0%}，報酬權重 {c1318['wshare']:.0%}）" if (ge80 and same_sign2) else
         f"存活者偏誤未排除：實際營收覆蓋 2013～2018 槽位 {c1318['share']:.1%}（報酬權重 {c1318['wshare']:.1%}）< 80%；2013～2018 超額區間 opt {IV['seg2']['opt']:.2f}／p90 {IV['seg2']['p90']:.2f}／carry {IV['seg2']['carry']:.2f}pp（壓力成本 opt {IVS['seg2']['opt']:.2f}pp）")
chk = dict(paper_t_ge_2643=main['ALL']['t_vs_0050'] >= 2.643, v7x_t_ge_1=V('REAL_opt', 'cons_main', 'normal', '100萬', 't_0050_med') >= 1.0,
           v7x_y19_pos=V('REAL_opt', 'cons_main', 'normal', '100萬', 'exc_Y19on_med') > 0)
grade = '可用（勘誤生效）' if ge80 else '待觀察'
DEC = dict(coverage_ge_80=ge80, branch='opt 補缺版（不利端）＋p90／carry 區間' if not ge80 else 'ALLPIT_REAL（未覆蓋用 opt）',
           new_official_paper=dict(variant='ALLPIT_REAL_opt', engine='cons_main', cost='normal', ann=main['ALL']['ann'], t=main['ALL']['t_vs_0050'],
                                   seg=[main['SEG1']['gex'], main['SEG2']['gex'], main['SEG3']['gex']], hold=main['HOLD']['gex'], t_drop_best3=main['ALL']['t_drop_best3'],
                                   mdd_window=main['ALL']['mdd'], mdd_daily=main['ALL']['mdd_daily'], stress_ann=P('ALLPIT_REAL_opt', 'stress')['ALL']['ann'], stress_t=P('ALLPIT_REAL_opt', 'stress')['ALL']['t_vs_0050']),
           interval_paper=IV, interval_paper_stress=IVS, interval_v7x_100m_normal=IVV,
           new_official_v7x={f'{c}|{t}': {f: V('REAL_opt', 'cons_main', c, t, f) for f in ('CAGR_med', 'exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 'exc_Y19on_p10', 't_Y19on_med', 'exc_HOLD_med', 'exc_S0612_med', 'exc_S1318_med', 'exc_S19on_med', 'maxDD_med')}
                             for c in ('normal', 'stress') for t in ('100萬', '300萬', '1000萬')},
           replace_rows=rows, replace=replace, replace_text='取代（不只加註）；舊數字移附錄，標「含 2013 年後營收存活者偏誤」' if replace else '不取代',
           G4_note=G4txt, data_alarm_checks=chk, grade=grade)
R['DECISION'] = DEC
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in DEC.items() if k not in ('replace_rows', 'new_official_v7x')}, ensure_ascii=False, indent=1, default=str)[:4000])
for r in rows: print(r['part'], r['field'], round(r['old'], 3), '→', round(r['new'], 3), round(r['delta'], 3), r['exceed'])

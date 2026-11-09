#!/usr/bin/env python3
"""正式執行（只跑一次）：紙上 ia（patched ia_core 副本，DISP_OFFICIAL 選項；cz_engine._sim_cons＋cz3 v3 規則）。
C0×{orig, union, replace, union_noovl, union14}、U_opt×{orig, union}；引擎 freeze／v3_main／v3_lo／v3_hi／o17 cons_main；一般／壓力。
斷言 PREREG 存在且雜湊相符、RESULTS_IA 尚不存在。python3 -I -B s9a_ia.py"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'RESULTS_IA' not in R, '只跑一次'
IS = R['PREREG']['input_sha256']
for f in ('events_v3.csv', 'cz3.py', 'dispmask_official.py', 'ia_lib.py', 'ia_core_dispmask_official.patch'): assert sha(HERE / f) == IS[f], f
assert sha(AG / 'independent_audit' / 'ia_core.py') == IS['ia_core.py']
T0 = time.time()
sys.path.insert(0, str(HERE)); import ia_lib as IL; sys.path.pop(0)
P, CE = IL.load(patched=True); K = P.K; COLS = P.COLS; IDX = P.IDX
sys.path.insert(0, str(HERE)); import cz3, dispmask_official as DO; sys.path.pop(0)
assert cz3.CE is CE
E2, E3 = cz3.load_E()
kw = dict(TR=K.TRADED, IDX=K.IDX, Cn=K.Cn, CI=K.CI, On=K.On)
ST = {'v3_main': cz3.build(E2, E3, 'main', **kw), 'v3_lo': cz3.build(E2, E3, 'lo', **kw), 'v3_hi': cz3.build(E2, E3, 'hi', **kw),
      'o17_cons_main': CE.build_settlement(E2, 'main', d=0.057)}
def mp(mode):
    """union_noovl／union14：monkeypatch（不在 patch 內，只作對照）。"""
    f = DO.make_mask(K._disposal_mask_5y, IDX, K.CI, K.N, mode)
    def run(fn):
        old = K.disposal_mask; K.disposal_mask = f
        try: return fn()
        finally: K.disposal_mask = old
    return run
B = {}
for m in (None, 'union', 'replace'):
    B[('C0', m or 'orig')] = IL.with_mask(P, m, lambda: IL.c0_plan(P))
for m in ('union_noovl', 'union14'):
    B[('C0', m)] = mp(m)(lambda: IL.c0_plan(P))
B[('U_opt', 'orig')] = IL.uopt_plan(P)
B[('U_opt', 'union')] = IL.with_mask(P, 'union', lambda: IL.uopt_plan(P))
print('lists', round(time.time() - T0), flush=True)
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
RUN = {('C0', 'orig'): {'freeze': 'ns', 'v3_main': 'ns', 'v3_lo': 'ns', 'v3_hi': 'n'}, ('C0', 'union'): {'freeze': 'ns', 'v3_main': 'ns', 'v3_lo': 'ns', 'v3_hi': 'n'},
       ('C0', 'replace'): {'freeze': 'ns'}, ('C0', 'union_noovl'): {'freeze': 'n'}, ('C0', 'union14'): {'freeze': 'n'},
       ('U_opt', 'orig'): {'freeze': 'n', 'o17_cons_main': 'ns', 'v3_main': 'ns', 'v3_lo': 'ns', 'v3_hi': 'n'}, ('U_opt', 'union'): {'v3_main': 'ns', 'v3_lo': 'n', 'v3_hi': 'n'}}
NAV = {}; LOG = {}
for bk, egs in RUN.items():
    pl = [(p[0], p[1]) for p in B[bk]['plan']]
    for eg, cs in egs.items():
        for c in (['normal'] if cs == 'n' else ['normal', 'stress']):
            if eg == 'freeze': nav, lg = CE.simulate(pl, *COST[c], B[bk]['t0'], B[bk]['t1'], mode='freeze')
            else: nav, lg = CE.simulate(pl, *COST[c], B[bk]['t0'], B[bk]['t1'], mode='consideration', settle=ST[eg])
            NAV[bk + (eg, c)] = nav
            if c == 'normal':
                LOG['|'.join(bk + (eg,))] = dict(n_frozen=len(lg['frozen']), frozen_max_share=max([x[2] for x in lg['frozen']] or [0.0]),
                                                settled=[(s['code'], s['type'], s['date'], round(s['value_lastclose'], 6), s['parts']) for s in lg.get('settled', [])])
print('sims', round(time.time() - T0), flush=True)
OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')
CHK = {c: float(np.nanmax(np.abs(NAV[('C0', 'orig', 'freeze', c)].reindex(OFF[f'EXFIN|n10|{c}'].dropna().index).to_numpy() - OFF[f'EXFIN|n10|{c}'].dropna().to_numpy()))) for c in COST}
assert max(CHK.values()) < 1e-9, CHK
SD = B[('C0', 'orig')]['sd']
WIN = {k: IL.win(v, B[k[:2]]['sd']) for k, v in NAV.items()}
STATS = {}
for k, W in WIN.items():
    ref = WIN[('C0', 'orig', 'freeze', k[3])]
    J = ref.join(W[['r']], rsuffix='_k', how='inner')
    st = IL.stats_one(J['r_k'].to_numpy(), J['r50'].to_numpy(), J['r'].to_numpy(), J.index, J['yr'].to_numpy())
    st['ALL']['mdd_daily'] = float((NAV[k] / NAV[k].cummax() - 1).min() * 100)
    STATS['|'.join(k)] = st
# 描述：處置遮罩異動槽（移出／移入股的下一期報酬＝成交日收盤→下一成交日收盤含息，ia GROW）
def slot_desc(a, b):
    PA = {p[3]: (p[0], [COLS[j] for j in p[1]]) for p in B[a]['plan']}; PB = {p[3]: (p[0], [COLS[j] for j in p[1]]) for p in B[b]['plan']}
    days = sorted(PA); ex = [PA[d][0] for d in days] + [IDX.get_loc(pd.Timestamp('2026-10-12'))]
    rows = []
    for i, d in enumerate(days):
        e, e2 = PA[d][0], ex[i + 1]
        g = lambda c: float(np.prod(K.GROW[e + 1:e2 + 1, K.CI[c]]) - 1)
        out_ = sorted(set(PA[d][1]) - set(PB[d][1])); in_ = sorted(set(PB[d][1]) - set(PA[d][1]))
        if out_ or in_:
            ro = [g(c) for c in out_]; ri = [g(c) for c in in_]
            rows.append(dict(sig=str(d.date()), out=out_, inn=in_, r_out=ro, r_in=ri, in_minus_out_pp=(sum(ri) - sum(ro)) * 100, port_pp=(sum(ri) - sum(ro)) * 10))
    df = pd.DataFrame(rows)
    allr = [x for r in rows for x in r['r_out']]
    return dict(rows=rows, n_out=int(sum(len(r['out']) for r in rows)), mean_r_out=float(np.mean(allr)) if allr else None,
                sum_in_minus_out_pp=float(df.in_minus_out_pp.sum()) if len(df) else 0.0,
                sum_in_minus_out_pp_2019on=float(df[df.sig >= '2019-01-01'].in_minus_out_pp.sum()) if len(df) else 0.0,
                by_year={int(y): float(v) for y, v in df.groupby(df.sig.str[:4].astype(int)).port_pp.sum().items()} if len(df) else {},
                top_dates=df.reindex(df.port_pp.abs().sort_values(ascending=False).index).head(5)[['sig', 'port_pp']].to_dict('records') if len(df) else [])
DESC = {f'C0|{m}': slot_desc(('C0', 'orig'), ('C0', m)) for m in ('union', 'replace', 'union_noovl', 'union14')}
DESC['U_opt|union'] = slot_desc(('U_opt', 'orig'), ('U_opt', 'union'))
R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS_IA' not in R
R['RESULTS_IA'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - T0, 1), C0_reproduce=CHK, stats=STATS, logs=LOG, slot_desc=DESC,
                       lists_sha={'|'.join(k): hashlib.sha256(json.dumps({str(p[3].date()): [COLS[j] for j in p[1]] for p in v['plan']}, sort_keys=True).encode()).hexdigest() for k, v in B.items()})
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('done', round(time.time() - T0), 's', CHK)

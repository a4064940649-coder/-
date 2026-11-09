#!/usr/bin/env python3
"""o18 題 1 正式執行（紙上 ia 口徑，只跑一次）：斷言 PREREG 存在、雜湊相符、RESULTS_PAPER 尚不存在、輸入檔 sha 相符。
變體見 PREREG.paper.variants；引擎 cons_main（主）、cons_lo、cons_hi、freeze（對照）；成本一般與壓力。
python3 -I -B run_paper.py"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
RJ = HERE / 'results.json'
T0 = time.time()
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R, 'PREREG 不存在：不得計算'
PR = R['PREREG']
assert hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改'
assert sha(HERE / 'SPEC.json') == R['SPEC_sha256'], 'SPEC.json 被改'
assert 'RESULTS_PAPER' not in R, '只跑一次'
for f, h in PR['file_sha256'].items():
    assert sha(HERE / f if not f.startswith('/') else f) == h, f'{f} 雜湊不符'
sys.path.insert(0, str(HERE)); import o18lib as O; sys.path.pop(0)
P, V, K, COLS = O.P, O.V, O.K, O.COLS
BV = PR['backfill_version']
KY = PR['ky_codes']; TOP10 = PR['top10']
VARS = {k: (tuple(v['kind']), tuple(v.get('excl', []))) for k, v in PR['paper']['variants'].items()}
ENG = PR['paper']['engines']
B = {}
for k, (kind, ex) in VARS.items():
    B[k] = O.build(tuple(None if x == 'None' else x for x in kind), excl=ex); log('名單', k, len(B[k]['plan']))
SD0 = B['C0']['sd']
assert all(B[k]['sd'] == SD0 for k in VARS), '訊號日不一致'
PK = {k: {p[3]: [COLS[j] for j in p[1]] for p in B[k]['plan']} for k in VARS}
DAYS = list(PK['C0'])
def ins(k, a='2013-01-01', b='2026-12-31', which='base_missing'):
    out = []
    for p in B[k]['plan']:
        d = p[3]
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            A_ = set(B[k][which].get(d, [])) if which else set()
            out += [(str(d.date()), COLS[j]) for j in p[1] if COLS[j] in A_]
    return out
# P2：放回入選槽 ≤ U_opt
CHK = {}
for k in VARS:
    if '_REAL_' in k:
        n13, n1318 = len(ins(k)), len(ins(k, '2013-01-01', '2018-12-31'))
        CHK[k] = dict(ins_2013on=n13, ins_1318=n1318, ins_fill=len(ins(k, which='aff')), ins_0613=len(ins(k, '2006-01-01', '2012-12-31', 'base_missing')) if k.startswith(('UFULL', 'ALLPIT')) else None,
                      changed_vs_C0=int(sum(len(set(PK['C0'][d]) - set(PK[k][d])) for d in DAYS)))
        if k.startswith('U13'): assert n13 <= 251 and n1318 <= 227, ('P2 違反', k, n13, n1318)
log('P2', CHK)
NAV, LOGS = {}, {}
for k in VARS:
    for eg, costs in ENG[PR['paper']['engine_set'][k]].items():
        for c in costs:
            nav, lg = O.sim(B[k], c, eg); NAV[(k, c, eg)] = nav
            if c == 'normal':
                LOGS[f'{k}|{eg}'] = dict(n_frozen=len(lg.get('frozen', [])), frozen_codes=sorted({x for _, cs, _ in lg.get('frozen', []) for x in cs}),
                                         frozen_max_share=max([x[2] for x in lg.get('frozen', [])] or [0.0]),
                                         settled_codes=sorted({s['code'] for s in lg.get('settled', [])}),
                                         settled_types=pd.Series([s['type'] for s in lg.get('settled', [])]).value_counts().to_dict() if lg.get('settled') else {})
    log('模擬', k)
OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')
for c in ('normal', 'stress'):
    ref = OFF[f'EXFIN|n10|{c}'].dropna(); nav = NAV[('C0', c, 'freeze')]
    assert float(np.nanmax(np.abs(nav.reindex(ref.index).to_numpy() - ref.to_numpy()))) < 1e-12, 'C0 未重現'
WIN = {key: O.win(nav, SD0) for key, nav in NAV.items()}
ST = {}
for (k, c, eg), W in WIN.items():
    refs = [('C0', c, 'freeze')]
    if k.startswith('ALLPIT_REAL') and ('ALLPIT_opt_o17', c, 'cons_main') in WIN: refs.append(('ALLPIT_opt_o17', c, 'cons_main'))
    if k.startswith('U13_REAL') and ('U13_opt_o17', c, 'cons_main') in WIN: refs.append(('U13_opt_o17', c, 'cons_main'))
    for rk in refs:
        s = O.stats_pair(W, WIN[rk]); nav = NAV[(k, c, eg)]
        s['ALL']['mdd_daily'] = float((nav / nav.cummax() - 1).min() * 100)
        ST[f'{k}|{c}|{eg}|vs_{rk[0]}'] = s
log('統計')
# ---------------- P4 子樣本：放回槽 vs 被擠掉的 C0 股（下一期＝訊號日收盤→下一訊號日收盤，含息還原價；描述）
ADJ = K.ADJ
SDI = pd.DatetimeIndex(SD0)
def r_next(c, d):
    i = SDI.get_loc(d)
    if i + 1 >= len(SDI) or c not in ADJ.columns: return np.nan
    a = ADJ[c].loc[:d].dropna(); b = ADJ[c].loc[:SDI[i + 1]].dropna()
    if not len(a) or not len(b) or a.index[-1] < d - pd.Timedelta(days=10): return np.nan
    return float(b.iloc[-1] / a.iloc[-1] - 1)
def p4(k):
    rows = []
    for p in B[k]['plan']:
        d = p[3]
        if d < pd.Timestamp('2013-01-01'): continue
        bm = set(B[k]['base_missing'].get(d, [])); fill = set(B[k]['aff'].get(d, []))
        put = [c for c in PK[k][d] if c in bm]
        if not put: continue
        disp = sorted(set(PK['C0'][d]) - set(PK[k][d]))
        rd = np.nanmean([r_next(x, d) for x in disp]) if disp else np.nan
        for c in put:
            dd = P.delist_date(c); mt = None if pd.isna(dd) else (dd - d).days / 30.44
            g = 'survivor' if mt is None else ('le6' if mt <= 6 else ('m6_12' if mt <= 12 else 'gt12'))
            rows.append(dict(sig=str(d.date()), code=c, grp=g, how='fill' if c in fill else 'real', r_put=r_next(c, d), r_disp=rd, diff=r_next(c, d) - rd))
    D = pd.DataFrame(rows)
    out = {'n': int(len(D))}
    if not len(D): return out, D
    def g_(x):
        z = x.groupby('sig')['diff'].mean().dropna()
        return dict(n_slots=int(len(x)), n_days=int(z.size), mean_diff_pct=float(x['diff'].mean() * 100), mean_put_pct=float(x.r_put.mean() * 100),
                    mean_disp_pct=float(x.r_disp.mean() * 100), t_by_day=O.tt(z.to_numpy()))
    out['by_grp'] = {gg: g_(x) for gg, x in D.groupby('grp')}
    out['by_how'] = {gg: g_(x) for gg, x in D.groupby('how')}
    D['contrib_pp'] = D['diff'] / 10 * 100
    cb = D.groupby('code').contrib_pp.sum().sort_values()
    out['contrib_by_issuer_pp_sum'] = cb.round(3).to_dict()
    tot = cb.sum(); out['contrib_total_pp_sum'] = float(tot)
    out['top3_share_of_total'] = float(cb.reindex(['2325', '2311', '1704']).fillna(0).sum() / tot) if tot != 0 else None
    out['seg'] = {s: g_(x) for s, x in D.groupby(np.where(D.sig <= '2018-12-31', '2013_2018', '2019_2026'))}
    return out, D
P4 = {}
for k in ('U13_REAL_opt', 'ALLPIT_REAL_opt', 'U13_REAL_p90', 'U13_REAL_carry'):
    P4[k], DD = p4(k)
    if k == 'ALLPIT_REAL_opt': DD.to_csv(HERE / 'data/p4_slots_ALLPIT_REAL_opt.csv', index=False)
log('P4')
R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'RESULTS_PAPER' not in R
R['RESULTS_PAPER'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - T0, 1), checks=CHK, logs=LOGS, stats=ST, P4=P4,
                          lists_sha={k: hashlib.sha256(json.dumps({str(d.date()): v for d, v in PK[k].items()}, sort_keys=True).encode()).hexdigest() for k in VARS})
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame({f'{k}|{c}|{eg}': v for (k, c, eg), v in NAV.items()}).to_parquet(Path(PR['scratch']) / 'navs_paper.parquet')
log('寫入 results.json RESULTS_PAPER')
def show(k, c='normal', eg='cons_main'):
    s = ST.get(f'{k}|{c}|{eg}|vs_C0')
    if s: print(k, c, eg, 'ann %.2f t %.3f seg %.2f/%.2f/%.2f 19on %.2f d_ann %.2f tdiff %s' % (s['ALL']['ann'], s['ALL']['t_vs_0050'], s['SEG1']['gex'], s['SEG2']['gex'], s['SEG3']['gex'], s['SEG3']['gex'], s['ALL']['ann_diff_vs_ref'], s['ALL']['t_diff_vs_ref']))
for k in VARS: show(k)

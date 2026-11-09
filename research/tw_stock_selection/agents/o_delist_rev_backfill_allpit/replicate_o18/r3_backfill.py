#!/usr/bin/env python3
"""G6 複製員 r3：閘門（只在同期存活公司上與 tw_revenue 比對，依年分期）＋補回規則 B1 → 自己的補回表；與研究員 rev_backfill_B1.csv 逐格比對。
python3 -I -B r3_backfill.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TOP = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o18')
KY = {'1262', '1592', '6422', '6452'}

tw = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str, 'market': str})
tw['ym'] = tw.month.str[:7]; tw['pri'] = (tw.market != 'sii').astype(int)
tw = tw.sort_values(['ym', 'code', 'pri']).drop_duplicates(['ym', 'code'])
TW = dict(zip(zip(tw.code, tw.ym), tw.revenue_k_twd))
def pm(ym, k): return str(pd.Period(ym, 'M') + k)

A = pd.read_csv(SCR / 'r2_A.csv', dtype={'code': str, 'ym': str})
# A 各欄 → 目標月
Acand = {'rev': {}, 'prev': {}, 'ly': {}}
for c, ym, rv, pv, lv in zip(A.code, A.ym, A.rev, A.prev, A.ly):
    if np.isfinite(rv): Acand['rev'][(c, ym)] = rv
    if np.isfinite(pv): Acand['prev'].setdefault((c, pm(ym, -1)), pv)
    if np.isfinite(lv): Acand['ly'].setdefault((c, pm(ym, -12)), lv)
def gate(dct, tol=0.001):
    by = {}
    for (c, ym), v in dct.items():
        t = TW.get((c, ym))
        if t is None or not np.isfinite(t) or t <= 0: continue
        by.setdefault(ym[:4], []).append(abs(v / t - 1) <= tol)
    return {y: dict(n=len(x), ok=float(np.mean(x))) for y, x in sorted(by.items())}
G = {k: gate(v) for k, v in Acand.items()}
# YOY 欄（描述）：對 tw_revenue 水準法年增，|差| ≤ 0.1pp？——規格只說不採用；這裡以 ±0.01 四捨五入容忍報描述
yoyc = {}
for c, ym, yy in zip(A.code, A.ym, A.yoy):
    a, b = TW.get((c, ym)), TW.get((c, pm(ym, -12)))
    if np.isfinite(yy) and a is not None and b is not None and b > 0:
        yoyc.setdefault(ym[:4], []).append(abs((a / b - 1) * 100 - yy) <= 0.0051 + 1e-9)
G['YOY'] = {y: dict(n=len(x), ok=float(np.mean(x))) for y, x in sorted(yoyc.items())}
PASS = {k: {y: g['ok'] >= 0.98 for y, g in G[k].items()} for k in ('rev', 'prev', 'ly')}

# ============================== N
VAR = sys.argv[1] if len(sys.argv) > 1 else 'base'
TOLMODE = sys.argv[2] if len(sys.argv) > 2 else 'rel'
NV = pd.read_csv(SCR / ('r2_N.csv.gz' if VAR == 'base' else f'r2_N_{VAR}.csv.gz'), dtype={'code': str, 'ym': str, 'date': str})
def tol_round(v):
    # 億元 2 位小數 → 0.005 億＝500 千元；萬元整數 → 5 千元（以 rev_k 的最後有效位推）
    return 500.0 + 1e-6 if (abs(v / 1e5 * 100 - round(v / 1e5 * 100)) < 1e-6) else 5.0
def ok_single(c, ym, v, u):
    t = TW.get((c, ym))
    if t is None or not np.isfinite(t): return None
    if TOLMODE == 'unit': return abs(v - t) <= u * (1 + 1e-9)   # 報導最後一位的 1 個單位
    return t > 0 and abs(v / t - 1) <= 0.006                      # 相對 0.6%（與共識帶寬同）
sg = {}
for c, ym, v, u in zip(NV.code, NV.ym, NV.rev_k, NV.unit_k):
    o = ok_single(c, ym, v, u)
    if o is not None: sg.setdefault('all', []).append(o)
G['N_single'] = dict(n=len(sg.get('all', [])), ok=float(np.mean(sg['all'])))
cons = {}; consu = {}
for (c, ym), g in NV.groupby(['code', 'ym']):
    if g.seg.nunique() < 2: continue
    v = g.rev_k.to_numpy()
    if v.max() / v.min() - 1 > 0.006: continue
    cons[(c, ym)] = float(np.median(v)); consu[(c, ym)] = float(g.unit_k.max())
cg = {}
for (c, ym), v in cons.items():
    o = ok_single(c, ym, v, consu[(c, ym)])
    if o is not None: cg.setdefault(ym[:4], []).append(o)
G['N_cons_level'] = {y: dict(n=len(x), ok=float(np.mean(x))) for y, x in sorted(cg.items())}
PASS['N'] = {y: g['ok'] >= 0.98 for y, g in G['N_cons_level'].items()}

# ============================== B1
DEM = pd.read_csv(SCR / 'r1_aff.csv', dtype=str)
ISS = sorted(set(DEM.code) - KY)
lv = {}
for c in ISS:
    months = set()
    for k in Acand: months |= {ym for (cc, ym) in Acand[k] if cc == c}
    months |= {ym for (cc, ym) in cons if cc == c}
    months |= {ym for (cc, ym) in TW if cc == c}
    for ym in months:
        y = ym[:4]
        if (c, ym) in TW and np.isfinite(TW[(c, ym)]): lv[(c, ym)] = (TW[(c, ym)], 'tw_revenue'); continue
        if (c, ym) in Acand['rev'] and Acand['rev'][(c, ym)] > 0 and PASS['rev'].get(y, False): lv[(c, ym)] = (Acand['rev'][(c, ym)], 'A.rev'); continue
        if (c, ym) in Acand['prev'] and Acand['prev'][(c, ym)] > 0 and PASS['prev'].get(y, False): lv[(c, ym)] = (Acand['prev'][(c, ym)], 'A.prev'); continue
        if (c, ym) in Acand['ly'] and Acand['ly'][(c, ym)] > 0 and PASS['ly'].get(y, False): lv[(c, ym)] = (Acand['ly'][(c, ym)], 'A.ly'); continue
        if (c, ym) in cons and PASS['N'].get(y, False): lv[(c, ym)] = (cons[(c, ym)], 'N.level'); continue
rows = []
for (c, ym), (v, s) in sorted(lv.items()):
    b = lv.get((c, pm(ym, -12)))
    yy, ys = np.nan, ''
    if b is not None and b[0] > 0 and not (s == 'tw_revenue' and b[1] == 'tw_revenue'):
        yy = (v / b[0] - 1) * 100; ys = f'{s}/{b[1]}'
    rows.append(dict(code=c, ym=ym, rev_k=v, rev_src=s, yoy_pct=yy, yoy_src=ys))
B = pd.DataFrame(rows)
B.to_csv(HERE / ('my_backfill.csv' if VAR == 'base' else f'my_backfill_{VAR}.csv'), index=False)

# ============================== 與研究員 B1 比對
R = pd.read_csv(TOP / 'data' / 'rev_backfill_B1.csv', dtype={'code': str, 'ym': str})
my_y = B[B.yoy_pct.notna()].set_index(['code', 'ym']).yoy_pct
re_y = R[R.yoy_pct.notna()].set_index(['code', 'ym']).yoy_pct
U = my_y.index.union(re_y.index)
cmp = []
for k in U:
    a, b = my_y.get(k, np.nan), re_y.get(k, np.nan)
    agree = (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-6) or (not np.isfinite(a) and not np.isfinite(b))
    cmp.append(dict(code=k[0], ym=k[1], mine=a, theirs=b, agree=bool(agree)))
CMP = pd.DataFrame(cmp)
dis = CMP[~CMP.agree]
# 水準：研究員 B1 有列的 issuer-month（非 tw_revenue）
my_l = B[B.rev_src != 'tw_revenue'].set_index(['code', 'ym'])
re_l = R[R.rev_src != 'tw_revenue'].set_index(['code', 'ym'])
UL = my_l.index.union(re_l.index)
lag = [(k, my_l.rev_k.get(k, np.nan), re_l.rev_k.get(k, np.nan), my_l.rev_src.get(k, ''), re_l.rev_src.get(k, '')) for k in UL]
lagree = [(np.isfinite(a) and np.isfinite(b) and abs(a / b - 1) <= 1e-9) for _, a, b, _, _ in lag]
out = dict(gates=G, passes=PASS, n_cons=len(cons), n_rows_mine=len(B), n_yoy_mine=int(B.yoy_pct.notna().sum()),
           rev_src_mine=B.rev_src.value_counts().to_dict(), yoy_src_mine=B.yoy_src[B.yoy_pct.notna()].value_counts().to_dict(),
           yoy_cmp=dict(n_union=len(CMP), n_agree=int(CMP.agree.sum()), rate=float(CMP.agree.mean()), n_mine=len(my_y), n_theirs=len(re_y),
                        disagree=dis.to_dict('records')),
           level_cmp=dict(n_union=len(lag), n_agree=int(sum(lagree)), rate=float(np.mean(lagree)),
                          disagree=[dict(code=k[0], ym=k[1], mine=a, theirs=b, src_mine=sa, src_theirs=sb) for (k, a, b, sa, sb), g in zip(lag, lagree) if not g]))
(HERE / ('r3_backfill.json' if VAR == 'base' else f'r3_backfill_{VAR}.json')).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k not in ('yoy_cmp', 'level_cmp')}, ensure_ascii=False, indent=0, default=str)[:6000])
print('YOY', {k: v for k, v in out['yoy_cmp'].items() if k != 'disagree'}, 'LEVEL', {k: v for k, v in out['level_cmp'].items() if k != 'disagree'})
print(pd.DataFrame(out['yoy_cmp']['disagree']).to_string()[:4000])
print(pd.DataFrame(out['level_cmp']['disagree']).to_string()[:6000])

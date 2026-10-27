#!/usr/bin/env python3
"""迭代建模（只用挑選期，lock_selection 強制）：事前列出的 8 個版本（訊號規格 × 結合方式）全部回測、全部記錄。
挑選期：訊號 2006-04～2019-04（主策略營收腿 2006-04 起才有），評價到 2019-05 換股日收盤。
選擇規則（事前固定）：配對差（對原主策略）t ≥ 1.0 的版本中，取對同池等權 t 最高者；沒有就判「不可用」，不打開驗證期。
輸出 iterations.json。"""
import sys, json, time, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as L
L.lock_selection()
import combo as CB
K, V = L.K, L.V
IDX, COLS, N = L.IDX, L.COLS, L.N
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)

ITF = L.HERE / 'iterations.json'
VERSIONS = [
    ('v1', (126, 0, 'cum'), 'third', 'P1 最強的 IMOM（6 月、不跳過）當第三腿'),
    ('v2', (126, 0, 'share'), 'third', 'P1 最強的盤中占比當第三腿'),
    ('v3', (252, 0, 'diff'), 'third', '12 月盤中減隔夜當第三腿'),
    ('v4', (126, 0, 'share'), 'mult', '52 週腿乘以盤中占比百分位'),
    ('v5', (126, 0, 'cum'), 'mult', '52 週腿乘以 IMOM 百分位'),
    ('v6', (126, 0, 'share'), 'drop', '前 20 名剔除最隔夜型 5 檔（6 月占比）'),
    ('v7', (252, 21, 'share'), 'drop', '前 20 名剔除最隔夜型 5 檔（P2 挑選期最強：12-1 占比）'),
    ('v8', (252, 21, 'cum'), 'third', '研究計畫原始假說規格 12-1 IMOM 當第三腿'),
]
plan_doc = {'versions': [{'id': v, 'spec': {'lookback': s[0], 'skip': s[1], 'form': s[2]}, 'combo': c, 'why': w} for v, s, c, w in VERSIONS],
            'selection_rule': '挑選期配對差（新版 − 原主策略，訊號窗月報酬，一般成本）t ≥ 1.0 的版本中，取對同池等權 t 最高者；都沒有 → 不可用、不開驗證期',
            'chosen_from_explore': '版本由 explore.json 的 P1／P2 挑選期結果選出（12 個訊號規格中挑 6 個），探索本身也是多重比較，最後報告揭露 12 規格 × 8 版本'}
doc = json.loads(ITF.read_text(encoding='utf-8')) if ITF.exists() else {}
if 'plan' not in doc:
    doc['plan'] = plan_doc; doc['plan_time'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    ITF.write_text(L.jdump(doc), encoding='utf-8')
assert doc['plan']['versions'] == json.loads(L.jdump(plan_doc))['versions'], '版本清單和事前紀錄不同'

sp = L.Split()
YL, Y5, _ = V.load_revenue()
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
A, B = '2006-04-01', str(L.CUT_DAY.date())
SEGS = {'SEL': (A, B), 'SEL_2006_2012': (A, '2012-12-31'), 'SEL_2013_2019': ('2013-01-01', B)}

def trim(S):
    S.sig_days = [d for d in S.sig_days if d <= L.SEL_LAST_SIG]; S.end_day = L.CUT_DAY
    return S

base = trim(V.V2('gapfix', YL, Y5, exfin=True))
t0, t1, sd = base.bounds(); L.assert_locked_ok(t1)
plb = base.plan(10, 'official')
ew, _ = K.simulate([(p[0], p[2]) for p in plb], 0, 0, t0, t1)
navb = {c: K.simulate([(p[0], p[1]) for p in plb], bb, ss, t0, t1)[0] for c, (bb, ss) in COST.items()}
assert np.isfinite(navb['normal'].iloc[-1])

def segs(nav):
    o = {}
    for k, (a, b) in SEGS.items():
        s = V.seg_stats(nav, ew, a, b, sd, yearly=(k == 'SEL'))
        o[k] = {kk: s[kk] for kk in ('CAGR', 'maxDD', 't_0050', 't_EW', 'E50_CAGR', 'EW_CAGR', 'n_m')}
        if k == 'SEL': o[k]['yearly'] = s['yearly']
    return o
res = {'main': {c: segs(navb[c]) for c in COST}, 'versions': {}}
log('main', {k: (round(v['CAGR'], 2), round(v['t_0050'], 2), round(v['t_EW'], 2)) for k, v in res['main']['normal'].items()})
for vid, spec, cmb, why in VERSIONS:
    S = trim(CB.Combo(spec, cmb, sp, YL, Y5))
    pl = S.plan(10, 'official')
    assert [p[3] for p in pl] == [p[3] for p in plb]
    o = {'spec': spec, 'combo': cmb, 'blocked': S.nblk}
    ov = [len(set(p[1]) & set(q[1])) / 10 for p, q in zip(pl, plb)]
    o['overlap_with_main'] = float(np.mean(ov))
    for c, (bb, ss) in COST.items():
        nav, lg = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
        o[c] = segs(nav); o[c]['turnover'] = lg['turnover']
        o[c]['paired_vs_main'] = {k: L.paired(nav, navb[c], sd, a, b)[0] for k, (a, b) in SEGS.items()}
    res['versions'][vid] = o
    pv = o['normal']['paired_vs_main']['SEL']
    log(vid, spec, cmb, 'CAGR', round(o['normal']['SEL']['CAGR'], 2), 't0050', round(o['normal']['SEL']['t_0050'], 2), 'tEW', round(o['normal']['SEL']['t_EW'], 2),
        'paired', round(pv['mean_pct'], 3), round(pv['t'], 2), 'overlap', round(o['overlap_with_main'], 2))

ok = [v for v, o in res['versions'].items() if o['normal']['paired_vs_main']['SEL']['t'] >= 1.0]
chosen = max(ok, key=lambda v: res['versions'][v]['normal']['SEL']['t_EW']) if ok else None
res['passing_paired_t_ge_1'] = ok; res['chosen'] = chosen
res['verdict_if_none'] = '不可用（沒有版本配對差 t ≥ 1.0，不打開驗證期）' if chosen is None else None
doc['results'] = res; doc['results_time'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
ITF.write_text(L.jdump(doc), encoding='utf-8')
log('chosen', chosen, 'passing', ok)

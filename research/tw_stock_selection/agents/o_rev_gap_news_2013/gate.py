#!/usr/bin/env python3
"""步驟 3／5：正式閘門（PREREG 之後一次跑完）。斷言 PREREG 與 data_freeze_r 存在且未被改，才計算。
G1 不過 → 不跑任何策略淨值；只做覆蓋報告與描述（含「代理放回後會進前 10 名的缺口股月數」這種名單計數，不是報酬）。"""
import sys, json, hashlib, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
F = HERE / 'results.json'
res = json.loads(F.read_text(encoding='utf-8'))
assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在'
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert sha(HERE / 'data_freeze_r.json') == res['PREREG']['data_freeze_r_sha256'], 'data_freeze_r 與 PREREG 不符'
assert sha(HERE / 'gate.py') == res['PREREG']['gate_script_sha256'] and sha(HERE / 'firstvalue.py') == res['PREREG']['firstvalue_script_sha256']
fz = json.loads((HERE / 'data_freeze_r.json').read_text(encoding='utf-8'))
for k, v in fz['files'].items():
    if k in ('topic.iterations.json', 'topic.explore.json'): continue
    assert sha(v['path']) == v['sha256'], f'{k} 與凍結不符'
import rglib as G
import proxy as PX
K, V, L = G.K, G.V, G.L
sel = res['PREREG']['selected']; IT, VER = sel['parser'], sel['proxy_version']
b = G.base(); S = G.mk()
M, LV, L5, Q = PX.load_inputs(K, HERE / 'data' / f'mentions_{IT}.parquet')
P = PX.Proxy(M, LV, L5, Q, S.YL, S.Y5, V.SWITCH)
X = pd.read_csv(HERE / 'gap_cells_2013_2021.csv', dtype={'code': str})
pv = {}
for ver in PX.VERSIONS:
    pv[ver] = np.array([P.y3(c, pd.Timestamp(s), ver)[0] for s, c in zip(X.sig, X.code)])
cov = np.isfinite(pv[VER])
def rate(m): return {'n': int(m.sum()), 'cov': float(cov[m].mean()) if m.sum() else None}
g1 = {'version': f'{IT}/{VER}', 'cov_all_732': float(cov.mean()), 'n_cov': int(cov.sum()),
      'cov_2013_2020': rate((X.sig.str[:4] <= '2020').to_numpy()), 'cov_B10_picks_133': rate(X.in_B10.to_numpy()),
      'cov_by_type': {t: rate((X.type == t).to_numpy()) for t in ('merger', 'deterior', 'nondelist')},
      'cov_by_year': {y: rate((X.sig.str[:4] == y).to_numpy()) for y in sorted(X.sig.str[:4].unique())},
      'cov_all_versions': {v: float(np.isfinite(pv[v]).mean()) for v in PX.VERSIONS},
      'cov_B10_picks_all_versions': {v: float(np.isfinite(pv[v])[X.in_B10.to_numpy()].mean()) for v in PX.VERSIONS},
      'threshold': 0.60}
g1['pass'] = g1['cov_all_732'] >= 0.60
pd.DataFrame({'sig': X.sig, 'code': X.code, 'type': X.type, 'in_B10': X.in_B10, **{f'proxy_{v}': pv[v] for v in PX.VERSIONS}}).to_csv(HERE / 'data' / 'proxy_table_gap_cells.csv', index=False)
# G2
r = subprocess.run([sys.executable, '-I', '-B', str(HERE / 'firstvalue.py'), IT], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-2000:]
g2 = json.loads((HERE / f'firstvalue_{IT}.json').read_text(encoding='utf-8'))
# 描述：代理放回後（非 AB 名單），缺口股月進前 10 名的個數（名單計數，不是報酬）
cellset = {(s, c) for s, c in zip(X.sig, X.code)}
class V4P(L.V4):
    def __init__(self, *a, pmap=None, worst=False, **k):
        super().__init__(*a, **k); self.pmap = pmap; self.worst = worst
    def y3(self, d, back=0):
        y = super().y3(d, back)
        if back == 0 and G.W0 <= d <= G.W1:
            y = y.copy(); s = str(d.date())
            for (ss, c), v in self.pmap.items():
                if ss != s: continue
                j = K.CI[c]
                if np.isfinite(v): y[j] = v
                elif self.worst: y[j] = 1e9          # 未覆蓋格以最壞情況（營收百分位 1.0）
        return y
def picks_count(ver, worst):
    pm = {(s, c): v for s, c, v in zip(X.sig, X.code, pv[ver])}
    Sx = V4P(b['YFX'], b['Y5F'], b['DMT'], pmap=pm, worst=worst)
    n = {'covered': 0, 'uncovered_worst': 0, 'by_type_covered': {}, 'by_type_uncovered': {}}
    for e, pk, _, d in Sx.plan(10, 'official'):
        if not (G.W0 <= d <= G.W1): continue
        s = str(d.date())
        for j in pk:
            c = K.COLS[j]
            if (s, c) in cellset:
                v = pm[(s, c)]; t = X.type[(X.sig == s) & (X.code == c)].iloc[0]
                key = 'covered' if np.isfinite(v) else 'uncovered_worst'
                n[key] += 1; bt = n['by_type_' + ('covered' if np.isfinite(v) else 'uncovered')]; bt[t] = bt.get(t, 0) + 1
    return n
desc = {f'{v}_{"B10R" if w else "ABR"}': picks_count(v, w) for v in (VER, 'NQ') for w in (False, True)}
out = {'G1': g1, 'G2': {k: g2[k] for k in ('coverage', 'G2_pass', 'coverage_wilson95_strict_all3', 'n_top10_stock_month_seed', 'n_distinct_stock_months',
                                            'n_cells_corroborated_strict', 'n_cells_corroborated_loose', 'first_vs_latest')},
       'descriptive_gap_picks_non_AB（名單計數，非報酬；B10 原本 133）': desc}
if not g1['pass']:
    out['decision'] = 'G1 不過：維持 B10 為 SOP 保守口徑，主題結案；不跑 AB_V4R／B10R／AB_V4R_N1 淨值。首次值：覆蓋 {:.2%}（嚴格、三個月）→ 無法估計。'.format(g2['coverage']['strict_all3'])
else:
    out['decision'] = 'G1 通過：需執行 run_r.py（未實作，因為本次未通過時不得跑）'
res['GATE'] = out
F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))

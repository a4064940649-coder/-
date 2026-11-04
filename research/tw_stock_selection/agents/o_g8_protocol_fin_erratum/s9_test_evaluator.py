#!/usr/bin/env python3
"""判定器一致性測試：在 scratchpad 造合成前瞻資料夾（13 個 J 窗超額相同＝x_t、meta ts_ok／覆蓋率 1.0），
逐條比較 g8_evaluate.py 的終態與 g8lib.run_rule＋合併程序（OC 模擬用的同一套語意）。另測作廢規則與成本校準流程。
用法：python3 -I -B s9_test_evaluator.py <protocol.json>"""
import sys, json, shutil, subprocess, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import g8lib as G

PROTO = Path(sys.argv[1])
P = json.loads(PROTO.read_text(encoding='utf-8'))
TD = G.SCR / 'evaltest'
H = 36; NP = 120
rng = np.random.default_rng(20261112)
alphas = np.repeat([-1.0, 0.0, 1.0, 3.0], NP // 4)
X = np.vstack([G.gen('D1', a, 1, H, rng) for a in alphas])
X[::7] *= 1.8            # 部分路徑放大波動以觸發 refit
par = dict(sigma=P['params']['sigma_design'], c=P['params']['c'], c_d=P['params']['c_d'], b=P['params']['b'], up_looks=P['params']['up_looks'])
u, d = G.run_rule('R-FIX', X, par)
g8 = X[:, :12].sum(axis=1) > 0
anchors = [str(p) for p in pd.period_range('2026-11', periods=H, freq='M')]


def make_folder(x, void_months=()):
    if TD.exists(): shutil.rmtree(TD)
    (TD / 'paper_lists').mkdir(parents=True)
    rows = []
    for k, an in enumerate(anchors):
        st = pd.Timestamp(an + '-10')
        for j in range(13):
            rows.append(dict(line=f'J{j}', anchor=an, start=str((st + pd.Timedelta(days=j)).date()), end=str((st + pd.Timedelta(days=30 + j)).date()), status='complete', n_j='',
                             ret=x[k] / 100, ret_0050=0.0, excess_pp=x[k], ret_stress=x[k] / 100, excess_stress_pp=x[k] - 0.5, ret_cap10=x[k] / 100, excess_cap10_pp=x[k], cost='', n_hold=10, source='test', note=''))
            ok = an not in void_months
            meta = dict(ts_ok=ok, frozen_ts=f'{st.date()}T22:30:00+08:00', exec_open_ts=f'{(st + pd.Timedelta(days=1)).date()}T09:00:00+08:00', rev_gate=dict(coverage=1.0))
            (TD / 'paper_lists' / f'j{j}_{an.replace("-", "")}.meta.json').write_text(json.dumps(meta))
    pd.DataFrame(rows).to_csv(TD / 'forward_returns.csv', index=False)
    pd.DataFrame(columns=['kind', 'line', 'date', 'code', 'side', 'is_odd', 'shares', 'fill_px', 'open_px', 'share_of_open_vol', 'share_of_open_vol_actual', 'sigma60_prev', 'ticks_from_open']).to_csv(TD / 'trades.csv', index=False)


def evaluate(extra=()):
    r = subprocess.run([sys.executable, '-I', '-B', str(G.HERE / 'g8_evaluate.py'), 'status', '--forward', str(TD), '--protocol', str(PROTO), *extra],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def expected(i):
    if not g8[i]: return 'FORWARD_FAILED'
    if u[i] > 0: return 'S1_EXCEPTION_EVIDENCE'
    if d[i] > 0: return 'FORWARD_FAILED'
    return 'FORWARD_VALIDATING_NO_REJECTION_24'


mism = []; counts = {}
for i in range(NP):
    make_folder(X[i])
    o = evaluate()
    e = expected(i)
    counts[e] = counts.get(e, 0) + 1
    if o['state'] != e: mism.append(dict(i=i, alpha=float(alphas[i]), got=o['state'], exp=e, u=int(u[i]), d=int(d[i]), events=o['events']))
# 作廢規則：3 個錨定月作廢（在前 12 個有效月之前）→ 字面 G8 無法判定；n 只算有效月
make_folder(X[0], void_months=('2026-12', '2027-02', '2027-03'))
ov = evaluate()
void_ok = bool(ov['g8_literal'] is not None and 'label' in ov['g8_literal'] and sum(a['state'] == 'void' for a in ov['anchors']) == 3 and ov['n_valid'] == H - 3)
# 作廢 2 個月 → 正常判定
make_folder(X[0], void_months=('2026-12', '2027-02'))
ov2 = evaluate()
void2_ok = bool(ov2['g8_literal'] is not None and 'label' not in ov2['g8_literal'])
# 成本校準：資料不足 → performed＝False、Δ＝0
r = subprocess.run([sys.executable, '-I', '-B', str(G.HERE / 'g8_evaluate.py'), 'calibrate', '--forward', str(TD), '--protocol', str(PROTO), '--asof', '2027-03-31', '--out', str(G.SCR / 'calib_test.json')],
                   capture_output=True, text=True)
cal = json.loads(r.stdout) if r.returncode == 0 else dict(err=r.stderr[-500:])
out = dict(utc=G.now_utc(), n_paths=NP, n_mismatch=len(mism), mismatches=mism[:10], expected_counts=counts, void_gt2_ok=void_ok, void_eq2_ok=void2_ok,
           calibrate_insufficient_ok=bool(cal.get('performed') is False and cal.get('delta_per_side') == 0.0), protocol_sha256=G.sha(PROTO))
shutil.rmtree(TD)
print(json.dumps(out, ensure_ascii=False, default=G._js))
if '--save' in sys.argv: G.save_key('EVALUATOR_TEST', out)

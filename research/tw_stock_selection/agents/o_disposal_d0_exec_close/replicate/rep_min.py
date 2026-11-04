#!/usr/bin/env python3
"""自我交叉核對（不是 G6 獨立複製：同一代理）：不 import dlib，只依 frozen_spec.txt 第一～三節重寫
C1×CL、每槽 10 萬（2026 幣值）一般成本的逐筆可執行 net_x，與 s4 結果比對。資料層用 o9 core（x_cache）。
用法：python3 -I -B replicate/rep_min.py"""
import sys, math, json
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent; TOP = HERE.parent
sys.path.insert(0, str(TOP.parent / 'o_disposal_official_fullhist'))
import numpy as np, pandas as pd
import core as K

L = pd.read_csv(TOP / 'raw' / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
L['ordr'] = np.arange(len(L)); L['name'] = L['name'].fillna('')
L['is_common'] = (L.code.str.len() == 4) & (~L.code.str.startswith('0')) & (~L.name.str.contains('DR'))
U = K.universe(L)
SH = json.loads((TOP / 'calib5m.json').read_text())['frozen_shares']
SRC = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/src.parquet').reindex(index=K.IDX, columns=K.COLS).to_numpy()


def share(t, r, kind):
    inside = r.s_i <= t <= r.e_i
    if inside and r.regime == 'periodic_match' and np.isfinite(r.match_min) and r.match_min > 0:
        m = r.match_min
        g = '5' if 3 < m <= 7 else ('20' if 12 < m <= 22 else 'x')
        if kind == 'C': return SH['last_5'] if g == '5' else (SH['last_20'] if g == '20' else SH['last_2'])
    return SH['close_norm']


rows = []; busy = {}
for r in U.sort_values(['s_i', 'announce', 'ordr'], kind='stable').itertuples():
    if not r.take_R1 or not r.in_D or r.s_i >= K.T or r.e_i <= r.s_i: continue
    j = K.CI[r.code]; s = r.s_i
    if busy.get(r.code, -1) >= s: continue
    if K.bear_before(K.IDX[s]): continue
    if not (np.isfinite(K.V60p[s, j]) and K.V60p[s, j] >= 1e7 and np.isfinite(K.PC[s, j]) and K.PC[s, j] >= 10): continue
    c = K.C[s, j]; Lu = K.LIMUP[s, j]
    if not (K.TR[s, j] and np.isfinite(c) and c < Lu - 1e-9): continue
    lim = min(K.T - 1, K.LASTV[j]); k = min(r.e_i, lim)
    while k < lim and not (K.TR[k, j] and K.C[k, j] > K.LIMDN[k, j] + 1e-9): k += 1
    if k <= s: continue
    busy[r.code] = k
    A = 1e5 * K.cpi_factor(s); N = int(A // Lu); nl, no = N // 1000, N % 1000
    v = K.V[s, j]; sh = share(s, r, 'C'); v = v / (1 - sh) if SRC[s, j] == 3 else v
    Qa = v * sh; nl = min(nl, int(0.2 * Qa // (c * 1000)))
    fr = K.fee_rate(s, False); sg = K.SIGp[s, j] if np.isfinite(K.SIGp[s, j]) else 0.03
    tk = max(float(K.tick(c)) / c, 0.001)
    out = 0.0
    if nl:
        q = nl * 1000 * c; out += q * (1 + tk + 0.5 * sg * math.sqrt(q / Qa)) + max(20, fr * q)
    if no:
        q = no * c; out += q * 1.004 + max(20, fr * q)
    g = math.exp(K.LA[k, j] - K.LA[s, j])
    # 賣（簡化：整張若超過上限也假設當日全賣；只在核對差異時說明）
    pin = 0.0; ck = K.C[k, j]; fr2 = K.fee_rate(k, False)
    v2 = K.V[k, j]; sh2 = share(k, r, 'C'); v2 = v2 / (1 - sh2) if SRC[k, j] == 3 else v2; Qa2 = v2 * sh2
    sg2 = K.SIGp[k, j] if np.isfinite(K.SIGp[k, j]) else 0.03; tk2 = max(float(K.tick(ck)) / ck, 0.001)
    if nl:
        q = nl * 1000 * c * g; pin += q * (1 - tk2 - 0.5 * sg2 * math.sqrt(q / Qa2) - 0.003) - max(20, fr2 * q)
    if no:
        q = no * c * g; pin += q * (1 - 0.004 - 0.003) - max(20, fr2 * q)
    ew = math.expm1(K.LEW[k] - K.LEW[s])
    rows.append(dict(code=r.code, date=K.IDX[s], net_x=pin / out - 1 - ew))
df = pd.DataFrame(rows)
f = df[(df.date >= '2006-04-01') & (df.date <= '2026-10-22')]
v = f[(f.date >= '2024-01-02')]
print(json.dumps({'FULL_n': int(len(f)), 'FULL_mean_pct': round(float(f.net_x.mean() * 100), 3), 'VAL_n': int(len(v)), 'VAL_mean_pct': round(float(v.net_x.mean() * 100), 3)}))

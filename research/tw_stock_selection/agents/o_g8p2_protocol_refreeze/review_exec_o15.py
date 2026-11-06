#!/usr/bin/env python3
"""反方審查（成本與可成交角度）o15：G8P-2 V1 獨立純量重寫（不 import 研究員程式），只讀 G8_PROTOCOL_v2.json。
問題：判定線相對真實淨超額的正向漂移 d（pp／月）到多少時，V1 誤升級 > 5%？
  主口徑最差 d＝0.78−0.527＝0.253；若前瞻期較差線平移像 2019 後（0.611），同樣「真實＝一般＋0.78」只剩 d＝0.169。
  d 可能來源：零股（100 萬時約 84% 成交值）成本與成交機率、開盤衝擊、M2 遞補價，全部沒有真錢校準。
另測：漂移隨月份換手而變（d_t＝d·T_t/E[T]，T_t ~ lognormal sd 0.5），平均相同時是否更糟。
執行：python3 -I -B review_exec_o15.py
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'G8_PROTOCOL_v2.json').read_text(encoding='utf-8'))['params']
SD, C, CD, B = P['sigma_design'], P['c'], P['c_d'], P['b']
CS = C * SD


def run(dgp, sigma, d, n_paths, seed, tv=False):
    rng = np.random.default_rng(seed)
    if dgp == 'D1':
        e = rng.standard_normal((n_paths, 24))
    else:
        e = rng.standard_t(4, size=(n_paths, 24)) / math.sqrt(2.0)
    drift = np.full((n_paths, 24), d)
    if tv:
        T = rng.lognormal(-0.125, 0.5, size=(n_paths, 24))   # E[T]=1
        drift = d * T
    X = drift + sigma * e
    S = np.cumsum(X, 1)
    def z(n):
        s = X[:, :n].std(axis=1, ddof=1)
        return S[:, n - 1] / (np.maximum(CS, s) * math.sqrt(n))
    z12, z18, z24 = z(12), z(18), z(24)
    alive = S[:, 11] > 0                         # 字面 G8
    alive &= ~(z12 <= -CD)
    up12 = alive & (z12 >= B * math.sqrt(2.0))
    cont = alive & ~up12
    cont &= ~(z18 <= -CD)
    up24 = cont & ~(z24 <= -CD) & (z24 >= B)
    return float((up12 | up24).mean())


def main():
    out = {}
    N = 200_000
    sig = [6.141373712658076, 7.0, 8.0, 9.0, 10.4]
    for dgp in ('D1', 'D2'):
        for d in (0.0, 0.169, 0.253, 0.35, 0.45, 0.6):
            row = {}
            for i, s in enumerate(sig):
                row[f'{s:.2f}'] = run(dgp, s, d, N, 15_000 + 100 * i + int(d * 1000))
            out[f'{dgp}|d={d}'] = row
            print(dgp, d, {k: round(v, 4) for k, v in row.items()}, 'max', round(max(row.values()), 4), flush=True)
    for d in (0.253,):
        row = {f'{s:.2f}': run('D1', s, d, N, 99_000 + i, tv=True) for i, s in enumerate(sig)}
        out[f'D1|d={d}|turnover_varying'] = row
        print('D1 tv', d, {k: round(v, 4) for k, v in row.items()})
    # 等效：漂移 d 對應「每單位單邊換手」的額外每邊成本（假設單邊月換手 0.42，見審查文字）
    print('d per month → extra cost per side (one-way monthly turnover 0.42):',
          {d: round(d / (2 * 0.42), 3) for d in (0.169, 0.253, 0.35, 0.45)}, '% per side')
    (HERE / 'review_exec_o15_out.json').write_text(json.dumps(out, indent=1), encoding='utf-8') if '--save' in sys.argv else None


if __name__ == '__main__':
    main()

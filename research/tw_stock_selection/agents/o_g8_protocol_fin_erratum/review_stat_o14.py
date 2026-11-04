#!/usr/bin/env python3
"""反方審查（資料與統計）o14：G8 序貫協定在「前瞻 σ 不等於設計 σ」時的誤升級；獨立重算 b、c_d；殘差離群月敏感度。
只讀 G8_PROTOCOL.json，不 import g8lib、不寫任何研究員檔案。輸出只印到 stdout。
用法：python3 -I -B review_stat_o14.py"""
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'G8_PROTOCOL.json').read_text(encoding='utf-8'))
SD, C, CD, B = (P['params'][k] for k in ('sigma_design', 'c', 'c_d', 'b'))
RES = np.array(P['data']['resid_post2019'], float); RES = RES - RES.mean()
H = 24


def rule(X):
    """協定 layer2＋refit＋字面 G8 閘（只算到 24 個月）。回傳 (seq_up, combined_up, combined_fail)。"""
    n_ = np.arange(1, H + 1)
    S = np.cumsum(X, 1); S2 = np.cumsum(X ** 2, 1)
    Pn = X.shape[0]
    sig = np.full(Pn, C * SD); trig = np.zeros(Pn, bool)
    up = np.zeros(Pn, bool); dn = np.zeros(Pn, bool); done = np.zeros(Pn, bool)
    for n in (12, 18, 24):
        m = S[:, n - 1] / n; s_hat = np.sqrt(np.maximum((S2[:, n - 1] - n * m * m) / (n - 1), 0))
        new = (s_hat > 1.5 * SD) & ~trig
        sig = np.where(new, np.maximum(C * SD, s_hat), sig); trig |= new
        z = S[:, n - 1] / (sig * math.sqrt(n))
        d = (~done) & (z <= -CD); dn |= d; done |= d
        if n in (12, 24):
            u = (~done) & (z >= B * math.sqrt(24 / n)); up |= u; done |= u
    g8 = S[:, 11] > 0
    return up.mean(), (up & g8).mean(), ((~g8) | dn).mean()


def main():
    rng = np.random.default_rng(14014)
    # (1) 獨立重算 b（單位 BM、looks 12/24、5%）與 c_d
    Pb = 2_000_000
    z12 = rng.standard_normal(Pb); z24 = (z12 * math.sqrt(12) + rng.standard_normal(Pb) * math.sqrt(12)) / math.sqrt(24)
    M = np.maximum(z12 / math.sqrt(2), z24)
    print('b 重算', round(float(np.quantile(M, 0.95)), 4), '協定', round(B, 4))
    X = -1 + SD * rng.standard_normal((400_000, 24)); S = np.cumsum(X, 1)
    zmin = np.min(np.stack([S[:, n - 1] / (C * SD * math.sqrt(n)) for n in (12, 18, 24)], 1), 1)
    print('c_d 重算', round(float(-np.quantile(zmin, 0.55)), 4), '協定', round(CD, 4))
    print('殘差 sd', round(RES.std(ddof=1), 3), 'c·σ_d', round(C * SD, 3), '1.5σ_d（refit 門檻）', round(1.5 * SD, 3))
    # (2) 真實 σ 掃描：D1（iid 常態）與 D3s（區塊 bootstrap 殘差按比例放大）
    Pn = 200_000
    print('\nσ_true | D1 誤升級(序貫) 誤升級(含字面G8閘) α=+1升級 | D3s 誤升級(序貫) 含閘')
    for s in (3.93, 5.0, 6.14, 6.51, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0):
        X0 = s * rng.standard_normal((Pn, H))
        fu, fuc, _ = rule(X0)
        up1, up1c, _ = rule(X0 + 1.0)
        # D3s：平均區塊 6、環狀
        idx = np.empty((Pn, H), np.int64); idx[:, 0] = rng.integers(0, len(RES), Pn)
        for t in range(1, H):
            j = rng.random(Pn) < 1 / 6
            idx[:, t] = np.where(j, rng.integers(0, len(RES), Pn), (idx[:, t - 1] + 1) % len(RES))
        X3 = RES[idx] * (s / RES.std(ddof=1))
        f3, f3c, _ = rule(X3)
        print(f'{s:5.2f} | {fu:.4f} {fuc:.4f} {up1c:.4f} | {f3:.4f} {f3c:.4f}')
    # (3) 離群月（+26.1）敏感度：D3 去掉最大一個殘差（重新去均值、不縮放）
    r2 = np.delete(RES, np.argmax(RES)); r2 = r2 - r2.mean()
    print('\n去掉最大殘差後 sd', round(r2.std(ddof=1), 3), 'skew', round(float(((r2 - r2.mean()) ** 3).mean() / r2.std() ** 3), 3))
    # (4) σ 在前瞻中途上升：前 12 個月 σ_d、後 12 個月 1.4σ（o8 波動狀態上限）
    X = np.concatenate([SD * rng.standard_normal((Pn, 12)), 1.4 * SD * rng.standard_normal((Pn, 12))], 1)
    print('前 12 月 σ_d、後 12 月 1.4σ_d：誤升級', [round(v, 4) for v in rule(X)])
    X = 1.4 * SD * rng.standard_normal((Pn, 24))
    print('全程 1.4σ_d（o8 高波動狀態持續）：誤升級', [round(v, 4) for v in rule(X)])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""步驟「迭代建模」（只用挑選期 2006-04～2019-04 的資料與模擬，不看任何真實全期 p 值）：
A. SPA 實作版本（長期變異數 ω 的估計法）：V1 Politis-Romano 核（Hansen 2005 建議）、V2 bootstrap 變異數、
   V3 Newey-West Bartlett、V4 iid 樣本變異數。用挑選期 F_long 族群（157 窗 × 2,319 條）去均值後做平穩
   bootstrap 重抽，產生「已知真相」的模擬資料：
     NULL_LF：全部成員真實均值 0（最不利虛無）；NULL_RE：真實均值＝min(挑選期 d̄,0)，主策略與最佳成員設為 0；
     POW05／POW10：NULL_RE 但主策略真實均值 +0.5／+1.0 pp/窗。
   每組 150 次，內層 bootstrap 499 次，區塊長度 6。記錄 5% 拒絕率：SPA_c（主策略檢定量）、SPA_c（族群最大）、RC。
   選擇規則（事前寫在這裡）：NULL_LF 下 SPA_c（主策略檢定量）拒絕率落在 [2.5%, 7.5%] 的版本中，選 POW10 檢定力最高者；
   同分取最接近 5%。
B. 斷點 sup-F 的變異數版本：B1 HAC（NW 落後規則）、B2 iid、B3 HAC 固定落後 6。虛無：挑選期主策略對 0050 超額
   去均值後區塊重抽成 245 窗；臨界值來自 iid 常態有限樣本模擬；檢定力：在第 157 窗（2019-05）後均值下降 0.5 pp。
   選擇規則：虛無拒絕率最接近 5%（且 ≤ 7.5%）者；同分取檢定力高者。
C. 族群蒐集與清理的版本紀錄（不依 p 值）。
輸出 iterations.json。"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C, spa_lib as S, breaks_lib as BL
MAIN_ID = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'
SEL = ('2006-04-01', '2019-04-30')
T0 = time.time()

def sim_spa(Xd, mu_cfg, focus, nsim=150, Bin=499, L=6, seed=0):
    n, N = Xd.shape
    rng = np.random.default_rng(seed)
    res = {v: {'c_main': 0, 'c_max': 0, 'rc': 0} for v in ('pr', 'boot', 'nw', 'iid')}
    for s in range(nsim):
        # 平穩 bootstrap 路徑（保留區塊結構）產生一組模擬資料
        p = 1 / L; path = np.empty(n, int); path[0] = rng.integers(0, n)
        for t in range(1, n): path[t] = rng.integers(0, n) if rng.random() < p else (path[t - 1] + 1) % n
        D = Xd[path] + mu_cfg[None, :]
        Win = S.sb_counts(n, L, Bin, int(rng.integers(1 << 30)))
        Mb = S.boot_means(Win, D)
        dbar = D.mean(axis=0); sn = np.sqrt(n)
        for v in res:
            om2 = n * Mb.astype(float).var(axis=0) if v == 'boot' else S.lrv(D, v, L)
            om = np.sqrt(om2); tst = sn * dbar / om
            thr = -np.sqrt(om2 / n * 2 * np.log(np.log(n)))
            g = np.where(dbar >= thr, dbar, 0.0)
            Z = (Mb - g[None, :]) * (sn / om)[None, :]
            Tb = np.maximum(Z.max(axis=1), 0)
            res[v]['c_main'] += float((Tb >= max(tst[focus], 0)).mean() <= 0.05 and tst[focus] > 0)
            res[v]['c_max'] += float((Tb >= max(tst.max(), 0)).mean() <= 0.05)
            Vb = sn * (Mb - dbar[None, :]).max(axis=1)
            res[v]['rc'] += float((Vb >= sn * dbar.max()).mean() <= 0.05)
    return {v: {k: x / nsim for k, x in d.items()} for v, d in res.items()}

def part_A():
    D = pd.read_parquet(C.DATA / 'family_excess.parquet'); MM = pd.read_csv(C.DATA / 'family_members.csv')
    fl = MM[MM.F_long].member_id.tolist(); m = C.wmask(D.index, *SEL)
    X = D.loc[m, fl].fillna(0.0).to_numpy(); focus = fl.index(MAIN_ID)
    dbar = X.mean(axis=0); Xd = X - dbar
    re = np.minimum(dbar, 0.0); re[focus] = 0.0; re[np.argmax(dbar)] = 0.0
    cfgs = {'NULL_LF': np.zeros_like(dbar), 'NULL_RE': re}
    p05 = re.copy(); p05[focus] = 0.005; p10 = re.copy(); p10[focus] = 0.010
    cfgs['POW05'] = p05; cfgs['POW10'] = p10
    out = {}
    for i, (k, mu) in enumerate(cfgs.items()):
        out[k] = sim_spa(Xd, mu, focus, seed=100 + i)
        print('A', k, round(time.time() - T0), json.dumps(out[k]), flush=True)
    # 選擇
    cand = [v for v in ('pr', 'boot', 'nw', 'iid') if 0.025 <= out['NULL_LF'][v]['c_main'] <= 0.075]
    if cand:
        best = sorted(cand, key=lambda v: (-out['POW10'][v]['c_main'], abs(out['NULL_LF'][v]['c_main'] - 0.05)))[0]
    else:
        best = sorted(('pr', 'boot', 'nw', 'iid'), key=lambda v: abs(out['NULL_LF'][v]['c_main'] - 0.05))[0]
    names = {'pr': 'V1 Politis-Romano 核', 'boot': 'V2 bootstrap 變異數', 'nw': 'V3 Newey-West', 'iid': 'V4 iid'}
    return {'sims': out, 'candidates_in_size_band': cand, 'selected': best, 'selected_name': names[best], 'n_members': len(fl), 'n_windows': int(m.sum())}

def part_B():
    D = pd.read_parquet(C.DATA / 'family_excess.parquet')
    x = D[MAIN_ID]; m = C.wmask(x.index, *SEL)
    xs = x[m].to_numpy(); xd = xs - xs.mean()
    n = 245; nsim_crit = 1000; nser = 300
    rng0 = np.random.default_rng(7)
    variants = {'B1_hac_rule': dict(hac=True, lag=None), 'B2_iid': dict(hac=False, lag=None), 'B3_hac_lag6': dict(hac=True, lag=6)}
    def supf(y, hac, lag):
        if lag is None or not hac: return BL.supF(y, 0.15, hac)['stat']
        old = BL.nw_lag; BL.nw_lag = lambda n_: lag
        try: return BL.supF(y, 0.15, True)['stat']
        finally: BL.nw_lag = old
    out = {}
    for name, kw in variants.items():
        crit = np.quantile([supf(rng0.standard_normal(n), kw['hac'], kw['lag']) for _ in range(nsim_crit)], 0.95)
        rng = np.random.default_rng(11)
        rej0 = rej1 = 0
        for s in range(nser):
            p = 1 / 6; path = np.empty(n, int); path[0] = rng.integers(0, len(xd))
            for t in range(1, n): path[t] = rng.integers(0, len(xd)) if rng.random() < p else (path[t - 1] + 1) % len(xd)
            y0 = xd[path] + 0.01
            y1 = y0.copy(); y1[157:] -= 0.005
            rej0 += supf(y0, kw['hac'], kw['lag']) > crit
            rej1 += supf(y1, kw['hac'], kw['lag']) > crit
        out[name] = {'crit95': float(crit), 'size': rej0 / nser, 'power_drop0.5pp': rej1 / nser}
        print('B', name, out[name], round(time.time() - T0), flush=True)
    ok = [k for k, v in out.items() if v['size'] <= 0.075]
    pool = ok if ok else list(out)
    best = sorted(pool, key=lambda k: (abs(out[k]['size'] - 0.05), -out[k]['power_drop0.5pp']))[0]
    return {'sims': out, 'selected': best}

def main():
    it = {'note': '版本總數：SPA 實作 4 版、斷點變異數 3 版、族群蒐集清理 4 版；全部只用挑選期資料或模擬，未計算任何真實全期 p 值。'}
    it['C_family_collection'] = [
        {'v': 'F0', 'what': '初次掃描 3,956 欄，納入 1,788 條', 'issue': 'w12/w3 部分檔案的 net 欄是日報酬而非淨值、npos 是持股數 → 25 條窗報酬 > 300%'},
        {'v': 'F1', 'what': '日報酬欄自動轉淨值、排除 npos／r0050 → 1,800 條', 'issue': 'AB_V4 被 integrate_o4::V4|normal（同一序列）去重掉'},
        {'v': 'F2', 'what': '去重時主策略正式資料夾優先 → AB_V4 以 o_v4 正式來源保留；t 2.5118 與 o4 正式值逐位相同', 'issue': ''},
        {'v': 'F3', 'what': '重跑 strategy_ensemble 窮舉 5,076 組（只讀 import elib，S_long|E1 與原檔最大差 0.0）；建立缺淨值成員 121 條（合成，只做敏感度）', 'issue': ''},
    ]
    it['A_SPA_variants'] = part_A()
    it['B_break_variants'] = part_B()
    json.dump(it, open(HERE / 'iterations.json', 'w'), ensure_ascii=False, indent=1)
    print('selected', it['A_SPA_variants']['selected'], it['B_break_variants']['selected'], round(time.time() - T0))

if __name__ == '__main__':
    main()

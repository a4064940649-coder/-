#!/usr/bin/env python3
"""第二套實作互相核對（PREREG 後執行）：arch 8.0.0 的 SPA／StepM（平穩 bootstrap、studentize、不同種子），
與 statsmodels 0.15 的 OLS-CUSUM、遞迴殘差。arch 與 statsmodels 由 pip 安裝在 scratchpad 的獨立目錄（arch_site），
只在本程式載入。比對：同族群×窗×區塊長度的 SPA l/c/u p 值差 ≤ 0.02（不同種子）；CUSUM 統計量相同（iid 版）。
寫入 CROSSCHECK。"""
import sys
ARCH = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/arch_site'
sys.path.insert(0, ARCH)
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE))
import numpy as np, pandas as pd
import arch, statsmodels
from arch.bootstrap import SPA, StepM, StationaryBootstrap
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import fam_common as C, breaks_lib as BL
RES = C.check_prereg()
PR = RES['PREREG']; MAIN_ID = PR['main_id']

def main():
    D = pd.read_parquet(C.DATA / 'family_excess.parquet'); MM = pd.read_csv(C.DATA / 'family_members.csv')
    F = pd.read_parquet(C.DATA / 'family_ret.parquet'); E = pd.read_parquet(C.DATA / 'ensemble_ret.parquet'); B = pd.read_parquet(C.DATA / 'bench.parquet')
    R = pd.concat([F, E], axis=1)
    out = {'arch_version': arch.__version__, 'statsmodels_version': statsmodels.__version__, 'numpy': np.__version__, 'spa': {}}
    mine = RES.get('RESULTS_SPA', {}).get('grid', {})
    for fam, w, L in PR['crosscheck']['cells']:
        cov = MM[f'cov_{w}']
        mem = MM[cov & (MM.F_long if fam == 'F_long' else (MM.F_main if fam == 'F_main' else True))].member_id.tolist()
        m = C.wmask(D.index, *C.WINDOWS[w])
        X = D.loc[m, mem].fillna(0.0)
        bench_loss = np.zeros(len(X))                 # 損失＝−超額；基準（0050）損失 0
        model_loss = -X.to_numpy()
        spa = SPA(bench_loss, model_loss, block_size=L, reps=PR['spa']['B'], bootstrap='stationary', studentize=True, nested=False,
                  seed=PR['crosscheck']['seed'])
        spa.compute()
        pv = spa.pvalues
        sm = StepM(bench_loss, model_loss, size=0.05, block_size=L, reps=PR['spa']['B'], bootstrap='stationary', studentize=True, nested=False,
                   seed=PR['crosscheck']['seed'])
        sm.compute()
        sup = list(sm.superior_models) if sm.superior_models is not None else []
        sup_ids = [mem[int(j)] if not isinstance(j, str) else j for j in sup]
        key = f'{fam}|{w}|L{L}|x0050'
        my = mine.get(key, {})
        # 第二套「標準化」SPA：用 arch 的 StationaryBootstrap 產生重抽（不同亂數流）、逐次迴圈計算，
        # 變異數用 arch 自己的核估計（spa._loss_diff_var）；與自寫向量化版本比較 SPA_c（族群最大）與主策略 focus p。
        Xn = X.to_numpy(); n_ = Xn.shape[0]; f = mem.index(MAIN_ID)
        om2 = np.asarray(spa._loss_diff_var, float); om = np.sqrt(om2); dbar = Xn.mean(axis=0); sn = np.sqrt(n_)
        tst = sn * dbar / om
        thr = -np.sqrt(om2 / n_ * 2 * np.log(np.log(n_)))
        gc = np.where(dbar >= thr, dbar, 0.0)
        bs = StationaryBootstrap(L, Xn, seed=PR['crosscheck']['seed'] + 1)
        Tb = np.empty(PR['spa']['B'])
        for i, (pa, _) in enumerate(bs.bootstrap(PR['spa']['B'])):
            Tb[i] = max(float((sn * (pa[0].mean(0) - gc) / om).max()), 0.0)
        T = max(float(tst.max()), 0.0)
        o2 = {'second_impl_p_SPA_c_familymax': float((Tb >= T).mean()), 'second_impl_p_SPA_c_focus': float((Tb >= tst[f]).mean()) if tst[f] > 0 else 1.0,
              'mine_p_SPA_c_familymax': my.get('p_SPA_c'), 'mine_p_SPA_c_focus': my.get('p_SPA_c_focus')}
        if my:
            o2['studentized_max_abs_diff'] = max(abs(o2['second_impl_p_SPA_c_familymax'] - my['p_SPA_c']), abs(o2['second_impl_p_SPA_c_focus'] - my['p_SPA_c_focus']))
            o2['studentized_within_0.02'] = bool(o2['studentized_max_abs_diff'] <= 0.02)
        o = {'arch_p_lower': float(pv['lower']), 'arch_p_consistent': float(pv['consistent']), 'arch_p_upper': float(pv['upper']),
             'note': 'arch 8.0 的 SPA.compute 以「不標準化」平均的最大值為統計量（studentize 只用於 consistent 門檻），故與自寫的不標準化 SPA（p_SPAns_*）比對',
             'mine_p_lower': my.get('p_SPAns_l'), 'mine_p_consistent': my.get('p_SPAns_c'), 'mine_p_upper': my.get('p_SPAns_u'), **o2,
             'arch_StepM_n_superior': len(sup_ids), 'arch_StepM_main_superior': MAIN_ID in sup_ids,
             'mine_StepM_n_reject': my.get('StepM_5pct_n_reject'), 'mine_StepM_main_reject': my.get('StepM_5pct_rejects_focus')}
        if my:
            o['max_abs_diff'] = max(abs(o['arch_p_lower'] - o['mine_p_lower']), abs(o['arch_p_consistent'] - o['mine_p_consistent']), abs(o['arch_p_upper'] - o['mine_p_upper']))
            o['within_0.02'] = bool(o['max_abs_diff'] <= 0.02)
        out['spa'][key] = o
        print(key, o, flush=True)
    # CUSUM（iid 版）比對 statsmodels
    y = D[MAIN_ID][C.wmask(D.index, *C.WINDOWS['W_long'])].to_numpy()
    st, pval, _ = breaks_cusumolsresid(y - y.mean(), ddof=1)
    mine_c = BL.ols_cusum(y, hac=False)
    out['ols_cusum'] = {'statsmodels_stat': float(st), 'statsmodels_p': float(pval), 'mine_stat_iid': mine_c['stat'], 'mine_p_iid': mine_c['p']}
    print(out['ols_cusum'])
    C.save_key('CROSSCHECK', out)

if __name__ == '__main__':
    main()

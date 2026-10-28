#!/usr/bin/env python3
"""步驟 3：資料窺探檢定（PREREG 後才可執行；只跑一次）。
族群 {F_all, F_main, F_long} × 窗 {W_long, W_19, W_21} × 平均區塊長度 {3, 6, 12}，B=10,000，固定種子。
損失：對 0050 窗超額（主）、對同池等權（副）。另做：F_main_core（去掉 2330／0050／處置疊加）、
合成成員敏感度（報告值、悲觀版 t≥2）、挑選／驗證／保留子期間描述、有效試驗數、DSR。寫入 results.json 的 RESULTS_SPA。"""
import sys, time, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C, spa_lib as S
RES = C.check_prereg()
PR = RES['PREREG']
MAIN_ID = PR['main_id']
B = PR['spa']['B']; LS = PR['spa']['block_lengths']; VAR = PR['spa']['variance']
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

def seed_of(w, L): return PR['spa']['seed_base'] + 1000 * list(C.WINDOWS).index(w) + L

def families(MM, w):
    cov = MM[f'cov_{w}']
    core_re = re.compile(PR['family']['F_main_core_exclude_regex'])
    fams = {'F_all': MM[cov].member_id.tolist(),
            'F_main': MM[cov & MM.F_main].member_id.tolist(),
            'F_long': MM[cov & MM.F_long].member_id.tolist()}
    fams['F_main_core'] = [m for m in fams['F_main'] if not core_re.search(m) or m == MAIN_ID]
    return fams

def run_one(X, Wc, L, mem, extra=None):
    f = mem.index(MAIN_ID)
    o = S.mcp_tests(X, Wc, L, VAR, focus=f)
    o['top_members'] = [(mem[j], t) for j, t in zip(o.pop('top_idx'), o.pop('top_t'))]
    o['argmax_member'] = mem[o.pop('argmax')]
    if extra: o.update(extra)
    return o

def main():
    D = pd.read_parquet(C.DATA / 'family_excess.parquet'); Dew = pd.read_parquet(C.DATA / 'family_excess_ew.parquet')
    MM = pd.read_csv(C.DATA / 'family_members.csv'); SY = pd.read_parquet(C.DATA / 'synth_excess.parquet')
    out = {'grid': {}, 'eff_trials': {}, 'dsr': {}, 'synthetic': {}, 'subperiod': {}, 'family_sizes': {}}
    for w, (a, b) in C.WINDOWS.items():
        m = C.wmask(D.index, a, b); n = int(m.sum())
        fams = families(MM, w)
        out['family_sizes'][w] = {k: len(v) for k, v in fams.items()}
        for fam, mem in fams.items():
            X = D.loc[m, mem].fillna(0.0).to_numpy()
            et = S.eff_trials(X); out['eff_trials'][f'{fam}|{w}'] = et
            sr_all = X.mean(axis=0) / X.std(axis=0, ddof=1)
            xm = X[:, mem.index(MAIN_ID)]
            out['dsr'][f'{fam}|{w}'] = {k: S.dsr(xm, sr_all, nt) for k, nt in
                                        (('LiJi', et['M_eff_LiJi']), ('Nyholt', et['M_eff_Nyholt']), ('raw', et['N']))}
        for L in LS:
            Wc = S.sb_counts(n, L, B, seed_of(w, L))
            for fam, mem in fams.items():
                for loss, DD in (('x0050', D), ('xEW', Dew)):
                    if loss == 'xEW' and fam == 'F_main_core': continue
                    X = DD.loc[m, mem].fillna(0.0).to_numpy()
                    out['grid'][f'{fam}|{w}|L{L}|{loss}'] = run_one(X, Wc, L, mem)
                log(w, L, fam, {k: round(v, 4) for k, v in out['grid'][f'{fam}|{w}|L{L}|x0050'].items() if k.startswith('p_')})
            # 合成成員敏感度（只做 L=6）
            if L == 6:
                for fam in ('F_all', 'F_long'):
                    mem = fams[fam]
                    syn = [c for c in SY.columns if SY.loc[m, c].notna().mean() >= 0.975]
                    for ver in ('reported', 'pessimistic'):
                        Y = SY.loc[m, syn].fillna(0.0).copy()
                        if ver == 'pessimistic':
                            # 每個合成成員在窗內的 t 提高到至少 2.0（維持波動與相關）
                            for c in syn:
                                y = Y[c].to_numpy(); t = C.tstat(y)
                                if t < 2.0: Y[c] = y + (2.0 - t) * y.std(ddof=1) / np.sqrt(len(y))
                        X = np.hstack([D.loc[m, mem].fillna(0.0).to_numpy(), Y.to_numpy()])
                        mm = mem + syn
                        out['synthetic'][f'{fam}+SYN_{ver}|{w}|L6'] = run_one(X, Wc, L, mm, {'n_syn': len(syn)})
                log(w, 'synthetic done')
    # 子期間描述（F_long，L=6，對 0050）
    MMl = MM[MM.F_long]
    for sp, (a, b) in C.SUBPER.items():
        m = C.wmask(D.index, a, b); n = int(m.sum())
        mem = MMl.member_id.tolist()
        Wc = S.sb_counts(n, 6, B, PR['spa']['seed_base'] + 7000 + n)
        X = D.loc[m, mem].fillna(0.0).to_numpy()
        out['subperiod'][f'F_long|{sp}|L6'] = run_one(X, Wc, 6, mem, {'n_windows': n})
        log('sub', sp)
    # 判定欄
    jk = PR['decision']['spa_column']
    g = out['grid'][jk]
    out['decision_value'] = {'key': jk, 'p_SPA_c_focus': g['p_SPA_c_focus'], 'p_SPA_c_familymax': g['p_SPA_c'],
                             'focus_rank': g['focus_rank'], 'focus_t': g['focus_t']}
    p = g['p_SPA_c_focus']
    out['decision_grade'] = ('維持第 2 級，註記「通過資料窺探校正」' if p <= 0.05 else
                             ('第 2 級，註記「邊緣」' if p <= 0.10 else '對外級別改為第 3 級「有理論支持，但未通過資料窺探校正」，SOP 預期超額只報區間下緣'))
    C.save_key('RESULTS_SPA', out)
    log('done', out['decision_value'], out['decision_grade'])

if __name__ == '__main__':
    main()

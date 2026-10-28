#!/usr/bin/env python3
"""建立事前登記的先驗族群成員清單（只用相關係數，不含 α 資訊）：MAIN、ALL、LOCO、THEORY。輸出 data/families.json。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S
from explore import THEORY_FOLDERS

def main():
    X, M = S.load_family()
    cl = pd.read_csv(S.DATA / 'clusters_k161.csv')
    lab2, med2, _ = S.cluster_medoids(X, 161)
    assert list(cl.cluster) == list(lab2), '群集結果與 explore 不一致'
    med = cl[cl.is_medoid].member_id.tolist()
    main_c = int(cl.set_index('member_id').loc[S.MAIN_ID, 'cluster'])
    loco = [m for m in med if int(cl.set_index('member_id').loc[m, 'cluster']) != main_c]
    th_mask = M.F_main | M.folder.isin(THEORY_FOLDERS)
    tcols = M.member_id[th_mask].tolist()
    kt = int(round(S.li_ji(X[tcols].values)))
    labt, medt, _ = S.cluster_medoids(X[tcols], kt)
    fam = {'MAIN': med, 'ALL': list(X.columns), 'LOCO': loco, 'THEORY': medt,
           'meta': {'main_cluster': main_c, 'n_main_cluster': int((cl.cluster == main_c).sum()), 'k_theory': kt,
                    'main_in_MAIN_medoids': S.MAIN_ID in med, 'B10_in_MAIN_medoids': S.B10_ID in med,
                    'theory_main_cluster_size': int((labt == labt[tcols.index(S.MAIN_ID)]).sum())}}
    (S.DATA / 'families.json').write_text(json.dumps(fam, ensure_ascii=False, indent=1), encoding='utf-8')
    print({k: len(v) for k, v in fam.items() if k != 'meta'}, fam['meta'])

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""步驟三之一：重建登記重算集合的紙上名單（不看報酬）：C0（main_v2 gapfix 正式、排除金融 n10）、U_opt（ER2 樂觀放回）、U_opt_F2（FM 靜態遮罩）。
名單生成與 integrate_o16/run.py 完全相同（只讀 import o_finmask_delisted_fin_erratum 的 fclass／flib → erlib → v2_core → ia_core）。
核對：C0 名單與 o14 lists_C0 逐筆相同、U_opt 放回槽位與 ERRATUM2 相同（與 integrate_o16 相同斷言）。
輸出 scratchpad/o17_delist/plans_ia.pkl。python3 -I -B s4_plans.py"""
import sys, json, pickle, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
FM = AG / 'o_finmask_delisted_fin_erratum'; ER = AG / 'o_rev_trunc_2009_erratum'
sys.path.insert(0, str(HERE)); import cz_lib as Z; sys.path.pop(0)
SPEC = json.loads((FM / 'ERRATUM_SPEC_FM.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(FM))
import fclass as FCL, flib as FL
sys.path.pop(0)
EL, V, K = FL.E, FL.V, FL.K
COLS = FL.COLS; LO, HI = EL.LO, EL.HI
assert Z.sha(AG / 'independent_audit' / 'ia_core.py') == SPEC['files']['engine']['ia_core']
assert Z.sha(AG / 'main_v2' / 'v2_core.py') == SPEC['files']['engine']['v2_core']
cls = SPEC['classifiers']
CL = FCL.Classifier(cls['keywords'], cls['f2_manual_exclusion'], cls['name_exclusion'])
C0M = FL.C0MASK.copy()
F2V = np.array([CL.f2(c) for c in COLS])
scoped = lambda fn: (lambda d: fn(d) if LO <= pd.Timestamp(d) <= HI else C0M)
MASKF = {'C0': lambda d: C0M, 'F2': scoped(lambda d: F2V)}
VARS = {'C0': ('C0', 'base'), 'U_opt': ('C0', 'opt'), 'U_opt_F2': ('F2', 'opt')}
OUT = {}
for k, (mk, mode) in VARS.items():
    S = FL.VM(MASKF[mk], mode, 'SURV')
    pl = S.plan(10, 'official'); t0, t1, sd = S.bounds()
    OUT[k] = dict(plan=[(int(p[0]), np.array(p[1], int), str(pd.Timestamp(p[3]).date())) for p in pl], t0=int(t0), t1=int(t1), sd=[str(pd.Timestamp(x).date()) for x in sd])
    print(k, len(pl), t0, t1, flush=True)
PK = {k: {p[2]: [COLS[j] for j in p[1]] for p in v['plan']} for k, v in OUT.items()}
L0 = json.loads((AG / 'o_g8_protocol_fin_erratum' / 'ERRATUM.json').read_text(encoding='utf-8'))['lists_C0']
days13 = [d for d in PK['C0'] if pd.Timestamp(LO) <= pd.Timestamp(d) <= pd.Timestamp(HI)]
assert len(days13) == 93 and all(PK['C0'][d] == L0[d] for d in days13)
FMC = json.loads((FM / 'FM_COUNTS.json').read_text(encoding='utf-8'))
for k, key in (('C0', 'C0|C0'), ('U_opt', 'C0|U_opt'), ('U_opt_F2', 'F2|U_opt')):
    assert all(PK[k][d] == FMC['lists'][key][d] for d in days13), k
    assert all(PK[k][d] == PK['C0'][d] for d in PK[k] if pd.Timestamp(d) > pd.Timestamp(HI)), k
(Z.WORK / 'plans_ia.pkl').write_bytes(pickle.dumps(OUT))
meta = dict(n={k: len(v['plan']) for k, v in OUT.items()}, sha=hashlib.sha256((Z.WORK / 'plans_ia.pkl').read_bytes()).hexdigest(),
            lists_sha={k: hashlib.sha256(json.dumps(PK[k], sort_keys=True).encode()).hexdigest() for k in PK})
(HERE / 'plans_ia_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print(meta)

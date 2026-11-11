# -*- coding: utf-8 -*-
"""K 角色（目標產生器，非複製程式）：以原引擎 cz_v7x.CSim(cmode='orig')（＝x_exec.Sim）在獨立子行程跑指定 (j, seed, cell)，
記錄每個換股日／順延日處理後的持股、現金、待執行量與每日淨值。原程式只讀 import，不修改；輸出寫到本題 out/。
本檔不得被 A／B／C 的重建或追蹤程式 import。python3 -I -B k_orig_trace.py J SEED COST TIER"""
import sys, json, gzip
from pathlib import Path
import numpy as np, pandas as pd
AG = Path('/home/user/-/research/tw_stock_selection/agents')
sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV; sys.path.pop(0)
D, X = CV.D, CV.X
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
j, seed, cost, tier = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
pl = D.plans(js=[j], seeds=[seed])[(j, seed)]
cfg = dict(tier=tier, mode='GROW', model='K5T', cost=cost); cfg.update(BASE)
REC = []
class T(CV.CSim):
    def rebalance(self, t, pk, tr):
        pre = {D.COLS[i]: float(self.S[i]) for i in np.flatnonzero(self.S > 0)}; c0 = self.cash
        super().rebalance(t, pk, tr)
        REC.append(dict(t=int(t), date=str(D.IDX[t].date()), kind='reb', pre=pre, cash_pre=float(c0),
                        S={D.COLS[i]: float(self.S[i]) for i in np.flatnonzero(self.S > 0)}, cash=float(self.cash),
                        cash_late=float(self.cash_late), pend={D.COLS[k]: int(v) for k, v in self.pend.items()}))
    def deferred(self, t, tr):
        super().deferred(t, tr)
        REC.append(dict(t=int(t), date=str(D.IDX[t].date()), kind='def', S={D.COLS[i]: float(self.S[i]) for i in np.flatnonzero(self.S > 0)},
                        cash=float(self.cash), cash_late=float(self.cash_late), pend={D.COLS[k]: int(v) for k, v in self.pend.items()}))
sm = T(pl, cfg, settle=None, cmode='orig')
nav = sm.run()
R4 = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
tag = {'normal': 'normal', 'stress': 'stress'}[cost]; tn = {1e6: '100萬', 3e6: '300萬', 1e7: '1000萬', 3e5: '30萬'}[tier]
out = dict(j=j, seed=seed, cost=cost, tier=tier, cfg=cfg, rec=REC, nav={str(k.date()): float(v) for k, v in nav.items()}, acc={k: (v if isinstance(v, (int, float)) else str(v)) for k, v in sm.acc.items() if not isinstance(v, (list, dict))})
fn = AG / f'o_g6_allpit_list_trace_stage2/out/k_orig_J{j}_s{seed}_{cost}_{int(tier)}.json.gz'
with gzip.open(fn, 'wt') as f: json.dump(out, f)
print('ok', fn, len(REC), float(nav.iloc[-1]))

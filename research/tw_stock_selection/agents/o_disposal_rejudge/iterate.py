#!/usr/bin/env python3
"""迭代建模（只用挑選期：訊號日 2006-04～2021-08，評價到 2021-09 換股訊號日收盤）。輸出 iterations.json。
版本 E0～E5（事前列出）＋鄰近版本（E3@0.5、E3@0.7、各版 n30，只報告不選版）；一般成本與壓力成本（處置中交易每邊 +0.5%）。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import drj_core as R
K = R.K
HERE = R.HERE
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PLAN_sha256' in res
assert 'PREREG' not in res, 'PREREG 已存在：迭代只能在凍結前做'
LISTV = sys.argv[1] if len(sys.argv) > 1 else 'A'
D = R.load_list(use_proxy_gapyears=(LISTV == 'AP'))
DS = R.Disp(D)
NORMAL = K.rates(0.001)
A, B = R.SEL_A, R.SEL_B

def run_version(ver, n, cost):
    S = R.trim(R.make_strategy(ver if ver != 'E4' else 'E0', DS), B)
    t0, t1, sd = S.bounds()
    pl = S.plan(n, 'official')
    dm = R.defer_map(pl, DS) if ver == 'E4' else None
    b, s = NORMAL
    extra = 0.005 if cost == 'stress' else None
    nav, lg = R.simulate2(pl, b, s, t0, t1, defer=dm, extra=extra, DS=DS)
    return nav, sd, pl, lg

out = {'list_version': LISTV, 'period': f'signal days {A.date()}..{B.date()}', 'versions': {}}
NAV = {}
T0 = time.time()
for n in (10, 30):
    for cost in ('normal', 'stress'):
        vers = ['E1', 'E0', 'E2', 'E3', 'E4', 'E5'] + (['E3@0.5', 'E3@0.7'] if n == 10 else [])
        for ver in vers:
            nav, sd, pl, lg = run_version(ver, n, cost)
            NAV[(ver, n, cost)] = nav
            cg, mdd = R.cagr_mdd(nav, A, R.IDX[-1])
            o = dict(CAGR=cg, maxDD=mdd, t_vs_0050=R.tvs(nav, R.E50.reindex(nav.index), sd, A, B), turnover=lg['turnover'],
                     n_frozen=len(lg['frozen']), defer_entered=lg.get('defer_entered'), defer_never=lg.get('defer_never'))
            if ver != 'E1':
                o['paired_vs_E1'] = R.paired(nav, NAV[('E1', n, cost)], sd, A, B)
            # 名單差異的名額數
            if ver != 'E1':
                p1 = {p[3]: set(p[1].tolist()) for p in run_version.__defaults__ or []} if False else None
            out['versions'][f'{ver}|n{n}|{cost}'] = o
            print(f'[{time.time()-T0:5.0f}s]', ver, n, cost, round(cg, 2), round(mdd, 2), o.get('paired_vs_E1', {}).get('mean_pp'), o.get('paired_vs_E1', {}).get('t'), flush=True)
# 選版（事前規則）：n10 一般成本，E0/E2/E3/E4/E5 中 paired t 最高且 MDD 不比 E1 深 3pp 以上
e1 = out['versions']['E1|n10|normal']
cands = []
for v in ['E0', 'E2', 'E3', 'E4', 'E5']:
    o = out['versions'][f'{v}|n10|normal']
    ok = o['maxDD'] >= e1['maxDD'] - 3.0
    cands.append(dict(version=v, t=o['paired_vs_E1']['t'], mean_pp=o['paired_vs_E1']['mean_pp'], mdd=o['maxDD'], mdd_ok=bool(ok)))
valid = [c for c in cands if c['mdd_ok'] and np.isfinite(c['t'])]
sel = max(valid, key=lambda c: c['t']) if valid else None
out['selection'] = dict(rule='n10 一般成本：E0/E2/E3/E4/E5 中對 E1 配對 t 最高、且 MDD 不比 E1 深 3pp 以上', candidates=cands, selected=sel['version'] if sel else None)
# 最小可偵測效果（E0−E1，n10 一般）
p = out['versions']['E0|n10|normal']['paired_vs_E1']
se = p['sd_pp'] / np.sqrt(p['n'])
out['MDE_E0_minus_E1_pp_per_month'] = dict(se_pp=se, mde_80pow_5pct=2.8 * se)
pd.DataFrame({f'{k[0]}|n{k[1]}|{k[2]}': v for k, v in NAV.items()}).to_parquet(HERE / f'navs_sel_{LISTV}.parquet')
fn = HERE / 'iterations.json'
it = json.loads(fn.read_text(encoding='utf-8')) if fn.exists() else {}
it[LISTV] = out
it['n_versions_tested'] = '6 個事前版本（E0～E5）×n10/n30×一般/壓力，加鄰近版本 E3@0.5、E3@0.7（只報告）；名單 A 與 A+P 各一次'
fn.write_text(json.dumps(it, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(out['selection'], ensure_ascii=False), out['MDE_E0_minus_E1_pp_per_month'])

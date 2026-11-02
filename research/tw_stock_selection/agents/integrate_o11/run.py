#!/usr/bin/env python3
"""integrate_o11 正式執行（只跑一次）：先驗 PREREG 雜湊，再算主策略（排除金融 n10，AB_V7）× 0050 固定比例混合，
並把 o_g2_feasibility_stoprule 的 S2 曲線與 G3 族群套到每個組合。輸出寫入 results.json 的 RESULTS（只寫一次）。
python3 -I -B run.py"""
import json, hashlib, math, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
AG = HERE.parent
G2 = AG / 'o_g2_feasibility_stoprule'
RJ = HERE / 'results.json'

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'RESULTS' not in R, 'RESULTS 已存在：只跑一次'
P = R['PREREG']
assert hashlib.sha256(json.dumps(P, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改'
for f, h in P['sha256'].items():
    assert sha(f) == h, f'{f} 雜湊不符'

FC = _load('fam_common_ro', AG / 'o_main_spa_breaks' / 'fam_common.py')
SP = _load('spa_lib_ro', AG / 'o_main_spa_breaks' / 'spa_lib.py')
GRID, WIN = FC.GRID, FC.WIN
T = len(WIN)
r50 = FC.win_ret(FC.e50_nav()).to_numpy()
assert np.isfinite(r50).all()

JS = pd.read_parquet(AG / 'o_v7_listing_exec_engine' / 'navs_jscan_v7.parquet')
OFF = pd.read_parquet(AG / 'o_v7_listing_exec_engine' / 'navs_v7.parquet')
EX = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet')

def wr(nav):
    r = FC.win_ret(nav).to_numpy(); nan = int((~np.isfinite(r)).sum()); r = np.where(np.isfinite(r), r, 0.0); return r, nan
NANLOG = {}
def jmat(cols, tag):
    out = []; nn = []
    for c in cols:
        r, n = wr(c)
        out.append(r); nn.append(n)
    NANLOG[tag] = nn
    return np.column_stack(out)

LEG = {}
LEG['P|normal'] = jmat([JS[f'AB_V7|normal|J{j}'] for j in range(13)], 'P|normal')
LEG['P|stress'] = jmat([JS[f'AB_V7|stress|J{j}'] for j in range(13)], 'P|stress')
LEG['PB10|normal'] = jmat([JS[f'AB_V7_B10|normal|J{j}'] for j in range(13)], 'PB10|normal')
TIERS = ['100萬', '300萬', '1000萬']
for cost in ('normal', 'stress'):
    for tier in TIERS:
        LEG[f'X|{cost}|{tier}'] = jmat([EX[f'B10|J{j}|K5T|{cost}|{tier}|GROW'] for j in range(13)], f'X|{cost}|{tier}')
LEG['OFF|normal'] = jmat([OFF['AB_V7|normal']], 'OFF|normal')
E = np.repeat(r50[:, None], 13, axis=1)

# 移轉成本（每單位移轉金額）
TC = {'normal': {'M2E': 0.004855 + 0.001855, 'E2M': (0.001855 + 0.001) + 0.001855}, 'stress': {'M2E': 0.01, 'E2M': 0.01}}

def mix(rM, w, cost):
    """袖子法：每窗起點比例 w（主策略）/1−w（0050），窗末調回，移轉金額扣成本。rM: (T,J)。"""
    if w == 1.0: return rM.copy()
    if w == 0.0: return np.repeat(r50[:, None], rM.shape[1], axis=1)
    a = w * (1 + rM); b = (1 - w) * (1 + r50[:, None]); V = a + b
    m = a - w * V                      # >0：主策略腿超重 → 賣主策略買 0050
    c = np.where(m > 0, m * TC[cost]['M2E'], -m * TC[cost]['E2M'])
    return V - c - 1.0

YEARS = WIN.year.to_numpy()
PER = {'ALL': ('2006-04-01', '2026-10-12'), 'SEG1': ('2006-04-01', '2012-12-31'), 'SEG2': ('2013-01-01', '2018-12-31'),
       'SEG3': ('2019-01-01', '2026-10-12'), 'HOLD': ('2024-01-01', '2026-10-12')}
MASK = {k: (WIN >= pd.Timestamp(a)) & (WIN <= pd.Timestamp(b)) for k, (a, b) in PER.items()}

def tt(x):
    x = np.asarray(x, float); s = x.std(ddof=1); return float(x.mean() / s * math.sqrt(len(x))) if s > 0 else float('nan')
def ann(r): return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
def mdd(r):
    nav = np.cumprod(1 + r); pk = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]; return float((nav / pk - 1).min()) * 100
def drop3(r):
    uy = np.unique(YEARS)
    yex = np.array([np.prod(1 + r[YEARS == y]) - np.prod(1 + r50[YEARS == y]) for y in uy])
    bad = uy[np.argsort(-yex)[:3]]; keep = ~np.isin(YEARS, bad)
    return tt((r - r50)[keep])
def q(v): v = np.asarray(v, float); return {'med': float(np.median(v)), 'p10': float(np.percentile(v, 10)), 'min': float(v.min()), 'max': float(v.max())}

def stats_block(rX, rM):
    out = {}
    for pn, mk in MASK.items():
        J = rX.shape[1]
        a = [ann(rX[mk, j]) for j in range(J)]; a50 = ann(r50[mk])
        d = {'ann': q(a), 'ann_0050': a50, 'gex': q([x - a50 for x in a]), 'mdd': q([mdd(rX[mk, j]) for j in range(J)]), 'mdd_0050': mdd(r50[mk]),
             't_vs_0050': q([tt(rX[mk, j] - r50[mk]) for j in range(J)]), 'mex_pp': q([float((rX[mk, j] - r50[mk]).mean() * 100) for j in range(J)])}
        if rM is not None:
            d['diff_vs_M_pp'] = q([float((rX[mk, j] - rM[mk, j]).mean() * 100) for j in range(J)])
            d['t_diff_vs_M'] = q([tt(rX[mk, j] - rM[mk, j]) if np.std(rX[mk, j] - rM[mk, j]) > 1e-12 else 0.0 for j in range(J)])
            d['corr_with_M'] = q([float(np.corrcoef(rX[mk, j], rM[mk, j])[0, 1]) for j in range(J)])
            d['corr_with_0050'] = q([float(np.corrcoef(rX[mk, j], r50[mk])[0, 1]) for j in range(J)])
            d['corr_excess_with_M_excess'] = q([float(np.corrcoef(rX[mk, j] - r50[mk], rM[mk, j] - r50[mk])[0, 1]) for j in range(J)])
        if pn == 'ALL':
            d['t_drop3'] = q([drop3(rX[:, j]) for j in range(J)])
        out[pn] = d
    return out

WS = P['combos']['W_LIST']
RES = {'nan_windows_filled_as_cash': NANLOG, 'paper': {}, 'exec': {}, 'official_day': {}}
for w in WS:
    k = f'w={w}'
    RES['paper'][k] = {c: stats_block(mix(LEG[f'P|{c}'], w, c), LEG[f'P|{c}']) for c in ('normal', 'stress')}
    RES['paper'][k]['B10_normal'] = stats_block(mix(LEG['PB10|normal'], w, 'normal'), LEG['PB10|normal'])
    RES['official_day'][k] = stats_block(mix(LEG['OFF|normal'], w, 'normal'), LEG['OFF|normal'])
    RES['exec'][k] = {f'{c}|{t}': stats_block(mix(LEG[f'X|{c}|{t}'], w, c), LEG[f'X|{c}|{t}']) for c in ('normal', 'stress') for t in TIERS}
RES['paper']['0050'] = stats_block(E, None)

# ---- o_g2 的 S2 曲線與 G3 族群 ----
G2R = json.loads((G2 / 'results.json').read_text(encoding='utf-8'))
TAB = G2R['FINAL']['table_V1']
IRG = sorted(float(x) for x in TAB)
def interp(col, x): return float(np.interp(x, IRG, [TAB[str(g)][col] for g in IRG]))
FI = pd.read_parquet(G2 / 'data' / 'family_ir.parquet')
sub = FI[FI.keep_folder & FI.IR_ALL.notna()]
grid = np.round(np.arange(-2.0, 2.5 + 1e-9, 0.01), 2)
def npmle(x, se, iters=5000, tol=1e-10):
    L = np.exp(-0.5 * ((x[:, None] - grid[None, :]) / se[:, None]) ** 2) / se[:, None]
    wv = np.full(len(grid), 1.0 / len(grid)); ll0 = -np.inf
    for _ in range(iters):
        f = L @ wv; wv = wv * (L / f[:, None]).mean(axis=0); ll = np.log(L @ wv).sum()
        if ll - ll0 < tol: break
        ll0 = ll
    return wv
WPRI = npmle(sub.IR_ALL.to_numpy(), sub.se_iid.to_numpy())
share05 = float(WPRI[grid >= 0.5 - 1e-9].sum())
def post_mean(x0, se0):
    p = WPRI * np.exp(-0.5 * ((x0 - grid) / se0) ** 2); p /= p.sum(); return float((p * grid).sum()), float(p[grid >= 0.5 - 1e-9].sum())
SI = np.load(G2 / 'data' / 'sim_inputs.npz')
Tb_all = SI['Tb_all']; sr_all = SI['sr_all']
def ir(x): return float(x.mean() / x.std(ddof=1) * math.sqrt(12))
def se_iid(x):
    srm = x.mean() / x.std(ddof=1); return math.sqrt(12) * math.sqrt((1 + srm ** 2 / 2) / len(x))
def g3(xjavg):
    om = math.sqrt(float(SP.lrv(xjavg[:, None], 'pr', 6)[0])); t = math.sqrt(T) * xjavg.mean() / om
    p_lb = float((Tb_all >= t).mean()) if t > 0 else 1.0
    sr_c = xjavg.mean() / xjavg.std(ddof=1)
    d = SP.dsr(xjavg, np.append(sr_all, sr_c), len(sr_all) + 1)
    return {'t_studentized_PR': t, 'p_SPA_c_lower_bound': p_lb, 'DSR': d['DSR'], 'SR0_monthly': d['SR0'], 'SR_monthly': d['SR']}

S2 = {}; GATES = {}
for w in WS:
    k = f'w={w}'
    xs = {f'{c}|{t}': mix(LEG[f'X|{c}|{t}'], w, c) - r50[:, None] for c in ('normal', 'stress') for t in TIERS}
    xp = mix(LEG['P|normal'], w, 'normal') - r50[:, None]
    xj = xs['normal|100萬'].mean(1)
    irx = ir(xj); pmx, p05x = post_mean(irx, se_iid(xj))
    irp = ir(xp.mean(1)); pmp, p05p = post_mean(irp, se_iid(xp.mean(1)))
    row = {'IR_exec_javg_100w': irx, 'IR_paper_javg': irp, 'post_mean_exec': pmx, 'post_mean_paper': pmp, 'P_true_ge_0.5_exec': p05x, 'P_true_ge_0.5_paper': p05p}
    for lab, x in (('measured_exec', irx), ('post_exec', pmx), ('measured_paper', irp), ('post_paper', pmp)):
        row[lab] = {col: interp(col, max(x, 0.0)) for col in ('joint_G1_G7', 'joint_without_DSR', 'A', 'A_plus_G1_G7')}
    row['S2_open_topic'] = row['post_exec']['joint_G1_G7'] >= 0.20
    row['S2_open_topic_even_measured_paper'] = row['measured_paper']['joint_G1_G7'] >= 0.20
    S2[k] = row
    # G1～G8（可執行 100 萬 GROW 為主欄；13 個 j）
    eb = RES['exec'][k]
    g = {}
    a = eb['normal|100萬']['ALL']; s = eb['stress|100萬']['ALL']
    g['G1'] = {'pass': a['gex']['med'] > 0 and a['gex']['p10'] > 0 and s['gex']['med'] > 0, 'gex_med': a['gex']['med'], 'gex_p10': a['gex']['p10'], 'stress_gex_med': s['gex']['med']}
    g['G2'] = {'pass': a['t_vs_0050']['med'] >= 3.0 and eb['normal|100萬']['SEG3']['t_vs_0050']['med'] >= 2.0, 't_ALL_med': a['t_vs_0050']['med'],
               't_Y19_med': eb['normal|100萬']['SEG3']['t_vs_0050']['med'], 'paper_t_ALL_med': RES['paper'][k]['normal']['ALL']['t_vs_0050']['med'],
               'paper_t_Y19_med': RES['paper'][k]['normal']['SEG3']['t_vs_0050']['med']}
    g3r = g3(xj)
    g['G3'] = {'pass': (g3r['p_SPA_c_lower_bound'] < 0.05) and (g3r['DSR'] >= 0.95), **g3r,
               'note': 'p 為下界（只用族群最大值分布，未加候選自己的重新置中統計量）；下界 ≥ 0.05 即確定不過'}
    seg_ok = all(eb['normal|100萬'][sg]['gex']['med'] > 0 for sg in ('SEG1', 'SEG2', 'SEG3'))
    nb = [RES['paper'][k]['normal']['ALL']['gex']['med'], RES['paper'][k]['B10_normal']['ALL']['gex']['med']]
    i = WS.index(w); nbw = [WS[x] for x in (i - 1, i + 1) if 0 <= x < len(WS)]
    nb += [RES['exec'][f'w={v}']['normal|100萬']['ALL']['gex']['med'] for v in nbw]
    g['G4'] = {'pass': seg_ok and a['t_drop3']['med'] >= 2.0 and all(v > 0 for v in nb), 'seg_gex_med': {sg: eb['normal|100萬'][sg]['gex']['med'] for sg in ('SEG1', 'SEG2', 'SEG3')},
               't_drop3_med': a['t_drop3']['med'], 'neighbors_gex_med': nb, 'neighbor_w': nbw}
    g['G5'] = {'pass': None, 'note': '沿用主策略 V7／V7X 的資料判定；本整合沒有新增資料，不另驗'}
    g['G6'] = {'pass': False, 'note': '本整合沒有獨立複製'}
    g7 = {}
    for t in ('300萬', '1000萬'):
        b = eb[f'normal|{t}']
        g7[t] = {'gex_med': b['ALL']['gex']['med'], 'gex_p10': b['ALL']['gex']['p10'], 't_ALL_med': b['ALL']['t_vs_0050']['med'], 't_Y19_med': b['SEG3']['t_vs_0050']['med'],
                 'pass': b['ALL']['gex']['med'] > 0 and b['ALL']['gex']['p10'] > 0 and b['ALL']['t_vs_0050']['med'] >= 3.0 and b['SEG3']['t_vs_0050']['med'] >= 2.0}
    g['G7'] = {'pass': all(v['pass'] for v in g7.values()), **g7}
    g['G8'] = {'pass': False, 'note': '前瞻紙上追蹤尚未開始'}
    GATES[k] = g

# ---- 判定（事前規則）----
DEC = {}
m1 = RES['paper']['w=1.0']['normal']['ALL']['t_vs_0050']['med']
for w in WS:
    if w == 1.0: continue
    k = f'w={w}'; pb = RES['paper'][k]['normal']; eb = RES['exec'][k]
    c1 = pb['ALL']['t_vs_0050']['med'] >= 2.0
    c2 = pb['SEG3']['gex']['med'] > 0
    c3 = all(eb[f'normal|{t}']['ALL']['gex']['med'] > 0 and eb[f'normal|{t}']['ALL']['gex']['p10'] > 0 for t in TIERS)
    c4 = eb['stress|100萬']['ALL']['gex']['med'] > 0
    c5 = pb['ALL']['t_vs_0050']['med'] > m1
    v = '可用' if (c1 and c2 and c3 and c4 and c5) else ('待觀察' if c1 else '不可用')
    DEC[k] = {'c1_paper_ALL_t_ge2': c1, 'c2_paper_Y19_gex_pos': c2, 'c3_exec_normal_3tiers_med_p10_pos': c3, 'c4_exec_stress_100w_med_pos': c4,
              'c5_paper_ALL_t_gt_M_alone': c5, 'M_alone_paper_ALL_t': m1, 'verdict': v}

RES['S2'] = S2; RES['GATES'] = GATES; RES['DECISION'] = DEC
RES['prior'] = {'n_members': int(len(sub)), 'share_true_ge_0.5': share05, 'g2_reported_share_ge_0.5': G2R['FINAL']['stop_rule']['S1_c_share_true_ge05_NPMLE']}
R['RESULTS'] = RES
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps({'DECISION': DEC, 'S2': {k: {'post_exec': v['post_exec'], 'IR_exec': v['IR_exec_javg_100w'], 'IR_paper': v['IR_paper_javg']} for k, v in S2.items()}}, ensure_ascii=False, indent=1, default=float))

#!/usr/bin/env python3
"""合併 run_P0.json、run_P1.json → results.json 的 RESULTS，並印出 summary.txt（FINDINGS 的表格來源）。
python3 -I -B summarize.py"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
R0 = json.loads((HERE / 'run_P0.json').read_text(encoding='utf-8')); R1 = json.loads((HERE / 'run_P1.json').read_text(encoding='utf-8'))
RUNS = {**R0['runs'], **R1['runs']}
SEG = ['S1', 'S2', 'S3', 'S4', 'P1', 'ALL']
L = []
def p(*a): L.append(' '.join(str(x) for x in a))
def g(key, seg='ALL', f='CAGR'):
    return RUNS[key]['seg'][seg][f]
def K_(P, R, X, D, u, n, c='normal'): return f'P{P}|R{R}|X{X}|D{D}|{u}|n{n}|{c}'

# ---------- V3 的 D 端（事前規則） ----------
lo, hi = g(K_(1, 1, 1, 'lo', 'EXFIN', 10)), g(K_(1, 1, 1, 'hi', 'EXFIN', 10))
DW = 'lo' if lo <= hi else 'hi'
p(f'V3 的 D 端：D_lo {lo:.2f}、D_hi {hi:.2f} → D_worse＝{DW}')
V3 = lambda u, n, c='normal': K_(1, 1, 1, DW, u, n, c)
V2 = lambda u, n, c='normal': K_(0, 0, 0, 'v2', u, n, c)

# ---------- 主表 ----------
def row(key):
    s = RUNS[key]['seg']
    return ' | '.join(f"{s[k]['CAGR']:.1f}（{s[k]['maxDD']:.0f}）t {s[k]['t_0050']:.2f}/{s[k]['t_EW']:.2f}" for k in SEG)
p('\n## 主表（年化%（最大回撤）t 對0050/對等權；訊號窗）', ' | '.join(SEG))
VERS = [('V2 正式', (0, 0, 0, 'v2')), ('+R（R_DEL 營收）', (0, 1, 0, 'v2')), ('+X（單獨）', (0, 0, 1, 'v2')), ('+P（單獨）', (1, 0, 0, 'v2')),
        ('+D_lo（單獨）', (0, 0, 0, 'lo')), ('+D_hi（單獨）', (0, 0, 0, 'hi')), ('D_none（單獨，全期不排除）', (0, 0, 0, 'none')),
        ('R+X', (0, 1, 1, 'v2')), ('R+X+P', (1, 1, 1, 'v2')), ('V3_lo＝R+X+P+D_lo', (1, 1, 1, 'lo')), ('V3_hi＝R+X+P+D_hi', (1, 1, 1, 'hi')),
        ('V3−D＝R+X+P+D_none', (1, 1, 1, 'none')), ('X+P（無 R）', (1, 0, 1, 'v2')), ('X+P+D_hi（無 R）', (1, 0, 1, 'hi'))]
for u in ('EXFIN', 'ALL'):
    for n in (10, 30):
        for c in ('normal', 'stress'):
            p(f'\n### {u} n{n} {c}')
            for nm, (P_, R_, X_, D_) in VERS:
                p(f'{nm:28s}', row(K_(P_, R_, X_, D_, u, n, c)))
            p(f"{'等權（V2 池）':28s}", ' | '.join(f"{RUNS[V2(u, n, c)]['seg'][k]['EW_CAGR']:.1f}" for k in SEG))
            p(f"{'等權（V3 池）':28s}", ' | '.join(f"{RUNS[V3(u, n, c)]['seg'][k]['EW_CAGR']:.1f}" for k in SEG))
            p(f"{'0050':28s}", ' | '.join(f"{RUNS[V2(u, n, c)]['seg'][k]['E50_CAGR']:.1f}" for k in SEG))

# ---------- 邊際影響與交互作用（全因子；排除金融 n10 一般成本 ALL 與各段） ----------
def eff(u, n, seg, f='CAGR'):
    out = {}
    import itertools
    for D in ('lo', 'hi'):
        vals = {}
        for P_, R_, X_, DD in itertools.product((0, 1), (0, 1), (0, 1), ('v2', D)):
            vals[(P_, R_, X_, DD)] = g(K_(P_, R_, X_, DD, u, n), seg, f)
        def me(i):
            a = [v for k, v in vals.items() if k[i] in (1, D)]; b = [v for k, v in vals.items() if k[i] in (0, 'v2')]
            return np.mean(a) - np.mean(b)
        out[D] = {'P': me(0), 'R': me(1), 'X': me(2), 'D': me(3),
                  'PxR': (vals[(1, 1, 0, 'v2')] - vals[(0, 1, 0, 'v2')]) - (vals[(1, 0, 0, 'v2')] - vals[(0, 0, 0, 'v2')]),
                  'PxD': (vals[(1, 0, 0, D)] - vals[(0, 0, 0, D)]) - (vals[(1, 0, 0, 'v2')] - vals[(0, 0, 0, 'v2')]),
                  'RxX': (vals[(0, 1, 1, 'v2')] - vals[(0, 1, 0, 'v2')]) - (vals[(0, 0, 1, 'v2')] - vals[(0, 0, 0, 'v2')]),
                  'RxD': (vals[(0, 1, 0, D)] - vals[(0, 0, 0, D)]) - (vals[(0, 1, 0, 'v2')] - vals[(0, 0, 0, 'v2')])}
    return out
EFF = {}
p('\n## 邊際影響（全因子平均主效果，pp）與兩兩交互作用')
for u in ('EXFIN', 'ALL'):
    for n in (10, 30):
        for seg in SEG:
            e = eff(u, n, seg); EFF[f'{u}|n{n}|{seg}'] = e
        p(f'{u} n{n}: ' + ' ; '.join(f"{seg} " + ' '.join(f"{k} {v:+.2f}" for k, v in EFF[f'{u}|n{n}|{seg}'][DW].items()) for seg in ('ALL', 'S2', 'S3', 'P1')))
# 累加路徑
p('\n## 累加路徑（排除金融 n10 一般成本）：年化 ALL／S1／S2／S3／S4，t 對 0050 ALL，t 對等權 ALL')
path = [('V2', (0, 0, 0, 'v2')), ('+R', (0, 1, 0, 'v2')), ('+X', (0, 1, 1, 'v2')), ('+P', (1, 1, 1, 'v2')), (f'+D_{DW}＝V3', (1, 1, 1, DW))]
PATH = {}
for u in ('EXFIN', 'ALL'):
    for n in (10, 30):
        prev = None
        for nm, k in path:
            key = K_(*k, u, n); s = RUNS[key]['seg']
            d = '' if prev is None else f"（ALL {s['ALL']['CAGR'] - prev['ALL']['CAGR']:+.2f}pp，t {s['ALL']['t_0050'] - prev['ALL']['t_0050']:+.2f}，tEW {s['ALL']['t_EW'] - prev['ALL']['t_EW']:+.2f}）"
            p(f'{u} n{n} {nm:10s}', ' / '.join(f"{s[x]['CAGR']:.2f}" for x in ('ALL', 'S1', 'S2', 'S3', 'S4')), f"t {s['ALL']['t_0050']:.2f} tEW {s['ALL']['t_EW']:.2f}", d)
            PATH[f'{u}|n{n}|{nm}'] = {x: s[x] for x in SEG}
            prev = s

# ---------- 凍結事件 ----------
p('\n## 凍結事件（換股日有部位無法成交的次數；最大凍結占比；出清次數）')
for nm, k in VERS:
    r = RUNS[K_(*k, 'EXFIN', 10)]; r30 = RUNS[K_(*k, 'EXFIN', 30)]
    p(f'{nm:28s} n10 {r["n_frozen"]} ({r["max_frozen_share"]:.3f}) 出清 {r["n_exits"]} | n30 {r30["n_frozen"]} ({r30["max_frozen_share"]:.3f}) 出清 {r30["n_exits"]} {r30["exits"][:3]} | 等權 {r["ew_n_frozen"]} ({r["ew_max_frozen_share"]:.3f}) 出清 {r["ew_exits"]}')

# ---------- 界限 ----------
p('\n## 界限（以 R+X+P 為底；年化 ALL／S2／S3／S4，t 對 0050 ALL）')
BND = {}
for D in ('lo', 'hi', 'none'):
    for u in ('EXFIN', 'ALL'):
        for n in (10, 30):
            base = K_(1, 1, 1, D, u, n)
            items = [('X main', base)] + [(b, f'P1|R1|D{D}|{u}|{b}|n{n}|normal') for b in ('X_upper', 'X_lower', 'B05', 'B10', 'DRAG', 'DRAG6251', 'WORST1')]
            for nm, key in items:
                s = RUNS[key]['seg']; BND[f'{D}|{u}|n{n}|{nm}'] = {x: s[x] for x in SEG}
                extra = ''
                if nm in ('B05', 'B10'): extra = f"凍結(有X) {RUNS[key]['n_frozen']} 出清 {RUNS[key]['n_exits']}；無X：ALL {RUNS[key]['noX']['seg_ALL']['CAGR']:.2f} t {RUNS[key]['noX']['seg_ALL']['t_0050']:.2f} 凍結 {RUNS[key]['noX']['n_frozen']}（最大 {RUNS[key]['noX']['max_frozen_share']:.2f}）；入選缺營收股 {RUNS[key]['n_gap_picks']}"
                if nm == 'WORST1': extra = str(RUNS[key]['picks'])
                if nm == 'DRAG': extra = f"拖累合計 {RUNS[key]['drag_sum'] * 100:.2f}pp"
                p(f'D{D} {u} n{n} {nm:9s}', ' / '.join(f"{s[x]['CAGR']:.2f}" for x in ('ALL', 'S2', 'S3', 'S4')), f"t {s['ALL']['t_0050']:.2f} tEW {s['ALL']['t_EW']:.2f}", extra)

# ---------- 均值與波動拆解 ----------
p('\n## t 變動拆解（月均超額 vs 0050、標準差；ALL 訊號窗）')
for nm, k in VERS:
    m = RUNS[K_(*k, 'EXFIN', 10)]['msd']; p(f'{nm:28s} mean {m["mean_pp"]:.3f}pp sd {m["sd_pp"]:.3f} n {m["n"]} t {m["t"]:.3f}')
m2, m3 = RUNS[V2('EXFIN', 10)]['msd'], RUNS[V3('EXFIN', 10)]['msd']
dec = {'t_V2': m2['t'], 't_V3': m3['t'], 't_if_only_mean_changed': m3['mean_pp'] / m2['sd_pp'] * np.sqrt(m3['n']), 'mean_change_pp': m3['mean_pp'] - m2['mean_pp'], 'sd_change_pp': m3['sd_pp'] - m2['sd_pp']}
p('拆解', dec)

# ---------- 逐年 ----------
p('\n## 逐年（排除金融 n10 一般成本）')
yrs = sorted(set(RUNS[V2('EXFIN', 10)]['seg']['ALL']['yearly']) | set())
cols = [('V2', V2('EXFIN', 10)), ('+R', K_(0, 1, 0, 'v2', 'EXFIN', 10)), ('R+X+P', K_(1, 1, 1, 'v2', 'EXFIN', 10)), ('V3_lo', K_(1, 1, 1, 'lo', 'EXFIN', 10)), ('V3_hi', K_(1, 1, 1, 'hi', 'EXFIN', 10)), ('V3−D', K_(1, 1, 1, 'none', 'EXFIN', 10))]
p('年 | ' + ' | '.join(c for c, _ in cols) + ' | 0050')
E50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
YEARLY = {}
for y in RUNS[V2('EXFIN', 10)]['seg']['ALL']['yearly']:
    vals = [RUNS[k]['seg']['ALL']['yearly'].get(y) for _, k in cols]
    YEARLY[y] = dict(zip([c for c, _ in cols], vals))
    p(y, '|', ' | '.join(f'{v:.1f}' if v is not None else '-' for v in vals))

# ---------- 最大單股貢獻 ----------
p('\n## 最大單股貢獻（毛報酬近似）')
for nm, k in (('V2', V2('EXFIN', 10)), ('V3', V3('EXFIN', 10))):
    for seg, v in RUNS[k]['contrib'].items():
        p(nm, seg, v.get('top_code'), v.get('top_name'), f"{v.get('top_contrib_pp', 0):.1f}pp", f"剔除 −{v.get('ex_top_drop_pp', 0):.2f}pp", v.get('top5'))

# ---------- 名單差異 ----------
pk2, pk3 = RUNS[V2('EXFIN', 10)]['picks'], RUNS[V3('EXFIN', 10)]['picks']
diff = {d: sorted(set(pk2.get(d, [])) ^ set(pk3.get(d, []))) for d in pk3 if set(pk2.get(d, [])) != set(pk3.get(d, []))}
nslot = sum(len(set(pk3[d]) - set(pk2.get(d, []))) for d in pk3)
byyear = pd.Series({d: len(set(pk3[d]) - set(pk2.get(d, []))) for d in pk3}); byyear.index = pd.to_datetime(byyear.index)
p('\n## V3 對 V2 名單差異（排除金融 n10）：不同的換股日', len(diff), '名額', nslot, '逐年', byyear.groupby(byyear.index.year).sum().to_dict())

# ---------- 預測檢查 ----------
def d_(a, b, seg='ALL', f='CAGR'): return g(a, seg, f) - g(b, seg, f)
pred = {
  'P1_X_EXFIN_n10_ALL_pp': d_(K_(0, 0, 1, 'v2', 'EXFIN', 10), V2('EXFIN', 10)),
  'P1_X_EXFIN_n10_S2_pp': d_(K_(0, 0, 1, 'v2', 'EXFIN', 10), V2('EXFIN', 10), 'S2'),
  'P1_X_given_R_EXFIN_n10_ALL_pp': d_(K_(0, 1, 1, 'v2', 'EXFIN', 10), K_(0, 1, 0, 'v2', 'EXFIN', 10)),
  'P1_X_given_R_EXFIN_n30_ALL_pp': d_(K_(0, 1, 1, 'v2', 'EXFIN', 30), K_(0, 1, 0, 'v2', 'EXFIN', 30)),
  'P1_frozen_V2_n10': RUNS[V2('EXFIN', 10)]['n_frozen'], 'P1_frozen_V3_n10': RUNS[V3('EXFIN', 10)]['n_frozen'],
  'P2_P_S3_pp_given_RX': d_(K_(1, 1, 1, 'v2', 'EXFIN', 10), K_(0, 1, 1, 'v2', 'EXFIN', 10), 'S3'),
  'P2_P_S3_pp_alone': d_(K_(1, 0, 0, 'v2', 'EXFIN', 10), V2('EXFIN', 10), 'S3'),
  'P2_EW_S3_change_pp': RUNS[K_(1, 1, 1, 'v2', 'EXFIN', 10)]['seg']['S3']['EW_CAGR'] - RUNS[K_(0, 1, 1, 'v2', 'EXFIN', 10)]['seg']['S3']['EW_CAGR'],
  'P2_t0050_S3_change': d_(K_(1, 1, 1, 'v2', 'EXFIN', 10), K_(0, 1, 1, 'v2', 'EXFIN', 10), 'S3', 't_0050'),
  'P2_tEW_S3_change': d_(K_(1, 1, 1, 'v2', 'EXFIN', 10), K_(0, 1, 1, 'v2', 'EXFIN', 10), 'S3', 't_EW'),
  'P3_D_lo_P1seg_pp': d_(K_(1, 1, 1, 'lo', 'EXFIN', 10), K_(1, 1, 1, 'v2', 'EXFIN', 10), 'P1'),
  'P3_D_hi_P1seg_pp': d_(K_(1, 1, 1, 'hi', 'EXFIN', 10), K_(1, 1, 1, 'v2', 'EXFIN', 10), 'P1'),
  'P3_D_lo_S2_pp': d_(K_(1, 1, 1, 'lo', 'EXFIN', 10), K_(1, 1, 1, 'v2', 'EXFIN', 10), 'S2'),
  'P3_D_lo_S1_pp': d_(K_(1, 1, 1, 'lo', 'EXFIN', 10), K_(1, 1, 1, 'v2', 'EXFIN', 10), 'S1'),
  'P3_D_hi_S2_pp': d_(K_(1, 1, 1, 'hi', 'EXFIN', 10), K_(1, 1, 1, 'v2', 'EXFIN', 10), 'S2'),
  'P3_D_hi_S1_pp': d_(K_(1, 1, 1, 'hi', 'EXFIN', 10), K_(1, 1, 1, 'v2', 'EXFIN', 10), 'S1'),
  'P3_none_vs_v2_S3_pp': d_(K_(1, 1, 1, 'none', 'EXFIN', 10), K_(1, 1, 1, 'v2', 'EXFIN', 10), 'S3'),
  'P4_V3_ALL': g(V3('EXFIN', 10)), 'P4_V3_t': g(V3('EXFIN', 10), 'ALL', 't_0050')}
p('\n## 預測檢查', json.dumps({k: round(float(v), 3) for k, v in pred.items()}, ensure_ascii=False))

# ---------- 判定與 SOP 預期表 ----------
tV3 = g(V3('EXFIN', 10), 'ALL', 't_0050')
tier = 1 if tV3 >= 2.3 else (2 if tV3 >= 2.0 else 3)
p('\n## 判定：V3 排除金融 n10 ALL t', round(tV3, 3), '→ 第', tier, '級')
EXP = {}
p('\n## SOP 預期表（預期年化＝0050＋0.5×(V3−0050)）')
for u in ('EXFIN', 'ALL'):
    for n in (10, 30):
        for c in ('normal', 'stress'):
            s = RUNS[V3(u, n, c)]['seg']
            for seg in ('ALL', 'P1', 'S3', 'S4'):
                e = s[seg]['E50_CAGR']; b = s[seg]['CAGR']
                EXP[f'{u}|n{n}|{c}|{seg}'] = {'backtest': b, 'E50': e, 'excess': b - e, 'expected': e + 0.5 * (b - e), 'maxDD': s[seg]['maxDD'], 't': s[seg]['t_0050']}
            p(f'{u} n{n} {c}:', ' ; '.join(f"{seg} 回測 {EXP[f'{u}|n{n}|{c}|{seg}']['backtest']:.1f} 0050 {EXP[f'{u}|n{n}|{c}|{seg}']['E50']:.1f} 預期 {EXP[f'{u}|n{n}|{c}|{seg}']['expected']:.1f} 回撤 {EXP[f'{u}|n{n}|{c}|{seg}']['maxDD']:.0f}" for seg in ('ALL', 'P1', 'S3', 'S4')))
# 保守版預期（期望拖累界限）
bk = f'P1|R1|D{DW}|EXFIN|DRAG|n10|normal'; s = RUNS[bk]['seg']
p('保守（V3＋期望拖累）EXFIN n10:', ' ; '.join(f"{seg} {s[seg]['CAGR']:.1f} 預期 {s[seg]['E50_CAGR'] + 0.5 * (s[seg]['CAGR'] - s[seg]['E50_CAGR']):.1f}" for seg in ('ALL', 'P1', 'S3', 'S4')))
bk = f'P1|R1|D{DW}|EXFIN|B10|n10|normal'; s = RUNS[bk]['seg']
p('最壞（V3＋B10）EXFIN n10:', ' ; '.join(f"{seg} {s[seg]['CAGR']:.1f} t {s[seg]['t_0050']:.2f} 預期 {s[seg]['E50_CAGR'] + 0.5 * (s[seg]['CAGR'] - s[seg]['E50_CAGR']):.1f}" for seg in ('ALL', 'P1', 'S3', 'S4')))

RES['RESULTS'] = {'D_worse': DW, 'V3_key': V3('EXFIN', 10), 'tier': tier, 'path': PATH, 'effects': EFF, 'bounds': BND, 'decomp': dec, 'yearly': YEARLY,
                  'pred': pred, 'expectation': EXP, 'list_diff_n10': {'n_signal_days': len(diff), 'n_slots': int(nslot)},
                  'validate_vs_main_v2': R0['extra'].get('validate_vs_main_v2'),
                  'main_table': {f'{nm}|{u}|n{n}|{c}': RUNS[K_(*k, u, n, c)]['seg'] for nm, k in VERS for u in ('EXFIN', 'ALL') for n in (10, 30) for c in ('normal', 'stress')}}
(HERE / 'results.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
(HERE / 'summary.txt').write_text('\n'.join(L), encoding='utf-8')
print('\n'.join(L))

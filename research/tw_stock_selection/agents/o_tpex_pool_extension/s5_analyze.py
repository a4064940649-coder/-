#!/usr/bin/env python3
"""第 4～5 步：樣本外統計、G1～G8、預測 P1～P7、第 5 步整合比較、SPA／DSR、穩健性。只讀 out/ 的淨值（PREREG 後一次模擬的結果）。
輸出 results.json 的 OOS、GATES、PRED、STEP5、SPA、ROBUST、EXPECT。用法：python3 -I -B s5_analyze.py"""
import sys, json, hashlib, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RES = json.loads((HERE / 'results.json').read_text())
assert 'PREREG' in RES and hashlib.sha256((HERE / 'frozen_spec.txt').read_bytes()).hexdigest() == RES['PREREG']['spec_sha256']
import otlib as L, dlib as DL
D = L.D
NAV = pd.read_parquet(HERE / 'out' / 'navs_oos.parquet')
ACC = json.loads((HERE / 'out' / 'acc_oos.json').read_text())
E50 = pd.Series(D.E50, index=D.IDX)
PER = ['ALL', 'SEL', 'OOS', 'VAL', 'HOLD', 'Y19', 'P1', 'P2', 'P3']
OUT = {}


def col(name, j, cell): return NAV[f'{name}|J{j}|{cell}']


def cell_stats(name, cell, ref=None, holes=True):
    per = []
    for j in range(13):
        nav = col(name, j, cell)
        W = L.windows(nav, j) if ref is None else L.windows(nav, j, ref=col(ref, j, cell))
        r = {p: L.seg_stats(W, *L.PERIODS[p], drop_holes=holes) for p in PER}
        if ref is None:
            r['ALL_incl_holes'] = L.seg_stats(W, *L.PERIODS['ALL'], drop_holes=False)
            r['OOS_incl_holes'] = L.seg_stats(W, *L.PERIODS['OOS'], drop_holes=False)
            n = nav.dropna(); r['cagr_daily_ALL'] = L.cagr(n, n.index[0], ENDJ(j)); r['mdd'] = L.mdd(n.loc[:ENDJ(j)])
            r['cagr_0050_daily_ALL'] = L.cagr(E50, n.index[0], ENDJ(j))
        per.append(r)
    out = {}
    for p in per[0]:
        if isinstance(per[0][p], dict):
            out[p] = {k: L.jsumm([x[p].get(k) for x in per]) for k in ('ann', 'ann_b', 'ann_x', 'mean_pp', 't', 't_nw', 't_drop_best3y') if k in per[0][p]}
            out[p]['n'] = per[0][p].get('n')
        else:
            out[p] = L.jsumm([x[p] for x in per])
    return out, per


def ENDJ(j): return L.ENDS[j]


CELLS = [f'{t}|{m}|{c}' for t in ('100萬', '300萬', '1000萬') for m in ('FIX', 'GROW') for c in ('normal', 'stress')]
MAINC = '100萬|GROW|normal'


def brief(s):
    return {p: {k: (round(v['median'], 3) if isinstance(v, dict) and v.get('median') is not None else v) for k, v in s[p].items()} if isinstance(s[p], dict) and 'median' not in s[p] else s[p] for p in s}


# ---------------- 主表 ----------------
TAB = {}; PERJ = {}
for nm in ('M', 'R1', 'R2', 'BASE'):
    for c in CELLS:
        TAB[f'{nm}|{c}'], PERJ[f'{nm}|{c}'] = cell_stats(nm, c)
for nm in ('R2', 'BASE'):
    for c in ('30萬|GROW|normal', '50萬|GROW|normal'):
        TAB[f'{nm}|{c}'], PERJ[f'{nm}|{c}'] = cell_stats(nm, c)
PAIR = {}
for nm in ('M', 'R1', 'R2'):
    for c in CELLS:
        PAIR[f'{nm}-BASE|{c}'], _ = cell_stats(nm, c, ref='BASE')
OUT['TABLE'] = TAB; OUT['PAIR_vs_BASE'] = PAIR

# ---------------- 敏感度 ----------------
SENS = {}
for tag in ('sens_d05', 'sens_d00', 'sens_oa05'):
    f = HERE / 'out' / f'navs_{tag}.parquet'
    if not f.exists(): continue
    S = pd.read_parquet(f)
    for nm in ('M', 'BASE', 'R2'):
        per = []
        for j in range(13):
            W = L.windows(S[f'{nm}|J{j}|{MAINC}'], j)
            per.append({p: L.seg_stats(W, *L.PERIODS[p]) for p in ('ALL', 'OOS', 'Y19')})
        SENS[f'{tag}|{nm}'] = {p: {k: L.jsumm([x[p].get(k) for x in per]) for k in ('ann_x', 't')} for p in ('ALL', 'OOS', 'Y19')}
OUT['SENS'] = SENS

# ---------------- 鄰域 ----------------
NB = {}
for k in RES['PREREG']['neighbors']:
    s, _ = cell_stats('NB_' + k, MAINC); NB[k] = {p: {kk: s[p][kk] for kk in ('ann_x', 't')} for p in ('ALL', 'SEL', 'OOS')}
OUT['NEIGHBORS'] = NB
ITV = {}
for c in NAV.columns:
    if c.startswith('IT_') and c.endswith(MAINC) and '|J0|' in c:
        nm = c.split('|')[0]; s, _ = cell_stats(nm, MAINC); ITV[nm] = {p: {kk: s[p][kk] for kk in ('ann_x', 't')} for p in ('ALL', 'SEL', 'OOS')}
OUT['OTHER_VARIANTS_FULL'] = ITV


# ---------------- G1～G7 ----------------
def g12(name, cell, pref=''):
    s = TAB[f'{name}|{cell}']
    g1 = dict(ALL_ann_x_median=s['ALL']['ann_x']['median'], ALL_ann_x_p10=s['ALL']['ann_x']['p10'], OOS_ann_x_median=s['OOS']['ann_x']['median'], OOS_ann_x_p10=s['OOS']['ann_x']['p10'])
    g1['pass'] = bool(all(v is not None and v > 0 for v in g1.values()))
    g2 = dict(ALL_t_median=s['ALL']['t']['median'], Y19_t_median=s['Y19']['t']['median'])
    g2['pass'] = bool(g2['ALL_t_median'] >= 3.0 and g2['Y19_t_median'] >= 2.0)
    return g1, g2


G = {}
for nm in ('M', 'R1', 'R2', 'BASE'):
    g1, g2 = g12(nm, MAINC); g1s, _ = g12(nm, '100萬|GROW|stress')
    s = TAB[f'{nm}|{MAINC}']
    g4 = dict(P1=s['P1']['ann_x']['median'], P2=s['P2']['ann_x']['median'], P3=s['P3']['ann_x']['median'], t_drop_best3y=s['ALL']['t_drop_best3y']['median'])
    g4['segments_pass'] = bool(all(g4[k] is not None and g4[k] > 0 for k in ('P1', 'P2', 'P3'))); g4['drop3_pass'] = bool(g4['t_drop_best3y'] >= 2.0)
    g7 = {}
    for t in ('100萬', '300萬', '1000萬'):
        a, b = g12(nm, f'{t}|GROW|normal'); g7[t] = dict(G1=a['pass'], G2=b['pass'], ALL_ann_x=a['ALL_ann_x_median'], ALL_t=b['ALL_t_median'], Y19_t=b['Y19_t_median'])
    G[nm] = dict(G1=g1, G1_stress=g1s, G2=g2, G4=g4, G7=g7)
# 鄰域不翻號（主版本）
sgn = np.sign(TAB[f'M|{MAINC}']['ALL']['ann_x']['median'])
G['M']['G4']['neighbors_same_sign'] = bool(all(np.sign(v['ALL']['ann_x']['median']) == sgn for v in NB.values()))
G['M']['G4']['pass'] = bool(G['M']['G4']['segments_pass'] and G['M']['G4']['drop3_pass'] and G['M']['G4']['neighbors_same_sign'])
G['M']['G5'] = dict(sens={k: v['ALL']['ann_x']['median'] for k, v in SENS.items() if k.endswith('|M')},
                    G1_not_flipped=bool(all((v['ALL']['ann_x']['median'] > 0) == G['M']['G1']['pass'] for k, v in SENS.items() if k.endswith('|M'))),
                    incl_holes_ALL_ann_x=TAB[f'M|{MAINC}']['ALL_incl_holes']['ann_x']['median'], incl_holes_ALL_t=TAB[f'M|{MAINC}']['ALL_incl_holes']['t']['median'])
OUT['GATES'] = G

# ---------------- 第 5 步① 合併 vs 上市 ----------------
st5 = {}
for c in CELLS:
    p = PAIR[f'M-BASE|{c}']
    st5[c] = dict(SEL_mean_pp=p['SEL']['mean_pp']['median'], SEL_t=p['SEL']['t']['median'], OOS_mean_pp=p['OOS']['mean_pp']['median'], OOS_t=p['OOS']['t']['median'],
                  VAL_t=p['VAL']['t']['median'], HOLD_t=p['HOLD']['t']['median'], ALL_t=p['ALL']['t']['median'], ALL_mean_pp=p['ALL']['mean_pp']['median'])
st5['pass_100萬GROW'] = bool(st5[MAINC]['OOS_t'] >= 1.5 and np.sign(st5[MAINC]['SEL_mean_pp']) == np.sign(st5[MAINC]['OOS_mean_pp']))
OUT['STEP5_1'] = st5

# ---------------- 預測 ----------------
PRED = {}
PRED['P1'] = dict(explore=json.loads((HERE / 'explore.json').read_text())['E1_P1_liquidity_matched'])
PRED['P3'] = dict(SEL_mean_pp=st5[MAINC]['SEL_mean_pp'], OOS_mean_pp=st5[MAINC]['OOS_mean_pp'], pass_=bool(st5[MAINC]['SEL_mean_pp'] > 0 and st5[MAINC]['OOS_mean_pp'] > 0))
PRED['P4'] = {t: dict(ALL_mean_pp=PAIR[f'M-BASE|{t}|GROW|normal']['ALL']['mean_pp']['median'], OOS_mean_pp=PAIR[f'M-BASE|{t}|GROW|normal']['OOS']['mean_pp']['median']) for t in ('100萬', '300萬', '1000萬')}
a = [PRED['P4'][t]['ALL_mean_pp'] for t in ('100萬', '300萬', '1000萬')]
PRED['P4']['monotone_decreasing'] = bool(a[0] > a[1] > a[2]); PRED['P4']['100萬_pair_gt0'] = bool(a[0] > 0)
# P5：2019 年後 j0～j4 減 j8～j12 的超額（ann_x），上櫃 R2 vs 上市 BASE
def jgap(nm, p):
    v = [PERJ[f'{nm}|{MAINC}'][j][p].get('ann_x') for j in range(13)]
    return float(np.mean(v[:5]) - np.mean(v[8:])), v
for p in ('SEL', 'Y19'):
    g2_, v2 = jgap('R2', p); gb, vb = jgap('BASE', p)
    PRED.setdefault('P5', {})[p] = dict(TPEX_j0_4_minus_j8_12=g2_, TWSE_j0_4_minus_j8_12=gb, TPEX_by_j=v2, TWSE_by_j=vb,
                                       slope_TPEX=float(np.polyfit(range(13), v2, 1)[0]), slope_TWSE=float(np.polyfit(range(13), vb, 1)[0]))
PRED['P5']['note'] = 'P5 原文：2019 年後上櫃 j0～j4 優勢衰減應慢於上市；以 2019 年後與挑選期 j 剖面的「前段減後段」比較'

# P6：驗證期處置事件
def ev_period(a, b):
    ev = DL.events(('TWSE', 'TPEX'), False)
    rows = []
    for r in ev.itertuples():
        if not (pd.Timestamp(a) <= D.IDX[r.i0] <= pd.Timestamp(b)): continue
        j = r.j; i0 = r.i0
        if not (D.TR[i0, j] and D.TR[r.i1, j]): continue
        if not DL.BULL[i0] or (np.isfinite(D.R[i0, j]) and D.R[i0, j] >= 0.09): continue
        v60 = D.V60p[i0, j]
        if not (np.isfinite(v60) and v60 >= 1e7 and D.CF[i0 - 1, j] >= 10): continue
        k = r.i1
        while k < len(D.IDX) - 1 and np.isfinite(D.R[k, j]) and D.R[k, j] <= -0.09: k += 1
        rows.append(dict(start=D.IDX[i0], mkt=r.mkt, second=r.second, raw=DL.AIf[k, j] / DL.AIf[i0, j] - 1))
    return pd.DataFrame(rows)
def evs(df, cost=0.008):
    x = df.raw - cost; g = x.groupby(df.start).mean(); xs = np.sort(x.to_numpy()); tr = xs[:int(len(xs) * 0.95)]
    return dict(n=int(len(x)), mean_pct=float(x.mean() * 100), median_pct=float(np.median(x) * 100), t_cluster=L.tstat(g), trim95_mean_pct=float(tr.mean() * 100), trim95_median_pct=float(np.median(tr) * 100))
P6 = {}
for lab, (a_, b_) in (('SEL_2021_10_2023_12', ('2021-10-01', '2023-12-31')), ('VAL_2024_2026', ('2024-01-01', '2026-10-31'))):
    df = ev_period(a_, b_); o = {}
    for m in ('TWSE', 'TPEX'):
        s_ = df[df.mkt == m]
        if len(s_) >= 5:
            o[m] = evs(s_)
            for sec in (False, True):
                s2 = s_[s_.second == sec]
                if len(s2) >= 5: o[f'{m}|{"repeat" if sec else "first"}'] = evs(s2)
    mm = df.assign(m=df.start.dt.to_period('M'), x=df.raw - 0.008).groupby(['m', 'mkt']).x.mean().unstack()
    if 'TPEX' in mm and 'TWSE' in mm:
        dd = (mm.TPEX - mm.TWSE).dropna(); o['TPEX_minus_TWSE_paired'] = dict(mean_pct=float(dd.mean() * 100), t=L.tstat(dd), n_months=int(len(dd)))
    P6[lab] = o
PRED['P6'] = P6


# P7：衛星
def win_mix(j, sat, w, cell):
    nav = col(sat, j, cell); W = L.windows(nav, j)
    R = (1 - w) * W.b + w * W.s
    # 再平衡成本：窗末漂移後回到目標比例的換手 × 0.15%
    drift = np.abs(w * (1 + W.s) / (1 + R) - w)
    R = R - 2 * drift * 0.0015
    return pd.DataFrame({'s': R, 'b': W.b, 'x': R - W.b}, index=W.index)
SATR = {}
for sat in ('R2', 'BASE'):
    for w, cell in ((0.3, '30萬|GROW|normal'), (0.5, '50萬|GROW|normal')):
        per = []
        for j in range(13):
            Wm = win_mix(j, sat, w, cell)
            r = {p: L.seg_stats(Wm, *L.PERIODS[p]) for p in ('ALL', 'SEL', 'OOS', 'Y19')}
            r['TE_ALL'] = float(Wm.loc[L.PERIODS['ALL'][0]:].x.std() * math.sqrt(12) * 100)
            per.append(r)
        SATR[f'{sat}_{int(w * 100)}'] = {p: {k: L.jsumm([x[p].get(k) for x in per]) for k in ('ann', 'ann_x', 't')} for p in ('ALL', 'SEL', 'OOS', 'Y19')}
        SATR[f'{sat}_{int(w * 100)}']['TE_ALL'] = L.jsumm([x['TE_ALL'] for x in per])
te_main = L.jsumm([float(L.windows(col('M', j, MAINC), j).x.std() * math.sqrt(12) * 100) for j in range(13)])
cors = []
for j in range(13):
    a_ = L.windows(col('R2', j, '30萬|GROW|normal'), j).x; b_ = L.windows(col('BASE', j, MAINC), j).x
    cors.append(float(a_.corr(b_)))
PRED['P7'] = dict(corr_R2_vs_BASE_excess=L.jsumm(cors), TE_main_100pct=te_main, sat=SATR,
                  pass_=bool(np.median(cors) < 0.7 and SATR['R2_30']['TE_ALL']['median'] < te_main['median'] and SATR['R2_30']['ALL']['ann_x']['median'] > 0))
OUT['PRED'] = PRED

# ---------------- 第 5 步② 主策略＋處置 ----------------
st52 = {}
for dk, mk in (('d0_TWSE', ('TWSE',)), ('d1_ALL', ('TWSE', 'TPEX'))):
    for lab, (a_, b_) in (('SEL', ('2021-10-01', '2023-12-29')), ('VAL', ('2024-01-02', '2026-10-12'))):
        nav, tr, sk = DL.sleeve(DL.events(mk, False), pd.Timestamp(a_), pd.Timestamp(b_))
        mret = nav.resample('ME').last().pct_change().dropna(); rb = E50.reindex(nav.index).resample('ME').last().pct_change().dropna()
        x = (mret - rb).dropna()
        st52[f'{dk}|{lab}'] = dict(ann=float(nav.iloc[-1] ** (365.25 / (nav.index[-1] - nav.index[0]).days) - 1) * 100,
                                   ann_0050=float((E50[nav.index[-1]] / E50[nav.index[0]]) ** (365.25 / (nav.index[-1] - nav.index[0]).days) - 1) * 100,
                                   t_month=L.tstat(x), mdd=L.mdd(nav), n_trades=int(len(tr)), trade_mean_pct=float(tr.ret.mean() * 100) if len(tr) else None,
                                   tpex_trades=int((tr.mkt == 'TPEX').sum()) if len(tr) else 0, skipped=sk)
        st52[f'{dk}|{lab}|nav'] = nav
# 80/20：主版本 j 平均日報酬 × 0.8 ＋ 處置腿 × 0.2（每月初再平衡）
def javg(name, cell):
    r = pd.concat([col(name, j, cell).pct_change() for j in range(13)], axis=1).mean(axis=1)
    return (1 + r.fillna(0)).cumprod()
MJ = javg('M', MAINC); BJ = javg('BASE', MAINC)
for lab, (a_, b_) in (('SEL', ('2021-10-01', '2023-12-29')), ('VAL', ('2024-01-02', '2026-10-12'))):
    for mk, nv in (('M', MJ), ('BASE', BJ)):
        for dk in ('d0_TWSE', 'd1_ALL'):
            sl = st52[f'{dk}|{lab}|nav']
            m1 = nv.loc[a_:b_].resample('ME').last().pct_change().dropna(); m2 = sl.resample('ME').last().pct_change().dropna()
            rb = E50.loc[a_:b_].resample('ME').last().pct_change().dropna()
            r = (0.8 * m1 + 0.2 * m2).dropna(); x = (r - rb).dropna()
            st52[f'{mk}80+{dk}20|{lab}'] = dict(ann=float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100, ann_0050=float(np.prod(1 + rb.reindex(r.index)) ** (12 / len(r)) - 1) * 100, t_month=L.tstat(x))
for k in [k for k in st52 if k.endswith('|nav')]: del st52[k]
OUT['STEP5_2'] = st52

# ---------------- SPA／DSR ----------------
sys.path.insert(0, str(L.AG / 'o_main_spa_breaks'))
import fam_common as FC, spa_lib as SP
Fx = pd.read_parquet(FC.DATA / 'family_excess.parquet'); MM = pd.read_csv(FC.DATA / 'family_members.csv')
mem = MM[MM.F_long & MM.cov_W_long].member_id.tolist()
m = FC.wmask(Fx.index, *FC.WINDOWS['W_long'])
e50w = FC.win_ret(FC.e50_nav())
mine = {}
for nm in ['M', 'R1', 'R2', 'BASE'] + sorted({c.split('|')[0] for c in NAV.columns if c.startswith('IT_') or c.startswith('NB_')}):
    nv = javg(nm, MAINC); mine[nm] = (FC.win_ret(nv) - e50w)
MX = pd.DataFrame(mine).reindex(Fx.index)
X_ = np.hstack([Fx.loc[m, mem].fillna(0.0).to_numpy(), MX.loc[m].fillna(0.0).to_numpy()])
names = mem + ['OTPX:' + k for k in MX.columns]
foc = names.index('OTPX:M')
n = int(m.sum()); Wc = SP.sb_counts(n, 6, 10000, 20261102)
spa = SP.mcp_tests(X_, Wc, 6, 'pr', focus=foc)
sr_all = X_.mean(0) / X_.std(0, ddof=1)
xm = X_[:, foc]
spa_o = {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in spa.items() if k not in ('top_idx', 'top_t')}
spa_o['argmax_member'] = names[spa['argmax']]; spa_o['top'] = [(names[i], t) for i, t in zip(spa['top_idx'], spa['top_t'])]
spa_o['N_family'] = len(names); spa_o['n_windows'] = n
spa_o['DSR_raw'] = SP.dsr(xm, sr_all, len(names)); et = SP.eff_trials(X_); spa_o['eff_trials'] = et
spa_o['DSR_LiJi'] = SP.dsr(xm, sr_all, et['M_eff_LiJi'])
spa_o['focus_t_grid'] = FC.tstat(xm)
for k in ('argmax',): spa_o.pop(k, None)
OUT['SPA'] = spa_o

# ---------------- 穩健性：逐年、產業、去掉最大貢獻股 ----------------
yr = {}
for nm in ('M', 'BASE', 'R2'):
    per = []
    for j in range(13):
        W = L.windows(col(nm, j, MAINC), j)
        hm = set(L.hole_months()); W = W[[str(pd.Period(d, 'M')) not in hm for d in W.index]]
        per.append(W.groupby(W.index.year).apply(lambda g: (np.prod(1 + g.s) - np.prod(1 + g.b)) * 100))
    yr[nm] = pd.concat(per, axis=1).median(axis=1).round(2).to_dict()
ind = pd.read_csv(L.EXT / 'stock_list.csv', dtype=str).set_index('code')['industry'].to_dict()
contrib = {}; indc = {}
for j in range(13):
    pl = L.gen_plan(RES['PREREG']['main_spec'], j)
    for k in range(len(pl) - 1):
        e1, pk, sg = pl[k]; e2 = pl[k + 1][0]
        if D.IDX[sg] < pd.Timestamp('2006-04-01'): continue
        AO = D.AI[e1 - 1, pk] * D.O[e1, pk] / D.PC[e1, pk]; BO = DL.AIf[e2 - 1, pk] * np.where(np.isfinite(D.O[e2, pk]) & np.isfinite(D.PC[e2, pk]), D.O[e2, pk] / D.PC[e2, pk], 1.0)
        r = BO / AO - 1
        for jj, rr in zip(pk, r):
            if np.isfinite(rr):
                c = D.COLS[jj]; contrib[c] = contrib.get(c, 0) + rr / len(pk) / 13
                ii = ind.get(c, '未知（下市）'); indc[ii] = indc.get(ii, 0) + rr / len(pk) / 13
top = sorted(contrib.items(), key=lambda x: -x[1])[:10]
tot = sum(contrib.values())
OUT['ROBUST'] = dict(yearly_excess_jmedian_pp=yr, top10_stock_contrib_sum_ideal=top, total_ideal_sum=tot,
                     share_top1=float(top[0][1] / tot), share_top10=float(sum(t[1] for t in top) / tot),
                     industry_contrib_ideal=sorted(indc.items(), key=lambda x: -x[1])[:12],
                     note='貢獻＝零衝擊開盤到開盤窗報酬 ÷ 檔數 ÷ 13 個 j 的總和（毛、未扣成本），只用於相對大小')
# 去掉最大貢獻股：該股入選的窗，以池內其餘 9 檔等權報酬近似（毛）→ 對 t 的影響以毛窗超額估計
OUT['TPEX_SHARE_PICKS'] = {k: v for k, v in ACC['tpex_share'].items() if k.split('|')[0] in ('M', 'R1', 'BASE', 'R2')}
OUT['PARTICIPATION'] = {k: dict(part_median=v['part_median'], part_p90=v['part_p90'], n_cap_hit=v['n_cap_hit'], impact=v['impact'], tax=v['tax'])
                        for k, v in ACC['acc'].items() if '|J0|' in k and ('normal' in k)}

# ---------------- 實盤預期（打五折）----------------
EXP = {}
for nm in ('M', 'BASE', 'R2'):
    for t in ('100萬', '300萬', '1000萬'):
        for md in ('FIX', 'GROW'):
            s = TAB[f'{nm}|{t}|{md}|normal']; ss = TAB[f'{nm}|{t}|{md}|stress']
            EXP[f'{nm}|{t}|{md}'] = dict(ALL_ann_median=s['ALL']['ann']['median'], ALL_0050=s['ALL']['ann_b']['median'], ALL_x_median=s['ALL']['ann_x']['median'], ALL_x_p10=s['ALL']['ann_x']['p10'],
                                         OOS_x_median=s['OOS']['ann_x']['median'], OOS_x_p10=s['OOS']['ann_x']['p10'], HOLD_x_median=s['HOLD']['ann_x']['median'],
                                         expect_x_half=0.5 * s['ALL']['ann_x']['median'], stress_ALL_x_median=ss['ALL']['ann_x']['median'], mdd_median=s['mdd']['median'])
OUT['EXPECT'] = EXP
RES.update(OUT)
(HERE / 'results.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=float))
print(json.dumps({k: OUT['GATES'][k] for k in ('M', 'BASE')}, ensure_ascii=False, default=float)[:3000])
print('STEP5_1', json.dumps(st5[MAINC], default=float)); print('SPA', {k: spa_o[k] for k in ('p_SPA_c', 'p_SPA_c_focus', 'focus_t', 'focus_rank', 'N_family')}, spa_o['DSR_raw']['DSR'])

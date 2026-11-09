#!/usr/bin/env python3
"""凍結前：煙霧測試（只重現已公開數字）＋名單層探索（不算任何新變體的報酬、t 或名單報酬差）。
煙霧：C0 對 main_v2/navs_gapfix 逐日差 0；U13 opt cons_main 重現 24.92%／t 2.064；ALLPIT opt cons_main 重現 21.28%。
探索（只看名單）：REAL（補回 B1）放回槽數 ≤ U_opt（P2 機械性質）；捨入敏感度（新聞水準 ±0.005 億）；放回槽距下市月數分布；
放回股的營收百分位（真實值）分布。python3 -I -B s7_smoke_explore.py <B_ver>  → smoke.json、explore.json"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
BV = sys.argv[1] if len(sys.argv) > 1 else 'B1'
T0 = time.time()
sys.path.insert(0, str(HERE)); import o18lib as O; sys.path.pop(0)
P, V, K, COLS = O.P, O.V, O.K, O.COLS
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
SM = {}
# ---------------- 煙霧 1：C0
b0 = O.build(('base', None, None, False, None))
OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')
for c in ('normal', 'stress'):
    nav, _ = O.sim(b0, c, 'freeze'); ref = OFF[f'EXFIN|n10|{c}'].dropna()
    SM[f'C0_{c}_maxabs_vs_main_v2'] = float(np.nanmax(np.abs(nav.reindex(ref.index).to_numpy() - ref.to_numpy())))
    if c == 'normal': NAV0 = nav
assert max(SM['C0_normal_maxabs_vs_main_v2'], SM['C0_stress_maxabs_vs_main_v2']) < 1e-12, SM
W0 = O.win(NAV0, b0['sd'])
def full(W):
    s = O.stats_pair(W, W0); return dict(ann=s['ALL']['ann'], t=s['ALL']['t_vs_0050'], seg2=s['SEG2']['gex'])
SM['C0'] = full(W0); log('C0', SM['C0'])
# ---------------- 煙霧 2：U13 opt；煙霧 3：ALLPIT opt
bu = O.build(('opt', '2013-01-01', None, False, None)); nav, _ = O.sim(bu, 'normal', 'cons_main'); SM['U13_opt_cons_main'] = full(O.win(nav, bu['sd']))
ba = O.build(('opt', '2006-01-01', 'UB', True, None)); nav, _ = O.sim(ba, 'normal', 'cons_main'); SM['ALLPIT_opt_cons_main'] = full(O.win(nav, ba['sd']))
SM['checks'] = dict(U13_ann_2492=abs(SM['U13_opt_cons_main']['ann'] - 24.92) < 0.006, U13_t_2064=abs(SM['U13_opt_cons_main']['t'] - 2.064) < 0.0006,
                    ALLPIT_ann_2128=abs(SM['ALLPIT_opt_cons_main']['ann'] - 21.28) < 0.006)
log('smoke', SM)
assert all(SM['checks'].values()), SM['checks']
SM['utc'] = pd.Timestamp.now('UTC').isoformat()
P.jdump(SM, HERE / 'smoke.json')

# ---------------- 探索（名單）
def ins_slots(b, lo='2013-01-01', key='aff'):
    out = []
    for p in b['plan']:
        d = p[3]
        if d < pd.Timestamp(lo): continue
        A_ = set(b[key].get(d, []))
        out += [(str(d.date()), COLS[j]) for j in p[1] if COLS[j] in A_]
    return out
EX = {'note': '只看名單；沒有計算任何變體的報酬、t 或名單報酬差。', 'B_ver': BV}
uopt = ins_slots(bu)
EX['U13_opt_inserted'] = dict(n=len(uopt), n1318=sum(1 for s, _ in uopt if s <= '2018-12-31'))
for fill in ('opt', 'p90', 'carry'):
    for lo, tag in (('2013-01-01', 'U13'), ('2006-01-01', 'UFULL')):
        b = O.build((fill, lo, None, False, BV))
        ins_unc = ins_slots(b, lo, 'aff')                # 仍未覆蓋、被 fill 放回者
        ins_bm = ins_slots(b, lo, 'base_missing')         # 原本缺營收（補回或放回）而入選者
        ins_real = sorted(set(ins_bm) - set(ins_unc))     # 靠真實補回值入選者
        EX[f'{tag}_REAL_{fill}'] = dict(inserted_total=len(ins_bm), inserted_total_2013on=sum(1 for s, _ in ins_bm if s >= '2013-01-01'),
                                       inserted_total_1318=sum(1 for s, _ in ins_bm if '2013-01-01' <= s <= '2018-12-31'),
                                       by_real_backfill=len(ins_real), by_fill=len(ins_unc), real_slots=ins_real[:80])
        if tag == 'U13':
            EX[f'{tag}_REAL_{fill}']['P2_le_Uopt'] = EX[f'{tag}_REAL_{fill}']['inserted_total_2013on'] <= 251 and EX[f'{tag}_REAL_{fill}']['inserted_total_1318'] <= 227
        log(tag, fill, {k: v for k, v in EX[f'{tag}_REAL_{fill}'].items() if k != 'real_slots'})
assert EX['U13_REAL_opt']['P2_le_Uopt'], 'P2 違反：程式錯誤'
# 純「補回值」版（未覆蓋者不放回＝只有真實值的股票加入母體），看真實補回股會不會入選
b = O.build(('base', None, None, False, BV))
bm_real = []
for p in b['plan']:
    d = p[3]
    y0 = K.yoy3_for_month(P.Y5 if d >= V.SWITCH else P.YL, pd.Period(d, 'M'))
    bm_real += [(str(d.date()), COLS[j]) for j in p[1] if not np.isfinite(y0[j])]
EX['REAL_only_no_fill'] = dict(n_inserted=len(bm_real), slots=bm_real)
# 距下市月數（REAL opt U13 放回槽，含真實補回與 fill）
b = O.build(('opt', '2013-01-01', None, False, BV)); ins_bm = ins_slots(b, '2013-01-01', 'base_missing')
def mdl(c, s):
    dd = P.delist_date(c)
    return None if pd.isna(dd) else round((dd - pd.Timestamp(s)).days / 30.44, 1)
x = [mdl(c, s) for s, c in ins_bm]
EX['U13_REAL_opt_months_to_delist'] = dict(n=len(x), n_survivor=sum(v is None for v in x), le6=sum(v is not None and v <= 6 for v in x),
                                         m6_12=sum(v is not None and 6 < v <= 12 for v in x), gt12=sum(v is not None and v > 12 for v in x))
# 真實補回股（池內）的營收百分位分布（若 A_opt 放 1.0，真實值多半在哪）
pr = []
for p in b['plan']:
    d = p[3]
    if d < pd.Timestamp('2013-01-01'): continue
    cs = b_cs = None
SPrank = []
S = P.VU('base', YLx=O._pt(BV)[0], Y5x=O._pt(BV)[1])
for d in S.sig_days:
    if d < pd.Timestamp('2013-01-01') or d > pd.Timestamp('2021-02-28'): continue
    cs = S.cross_section(d); el = cs['el']
    y0 = K.yoy3_for_month(P.YL, pd.Period(d, 'M')); y0 = np.where(V.FINMASK, np.nan, y0)
    rp = pd.Series(np.where(el, cs['y3'], np.nan)).rank(pct=True).to_numpy()
    for j in np.where(el & ~np.isfinite(y0))[0]: SPrank.append(float(rp[j]))
SPrank = np.array(SPrank)
EX['real_backfilled_pool_rev_pct'] = dict(n=int(len(SPrank)), median=float(np.median(SPrank)) if len(SPrank) else None,
                                          share_ge_0p9=float((SPrank >= 0.9).mean()) if len(SPrank) else None,
                                          deciles=np.percentile(SPrank, [10, 25, 50, 75, 90]).round(3).tolist() if len(SPrank) else None)
log('pct', EX['real_backfilled_pool_rev_pct'])
# ---------------- 捨入敏感度（新聞水準 ±0.005 億＝±500 千元；只看名單）
RB = pd.read_csv(HERE / f'data/rev_backfill_{BV}.csv', dtype={'code': str})
nl = RB[RB.rev_src == 'N.level']
EX['rounding'] = dict(n_news_level_cells=int(len(nl)), n_yoy_cells_using_news=int(RB.yoy_src.fillna('').str.contains('N.level').sum()))
if len(nl):
    base_list = {p[3]: set(COLS[j] for j in p[1]) for p in b['plan']}
    LV = {(c, y): v for c, y, v in zip(RB.code, RB.ym, RB.rev_k) if isinstance(v, float) and np.isfinite(v)}
    sys.path.insert(0, str(HERE)); import bflib as BL; sys.path.pop(0)
    TL = BL.tw_levels()
    def lvl(c, p, dlt):
        v = LV.get((c, str(p)))
        if v is None:
            try: v = TL.at[p, c]
            except KeyError: v = np.nan
        elif ((RB.code == c) & (RB.ym == str(p)) & (RB.rev_src == 'N.level')).any(): v = v + dlt
        return v
    res = {}
    for tag, dlt in (('minus', -500.0), ('plus', 500.0)):
        R2 = RB.copy()
        for i, r in R2[R2.yoy_src.fillna('').str.contains('N.level')].iterrows():
            p = pd.Period(r.ym, 'M'); a, z = lvl(r.code, p, dlt), lvl(r.code, p - 12, dlt)
            R2.at[i, 'yoy_pct'] = (a / z - 1) * 100
        R2.to_csv(HERE / f'data/rev_backfill_{BV}_round{tag}.csv', index=False)
        bb = O.build(('opt', '2013-01-01', None, False, f'{BV}_round{tag}'))
        L2 = {p[3]: set(COLS[j] for j in p[1]) for p in bb['plan']}
        res[tag] = dict(n_days_changed=int(sum(L2[d] != base_list[d] for d in base_list)), n_slots_changed=int(sum(len(base_list[d] - L2[d]) for d in base_list)))
    EX['rounding'].update(res)
log('rounding', EX['rounding'])
EX['utc'] = pd.Timestamp.now('UTC').isoformat(); EX['runtime_s'] = round(time.time() - T0, 1)
P.jdump(EX, HERE / 'explore.json')

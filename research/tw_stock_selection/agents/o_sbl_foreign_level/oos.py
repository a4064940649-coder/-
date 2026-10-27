#!/usr/bin/env python3
"""樣本外（驗證期 2024-02～2025-09＝回推持股；保留期 2025-10～2026-09＝官方 MI_QFIIS 實值；合併 2024～2026 用於判定）。

必須先有 results.json 的 PREREG，且 sbl_lib.py／oos.py／data_freeze.json 的 sha256 與登記時相同。只跑一次，不改規格。
輸出 results_oos.json、oos_navs.parquet。
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import sbl_lib as L

H = L.HERE
SELFTEST = len(sys.argv) > 1 and sys.argv[1] == 'selftest'     # 只用挑選期資料檢查程式能跑（登記前除錯用）
if not SELFTEST:
    R = json.loads((H / 'results.json').read_text(encoding='utf-8'))
    assert 'PREREG' in R, 'PREREG 不存在：不得計算樣本外'
    PR = R['PREREG']
    for f in ('sbl_lib.py', 'oos.py', 'data_freeze.json'):
        assert L.sha(H / f) == PR['code_sha256'][f], f'{f} 在登記後被改動'
else:
    PR = {'created_utc': 'selftest'}
L.check_freeze()

K, V, IDX, COLS, N = L.K, L.V, L.IDX, L.COLS, L.N
E50 = K.E50
S = L.strategy(True)
if SELFTEST:
    a, b = L.SEL
    END = [d for d in S.sig_days if d > b][0]
    WIN = {'SEL_A': (a, pd.Timestamp('2022-12-31')), 'SEL_B': (pd.Timestamp('2023-01-01'), b), 'SEL': (a, b)}
else:
    a, b = L.VAL
    END = S.end_day                                  # 2026-10-12
    WIN = {'VAL_2024-02_2025-09': L.VAL_A, 'HOLD_2025-10_2026-09': L.HOLD, 'VAL_ALL_2024_2026': L.VAL}
t_end = IDX.get_loc(END)

base = L.main_plan(S, a, b)
SDw = [p[3] for p in base] + [END]
nav_main = {c: L.run_plan(base, c, t_end)[0] for c in L.COST}
ew = L.ew_nav(base, t_end)
sl = L.sleeve_plan(S, 'lvl_size', a, b)              # B1
assert [p[3] for p in sl] == [p[3] for p in base]
nav_s = {c: L.run_plan(sl, c, t_end)[0] for c in L.COST}
c2 = L.main_plan(S, a, b, third='lvl', w3=0.2)       # C2
nav_c2 = {c: L.run_plan(c2, c, t_end)[0] for c in L.COST}
nav_c1 = {c: L.combine(nav_main[c], nav_s[c], base, 0.8) for c in L.COST}

def block(nav, lo, hi, ref):
    o = {}
    for nm, bench in [('vs_EW', ew), ('vs_0050', E50), ('vs_main', ref)]:
        o[nm] = L.summ(L.monthly_t(nav, bench, SDw, lo, hi))
    sub = [d for d in SDw if lo <= d <= hi]
    nd = [d for d in SDw if d > hi]
    end = nd[0] if nd else END
    o.update(L.cagr_dd(nav, sub[0] + pd.Timedelta(days=1), end))
    x = L.monthly_t(nav, E50, SDw, lo, hi)
    o['TE_vs_0050_ann%'] = round(float(x.std(ddof=1) * np.sqrt(12) * 100), 2)
    return o

out = {'prereg_created': PR['created_utc'], 'windows': {k: [str(v[0].date()), str(v[1].date())] for k, v in WIN.items()}, 'res': {}}
for wn, (lo, hi) in WIN.items():
    r = {}
    for c in L.COST:
        r[c] = {'main': block(nav_main[c], lo, hi, nav_main[c]), 'B1_sleeve': block(nav_s[c], lo, hi, nav_main[c]),
                'C2_third_score': block(nav_c2[c], lo, hi, nav_main[c]), 'C1_80_20': block(nav_c1[c], lo, hi, nav_main[c]),
                'EW': L.cagr_dd(ew, lo, hi), '0050': L.cagr_dd(E50, lo, hi)}
    sx = L.monthly_t(nav_s['normal'], ew, SDw, lo, hi); mx = L.monthly_t(nav_main['normal'], ew, SDw, lo, hi)
    r['corr_B1_excess_vs_main_excess(vs_EW)'] = round(float(np.corrcoef(sx, mx.reindex(sx.index))[0, 1]), 3)
    out['res'][wn] = r

# 市值組內（B1 每組 10 檔對該組等權，一般成本）
sub = {}
for gg, nm in [(0, 'small'), (1, 'mid'), (2, 'large')]:
    plg, plew = [], []
    for p in sl:
        cs = S.cross_section(p[3]); f = L.fo_features(cs['t']); g = L.terciles(f['mcap'], cs['el'])
        plg.append((p[0], np.array([j for j in p[1] if g[j] == gg], int))); plew.append((p[0], np.where(cs['el'] & (g == gg))[0]))
    ng, _ = K.simulate(plg, *L.COST['normal'], sl[0][0], t_end); ne, _ = K.simulate(plew, 0, 0, sl[0][0], t_end)
    sub[nm] = {wn: L.summ(L.monthly_t(ng, ne, SDw, lo, hi)) for wn, (lo, hi) in WIN.items()}
out['B1_within_size_vs_group_EW'] = sub

# 逐年（月超額對等權、對主策略）
yr = {}
for nm, nv, ref in [('B1_vs_EW', nav_s['normal'], ew), ('C2_vs_main', nav_c2['normal'], nav_main['normal']),
                    ('C1_vs_main', nav_c1['normal'], nav_main['normal']), ('main_vs_EW', nav_main['normal'], ew)]:
    x = L.monthly_t(nv, ref, SDw, a, b)
    yr[nm] = {int(y): L.summ(g) for y, g in x.groupby(x.index.year)}
out['yearly'] = yr

# 去掉最大貢獻股（B1，無成本近似）
con, per = V.stock_contrib(sl, t_end, a=a, b=b)
top = sorted(con, key=con.get, reverse=True)[:5]
def sleeve_ex(excl):
    xs = []
    for (d, pk, r) in per:
        rr = r.copy(); m = np.array([COLS[j] in excl for j in pk])
        if m.any() and (~m).any(): rr[m] = rr[~m].mean()
        xs.append((d, rr.mean()))
    return pd.Series(dict(xs))
ewp = {}
for (d, pk, r), p in zip(per, sl):
    e2 = [q[0] for q in sl if q[3] > d]; e2 = e2[0] if e2 else t_end
    rr = L.open_to_open(p[0], e2, p[2], to_open=bool([q for q in sl if q[3] > d]))
    ewp[d] = float(np.nanmean(rr))
ewp = pd.Series(ewp)
out['B1_drop_top_contributors'] = {'top5': [(c, round(con[c] * 100, 1)) for c in top],
                                   'gross_vs_EW_all': L.summ(sleeve_ex(set()) - ewp), 'gross_vs_EW_ex_top1': L.summ(sleeve_ex({top[0]}) - ewp),
                                   'gross_vs_EW_ex_top5': L.summ(sleeve_ex(set(top)) - ewp)}
# 分產業：B1 名單的產業分布，以及去掉電子相關（半導體、電子零組件、電腦及週邊、光電、通信網路、其他電子、電子通路、資訊服務）
SLD = V.SL['industry']
elec = {'半導體業', '電子零組件業', '電腦及週邊設備業', '光電業', '通信網路業', '其他電子業', '電子通路業', '資訊服務業'}
cnt = {}
for p in sl:
    for j in p[1]:
        ind = SLD.get(COLS[j], '？'); cnt[ind] = cnt.get(ind, 0) + 1
tot = sum(cnt.values())
out['B1_industry_share'] = {k: round(v / tot, 3) for k, v in sorted(cnt.items(), key=lambda x: -x[1])[:10]}
xs = []
for (d, pk, r) in per:
    m = np.array([SLD.get(COLS[j], '') not in elec for j in pk])
    if m.any(): xs.append((d, r[m].mean()))
out['B1_non_electronics_gross_vs_EW'] = L.summ(pd.Series(dict(xs)) - ewp)

# H3b 描述：市值組內 Q5−Q1（無成本），樣本外
rows = []
for d in [p[3] for p in base]:
    cs = S.cross_section(d); el = cs['el']; t, e = cs['t'], cs['e']
    i = SDw.index(d); nd = SDw[i + 1]
    e2 = IDX.get_loc(nd) + 1 if nd != END else IDX.get_loc(nd)
    f = L.fo_features(t); g = L.terciles(f['mcap'], el); idx = np.where(el)[0]
    r = L.open_to_open(e, e2, idx, to_open=nd != END)
    for k, j in enumerate(idx): rows.append((d, g[j], f['lvl'][j], r[k]))
D = pd.DataFrame(rows, columns=['d', 'g', 'lvl', 'r'])
spr = {}
for d, x in D.groupby('d'):
    s = []
    for gg, y in x.groupby('g'):
        y = y.dropna()
        if gg < 0 or len(y) < 10: continue
        q = pd.qcut(y.lvl.rank(method='first'), 5, labels=False); m = y.r.groupby(q).mean(); s.append(m.iloc[-1] - m.iloc[0])
    spr[d] = np.mean(s)
spr = pd.Series(spr)
out['P6_within_size_Q5minusQ1_gross'] = {wn: L.summ(spr[(spr.index >= lo) & (spr.index <= hi)]) for wn, (lo, hi) in WIN.items()}

if not SELFTEST:
  # 回推誤差和報酬是否相關（診斷）：2026-10-21 回推到 2025-10-07 的誤差 vs 同期報酬
  Rq, Sq, Fq = L.load_qfiis(); W = L.load_flows()
  bc = L.backcast(Rq, Sq, W, Rq.index.max(), L.ANCHOR - pd.Timedelta(days=1))
  err = (bc.loc[L.ANCHOR] - Rq.loc[L.ANCHOR].reindex(bc.columns)).reindex(COLS)
  t0, t1 = IDX.get_loc(L.ANCHOR), IDX.get_loc(pd.Timestamp('2026-10-21'))
  A = K.MYADJ.to_numpy()
  ret = pd.Series(A[t1] / A[t0] - 1, index=COLS)
  k = err.notna() & ret.notna()
  out['backcast_error_vs_return_spearman'] = round(float(err[k].rank().corr(ret[k].rank())), 4)
  out['backcast_error_vs_return_note'] = '回推值 − 官方值（2025-10-07，回推 1 年）和 2025-10-07→2026-10-21 報酬的 Spearman；正值代表回推高估之後會漲的股票（對 H3b 有利的偏誤）'

SUF = '_selftest' if SELFTEST else ''
pd.DataFrame({'main_n10': nav_main['normal'], 'B1': nav_s['normal'], 'C2': nav_c2['normal'], 'C1': nav_c1['normal'], 'EW': ew,
              'main_stress': nav_main['stress'], 'B1_stress': nav_s['stress'], 'C2_stress': nav_c2['stress']}).to_parquet(H / f'oos_navs{SUF}.parquet')
(H / f'results_oos{SUF}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])

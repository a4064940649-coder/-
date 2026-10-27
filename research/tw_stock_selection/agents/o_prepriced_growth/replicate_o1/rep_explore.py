#!/usr/bin/env python3
"""探索段重做：G＝AVG 營收前 1/5 且 52 週百分位 0.4～0.85。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L
from scipy.stats import spearmanr

days = L.reb_days('2006-07-01', '2013-12-31'); END = pd.Timestamp('2013-12-31')
WD = days + [END]
ADJF = pd.DataFrame(L.ADJ, index=L.IDX).ffill().to_numpy() if (len(sys.argv) < 3 or sys.argv[2] == 'ffill') else (L.ADJ if sys.argv[2] == 'nan' else L.ADJP.to_numpy())
segmode = sys.argv[1] if len(sys.argv) > 1 else 'ceil'
CS = {}
for d in days:
    cs = L.pool(d); sc, x = L.leg2_score(cs, d, 0.0, 1.0)
    el = cs['el']
    ok = el & np.isfinite(sc)
    rs = L.prank(sc, ok)
    cs.update(avg=sc, ravg=rs, p52=x['p52'], strong=ok & (rs >= 0.8))   # 前 1/5：百分位 ≥ 0.8
    CS[d] = cs

rows = []
for i, d in enumerate(days):
    cs = CS[d]; t = cs['t']; t2 = L.IDX.get_loc(WD[i + 1])
    r = ADJF[t2] / ADJF[t] - 1
    el = cs['el']; ew = np.nanmean(r[el]); ex = r - ew
    p = cs['p52']
    seg = np.clip(np.ceil(p * 10), 1, 10) if segmode == 'ceil' else np.clip(np.floor(p * 10) + 1, 1, 10)
    G = cs['strong'] & (p >= 0.4) & (p <= 0.85)
    rows.append(dict(d=d, ex=ex, seg=seg, G=G, el=el, strong=cs['strong'], p=p))

out = {}
cnt = [int(r['G'].sum()) for r in rows]
out['G_count'] = {'mean': float(np.mean(cnt)), 'min': min(cnt), 'max': max(cnt)}
out['LOWSTRONG_count'] = float(np.mean([(r['strong'] & (r['p'] < 0.4)).sum() for r in rows]))
out['HIGHSTRONG_count'] = float(np.mean([(r['strong'] & (r['p'] > 0.85)).sum() for r in rows]))
dec = np.full((len(rows), 10), np.nan)
for i, r in enumerate(rows):
    for k in range(1, 11):
        m = r['strong'] & (r['seg'] == k) & np.isfinite(r['ex'])
        if m.any(): dec[i, k - 1] = r['ex'][m].mean()
dm = np.nanmean(dec, axis=0) * 100
out['deciles'] = [round(float(x), 2) for x in dm]
out['spearman'] = float(spearmanr(np.arange(10), dm).correlation)
def grp(lo, hi):
    v = []
    for r in rows:
        m = r['strong'] & (r['seg'] >= lo) & (r['seg'] <= hi) & np.isfinite(r['ex'])
        v.append(r['ex'][m].mean() if m.any() else np.nan)
    return np.array(v)
lo_, mid_, hi_ = grp(1, 3), grp(4, 8), grp(9, 10)
out['low_mid_high'] = [float(np.nanmean(x) * 100) for x in (lo_, mid_, hi_)]
out['low_mid_high_decmean'] = [float(np.mean(dm[a:b])) for a, b in ((0, 3), (3, 8), (8, 10))]
out['t_mid_low'] = L.tstat(mid_ - lo_); out['t_mid_high'] = L.tstat(mid_ - hi_)
dl, dmid, dh = np.nanmean(dec[:, :3], 1), np.nanmean(dec[:, 3:8], 1), np.nanmean(dec[:, 8:], 1)
out['low_mid_high_decavg_mean'] = [float(np.nanmean(x) * 100) for x in (dl, dmid, dh)]; out['t_mid_low_decavg'] = L.tstat(dmid - dl); out['t_mid_high_decavg'] = L.tstat(dmid - dh)

# G 下月超額、子樣本
gx = np.array([r['ex'][r['G'] & np.isfinite(r['ex'])].mean() for r in rows])
out['G_next'] = (float(np.mean(gx) * 100), L.tstat(gx))
def sub(mask_fn, sel=None):
    v = []
    for i, r in enumerate(rows):
        if sel is not None and not sel(r['d']): continue
        m = r['G'] & mask_fn(i, r) & np.isfinite(r['ex'])
        v.append(r['ex'][m].mean() if m.any() else np.nan)
    v = np.array(v); return (round(float(np.nanmean(v) * 100), 2), round(L.tstat(v), 2))
LIQ_EL = True   # 流動性名次：在股池（el）內排；False＝在前 300 名（含金融）內排
liq = {}
for d in days:
    cs = CS[d]; v = np.where(cs['el'] if LIQ_EL else cs['top'], L.V60[cs['t']], -np.inf); o = np.argsort(-v, kind='stable')
    rk = np.empty(L.N); rk[o] = np.arange(1, L.N + 1); liq[d] = rk
ELSET = ('半導體業', '電子零組件業', '電腦及週邊設備業', '光電業', '通信網路業', '其他電子業', '電子通路業', '資訊服務業')
ELM = np.array([L.SL['industry'].get(c, '') in ELSET for c in L.COLS])
KNOWN = np.array([c in L.SL.index for c in L.COLS])
EL6 = np.array([L.SL['industry'].get(c, '') in ELSET[:6] for c in L.COLS])
def sub2(mask_fn):
    # 子樣本內 G 對「同子樣本股池等權」的超額
    v = []
    for i, r in enumerate(rows):
        mk = mask_fn(i, r); m = r['G'] & mk & np.isfinite(r['ex']); pm = r['el'] & mk & np.isfinite(r['ex'])
        v.append(r['ex'][m].mean() - r['ex'][pm].mean() if m.any() else np.nan)
    v = np.array(v); return (round(float(np.nanmean(v) * 100), 2), round(L.tstat(v), 2))
out['G_share_liq_101_300'] = float(np.mean([(liq[r['d']][r['G']] > 100).mean() for r in rows]))
out['elec_share_G'] = float(np.mean([ELM[r['G']].mean() for r in rows]))
out['sub_vs_subpool'] = {'liq_101_300': sub2(lambda i, r: liq[r['d']] > 100), 'liq_top100': sub2(lambda i, r: liq[r['d']] <= 100),
                         'elec': sub2(lambda i, r: ELM), 'nonelec': sub2(lambda i, r: ~ELM)}
out['sub'] = {
    'liq_101_300': sub(lambda i, r: liq[r['d']] > 100), 'liq_top100': sub(lambda i, r: liq[r['d']] <= 100),
    'elec': sub(lambda i, r: ELM), 'nonelec': sub(lambda i, r: ~ELM),
    'nonelec_known': sub(lambda i, r: ~ELM & KNOWN), 'elec_strict6': sub(lambda i, r: EL6), 'nonelec_strict6': sub(lambda i, r: ~EL6),
    '2006_2008': sub(lambda i, r: np.ones(L.N, bool), lambda d: d.year <= 2008),
    '2009_2012': sub(lambda i, r: np.ones(L.N, bool), lambda d: 2009 <= d.year <= 2012),
    '2013': sub(lambda i, r: np.ones(L.N, bool), lambda d: d.year == 2013)}

# 事件時間：G 形成後第 k 個窗的超額（以 G 形成月的成員，看第 k 個之後窗）
ev = {}
for k in range(6):
    v = []
    for i in range(len(rows) - k):
        g = rows[i]['G']; x = rows[i + k]['ex']; m = g & np.isfinite(x)
        if m.any(): v.append(x[m].mean())
    ev[k] = (round(float(np.mean(v) * 100), 2), round(L.tstat(v), 2))
out['event_time'] = ev

# 3 個月內創 52 週新高、進入主策略前 30 名
NEWH = np.isclose(L.S52, 1.0) | (L.S52 >= 1.0)
def within3(i, mask):
    d = rows[i]['d']; t = CS[d]['t']; t3 = min(t + 63, L.IDX.get_loc(END))
    hit = NEWH[t + 1:t3 + 1].any(axis=0)
    return hit[mask]
LOWCUT, HICUT = 0.3, 0.85   # 營收強低段＝p52 ≤ 0.3（第 1～3 段）；高段＝p52 > 0.85（與研究員檔數 10.4／11.2 一致）
grpdef = {'G': lambda r: r['G'], 'REST': lambda r: r['el'] & ~r['G'],
          'LOWSTRONG': lambda r: r['strong'] & (r['p'] <= LOWCUT if LOWCUT == 0.3 else r['p'] < LOWCUT), 'HIGHSTRONG': lambda r: r['strong'] & (r['p'] > HICUT)}
out['LOW_HIGH_counts'] = [float(np.mean([f(r).sum() for r in rows])) for f in (grpdef['LOWSTRONG'], grpdef['HIGHSTRONG'])]
nh_ = {}
for g, f in grpdef.items():
    v = [within3(i, f(r)).mean() for i, r in enumerate(rows) if f(r).any()]
    nh_[g] = float(np.mean(v))
    v87 = [within3(i, f(r)).mean() for i, r in enumerate(rows[:87]) if f(r).any()]
    nh_[g + '_87m'] = float(np.mean(v87))
out['newhigh_3m'] = nh_
# 主策略前 30：之後 3 個訊號日（m+1..m+3）任一進前 30
TOP30 = {}
for d in days:
    cs = CS[d]; s = L.main_score(cs); o = L.order(s, cs['y3'])[:30]
    m = np.zeros(L.N, bool); m[o] = True; TOP30[d] = m
p2 = {}; p2s = {}
for g, f in grpdef.items():
    v = []
    for i, r in enumerate(rows):
        nxt = [TOP30[x] for x in days[i + 1:i + 4]]
        if not nxt or not f(r).any(): v.append(np.nan); continue
        hit = np.any(nxt, axis=0); v.append(hit[f(r)].mean())
    p2[g] = float(np.nanmean(v)); p2s[g] = np.array(v)
out['P2'] = p2; out['P2_t_G_REST'] = L.tstat(p2s['G'] - p2s['REST'])
# 變體：當月不在前 30、之後 3 個月內進入（只算有完整 3 個月的訊號月）
p2b = {}; p2bs = {}
for g, f in grpdef.items():
    v = []
    for i, r in enumerate(rows[:-3]):
        nxt = [TOP30[x] for x in days[i + 1:i + 4]]
        hit = np.any(nxt, axis=0) & ~TOP30[r['d']]; mm = f(r)
        v.append(hit[mm].mean() if mm.any() else np.nan)
    p2b[g] = float(np.nanmean(v)); p2bs[g] = np.array(v)
for lag in (1, 2, 3):
    vv = {}
    for g, f in grpdef.items():
        v = [TOP30[days[i + lag]][f(r)].mean() for i, r in enumerate(rows[:-3]) if f(r).any()]
        vv[g] = round(float(np.mean(v)), 3)
    out[f'P2_at_m+{lag}'] = vv
p2c = {}; p2cs = {}
for g, f in grpdef.items():
    v = []
    for i, r in enumerate(rows[:-3]):
        nxt = [TOP30[x] for x in days[i + 1:i + 4]]
        mm = f(r) & ~TOP30[r['d']]; hit = np.any(nxt, axis=0)
        v.append(hit[mm].mean() if mm.any() else np.nan)
    p2c[g] = float(np.nanmean(v)); p2cs[g] = np.array(v)
out['P2_cond_notcur'] = p2c; out['P2_cond_t_G_REST'] = L.tstat(p2cs['G'] - p2cs['REST']); out['P2_cond_t_G_LOW'] = L.tstat(p2cs['G'] - p2cs['LOWSTRONG'])
out['P2_notcur'] = p2b; out['P2_notcur_t_G_REST'] = L.tstat(p2bs['G'] - p2bs['REST']); out['P2_notcur_t_G_LOW'] = L.tstat(p2bs['G'] - p2bs['LOWSTRONG'])
# 3 個月超額（事後分組）
x3h, x3n = [], []
for i, r in enumerate(rows):
    t = CS[r['d']]['t']; t3 = min(L.IDX.get_loc(WD[min(i + 3, len(WD) - 1)]), L.IDX.get_loc(END))
    rr = ADJF[t3] / ADJF[t] - 1; ex3 = rr - np.nanmean(rr[r['el']])
    hit = NEWH[t + 1:min(t + 63, L.IDX.get_loc(END)) + 1].any(axis=0)
    a = r['G'] & hit & np.isfinite(ex3); b = r['G'] & ~hit & np.isfinite(ex3)
    if a.any(): x3h.append(ex3[a].mean())
    if b.any(): x3n.append(ex3[b].mean())
out['ex3m_hit_nohit'] = (float(np.mean(x3h) * 100), float(np.mean(x3n) * 100))
json.dump(out, open(HERE / f'rep_explore_{segmode}.json', 'w'), ensure_ascii=False, indent=1, default=float)
for k, v in out.items(): print(k, v)

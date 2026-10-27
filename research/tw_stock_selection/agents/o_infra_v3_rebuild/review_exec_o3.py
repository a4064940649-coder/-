#!/usr/bin/env python3
"""反方審查（成本與可成交）：只讀 import o3_core，不修改研究員任何檔案。輸出只印到 stdout。
python3 -I -B review_exec_o3.py"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import o3_core as O
V, K = O.V, O.K

RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
print('patched', O.apply_patch())
E = O.load_events(); H = O.load_disposal_hist()
EXTRA = {pd.Timestamp(k): v for k, v in RES['PREREG_params']['D_hi_extra'].items()}
YL, Y5, _ = V.load_revenue(); YR = O.load_rdel()
XR = O.exit_rules(E, 'main')
O.set_disposal('all_plus', H, EXTRA)

class V3AB(O.V3):
    U = None
    def y3(self, d, back=0):
        y = super().y3(d, back)
        if back == 0 and self.U is not None:
            u = self.U(d)
            if u is not None and u.any(): y = np.where(u, super().y3(d, 1), y)
        return y

S = O.V3('gapfix', YR, Y5, exfin=True)
t0, t1, sd = S.bounds()
pl = S.plan(10, 'official')
print('gap-up blocked replacements (n10):', S.nblk)
ref = pd.read_parquet(HERE / 'navs_P1.parquet')['P1|R1|X1|Dhi|EXFIN|n10|normal'].dropna()
ew, _ = O.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1, xrules=XR)

def exs(nav):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    nd = nav.reindex(D); pr = (nd.shift(-1) / nd - 1).iloc[:-1]
    bd = K.E50.reindex(D); bp = (bd.shift(-1) / bd - 1).iloc[:-1]
    x = (pr - bp).dropna(); return x[(x.index >= '2006-04-01') & (x.index <= '2026-10-12')]

def summ(nav):
    s = V.seg_stats(nav, ew, *V.SEGS['ALL'][:2], sd, ad=V.SEGS['ALL'][2])
    p1 = V.seg_stats(nav, ew, *V.SEGS['P1'][:2], sd, ad=V.SEGS['P1'][2])
    x = exs(nav)
    return dict(CAGR=round(s['CAGR'], 2), t=round(s['t_0050'], 3), t_cal=round(s['t_0050_cal'], 2), mean_pp=round(x.mean() * 100, 3), P1=round(p1['CAGR'], 2), tP1=round(p1['t_0050'], 2))

nav, lg = O.simulate([(p[0], p[1]) for p in pl], *K.rates(0.001), t0, t1, xrules=XR)
print('reproduce max |diff|', float((nav.reindex(ref.index) / ref - 1).abs().max()), 'turnover', round(lg['turnover'], 4))
print('normal', summ(nav))

print('\n== 1. 成本敏感度（單邊滑價；手續費 0.1425%×0.6、賣方證交稅 0.3%）==')
for slip in (0.0, 0.0005, 0.001, 0.0013, 0.0015, 0.002, 0.0025, 0.003, 0.004):
    b, s_ = K.rates(slip)
    nv, _ = O.simulate([(p[0], p[1]) for p in pl], b, s_, t0, t1, xrules=XR)
    print(f'slip {slip*100:.2f}%', summ(nv))
for fee_disc in (1.0,):
    fee = 0.001425 * fee_disc
    nv, _ = O.simulate([(p[0], p[1]) for p in pl], fee + 0.001, fee + 0.003 + 0.001, t0, t1, xrules=XR)
    print('全額手續費 0.1425% ＋ 滑價 0.1%', summ(nv))

print('\n== 2. 容量與成交量占比（部位＝資金/10，訊號日 60 日均成交值；換股日全日成交值）==')
S52, V60 = V.rolling_signals('gapfix')
rows = []
prev = set()
for (e, pk, pool, d) in pl:
    for j in pk:
        rows.append(dict(d=d, e=K.IDX[e], code=K.COLS[j], new=(j not in prev), v60=V60[K.IDX.get_loc(d), j], vday=K.V.iat[e, j],
                         openp=K.On[e, j], prevc=K.PREVC[e, j], ref=K.ref_price(e)[j]))
    prev = set(pk)
R = pd.DataFrame(rows)
R['seg'] = np.select([R['d'] < '2014-01-01', R['d'] < '2021-01-01'], ['S1', 'S2'], 'S3')
for C in (3e5, 1e6, 3e6, 1e7):
    pos = C / 10
    R['pv60'] = pos / R['v60']; R['pday'] = pos / R['vday']
    g = R.groupby('seg')
    print(f'資金 {C/1e4:.0f} 萬：部位/60日均值 中位 {R.pv60.median()*100:.3f}%  >1% 比率 {(R.pv60>0.01).mean():.3f}  >2% {(R.pv60>0.02).mean():.3f}  >5% {(R.pv60>0.05).mean():.3f}',
          ' 分段>1%:', {k: round((v.pv60 > 0.01).mean(), 3) for k, v in g}, ' 分段>2%:', {k: round((v.pv60 > 0.02).mean(), 3) for k, v in g})
print('入選股 60 日均成交值分位（百萬）', (R['v60'].quantile([0.01, 0.05, 0.1, 0.25, 0.5]) / 1e6).round(1).to_dict())
print('分段 60 日均值 10% 分位（百萬）', {k: round(v['v60'].quantile(0.1) / 1e6, 1) for k, v in R.groupby('seg')})

print('\n== 3. 一張（1000 股）買不買得起：開盤價×1000 > 資金/10 ==')
R['lot'] = R['openp'] * 1000
odd = pd.Timestamp('2020-10-26')
for C in (3e5, 1e6, 3e6):
    pos = C / 10
    m = R['lot'] > pos
    pre = R['e'] < odd
    print(f'資金 {C/1e4:.0f} 萬：一張 > 部位 的入選名額比率 全期 {m.mean():.3f}；2020-10-26 前（零股只能盤後）{m[pre].mean():.3f}；之後 {m[~pre].mean():.3f}；'
          f'一張 > 2×部位（整股化誤差 >50%）{(R["lot"] > 2 * pos).mean():.3f}')

print('\n== 4. 開盤接近漲停（未達漲停、被引擎視為以開盤價全數成交）／賣出日開盤跌停 ==')
def lim(t): return 0.07 if K.IDX[t] < pd.Timestamp('2015-06-01') else 0.10
nb = R[R['new']].copy()
lu = np.array([K.limit_up_price(np.array([r]), lim(K.IDX.get_loc(e)))[0] if np.isfinite(r) else np.nan for r, e in zip(nb['ref'], nb['e'])])
nb['gap'] = nb['openp'] / nb['ref'] - 1
nb['to_lu'] = lu / nb['openp'] - 1
print('新買進名額', len(nb), '；開盤跳空 ≥ +3%', round((nb.gap >= 0.03).mean(), 3), '；≥ +5%', round((nb.gap >= 0.05).mean(), 3),
      '；開盤距漲停 ≤ 1 檔內（未封）', int(((nb.to_lu >= 0) & (nb.to_lu <= 0.012)).sum()))
# 賣出：前一期持有、本期不在名單、且當日有開盤
ld_n = 0; sells = 0; ld_list = []
for i in range(1, len(pl)):
    e = pl[i][0]; prevpk = set(pl[i - 1][1]); cur = set(pl[i][1])
    rf = K.ref_price(e); L = lim(e)
    for j in prevpk - cur:
        if not (K.TRADED[e, j] and np.isfinite(K.On[e, j])): continue
        sells += 1
        raw = rf[j] * (1 - L); tk = K.tick(np.array([raw]))[0]; ldp = math.ceil(raw / tk - 1e-9) * tk
        if K.On[e, j] <= ldp + 1e-6: ld_n += 1; ld_list.append((str(K.IDX[e].date()), K.COLS[j]))
print('賣出名額', sells, '；開盤即跌停（可能賣不掉，引擎當作成交）', ld_n, ld_list[:10])

print('\n== 5. 處置：換股日（成交日）當天已在處置中的新買進／賣出 ==')
def in_disp(code, day):
    T = O.DSP5 if day >= O.D5_START else H
    x = T[(T['code'] == code) & (T['start'] <= day) & (T['end'] >= day)]
    return len(x) > 0
nb['disp_e'] = [in_disp(c, e) for c, e in zip(nb['code'], nb['e'])]
print('新買進在成交日已進入處置（訊號日晚間公告、隔日生效，SOP 實盤會看到）：', int(nb['disp_e'].sum()), '/', len(nb),
      '；2021-09-24 後（真實名單）', int(nb[nb.e >= O.D5_START]['disp_e'].sum()), '/', int((nb.e >= O.D5_START).sum()))
print(nb[nb['disp_e']][['d', 'e', 'code']].astype(str).values.tolist()[:20])
# 這些名額的持有期報酬 vs 同期組合
ex = [p[0] for p in pl]
G = K.GROW
def hp(j, i):
    t = pl[i][0]; t2 = ex[i + 1] if i + 1 < len(ex) else t1
    g = K.Cn[t, j] / K.On[t, j] * np.prod(G[t + 1:t2 + 1, j])
    if i + 1 < len(ex) and K.TRADED[t2, j] and np.isfinite(K.On[t2, j]): g *= K.On[t2, j] / K.Cn[t2, j]
    return g - 1
di = []
for i, (e, pk, pool, d) in enumerate(pl):
    for j in pk:
        if ((nb['e'] == K.IDX[e]) & (nb['code'] == K.COLS[j]) & nb['disp_e']).any():
            others = [hp(k, i) for k in pk if k != j]
            di.append((str(d.date()), K.COLS[j], round(hp(j, i) * 100, 1), round(np.mean(others) * 100, 1)))
if di:
    print('處置中買進名額 持有期報酬% vs 同期其餘持股平均%：', di, ' 平均差', round(np.mean([a[2] - a[3] for a in di]), 2), 'pp')

print('\n== 6. 只用已公告營收（AB，main_v2 自己的實盤基準）套到 V3 ==')
for mode, seeds in (('worst', [None]), ('mc', list(range(8)))):
    out = []
    for sd_ in seeds:
        SA = V3AB('gapfix', YR, Y5, exfin=True); SA.U = V.make_unavail(mode, sd_)
        pa = SA.plan(10, 'official')
        nv, _ = O.simulate([(p[0], p[1]) for p in pa], *K.rates(0.001), t0, t1, xrules=XR)
        out.append(summ(nv))
        if mode == 'mc':
            nv2, _ = O.simulate([(p[0], p[1]) for p in pa], *K.rates('stress'), t0, t1, xrules=XR)
            out[-1]['stress_t'] = summ(nv2)['t']
    print(mode, out if mode == 'worst' else {k: round(float(np.mean([o[k] for o in out])), 3) for k in out[0]}, '' if mode == 'worst' else ('t 範圍', min(o['t'] for o in out), max(o['t'] for o in out)))
print('done')

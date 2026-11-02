#!/usr/bin/env python3
"""反方審查（資料與統計）o10：只讀 import 研究員程式，不寫任何研究員檔案。輸出印到 stdout。
A. 處置腿 d1−d0 同月配對差 t（SEL 2021-10～2023-12、VAL 2024-01～2026-10）
B. 處置腿加入衝擊成本（上櫃每邊 +0.41%、上市每邊 +0.24%，取研究員 K5T 100 萬估計）後 d1 vs d0
C. d1 VAL 逐年對 0050 超額
D. 處置事件以「月」群聚的 t（研究員用進場日群聚，持有期重疊）
E. P1 以桶內前瞻報酬橫斷面標準差標準化後，上櫃－上市價差是否仍在（檢查是否只是波動度較大）
"""
import sys, math
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/o_tpex_pool_extension')
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import otlib as L, dlib as DL
import s1_explore as S1
D = L.D
E50 = pd.Series(D.E50, index=D.IDX)


def mret(nav):
    return nav.resample('ME').last().pct_change().dropna()


def run(mk, a, b, cost=(0.0025, 0.0055)):
    nav, tr, sk = DL.sleeve(DL.events(markets=mk), a, b, cost=cost)
    return nav, tr, sk


PER = {'SEL': (pd.Timestamp('2021-10-01'), pd.Timestamp('2023-12-29')), 'VAL': (pd.Timestamp('2024-01-01'), pd.Timestamp('2026-10-30'))}
print('=== A/B/C 處置腿 ===')
for p, (a, b) in PER.items():
    out = {}
    for lab, c0, c1 in (('normal', (0.0025, 0.0055), (0.0025, 0.0055)), ('impact', (0.0049, 0.0079), (0.0066, 0.0096))):
        n0, t0, _ = run(('TWSE',), a, b, c0)
        n1, t1, _ = run(('TWSE', 'TPEX'), a, b, c1)   # impact 版：d1 全部以上櫃衝擊計（保守，偏不利）
        r0, r1 = mret(n0), mret(n1)
        rb = E50.reindex(n0.index).resample('ME').last().pct_change().dropna()
        d = (r1 - r0).dropna()
        ann = lambda n: (n.iloc[-1] ** (365.25 / (n.index[-1] - n.index[0]).days) - 1) * 100
        print(p, lab, f'd0 ann {ann(n0):.1f} t_vs0050 {L.tstat((r0-rb).dropna()):.2f} | d1 ann {ann(n1):.1f} t_vs0050 {L.tstat((r1-rb).dropna()):.2f} | '
              f'd1-d0 mean {d.mean()*100:.2f}pp/月 t {L.tstat(d):.2f} nw {L.nw_t(d):.2f} n {len(d)}')
        if lab == 'normal':
            out['n0'], out['n1'], out['t1'] = n0, n1, t1
    if p == 'VAL':
        n1 = out['n1']; y1 = n1.resample('YE').last(); y1 = y1 / y1.shift(1).fillna(1) - 1
        yb = E50.reindex(n1.index); yb = yb / yb.iloc[0]; yb = yb.resample('YE').last(); yb = yb / yb.shift(1).fillna(1) - 1
        print('d1 VAL 逐年', {str(k.year): round(float(v * 100), 1) for k, v in y1.items()}, '0050', {str(k.year): round(float(v * 100), 1) for k, v in yb.items()})
        n0 = out['n0']; y0 = n0.resample('YE').last(); y0 = y0 / y0.shift(1).fillna(1) - 1
        print('d0 VAL 逐年', {str(k.year): round(float(v * 100), 1) for k, v in y0.items()})
        tr = out['t1']
        tr['m'] = pd.to_datetime(tr.entry).dt.to_period('M')
        for mk in ('TWSE', 'TPEX'):
            s = tr[tr.mkt == mk]
            g = s.groupby('m').ret.mean()
            print('VAL 已成交', mk, 'n', len(s), 'mean', round(s.ret.mean() * 100, 2), 'median', round(s.ret.median() * 100, 2), 't_month_cluster', round(L.tstat(g), 2), 'n_month', len(g),
                  'drop_top5%', round(np.sort(s.ret.to_numpy())[:int(len(s) * .95)].mean() * 100, 2))
        top = tr.sort_values('ret', ascending=False).head(10)
        print('VAL top10 trades', top[['code', 'mkt', 'entry', 'ret']].to_dict('records'))
        print('VAL top10 share of sum ret', round(top.ret.sum() / tr.ret.sum(), 3))

print('=== D 處置事件（研究員 explore 口徑）月群聚 t ===')
for p, (a, b) in PER.items():
    ev = S1.disposal_events(a, b); ok = ev[ev.ok == True].copy(); ok['x'] = ok.raw - 0.008
    for mk in ('TWSE', 'TPEX'):
        s = ok[ok.mkt == mk]
        gd = s.groupby('start').x.mean(); gm = s.groupby(s.start.dt.to_period('M')).x.mean()
        print(p, mk, 'n', len(s), 'mean', round(s.x.mean() * 100, 2), 't_date', round(L.tstat(gd), 2), 't_month', round(L.tstat(gm), 2), 'nw_month', round(L.nw_t(gm), 2), 'n_month', len(gm))

print('=== E P1 波動度標準化 ===')
S = L.SIGS[0]; rows = []
for k in range(len(S) - 1):
    sig, ex = S[k]; ex2 = S[k + 1][1]; d = D.IDX[sig]
    if d < S1.SELA or d > S1.SELB or str(pd.Period(d, 'M')) in L.hole_months(): continue
    ok, y3 = S1.elig(sig, ex); w = np.where(ok)[0]
    if (D.MK[sig, w]).sum() < 50: continue
    sc = 0.5 * pd.Series(D.S52[sig, w]).rank(pct=True).to_numpy() + 0.5 * pd.Series(y3[w]).rank(pct=True).to_numpy()
    q = pd.qcut(pd.Series(D.V60[sig, w]).rank(method='first'), 5, labels=False).to_numpy()
    fr = S1.fwd(ex, ex2, w)
    raw, nrm, rk = [], [], []
    for bb in range(5):
        res = {}
        for m in (0, 1):
            sel = (q == bb) & (D.MK[sig, w] == bool(m)) & np.isfinite(fr)
            if sel.sum() < 10: continue
            s_ = sc[sel]; f_ = fr[sel]; thr = np.quantile(s_, 0.8); top = s_ >= thr
            sp = f_[top].mean() - f_.mean(); sd = f_.std(ddof=1)
            # 秩版本：前瞻報酬桶內百分位（去掉極端值與尺度）
            fr_r = pd.Series(f_).rank(pct=True).to_numpy()
            res[m] = (sp, sp / sd if sd > 0 else np.nan, fr_r[top].mean() - fr_r.mean(), np.median(f_[top]) - np.median(f_))
        if 0 in res and 1 in res:
            raw.append(res[1][0] - res[0][0]); nrm.append(res[1][1] - res[0][1]); rk.append(res[1][2] - res[0][2])
    if raw: rows.append(dict(date=d, raw=np.mean(raw), nrm=np.mean(nrm), rk=np.mean(rk)))
df = pd.DataFrame(rows).set_index('date')
for c in ('raw', 'nrm', 'rk'):
    x = df[c]
    print('P1', c, 'mean', round(x.mean() * (100 if c == 'raw' else 1), 4), 't', round(L.tstat(x), 2), 'nw', round(L.nw_t(x), 2), 'n', len(x),
          '| drop best 3 years t', round(L.tstat(x[~x.index.year.isin(x.groupby(x.index.year).mean().nlargest(3).index)]), 2),
          '| by year', {int(y): round(float(v * (100 if c == 'raw' else 1)), 3) for y, v in x.groupby(x.index.year).mean().items()})

print('=== F P1 樣本外（2019-05～2026-10，剔除洞月；研究員未做）===')
r = S1.e1_p1(pd.Timestamp('2019-05-01'), pd.Timestamp('2026-10-31'))
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith('bucket')})
print({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in r.items() if k.startswith('bucket')})

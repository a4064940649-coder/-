#!/usr/bin/env python3
"""反方審查（成本與可成交）：python3 -I -B review_exec_o2.py
只讀研究員檔案、只 import osurv_core（間接 v2_core/ia_core）；不寫任何檔案，輸出到 stdout。
A. B10_DEL 的凍結部位：引擎把下市持股凍在最後收盤、永不變現 → 拆出「死錢」造成的年化損失
   （freeze=False＝換股日以最後收盤出清再投入；併購下市的實際情況較接近此）
B. R_DEL 相對 ORIG 新進股（排除金融 n10）的可成交性：訊號日 V60、成交日開盤漲停、跳空、持有期內收跌停日、最後交易日
C. 換手與成本：R_DEL/ORIG 換手、0.3% 稅＋滑價拖累
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import osurv_core as O
V, K = O.V, O.K

COST = K.rates(0.001)
def seg(nav, ew, sd, key):
    a, b, ad = V.SEGS[key]
    s = V.seg_stats(nav, ew, a, b, sd, ad=ad)
    return round(s['CAGR'] * 100 if abs(s['CAGR']) < 5 else s['CAGR'], 2), round(s['t_0050'], 2)

# ---------- A ----------
YL, Y5 = O.load_yl('R_DEL')
GAP = O.gapsets('R_DEL')
for pf in (1.0, 0.5):
    S = O.V2B('gapfix', YL, Y5, exfin=True, gap=GAP, pfix=pf)
    t0, t1, sd = S.bounds()
    pl = S.plan(10, 'official')
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    navF, lgF = K.simulate([(p[0], p[1]) for p in pl], *COST, t0, t1, freeze=True)
    navL, lgL = K.simulate([(p[0], p[1]) for p in pl], *COST, t0, t1, freeze=False)
    print(f'A. pfix={pf} EXFIN n10 normal：freeze=True ALL', seg(navF, ew, sd, 'ALL'), 'S2', seg(navF, ew, sd, 'S2'),
          '| freeze=False ALL', seg(navL, ew, sd, 'ALL'), 'S2', seg(navL, ew, sd, 'S2'))
    fr = lgF['frozen']
    if fr:
        ws = [f[2] for f in fr]
        print('   凍結事件', len(fr), '首', fr[0][0], '末', fr[-1][0], '最大占比', round(max(ws), 3), '平均占比', round(float(np.mean(ws)), 3))
        last = fr[-1]; print('   最後一次凍結的代碼', last[1], '占比', round(last[2], 3))
        lt = {c: str(K.IDX[np.where(K.TRADED[:, K.CI[c]])[0][-1]].date()) for f in fr for c in f[1]}
        print('   凍結股最後交易日', lt)

# ---------- B ----------
P0 = json.loads((HERE / 'plans_ORIG.json').read_text())['EXFIN|n10']
P1 = json.loads((HERE / 'plans_R_DEL.json').read_text())['EXFIN|n10']
d0 = {p['sig']: set(p['codes']) for p in P0}
S1 = O.V2B('gapfix', *O.load_yl('R_DEL'), exfin=True)
S1.plan(10, 'official')
rows = []
sigs = [p['sig'] for p in P1]
for i, p in enumerate(P1):
    new = set(p['codes']) - d0.get(p['sig'], set())
    if not new: continue
    d = pd.Timestamp(p['sig']); t = K.IDX.get_loc(d); e = t + 1
    t2 = K.IDX.get_loc(pd.Timestamp(sigs[i + 1])) + 1 if i + 1 < len(sigs) else len(K.IDX) - 1
    gu = K.gap_up_open(e)
    for c in sorted(new):
        j = K.CI[c]
        r = K.Rn[e:t2 + 1, j]
        lim = 0.07 if d < pd.Timestamp('2015-06-01') else 0.10
        ndown = int(np.nansum(r <= -lim + 0.005)); nup = int(np.nansum(r >= lim - 0.005))
        lt = K.IDX[np.where(K.TRADED[:, j])[0][-1]].date()
        rows.append(dict(sig=p['sig'], code=c, V60_mNTD=round(float(S1.V60[t, j]) / 1e6, 1),
                         open_gap_pct=round(float(K.On[e, j] / K.Cn[t, j] - 1) * 100, 2), limit_up_open=bool(gu[j]),
                         n_limdown_days=ndown, n_limup_days=nup, last_trade=str(lt),
                         hold_ret=round(p['ret'][p['codes'].index(c)] * 100, 1)))
B = pd.DataFrame(rows)
print('B. R_DEL 新進股（排除金融 n10）'); print(B.to_string(index=False))
# 全體入選股的 V60 分布作對照
v_all = []
for p in P1:
    t = K.IDX.get_loc(pd.Timestamp(p['sig']))
    v_all += [float(S1.V60[t, K.CI[c]]) / 1e6 for c in p['codes']]
v_all = np.array(v_all)
print('   全體入選股 V60（百萬元）中位數', round(float(np.nanmedian(v_all)), 1), '第10百分位', round(float(np.nanpercentile(v_all, 10)), 1),
      '；1,000 萬 n10 每檔 100 萬 ÷ V60 > 5% 的比例', round(float(np.mean(1.0 / v_all > 0.05)), 3))

# ---------- C ----------
for ver in ('ORIG', 'R_DEL'):
    r = json.loads((HERE / f'run_{ver}.json').read_text())['BASE']['EXFIN|n10']
    drag = r['normal_turnover'] * 12 * (COST[0] + COST[1])
    print(f'C. {ver} 月換手 {r["normal_turnover"]:.3f}，年化成本拖累約 {drag * 100:.2f}pp（一般）；ALL 一般 {r["normal"]["ALL"]["CAGR"]} 壓力 {r["stress"]["ALL"]["CAGR"]}（t {r["stress"]["ALL"]["t_0050"]}）；S2 壓力 {r["stress"]["S2"]["CAGR"]}（t {r["stress"]["S2"]["t_0050"]}）')
DSP = K.DSP
print('C. 處置資料起點', DSP['start'].min().date(), '→ 2021-09 以前所有回測（含本題全部修補期 2015~2019）都沒有排除處置股')

# ---------- D ----------
# 處置排除在 2021-09 以前形同不存在：用 2021-09 之後的資料估「有沒有處置過濾」的差距（ORIG、排除金融 n10、一般成本）
if __name__ == '__main__':
    YL0, Y50 = O.load_yl('ORIG')
    outs = {}
    for tag in ('with_disposal', 'no_disposal'):
        if tag == 'no_disposal':
            K.disposal_mask = lambda d: np.zeros(K.N, bool)
        S = O.V2B('gapfix', YL0, Y50, exfin=True)
        t0, t1, sd = S.bounds()
        pl = S.plan(10, 'official')
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
        nav, lg = K.simulate([(p[0], p[1]) for p in pl], *COST, t0, t1)
        sel = [(str(p[3].date()), set(K.COLS[j] for j in p[1])) for p in pl if p[3] >= pd.Timestamp('2021-10-01')]
        outs[tag] = sel
        a = pd.Timestamp('2021-10-01')
        r = nav.pct_change(); r = r[r.index >= a]
        cg = float((1 + r.fillna(0)).prod() ** (365.25 / (r.index[-1] - a).days) - 1) * 100
        print(f'D. {tag}: 2021-10~2026-10 年化 {cg:.2f}%；S4', seg(nav, ew, sd, 'S4'), 'ALL', seg(nav, ew, sd, 'ALL'))
    nchg = sum(len(b - a) for (_, a), (_, b) in zip(outs['with_disposal'], outs['no_disposal']))
    print('D. 2021-10 起不排處置會換掉的名額', nchg, '／', 10 * len(outs['with_disposal']))

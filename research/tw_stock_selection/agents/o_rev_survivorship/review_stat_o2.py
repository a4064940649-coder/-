#!/usr/bin/env python3
"""反方審查（資料與統計）：python3 -I -B review_stat_o2.py
只讀研究員的檔案；不寫任何研究員檔案。輸出印到 stdout。
A. 價格面板本身的存活者結構（最後交易日分布）
B. 由 navs 重算 EXFIN n10 的 ALL 年化與訊號窗 t；拆年份、去最佳年
C. 額外界限：pfix 0.6/0.75/0.9；B10 只套 2013-01~2015-03；B10 只套 2021+；（BASE、EXFIN n10、一般成本）
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import osurv_core as O
V, K = O.V, O.K

PART = sys.argv[1] if len(sys.argv) > 1 else 'ABC'
if PART == 'ABC':
    # ---------- A ----------
    lt = pd.Series({c: K.IDX[np.where(K.TRADED[:, j])[0][-1]] for j, c in enumerate(K.COLS) if K.TRADED[:, j].any()})
    end = K.IDX[-1]
    dl = lt[lt < end - pd.Timedelta(days=30)]
    print('A. 面板檔數', len(lt), '已停止交易', len(dl))
    print('   最後交易日=2019-05-16 檔數', int((dl == pd.Timestamp('2019-05-16')).sum()))
    print('   停止交易年分布', dl.groupby(dl.dt.year).size().to_dict())
    fi = pd.Series({c: K.IDX[np.where(K.TRADED[:, j])[0][0]] for j, c in enumerate(K.COLS) if K.TRADED[:, j].any()})
    both = dl[(dl > pd.Timestamp('2019-05-16')) & (dl < pd.Timestamp('2024-01-01'))]
    print('   2019-05-17~2023-12 停止交易檔數', len(both), list(both.index[:30]))
    nl = fi[(fi > pd.Timestamp('2019-06-01'))]
    print('   2019-06 後才出現的檔數', len(nl), ' 其中已停止交易', int(nl.index.isin(dl.index).sum()))

    # ---------- B ----------
    S0 = O.V2B('gapfix', *O.load_yl('ORIG'), exfin=True)
    t0, t1, sd = S0.bounds()
    D = pd.DatetimeIndex([d for d in sd if d in K.IDX])
    def mex(nav):
        nd = nav.reindex(D); pr = (nd.shift(-1) / nd - 1).iloc[:-1]
        bd = K.E50.reindex(D); bp = (bd.shift(-1) / bd - 1).iloc[:-1]
        x = (pr - bp).dropna(); return x[(x.index >= '2006-04-01') & (x.index <= '2026-10-12')]
    out = {}
    for ver in ['ORIG', 'R_DEL', 'B05_DEL', 'B10_DEL', 'R_ALL']:
        N_ = pd.read_parquet(HERE / f'navs_{ver}.parquet')
        nav = N_['EXFIN|n10|normal'].dropna(); ew = N_['EXFIN|EW'].dropna()
        st = V.seg_stats(nav, ew, '2006-04-01', '2026-10-12', sd, ad='2006-04-01')
        x = mex(nav); out[ver] = x
        yr = x.groupby(x.index.year).sum()
        best = yr.sort_values(ascending=False)
        def t_drop(k):
            keep = ~x.index.year.isin(best.index[:k]); return K.tstat(x[keep])
        print(f'B. {ver}: CAGR {st["CAGR"]:.2f} t0050 {st["t_0050"]:.3f} tEW {st["t_EW"]:.3f} n {len(x)} mean {x.mean()*100:.3f}pp sd {x.std()*100:.3f}'
              f' | 去最佳1年 t {t_drop(1):.2f} 去最佳2年 {t_drop(2):.2f} 去最佳3年 {t_drop(3):.2f} 最佳年 {list(best.index[:3])}'
              f' | 2014+ t {K.tstat(x[x.index >= "2014-01-01"]):.2f} 2019-05+ t {K.tstat(x[x.index >= "2019-05-01"]):.2f}')
    d = (out['R_DEL'] - out['ORIG']).dropna()
    print('   R_DEL-ORIG 配對 mean %.4fpp t %.2f nonzero %d' % (d.mean() * 100, K.tstat(d), int((d.abs() > 1e-12).sum())))
    nz = d[d.abs() > 1e-12]; print('   非零月份', [(str(i.date()), round(v * 100, 2)) for i, v in nz.items()])
    print('   非零月份 t（只用非零月）%.2f' % K.tstat(nz))

    # ---------- C ----------
    YL, Y5 = O.load_yl('R_DEL')
    GAP = O.gapsets('R_DEL')
    def run(gap, pfix, tag):
        S = O.V2B('gapfix', YL, Y5, exfin=True, gap=gap, pfix=pfix)
        pl = S.plan(10, 'official')
        a, b, sdd = S.bounds()
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, a, b)
        bb, ss = K.rates(0.001)
        nav, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, a, b)
        r = {k: V.seg_stats(nav, ew, x, y, sdd, ad=z) for k, (x, y, z) in V.SEGS.items()}
        print(f'C. {tag}: ' + ' '.join(f'{k} {v["CAGR"]:.2f}/{v["t_0050"]:.2f}' for k, v in r.items()), 'gap_used_mean', round(np.mean(list(S.ngap_used.values())), 2))
        return r
    run(GAP, 0.5, 'B05_DEL 重現')
    for p in (0.6, 0.75, 0.9):
        run(GAP, p, f'pfix={p}')
    g1 = {d: s for d, s in GAP.items() if d < pd.Timestamp('2015-04-01')}
    g2 = {d: s for d, s in GAP.items() if d >= pd.Timestamp('2021-01-01')}
    g3 = {d: s for d, s in GAP.items() if pd.Timestamp('2015-04-01') <= d < pd.Timestamp('2021-01-01')}
    run(g1, 1.0, 'B10 只套 2013-01~2015-03')
    run(g2, 1.0, 'B10 只套 2021+')
    run(g3, 1.0, 'B10 只套 2015-04~2020')
    run(g1, 0.75, 'pfix0.75 只套 2013-01~2015-03')


# ---------- D：凍結死錢的影響（freeze=False＝無法成交時以最後收盤出清，近似併購現金對價） ----------
if PART == 'D':
    YL, Y5 = O.load_yl('R_DEL'); GAP = O.gapsets('R_DEL')
    def runf(gap, pfix, tag, freeze):
        S = O.V2B('gapfix', YL, Y5, exfin=True, gap=gap, pfix=pfix)
        pl = S.plan(10, 'official'); a, b, sdd = S.bounds()
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, a, b, freeze=freeze)
        bb, ss = K.rates(0.001)
        nav, lg = K.simulate([(p[0], p[1]) for p in pl], bb, ss, a, b, freeze=freeze)
        r = {k: V.seg_stats(nav, ew, x, y, sdd, ad=z) for k, (x, y, z) in V.SEGS.items()}
        fz = lg['frozen']; mx = max([f[2] for f in fz], default=0)
        print(f'D. {tag} freeze={freeze}: ' + ' '.join(f'{k} {v["CAGR"]:.2f}/{v["t_0050"]:.2f}' for k, v in r.items()), f'n_frozen {len(fz)} max_frozen_share {mx:.3f}', flush=True)
    runf(None, None, 'R_DEL', True)
    runf(None, None, 'R_DEL', False)
    for p in (0.75, 0.9, 1.0):
        runf(GAP, p, f'pfix={p}', True)
        runf(GAP, p, f'pfix={p}', False)

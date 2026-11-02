#!/usr/bin/env python3
"""反方審查（成本與可成交）o10：只讀 import 研究員程式，不寫任何研究員檔案，結果印 stdout。
處置腿 d1（上市＋上櫃）vs d0（只上市）在可執行限制下是否仍「兩段都較好」：
 E1 同日事件排序：研究員依代碼排序填 10 槽（d1 有 515 次槽滿），改成 20 組隨機排序看 d1−d0 的 t 分布
 E2 成本：上櫃每邊另加 0.2%／0.4%（盤中零股＋分盤集合競價），上市另加 0.1%／0.2%；最低手續費 20 元（2 萬／槽）
 E3 預收款券：第二次（含以上）處置需全額預收 → 不能用當日賣 0050（T+2）的錢買 → 剔除 second 事件
 E4 決策資訊：「進場日漲幅 ≥ 9% 不買」「5% 成交值上限」用當日收盤資訊 → 改用前一日資訊（13:25 前可知的保守替代）
"""
import sys
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/o_tpex_pool_extension')
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import otlib as L, dlib as DL
D = L.D
E50 = pd.Series(D.E50, index=D.IDX)
PER = {'SEL': (pd.Timestamp('2021-10-01'), pd.Timestamp('2023-12-29')), 'VAL': (pd.Timestamp('2024-01-02'), pd.Timestamp('2026-10-12'))}


def sleeve(ev, a, b, seed=None, extra=(0.0, 0.0), minfee=False, prev_info=False, equity0=2e5):
    ta = int(np.searchsorted(D.IDX, a)); tb = int(np.searchsorted(D.IDX, b, side='right')) - 1
    ev = ev[(ev.i0 >= ta) & (ev.i0 <= tb)]
    rng = np.random.default_rng(seed) if seed is not None else None
    byday = {}
    for r in ev.itertuples(): byday.setdefault(r.i0, []).append(r)
    if rng is not None:
        for k in byday: rng.shuffle(byday[k])
    cash50 = equity0 * D.cpi_scale(D.IDX[ta].year); pos = {}; nav = np.empty(tb - ta + 1); trades = []
    for t in range(ta, tb + 1):
        if t > ta:
            cash50 *= D.E50[t] / D.E50[t - 1]
            for j, p in pos.items():
                g = DL.AIf[t, j] / DL.AIf[t - 1, j]; p[0] *= g if np.isfinite(g) else 1.0
        for j in list(pos):
            p = pos[j]
            if t >= p[1]:
                if np.isfinite(D.R[t, j]) and D.R[t, j] <= -0.09 and D.TR[t, j]: continue
                if not D.TR[t, j]: continue
                ex = extra[1] if p[3] == 'TPEX' else extra[0]
                fee = 0.0055 + ex
                if minfee: fee = max(fee, 20 / p[0] + 0.003 + 0.0025 + ex)   # 最低手續費 20 元＋稅＋原滑價
                v = p[0] * (1 - fee); cash50 += v
                trades.append(dict(mkt=p[3], ret=v / p[4] - 1, second=p[6], t=t)); del pos[j]
        for r in byday.get(t, []):
            j = r.j
            if j in pos or not (D.TR[t, j] and np.isfinite(D.C[t, j])) or not DL.BULL[t]: continue
            rr = D.R[t - 1, j] if prev_info else D.R[t, j]
            if np.isfinite(rr) and rr >= 0.09: continue
            v60 = D.V60p[t, j]
            if not (np.isfinite(v60) and v60 >= 1e7 and D.CF[t - 1, j] >= 10): continue
            if len(pos) >= 10: continue
            eq = cash50 + sum(p[0] for p in pos.values())
            vcap = np.nan_to_num(D.V[t - 1, j] if prev_info else D.V[t, j])
            amt = min(eq / 10, 0.05 * vcap, cash50 / 1.0015)
            if amt <= 0: continue
            ex = extra[1] if r.mkt == 'TPEX' else extra[0]
            fb = 0.0025 + ex
            if minfee: fb = max(fb, 20 / amt + 0.0015 + ex)
            cash50 -= amt * 1.0015
            pos[j] = [amt * (1 - fb), r.i1, r.code, r.mkt, amt, t, r.second]
        nav[t - ta] = cash50 + sum(p[0] for p in pos.values())
    return pd.Series(nav / nav[0], index=D.IDX[ta:tb + 1]), pd.DataFrame(trades)


def mx(nav):
    m = nav.resample('ME').last().pct_change().dropna(); rb = E50.reindex(nav.index).resample('ME').last().pct_change().dropna()
    return (m - rb).dropna()


def ann(nav):
    return float(nav.iloc[-1] ** (365.25 / (nav.index[-1] - nav.index[0]).days) - 1) * 100


def comp(lab, **kw):
    out = {}
    for per, (a, b) in PER.items():
        r = {}
        for dk, mk, fo in (('d0', ('TWSE',), False), ('d1', ('TWSE', 'TPEX'), False)):
            ev = DL.events(markets=mk, first_only=kw.pop('_fo', fo) if False else fo)
            if kw.get('nosecond'): ev = ev[~ev.second]
            kk = {k: v for k, v in kw.items() if k != 'nosecond'}
            nav, tr = sleeve(ev, a, b, **kk)
            r[dk] = (nav, tr)
        x0, x1 = mx(r['d0'][0]), mx(r['d1'][0]); dd = (x1 - x0).dropna()
        out[per] = dict(ann0=round(ann(r['d0'][0]), 1), ann1=round(ann(r['d1'][0]), 1), t0=round(L.tstat(x0), 2), t1=round(L.tstat(x1), 2),
                        diff_pp=round(dd.mean() * 100, 2), t_diff=round(L.tstat(dd), 2),
                        tp_mean=round(r['d1'][1].query('mkt=="TPEX"').ret.mean() * 100, 2) if len(r['d1'][1]) else None)
    print(lab, out, flush=True)
    return out


if __name__ == '__main__':
    base = comp('BASE(代碼排序, 研究員成本)')
    # E1 隨機排序
    rows = []
    for s in range(20):
        o = {}
        for per, (a, b) in PER.items():
            n0, _ = sleeve(DL.events(markets=('TWSE',)), a, b, seed=s); n1, _ = sleeve(DL.events(markets=('TWSE', 'TPEX')), a, b, seed=s)
            x0, x1 = mx(n0), mx(n1)
            o[per] = (L.tstat(x0), L.tstat(x1), L.tstat((x1 - x0).dropna()), ann(n1) - ann(n0))
        rows.append(o)
    for per in PER:
        A = np.array([r[per] for r in rows])
        print(f'E1 {per} 隨機排序 20 組: t0 med {np.median(A[:,0]):.2f} t1 med {np.median(A[:,1]):.2f} t1>t0 比例 {(A[:,1]>A[:,0]).mean():.2f} '
              f't_diff med {np.median(A[:,2]):.2f} [min {A[:,2].min():.2f}, max {A[:,2].max():.2f}] 年化差 med {np.median(A[:,3]):.1f}pp min {A[:,3].min():.1f}', flush=True)
    comp('E2a 加成本 上市+0.1/上櫃+0.2 每邊', extra=(0.001, 0.002))
    comp('E2b 加成本 上市+0.2/上櫃+0.4 每邊', extra=(0.002, 0.004))
    comp('E2c 最低手續費 20 元', minfee=True)
    comp('E3 剔除第二次以上處置(預收款券)', nosecond=True)
    comp('E4 前一日資訊判 9%/5% 上限', prev_info=True)
    comp('E2b+E3+E4 合併', extra=(0.002, 0.004), minfee=True, nosecond=True, prev_info=True)
    # 第二次處置在 d1 新增上櫃交易中的占比
    for per, (a, b) in PER.items():
        _, tr = sleeve(DL.events(markets=('TWSE', 'TPEX')), a, b)
        g = tr.groupby(['mkt', 'second']).ret.agg(['count', 'mean'])
        print('second 分組', per, g.to_dict(), flush=True)


def stress_spec():
    """規格寫了處置腿壓力來回 2%，研究員 STEP5_2 沒報；這裡補：買 0.8%、賣 1.2%。另報 E2b+E3（不含對 d1 有利的 E4）。"""
    for per, (a, b) in PER.items():
        n0, _, _ = DL.sleeve(DL.events(markets=('TWSE',)), a, b, cost=(0.008, 0.012))
        n1, _, _ = DL.sleeve(DL.events(markets=('TWSE', 'TPEX')), a, b, cost=(0.008, 0.012))
        x0, x1 = mx(n0), mx(n1)
        print('STRESS2%', per, dict(ann0=round(ann(n0), 1), ann1=round(ann(n1), 1), t0=round(L.tstat(x0), 2), t1=round(L.tstat(x1), 2),
                                     t_diff=round(L.tstat((x1 - x0).dropna()), 2), ann_0050=round(ann(E50.loc[a:b] / E50.loc[a:b].iloc[0]), 1)), flush=True)
    comp('E2b+E3（不含 E4）', extra=(0.002, 0.004), minfee=True, nosecond=True)

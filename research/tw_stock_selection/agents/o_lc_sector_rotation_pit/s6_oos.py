#!/usr/bin/env python3
"""凍結後：驗證期 VAL（錨定月 2017-01～2026-10）、正式全期 FORMAL（2014-01～2026-10，PIT）、描述段 DESC（2006-04～2013-12，非 PIT 市值）。
一開始先斷言 PREREG 存在且雜湊相符，才讀任何 2017 年以後或 2014 年以前的報酬。紙上引擎（tcore.rates 成本）；可執行版見 s7_exec.py。"""
import sys, json, time, math
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lclib as X
import os
DRY = os.environ.get('LC_DRY') == '1'     # 凍結前的程式除錯：只用挑選期、輸出到 scratch，不碰 results.json
if DRY:
    _it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
    PRE = dict(chosen=dict(cfg=_it['chosen']['cfg'], params=_it['versions'][_it['chosen']['version']]['params']), default=dict(cfg='COMP|k5|A30|M'),
               placebo_persistence=json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))['persistence_jmed'], code_sha256={}, data_sha256={})
    RES = dict(PREREG=PRE)
    RESF = X.SCR / f'dry_{Path(__file__).stem}.json'
    if RESF.exists(): RES.update(json.loads(RESF.read_text(encoding='utf-8')))
    for f in ('dry_s6_oos.json',):
        if (X.SCR / f).exists(): RES.update({k: v for k, v in json.loads((X.SCR / f).read_text(encoding='utf-8')).items() if k == 'OOS'})
else:
    RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    PRE = RES['PREREG']
    RESF = HERE / 'results.json'
    assert X.canon(PRE) == RES['PREREG_sha256'], 'PREREG 雜湊不符'
    for f, h in PRE['code_sha256'].items(): assert X.sha(HERE / f) == h, f'{f} 與 PREREG 不符'
    for f, h in PRE['data_sha256'].items(): assert X.sha(HERE / f) == h, f'{f} 與 PREREG 不符'
import lcrun as R
L, TC = X.L, X.TC
SEL = PRE['chosen']['cfg']; DEF = PRE['default']['cfg']
P = PRE['chosen']['params']; assert P == R.P0, '本題選定版是 V0；若不是，需另跑選定版參數'
PERIODS = dict(FORMAL=('2014-01', '2026-10', '2026-11'), DESC=('2006-04', '2013-12', '2014-01'))
if DRY: PERIODS = dict(FORMAL=('2014-01', '2016-12', '2017-01'), DESC=('2014-01', '2015-06', '2015-07'))
MASKS = dict(VAL=('2017-01-01', '2026-12-31'), FORMAL=('2014-01-01', '2026-12-31'), Y19=('2019-01-01', '2026-12-31'), SEG2=('2014-01-01', '2018-12-31'),
             SEG3=('2019-01-01', '2026-12-31'), DESC=('2006-04-01', '2013-12-31'), SEG1=('2006-04-01', '2012-12-31'))
if DRY: MASKS = {k: ('2014-01-01', '2016-12-31') for k in MASKS}
AI_SECT = ['半導體業', '電腦及週邊設備業', '電子零組件業', '其他電子業', '通信網路業']
R.init()

def msk(idx, k):
    a, b = MASKS[k]; return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))

def job(args):
    j, per = args
    m0, m1, me = PERIODS[per]
    B = R.build(j, m0, m1, me, P=P, keep_cs=True)
    days = B['days']; ti = [X.IDX.get_loc(d) for d in days]
    res = {}
    for cost in ('normal', 'zero', 'stress'):
        for nm in ['B'] + list(B['plans']):
            pl = B['Bplan'] if nm == 'B' else B['plans'][nm]
            dly = cost == 'normal' and per == 'FORMAL'
            s, inf = X.run_paper(days, pl, cost, daily=dly)
            res[(nm, cost)] = dict(s=s, turnover=inf['turnover'], cost=inf['cost'], nav=inf.get('nav'))
    e50 = X.bench_win(days)
    # 產業層級紀錄（選定與預設）：Brinson、右尾、離散度、產業權重
    sec = {}
    for k, d in enumerate(days[:-1]):
        cs = B['rec'][d]['cs']; ST = cs['ST']
        sr, rb = R.sector_window_returns(cs, ti[k], ti[k + 1])
        r = dict(rb=rb, disp=float(ST['REV3'].std()) if ST['REV3'].notna().sum() > 3 else np.nan, sr=sr, B={s: float(ST.loc[s, 'B'] / cs['mc'].sum()) for s in ST.index},
                 r2330=float(TC.factor_close_to_close(ti[k], ti[k + 1])[X.J2330] - 1), w2330=B['rec'][d]['w2330'])
        for nm in (SEL, DEF):
            pl = B['plans'][nm][d]
            if pl is None: r[nm] = None; continue
            U, w = pl
            r[nm] = {s: float(w[(cs['ind'] == s) & (U != X.J2330) & ~cs['fin']].sum()) for s in ST.index}
            r[nm + '|top'] = B['rec'][d]['secw'][nm]['top']
        sec[d] = r
    last = None
    if per == 'FORMAL':
        d = days[-2]; cs = B['rec'][d]['cs']
        last = dict(d=str(d.date()), ST=cs['ST'].round(4).reset_index().to_dict('records'), U=[X.COLS[i] for i in cs['U']], b=cs['b'].tolist(),
                    w_sel=B['plans'][SEL][d][1].tolist() if B['plans'][SEL][d] is not None else None, ind=list(cs['ind']), top=B['rec'][d]['secw'].get(SEL, {}).get('top'))
    return j, per, res, e50, sec, days, last

def ex_stats(x, key):
    x = x[msk(x.index, key)].dropna()
    s = X.stats(x); s['t_NW'] = X.nw_t(x); return s

def drop_best3(x):
    x = x.dropna(); yr = x.groupby(x.index.year).sum()
    best = yr.sort_values(ascending=False).index[:3]
    return X.tstat(x[~x.index.year.isin(best)]), [int(b) for b in best]

def jmed(lst, k): return float(np.median([d[k] for d in lst]))
def jp10(lst, k): return float(np.percentile([d[k] for d in lst], 10))

if __name__ == '__main__':
    t0 = time.time()
    with Pool(4) as p: out = p.map(job, [(j, per) for per in PERIODS for j in range(13)])
    print('runs done', round(time.time() - t0), flush=True)
    F = {j: o for o in out if o[1] == 'FORMAL' for j in [o[0]]}
    Dd = {j: o for o in out if o[1] == 'DESC' for j in [o[0]]}
    R_ = {}
    names = [n for n in F[0][2] if n[1] == 'normal']
    # ---------- 主要統計（選定、預設、B）----------
    main = {}
    for nm in (SEL, DEF, 'B'):
        for cost in ('normal', 'zero', 'stress'):
            for bench in ('0050', 'B'):
                if nm == 'B' and bench == 'B': continue
                for per in ('VAL', 'FORMAL', 'Y19', 'SEG2', 'SEG3', 'DESC', 'SEG1'):
                    src = Dd if per in ('DESC', 'SEG1') else F
                    lst = []
                    for j in range(13):
                        _, _, res, e50, _, _, _ = src[j]
                        s = res[(nm, cost)]['s']; b = e50 if bench == '0050' else res[('B', cost)]['s']
                        x = (s - b)
                        st = ex_stats(x, per)
                        xs = s[msk(s.index, per)]; bs = b[msk(b.index, per)]
                        st['cagr'] = X.cagr(xs); st['cagr_bench'] = X.cagr(bs); st['ann_ex_pp'] = (st['cagr'] - st['cagr_bench']) * 100
                        if per in ('FORMAL', 'VAL', 'DESC'):
                            st['t_drop3'], st['best3'] = drop_best3(x[msk(x.index, per)])
                        lst.append(st)
                    key = f'{nm}|{cost}|vs{bench}|{per}'
                    main[key] = dict(jmed={k: jmed(lst, k) for k in ('mean_pp', 't', 't_NW', 'IR', 'TE', 'cagr', 'cagr_bench', 'ann_ex_pp')},
                                     p10={k: jp10(lst, k) for k in ('mean_pp', 't', 'IR')},
                                     t_drop3_jmed=float(np.median([d['t_drop3'] for d in lst])) if 't_drop3' in lst[0] else None,
                                     best3_j6=lst[6].get('best3'), n=lst[0]['n'])
    R_['main'] = main
    print('main done', flush=True)
    # ---------- 逐年（j 平均超額，對 0050 與對 B，一般成本）----------
    def javg(nm, cost, bench, src=F):
        xs = []
        for j in range(13):
            _, _, res, e50, _, days, _ = src[j]
            s = res[(nm, cost)]['s']; b = e50 if bench == '0050' else res[('B', cost)]['s']
            x = (s - b); x.index = pd.period_range(PERIODS['FORMAL' if src is F else 'DESC'][0], periods=len(x), freq='M'); xs.append(x)   # 以錨定月對齊（2 月農曆年後 j 大時 d 可能落到 3 月）
        return pd.concat(xs, axis=1).mean(axis=1)
    yearly = {}
    for nm in (SEL, DEF, 'B'):
        for bench in ('0050', 'B'):
            if nm == 'B' and bench == 'B': continue
            xa = pd.concat([javg(nm, 'normal', bench, Dd), javg(nm, 'normal', bench, F)])
            yearly[f'{nm}|vs{bench}'] = {int(y): round(float(v) * 100, 3) for y, v in xa.groupby(xa.index.year).sum().items()}
    R_['yearly_sum_pp'] = yearly
    # ---------- 年化與回撤（2019 起與 0050 並列；j 平均日淨值）----------
    def javg_nav(nm):
        rs = [F[j][2][(nm, 'normal')]['nav'].pct_change() for j in range(13)]
        r = pd.concat(rs, axis=1).mean(axis=1).fillna(0); return (1 + r).cumprod()
    e50d = L.E50
    nav_tbl = {}
    for nm in (SEL, DEF, 'B'):
        nv = javg_nav(nm)
        for a in ('2014-01-01', '2017-01-01', '2019-01-01'):
            v = nv[nv.index >= a]
            if len(v) < 20: continue
            e = e50d[(e50d.index >= v.index[0]) & (e50d.index <= v.index[-1])]
            yrs = (v.index[-1] - v.index[0]).days / 365.25
            nav_tbl[f'{nm}|from{a[:4]}'] = dict(cagr=float((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1), mdd=X.mdd_nav(v),
                                               cagr_0050=float((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1), mdd_0050=X.mdd_nav(e))
    R_['nav_daily_javg'] = nav_tbl
    # ---------- 40 格熱圖（VAL、FORMAL；一般成本 j 中位數）----------
    heat = {}
    for (nm, c) in names:
        if nm == 'B': continue
        for per in ('VAL', 'FORMAL'):
            lb = [ex_stats(F[j][2][(nm, 'normal')]['s'] - F[j][2][('B', 'normal')]['s'], per) for j in range(13)]
            l5 = [ex_stats(F[j][2][(nm, 'normal')]['s'] - F[j][3], per) for j in range(13)]
            heat[f'{nm}|{per}'] = dict(vsB_mean=jmed(lb, 'mean_pp'), vsB_IR=jmed(lb, 'IR'), vsB_t=jmed(lb, 't'), vs0050_mean=jmed(l5, 'mean_pp'), vs0050_t=jmed(l5, 't'))
    R_['heat'] = heat
    sel_sign = np.sign(heat[f'{SEL}|VAL']['vsB_mean'])
    nb = [k for k in heat if k.endswith('|VAL') and k.split('|')[0] == SEL.split('|')[0]]
    R_['neighborhood_flip'] = dict(sel_sign=float(sel_sign), same_signal_cells={k: heat[k]['vsB_mean'] for k in nb},
                                   n_flip_all40=int(sum(np.sign(heat[k]['vsB_mean']) != sel_sign for k in heat if k.endswith('|VAL'))))
    # ---------- 周轉、成本 ----------
    R_['turnover_cost'] = {nm: dict(turnover_m=float(np.median([F[j][2][(nm, 'normal')]['turnover'].mean() for j in range(13)])),
                                    cost_pp_m=float(np.median([F[j][2][(nm, 'normal')]['cost'].mean() * 100 for j in range(13)])))
                           for nm in (SEL, DEF, 'B')}
    # ---------- Brinson、右尾、機制（選定與預設；FORMAL 與 DESC；每 j 再取中位數）----------
    mech = {}
    for nm in (SEL, DEF):
        for per, src in (('FORMAL', F), ('DESC', Dd)):
            alloc_t, tot_t, rt_t, top10_t, al_m, wi_m = [], [], [], [], [], []
            disp_hi, disp_lo, post_bear, semi_c, y19c, seg2c = [], [], [], [], [], []
            for j in range(13):
                _, _, res, e50, sec, days, _ = src[j]
                g = res[(nm, 'zero')]['s'] - res[('B', 'zero')]['s']
                rows = []; heldw = None
                for d in days[:-1]:
                    r = sec[d]; w = r[nm] if r[nm] is not None else heldw
                    heldw = w
                    for s, bw in r['B'].items():
                        ww = w.get(s, bw) if w else bw
                        rows.append((d, s, (ww - bw) * (r['sr'][s] - r['rb'])))
                C = pd.DataFrame(rows, columns=['d', 's', 'c'])
                al = C.groupby('d').c.sum().reindex(g.index).fillna(0)
                within = g - al
                al_m.append(al.mean() * 100); wi_m.append(within.mean() * 100)
                alloc_t.append(X.tstat(al)); tot_t.append(X.tstat(g))
                Cn = C.copy(); yr = pd.DatetimeIndex(Cn.d).year
                rm = ((Cn.s == '航運業') & (yr == 2021)) | (Cn.s.isin(AI_SECT) & (yr == 2025))
                adj = g - Cn[rm].groupby('d').c.sum().reindex(g.index).fillna(0)
                rt_t.append(X.tstat(adj))
                top = Cn.reindex(Cn.c.sort_values(ascending=False).index).head(10)
                adj2 = g - top.groupby('d').c.sum().reindex(g.index).fillna(0)
                top10_t.append(X.tstat(adj2))
                dsp = pd.Series({d: sec[d]['disp'] for d in days[:-1]})
                q = dsp.quantile(2 / 3); disp_hi.append(g[dsp >= q].mean() * 100); disp_lo.append(g[dsp < q].mean() * 100)
                pb = g[(g.index >= '2020-04-01') & (g.index <= '2020-06-30') | (g.index >= '2022-11-01') & (g.index <= '2023-01-31') | (g.index >= '2009-03-01') & (g.index <= '2009-05-31')]
                post_bear.append(pb.mean() * 100 if len(pb) else np.nan)
                semi_c.append(C[C.s == '半導體業'].c.sum() / max(len(g), 1) * 100)
                y19c.append(g[g.index >= '2019-01-01'].mean() * 100 if (g.index >= '2019-01-01').any() else np.nan)
                seg2c.append(g[(g.index >= '2014-01-01') & (g.index < '2019-01-01')].mean() * 100 if ((g.index >= '2014-01-01') & (g.index < '2019-01-01')).any() else np.nan)
                if j == 6:
                    topc = top[['d', 's', 'c']].assign(d=lambda z: z.d.astype(str), c=lambda z: (z.c * 100).round(3)).to_dict('records')
                    secsum = (C.groupby('s').c.sum() / len(g) * 100).sort_values().round(4).to_dict()
            mech[f'{nm}|{per}'] = dict(gross_alloc_pp=float(np.median(al_m)), gross_within_pp=float(np.median(wi_m)), alloc_t=float(np.median(alloc_t)),
                                       gross_total_t=float(np.median(tot_t)), t_drop_2021ship_2025AI=float(np.median(rt_t)), t_drop_top10_sectormonths=float(np.median(top10_t)),
                                       disp_hi_pp=float(np.nanmedian(disp_hi)), disp_lo_pp=float(np.nanmedian(disp_lo)), post_bear_pp=float(np.nanmedian(post_bear)),
                                       semis_contrib_pp=float(np.median(semi_c)), gross_Y19_pp=float(np.nanmedian(y19c)), gross_SEG2_pp=float(np.nanmedian(seg2c)),
                                       top10_j6=topc, sector_contrib_j6=secsum)
    R_['mech'] = mech
    # P3 j 曲線（營收 vs 價格；毛超額對 B，VAL）
    jc = {}
    for nm in ('REV3|k5|A30|M', 'RS|k5|A30|M', 'COMP|k5|A30|M', SEL):
        jc[nm] = {j: float((F[j][2][(nm, 'zero')]['s'] - F[j][2][('B', 'zero')]['s'])[msk(F[j][2][(nm, 'zero')]['s'].index, 'VAL')].mean() * 100) for j in range(13)}
    R_['P3_jcurve_gross_vsB_VAL'] = jc
    R_['P3_REV3_j0_4_minus_j8_12'] = float(np.mean([jc['REV3|k5|A30|M'][j] for j in range(5)]) - np.mean([jc['REV3|k5|A30|M'][j] for j in range(8, 13)]))
    # ---------- P6：與主策略（V7X 可執行，100 萬 GROW，同 j）----------
    nav7 = pd.read_parquet(X.AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet')
    p6 = []; comb = {}
    for j in range(13):
        _, _, res, e50, sec, days, _ = F[j]
        mv = nav7[f'B10|J{j}|K5T|normal|100萬|GROW'].reindex(pd.DatetimeIndex(days)).ffill().to_numpy()
        mr = pd.Series(mv[1:] / mv[:-1] - 1, index=pd.DatetimeIndex(days[:-1]))
        mx = mr - e50; sx = res[(SEL, 'normal')]['s'] - e50
        r23 = pd.Series({d: sec[d]['r2330'] for d in days[:-1]}) - e50
        m19 = (sx.index >= '2019-01-01') & (r23 > 0)
        if DRY: m19 = (r23 > 0)
        def beta(y, x): return float(np.polyfit(x[m19], y[m19], 1)[0]) if m19.sum() >= 5 else float('nan')
        sxB = res[(SEL, 'normal')]['s'] - res[('B', 'normal')]['s']
        p6.append(dict(corr=float(np.corrcoef(sx[msk(sx.index, 'FORMAL')].dropna(), mx[msk(sx.index, 'FORMAL')].reindex(sx[msk(sx.index, 'FORMAL')].dropna().index).fillna(0))[0, 1]),
                       beta_sel=beta(sx, r23), beta_sel_vsB=beta(sxB, r23), beta_main=beta(mx.fillna(0), r23), n19=int(m19.sum())))
        for wm in (0.7, 0.5):
            c = wm * mx + (1 - wm) * sx
            comb.setdefault(wm, []).append((c, mx))
    R_['P6'] = dict(corr_jmed=float(np.median([d['corr'] for d in p6])), beta_sel_jmed=float(np.median([d['beta_sel'] for d in p6])),
                    beta_sel_vsB_jmed=float(np.median([d['beta_sel_vsB'] for d in p6])),
                    beta_main_jmed=float(np.median([d['beta_main'] for d in p6])), n_months=p6[0]['n19'])
    rng = np.random.default_rng(7)
    cb = {}
    for wm, lst in comb.items():
        for per in ('VAL', 'Y19'):
            irc = [X.stats(c[msk(c.index, per)])['IR'] for c, m in lst]; irm = [X.stats(m[msk(m.index, per)])['IR'] for c, m in lst]
            # 配對區塊 bootstrap（j=6 路徑，區塊 6）：IR(組合) − IR(主策略) 的 p
            c, m = lst[6]; c = c[msk(c.index, per)].dropna(); m = m.reindex(c.index).fillna(0); n = len(c); diffs = []
            for _ in range(2000 if n >= 12 else 0):
                st = rng.integers(0, n, n // 6 + 1); ix = np.concatenate([np.arange(s, s + 6) % n for s in st])[:n]
                cc, mm = c.to_numpy()[ix], m.to_numpy()[ix]
                diffs.append(cc.mean() / cc.std() - mm.mean() / mm.std())
            cb[f'main{int(wm*100)}_{per}'] = dict(IR_comb_jmed=float(np.median(irc)), IR_main_jmed=float(np.median(irm)), p_boot_improve=float(np.mean(np.array(diffs) <= 0)) if diffs else None)
    R_['combo'] = cb
    # ---------- 2006～2012 描述：2007-07 改制前後 ----------
    if True:
        lst = []
        for j in range(13):
            _, _, res, e50, sec, days, _ = Dd[j]
            g = res[(SEL, 'zero')]['s'] - res[('B', 'zero')]['s']
            lst.append((g[g.index < '2007-07-01'].mean() * 100, g[(g.index >= '2007-07-01') & (g.index < '2008-07-01')].mean() * 100))
        R_['desc_2007_break'] = dict(pre_200707_pp=float(np.median([a for a, b in lst])), post_12m_pp=float(np.median([b for a, b in lst])))
    R_['last_signal'] = F[5][6]
    R_['last_signal_j'] = 5
    # j 平均日淨值存 scratch（G3 與報告用）
    navs = {}
    for (nm, c) in names:
        rs = [F[j][2][(nm, 'normal')]['nav'].pct_change() for j in range(13)]
        navs[nm] = (1 + pd.concat(rs, axis=1).mean(axis=1).fillna(0)).cumprod()
    pd.DataFrame(navs).to_parquet(X.SCR / 'navs_javg_formal.parquet')
    # 每 j 窗報酬（選定、預設、B）存 data（小）
    wr = {}
    for j in range(13):
        for nm in (SEL, DEF, 'B'):
            for cost in ('normal', 'zero'):
                s = F[j][2][(nm, cost)]['s']; wr[f'{nm}|{cost}|J{j}'] = s
        wr[f'0050|J{j}'] = F[j][3]
    pd.DataFrame(wr).to_parquet(X.DATA / 'oos_windows.parquet')
    RES = json.loads(RESF.read_text(encoding='utf-8')) if RESF.exists() else {}
    RES['OOS'] = R_
    RESF.write_text(X.dumps(RES), encoding='utf-8')
    for k in (f'{SEL}|normal|vs0050|VAL', f'{SEL}|normal|vsB|VAL', f'{SEL}|normal|vs0050|FORMAL', f'{SEL}|normal|vs0050|Y19', f'{DEF}|normal|vs0050|VAL', 'B|normal|vs0050|VAL'):
        print(k, {a: round(b, 3) for a, b in main[k]['jmed'].items()}, 'p10', {a: round(b, 3) for a, b in main[k]['p10'].items()})
    print('time', round(time.time() - t0))

#!/usr/bin/env python3
"""第 1 步：探索（只用挑選期：價格類 2006-04～2019-04；處置與法人 2021-10～2023-12）。輸出 explore.json。
E1（P1）流動性配對價差：上櫃 vs 上市，同 V60 五分位桶內「分數前 1/5 − 桶內等權」。
E2（P2）上櫃池法人活動三分位（2021-10～2023-12，inst_5y 淨買賣張數絕對值 ÷ 成交量）。
E3 j 剖面（零衝擊與 K5T 100 萬 GROW）：上市 n10 前 300、只上櫃 n10 前 300、合併 n10 前 400。
E4 容量：入選股 V60、預測開盤量、各級距參與率與 K5T 衝擊估計。
E5 處置事件（2021-10～2023-12）：上市 vs 上櫃、第一次 vs 重複處置，事件日分群 t。
E6（P7）上櫃衛星超額與上市主策略超額的相關。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import otlib as L
D, X = L.D, L.X
SELA, SELB = pd.Timestamp('2006-04-01'), pd.Timestamp('2019-04-30')
DA, DB = pd.Timestamp('2021-10-01'), pd.Timestamp('2023-12-31')
OUT = {}


def elig(sig, ex):
    y3 = L.yoy3_at(sig)
    ok = np.isfinite(D.V60[sig]) & (D.V60[sig] >= 1e7) & (D.C[sig] >= 10) & D.TR[sig] & ~L.EXCL & ~D.DISP[sig] & ~D.DISP[ex] \
        & np.isfinite(D.O[ex]) & np.isfinite(D.S52[sig]) & np.isfinite(y3)
    return ok, y3


AO = D.AI * D.O / np.where(np.isfinite(D.C) & (D.C > 0), D.C, np.nan)
AIf = pd.DataFrame(np.where(D.TR, D.AI, np.nan)).ffill().to_numpy()


def fwd(e1, e2, idx):
    a = AO[e1, idx]; b = np.where(np.isfinite(AO[e2, idx]), AO[e2, idx], AIf[e2, idx])
    return b / a - 1


def e1_p1(lo, hi):
    S = L.SIGS[0]; rows = []
    for k in range(len(S) - 1):
        sig, ex = S[k]; ex2 = S[k + 1][1]
        d = D.IDX[sig]
        if d < lo or d > hi or str(pd.Period(d, 'M')) in L.hole_months(): continue
        ok, y3 = elig(sig, ex)
        w = np.where(ok)[0]
        if (D.MK[sig, w]).sum() < 50: continue
        r52 = pd.Series(D.S52[sig, w]).rank(pct=True).to_numpy(); rr = pd.Series(y3[w]).rank(pct=True).to_numpy()
        sc = 0.5 * r52 + 0.5 * rr
        q = pd.qcut(pd.Series(D.V60[sig, w]).rank(method='first'), 5, labels=False).to_numpy()
        fr = fwd(ex, ex2, w)
        diffs = []; per = {}
        for b in range(5):
            res = {}
            for m in (0, 1):
                sel = (q == b) & (D.MK[sig, w] == bool(m)) & np.isfinite(fr)
                if sel.sum() < 10: continue
                s_ = sc[sel]; f_ = fr[sel]; thr = np.quantile(s_, 0.8)
                res[m] = float(f_[s_ >= thr].mean() - f_.mean())
            if 0 in res and 1 in res:
                diffs.append(res[1] - res[0]); per[b] = (res[0], res[1])
        if diffs:
            rows.append(dict(date=d, diff=float(np.mean(diffs)), n_b=len(diffs), **{f'tw_b{b}': v[0] for b, v in per.items()}, **{f'tp_b{b}': v[1] for b, v in per.items()}))
    df = pd.DataFrame(rows).set_index('date')
    out = dict(n_months=len(df), mean_diff_pp=float(df['diff'].mean() * 100), t=L.tstat(df['diff']), t_nw=L.nw_t(df['diff']))
    for b in range(5):
        if f'tw_b{b}' in df:
            out[f'bucket{b}'] = dict(tw_spread_pp=float(df[f'tw_b{b}'].mean() * 100), tp_spread_pp=float(df[f'tp_b{b}'].mean() * 100),
                                     t_tw=L.tstat(df[f'tw_b{b}']), t_tp=L.tstat(df[f'tp_b{b}']), t_diff=L.tstat(df[f'tp_b{b}'] - df[f'tw_b{b}']), n=int(df[f'tw_b{b}'].notna().sum()))
    out['S1a_2008_2012'] = dict(mean_diff_pp=float(df.loc['2008':'2012', 'diff'].mean() * 100), t=L.tstat(df.loc['2008':'2012', 'diff']))
    out['2013_2019'] = dict(mean_diff_pp=float(df.loc['2013':'2019', 'diff'].mean() * 100), t=L.tstat(df.loc['2013':'2019', 'diff']))
    return out


def e2_p2():
    inst = pd.read_csv(L.EXT / 'extras' / 'inst_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    inst = inst[inst.code.isin(D.CI)]
    for c in ('foreign', 'trust', 'dealer'): inst[c] = pd.to_numeric(inst[c], errors='coerce').abs()
    act = inst.assign(a=inst.foreign + inst.trust + inst.dealer).pivot_table(index='date', columns='code', values='a', aggfunc='sum')
    tru = inst.pivot_table(index='date', columns='code', values='trust', aggfunc='sum')
    act = act.reindex(index=D.IDX, columns=D.COLS); tru = tru.reindex(index=D.IDX, columns=D.COLS)
    vol = pd.DataFrame(D.V / D.C / 1000.0, index=D.IDX, columns=D.COLS)       # 張
    share = act.rolling(60, min_periods=40).sum() / vol.rolling(60, min_periods=40).sum()
    tz = tru.fillna(0).rolling(60, min_periods=40).sum()
    S = L.SIGS[0]; rows = []
    for k in range(len(S) - 1):
        sig, ex = S[k]; ex2 = S[k + 1][1]; d = D.IDX[sig]
        if d < DA or d > DB or str(pd.Period(d, 'M')) in L.hole_months(): continue
        ok, y3 = elig(sig, ex)
        w = np.where(ok & D.MK[sig])[0]
        w = w[np.argsort(-D.V60[sig, w])][:300]
        sh = share.iloc[sig].to_numpy()[w]; fr = fwd(ex, ex2, w)
        g = np.isfinite(sh) & np.isfinite(fr)
        w, sh, fr = w[g], sh[g], fr[g]
        if len(w) < 60: continue
        sc = 0.5 * pd.Series(D.S52[sig, w]).rank(pct=True).to_numpy() + 0.5 * pd.Series(y3[w]).rank(pct=True).to_numpy()
        ter = pd.qcut(pd.Series(sh).rank(method='first'), 3, labels=False).to_numpy()
        r = dict(date=d)
        for tt in range(3):
            s_ = sc[ter == tt]; f_ = fr[ter == tt]; thr = np.quantile(s_, 0.8)
            r[f't{tt}'] = float(f_[s_ >= thr].mean() - f_.mean())
        z = tz.iloc[sig].to_numpy()[w] == 0
        if z.sum() >= 15:
            thr = np.quantile(sc[z], 0.8); r['trust0'] = float(fr[z][sc[z] >= thr].mean() - fr[z].mean())
        rows.append(r)
    df = pd.DataFrame(rows).set_index('date')
    out = {f'tercile{t}_spread_pp': float(df[f't{t}'].mean() * 100) for t in range(3)}
    out['low_minus_high_pp'] = float((df.t0 - df.t2).mean() * 100); out['t_low_minus_high'] = L.tstat(df.t0 - df.t2)
    if 'trust0' in df: out['trust_zero60_spread_pp'] = float(df.trust0.mean() * 100); out['t_trust0'] = L.tstat(df.trust0)
    out['n_months'] = len(df); out['note'] = 'tercile0＝法人活動占比最低；inst_5y 只有淨買賣（張），以 |外資|+|投信|+|自營| 淨額 60 日合計 ÷ 60 日成交張數近似成交占比'
    return out


VARS0 = {'TW300': dict(pool='TW', K=300, n=10), 'TP300': dict(pool='TP', K=300, n=10), 'ALL400': dict(pool='ALL', K=400, n=10)}


def e3_jprofile():
    out = {}; navs = {}
    for nm, v in VARS0.items():
        rows = {}
        for j in range(13):
            pl = L.gen_plan(v, j)
            iw = L.ideal_windows(pl, j)
            b = L.windows(pd.Series(D.E50, index=D.IDX), j).s
            ix = iw.index[(iw.index >= SELA) & (iw.index <= SELB)]
            xi = (iw.reindex(ix) - b.reindex(ix)).dropna()
            t1 = int(np.searchsorted(D.IDX, SELB + pd.Timedelta(days=40)))
            nav, acc = L.sim(pl, L.cfg('100萬', 'GROW'), t1=t1)
            W = L.windows(nav, j); s = L.seg_stats(W, SELA, SELB)
            rows[j] = dict(ideal_x_pp=float(xi.mean() * 100), ideal_t=L.tstat(xi), k5t_ann_x=s.get('ann_x'), k5t_t=s.get('t'), part_median=acc['part_median'])
            navs[(nm, j)] = nav
        out[nm] = dict(by_j=rows, ideal_x_median=float(np.median([r['ideal_x_pp'] for r in rows.values()])),
                       k5t_ann_x_median=float(np.median([r['k5t_ann_x'] for r in rows.values()])), k5t_t_median=float(np.median([r['k5t_t'] for r in rows.values()])),
                       slope_ideal_x_per_j=float(np.polyfit(range(13), [rows[j]['ideal_x_pp'] for j in range(13)], 1)[0]))
    # E6：上櫃衛星超額與上市主策略超額相關（K5T 100 萬 GROW，窗超額）
    cors = []
    for j in range(13):
        a = L.windows(navs[('TP300', j)], j).loc[SELA:SELB].x; b = L.windows(navs[('TW300', j)], j).loc[SELA:SELB].x
        cors.append(float(a.corr(b)))
    return out, dict(corr_TPsat_vs_TWmain_excess_j_median=float(np.median(cors)), by_j=cors)


def e4_capacity():
    out = {}
    for nm, v in VARS0.items():
        pl, dg = L.gen_plan(v, 0, return_diag=True)
        vv = []; tp = []; part = {t: [] for t in ('100萬', '300萬', '1000萬')}; imp = {t: [] for t in part}
        for (ex, pk, sg) in pl:
            if not (SELA <= D.IDX[sg] <= SELB): continue
            for jj in pk:
                v60 = D.V60p[ex, jj]
                if not np.isfinite(v60): continue
                vv.append(v60); tp.append(bool(D.MK[sg, jj]))
                oa = X.OASH[ex, jj] * v60; sg_ = D.SIGp[ex, jj]
                for t in part:
                    q = L.TIERS[t] * D.cpi_scale(D.IDX[ex].year) / v['n']
                    part[t].append(q / oa)
                    imp[t].append(0.001 + 0.5 * sg_ * math.sqrt(q / oa) if np.isfinite(sg_) else np.nan)
        vv = np.array(vv); tp = np.array(tp)
        o = dict(n=int(len(vv)), share_tpex_picks=float(tp.mean()), v60_median_all=float(np.median(vv)) / 1e8,
                 v60_median_tpex=float(np.median(vv[tp])) / 1e8 if tp.any() else None, v60_p10_tpex=float(np.percentile(vv[tp], 10)) / 1e8 if tp.any() else None,
                 v60_median_twse=float(np.median(vv[~tp])) / 1e8 if (~tp).any() else None)
        for t in part:
            pa = np.array(part[t]); im = np.array(imp[t])
            o[f'part_{t}_median'] = float(np.nanmedian(pa)); o[f'part_{t}_p90'] = float(np.nanpercentile(pa, 90))
            o[f'k5t_imp_{t}_median_pct'] = float(np.nanmedian(im) * 100)
            if tp.any():
                o[f'part_{t}_median_tpex'] = float(np.nanmedian(pa[tp])); o[f'k5t_imp_{t}_median_tpex_pct'] = float(np.nanmedian(im[tp]) * 100)
        out[nm] = o
    out['unit'] = 'V60 億元；part＝單檔下單金額 ÷ 事前預測開盤量（share_dec×V60）；K5T 衝擊為每邊％'
    return out


def disposal_events(a, b):
    ds = pd.read_csv(L.EXT / 'extras' / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    ds = ds[ds.code.isin(D.CI)].sort_values(['code', 'start']).drop_duplicates(['code', 'start'])
    tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close']
    ma = tx.rolling(200).mean(); bull = (tx >= ma).reindex(D.IDX).ffill().shift(1).fillna(False).to_numpy()
    rows = []; prev_end = {}
    for c, s, e in zip(ds.code, ds.start, ds.end):
        j = D.CI[c]
        i0 = int(np.searchsorted(D.DATES, np.datetime64(s))); i1 = int(np.searchsorted(D.DATES, np.datetime64(e), side='right')) - 1
        pe = prev_end.get(c); second = pe is not None and 0 <= i0 - pe <= 30
        prev_end[c] = i1
        if not (a <= s <= b) or i0 >= len(D.IDX) or i1 <= i0: continue
        if not (D.TR[i0, j] and D.TR[i1, j]): rows.append(dict(code=c, start=s, mkt='TPEX' if D.MK[i0, j] else 'TWSE', ok=False, why='no_price')); continue
        r0 = D.R[i0, j]
        v60 = D.V60p[i0, j]
        why = None
        if not bull[i0]: why = 'bear'
        elif r0 >= 0.09: why = 'limitup_entry'
        elif not (np.isfinite(v60) and v60 >= 1e7 and D.C[i0 - 1, j] >= 10): why = 'illiquid'
        # 出場日跌停（收盤跌幅 ≤ −9%）延到第一個非跌停日
        k = i1
        while k < len(D.IDX) - 1 and np.isfinite(D.R[k, j]) and D.R[k, j] <= -0.09: k += 1
        raw = AIf[k, j] / AIf[i0, j] - 1
        rows.append(dict(code=c, start=s, i0=i0, i1=k, mkt='TPEX' if D.MK[i0, j] else 'TWSE', second=bool(second), raw=float(raw), ok=why is None, why=why or ''))
    return pd.DataFrame(rows)


def ev_stats(df, cost):
    x = df.raw - cost
    g = x.groupby(df.start).mean()
    xs = np.sort(x.to_numpy()); trim = xs[: int(len(xs) * 0.95)]
    return dict(n=int(len(x)), mean_pct=float(x.mean() * 100), median_pct=float(np.median(x) * 100), t_cluster=L.tstat(g), n_dates=int(len(g)),
                mean_trim95_pct=float(trim.mean() * 100) if len(trim) else None, median_trim95_pct=float(np.median(trim) * 100) if len(trim) else None, win=float((x > 0).mean()))


def e5_disposal():
    ev = disposal_events(DA, DB)
    ok = ev[ev.ok == True]
    out = {'n_all': int(len(ev)), 'drop_reasons': ev[ev.ok != True].groupby(['mkt', 'why']).size().to_dict() if len(ev) else {}}
    out = {k: ({str(kk): int(vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in out.items()}
    for cost, lab in ((0.008, 'normal'), (0.02, 'stress')):
        for m in ('TWSE', 'TPEX'):
            sub = ok[ok.mkt == m]
            if len(sub) >= 5:
                out[f'{m}|{lab}'] = ev_stats(sub, cost)
                for sec in (False, True):
                    s2 = sub[sub.second == sec]
                    if len(s2) >= 5: out[f'{m}|{lab}|{"repeat" if sec else "first"}'] = ev_stats(s2, cost)
        # 同窗格配對：同一起始月內上櫃－上市
        mm = ok.assign(m=ok.start.dt.to_period('M'), x=ok.raw - cost).groupby(['m', 'mkt']).x.mean().unstack()
        if 'TPEX' in mm and 'TWSE' in mm:
            dd = (mm.TPEX - mm.TWSE).dropna(); out[f'TPEX_minus_TWSE_month_paired|{lab}'] = dict(mean_pct=float(dd.mean() * 100), t=L.tstat(dd), n_months=int(len(dd)))
    return out


if __name__ == '__main__':
    OUT['E1_P1_liquidity_matched'] = e1_p1(SELA, SELB); print('E1', OUT['E1_P1_liquidity_matched'], flush=True)
    OUT['E2_P2_inst_terciles'] = e2_p2(); print('E2', OUT['E2_P2_inst_terciles'], flush=True)
    OUT['E3_jprofile_SEL'], OUT['E6_P7_corr'] = e3_jprofile(); print('E3', {k: {kk: vv for kk, vv in v.items() if kk != 'by_j'} for k, v in OUT['E3_jprofile_SEL'].items()}, OUT['E6_P7_corr']['corr_TPsat_vs_TWmain_excess_j_median'], flush=True)
    OUT['E4_capacity_SEL'] = e4_capacity(); print('E4', OUT['E4_capacity_SEL'], flush=True)
    OUT['E5_disposal_2021_10_2023_12'] = e5_disposal(); print('E5', OUT['E5_disposal_2021_10_2023_12'], flush=True)
    OUT['scope'] = '挑選期：價格類 2006-04～2019-04 訊號日；處置與法人 2021-10～2023-12。資料洞月份（gap_map hole）排除。未看驗證期。'
    (HERE / 'explore.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))

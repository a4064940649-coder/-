#!/usr/bin/env python3
"""反方統計審查 o9（不修改研究員檔案；只讀 out/*.npz 與 results.json；結果印到 stdout 並寫 out_review_o9.json 於 scratch）。
python3 -I -B review_stat_o9.py [A|B|AB]
A：npz 事後分析（Brinson 層別配置 vs 層內選股、平均市值名次、去前 k 大貢獻股／去航運、毛口徑分段）。
B：特徵配對安慰劑（重跑引擎，版本 a、MAIN）：
   N_size：實際分數在「市值名次相鄰 10 檔」合格股內隨機重排（保留每個名次位置的規模組成，只拿掉層內選股）；
   N_ind ：實際分數在同產業合格股內隨機重排（保留產業配置，只拿掉產業內選股）；
   N_rand：原隨機分數（對照，確認與 s3 一致）。
   一般成本與零成本各跑。
"""
import sys, json, math, time, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from s4_stats import load, tstat, JK
import pcore_lite as PL

MODE = sys.argv[1] if len(sys.argv) > 1 else 'AB'
OUTF = Path(os.environ.get('O9OUT', '/tmp/review_o9.json'))
RES = json.loads(OUTF.read_text()) if OUTF.exists() else {}
def save(): OUTF.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=float))

codes = PL.COLS
IND = np.array([PL.IND.get(c, '?') for c in codes])

def partA():
    O = {}
    for cfg in ['a', 'b']:
        D = load(f'run_{cfg}_MAIN')
        Z = load(f'run_{cfg}_MAIN_zero')
        o = {}
        # --- 1. Brinson：層 = 訊號日市值名次 ---
        tiers = [(1, 50), (51, 100), (101, 150), (151, 200), (201, 300)]
        alloc_t, sel_t, alloc_m, sel_m, tot_m = [], [], [], [], []
        rk_sat_a, rk_sat_r = [], []
        for k in JK:
            d = D[k]
            Wa = d['Wa'].astype(float); Wr = d['Wr'].astype(float); F = np.nan_to_num(d['F'].astype(float), nan=1.0) - 1
            rk = d['rk']; Rrep = d['Rrep'].astype(float)
            al = np.zeros(len(Wa)); se = np.zeros(len(Wa))
            for i in range(len(Wa)):
                rR_tot = (Wr[i] * F[i]).sum() / Wr[i].sum()
                for a, b in tiers + [(301, 10 ** 9)]:
                    m = (rk[i] >= a) & (rk[i] <= b) if b < 10 ** 9 else ((rk[i] > 300) | (rk[i] == 0))
                    wa, wr = Wa[i, m].sum(), Wr[i, m].sum()
                    ra = (Wa[i, m] * F[i, m]).sum() / wa if wa > 1e-12 else 0.0
                    rr = (Wr[i, m] * F[i, m]).sum() / wr if wr > 1e-12 else 0.0
                    al[i] += (wa - wr) * (rr - rR_tot)
                    se[i] += wa * (ra - rr) if wr > 1e-12 else wa * (ra - rR_tot)
            alloc_t.append(tstat(al)); sel_t.append(tstat(se)); alloc_m.append(al.mean() * 100); sel_m.append(se.mean() * 100)
            tot_m.append((d['ra'] - d['Rr'].mean(axis=0)).mean() * 100)
            # 衛星（超出複製權重的部分）的權重加權平均市值名次
            sa = np.clip(Wa - Rrep, 0, None); sr = np.clip(Wr - Rrep, 0, None)
            rkf = np.where(rk == 0, np.nan, rk).astype(float)
            rk_sat_a.append(np.nanmean([np.nansum(sa[i] * rkf[i]) / sa[i][np.isfinite(rkf[i])].sum() for i in range(len(sa))]))
            rk_sat_r.append(np.nanmean([np.nansum(sr[i] * rkf[i]) / sr[i][np.isfinite(rkf[i])].sum() for i in range(len(sr))]))
        o['brinson_tiers'] = [list(x) for x in tiers]
        o['brinson_alloc_mean_pp_jmed'] = float(np.median(alloc_m)); o['brinson_alloc_t_jmed'] = float(np.median(alloc_t))
        o['brinson_sel_mean_pp_jmed'] = float(np.median(sel_m)); o['brinson_sel_t_jmed'] = float(np.median(sel_t))
        o['brinson_sel_t_p10'] = float(np.percentile(sel_t, 10))
        o['actual_minus_random_mean_pp_jmed'] = float(np.median(tot_m))
        o['sat_wavg_rank_actual_jmed'] = float(np.median(rk_sat_a)); o['sat_wavg_rank_random_jmed'] = float(np.median(rk_sat_r))
        # --- 2. 去前 k 大貢獻股（靜態歸因近似，含航運整個產業）---
        stock_tot = np.zeros(len(codes)); As = {}
        for k in JK:
            d = D[k]
            A = (d['Wa'].astype(float) - d['Wr'].astype(float)) * (np.nan_to_num(d['F'].astype(float), nan=1.0) - 1)
            As[k] = A; stock_tot += A.sum(axis=0) / len(A)
        order = np.argsort(-stock_tot)
        o['top10'] = [(codes[j], PL.NAME.get(codes[j]), PL.IND.get(codes[j]), round(float(stock_tot[j] / 13 * 100), 4)) for j in order[:10]]
        drop = {}
        for kk in [1, 3, 5, 10, 20]:
            S = order[:kk]; ts = []
            for k in JK:
                x = D[k]['ra'] - D[k]['Rr'].mean(axis=0); ts.append(tstat(x - As[k][:, S].sum(axis=1)))
            drop[f'top{kk}'] = float(np.median(ts))
        for nm, mask in {'航運業': IND == '航運業', '電子零組件業': IND == '電子零組件業', '航運+電子零組件+電腦週邊': np.isin(IND, ['航運業', '電子零組件業', '電腦及週邊設備業'])}.items():
            ts = []; ms = []
            for k in JK:
                x = D[k]['ra'] - D[k]['Rr'].mean(axis=0); y = x - As[k][:, mask].sum(axis=1); ts.append(tstat(y)); ms.append(y.mean() * 100)
            drop[f'ind_{nm}'] = (float(np.median(ms)), float(np.median(ts)))
        # 產業層 Brinson：產業配置 vs 產業內選股
        ia, isel = [], []
        uind = sorted(set(IND))
        for k in JK:
            d = D[k]; Wa = d['Wa'].astype(float); Wr = d['Wr'].astype(float); F = np.nan_to_num(d['F'].astype(float), nan=1.0) - 1
            al = np.zeros(len(Wa)); se = np.zeros(len(Wa))
            for i in range(len(Wa)):
                rtot = (Wr[i] * F[i]).sum() / Wr[i].sum()
                for g in uind:
                    m = IND == g; wa, wr = Wa[i, m].sum(), Wr[i, m].sum()
                    if wa < 1e-12 and wr < 1e-12: continue
                    ra = (Wa[i, m] * F[i, m]).sum() / wa if wa > 1e-12 else 0.0
                    rr = (Wr[i, m] * F[i, m]).sum() / wr if wr > 1e-12 else rtot
                    al[i] += (wa - wr) * (rr - rtot); se[i] += wa * (ra - rr)
            ia.append((al.mean() * 100, tstat(al))); isel.append((se.mean() * 100, tstat(se)))
        o['industry_brinson_alloc_jmed'] = (float(np.median([a for a, _ in ia])), float(np.median([b for _, b in ia])))
        o['industry_brinson_within_jmed'] = (float(np.median([a for a, _ in isel])), float(np.median([b for _, b in isel])))
        o['drop'] = drop
        # --- 3. 毛口徑分段 ---
        if Z is not None:
            for nm, (a, b) in {'2019-05_2022-12': ('2019-05-01', '2022-12-31'), '2023_2026': ('2023-01-01', '2026-12-31')}.items():
                ts = []; ms = []
                for k in JK:
                    sd = pd.DatetimeIndex(Z[k]['sd']); m = (sd >= a) & (sd <= b); x = (Z[k]['ra'] - Z[k]['Rr'].mean(axis=0))[m]
                    ts.append(tstat(x)); ms.append(x.mean() * 100)
                o[f'gross_{nm}'] = (float(np.median(ms)), float(np.median(ts)), float(np.percentile(ts, 10)))
            o['gross_j_t_all'] = [round(tstat(Z[k]['ra'] - Z[k]['Rr'].mean(axis=0)), 2) for k in JK]
            o['gross_frac_j_t_ge2'] = float(np.mean([tstat(Z[k]['ra'] - Z[k]['Rr'].mean(axis=0)) >= 2 for k in JK]))
        o['net_j_t_all'] = [round(tstat(D[k]['ra'] - D[k]['Rr'].mean(axis=0)), 2) for k in JK]
        # 2019-05～2022-12 淨口徑逐 j
        o['net_j_t_2019_2022'] = [round(tstat((D[k]['ra'] - D[k]['Rr'].mean(axis=0))[pd.DatetimeIndex(D[k]['sd']) <= '2022-12-31']), 2) for k in JK]
        # 對 0050 t 的 2019-05～2022-12
        o['vs0050_t_2019_2022_jmed'] = float(np.median([tstat((D[k]['ra'] - D[k]['r5'])[pd.DatetimeIndex(D[k]['sd']) <= '2022-12-31']) for k in JK]))
        o['vs0050_t_2023_2026_jmed'] = float(np.median([tstat((D[k]['ra'] - D[k]['r5'])[pd.DatetimeIndex(D[k]['sd']) >= '2023-01-01']) for k in JK]))
        O[cfg] = o
        print(cfg, json.dumps(o, ensure_ascii=False, default=float), flush=True)
    RES['A'] = O; save()

def partB(nseed=150):
    import pcore as P
    from multiprocessing import Pool
    T = P.setup('a'); L = P.L
    DS = P.daysets('MAIN')
    for k, (sd, de) in DS.items():
        for d in sd: T.cross_section(d)
    # 存活者檢查：前 300 名、其他條件都合格但營收缺值、之後下市者
    surv = []
    end_t = len(L.IDX) - 1
    for d in DS['OFF'][0]:
        cs = T.cross_section(d); t = cs['t']; e = cs['e']
        base = cs['capuni'] & (cs['rank'] <= 300) & ~L.FINMASK & ~L.FORMASK & np.isfinite(cs['s52']) & ~np.isfinite(cs['y3'])
        dl = base & (L.LAST_TRADE < end_t - 5)
        surv.append((str(d.date()), int(base.sum()), int(dl.sum()), [codes[j] for j in np.where(dl)[0]]))
    RES['B_surv_missing_rev_top300'] = dict(mean_missing=float(np.mean([s[1] for s in surv])), mean_missing_later_delisted=float(np.mean([s[2] for s in surv])),
                                            codes=sorted(set(c for s in surv for c in s[3])))
    # 下市且營收缺之名單在下一窗的報酬 vs 合格池平均
    sd, de = DS['OFF']; ei = [T.cross_section(d)['e'] for d in sd] + [L.IDX.get_loc(de)]
    gaps = []
    for i, d in enumerate(sd):
        cs = T.cross_section(d)
        base = cs['capuni'] & (cs['rank'] <= 300) & ~L.FINMASK & ~L.FORMASK & np.isfinite(cs['s52']) & ~np.isfinite(cs['y3']) & (L.LAST_TRADE < end_t - 5)
        if not base.any(): continue
        F = L.window_factors_all(ei[i], ei[i + 1], P.GX)
        fe = np.nanmean(F[cs['elig']]); fb = F[base]
        gaps.append((str(d.date()), float(np.nanmean(fb) - fe), int(base.sum())))
    RES['B_surv_window_gap'] = dict(n_windows=len(gaps), mean_gap_pp=float(np.mean([g[1] for g in gaps]) * 100) if gaps else None, rows=gaps[:60])
    print('surv', json.dumps(RES['B_surv_missing_rev_top300'], ensure_ascii=False), RES['B_surv_window_gap']['mean_gap_pp'], flush=True)
    save()

    def perm_size(rg):
        def f(cs):
            sc = cs['sc'].copy(); el = np.where(cs['elig'])[0]
            el = el[np.argsort(cs['rank'][el], kind='stable')]
            out = np.full(len(sc), np.nan)
            for g in range(0, len(el), 10):
                ix = el[g:g + 10]; out[ix] = sc[ix][rg.permutation(len(ix))]
            return out
        return f
    def perm_ind(rg):
        def f(cs):
            sc = cs['sc'].copy(); el = np.where(cs['elig'])[0]; out = np.full(len(sc), np.nan)
            for g in set(IND[el]):
                ix = el[IND[el] == g]; out[ix] = sc[ix][rg.permutation(len(ix))]
            return out
        return f
    NULLS = {'N_size': perm_size, 'N_ind': perm_ind, 'N_rand': P.rand_fn}
    keys = list(DS.keys())
    global _JOB
    def _job(a):
        nm, k, cost, s0, s1 = a
        sd, de = DS[k]; out = []
        for s in range(s0, s1):
            rg = np.random.default_rng(777000 + s)
            out.append(P.run_scores(sd, de, NULLS[nm](rg), cost=cost))
        return a, np.array(out)
    _JOB = _job
    jobs = [(nm, k, cost, s, min(s + 50, nseed)) for nm in NULLS for k in keys for cost in ['normal', 'zero'] for s in range(0, nseed, 50)]
    t0 = time.time()
    with Pool(4) as pool: res = pool.map(job, jobs, chunksize=1)
    acc = {}
    for a, arr in res: acc.setdefault(a[:3], []).append(arr)
    act = {(k, c): P.run_scores(DS[k][0], DS[k][1], None, cost=c) for k in keys for c in ['normal', 'zero']}
    O = {}
    for nm in NULLS:
        for cost in ['normal', 'zero']:
            ts, ms, t1, t2 = [], [], [], []
            for k in keys:
                Rr = np.vstack(acc[(nm, k, cost)]); x = act[(k, cost)] - Rr.mean(axis=0)
                sd = pd.DatetimeIndex(DS[k][0])
                if k == 'OFF':
                    O[f'{nm}|{cost}|OFF'] = (float(x.mean() * 100), tstat(x)); continue
                ts.append(tstat(x)); ms.append(x.mean() * 100)
                t1.append(tstat(x[sd <= '2022-12-31'])); t2.append(tstat(x[sd >= '2023-01-01']))
            O[f'{nm}|{cost}'] = dict(mean_pp_jmed=float(np.median(ms)), t_jmed=float(np.median(ts)), t_p10=float(np.percentile(ts, 10)), t_min=float(np.min(ts)),
                                     t_jmed_2019_2022=float(np.median(t1)), t_jmed_2023_2026=float(np.median(t2)))
            print(nm, cost, json.dumps(O[f'{nm}|{cost}'], default=float), O[f'{nm}|{cost}|OFF'], flush=True)
    O['nseed'] = nseed; O['runtime_s'] = time.time() - t0
    RES['B'] = O; save()

_JOB = None
def job(a): return _JOB(a)

if __name__ == '__main__':
    if 'A' in MODE: partA()
    if 'B' in MODE: partB()

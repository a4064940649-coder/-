#!/usr/bin/env python3
"""凍結後一次性樣本外與全期報告（先斷言 PREREG／PREREG_POWER／PREREG_FREEZE 與程式、資料雜湊）。
步驟 1（模擬）：依 PREREG_FREEZE.oos_plan 跑全期；20 seeds 平均淨值存 data/oos_navs.parquet、逐日成本追蹤存 data/oos_track.parquet。
步驟 2（統計）：主欄、各格、三段、拿掉最好 3 年、拿掉最大一窗、鄰域、配對差三層＋月 t＋bootstrap、成本前後、會計成本、周轉、下單日占比、兩平滑價、G3、G1～G8。
寫入 results.json['OOS']。python3 -I -B oos.py [sim|stats|all]"""
import sys, json, time, math, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; import run_lib as RL; sys.path.pop(0)
RES = HERE / 'results.json'
R = json.loads(RES.read_text(encoding='utf-8'))
for k in ('PREREG', 'PREREG_POWER', 'PREREG_FREEZE'):
    assert k in R and CL.canon(R[k]) == R[k + '_sha256'], f'{k} 缺或被改'
FZ = R['PREREG_FREEZE']
for f, h in FZ['code_sha256'].items():
    if f in ('oos.py', 'g3.py', 'post.py'): continue
    assert CL.sha(HERE / f) == h, f'{f} 與凍結雜湊不符'
for f, h in FZ['data_sha256'].items(): assert CL.sha(HERE / f) == h, f'{f} 與凍結雜湊不符'
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
CH = FZ['chosen']; CHO = FZ['chosen_opt']
OPTS = {k: v['opt'] for k, v in IT['versions'].items() if 'opt' in v}; OPTS['BASE'] = {}
V8 = ['CA80', 'CA90', 'MT', 'NR', 'DC5C', 'DC5E', 'CB1', 'CB2']
NAVF = HERE / 'data/oos_navs.parquet'; TRF = HERE / 'data/oos_track.parquet'
VAL = ('2015-01-01', '2020-12-31'); HOLD = ('2021-01-01', '2026-12-31'); FULL = ('2006-01-01', '2026-12-31'); Y19 = ('2019-01-01', '2026-12-31')
SEL = ('2006-01-01', '2014-12-31')

def sim():
    B = []   # (ckpt 名, jobs)
    for tg in ['BASE', 'A20'] + V8:
        B.append((f'N100|{tg}', RL.jobs_for(f'N100|{tg}', '100萬', 'normal', OPTS[tg], track=tg in ('BASE', CH, 'MT', 'NR', 'CB1', 'CB2'))))
    cells = {'OFF100': ('100萬', 'normal', CL.OFFC), 'S100': ('100萬', 'stress', None), 'N30': ('30萬', 'normal', None), 'S30': ('30萬', 'stress', None),
             'N300': ('300萬', 'normal', None), 'N1000': ('1000萬', 'normal', None)}
    for c, (tier, cost, sw) in cells.items():
        for tg in ('BASE', CH):
            B.append((f'{c}|{tg}', RL.jobs_for(f'{c}|{tg}', tier, cost, OPTS[tg], sw=sw)))
    for nm, o in FZ['neighbors'].items():
        B.append((f'N100|NB_{nm}', RL.jobs_for(f'N100|NB_{nm}', '100萬', 'normal', o)))
    for c, tier in (('N30', '30萬'), ('N300', '300萬'), ('N1000', '1000萬')):
        for tg in ('BASE', 'CA80', 'CA90', 'MT'):
            B.append((f'AUX{c}|{tg}', RL.jobs_for(f'AUX{c}|{tg}', tier, 'normal', OPTS[tg], seeds=range(5))))
    for tg in ('BASE', 'CA80', 'CA90', 'MT'):
        B.append((f'AUXOFF100|{tg}', RL.jobs_for(f'AUXOFF100|{tg}', '100萬', 'normal', OPTS[tg], sw=CL.OFFC, seeds=range(5))))
    print('n_jobs', sum(len(j) for _, j in B), flush=True)
    NV = {}; TR = {}; ST = {}
    for nm, jobs in B:
        res = RL.run_batch(jobs, nproc=2, log=f'oos:{nm}', ckpt='oos_' + nm.replace('|', '_') + '.pkl')
        for k, v in res.items():
            key = f'{k[0]}|{k[1]}|{k[2]}'
            NV[key] = v['nav'].astype('float32'); ST[key] = v['st']
            if v['tr'] is not None:
                for c in v['tr'].columns: TR[f'{key}|{c}'] = v['tr'][c].astype('float32')
        del res
    pd.DataFrame(NV).to_parquet(NAVF, compression='zstd')
    pd.DataFrame(TR).to_parquet(TRF, compression='zstd')
    (HERE / 'data/oos_simstats.json').write_text(CL.dumps(ST), encoding='utf-8')

# ---------------- 統計 ----------------
def load():
    NV = pd.read_parquet(NAVF).astype(float)
    res = {}
    for c in NV.columns:
        a, b, v, j = c.split('|'); s = NV[c].dropna()
        res[(f'{a}|{b}', v, int(j))] = dict(nav=s / s.iloc[0])
    return res
E = None; S = None
def gstats(res, tag):
    """對 0050：x_stats.stats 逐 j → jsumm；三段；拿掉最好 3 年 t；放回版本取最差。"""
    out = {}
    for v in RL.VARS:
        per = []; seg = []; d3 = []; dmax = []
        for j in range(13):
            nav = res[(tag, v, j)]['nav']
            st = S.stats(nav, 'AB_V7_B10', j); per.append(st)
            e = S.end_day('AB_V7_B10', j)
            sg = [CL.cagr(nav, a, min(pd.Timestamp(b), e)) - CL.cagr(S.E50, max(pd.Timestamp(a), nav.index[0]), min(pd.Timestamp(b), e))
                  for a, b in (('2006-04-01', '2012-12-31'), ('2013-01-01', '2018-12-31'), ('2019-01-01', '2026-12-31'))]
            seg.append(sg)
            x = S.windows(nav.loc[:e], 'AB_V7_B10', j); ye = S.yearly_exc(nav, 'AB_V7_B10', j)
            b3 = sorted(ye, key=lambda y: -ye[y])[:3]
            d3.append(S.tstat(x[~x.index.year.isin(b3)]))
            dmax.append(S.tstat(x.drop(x.idxmax())))
        js = S.jsumm(per)
        o = {k: js[k] for k in ('exc_med', 'exc_p10', 't_0050_med', 't_0050_p10', 'CAGR_med', 'CAGR_Y19on_med', 'E50_CAGR_Y19on_med', 'exc_Y19on_med',
                                't_Y19on_med', 'maxDD_med', 'exc_SEL_med', 'exc_POSTODD_med', 'exc_PREODD_med', 'CAGR_POSTODD_med', 'n_j_exc_Y19on_pos')}
        sg = np.array(seg); o['seg_exc_med'] = np.median(sg, 0).tolist()
        o['t_drop_best3y_med'] = float(np.median(d3)); o['t_drop_maxwin_med'] = float(np.median(dmax))
        o['by_j_exc'] = [p['exc'] for p in per]; o['by_j_t'] = [p['t_0050'] for p in per]
        out[v] = o
    w = {}
    for k in out['REAL_opt']:
        if k.startswith('by_j'): continue
        vals = {v: out[v][k] for v in RL.VARS}
        if k == 'seg_exc_med': w[k] = [min(vals[v][i] for v in RL.VARS) for i in range(3)]
        elif k == 'maxDD_med': w[k] = min(vals.values())
        else: w[k] = min(vals.values())
    w['worst_variant_exc'] = min(RL.VARS, key=lambda v: out[v]['exc_med'])
    return dict(by_variant=out, worst=w)

def pairs(res, tag, base):
    o = {}
    for nm, (a, b) in dict(selection=SEL, validation=VAL, holdout=HOLD, full=FULL, Y19on=Y19).items():
        p = RL.paired(res, tag, base, a, b); pia = RL.paired(res, tag, base, a, b, ia=True)
        o[nm] = dict(worst=p['worst'], by_variant={v: {k: p[v][k] for k in ('med', 'mean', 'max', 'min', 'p10', 'med_j')} for v in RL.VARS}, ia_worst_med=pia['worst']['med'])
    fw = o['full']['worst']['variant']
    mo = {}
    for v in RL.VARS:
        dm = RL.monthly_diff(res, tag, base, '2006-05-01', '2026-10-31', v)
        b = RL.boot_ci(dm.values)
        dmx = dm.drop(dm.idxmax())
        mo[v] = dict(mean_pp_yr=float(dm.mean() * 1200), t=CL.tstat(dm), n_m=int(len(dm)), boot=dict(ci90=[x * 1200 for x in b['ci90']], ci95=[x * 1200 for x in b['ci95']], p_le0=b['p_le0']),
                     drop_maxmonth_pp_yr=float(dmx.mean() * 1200), max_month=str(dm.idxmax()), t_val=CL.tstat(dm[(dm.index >= pd.Period('2015-01', 'M')) & (dm.index <= pd.Period('2020-12', 'M'))]),
                     t_hold=CL.tstat(dm[dm.index >= pd.Period('2021-01', 'M')]), t_sel=CL.tstat(dm[dm.index <= pd.Period('2014-12', 'M')]))
    o['monthly'] = mo; o['full_worst_variant'] = fw
    return o

def track(trd, tag):
    out = {}
    for nm, (a, b) in dict(full=FULL, selection=SEL, Y19on=Y19, validation=VAL, holdout=HOLD).items():
        y = {}
        for k in ('tax', 'fee', 'tick', 'imp', 'odd', 'disp', 'turn', 'ord'):
            vals = []
            for v in RL.VARS:
                for j in range(13):
                    c = trd.get(f'{tag}|{v}|{j}|{k}')
                    if c is None: continue
                    c = c.dropna(); c = c[(c.index >= pd.Timestamp(a)) & (c.index <= pd.Timestamp(b))]
                    if len(c) < 20: continue
                    yrs = (c.index[-1] - c.index[0]).days / 365.25
                    vals.append(float(c.mean()) if k == 'ord' else float(c.sum() / yrs))
            y[k] = float(np.median(vals)) * (100 if k not in ('turn', 'ord') else 1) if vals else None
        if y['tax'] is not None: y['cost_total_pp'] = sum(y[k] for k in ('tax', 'fee', 'tick', 'imp', 'odd', 'disp'))
        out[nm] = y
    return out

def g3(res):
    spec = importlib.util.spec_from_file_location('g2lib_ro', AG / 'o_g2_feasibility_stoprule' / 'g2lib.py')
    G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
    FX, meta = G.family()
    mine = {}
    for tg in ['BASE'] + V8:
        jn = G.javg_nav({j: res[(f'N100|{tg}', 'REAL_opt', j)]['nav'] for j in range(13)})
        mine[f'OCAS:{tg}'] = G.win_ex(jn)
    mine = pd.DataFrame(mine)
    D = pd.concat([FX, mine.drop(columns=['OCAS:BASE'])], axis=1)
    m = mine.notna().all(axis=1).to_numpy()
    D = D[m].fillna(0.0)
    Xf = D.to_numpy(); n, Nm = Xf.shape
    W = G.SP.sb_counts(n, 6, 10000, 20261103)
    Mb = G.SP.boot_means(W, Xf)
    om2 = G.SP.lrv(Xf, 'pr', 6); om = np.sqrt(om2); sn = np.sqrt(n)
    dbar = Xf.mean(0); tst = sn * dbar / om
    thr = -np.sqrt(om2 / n * 2 * np.log(np.log(n)))
    gc = np.where(dbar >= thr, dbar, 0.0)
    Z = (Mb - gc[None, :].astype(np.float32)) * (sn / om).astype(np.float32)[None, :]
    Tb = np.maximum(Z.max(axis=1), 0.0)
    sr_all = Xf.mean(0) / Xf.std(0, ddof=1); sr_all = sr_all[np.isfinite(sr_all)]
    et = G.SP.eff_trials(Xf[:, np.std(Xf, 0) > 0])
    out = dict(N_matrix=int(Nm), N_nominal=2581, n_windows=int(n), first=str(D.index[0].date()), last=str(D.index[-1].date()), Tb_q95=float(np.quantile(Tb, 0.95)),
               max_t_family=float(tst.max()), argmax=str(D.columns[int(np.argmax(tst))]), M_eff_LiJi=float(et['M_eff_LiJi']),
               p_SPA_family=float((Tb >= tst.max()).mean()))
    for c in mine.columns:
        if c == 'OCAS:BASE':
            x = mine[c].to_numpy()[m]; k = None
        else:
            k = list(D.columns).index(c); x = Xf[:, k]
        t = float(math.sqrt(n) * x.mean() / math.sqrt(G.SP.lrv(x[:, None], 'pr', 6)[0]))
        out[c] = dict(t=t, p_SPA=float((Tb >= t).mean()) if t > 0 else 1.0, DSR_N2581=G.SP.dsr(x, sr_all, 2581)['DSR'],
                      DSR_LiJi=G.SP.dsr(x, sr_all, et['M_eff_LiJi'])['DSR'], SR_m=float(x.mean() / x.std(ddof=1)), mean_pp=float(x.mean() * 100))
    return out

def stats():
    global E, S
    E = CL.engine(); S = E['S']
    res = load(); trd = pd.read_parquet(TRF).astype(float)
    O = dict(written_utc=pd.Timestamp.now('UTC').isoformat(), chosen=CH, chosen_opt=CHO)
    # 重現核對
    gb = gstats(res, 'N100|BASE')
    O['repro_base'] = dict(exc_med_worst=gb['worst']['exc_med'], t_worst=gb['worst']['t_0050_med'], jp10_worst=gb['worst']['exc_p10'],
                           y19_cagr_opt=gb['by_variant']['REAL_opt']['CAGR_Y19on_med'], published=dict(exc=0.86, t=0.45, jp10=-2.25, y19=23.27))
    print('repro', O['repro_base'], flush=True)
    O['G_vs_0050'] = {}
    for tg in ['BASE', 'A20'] + V8: O['G_vs_0050'][f'N100|{tg}'] = gstats(res, f'N100|{tg}')
    for c in ('OFF100', 'S100', 'N30', 'S30', 'N300', 'N1000'):
        for tg in ('BASE', CH): O['G_vs_0050'][f'{c}|{tg}'] = gstats(res, f'{c}|{tg}')
    for nm in FZ['neighbors']: O['G_vs_0050'][f'N100|NB_{nm}'] = gstats(res, f'N100|NB_{nm}')
    O['paired'] = {}
    for tg in ['A20'] + V8: O['paired'][f'N100|{tg}'] = pairs(res, f'N100|{tg}', 'N100|BASE')
    for c in ('OFF100', 'S100', 'N30', 'S30', 'N300', 'N1000'): O['paired'][f'{c}|{CH}'] = pairs(res, f'{c}|{CH}', f'{c}|BASE')
    for nm in FZ['neighbors']: O['paired'][f'N100|NB_{nm}'] = pairs(res, f'N100|NB_{nm}', 'N100|BASE')
    for c in ('AUXN30', 'AUXN300', 'AUXN1000', 'AUXOFF100'):
        for tg in ('CA80', 'CA90', 'MT'):
            p = {}
            for nm, (a, b) in dict(selection=SEL, full=FULL, pre_odd=('2006-01-01', '2020-10-23'), post_odd=('2020-10-26', '2026-12-31'), Y19on=Y19).items():
                pp = RL.paired(res, f'{c}|{tg}', f'{c}|BASE', a, b); p[nm] = pp['worst']
            O['paired'][f'{c}|{tg}'] = p
    # 100 萬 MT 的盤中零股前後（P4）
    for tg in ('MT', 'CA80', 'CA90', 'NR', 'DC5C', 'DC5E', 'CB1', 'CB2'):
        O['paired'][f'N100|{tg}']['pre_odd'] = RL.paired(res, f'N100|{tg}', 'N100|BASE', '2006-01-01', '2020-10-23')['worst']
        O['paired'][f'N100|{tg}']['post_odd'] = RL.paired(res, f'N100|{tg}', 'N100|BASE', '2020-10-26', '2026-12-31')['worst']
        O['paired'][f'N100|{tg}']['seg0612'] = RL.paired(res, f'N100|{tg}', 'N100|BASE', '2006-01-01', '2012-12-31')['worst']
    O['track'] = {f'N100|{tg}': track(trd, f'N100|{tg}') for tg in sorted(set(('BASE', CH, 'MT', 'NR', 'CB1', 'CB2')))}
    O['simstats'] = json.loads((HERE / 'data/oos_simstats.json').read_text(encoding='utf-8'))
    O['simstats'] = {k: v for k, v in O['simstats'].items() if k.startswith('N100|') and k.endswith('|0')}
    # 兩平滑價（每邊額外滑價使全期超額歸零；近似＝超額 ÷ 雙邊周轉）
    be = {}
    for tg in ('BASE', CH):
        tt = O['track'][f'N100|{tg}']['full']['turn']; ex = O['G_vs_0050'][f'N100|{tg}']['worst']['exc_med']
        be[tg] = dict(turn_twosided=tt, exc_full=ex, breakeven_extra_slip_per_side_pct=ex / tt if tt else None)
    O['breakeven'] = be
    print('G3 ...', flush=True)
    O['G3'] = g3(res)
    # 判定
    gw = O['G_vs_0050'][f'N100|{CH}']['worst']; sw_ = O['G_vs_0050'][f'S100|{CH}']['worst']
    pr = O['paired'][f'N100|{CH}']; fw = pr['full_worst_variant']; mo = pr['monthly'][fw]
    nbs = [O['G_vs_0050'][f'N100|NB_{nm}']['worst']['exc_med'] for nm in FZ['neighbors']]
    g3c = O['G3'].get(f'OCAS:{CH}', {})
    O['improvement_vs_main'] = dict(val=pr['validation']['worst']['med'], hold=pr['holdout']['worst']['med'], full_t=mo['t'], ci95=mo['boot']['ci95'],
                                    established=bool(pr['validation']['worst']['med'] > 0 and pr['holdout']['worst']['med'] > 0 and mo['t'] >= 2.0 and mo['boot']['ci95'][0] > 0))
    G = dict(G1=bool(gw['exc_p10'] > 0 and sw_['exc_med'] > 0), G2=bool(gw['t_0050_med'] >= 3.0 and gw['t_Y19on_med'] >= 2.0),
             G3=bool(g3c.get('p_SPA', 1) < 0.05 and g3c.get('DSR_N2581', 0) >= 0.95),
             G4=bool(min(gw['seg_exc_med']) > 0 and gw['t_drop_best3y_med'] >= 2.0 and all(np.sign(x) == np.sign(gw['exc_med']) for x in nbs)),
             G5='設計上 PIT（ĉ 只用 d 收盤前資料；名單沿用官方 ALLPIT），繼承主策略資料限制', G6='待工作流層分派不同子代理複製',
             G7=bool(all(O['G_vs_0050'][f'{c}|{CH}']['worst']['exc_med'] > 0 for c in ('N300', 'N1000')) and gw['exc_med'] > 0), G8='未開始（未過 G1～G7 不進前瞻）')
    O['gates'] = G
    O['verdict'] = '不可用' if not all(G[k] for k in ('G1', 'G2', 'G3', 'G4', 'G7')) or FZ['verdict_at_freeze'] != 'candidate' else '歷史最高標準通過，前瞻驗證中'
    r = json.loads(RES.read_text(encoding='utf-8')); r['OOS'] = O
    RES.write_text(CL.dumps(r), encoding='utf-8')
    print('verdict', O['verdict'], G, flush=True)

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('sim', 'all'): sim()
    if mode in ('stats', 'all'): stats()

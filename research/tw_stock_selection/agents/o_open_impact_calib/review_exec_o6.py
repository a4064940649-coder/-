#!/usr/bin/env python3
"""反方審查（成本與可成交角度）o_open_impact_calib。只讀研究員檔案，不寫入研究員檔案；輸出到 scratchpad。
A. 「Y≈0.79」是不是只是成交量時鐘下的擴散波動（√(2/π)≈0.798），而不是自己委託的衝擊。
B. OA_hat（參與上限分母）相對 MIS 真值與 OA_b 的偏誤 → 20%×OA_hat 實際等於真實開盤集合競價的幾成。
C. cap_rows：以真實開盤量口徑換算的參與率；單一跳動單位相對損益兩平滑價。
D. 級距之間的配對差（是否只是雜訊）。
E. 不利重跑（AB、40 seeds）：參與上限改為真實開盤量 20%（cap 0.08×OA_hat）＋Z part 0.2；以及基礎滑價 0.2%（約一跳）。
用法：python3 -I -B review_exec_o6.py [A|B|C|D|E ...]
"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
OUT = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_o6')
OUT.mkdir(parents=True, exist_ok=True)
STEPS = set(sys.argv[1:]) or set('ABCDE')
R = {}
def log(*a): print(*a, flush=True)

if 'A' in STEPS or 'B' in STEPS:
    import iterate as I, calib_lib as C
    X, Rb = I.load()
    oa = I.fit_oa(X); X['OA_hat'] = I.oa_hat(X, oa)
if 'A' in STEPS:
    cal = (X.U & (X.period == 'CAL')).to_numpy()
    Lc = I.long_bars(X, Rb, cal)
    r = Lc.y.to_numpy(); s2 = (Lc.sig ** 2 * Lc.q / Lc.V).to_numpy(); s = np.sqrt(s2)
    # 帶正負號的 bar 報酬，用來算變異數比
    sr = []
    for b in C.BARS:
        sr.append(Rb.loc[cal, b].to_numpy())
    A = dict(Y_ratio=float(r.sum() / s.sum()), sqrt_2_over_pi=math.sqrt(2 / math.pi),
             var_ratio_sum_r2_over_sum_sig2q_over_V=float((r ** 2).sum() / s2.sum()),
             mean_abs_over_rms=float(r.mean() / math.sqrt((r ** 2).mean())))
    # 若完全是成交量時鐘下的高斯擴散，E|r| = √(2/π)·σ√(q/V)；把 Y 拆成 √(VR)×(E|r|/rms) 的形狀因子
    A['note'] = 'Y≈√(2/π) 表示 M2 只是在量時鐘下的波動，不能辨識自己委託的帶方向衝擊'
    # 在同一資料上，用「當日 bar 量份額」解釋 |r| 與用「完全不含量」的 σ/√48 比較
    A['Y_if_no_volume_info_sigma_over_sqrt48'] = float(r.mean() / (Lc.sig / math.sqrt(48)).mean())
    R['A_volume_clock'] = A; log('A', A)
if 'B' in STEPS:
    cal = (X.U & (X.period == 'CAL')).to_numpy(); val = (X.U & (X.period == 'VAL')).to_numpy()
    m = (cal | val) & np.isfinite(X.OA_b).to_numpy() & (X.OA_b > 0).to_numpy()
    B = dict(OA_hat_share_day_median=float((X.OA_hat / X.vol_off)[cal | val].median()),
             OA_b_share_day_median=float((X.OA_b / X.vol_off)[m].median()),
             OA_hat_over_OA_b_median=float((X.OA_hat / X.OA_b)[m].median()),
             OA_hat_over_OA_b_p25_p75=[float(x) for x in np.percentile((X.OA_hat / X.OA_b)[m], [25, 75])],
             OA_hat_over_V60sh_median=float((X.OA_hat / X.V60sh)[cal | val].median()))
    V = C.mis()
    sp = json.loads((HERE / 'data' / 'model_calib.json').read_text())['oa']
    Vp = V[(V.vrank_l1 <= 300) & np.isfinite(V.sig60_l1) & np.isfinite(V.V60sh) & (V.OA > 0)].copy()
    xx = np.column_stack([np.ones(len(Vp)), np.log(Vp.vrank_l1.clip(lower=1)), np.log(Vp.prevc), np.log(Vp.sig60_l1.clip(lower=1e-4)), np.zeros(len(Vp)), np.zeros(len(Vp))])
    Vp['OA_hat'] = sp['smear'] * np.exp(xx @ np.array(sp['coef'])) * Vp.V60sh
    B['MIS_in_pool_n'] = int(len(Vp))
    B['MIS_in_pool_OA_hat_over_OA_true_median'] = float((Vp.OA_hat / Vp.OA).median())
    B['MIS_in_pool_OA_true_share_day_median'] = float((Vp.OA / Vp.volume).median())
    B['MIS_in_pool_OA_true_over_V60sh_median'] = float((Vp.OA / Vp.V60sh).median())
    B['MIS_vol_surprise_median'] = float((Vp.volume / Vp.V60sh).median())
    R['B_oa_hat_bias'] = B; log('B', B)
if 'C' in STEPS:
    rows = pd.read_parquet(HERE / 'data' / 'cap_rows.parquet')
    rows = rows[(rows.strat == 'AB') & (rows.lot == 1)]
    k = R.get('B_oa_hat_bias', {}).get('OA_hat_over_OA_b_median', 2.3)
    Cc = {}
    def tick(p): return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0
    for (ti, ve), g in rows.groupby(['tier', 'ver']):
        qt = g.q_over_oa * k
        px = g.Q / (g.n * 1000.0)
        tk = np.array([tick(p) for p in px.to_numpy()]) / px.to_numpy()
        Cc[f'{ti}|{ve}'] = dict(n=int(len(g)), share_gt20pct_true_OA=float((qt > 0.2).mean()), share_gt50pct_true_OA=float((qt > 0.5).mean()),
                                median_true_participation=float(qt.median()), p90_true_participation=float(qt.quantile(0.9)),
                                one_tick_pct_median=float(np.median(tk) * 100), share_one_tick_gt_0117pct=float((tk > 0.00117).mean()),
                                share_one_tick_gt_0025pct=float((tk > 0.00025).mean()))
    R['C_participation'] = dict(scale_OA_hat_to_true=k, cells=Cc); log('C', json.dumps(Cc, ensure_ascii=False, indent=0))
if 'D' in STEPS:
    import ecore6 as E
    A = pd.read_parquet(HERE / 'data' / 'cap_avg_navs.parquet')
    D = E.Data(); sd = D.sig_days
    def wr(col): return E.win_ret(A[col].dropna(), sd)
    Dd = {}
    for ve in ('FIX', 'GROW'):
        for a, b in (('300萬', '100萬'), ('300萬', '30萬'), ('300萬', '1000萬'), ('100萬', '1000萬'), ('30萬', '1000萬')):
            x = (wr(f'AB|M3V_point|{a}|{ve}') - wr(f'AB|M3V_point|{b}|{ve}')).dropna()
            Dd[f'{ve} {a}-{b}'] = dict(mean_pp_per_window=float(x.mean() * 100), t=E.tstat(x))
        for v in ('O5', 'M2', 'Z02'):
            for ti in ('300萬', '1000萬'):
                x = (wr(f'AB|M3V_point|{ti}|{ve}') - wr(f'AB|{v}|{ti}|{ve}')).dropna()
                Dd[f'{ve} {ti} M3V-{v}'] = dict(mean_pp_per_window=float(x.mean() * 100), t=E.tstat(x))
    R['D_tier_pairs'] = Dd; log('D', json.dumps(Dd, ensure_ascii=False, indent=0))
if 'E' in STEPS:
    import ecore6 as E
    from concurrent.futures import ProcessPoolExecutor
    SPEC = json.loads((E.DATA / 'model_calib.json').read_text(encoding='utf-8'))
    VAR = {'CAPTRUE_Z02': dict(cap=0.08, opts=dict(E.FRICTIONS, zpart=0.2)),
           'BASE02': dict(cap=0.2, opts=dict(E.FRICTIONS, zpart=1.0, base=0.002)),
           'CAPTRUE_Z02_BASE02': dict(cap=0.08, opts=dict(E.FRICTIONS, zpart=0.2, base=0.002))}
    _W = {}
    def init(): _W['D'] = E.Data()
    def job(sp):
        D = _W['D']; M = E.Model(SPEC, impact='M3V', q='point'); M.cap = VAR[sp[0]]['cap']
        nav, acc, _ = E.Sim(D, M, VAR[sp[0]]['opts']).run(D.plans[('AB', sp[3])], D.t0, D.t1, E.TIERS[sp[1]], sp[2], 'A')
        return sp, nav, acc['n_cap_hit'], acc['n_zcap_hit']
    L = [(v, ti, ve, s) for v in VAR for ti in E.TIERS for ve in ('FIX', 'GROW') for s in range(40)]
    T0 = time.time(); navs = {}; hits = {}
    with ProcessPoolExecutor(4, initializer=init) as ex:
        for i, (sp, nav, h1, h2) in enumerate(ex.map(job, L, chunksize=4)):
            navs[sp] = nav; hits.setdefault(sp[:3], []).append((h1, h2))
            if i % 200 == 0: log('E', i, round(time.time() - T0))
    D = E.Data(); sd = D.sig_days; Ev = {}
    for v in VAR:
        for ti in E.TIERS:
            for ve in ('FIX', 'GROW'):
                Rr = pd.concat([navs[(v, ti, ve, s)].pct_change() for s in range(40)], axis=1).mean(axis=1).fillna(0)
                nav = (1 + Rr).cumprod()
                x = E.excess(nav, D.E50.reindex(nav.index), sd)
                h = np.array(hits[(v, ti, ve)])
                Ev[f'{v}|{ti}|{ve}'] = dict(CAGR=E.cagr_mdd(nav, E.A_, E.B_)[0], t=E.tstat(x), cap_hits_per_seed=float(h[:, 0].mean()), zcap_hits_per_seed=float(h[:, 1].mean()))
    R['E_unfavorable_reruns'] = Ev; log('E', json.dumps(Ev, ensure_ascii=False, indent=0))
f = OUT / ('review_o6_' + ''.join(sorted(STEPS)) + '.json')
json.dump(R, open(f, 'w'), ensure_ascii=False, indent=1); log('saved', f)

#!/usr/bin/env python3
"""步驟 1（探索，不看任何前瞻報酬；目前也沒有前瞻報酬）：
 (1) 盤點前瞻檔欄位（live 只讀表頭）與 o9 重播 forward_returns，逐條列 G8 需要的線能否機械計算；
 (2) 歷史決策線（o8 fslib，只讀 import）：設計 σ、殘差形狀、自相關、HAC、壓力成本平移（挑選期估）；
 (3) iid 重現 o9 G8 檢定力表（誤差 ≤ 0.02 才算程式正確）；
 (4) 相依性膨脹係數 c 的候選數字。
輸出：explore.json、results.json 的 EXPLORE。用法：python3 -I -B s1_explore.py"""
import sys, json, math, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import g8lib as G

AG = G.AG
LIVE = AG / 'live_tracking/main_forward'
INFRA = AG / 'o_forward_infra_live'
REPLAY = INFRA / 'replay/seq'
O8 = AG / 'o_forward_sequential_v2'


def header(p):
    with open(p, encoding='utf-8') as f:
        return next(csv.reader(f))


def inventory():
    live_fr, live_tr = header(LIVE / 'forward_returns.csv'), header(LIVE / 'trades.csv')
    live_fi = header(LIVE / 'fills_input.csv')
    rp = pd.read_csv(REPLAY / 'forward_returns.csv')
    tr = pd.read_csv(REPLAY / 'trades.csv', dtype={'code': str})
    assert header(REPLAY / 'forward_returns.csv') == live_fr, 'live 與重播欄位不同'
    comp = rp[rp.status == 'complete']
    out = {'live_forward_returns_columns': live_fr, 'live_trades_columns': live_tr, 'live_fills_input_columns': live_fi,
           'live_n_rows': {'forward_returns': int(sum(1 for _ in open(LIVE / 'forward_returns.csv')) - 1),
                           'trades': int(sum(1 for _ in open(LIVE / 'trades.csv')) - 1)},
           'replay_lines': sorted(rp.line.unique().tolist()), 'replay_trade_kinds': tr.groupby(['kind']).size().to_dict()}
    g = lambda line, col: comp.loc[comp.line == line, col]
    a = '2026-09'
    row = lambda line: comp[(comp.line == line) & (comp.anchor == a)].iloc[0]
    J = comp[comp.line.str.match(r'^J\d+$') & (comp.anchor == a)]
    jmed_re = float(np.median(J.excess_pp)); jmed_rep = float(row('JMED').excess_pp)
    jp10 = float(np.percentile(J.excess_pp, 10))
    mo, e5 = row('MAIN_OFFICIAL'), row('E50')
    mix_re = 0.3 * mo.ret + 0.7 * e5.ret                       # 第一窗沒有漂移成本
    m30_re = 0.7 * mo.ret + 0.3 * e5.ret
    lines = {
        'JMED（主欄，V7X K5T GROW 100 萬，一般成本）': dict(ok=True, how='forward_returns line=JMED, excess_pp；status=complete 且 n_j=13', replay_2026_09=jmed_rep,
                                                  recompute_median_J0_J12=jmed_re, diff=abs(jmed_re - jmed_rep)),
        'JMED 壓力成本': dict(ok=True, how='line=JMED, excess_stress_pp', replay_2026_09=float(row('JMED').excess_stress_pp)),
        'JMED 參與上限 10%': dict(ok=True, how='line=JMED, excess_cap10_pp', replay_2026_09=float(row('JMED').excess_cap10_pp)),
        'j 的 p10（G1 前瞻版）': dict(ok=True, how='同錨定月 line=J0..J12 的 excess_pp 取 numpy percentile 10（線性內插）', replay_2026_09=jp10),
        'MAIN_OFFICIAL': dict(ok=True, how='line=MAIN_OFFICIAL, excess_pp', replay_2026_09=float(mo.excess_pp)),
        'E50（0050 含息）': dict(ok=True, how='line=E50, ret', replay_2026_09=float(e5.ret)),
        'C_E50_70＝MIX_C70（0050 70%＋MAIN_OFFICIAL 30%）': dict(ok=True, how='line=MIX_C70；可由 MAIN_OFFICIAL.ret、E50.ret 與上一窗漂移流量×0.5% 重算',
                                                          replay_2026_09=float(row('MIX_C70').ret), recompute=float(mix_re), diff=abs(mix_re - float(row('MIX_C70').ret))),
        'M_E50_30（主策略 70%＋0050 30%）': dict(ok=True, how='衍生：0.7×MAIN_OFFICIAL.ret＋0.3×E50.ret − |w′−w|×0.5%（與 MIX_C70 同法），不在 forward_returns 內，需離線計算',
                                            replay_2026_09_ret=float(m30_re), replay_2026_09_excess_pp=float((m30_re - e5.ret) * 100)),
        '100 萬 FIX': dict(ok=False, how='缺口：前瞻引擎只跑 GROW 100 萬（x_exec 設定 mode＝GROW），forward_returns 沒有 FIX 欄；24 個月內 GROW 100 萬與 FIX 100 萬只差淨值規模（零股化程度），以 GROW 代替並標註'),
        '300 萬 GROW（G7 前瞻）': dict(ok=False, how='缺口：forward_returns 沒有 300 萬欄；需離線以 x_exec.py 讀凍結 paper_lists 與 .exec.json 重算（描述用，不得作為 G8 判定線）'),
        '1,000 萬 GROW（G7 前瞻）': dict(ok=False, how='缺口：同上'),
    }
    out['lines'] = lines
    out['n_gaps'] = sum(not v['ok'] for v in lines.values())
    return out


def hist_lines():
    sys.path.insert(0, str(O8))
    import fslib as F
    sys.path.pop(0)
    def line(tier, mode, cost):
        M, _, _ = F.cell_jmatrix(tier, mode, cost=cost)
        s = F.jmed(M); s.index = pd.PeriodIndex(s.index, freq='M'); return s
    L = {'FIX100_n': line('100萬', 'FIX', 'normal'), 'GROW100_n': line('100萬', 'GROW', 'normal'), 'FIX100_s': line('100萬', 'FIX', 'stress'),
         'GROW100_s': line('100萬', 'GROW', 'stress'), 'GROW300_n': line('300萬', 'GROW', 'normal'), 'GROW1000_n': line('1000萬', 'GROW', 'normal')}
    return L, F


def acf(x, k):
    x = np.asarray(x) - np.mean(x)
    return float(np.sum(x[k:] * x[:-k]) / np.sum(x * x))


def lrv(x, L):
    x = np.asarray(x) - np.mean(x); n = len(x)
    g0 = np.sum(x * x) / n; v = g0
    for k in range(1, L + 1):
        v += 2 * (1 - k / (L + 1)) * np.sum(x[k:] * x[:-k]) / n
    return float(v)


def main():
    inv = inventory()
    L, F = hist_lines()
    d = L['FIX100_n']
    post = d[d.index >= pd.Period('2019-05', 'M')]
    sel = d[(d.index >= pd.Period('2006-04', 'M')) & (d.index <= pd.Period('2019-04', 'M'))]
    p24 = d[d.index >= pd.Period('2024-01', 'M')]
    resid = (post - post.mean()).to_numpy(); resid24 = (p24 - p24.mean()).to_numpy()
    sig_hac6 = F.hac_sd(post)
    assert abs(sig_hac6 - G.SIGMA_D) < 1e-6, f'σ 重現失敗 {sig_hac6}'
    desc = {}
    for nm, x in (('post2019', post.to_numpy()), ('sel_2006_2019', sel.to_numpy()), ('2024_2026', p24.to_numpy())):
        x = np.asarray(x, float)
        desc[nm] = dict(n=len(x), first=None, sd=float(np.std(x, ddof=1)), hac_sd={L_: math.sqrt(lrv(x, L_)) for L_ in (3, 6, 12)},
                        acf={k: acf(x, k) for k in range(1, 7)}, kurt_excess=float(pd.Series(x).kurt()), skew=float(pd.Series(x).skew()))
    # 相依性膨脹係數候選（只用殘差形狀；定義見 PREREG_DESIGN）
    c2 = {'plain_sd': (np.std(resid, ddof=1) / G.SIGMA_D) ** 2}
    for L_ in (3, 6, 12): c2[f'hac_L{L_}'] = lrv(resid, L_) / G.SIGMA_D ** 2
    rng = np.random.default_rng(20261104)
    Xb = G.gen('D3', 0.0, 200_000, 24, rng, resid=resid)
    for n in (12, 24): c2[f'boot_L6_var_S{n}'] = float(np.var(Xb[:, :n].sum(axis=1)) / (n * G.SIGMA_D ** 2))
    Xb3 = G.gen('D3', 0.0, 200_000, 24, rng, resid=resid, block=3); c2['boot_L3_var_S24'] = float(np.var(Xb3.sum(axis=1)) / (24 * G.SIGMA_D ** 2))
    Xb12 = G.gen('D3', 0.0, 200_000, 24, rng, resid=resid, block=12); c2['boot_L12_var_S24'] = float(np.var(Xb12.sum(axis=1)) / (24 * G.SIGMA_D ** 2))
    Xb24 = G.gen('D3', 0.0, 200_000, 24, rng, resid=resid24); c2['boot2024_L6_var_S24'] = float(np.var(Xb24.sum(axis=1)) / (24 * G.SIGMA_D ** 2))
    # 壓力成本平移（挑選期估計；驗證期只描述）
    def shift(a, b, lo, hi):
        z = pd.concat([a, b], axis=1).dropna(); z = z[(z.index >= pd.Period(lo, 'M')) & (z.index <= pd.Period(hi, 'M'))]
        return float((z.iloc[:, 1] - z.iloc[:, 0]).mean())
    shifts = {'stress_minus_normal_FIX100_sel': shift(L['FIX100_n'], L['FIX100_s'], '2006-04', '2019-04'),
              'stress_minus_normal_FIX100_post2019_desc': shift(L['FIX100_n'], L['FIX100_s'], '2019-05', '2026-10'),
              'stress_minus_normal_GROW100_sel': shift(L['GROW100_n'], L['GROW100_s'], '2006-04', '2019-04'),
              'GROW100_minus_FIX100_sel': shift(L['FIX100_n'], L['GROW100_n'], '2006-04', '2019-04'),
              'GROW300_minus_GROW100_sel': shift(L['GROW100_n'], L['GROW300_n'], '2006-04', '2019-04'),
              'GROW1000_minus_GROW100_sel': shift(L['GROW100_n'], L['GROW1000_n'], '2006-04', '2019-04')}
    corr_fix_grow = float(pd.concat([L['FIX100_n'], L['GROW100_n']], axis=1).dropna().corr().iloc[0, 1])
    # (3) o9 G8 檢定力表重現（iid 常態，P＝400,000）
    o9 = json.loads((INFRA / 'results.json').read_text(encoding='utf-8'))['PREREG_FORWARD']['power_warning']['p_pass_given_alpha']
    rng = np.random.default_rng(20261107)
    rep = {}
    for k, v in o9.items():
        a = float(k.split('_')[1].replace('pp', ''))
        X = G.gen('D1', a, 400_000, 12, rng)
        p = float((X.sum(axis=1) > 0).mean())
        rep[k] = dict(o9=v, sim=p, analytic=float(__import__('scipy.stats', fromlist=['norm']).norm.cdf(a * math.sqrt(12) / G.SIGMA_D)), diff=abs(p - v))
    maxdiff = max(r['diff'] for r in rep.values())
    out = dict(utc=G.now_utc(), inventory=inv, sigma_design=G.SIGMA_D, sigma_reproduced=sig_hac6,
               decision_line='o8 fslib：B10 K5T 100 萬 FIX 一般成本，13 個 j 的錨定月窗超額中位數（V7 navs_exec7）；前瞻線為 V7X GROW 100 萬',
               resid_post2019=dict(n=len(resid), first=str(post.index[0]), last=str(post.index[-1])),
               resid_2024=dict(n=len(resid24), first=str(p24.index[0]), last=str(p24.index[-1])),
               desc=desc, inflation_c2_candidates=c2, cost_shifts_pp=shifts, corr_FIX_GROW_100=corr_fix_grow,
               o9_power_repro=rep, o9_power_repro_maxdiff=maxdiff, o9_power_repro_ok=bool(maxdiff <= 0.02),
               note='平均超額（α̂）沒有被任何設計步驟使用；殘差一律去均值。壓力成本平移只用挑選期估計。')
    (G.HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=G._js), encoding='utf-8')
    G.save_key('EXPLORE', out)
    # 供 PREREG 內嵌：殘差序列（去均值、四捨五入到 1e-10 以利 sha 穩定）
    np.save(G.SCR / 'resid_post2019.npy', np.round(resid, 10)); np.save(G.SCR / 'resid_2024.npy', np.round(resid24, 10))
    pd.DataFrame({k: v.astype(float) for k, v in L.items()}).to_csv(G.SCR / 'hist_lines.csv')
    print(json.dumps({'gaps': inv['n_gaps'], 'desc_post': desc['post2019'], 'c2': c2, 'shifts': shifts, 'repro_max': maxdiff}, ensure_ascii=False, indent=0, default=G._js))


if __name__ == '__main__':
    main()

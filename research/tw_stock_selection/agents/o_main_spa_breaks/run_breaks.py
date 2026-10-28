#!/usr/bin/env python3
"""步驟 4：結構斷點（PREREG 後才可執行；只跑一次）。
序列（W_long 245 窗）：x50_net／x50_gross（AB_V4 對 0050，扣成本後／前）、xEW_net／xEW_gross（對同池等權）、
Q5mQ1_gross／Q5mQ1_net（池內綜合分數五分位價差）。
方法：OLS-CUSUM、遞迴殘差 CUSUM、Andrews sup-F（15%）、Bai-Perron（m≤3；BIC、LWZ、序列 5%）、
事前固定日期 Chow（2019-05、2024-01）＋各自 80% 檢定力 MDE。主版 HAC；iid 版並列（迭代選出的版本）。
另做：模擬檢定力（區塊重抽挑選期序列，2019-05 起均值下降 25%、50%、100%）。寫入 RESULTS_BREAKS。"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C, breaks_lib as BL
RES = C.check_prereg()
PR = RES['PREREG']; BP = PR['breaks']
MAIN_ID = PR['main_id']
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

def series():
    D = pd.read_parquet(C.DATA / 'family_excess.parquet'); Dew = pd.read_parquet(C.DATA / 'family_excess_ew.parquet')
    B = pd.read_parquet(C.DATA / 'bench.parquet'); BI = pd.read_parquet(C.DATA / 'break_inputs.parquet')
    s = {'x50_net': D[MAIN_ID], 'x50_gross': BI['AB_V4|zero'] - B.r0050, 'xEW_net': Dew[MAIN_ID], 'xEW_gross': BI['AB_V4|zero'] - B.rEW_ABV4,
         'Q5mQ1_gross': BI['Q5|zero'] - BI['Q1|zero'], 'Q5mQ1_net': BI['Q5|normal'] - BI['Q1|normal']}
    return {k: v[C.wmask(v.index, *C.WINDOWS['W_long'])].dropna() for k, v in s.items()}

def main():
    S = series()
    n = len(S['x50_net']); idx = S['x50_net'].index
    for k, v in S.items(): assert len(v) == n and (v.index == idx).all(), k
    nulld = {h: BL.null_dists(n, BP['nsim_null'], BP['seed_null'] + int(h), hac=h) for h in (True, False)}
    log('null dists', {h: {k: float(np.quantile(v, 0.95)) for k, v in d.items()} for h, d in nulld.items()})
    kchow = {lab: int(np.argmax(idx >= pd.Timestamp(d))) for lab, d in BP['chow_dates'].items()}
    out = {'n': n, 'chow_index': kchow, 'crit95': {str(h): {k: float(np.quantile(v, 0.95)) for k, v in d.items()} for h, d in nulld.items()},
           'argmax_c90': BL.argmax_crit(0.90), 'series': {}}
    for name, x in S.items():
        y = x.to_numpy()
        o = {'mean_pp': float(y.mean() * 100), 'sd_pp': float(y.std(ddof=1) * 100), 't': C.tstat(y)}
        for h in (True, False):
            tag = 'HAC' if h else 'iid'
            bp = BL.bai_perron(y, nulld[h], BP['trim'], h)
            for lab in ('sel_BIC', 'sel_LWZ', 'sel_seq'):
                bp[lab]['break_dates'] = [str(idx[k].date()) for k in bp[lab]['breaks']]
                bp[lab]['ci90_dates'] = [(str(idx[max(a, 0)].date()), str(idx[min(b, n - 1)].date())) for a, b in bp[lab]['ci90']]
            bp['supF1_date'] = str(idx[bp['supF1_k']].date())
            bp['breaks_by_m'] = {m: [str(idx[k].date()) for k in v] for m, v in bp['breaks_by_m'].items()}
            o[f'BP_{tag}'] = bp
            o[f'OLS_CUSUM_{tag}'] = BL.ols_cusum(y, hac=h)
            o[f'Chow_{tag}'] = {lab: BL.chow(y, k, hac=h) for lab, k in kchow.items()}
        o['REC_CUSUM'] = BL.rec_cusum(y)
        o['OLS_CUSUM_HAC']['argmax_date'] = str(idx[o['OLS_CUSUM_HAC']['argmax']].date())
        # 斷點後均值 ≤ 斷點前一半？（序列 5% 選出的斷點；只看 2019 年後的斷點）
        sel = o['BP_HAC']['sel_seq']
        post19 = [(k, i) for i, k in enumerate(sel['breaks']) if idx[k] >= pd.Timestamp('2019-01-01')]
        o['decline_check'] = None
        if post19:
            k, i = post19[0]; mu = sel['seg_means']
            o['decline_check'] = {'break_date': str(idx[k].date()), 'mean_before_pp': mu[i] * 100, 'mean_after_pp': mu[i + 1] * 100,
                                  'after_le_half_before': bool(mu[i + 1] <= 0.5 * mu[i])}
        out['series'][name] = o
        log(name, 'supF', round(o['BP_HAC']['supF1'], 2), 'p', o['BP_HAC']['p_supF1'], o['BP_HAC']['supF1_date'], 'm_seq', o['BP_HAC']['m_seq5'],
            'chow19 p', round(o['Chow_HAC']['2019-05']['p'], 3), 'chow24 p', round(o['Chow_HAC']['2024-01']['p'], 3))
    # 模擬檢定力：區塊重抽 2006-04～2019-04 的序列（去均值後加回挑選期均值），在 2019-05 起均值下降 d
    rng = np.random.default_rng(BP['seed_power'])
    k19 = kchow['2019-05']
    pw = {}
    crit_sup = float(np.quantile(nulld[True]['supF1'], 0.95))
    for name in ('x50_net', 'xEW_net', 'Q5mQ1_gross'):
        y = S[name].to_numpy(); sel = y[:k19]; mu = sel.mean(); yd = sel - mu
        res = {}
        for frac in (0.25, 0.5, 1.0):
            rej_sup = rej_chow = 0
            for s in range(BP['nsim_power']):
                p = 1 / 6; path = np.empty(n, int); path[0] = rng.integers(0, len(yd))
                for t in range(1, n): path[t] = rng.integers(0, len(yd)) if rng.random() < p else (path[t - 1] + 1) % len(yd)
                z = yd[path] + mu; z[k19:] -= frac * mu
                rej_sup += BL.supF(z, BP['trim'], True)['stat'] > crit_sup
                rej_chow += BL.chow(z, k19, True)['p'] < 0.05
            res[f'drop{int(frac * 100)}pct'] = {'drop_pp': float(frac * mu * 100), 'power_supF': rej_sup / BP['nsim_power'], 'power_chow2019': rej_chow / BP['nsim_power']}
        pw[name] = res
        log('power', name, res)
    out['power_sim'] = pw
    # 判定（事前規則）
    sig = lambda nm: (out['series'][nm]['BP_HAC']['m_seq5'] > 0 or out['series'][nm]['Chow_HAC']['2019-05']['p'] < 0.05 or out['series'][nm]['Chow_HAC']['2024-01']['p'] < 0.05)
    half = lambda nm: (out['series'][nm]['decline_check'] or {}).get('after_le_half_before', False) or \
        any(out['series'][nm]['Chow_HAC'][d]['p'] < 0.05 and out['series'][nm]['Chow_HAC'][d]['mean_after'] <= 0.5 * out['series'][nm]['Chow_HAC'][d]['mean_before'] for d in ('2019-05', '2024-01'))
    sel_decay = any(sig(nm) and half(nm) for nm in ('xEW_net', 'Q5mQ1_gross'))
    only50 = (sig('x50_net') or sig('x50_gross')) and not any(sig(nm) for nm in ('xEW_net', 'xEW_gross', 'Q5mQ1_gross', 'Q5mQ1_net'))
    out['decision'] = {'selection_power_decay_flag': sel_decay, 'only_vs0050_break_flag': only50,
                       'sig_by_series': {nm: bool(sig(nm)) for nm in S},
                       'sop_note': ('選股力已衰退' if sel_decay else ('落後來自權值集中' if only50 else '未偵測到斷點（附 MDE）'))}
    C.save_key('RESULTS_BREAKS', out)
    log('done', out['decision'])

if __name__ == '__main__':
    main()

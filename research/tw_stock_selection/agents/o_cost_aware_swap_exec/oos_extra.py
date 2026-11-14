#!/usr/bin/env python3
"""凍結後補充統計（只描述，不選擇；先斷言三份 PREREG 與 results.json['OOS'] 已存在）。寫入 results.json['EXTRA']。
(1) 選中變體（依事前規則的描述代表）各格配對差：挑選／驗證／保留／全期／2019 起／三段／盤中零股前後（30 萬 2020-10 前後分列）。
(2) 逐年配對差（100 萬一般，各變體）；(3) 月配對差的均值、標準差、t 與有效樣本數（13 j 重疊）；淨值波動與最大回撤（t 的均值／波動拆解）。
(4) 對同池等權（o_main_spa_breaks/bench.parquet 的 rEW_ABV4，官方換股窗）。
(5) G3 的 N：PREREG 登記 2,581（2,573＋8）；研究中心在本題暫停期間另登記 46 條（外部貼文 2,573→2,619），故累計 2,627；DSR 兩個 N 並列。
(6) P1～P7 逐條核對。(7) P2 的 KS 以有效樣本數（÷4）重算。
python3 -I -B oos_extra.py"""
import sys, json, math, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sps
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; import run_lib as RL; import oos as OO; sys.path.pop(0)
RES = HERE / 'results.json'
R = json.loads(RES.read_text(encoding='utf-8'))
for k in ('PREREG', 'PREREG_POWER', 'PREREG_FREEZE'): assert CL.canon(R[k]) == R[k + '_sha256']
assert 'OOS' in R
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
EX = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
CH = R['PREREG_FREEZE']['chosen']; V8 = OO.V8
res = OO.load()
PER = dict(selection=('2006-01-01', '2014-12-31'), validation=('2015-01-01', '2020-12-31'), holdout=('2021-01-01', '2026-12-31'), full=('2006-01-01', '2026-12-31'),
           Y19on=('2019-01-01', '2026-12-31'), seg0612=('2006-01-01', '2012-12-31'), seg1318=('2013-01-01', '2018-12-31'),
           pre_odd=('2006-01-01', '2020-10-23'), post_odd=('2020-10-26', '2026-12-31'))
out = dict(written_utc=pd.Timestamp.now('UTC').isoformat())

def pw(tag, base, a, b):
    p = RL.paired(res, tag, base, a, b)
    return dict(worst_med=p['worst']['med'], worst_variant=p['worst']['variant'], mean_min=p['worst']['mean'], max_abs=p['worst']['max_abs'],
                by_variant={v: dict(med=p[v]['med'], mean=p[v]['mean'], p10=p[v]['p10'], min=p[v]['min'], max=p[v]['max'], med_j=p[v]['med_j']) for v in RL.VARS})

# (1) 選中變體各格
c1 = {}
for c in ('N100', 'OFF100', 'S100', 'N30', 'S30', 'N300', 'N1000'):
    c1[c] = {nm: pw(f'{c}|{CH}', f'{c}|BASE', a, b) for nm, (a, b) in PER.items()}
out['chosen_cells_paired'] = c1
# 全部變體 100 萬一般各期間
out['all_variants_N100_paired'] = {tg: {nm: pw(f'N100|{tg}', 'N100|BASE', a, b) for nm, (a, b) in PER.items()} for tg in ['A20'] + V8}

# (2) 逐年配對差（年報酬差，pp；13 j 中位數；放回版本取最小）
def yearly(tag, base):
    o = {}
    for v in RL.VARS:
        rows = []
        for j in range(13):
            n1 = res[(tag, v, j)]['nav']; n0 = res[(base, v, j)]['nav']
            y1 = n1.groupby(n1.index.year).last().pct_change(); y0 = n0.groupby(n0.index.year).last().pct_change()
            rows.append((y1 - y0) * 100)
        o[v] = pd.concat(rows, axis=1).median(axis=1)
    d = pd.DataFrame(o).dropna()
    return dict(worst={int(y): float(x) for y, x in d.min(axis=1).items()}, opt={int(y): float(x) for y, x in d['REAL_opt'].items()},
                n_pos_worst=int((d.min(axis=1) > 0).sum()), n_years=int(len(d)))
out['yearly_paired_N100'] = {tg: yearly(f'N100|{tg}', 'N100|BASE') for tg in ['A20'] + V8}

# (3) 月配對差：均值、標準差、t、有效樣本數；波動與回撤
def mstats(tag, base):
    o = {}
    for v in RL.VARS:
        ds = []
        for j in range(13):
            ds.append(CL.monthly(res[(tag, v, j)]['nav']) - CL.monthly(res[(base, v, j)]['nav']))
        dm = pd.concat(ds, axis=1)
        dm = dm[(dm.index >= pd.Period('2006-05', 'M')) & (dm.index <= pd.Period('2026-10', 'M'))]
        avg = dm.mean(axis=1).dropna(); sj = float(dm.std(ddof=1).mean()); sa = float(avg.std(ddof=1))
        # 13 j 的有效獨立條數：var(avg)=sj^2/k_eff
        keff = (sj / sa) ** 2 if sa > 0 else float('nan')
        vol1 = []; vol0 = []; dd1 = []; dd0 = []
        for j in range(13):
            n1 = res[(tag, v, j)]['nav']; n0 = res[(base, v, j)]['nav']
            vol1.append(n1.pct_change().std() * math.sqrt(252) * 100); vol0.append(n0.pct_change().std() * math.sqrt(252) * 100)
            dd1.append((n1 / n1.cummax() - 1).min() * 100); dd0.append((n0 / n0.cummax() - 1).min() * 100)
        o[v] = dict(mean_pp_m=float(avg.mean() * 100), sd_avg13_pp_m=sa * 100, sd_single_j_pp_m=sj * 100, t=CL.tstat(avg), n_m=int(len(avg)), k_eff_of_13=float(keff),
                    vol_var=float(np.median(vol1)), vol_base=float(np.median(vol0)), maxDD_var=float(np.median(dd1)), maxDD_base=float(np.median(dd0)))
    return o
out['monthly_stats'] = {f'{c}|{CH}': mstats(f'{c}|{CH}', f'{c}|BASE') for c in ('N100', 'OFF100', 'S100', 'N30', 'S30', 'N300', 'N1000')}
for tg in V8 + ['A20']:
    if tg != CH: out['monthly_stats'][f'N100|{tg}'] = mstats(f'N100|{tg}', 'N100|BASE')

# (4)(5) 同池等權與 G3 的 N
spec = importlib.util.spec_from_file_location('g2lib_ro', AG / 'o_g2_feasibility_stoprule' / 'g2lib.py')
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
B = pd.read_parquet(AG / 'o_main_spa_breaks' / 'data' / 'bench.parquet')
FX, meta = G.family()
mine = {}; ew = {}
for tg in ['BASE', 'A20'] + V8:
    jn = G.javg_nav({j: res[(f'N100|{tg}', 'REAL_opt', j)]['nav'] for j in range(13)})
    wr = G.FC.win_ret(jn)
    x50 = pd.Series(wr.to_numpy() - G.R50.to_numpy(), index=G.WIN); xew = pd.Series(wr.to_numpy() - B['rEW_ABV4'].reindex(G.WIN).to_numpy(), index=G.WIN)
    mine[tg] = x50
    seg = lambda x, a, b: x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
    ew[tg] = dict(t_vs_EW_full=G.tstat(xew.dropna()), mean_pp_m_vs_EW=float(xew.mean() * 100), t_vs_EW_Y19on=G.tstat(seg(xew, '2019-01-01', '2026-12-31').dropna()),
                  t_vs_0050_full=G.tstat(x50.dropna()), mean_pp_m_vs_0050=float(x50.mean() * 100), t_vs_0050_Y19on=G.tstat(seg(x50, '2019-01-01', '2026-12-31').dropna()),
                  t_vs_0050_sel=G.tstat(seg(x50, '2006-01-01', '2014-12-31').dropna()), t_vs_0050_val=G.tstat(seg(x50, '2015-01-01', '2020-12-31').dropna()),
                  t_vs_0050_hold=G.tstat(seg(x50, '2021-01-01', '2026-12-31').dropna()), n_win=int(x50.notna().sum()))
out['vs_pool_EW_and_0050_windows_REAL_opt_javg'] = ew
M = pd.DataFrame(mine)
D = pd.concat([FX, M[V8].add_prefix('OCAS:')], axis=1)
m = M.notna().all(axis=1).to_numpy()
D = D[m].fillna(0.0); Xf = D.to_numpy()
sr_all = Xf.mean(0) / Xf.std(0, ddof=1); sr_all = sr_all[np.isfinite(sr_all)]
et = G.SP.eff_trials(Xf[:, np.std(Xf, 0) > 0])
dsr = {}
for tg in ['BASE'] + V8:
    x = M[tg].to_numpy()[m]
    dsr[tg] = {f'DSR_N{n}': G.SP.dsr(x, sr_all, n)['DSR'] for n in (2581, 2627)}
    dsr[tg]['DSR_LiJi'] = G.SP.dsr(x, sr_all, et['M_eff_LiJi'])['DSR']; dsr[tg]['SR_m'] = float(x.mean() / x.std(ddof=1))
out['G3_N'] = dict(prereg_N=2581, center_N_before_this_topic_now=2619, cumulative_N_after=2627, N_matrix=int(Xf.shape[1]), M_eff_LiJi=float(et['M_eff_LiJi']),
                   note='PREREG 寫於 N＝2,573 時（→2,581）；本題暫停期間研究中心另登記外部貼文 46 條（2,573→2,585→2,619），本題 8 條加上後累計 2,627。矩陣只含可取得序列（2,334＋8）。',
                   DSR=dsr)

# (6) P1～P7
O = R['OOS']; V = IT['versions']
g = lambda tg, c: V[tg]['m'][c]['worst']['med']
tr = lambda tg: V[tg]['m']['track_N100']
bt = V['BASE_track']['track_N100']
P = {}
P['P1'] = dict(CA80=dict(pre_cost_sel=g('CA80', 'OFF100'), post_cost_sel=g('CA80', 'N100')), CA90=dict(pre_cost_sel=g('CA90', 'OFF100'), post_cost_sel=g('CA90', 'N100')),
               passed=bool(all(abs(g(t, 'OFF100')) <= 0.5 and g(t, 'N100') >= 0.5 for t in ('CA80', 'CA90'))),
               mechanism_rejected=bool(any(g(t, 'OFF100') < -0.5 for t in ('CA80', 'CA90'))))
e3 = EX['E3_P2_righttail']
ne = (e3['n_tail'] / 4) * ((e3['n_slots'] - e3['n_tail']) / 4) / (e3['n_slots'] / 4)
ks_p_eff = float(sps.kstwobign.sf(math.sqrt(ne) * e3['ks_stat']))
P['P2'] = dict(chat_diff_pct=e3['chat_diff'] * 100, ks_p_raw=e3['ks_p'], ks_stat=e3['ks_stat'], ks_p_neff_div4=ks_p_eff,
               passed_mean=bool(abs(e3['chat_diff']) < 0.0015), passed_ks=bool(ks_p_eff > 0.10), passed=bool(abs(e3['chat_diff']) < 0.0015 and ks_p_eff > 0.10))
aux = {k: v for k, v in O['paired'].items() if k.startswith('AUX')}
P['P3'] = dict(aux_full={k: v['full']['med'] for k, v in aux.items() if k.split('|')[1] in ('CA80', 'CA90')},
               aux_sel={k: v['selection']['med'] for k, v in aux.items() if k.split('|')[1] in ('CA80', 'CA90')},
               N100_full={t: O['paired'][f'N100|{t}']['full']['worst']['med'] for t in ('CA80', 'CA90')},
               subsamples=(R.get('POST') or {}).get('P3_subsamples_selection'))
P['P4'] = dict(MT_N100_pre_odd=O['paired']['N100|MT']['pre_odd']['med'], MT_N100_post_odd=O['paired']['N100|MT']['post_odd']['med'],
               MT_aux={k: dict(pre_odd=v['pre_odd']['med'], post_odd=v['post_odd']['med'], full=v['full']['med']) for k, v in aux.items() if k.endswith('|MT')},
               order_day_share_full=dict(BASE=O['track']['N100|BASE']['full']['ord'], MT=O['track']['N100|MT']['full']['ord']),
               order_day_share_sel=dict(BASE=bt['ord'], MT=tr('MT')['ord']),
               passed_order_days=bool(O['track']['N100|MT']['full']['ord'] <= 0.25))
P['P5'] = dict(cost_saved_sel_pp=bt['cost_total_pp'] - tr('NR')['cost_total_pp'],
               cost_saved_full_pp=O['track']['N100|BASE']['full']['cost_total_pp'] - O['track']['N100|NR']['full']['cost_total_pp'],
               paired_sel=g('NR', 'N100'), MDE_sel=IT['MDE_selection_pp'], passed=bool(bt['cost_total_pp'] - tr('NR')['cost_total_pp'] <= 0.4),
               reading='低於 MDE → 無資訊' if abs(g('NR', 'N100')) < IT['MDE_selection_pp'] else '高於 MDE')
P['P6'] = dict(DC5C_pre_cost_sel=g('DC5C', 'OFF100'), DC5E_minus_DC5C_sel=g('DC5E', 'N100') - g('DC5C', 'N100'),
               passed=bool(g('DC5C', 'OFF100') >= -0.3 and g('DC5E', 'N100') - g('DC5C', 'N100') >= -0.3))
P['P7'] = dict(CB1_pre=g('CB1', 'OFF100'), CB2_pre=g('CB2', 'OFF100'), CB1_post=g('CB1', 'N100'), CB2_post=g('CB2', 'N100'), CA_single_post=g(IT['combo_choice']['CA'], 'N100'),
               passed=bool(g('CB1', 'OFF100') >= -0.5 and g('CB2', 'OFF100') >= -0.5 and g('CB1', 'N100') >= g(IT['combo_choice']['CA'], 'N100') and g('CB2', 'N100') >= g(IT['combo_choice']['CA'], 'N100')))
P['weaker'] = dict(MT_seg0612=O['paired']['N100|MT']['seg0612']['med'], MT_full=O['paired']['N100|MT']['full']['worst']['med'])
out['predictions_check'] = P

r = json.loads(RES.read_text(encoding='utf-8')); r['EXTRA'] = out
RES.write_text(CL.dumps(r), encoding='utf-8')
print(CL.dumps({k: out[k] for k in ('G3_N', 'predictions_check')})[:6000])

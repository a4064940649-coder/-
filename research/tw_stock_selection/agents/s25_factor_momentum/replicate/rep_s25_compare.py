#!/usr/bin/env python3
"""S25 獨立複製：把 a1.json／a23.json／b.json（scratchpad）併成 replicate/results_rep.json，並與原研究存檔數字逐項對照。
原研究數字只讀 results_describe.json／results_train.json／results.json／FINDINGS.md（不讀程式）。
事前標準：年化差與超額差（%、百分點）≤ 0.3 個百分點、t 差 ≤ 0.3，判定相同 → 一致。python3 -I -B rep_s25_compare.py"""
import json, hashlib
from pathlib import Path

H = Path(__file__).resolve().parent
OR = H.parent
REP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s25rep')
a1 = json.loads((REP / 'a1.json').read_text()); a23 = json.loads((REP / 'a23.json').read_text()); b = json.loads((REP / 'b.json').read_text())
od = json.loads((OR / 'results_describe.json').read_text()); ot = json.loads((OR / 'results_train.json').read_text()); ov = json.loads((OR / 'results.json').read_text())
TOL = {'pct': 0.3, 't': 0.3, 'sharpe': 0.3, 'n': 0}
rows = []


def add(item, orig, rep, kind):
    d = rep - orig
    ok = abs(d) <= TOL[kind] + 1e-12
    rows.append(dict(item=item, original=orig, replicated=rep, diff=d, kind=kind, tol=TOL[kind], consistent=bool(ok)))


A = a1['A1']; OA = od['A1']
for f in ('SMB', 'HML', 'EPF', 'MOM', 'RMOM'):
    add(f'A1 {f} 差（%／月）', OA['per_factor'][f]['diff_pct'], A['per_factor'][f]['diff_pct'], 'pct')
    add(f'A1 {f} 差 Welch t', OA['per_factor'][f]['diff_welch_t'], A['per_factor'][f]['diff_welch_t'], 't')
add('A1 合併 趨勢>0（%／月）', OA['pooled']['win_mean_pct'], A['pooled']['win_mean_pct'], 'pct')
add('A1 合併 趨勢≤0（%／月）', OA['pooled']['lose_mean_pct'], A['pooled']['lose_mean_pct'], 'pct')
add('A1 合併 差（%／月）', OA['pooled']['diff_pct'], A['pooled']['diff_pct'], 'pct')
add('A1 合併 差 t（月時序）', OA['pooled']['diff_t_monthly_ts'], A['pooled']['diff_t_monthly_ts'], 't')
add('A1 合併 趨勢>0 筆數', OA['pooled']['win_n'], A['pooled']['win_n'], 'n')
add('A1 合併 趨勢≤0 筆數', OA['pooled']['lose_n'], A['pooled']['lose_n'], 'n')
for s in ('WIN', 'EW', 'WML'):
    add(f'A1 TSFM {s} 年化平均（%）', OA['tsfm'][s]['mean_ann_pct'], A['tsfm'][s]['mean_ann_pct'], 'pct')
    add(f'A1 TSFM {s} Sharpe', OA['tsfm'][s]['sharpe'], A['tsfm'][s]['sharpe'], 'sharpe')
add('A1 TSFM WIN 年化波動（%）', OA['tsfm']['WIN']['vol_ann_pct'], A['tsfm']['WIN']['vol_ann_pct'], 'pct')
add('A1 TSFM EW 年化波動（%）', OA['tsfm']['EW']['vol_ann_pct'], A['tsfm']['EW']['vol_ann_pct'], 'pct')
add('A1 WIN−EW Sharpe 差', OA['tsfm']['WIN_minus_EW_sharpe']['diff'], A['tsfm']['WIN_minus_EW_sharpe']['diff'], 'sharpe')
add('A1 WIN 對 EW α 的 NW t', OA['tsfm']['WIN_on_EW']['t'][0], A['tsfm']['WIN_on_EW']['t'][0], 't')
sb, osb = a1['A1_strategy_by_trend']['RMOM'], od['A1_strategy_by_trend']['RMOM']
add('A1 RMOM 趨勢≤0 現行對 0050 月超額（%）', osb['lose_ex_mean_pct'], sb['lose_ex_mean_pct'], 'pct')
add('A1 RMOM 趨勢>0 現行對 0050 月超額（%）', osb['win_ex_mean_pct'], sb['win_ex_mean_pct'], 'pct')
add('A1 RMOM 分組差 Welch t', osb['diff_welch_t'], sb['diff_welch_t'], 't')
for nm, k in (('A1 穩健 6 月趨勢 合併差', 'trend6'), ('A1 穩健 去 2020 合併差', 'ex2020'), ('A1 穩健 2019 起 合併差', 'since2019')):
    add(nm + '（%／月）', od['A1_robust'][k]['pooled']['diff_pct'], a1['A1_robust'][k]['pooled']['diff_pct'], 'pct')
    add(nm + ' t', od['A1_robust'][k]['pooled']['diff_t_monthly_ts'], a1['A1_robust'][k]['pooled']['diff_t_monthly_ts'], 't')

for w in ('VW', 'EW'):
    R, O = a23['A2'][w]['regs'], od['A2'][w]['regs']
    for rg in ('IND_on_const', 'IND_on_WML', 'IND_on_WML_MKT', 'IND_on_WML_MKT_MOM', 'WML_on_const', 'WML_on_IND'):
        add(f'A2 {w} {rg} α 年化（%）', O[rg]['alpha_ann_pct'], R[rg]['alpha_ann_pct'], 'pct')
        add(f'A2 {w} {rg} α t', O[rg]['alpha_t'], R[rg]['alpha_t'], 't')
add('A2 WML 與 MOM 相關', od['A2']['VW']['corr_WML_MOM'], a23['A2']['VW']['corr_WML_MOM'], 'sharpe')

add('A3 驗收 2019 起年化（%）', od['A3_acceptance']['since2019'] * 100, a23['A3_acceptance']['since2019'] * 100, 'pct')
add('A3 驗收 全期年化（%）', od['A3_acceptance']['full'] * 100, a23['A3_acceptance']['full'] * 100, 'pct')
add('A3 驗收 最大回撤（%）', od['A3_acceptance']['maxdd'] * 100, a23['A3_acceptance']['maxdd'] * 100, 'pct')
add('A3 交易筆數', od['A3']['n_trades'], a23['A3']['n_trades'], 'n')
add('A3 同業≥5 筆數', od['A3']['all']['n_peer_ge5'], a23['A3']['all']['n_peer_ge5'], 'n')
add('A3 同業家數中位數', od['A3']['peer_count_U']['median'], a23['A3']['peer_count_U']['median'], 'n')
MK = 'main（同業≥5，U 股池等權）'
lab = {'ex50': '對 0050 超額', 'exP_U': '對同業超額', 'peer50_U': '同業 − 0050', 'ret': '個股毛報酬', 'peer_U': '同業籃子'}
for per, pn in (('all', '全期'), ('train', '訓練'), ('valid', '驗證')):
    for c in ('ex50', 'exP_U', 'peer50_U') + (('ret', 'peer_U') if per == 'all' else ()):
        o, r = od['A3'][per][MK][c], a23['A3'][per]['main'][c]
        add(f'A3 {pn} {lab[c]} 平均（%／筆）', o['mean_pct'], r['mean_pct'], 'pct')
        if per == 'all' or c == 'exP_U':
            add(f'A3 {pn} {lab[c]} 群聚 t', o['t_cluster_month'], r['t_cluster_month'], 't')
    o, r = od['A3'][per][MK]['exP_U'], a23['A3'][per]['main']['exP_U']
    add(f'A3 {pn} 對同業 去最賺 5 筆 平均（%）', o['drop_top5_mean_pct'], r['drop_top5_mean_pct'], 'pct')
    add(f'A3 {pn} 對同業 去最賺 5 筆 群聚 t', o['drop_top5_t_cluster'], r['drop_top5_t_cluster'], 't')
o, r = od['A3']['all'][MK]['exP_U'], a23['A3']['all']['main']['exP_U']
add('A3 全期 對同業 中位數（%）', o['median_pct'], r['median_pct'], 'pct')
add('A3 全期 對同業 勝率（%）', o['win_rate'] * 100, r['win_rate'] * 100, 'pct')
add('A3 全期 對 0050 逐筆 t', od['A3']['all'][MK]['ex50']['t'], a23['A3']['all']['main']['ex50']['t'], 't')
add('A3 全期 對同業 逐筆 t', o['t'], r['t'], 't')
add('A3 穩健 全上市同業 對同業超額（%）', od['A3']['all']['robust_BROAD']['exP']['mean_pct'], a23['A3']['all']['robust_BROAD']['exP']['mean_pct'], 'pct')
add('A3 穩健 全上市同業 筆數', od['A3']['all']['robust_BROAD']['n'], a23['A3']['all']['robust_BROAD']['exP']['n'], 'n')
add('A3 穩健 市值加權同業 對同業超額（%）', od['A3']['all']['robust_VW_U']['mean_pct'], a23['A3']['all']['robust_VW_U']['mean_pct'], 'pct')
add('A3 全部 387 筆 對 0050 超額（%）', od['A3']['all']['all_trades_ex50']['mean_pct'], a23['A3']['all']['all_trades_ex50']['mean_pct'], 'pct')
add('A3 航運業 同業 − 0050（%）', od['A3']['by_industry']['航運業']['peer50_mean_pct'], a23['A3']['by_industry']['航運業']['peer50_mean_pct'], 'pct')

S, OS = b['train']['summary'], ot['summary']
for c in ('RMOM_STOP', 'RMOM_HALF', 'MOM_HALF', 'MOM_STOP'):
    add(f'B 訓練 {c} 分數（百分點）', OS[c]['score_median_pp'], S[c]['score_median_pp'], 'pct')
    add(f'B 訓練 {c} 改善 ≥0.25 路徑數', OS[c]['n_imp_ge_min'], S[c]['n_imp_ge_min'], 'n')
    add(f'B 訓練 {c} 觸發月數（位移 0）', OS[c]['months_triggered_shift0'], S[c]['months_triggered_shift0'], 'n')
    add(f'B 訓練 {c} 少買檔數（位移 0）', OS[c]['f1_dropped_shift0'], S[c]['f1_dropped_shift0'], 'n')
add('B CUR 訓練窗年化中位數（%）', OS['CUR']['train_median'] * 100, S['CUR']['train_median'] * 100, 'pct')
pv, op = b['valid']['pairs_FZ_vs_CUR'], ov['pairs']['FZ_vs_CUR|fresh']
add('B 重新起跑資金比例', ov['acceptance']['ratio_nav_TB'], b['fresh_capital_ratio'], 'sharpe')
add('B V1 年化差 > 0 路徑數', op['valid']['npos'], pv['valid']['npos'], 'n')
add('B V2 配對差中位數（百分點）', op['valid']['median_pp'], pv['valid']['median_pp'], 'pct')
ob = ov['pairs'].get('FZ_vs_CUR|fresh_blk', {}).get('valid', {}).get('median_pp')
if ob is None:
    ob = (ov['paths_median']['FZ|fresh_blk']['valid'] - ov['paths_median']['CUR|fresh_blk']['valid']) * 100
add('B V3 禁買聯集後配對差中位數（百分點）', ob, pv['blk']['median_pp'], 'pct')
add('B V4 2022 段回撤差中位數（百分點）', op['seg']['2022']['median_pp'], pv['seg']['2022']['median_pp'], 'pct')
add('B V4 2024-07～2025-04 段回撤差中位數（百分點）', op['seg']['2024-07~2025-04']['median_pp'], pv['seg']['2024-07~2025-04']['median_pp'], 'pct')
add('B 主跑 CUR 驗證窗年化（%）', ov['main']['CUR|fresh']['windows']['valid']['cagr'] * 100, b['main_fresh']['CUR']['valid'] * 100, 'pct')
add('B 連續 凍結−CUR 2019 起差中位數（百分點）', ov['pairs']['FZ_vs_CUR|cont']['since2019']['median_pp'], b['cont_FZ_vs_CUR']['since2019']['median_pp'], 'pct')

cats = dict(
    frozen=(ot['selection']['frozen'], b['train']['frozen_rep']), order=(ot['selection']['order'], b['train']['order']),
    T0=(ot['selection']['T0_pass'], b['train']['T0_pass']),
    verdict={k: (ov['verdict'][k], b['verdict'][k]) for k in ('T0', 'V1', 'V2', 'V3', 'V4', 'overall')},
    trig_months_valid=(ov['triggered_months']['valid'], b['train']['rule_effect']['RMOM_STOP']['+0']['trig_months_valid']),
    V3_block_union_codes=(sorted(ov['V3_block_union']), sorted(b['V3_block_union'])),
    A0_ok=(od['A3_acceptance']['ok'], a23['A3_acceptance']['ok']))
cat_ok = {k: (v[0] == v[1]) if isinstance(v, tuple) else all(x[0] == x[1] for x in v.values()) for k, v in cats.items()}
n_ok = sum(r['consistent'] for r in rows)
res = dict(
    note='S25 獨立複製（複製員自寫；未 import／執行／讀取 s25lib.py、prep.py、describe.py、train.py、validate.py、posthoc.py）。引擎 s20lib→s17lib→s10lib 預設旗標。',
    criterion='年化差與超額差 ≤ 0.3 個百分點、t 差 ≤ 0.3（Sharpe、相關、比例同用 0.3）、筆數／路徑數須相同，判定相同 → 一致',
    summary=dict(n_numeric=len(rows), n_consistent=n_ok, max_abs_diff_pct=max(abs(r['diff']) for r in rows if r['kind'] == 'pct'),
                 max_abs_diff_t=max(abs(r['diff']) for r in rows if r['kind'] == 't'), categorical_all_same=all(cat_ok.values()),
                 conclusion='成功' if (n_ok == len(rows) and all(cat_ok.values())) else '部分'),
    comparisons=rows, categorical=dict(values=cats, same=cat_ok),
    not_compared_or_differs=dict(
        bootstrap_ci90=dict(original=od['A1']['tsfm']['WIN_minus_EW_sharpe']['ci90'], replicated=a1['A1']['tsfm']['WIN_minus_EW_sharpe']['ci90'],
                            note='區塊 bootstrap 抽法（循環區塊、起點抽法）SPEC 未寫死，複製員自寫；點估計相同，區間下界不同（−0.43 vs −0.59），結論（含 0）相同'),
        peer_count_q90=dict(original=od['A3']['peer_count_U']['q90'], replicated=a23['A3']['peer_count_U']['q90'], note='分位數插值口徑差，中位數 32、q10 11 相同'),
        random_control=dict(original_FINDINGS='第 45 百分位', replicated=b['random_control']['FZ_percentile'], note='複製員用種子 20261116＋i（PREREG 未寫死抽法）'),
        CUR_fresh_trades=dict(original_closed=ov['main']['CUR|fresh']['trades']['n'], replicated_incl_open_end=b['main_fresh']['CUR']['n_trades'],
                              note='原研究交易統計只算已平倉；複製員的 205 含期末未平倉 8 筆')),
    A1=a1, A2=a23['A2'], A3=a23['A3'], A3_acceptance=a23['A3_acceptance'],
    B={k: v for k, v in b.items() if not k.startswith('paths_')}, B_paths=dict(cont=b['paths_cont'], fresh=b['paths_fresh'], fresh_blk=b['paths_fresh_blk']),
    meta=dict(a23=a23['meta'], b=b['meta'],
              code_sha256={f: hashlib.sha256((H / f).read_bytes()).hexdigest() for f in
                           ('rep_s25_stats.py', 'rep_s25_lib.py', 'rep_s25_a1.py', 'rep_s25_a23.py', 'rep_s25_b.py', 'rep_s25_compare.py')}))
(H / 'results_rep.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(res['summary'], ensure_ascii=False), cat_ok)
for r in rows:
    if not r['consistent'] or abs(r['diff']) > 1e-6:
        print('差', r['item'], round(r['original'], 4), round(r['replicated'], 4), round(r['diff'], 4), r['consistent'])

#!/usr/bin/env python3
"""數據稽核部｜已發布數字總對帳（第 2、3 部分）：風險表逐列對帳（表 b）＋文件數字對帳（表 c）。
前置：rerun_official.py、audit_core.py 已跑完；scratchpad/audit_pub/resync/run/ 下有用「official_nav/resync/harness.py 的機械轉出版」
（只改暫存與輸出路徑）重跑的 S7、S9、S27、S42 正式口徑結果（重跑完後摘要存 out/resync_rerun_extract.json，暫存可刪）。
用法：python3 -I -B audit_tables.py   → table_b_risk_rows.csv、table_c_doc_numbers.csv、table_c_stale_conflicts.csv、results.json"""
import json, gzip, csv, math, ast, re, statistics as st_
from pathlib import Path
from datetime import date

HERE = Path(__file__).resolve().parent
AG = HERE.parent
ROOT = Path('/home/user/-')
SITE = ROOT / 'xq_scripts' / 'site'
OUT = HERE / 'out'
RUN = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/audit_pub/resync/run')
J = lambda p: json.load(open(p, encoding='utf-8'))
pp = lambda x: float(x) * 100
core = J(OUT / 'core.json'); rr = J(OUT / 'rerun_res_official.json'); pub = J(AG / 'official_nav' / 'out' / 'res_official.json')
RS = J(AG / 'official_nav' / 'resync' / 'resync_results.json')['new']
E = core['independent_metrics']['E0|W10_70']; B0 = core['independent_metrics']['0050']
e0 = rr['strats']['E0|W10_70']; ex = e0['extras']; ps = e0['paths_summary']
LAG = {'A10': 0.0, 'A11': 0.6201550387596899, 'A12': 1.2945736434108528, 'A13': 1.9767441860465116, 'A15': 3.356589147286822}


# ---------------------------------------------------------------- 重跑結果的摘要（欄位選取同 resync/extract.py；迴歸自己寫最小平方法）
def ols(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def extract():
    X = {}
    r = J(RUN / 's7_official' / 'results.json'); q = J(RUN / 's7_official' / 'results_posthoc.json')['Q7_delay_capitals']['W10_70L']; c = r['cells']['W10_70L']
    X['S7'] = dict(late1=-q['1']['median_cost_pp'], late2=-q['2']['median_cost_pp'], late2_maxdd=pp(c['delay']['2']['since2019']['maxdd']))
    r = J(RUN / 's9_official' / 'results.json'); s3 = r['family']['W10_70|open']['S3']; xs, ys = [], []
    for d, lag in LAG.items():
        for k, v in r['combos'][f'{d}|open|W10_70']['paths'].items():
            xs.append(lag + int(k.split('+')[1])); ys.append(v * 100)
    a, b = ols(xs, ys)
    X['S9'] = dict(late_median=pp(s3['fam2_r19_median']), late_p10=pp(s3['fam2_r19_p10']), late_dd_median=pp(s3['fam2_maxdd_median']), late_dd_p10=pp(s3['fam2_maxdd_p10']),
                   ontime=a, slope=b, A10_median=pp(s3['A10_paths_r19_median']))
    r = J(RUN / 's27_official' / 'results.json'); p3 = r['part3_alerts']; sd = p3['history_check_std']; m = r['part2_simulation']['main_2016_b6to12']
    X['S27'] = dict(now={k: sd[k]['current_2026_10'] for k in 'ABCDE'}, thresholds=p3['thresholds_fa10'], hist=p3['history_check_summary'],
                    scen={sc: {h: dict(p10=m['scenarios'][sc][h]['strat_ann']['p10'], p50=m['scenarios'][sc][h]['strat_ann']['p50'], p90=m['scenarios'][sc][h]['strat_ann']['p90'],
                                       lose=m['scenarios'][sc][h]['p_lose_to_0050'], b10=m['scenarios'][sc][h]['b0050_ann']['p10'], b50=m['scenarios'][sc][h]['b0050_ann']['p50'],
                                       b90=m['scenarios'][sc][h]['b0050_ann']['p90']) for h in ('12m', '36m', '60m')} for sc in ('EFF', 'HALF', 'ZERO')},
                    alerts={k: v['prob'] for k, v in m['alerts']['fa10'].items()}, events=m['scenarios']['EFF']['events_5y'])
    r = J(RUN / 's42_official' / 'results.json'); g = r['grid']
    X['S42'] = dict(neutral=pp(g['COMB|0.004|LATE']['since2019']['median']), neutral_dd=pp(g['COMB|0.004|LATE']['maxdd_full']['median']), neutral_dd_p10=pp(g['COMB|0.004|LATE']['maxdd_full']['p10']),
                    half=pp(g['COMB|0.004|LATE']['HALF_since2019']['median']), half_dd=pp(g['COMB|0.004|LATE']['HALF_maxdd_full']['median']),
                    cons=pp(g['COMB|0.005|LATE']['HALF_since2019']['p10']), cons_dd=pp(g['COMB|0.005|LATE']['HALF_maxdd_full']['p10']),
                    opt=pp(g['COMB|0.003|ON']['since2019']['median']), opt_dd=pp(g['COMB|0.003|ON']['maxdd_full']['median']),
                    stack=[(x['step'], pp(x['since2019_median']), x['d_vs_CUR_since2019_median_pp'], pp(x['since2019_standard'])) for x in r['stack']],
                    paired={k: (v['median'] if isinstance(v, dict) else v) for k, v in r['paired_effects'].items()},
                    fpm_window_pp=r['acceptance']['A1_FPM']['d_window_pp'], cur_std=pp(r['acceptance']['A0']['since2019']))
    return X


if (RUN / 's42_official' / 'results.json').exists():
    X = extract(); json.dump(X, open(OUT / 'resync_rerun_extract.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
else:
    X = J(OUT / 'resync_rerun_extract.json')
S7, S9, S27, S42 = X['S7'], X['S9'], X['S27'], X['S42']


# ---------------------------------------------------------------- S27 現值：從 navs_official 自己算（月底口徑）
def read_nav(p):
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        rd = csv.reader(f); hdr = next(rd); rows = list(rd)
    return [date.fromisoformat(r[0][:10]) for r in rows], {h: [float(r[i]) for r in rows] for i, h in enumerate(hdr) if i}


dp, cp = read_nav(SITE / 'navs_official.csv.gz')
me = [i for i in range(len(dp)) if i == len(dp) - 1 or (dp[i + 1].year, dp[i + 1].month) != (dp[i].year, dp[i].month)]
s = [cp['E0|W10_70'][i] for i in me]; b = [cp['0050'][i] for i in me]; rel = [x / y for x, y in zip(s, b)]
A_now = (s[-1] / s[-37]) ** (1 / 3) - (b[-1] / b[-37]) ** (1 / 3)
B_now = (s[-1] / s[-13]) - (b[-1] / b[-13])
C_now = s[-1] / max(s) - 1
E_now = rel[-1] / max(rel) - 1
D_now = len(rel) - 1 - max(i for i, x in enumerate(rel) if x == max(rel))
sd = cp['E0|W10_70']; C_daily = sd[-1] / max(sd) - 1
IND27 = dict(A=pp(A_now), B=pp(B_now), C=pp(C_now), E=pp(E_now), D=D_now, C_daily=pp(C_daily))

# ---------------------------------------------------------------- S46：讀 s46 results.json 的 45 格 × 20 條，自己算家族統計
s46 = J(AG / 's46_momentum_gate_family' / 'results.json')
cellmed = {k: st_.median([p['since2019'] for p in v['paths'].values()]) for k, v in s46['cells'].items()}
allp = sorted(p['since2019'] for v in s46['cells'].values() for p in v['paths'].values())


def pct(sorted_v, q):                           # 線性內插分位數（同 numpy 預設）
    h = (len(sorted_v) - 1) * q; lo = int(math.floor(h)); hi = min(lo + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (h - lo)


cur46 = cellmed['g10_L20_D10']
F46 = dict(n_cells=len(cellmed), n_paths=len(allp), all_median=pp(st_.median(allp)), all_p10=pp(pct(allp, 0.10)), fam_median=pp(st_.median(cellmed.values())),
           cur=pp(cur46), cur_minus_median=pp(cur46 - st_.median(cellmed.values())), rank=1 + sum(1 for v in cellmed.values() if v > cur46))
r46 = list(csv.DictReader(open(AG / 'r46_replicate_s46' / 'compare.csv', encoding='utf-8-sig')))
r64 = list(csv.DictReader(open(AG / 'r64_replicate_retail' / 'compare.csv', encoding='utf-8-sig')))
r65 = list(csv.DictReader(open(AG / 'r65_replicate_s63_s65' / 'compare.csv', encoding='utf-8-sig')))
R64 = {x['項目']: float(x['重現值']) for x in r64}; R65 = {x['項目']: float(x['重現值']) for x in r65}
s64 = J(AG / 's64_retail_behavior_tax' / 'results.json'); SC = s64['scenarios']
s39 = J(AG / 's39_first_print_revenue' / 'results.json')

# ---------------------------------------------------------------- 表 b：風險表逐列
src_py = (SITE / 'build_site_data.py').read_text(encoding='utf-8')
tree = ast.parse(src_py)
RISK = [ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'RISK'][0]
riskj = J(SITE / 'data' / 'risk.json')['rows']
assert len(RISK) == len(riskj)
const_vs_json = all(a[0] == r['項目'] and a[1] == r['數值'] and a[3] == r['說明'] and a[4] == r['出處'] for a, r in zip(RISK, riskj))
P_OFF = 'agents/official_nav/out/res_official.json'; P_RS = 'agents/official_nav/resync/resync_results.json'
I1 = '獨立重算（讀 navs_official.csv.gz，自寫年化／回撤）'; I2 = '獨立重算（讀重跑交易紀錄，由成本／得款自算）'
G1 = '同引擎重跑（rerun_official.py）逐位相同'; G2 = '同程式重跑（harness 轉出版，正式口徑）'
NOREP = '未重現（正式口徑沒有另一位研究員獨立重現；本次只是同程式重跑）'
# 項目 → (出處檔, 出處值, 本次重算值, 重算方式, 重現狀態, 容許, 備註)
SPEC = {
    '回撤_單一路徑': (P_OFF, pp(pub['strats']['E0|W10_70']['std']['maxdd']), pp(E['maxdd']), I1 + '；' + G1, '已重現（本次獨立重算）', 0.05, f"{E['dd_peak']}～{E['dd_trough']}"),
    '回撤_20條中位數': (P_OFF, pp(pub['strats']['E0|W10_70']['paths_summary']['maxdd_median']), pp(ps['maxdd_median']), G1, '已重現（R64 重現 20 條基準中位數；本次同引擎重跑）', 0.05, ''),
    '回撤_壞的時候': (P_RS, RS['S9']['late_dd_p10'], S9['late_dd_p10'], G2, NOREP, 0.05, ''),
    '0050_回撤': (P_OFF, pp(pub['strats']['E0|W10_70']['extras']['e50_maxdd']), pp(B0['maxdd']), I1, '已重現（本次獨立重算）', 0.05, f"{B0['dd_peak']}～{B0['dd_trough']}"),
    '落後0050最多': (P_OFF, -pp(pub['strats']['E0|W10_70']['extras']['roll252_worst']), None, I1, '已重現（本次獨立重算）', 0.05, ''),
    '落後時間比例': (P_OFF, pp(pub['strats']['E0|W10_70']['extras']['roll252_frac_behind']), None, I1, '已重現（本次獨立重算）', 0.05, ''),
    '最賺5筆占比': (P_OFF, pp(pub['strats']['E0|W10_70']['extras']['top5_pnl_share']), pp(core['trades']['top5_share']), I2, '已重現（本次由交易紀錄重算；交易紀錄本身是同引擎）', 0.5, ''),
    '年化_全部': (P_OFF, pp(pub['strats']['E0|W10_70']['std']['cagr_since2019']), pp(E['c19']), I1 + '；' + G1, '已重現（R46、R55、R64、R65、R72 都重現 48.76；本次獨立重算）', 0.005, ''),
    '年化_不買最賺5檔': (P_OFF, pp(pub['strats']['E0|W10_70']['extras']['block5_since2019']), pp(ex['block5_since2019']), G1, NOREP, 0.005, ''),
    '勝率': (P_OFF, pp(pub['strats']['E0|W10_70']['extras']['win_rate']), pp(core['trades']['win_rate']), I2, '已重現（本次由交易紀錄重算）', 0.05, ''),
    '賺的平均': (P_OFF, pp(pub['strats']['E0|W10_70']['extras']['avg_win']), pp(core['trades']['avg_win']), I2, '已重現（本次由交易紀錄重算）', 0.005, ''),
    '賠的平均': (P_OFF, pp(pub['strats']['E0|W10_70']['extras']['avg_loss']), pp(core['trades']['avg_loss']), I2, '已重現（本次由交易紀錄重算）', 0.005, ''),
    '晚1天少': (P_RS, RS['S7']['late1_cost_5cap_median_pp'], S7['late1'], G2, NOREP, 0.05, ''),
    '晚2天少': (P_RS, RS['S7']['late2_cost_5cap_median_pp'], S7['late2'], G2, NOREP, 0.05, ''),
    '晚2天回撤': (P_RS, RS['S7']['late2_maxdd'], S7['late2_maxdd'], G2, NOREP, 0.05, ''),
    '準時可預期': (P_RS, RS['S9']['ontime_intercept'], S9['ontime'], G2 + '；迴歸截距自寫最小平方法', NOREP, 0.05, ''),
    '有時晚可預期': (P_RS, RS['S9']['late_median'], S9['late_median'], G2, NOREP, 0.5, f"中位數 {S9['late_median']:.2f}、差的 10% {S9['late_p10']:.2f}（說明寫 42.9／40.2）"),
    '警戒A_3年超額': (P_RS, RS['S27']['A_now'], IND27['A'], '獨立重算（navs_official 月底值）；' + G2 + f"＝{pp(S27['now']['A']):.2f}", '已重現（本次獨立重算；S27 舊口徑另有獨立複製）', 0.05, ''),
    '警戒E_相對前高': (P_RS, RS['S27']['E_now'], IND27['E'], '獨立重算（navs_official 月底值）；' + G2 + f"＝{pp(S27['now']['E']):.2f}", '已重現（本次獨立重算）', 0.05, ''),
    '警戒B_1年超額': (P_RS, RS['S27']['B_now'], IND27['B'], '獨立重算（navs_official 月底值）；' + G2 + f"＝{pp(S27['now']['B']):.2f}", '已重現（本次獨立重算）', 0.05, ''),
    '警戒C_月底回撤': (P_RS, RS['S27']['C_now'], IND27['C'], '獨立重算（navs_official 月底值）；' + G2 + f"＝{pp(S27['now']['C']):.2f}", '已重現（本次獨立重算）', 0.05, f"日內 {IND27['C_daily']:.2f}（說明寫 −10.6）"),
    '紅燈規則': ('agents/s27_expectation_ranges/results.json', 2, 2, '規則常數；說明裡「5 年誤報約 7%」' + G2 + f"＝{pp(S27['alerts']['TWO_OR_MORE_posthoc']['EFF']['5y']):.2f}%", '誤報率未重現（正式口徑）；舊口徑 S27 已獨立複製', 0, ''),
    '實盤可預期_中性': ('agents/s42_realistic_expectation/results.json＋' + P_RS, RS['S42']['neutral'], S42['neutral'], G2, '未重現（S42 本身沒有獨立重現；只有它依賴的 S39、S40 有）', 0.05,
                 f"回撤中位數 {S42['neutral_dd']:.2f}、壞時 {S42['neutral_dd_p10']:.2f}"),
    '實盤可預期_保守': ('agents/s42_realistic_expectation/results.json＋' + P_RS, RS['S42']['half'], S42['half'], G2, '未重現（同上）', 0.05,
                 f"保守 {S42['cons']:.2f}、回撤 {S42['cons_dd']:.2f}（說明寫 34.0、約 −40）"),
    '營收更正高估': ('agents/s39_first_print_revenue/results.json', pp(s39['engine']['standard']['FPM']['cagr_window'] - s39['engine']['standard']['CUR']['cagr_window']), S42['fpm_window_pp'],
               G2 + '（S42 正式口徑的 A1_FPM 視窗差）', '已重現（S39 有獨立複製；但 S39 的 −0.78 是只修除權息口徑，正式口徑是 −0.77）', 0.05,
               f"20 條中位數 {s39['engine']['paths_pairs']['FPM']['cagr']['median_pp']:.2f}、最多 {s39['engine']['paths_pairs']['FPM']['cagr']['min_pp']:.2f}"),
    '期限前視高估': ('agents/s40_revenue_deadline_pit/results.json；agents/deadline_engine_fix/out/results.json', None, None, '獨立重算（navs_official − navs_exright 的 2019 起年化）', '已重現（S40 有獨立複製；本次獨立重算）', 0.005, ''),
    '正常範圍_1年': (P_RS, RS['S27']['range1y_median'], pp(S27['scen']['EFF']['12m']['p50']), G2 + '（bootstrap 10,000 次，同亂數種子）', '未重現（正式口徑；舊口徑 30.5 有獨立複製）', 0.5,
                f"P10～P90 {pp(S27['scen']['EFF']['12m']['p10']):.2f}～{pp(S27['scen']['EFF']['12m']['p90']):.2f}、輸 0050 {pp(S27['scen']['EFF']['12m']['lose']):.1f}%"),
    '參數挑選後可預期': ('agents/s46_momentum_gate_family/results.json', pp(s46['families']['main45']['all_paths_median']), F46['all_median'], '讀 S46 的 45 格 × 20 條路徑值，家族統計自己算；R46 compare.csv 56 列全符', '已重現（R46）', 0.05,
                 f"現行第 {F46['rank']} 名、高出家族中位數 {F46['cur_minus_median']:.2f}（說明寫 6.25）"),
    '參數挑選_差的情況': ('agents/s46_momentum_gate_family/results.json', pp(s46['families']['main45']['all_paths_p10']), F46['all_p10'], '同上（900 條的第 10 百分位）', '已重現（R46）', 0.05, ''),
    '照做_只買一半少': ('agents/s64_retail_behavior_tax/results.json；r64 compare.csv', -SC['PICK_RAND']['d19_pp'], -R64['S64 隨機只買一半：相對照做（百分點）'], '讀 R64 compare.csv 的重現值', '已重現（R64；換一組亂數是 5.8）', 0.31,
                f"說明寫 46.4→40.1：S64 中位數 {pp(SC['PICK_RAND']['since2019']):.2f}"),
    '照做_回落20%停手少': ('agents/s64_retail_behavior_tax/results.json；r64 compare.csv', -SC['STOP_20_6M']['d19_pp'], -R64['S64 回落 20% 停手等 6 個月：相對照做（百分點）'], '讀 R64 compare.csv 的重現值', '已重現（R64）', 0.05,
                  f"說明寫 46.4→41.5：{pp(SC['STOP_20_6M']['since2019']):.2f}"),
    '照做_漏做四分之一少': ('agents/s64_retail_behavior_tax/results.json；r64 compare.csv', -SC['SKIP_25']['d19_pp'], -R64['S64 漏做 25%：相對照做（百分點）'], '讀 R64 compare.csv 的重現值', '已重現（R64；換一組亂數是 2.2）', 0.31,
                  f"說明寫 46.4→43.8：{pp(SC['SKIP_25']['since2019']):.2f}；兩個中位數相減是 {46.40 - pp(SC['SKIP_25']['since2019']):.2f}，不是 2.1"),
    '照做_開盤漲3%不買少': ('agents/s64_retail_behavior_tax/results.json；r64 compare.csv', -SC['GAP_3']['d19_pp'], -R64['S64 開盤漲超過 3% 不買：相對照做（百分點）'], '讀 R64 compare.csv 的重現值', '已重現（R64）', 0.05,
                   f"⚠ 說明寫「46.4% → 45.6%」但數值寫少 1.6：兩個中位數相減只有 {46.40 - pp(SC['GAP_3']['since2019']):.2f}；1.6 是「逐條相減後取中位數」。同一句話裡兩個口徑，讀者算不出 1.6"),
    '小資金_10萬零股超額': ('agents/s62_small_capital_retail/results.json；r64 compare.csv', None, R64['S62 10 萬 B 基準：對 0050 超額（百分點）'], '讀 R64 compare.csv 的重現值', '已重現（R64）', 0.05,
                   f"43.6 對 28.9：重現 {R64['S62 10 萬 B 基準：2019 起（%）']:.2f}"),
    '小資金_10萬只買整張超額': ('agents/s62_small_capital_retail/results.json；r64 compare.csv', None, R64['S62 10 萬只買整張：對 0050 超額（百分點）'], '讀 R64 compare.csv 的重現值', '已重現（R64）', 0.05,
                     f"29.4 對 28.9：重現 {R64['S62 10 萬只買整張：2019 起（%）']:.2f}"),
}
ta = {(x['分組'], x['項目']): x for x in csv.DictReader(open(HERE / 'table_a_official_rerun.csv', encoding='utf-8-sig'))}
find_a = lambda key: float([v for k, v in ta.items() if key in k[1]][0]['重算'])
ovr = {'落後0050最多': find_a('滾動 12 個月最多落後'), '落後時間比例': find_a('落後時間比例'), '期限前視高估': find_a('2019 起差')}
TB = []
for (item, val, unit, desc, src), rj in zip(RISK, riskj):
    f_, sv, mine, how, status, tol, note = SPEC[item]
    if item in ovr:
        mine = ovr[item]
    if item == '期限前視高估':
        sv = pp(J(AG / 'deadline_engine_fix' / 'out' / 'results.json')['standard']['ALL']['since2019']) - 48.991278775035196
    if sv is None:
        sv = mine
    paths = [p for p in re.split('[；＋]', f_)]
    exist = all((AG.parent / 'agents' / p.replace('agents/', '')).exists() if p.startswith('agents/') else True for p in [x.split('；')[0] for x in paths if x.startswith('agents/')])
    # 「出處」欄裡寫的資料夾是否存在
    cited = re.findall(r'agents/[A-Za-z0-9_/\.]+', src)
    cited_ok = all((AG / c[len('agents/'):]).exists() for c in cited)
    readme_cited = 'README.md' in src
    ok_src = abs(val - sv) <= tol + 1e-9; ok_mine = abs(val - mine) <= tol + 1e-9
    TB.append(dict(項目=item, 網站值=val, 單位=unit, 網站出處欄=src, 出處欄列的檔案都存在='是' if cited_ok else '否', 實際結果檔=f_, 結果檔存在='是' if exist else '否',
                   結果檔數值=round(sv, 4), 與結果檔一致='一致' if ok_src else '不一致', 本次重算值=round(mine, 4), 與本次重算一致='一致' if ok_mine else '不一致',
                   重算方式=how, 重現狀態=status, 備註=note))
# ---------------------------------------------------------------- 表 c：文件數字清單
TC = []
RD = 'xq_scripts/README.md'; ST = 'README.md'


def c(doc, loc, what, pubv, calc, tol, src, note='', force=None):
    if force:
        res = force
    elif calc is None:
        res = '無法驗證'
    elif isinstance(pubv, (int, float)):
        res = '一致' if abs(pubv - calc) <= tol + 1e-9 else '不一致'
    else:
        res = '一致' if pubv == calc else '不一致'
    TC.append(dict(文件=doc, 位置=loc, 數字=what, 文件寫的=pubv, 核對值=round(calc, 3) if isinstance(calc, float) else calc, 容許=tol, 核對結果=res, 出處=src, 備註=note))


A = lambda key, grp='': float([v for k, v in ta.items() if key in k[1] and grp in k[0]][0]['重算'])
L12 = '第 12 節「2026-11-17 起正式數字」'
src_a = '本次重跑＋獨立重算（表 a）'
for what, v, key, grp, tol in [('2019 年起每年', 48.76, '2019 年起每年', '現行', .005), ('2015-07 起每年', 34.40, '全期', '現行', .005), ('2024 年起每年', 70.46, '2024 年起每年', '現行', .005),
                               ('最大回撤', -32.1, '最大回撤', '現行（E0｜W10_70，100 萬，標準路徑）', .05), ('贏 0050 的年數', 9, '贏 0050 的年數', '', 0),
                               ('20 條最少', 42.01, '2019 起：最少', '', .005), ('20 條中位數', 46.40, '2019 起：中位數', '', .005), ('20 條最多', 49.29, '2019 起：最多', '', .005),
                               ('20 條回撤中位數', -33.4, '最大回撤中位數', '', .05), ('20 條回撤最深', -39.4, '最大回撤最深', '', .05),
                               ('成本加倍 2019 起', 43.11, '成本加倍：2019 起', '', .005), ('成本加倍 全期', 30.18, '成本加倍：全期', '', .005),
                               ('不買最賺 5 檔 2019 起', 44.88, '不買最賺 5 檔：2019 起', '', .005), ('不買最賺 5 檔 全期', 31.97, '不買最賺 5 檔：全期', '', .005),
                               ('前半段', 26.07, '前半段', '', .005), ('後半段', 43.22, '後半段', '', .005), ('已出場筆數', 381, '已出場筆數', '', 0), ('勝率', 50.4, '勝率', '', .05),
                               ('賺的平均', 27.43, '賺的平均', '', .005), ('賠的平均', -9.11, '賠的平均', '', .005), ('每筆期望值', 9.30, '每筆期望值', '', .0051), ('平均抱', 34, '平均抱', '', .5),
                               ('最賺 5 筆占比', 36, '最賺 5 筆占已平倉損益', '', .5), ('最多落後', 15.3, '最多落後（個百分點）', '', .05), ('落後時間比例', 20.2, '落後時間比例', '', .05),
                               ('與只修除權息的差：2019 起', -0.24, '2019 起差', '', .005), ('差：全期', -0.14, '全期差', '', .01), ('差：2024 起', -0.29, '2024 起差', '', .01),
                               ('20 條逐條差的中位數', -0.35, '逐條差的中位數', '', .005), ('20 條中位數的差', -0.14, '20 條中位數的差', '', .006),
                               ('XQ版 A10 2019 起', 45.09, 'A10：2019 起', '', .005), ('XQ版 A10 最大回撤', -38.8, 'A10：最大回撤', '', .05), ('XQ版 A10 20 條中位數', 43.90, 'A10：20 條中位數', '', .005),
                               ('XQ版 B11 2019 起', 46.36, 'B11：2019 起', '', .005), ('XQ版 B11 最大回撤', -38.1, 'B11：最大回撤', '', .05), ('XQ版 B11 20 條中位數', 44.04, 'B11：20 條中位數', '', .005)]:
    c(RD, L12, what, v, A(key, grp), tol, src_a, '精確 9.305，四捨五入在 9.30／9.31 邊界' if what == '每筆期望值' else '')
for y, v in {2015: -10.82, 2016: 15.44, 2017: 22.24, 2018: 0.95, 2019: 31.17, 2020: 100.75, 2021: 44.94, 2022: -17.45, 2023: 57.72, 2024: 63.38, 2025: 56.69, 2026: 75.24}.items():
    c(RD, L12 + ' 逐年', f'{y} 年', v, A(f'{y} 年', '逐年'), .005, src_a)
c(RD, L12, '換股日 2019-02-12 → 02-18', '2019-02-18', rr['av_changed_vs_rule'].get('2019-01'), 0, '本次重跑 av_changed_vs_rule')
c(RD, L12, '換股日 2023-04-11 → 04-13', '2023-04-13', rr['av_changed_vs_rule'].get('2023-03'), 0, '本次重跑 av_changed_vs_rule')
src_r = '本次同程式重跑（S7／S9／S27／S42 正式口徑）'
c(RD, L12 + ' 重算表', '晚 1 天少', 3.7, S7['late1'], .05, src_r); c(RD, L12 + ' 重算表', '晚 2 天少', 6.2, S7['late2'], .05, src_r)
c(RD, L12 + ' 重算表', '晚 2 天最大回撤', -35.3, S7['late2_maxdd'], .05, src_r); c(RD, L12 + ' 重算表', '有時晚：中位數', 42.9, S9['late_median'], .05, src_r)
c(RD, L12 + ' 重算表', '有時晚：差的 10%', 40.2, S9['late_p10'], .05, src_r); c(RD, L12 + ' 重算表', '有時晚：回撤中位數', -32.7, S9['late_dd_median'], .05, src_r)
c(RD, L12 + ' 重算表', '100 條回撤差的 10%', -36.4, S9['late_dd_p10'], .05, src_r); c(RD, L12 + ' 重算表', '準時可預期', 46.6, S9['ontime'], .05, src_r)
for what, v, k in [('只有衛星 70 萬：2019 起', 42.91, 'since2019'), ('只有衛星：最大回撤', -33.4, 'maxdd'), ('只有衛星：20 條中位數', 37.41, 'paths_median'), ('只有衛星：路徑最少', 32.57, 'paths_min'),
                   ('只有衛星：路徑最多', 44.15, 'paths_max'), ('只有衛星：20 條回撤中位數', -35.3, 'paths_dd_median')]:
    c(RD, L12 + ' XQ 表', what, v, None, .05, P_RS + f"（結果檔值 {RS['XAS'][k]}）", '本次沒有重跑 XAS；只核對到結果檔，數值相符' if abs(RS['XAS'][k] - v) <= .05 else '與結果檔不符')
c(RD, '第 12 節第一張表（0050 列）', '0050 2019 起', 28.91, A('2019 年起每年', '0050'), .005, src_a); c(RD, '第 12 節第一張表', '0050 全期', 21.14, A('全期每年', '0050'), .005, src_a)
c(RD, '第 12 節第一張表', '0050 2024 起', 54.48, A('2024 年起每年', '0050'), .005, src_a); c(RD, '第 12 節第一張表', '0050 最大回撤', -34.0, A('最大回撤', '0050'), .05, src_a)
# 第 30 節
L30 = '第 30 節'
c(RD, L30 + ' 現值', 'A（3 年超額）', 15.0, IND27['A'], .05, '獨立：navs_official 月底值'); c(RD, L30 + ' 現值', 'E（相對前高）', -7.1, IND27['E'], .05, '獨立：navs_official 月底值')
c(RD, L30 + ' 現值', 'B（1 年超額）', 33.5, IND27['B'], .05, '獨立：navs_official 月底值'); c(RD, L30 + ' 現值', 'C（月底回撤）', -5.4, IND27['C'], .05, '獨立：navs_official 月底值')
c(RD, L30 + ' 現值', 'C 日內', -10.6, IND27['C_daily'], .05, '獨立：navs_official 日值'); c(RD, L30 + ' 現值', 'D（月數）', 9, IND27['D'], 0, '獨立：navs_official 月底值')
th = S27['thresholds']
for k, v in [('A', -8), ('E', -24), ('B', -26), ('C', -29.9)]:
    c(RD, L30 + ' 門檻', f'{k} 門檻', v, pp(th[k]), .05, src_r)
c(RD, L30 + ' 門檻', 'D 門檻（月）', 50, th['D'], 0, src_r)
AL = {'A': ([2, 5, 8], [42, 65, 80]), 'E': ([2, 6, 9], [49, 73, 85]), 'B': ([2, 5, 8], [14, 34, 48]), 'C': ([2, 6, 9], [12, 26, 37]), 'D': ([0, 5, 10], [0, 51, 71]),
      'ANY': ([6, 18, 27], [58, 85, 94]), 'TWO_OR_MORE_posthoc': ([1, 4, 7], [40, 67, 81])}
NM = {'ANY': '黃燈', 'TWO_OR_MORE_posthoc': '紅燈'}
for k, (ef, ze) in AL.items():
    for sc, arr, lab in (('EFF', ef, '策略有效時誤報'), ('ZERO', ze, '超額歸零時觸發')):
        for h, v in zip(('1y', '3y', '5y'), arr):
            c(RD, L30 + ' 警戒線表', f'{NM.get(k, k)} {lab} {h}', v, pp(S27['alerts'][k][sc][h]), .5, src_r)
c(RD, L30 + ' 警戒線表', 'A 歷史最差', -3.5, pp(S27['hist']['A']['worst_value_worst']), .05, src_r); c(RD, L30 + ' 警戒線表', 'E 歷史最差', -19.3, pp(S27['hist']['E']['worst_value_worst']), .05, src_r)
SCN = {('EFF', '12m'): (0.5, 30, 99), ('HALF', '12m'): (-5, 23, 89), ('ZERO', '12m'): (-10, 17, 80), ('EFF', '36m'): (12, 35, 68), ('HALF', '36m'): (6, 28, 60), ('ZERO', '36m'): (1, 21, 51),
       ('EFF', '60m'): (17, 36, 60), ('HALF', '60m'): (11, 29, 52), ('ZERO', '60m'): (5, 22, 44)}
for (sc, h), (a_, m_, z_) in SCN.items():
    x = S27['scen'][sc][h]
    for lab, v, k in (('差的 10%', a_, 'p10'), ('中位數', m_, 'p50'), ('好的 10%', z_, 'p90')):
        c(RD, L30 + ' 前瞻模擬表', f'{sc} {h} {lab}', v, pp(x[k]), .51, src_r, '文件寫「約 0～1%」' if (sc, h, k) == ('EFF', '12m', 'p10') else '')
for h, (a_, m_, z_) in {'12m': (-4, 22, 66), '36m': (6, 23, 46), '60m': (10, 24, 41)}.items():
    x = S27['scen']['EFF'][h]
    for lab, v, k in (('差的 10%', a_, 'b10'), ('中位數', m_, 'b50'), ('好的 10%', z_, 'b90')):
        c(RD, L30 + ' 前瞻模擬表', f'0050 {h} {lab}', v, pp(x[k]), .51, src_r)
for h, arr in {'12m': (24, 41, 58), '36m': (9, 30, 58), '60m': (4, 23, 59)}.items():
    for sc, v in zip(('EFF', 'HALF', 'ZERO'), arr):
        c(RD, L30 + ' 輸 0050 機率', f'{sc} {h}', v, pp(S27['scen'][sc][h]['lose']), .5, src_r)
c(RD, L30, '5 年內某 12 個月輸 0050', 97, pp(S27['events']['any_12m_behind']), .5, src_r); c(RD, L30, '連續一年以上沒追回相對高點', 88, pp(S27['events']['lag_ge12']), .5, src_r)
# 第 31 節
L31 = '第 31 節'
c(RD, L31, '回測 48.76%', 48.76, A('2019 年起每年', '現行'), .005, src_a)
for what, v, k in [('樂觀', 45.4, 'opt'), ('樂觀 回撤', -34.8, 'opt_dd'), ('中性', 41.8, 'neutral'), ('中性 回撤', -34.8, 'neutral_dd'), ('中性 壞時回撤', -37.5, 'neutral_dd_p10'),
                   ('超額打一半', 36.1, 'half'), ('打一半 回撤', -36.7, 'half_dd'), ('保守', 34.0, 'cons'), ('保守 回撤', -39.9, 'cons_dd')]:
    c(RD, L31 + ' S42 表', what, v, S42[k], .051, src_r, 'S42 沒有獨立重現')
c(RD, L31 + ' S42 表', '0050 同期 2019 起', 28.9, A('2019 年起每年', '0050'), .05, src_a)
c(RD, L31, '滑價每邊多 0.1% 約 −0.85', -0.85, S42['paired']['slip_0.004_vs_0.003_since2019_pp'], .005, src_r)
c(RD, L31, '晚下單約 −3', -3, S42['paired']['COMB_LATE_minus_ON_median_pp'], .5, src_r, f"COMB 晚−準時 {S42['paired']['COMB_LATE_minus_ON_median_pp']:.2f}；CUR {S42['paired']['CUR_LATE_minus_ON_median_pp']:.2f}")
c(RD, L31, '正式基準 20 條中位數 46.4', 46.4, S42['stack'][0][1], .05, src_r)
c(RD, L31, '首次公布營收約 −1.0', -1.0, S42['stack'][1][2], .05, src_r)
c(RD, L31, 'S39 約 −0.8（視窗 2024-07 起）', -0.8, S42['fpm_window_pp'], .05, src_r)
c(RD, L31, 'S40 約 −0.24', -0.24, ovr['期限前視高估'], .005, src_a)
old42 = J(AG / 's42_realistic_expectation' / 'results.json')
o_std = [pp(x['since2019_standard']) for x in old42['stack']]
c(RD, L31, '「2019 起合併 −0.60」', -0.60, None, .05, 's42_realistic_expectation/results.json',
  f"出處是 S42 FINDINGS 的「−0.25＋−0.35＝−0.60」（逐條配對差，舊口徑），結果檔沒有這個欄位；舊口徑標準路徑 CUR {o_std[0]:.2f} → COMB {o_std[2]:.2f}＝{o_std[2]-o_std[0]:.2f}；20 條中位數 {old42['stack'][2]['d_vs_CUR_since2019_median_pp']:.2f}")
c(RD, L31 + ' S46', '現行比家族中位數高', 6.25, F46['cur_minus_median'], .005, '讀 S46 路徑值自算；R46 重現'); c(RD, L31 + ' S46', '900 條中位數', 40.4, F46['all_median'], .05, '同上')
c(RD, L31 + ' S46', '差的情況', 35.9, F46['all_p10'], .05, '同上'); c(RD, L31 + ' S46', '45 格排第 1', 1, F46['rank'], 0, '同上')
# 第 32 節
L32 = '第 32 節'
c(RD, L32, '只買一半少', 6.0, -SC['PICK_RAND']['d19_pp'], .05, 'S64 results.json；R64 重現 5.8'); c(RD, L32, '只買股價最低的一半少', 6.4, -SC['PICK_LOWPX']['d19_pp'], .05, 'S64 results.json（R64 沒重現這格）')
c(RD, L32, '只買成交值最大的一半少', 3.3, -SC['PICK_BIGAMT']['d19_pp'], .05, 'S64 results.json（R64 沒重現這格）'); c(RD, L32, '回落 20% 停手少', 4.8, -SC['STOP_20_6M']['d19_pp'], .05, 'S64；R64 重現')
c(RD, L32, '開盤漲 3% 不買少', 1.6, -SC['GAP_3']['d19_pp'], .05, 'S64；R64 重現'); c(RD, L32, '5% 才不買只少', 0.2, -SC['GAP_5']['d19_pp'], .05, 'S64；R64 重現')
c(RD, L32, '漏做四分之一少', 2.1, -SC['SKIP_25']['d19_pp'], .05, 'S64；R64 重現 2.2'); c(RD, L32, '漏一成少', 0.7, -SC['SKIP_10']['d19_pp'], .05, 'S64'); c(RD, L32, '漏一半少', 6.1, -SC['SKIP_50']['d19_pp'], .05, 'S64')
c(RD, L32, '完全照做 20 條中位數', 46.4, pp(s64['baseline']['since2019']), .05, 'S64；本次重跑同值'); c(RD, L32, '完全照做回撤', -33.4, pp(s64['baseline']['maxdd_full']), .05, 'S64')
c(RD, L32, '漏 25%＋開盤漲 3%＋只買一半約', 37.1, pp(SC['L3']['since2019']), .05, 'S64（R64 沒重現）'); c(RD, L32, '漏 50%＋只買一半＋回落停手約', 33.6, pp(SC['L4']['since2019']), .05, 'S64（R64 沒重現）')
c(RD, L32, '10 萬零股 2019 起', 43.6, R64['S62 10 萬 B 基準：2019 起（%）'], .05, 'R64 compare.csv'); c(RD, L32, '10 萬零股對 0050 超額', 14.7, R64['S62 10 萬 B 基準：對 0050 超額（百分點）'], .05, 'R64 compare.csv')
c(RD, L32, '10 萬只買整張', 29.4, R64['S62 10 萬只買整張：2019 起（%）'], .05, 'R64 compare.csv'); c(RD, L32, '只買整張超額', 0.5, R64['S62 10 萬只買整張：對 0050 超額（百分點）'], .05, 'R64 compare.csv')
c(RD, L32, '一半 0050：2019 起', 36.7, R65['S63 G 衛星 35%：2019 起（%）'], .05, 'R65 compare.csv'); c(RD, L32, '一半 0050：回撤', -30.4, R65['S63 G 衛星 35%：最大回撤（%）'], .05, 'R65 compare.csv')
b65 = R65['S65 A：36 個月年化超額中位數（百分點）']
c(RD, L32, '分 3 個月進場少約', 2.1, b65 - R65['S65 C 分 3 個月：36 個月年化超額中位數（百分點）'], .06, 'R65 compare.csv'); c(RD, L32, '分 6 個月少約', 3.1, b65 - R65['S65 D 分 6 個月：36 個月年化超額中位數（百分點）'], .06, 'R65 compare.csv')
c(RD, L32, '等 0050 回落 10% 少約', 3.6, b65 - R65['S65 E 等回落：36 個月年化超額中位數（百分點）'], .05, 'R65 compare.csv')
c(RD, L32, '只買 0050：2019 起', 28.9, A('2019 年起每年', '0050'), .05, src_a); c(RD, L32, '只買 0050：回撤', -34.0, A('最大回撤', '0050'), .05, src_a)
c(RD, L32, '2008～2013 每年約多（S55）', 1.3, None, .1, 'r55_replicate_s55/compare.csv', 'S55 宣稱 1.27、R55 重現 1.34；沒有在本次重算')
c(RD, L32, '低消 20 元每年多付 2～4／1～2 個百分點、歷史制度 10.2、稅 0～0.9', '多個', None, 0, 'S62／S64 results.json', '本次沒有逐格核對；S62 這幾格 R64 沒有重現')
c(RD, '第 33 節', '+5 時約 27% 被誤停、沒用時約八成被停', '27%／八成', None, 0, 's69_forward_validation_plan/FINDINGS.md；R69 compare.csv 全符', '本次沒有重跑模擬；只確認 R69 對照表沒有不符列')
# 第 26 節（全部是修正前口徑）
L26 = '第 26 節表格'
c(RD, L26, 'XQ 門檻版 只有衛星 2019 起', 40.5, RS['XAS']['since2019'], .05, P_RS, '舊口徑（修正前 40.46）；正式是 42.91%／−33.4%（37.41%）。表上沒標「修正前」，只在節尾有一句指到第 12 節', force='過期（舊口徑，未標註）')
c(RD, L26, 'XQ 門檻版 衛星 70% 2019 起', 46.5, A('A10：2019 起'), .05, src_a, '舊口徑（修正前、舊換股日 46.47）；正式（A10）是 45.09%／−38.8%（43.90%）', force='過期（舊口徑，未標註）')
c(RD, L26, '研究中心名次版 衛星 70%', 48.1, A('2019 年起每年', '現行'), .05, src_a, '舊口徑（修正前 48.13／−33.4／46.2）；正式是 48.76%／−32.1%（46.40%）', force='過期（舊口徑，未標註）')
c(RD, L26, '覆核補充：只算衛星約 40%（33%～45%）', 40, RS['XAS']['since2019'], 2.0, P_RS, '舊口徑；正式 42.91（路徑 32.57～44.15、中位數 37.41）。「差 4 個百分點內正常」的說法還成立', force='過期（舊口徑，未標註）')
# 從這裡開始
c(ST, '回測數字與必讀的限制', '2019 年起年化約 48.8%', 48.8, A('2019 年起每年', '現行'), .05, src_a); c(ST, '同上', '最大回撤約 −32%', -32, A('最大回撤', '現行（E0｜W10_70，100 萬，標準路徑）'), .5, src_a)
c(ST, '同上', '0050 約 28.9%', 28.9, A('2019 年起每年', '0050'), .05, src_a); c(ST, '同上', '0050 回撤 −34%', -34, A('最大回撤', '0050'), .5, src_a)
c(ST, '同上', '參考預期約 40%（S46）', 40, F46['all_median'], .5, '讀 S46 路徑值自算；R46 重現')
c(ST, '同上', '約 3,000 種組合（S52）', 3000, 3006, 10, 's52 FINDINGS（N＝3,006）；R52 對照表全符', '沒有重算，只讀出處')
c(ST, '同上', '2008～2013 約 1.3（S55）', 1.3, 1.3422, .1, 'r55 compare.csv（R55 重現 1.34；S55 宣稱 1.27）', '沒有重算，只讀出處')
c(ST, '同上', '2013～2015 約 3.7（S71）', 3.7, 3.68, .05, 's71 FINDINGS（+3.68）', '⚠ S71 沒有獨立重現（BACKLOG 寫「未重現」），依稽核部章程對外文件應標「未重現」')
c(ST, '同上', '上櫃 −1.6（S68）', -1.6, -1.61, .05, 's68 FINDINGS（2019 起 −1.61）', '⚠ S68 沒有獨立重現（沒有 r68 資料夾），應標「未重現」')
c(ST, '同上', '隨機排名沒有一次追上（S72）', '0／300', '0／300', 0, 'BACKLOG；r72 compare.csv 全符', '沒有重算，只讀出處')
# ---------------------------------------------------------------- 表 c2：矛盾／過期數字
STALE = [
 dict(優先=1, 檔案='xq_scripts/site/build_site_data.py（RISK）→ data/risk.json「照做_開盤漲3%不買少」', 現在寫的='每年少賺約 1.6 個百分點（46.4% → 45.6%）', 問題='同一句裡兩個口徑：46.4 − 45.6 只有 0.8，1.6 是「20 條逐條相減後取中位數」。讀者照括號算會以為寫錯',
      正式數字=f"逐條差中位數 {-SC['GAP_3']['d19_pp']:.2f}；中位數相減 {46.40 - pp(SC['GAP_3']['since2019']):.2f}", 建議='改成「20 條路徑逐條比較，中位數少 1.6 個百分點（20 條中位數 46.4% → 45.6%，中位數相減是 0.8；兩種算法不同）」；README 第 32 節同一格比照'),
 dict(優先=1, 檔案='同上「照做_漏做四分之一少」「照做_只買一半少」「照做_回落20%停手少」', 現在寫的='少 2.1（46.4% → 43.8%）；少 6.0（46.4% → 40.1%）；少 4.8（46.4% → 41.5%）', 問題='同上：括號裡兩個中位數相減是 2.6／6.3／4.9，和前面的數字對不起來',
      正式數字=f"逐條差中位數 2.12／6.00／4.76；中位數相減 {46.40 - pp(SC['SKIP_25']['since2019']):.2f}／{46.40 - pp(SC['PICK_RAND']['since2019']):.2f}／{46.40 - pp(SC['STOP_20_6M']['since2019']):.2f}", 建議='說明加一句「少幾個百分點＝逐條相減後取中位數，不等於括號兩數相減」'),
 dict(優先=1, 檔案='xq_scripts/底倉衛星_操作說明.md 第 1 節表、第 70、110 行', 現在寫的='2019 起 48.13%、全期 34.08%、2024 起 70.16%、最大回撤 −33.4%、最多落後 17.6 個百分點；準時 48.13%', 問題='修正前口徑，全檔沒有任何「修正前」或指向正式數字的註記；這是給使用者照做的操作說明',
      正式數字='48.76%／34.40%／70.46%／−32.1%／15.3 個百分點', 建議='表頭加「（2026-11-15 修正前；正式數字 48.76%／34.40%／70.46%／−32.1%，見 README 第 12 節最後）」，或直接換成正式數字'),
 dict(優先=1, 檔案='xq_scripts/策略利潤圖.html', 現在寫的='現行 2,758 萬、2019 起 48.1%、全期 34.1%、回撤 −33.4%、20 條中位數 46.2%；XQ 版 2,375 萬、47.0%／46.5%、−38.6%、44.0%；逐年 2018 1.8、2019 33.6、2020 99.4、2021 46.4、2022 −18.9、2023 52.7、2024 60.2、2025 56.5、2026 78.1',
      問題='用的是修正前淨值（s19_xq_threshold/navs.csv.gz），XQ 版還是舊換股日 B11；頁面沒有標示是舊口徑。和網站、README 正式數字不一致',
      正式數字=f"現行 {E['final']/1e4:,.0f} 萬、48.76%／34.40%／−32.1%／46.40%；XQ 版（A10）{core['independent_metrics']['XT|W10_70']['final']/1e4:,.0f} 萬、45.09%／−38.8%／43.90%；逐年 2018 0.95、2019 31.17、2020 100.75、2021 44.94、2022 −17.45、2023 57.72、2024 63.38、2025 56.69、2026 75.24",
      建議='用 navs_official.csv.gz 重產，或在標題下加「本圖是 2026-11-15 修正前的淨值，正式數字見網站」'),
 dict(優先=2, 檔案='README.md「回測數字與必讀的限制」', 現在寫的='2013～2015 約 3.7（S71）、上櫃股票 −1.6（S68）', 問題='數字與出處相符，但 S71、S68 都沒有獨立重現，文件沒有標「未重現」（稽核部章程職責 1）', 正式數字='S71 +3.68（未重現）；S68 −1.61（未重現）',
      建議='在兩處後面加「（未重現）」，或安排 R71、R68'),
 dict(優先=2, 檔案='xq_scripts/README.md 第 26 節表格', 現在寫的='XQ 只有衛星 40.5%／−32.4%（39.98%）；衛星 70% 46.5%／−38.6%（44.0%）；名次版 48.1%／−33.4%（46.2%）、41.6%／−25.1%（39.5%）', 問題='整張表是修正前、舊換股日口徑，表上沒有標；節尾只有一句「修正後見第 12 節最後」。這一節的標題是「你在 XQ 回測應該看到什麼」，使用者會直接拿來對帳',
      正式數字='XQ 只有衛星 42.91%／−33.4%（37.41%；路徑 32.57%～44.15%）；XQ 衛星 70%（A10）45.09%／−38.8%（43.90%）；名次版 48.76%／−32.1%（46.40%）；名次版只有衛星的正式口徑沒有算過', 建議='表頭加「修正前」，下面補一列正式數字；「約 40%（33%～45%）」改「約 43%（20 條 33%～44%，中位數 37%）」'),
 dict(優先=2, 檔案='xq_scripts/README.md 第 12 節第一張表與「主要版本（修正前）」「獨立重算」兩張表', 現在寫的='48.13%／34.08%／70.16%／−33.4%／10 年；46.23%；379 筆 51.2%×27.16%−48.8%×9.24%；17.6', 問題='節首有一行註記說是修正前（有標），但「獨立重算：379 筆逐筆相同（48.14%）」這句容易被讀成正式數字已獨立重現。正式口徑（381 筆、48.76%）只有 R46／R64 等用同一套共用引擎重現',
      正式數字='48.76%／34.40%／70.46%／−32.1%／9 年；46.40%；381 筆', 建議='在「獨立重算」那段加「（這是修正前口徑；正式口徑 48.76%／381 筆由 R46、R55、R64、R65、R72 與 2026-11-20 稽核重跑確認，都共用同一套引擎）」'),
 dict(優先=2, 檔案='xq_scripts/README.md 第 13～24 節（388、483、501、508、529、543、560、589、606、622、639 行等）', 現在寫的='現行 46.2%／−34.0%（第 17～24 節的對照基準）；48.13→48.99%；46.5%／−33.5%；準時 46.7%；約 43%（43.17%）／40%', 問題='第 17～24 節的「現行 46.2%／−34.0%」是修正前，表上沒有標；第 13～16 節的「→ 48.99%、46.7%、43.17%」是只修除權息，不是正式（第 12 節有說明這幾節沒改，但各表本身沒標）',
      正式數字='現行 20 條中位數 46.40%／−33.4%；單一路徑 48.76%；準時 46.6%；有時晚 42.9%／40.2%', 建議='各表的「現行」列加註「修正前；正式 46.4%／−33.4%」；第 13～16 節的 48.99% 後面加「（只修除權息；正式 48.76%）」'),
 dict(優先=2, 檔案='xq_scripts/每檔金額怎麼算_補充說明.md 第 77 行', 現在寫的='W10 衛星 70%（S5 主格）48.13%／34.08%／−33.4%', 問題='修正前口徑，全檔沒有註記', 正式數字='48.76%／34.40%／−32.1%', 建議='加「修正前」註記或換數字'),
 dict(優先=3, 檔案='xq_scripts/README.md 第 31 節', 現在寫的='資料問題合計只約 −0.6 個百分點（S39 約 −0.8、S40 約 −0.24，2019 起合併 −0.60）', 問題='一句話混了三種口徑：−0.8 是 2024-07 起視窗、−0.24 是標準路徑、−0.60 是 S42 FINDINGS 的「逐條配對差 −0.25＋−0.35」（舊口徑）；三個數字加不起來，同節後段又寫正式口徑「首次公布營收約 −1.0」（20 條中位數）',
      正式數字=f"正式口徑：20 條中位數 {S42['stack'][1][2]:.2f}（只剩首次公布營收，期限已在基準裡）；標準路徑 {S42['stack'][1][3]-S42['stack'][0][3]:.2f}", 建議='把這一行改成「正式基準下只剩首次公布營收：20 條中位數約 −1.0 個百分點（標準路徑約 −0.2）」，刪掉 −0.60'),
 dict(優先=3, 檔案='xq_scripts/README.md 第 30 節警戒線表 E 列', 現在寫的='E 超額歸零時觸發 1 年 49%', 問題=f"49% 是只修除權息的舊值（48.59%）；正式口徑重算是 {pp(S27['alerts']['E']['ZERO']['1y']):.1f}%（四捨五入 48%）；2026-11-17 的更新清單漏列這一格", 正式數字='48%', 建議='49% → 48%'),
 dict(優先=3, 檔案='xq_scripts/README.md 第 12 節正式表「交易統計」', 現在寫的='50.4% × 27.43% − 49.6% × 9.11% ＝ +9.30%', 問題='精確值 9.305；用表上四捨五入後的數字算是 9.31。差 0.01，不影響判讀', 正式數字='9.30～9.31', 建議='可不改；要改就寫「約 +9.3%」'),
 dict(優先=3, 檔案='xq_scripts/README.md 第 12 節正式表「差」欄', 現在寫的='全期 −0.14、2024 起 −0.29 個百分點', 問題='是兩個四捨五入後的數字相減；精確差是 −0.147、−0.297（四捨五入 −0.15、−0.30）', 正式數字='−0.15、−0.30', 建議='可不改'),
 dict(優先=3, 檔案='xq_scripts/site/data/nav.json 的 meta「日期」', 現在寫的='每週取最後一個交易日', 問題='有 8 週遇到補行交易的週六（2016-01-30 等），網站取的是週五；數值正確，說明不精確', 正式數字='—', 建議='說明改「每週取週一到週五的最後一個交易日」'),
 dict(優先=3, 檔案='xq_scripts/site/data/risk.json「營收更正高估」', 現在寫的='−0.8（2024-07～2026-10 年化約低 0.8）', 問題='數值相符，但出處 S39 是只修除權息口徑（−0.78）；正式口徑是 −0.77，四捨五入相同', 正式數字='−0.77', 建議='不用改'),
]
for x in STALE:
    pass
# 檔案裡仍出現舊數字的行（機械掃描，供人工複查）
SCAN = []
pat = re.compile(r'48\.99|48\.13|48\.14|48\.1%|46\.2%|46\.23|34\.08|34\.54|70\.16|70\.75')
for f in [ROOT / 'README.md', ROOT / '主策略說明.md', ROOT / '交易策略規則書.md', ROOT / '選股SOP.md'] + sorted((ROOT / 'xq_scripts').glob('*.md')) + sorted((ROOT / 'xq_scripts').glob('*.html')) + \
        [SITE / 'index.html', SITE / 'state' / 'README.md']:
    if not f.exists():
        continue
    for i, line in enumerate(f.read_text(encoding='utf-8').split('\n'), 1):
        if len(line) > 20000:
            hits = sorted(set(pat.findall(re.sub(r'\[[\d\.,\-\s"]{300,}\]', '', line))))
            if hits:
                SCAN.append(dict(檔案=str(f.relative_to(ROOT)), 行=i, 舊數字='、'.join(hits), 該行有標註='否（內嵌資料行）', 摘要='（html 內嵌資料與 PUB 常數）'))
            continue
        hits = sorted(set(pat.findall(line)))
        if hits:
            marked = any(k in line for k in ('修正前', '只修除權息', '期限修正前', '原寫法'))
            SCAN.append(dict(檔案=str(f.relative_to(ROOT)), 行=i, 舊數字='、'.join(hits), 該行有標註='是' if marked else '否', 摘要=line.strip()[:120]))


def dump_csv(p, rows):
    with open(p, 'w', newline='', encoding='utf-8-sig') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


dump_csv(HERE / 'table_b_risk_rows.csv', TB); dump_csv(HERE / 'table_c_doc_numbers.csv', TC); dump_csv(HERE / 'table_c_stale_conflicts.csv', STALE); dump_csv(HERE / 'table_c_old_number_scan.csv', SCAN)
cnt = lambda rows, k: {v: sum(1 for r in rows if r[k] == v) for v in sorted({r[k] for r in rows})}
res = dict(date='2026-11-20', scope='正式回測總重跑＋已發布數字總對帳',
           part1=dict(rerun_vs_res_official=core['rerun_vs_res_official'], nav_daily_compare=core['nav_daily_compare'], table_a_rows=core['table_a_n'], table_a_bad=core['table_a_bad'],
                      independent_metrics_E0={k: v for k, v in E.items()}, conv_252=core['conv_252'], trades=core['trades']),
           part2=dict(n_rows=len(TB), risk_const_equals_risk_json=const_vs_json, vs_source=cnt(TB, '與結果檔一致'), vs_recompute=cnt(TB, '與本次重算一致'),
                      status=cnt([dict(s=r['重現狀態'].split('（')[0]) for r in TB], 's'), s27_now_independent=IND27, s46_family_recomputed=F46, resync_rerun=X),
           part3=dict(n_numbers=len(TC), result=cnt(TC, '核對結果'), bad=[r for r in TC if r['核對結果'] == '不一致'], stale=STALE, old_number_lines=len(SCAN),
                      old_number_lines_unmarked=sum(1 for r in SCAN if r['該行有標註'].startswith('否'))),
           part4=dict(n=core['site_n'], bad=core['site_bad'], live=core['live']))
json.dump(res, open(HERE / 'results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print('RISK 常數＝risk.json：', const_vs_json)
print('表 b：', len(TB), cnt(TB, '與結果檔一致'), cnt(TB, '與本次重算一致'), res['part2']['status'])
for r in TB:
    if r['與結果檔一致'] != '一致' or r['與本次重算一致'] != '一致' or r['出處欄列的檔案都存在'] != '是':
        print('  ✗', r['項目'], r['網站值'], r['結果檔數值'], r['本次重算值'], r['出處欄列的檔案都存在'])
print('表 c：', len(TC), cnt(TC, '核對結果'))
for r in TC:
    if r['核對結果'] == '不一致':
        print('  ✗', r['位置'], r['數字'], r['文件寫的'], r['核對值'])
print('S27 現值獨立', IND27, 'S46', F46)
print('舊數字掃描', len(SCAN), '行；沒標註', res['part3']['old_number_lines_unmarked'])

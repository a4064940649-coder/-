#!/usr/bin/env python3
"""K 保管員：把既有正式結果檔裡的目標數字整理成 sealed/targets.json。不跑任何引擎、不產生新績效數字；
每個數字都附來源（檔案＋JSON 路徑）。另做三項只讀的一致性核對（中位數／p10 與逐 j 值相符、integrate_o19 正式表相符、封存淨值檔與目標相符）。
python3 -I -B k_build_targets.py"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent; SEAL = HERE / 'sealed'
F20 = AG / 'integrate_o20/results.json'; F19 = AG / 'integrate_o19/results.json'
R20 = json.loads(F20.read_text(encoding='utf-8')); R19 = json.loads(F19.read_text(encoding='utf-8'))
V = R20['RESULTS_V7X']; TAB, PERJ = V['table'], V['per_j']; O19 = R19['RESULTS_V7X']['table']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
MASK = {'M1': 'base', 'M2': 'M2'}                      # M1＝現行正式（results.json 的 'base'＝DM_BASE）；M2＝全期統一
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']
CELLS = {'100萬一般': 'normal|100萬|std', '100萬壓力': 'stress|100萬|std', '300萬一般': 'normal|300萬|std', '1000萬一般': 'normal|1000萬|std',
         '30萬標準': 'normal|30萬|std', '30萬整股': 'normal|30萬|wholelot', '30萬壓力': 'stress|30萬|std'}
# 輸出欄名 → (table 欄位字首, per_j 欄位或 None)
FLD = {'exc': ('exc', 'exc'), 't': ('t_0050', 't_0050'), 'CAGR_Y19on': ('CAGR_Y19on', 'CAGR_Y19on'), 'exc_Y19on': ('exc_Y19on', 'exc_Y19on'),
       'maxDD': ('maxDD', 'maxDD'), 'CAGR': ('CAGR', 'CAGR'), 't_Y19on': ('t_Y19on', None), 'E50_CAGR': ('E50_CAGR', None), 'E50_CAGR_Y19on': ('E50_CAGR_Y19on', None)}
NS = '未保存'
S20 = 'integrate_o20/results.json :: '               # 每個 src＝來源檔 :: JSON 路徑
chk = dict(med_eq_median_of_per_j_maxabs=0.0, p10_eq_percentile_of_per_j_maxabs=0.0, o20_base_vs_o19_strict_maxabs=0.0, o19_keys_compared=0)
v7x = {}
for mk, dv in MASK.items():
    v7x[mk] = {}
    for v in VARS:
        v7x[mk][v] = {}
        for cn, ck in CELLS.items():
            key = f'{dv}|{v}|{ck}'; T = TAB[key]; PJ = PERJ[key]
            o = dict(src_key=key, j_median={}, j_p10={}, per_j={}, median_j={}, periods={})
            for f, (tf, pf) in FLD.items():
                o['j_median'][f] = dict(value=T[f'{tf}_med'], src=f"{S20}RESULTS_V7X.table['{key}'].{tf}_med")
                o['j_p10'][f] = dict(value=T[f'{tf}_p10'], src=f"{S20}RESULTS_V7X.table['{key}'].{tf}_p10")
                if pf is None:
                    o['per_j'][f] = NS
                else:
                    a = np.array(PJ[pf], float); assert len(a) == 13
                    o['per_j'][f] = dict(values=PJ[pf], src=f"{S20}RESULTS_V7X.per_j['{key}'].{pf}")
                    chk['med_eq_median_of_per_j_maxabs'] = max(chk['med_eq_median_of_per_j_maxabs'], abs(float(np.median(a)) - T[f'{tf}_med']))
                    chk['p10_eq_percentile_of_per_j_maxabs'] = max(chk['p10_eq_percentile_of_per_j_maxabs'], abs(float(np.percentile(a, 10)) - T[f'{tf}_p10']))
                    o['median_j'][f] = int(np.argsort(a, kind='stable')[6])   # 由逐 j 值導出：排序第 7 個的 j
            for seg in ('S0612', 'S1318', 'S19on'):
                P = T['periods'][seg]
                o['periods'][seg] = {q: dict(med=P[q]['med'], p10=P[q]['p10'], src=f"{S20}RESULTS_V7X.table['{key}'].periods.{seg}.{q}") for q in ('exc', 't', 'cagr', 'cagr50', 'mdd')}
                o['periods'][seg]['per_j'] = NS
            if dv == 'base':
                b = O19.get(f'strict|{v}|{ck}')
                if b is not None:
                    chk['o19_keys_compared'] += 1
                    for f in ('exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 'maxDD_med', 'CAGR_Y19on_med', 'CAGR_med'):
                        chk['o20_base_vs_o19_strict_maxabs'] = max(chk['o20_base_vs_o19_strict_maxabs'], abs(T[f] - b[f]))
                    o['o19_formal_table_key'] = f"integrate_o19/results.json RESULTS_V7X.table['strict|{v}|{ck}']"
            v7x[mk][v][cn] = o
# 逐格取三變體最差（S-04；由上面的中位數／p10 取最小值導出，非另存欄位）
worst = {}
for mk in MASK:
    worst[mk] = {}
    for cn in CELLS:
        worst[mk][cn] = {}
        for agg in ('j_median', 'j_p10'):
            for f in ('exc', 't', 'CAGR_Y19on', 'exc_Y19on'):
                vals = {v: v7x[mk][v][cn][agg][f]['value'] for v in VARS}; wv = min(vals, key=vals.get)
                worst[mk][cn][f'{f}|{agg}'] = dict(value=vals[wv], variant=wv)
# 與 integrate_o20 DECISION.D1 的逐格最差核對（該處四捨五入到小數 4 位）
D1 = R20['DECISION']['D1']['cells']; dchk = {}
MAPD = {'V7X 100萬一般 全期超額': ('100萬一般', 'exc|j_median'), 'V7X 100萬一般 t': ('100萬一般', 't|j_median'), 'V7X 100萬一般 j p10': ('100萬一般', 'exc|j_p10'),
        'V7X 2019 起年化': ('100萬一般', 'CAGR_Y19on|j_median'), 'V7X 2019 起超額': ('100萬一般', 'exc_Y19on|j_median'), 'V7X 100萬壓力 全期超額': ('100萬壓力', 'exc|j_median'),
        'V7X 300萬一般 全期超額': ('300萬一般', 'exc|j_median'), 'V7X 1000萬一般 全期超額': ('1000萬一般', 'exc|j_median'), 'V7X 30萬一般（標準）全期超額': ('30萬標準', 'exc|j_median'),
        'V7X 30萬整股限制 全期超額': ('30萬整股', 'exc|j_median'), 'V7X 30萬壓力 全期超額': ('30萬壓力', 'exc|j_median')}
for name, (cn, ff) in MAPD.items():
    dchk[name] = dict(M1_absdiff=abs(D1[name]['base_worst'] - worst['M1'][cn][ff]['value']), M2_absdiff=abs(D1[name]['M2_worst'] - worst['M2'][cn][ff]['value']),
                      M1_variant_eq=D1[name]['base_worst_variant'] == worst['M1'][cn][ff]['variant'], M2_variant_eq=D1[name]['M2_worst_variant'] == worst['M2'][cn][ff]['variant'])
chk['decision_D1_worst_check'] = dchk
# 紙上
PS = R20['RESULTS_PAPER']['stats']; LS = R20['RESULTS_PAPER']['lists_sha']; paper = {}
for mk in ('M1', 'M2'):
    paper[mk] = {}
    for v in ('ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'):
        paper[mk][v] = dict(lists_sha=dict(value=LS[f'{mk}|{v}'], src=f"{S20}RESULTS_PAPER.lists_sha['{mk}|{v}']"))
        for c in ('normal', 'stress'):
            key = f'{mk}|{v}|{c}'; S = PS[key]; o = dict(src_key=key)
            nm = {'ann_cal': '全期日曆年化（主口徑，%）', 'ann': '全期 ia 年化（並列，%）', 't_vs_0050': '全期對 0050 t', 'gex_cal': '全期超額（日曆，pp）', 'gex': '全期超額（ia，pp）',
                  't_drop_best3': '拿掉最好 3 年 t', 'mdd': '最大回撤（窗，%）', 'mdd_daily': '最大回撤（日，%）', 'ann_0050': '0050 ia 年化', 'ann_cal_0050': '0050 日曆年化', 'n': '窗數'}
            o['ALL'] = {f: dict(value=S['ALL'][f], desc=d, src=f"{S20}RESULTS_PAPER.stats['{key}'].ALL.{f}") for f, d in nm.items()}
            for seg in ('SEG1', 'SEG2', 'SEG3'):
                o[seg] = {f: dict(value=S[seg][f], src=f"{S20}RESULTS_PAPER.stats['{key}'].{seg}.{f}") for f in ('gex_cal', 'gex', 'ann_cal', 'ann', 't_vs_0050', 'n')}
            paper[mk][v][c] = o
pworst = {mk: {c: {f: (lambda d: dict(value=min(d.values()), variant=min(d, key=d.get)))({v: paper[mk][v][c]['ALL'][f]['value'] for v in paper[mk]}) for f in ('ann_cal', 'ann', 't_vs_0050', 'gex_cal', 'gex')}
               for c in ('normal', 'stress')} for mk in paper}
# 與 DECISION.D1 的紙上五格核對（該處四捨五入到小數 4 位；D1 的「紙上全期年化」是 ia 口徑、「2019 起超額」是 SEG3 的 ia 超額）
MAPP = {'紙上全期年化': ('normal', 'ALL', 'ann'), '紙上全期 t': ('normal', 'ALL', 't_vs_0050'), '紙上壓力全期年化': ('stress', 'ALL', 'ann'), '紙上壓力全期 t': ('stress', 'ALL', 't_vs_0050'),
        '紙上 2019 起超額': ('normal', 'SEG3', 'gex')}
for name, (c, seg, f) in MAPP.items():
    o = {}
    for mk, dk in (('M1', 'base'), ('M2', 'M2')):
        vals = {v: paper[mk][v][c][seg][f]['value'] for v in paper[mk]}; wv = min(vals, key=vals.get)
        o[f'{mk}_absdiff'] = abs(D1[name][f'{dk}_worst'] - vals[wv]); o[f'{mk}_variant_eq'] = D1[name][f'{dk}_worst_variant'] == wv
    dchk[name] = o
assert set(dchk) == set(D1), set(D1) ^ set(dchk)
chk['decision_D1_cells_checked'] = len(dchk)
chk['decision_D1_all_within_rounding'] = all(v['M1_absdiff'] < 5.1e-5 and v['M2_absdiff'] < 5.1e-5 and v['M1_variant_eq'] and v['M2_variant_eq'] for v in dchk.values())
# 封存淨值檔與目標的一致性（只讀；不跑引擎）
fchk = {}
nf = SEAL / 'v7x_navs_jmean_100w.csv.gz'
if nf.exists():
    N = pd.read_csv(nf, index_col=0, parse_dates=True)
    ED = json.loads((AG / 'o_v7_listing_exec_engine/data/jscan_end_days.json').read_text())
    def endday(j):
        x = ED[f'AB_V7_B10|J{j}'] if f'AB_V7_B10|J{j}' in ED else ED['AB_V7_B10'][f'J{j}']
        return pd.Timestamp(x)
    mx = 0.0
    for col in N.columns:
        dv, v, j, c, t, m = col.split('|'); s = N[col].dropna(); a = s.index[0]; e = endday(int(j))
        cagr = ((s.loc[e] / s.loc[a]) ** (365.25 / (e - a).days) - 1) * 100
        mx = max(mx, abs(cagr - PERJ[f'{dv}|{v}|{c}|{t}|{m}']['CAGR'][int(j)]))
    fchk['v7x_navs_100w_CAGR_vs_per_j_maxabs_pp'] = mx; fchk['v7x_navs_100w_shape'] = list(N.shape)
wf = SEAL / 'paper_windows.npz'
if wf.exists():
    Z = np.load(wf); mx1 = mx2 = 0.0
    for mk in ('M1', 'M2'):
        for v in paper[mk]:
            for c in ('normal', 'stress'):
                r = Z[f'{mk}|{v}|{c}|r']; r50 = Z[f'{mk}|{v}|{c}|r50']; x = r - r50
                mx1 = max(mx1, abs((np.prod(1 + r) ** (12 / len(r)) - 1) * 100 - PS[f'{mk}|{v}|{c}']['ALL']['ann']))
                mx2 = max(mx2, abs(x.mean() / x.std(ddof=1) * np.sqrt(len(x)) - PS[f'{mk}|{v}|{c}']['ALL']['t_vs_0050']))
    fchk['paper_windows_ia_ann_vs_stats_maxabs_pp'] = mx1; fchk['paper_windows_t_vs_stats_maxabs'] = mx2
chk['sealed_files_vs_targets'] = fchk
SP = json.loads((HERE / 'k_seal_plans.json').read_text(encoding='utf-8')) if (HERE / 'k_seal_plans.json').exists() else {}
files = {
 'v7x_plans_M1.csv.gz': 'V7X REAL 名單，遮罩 DM_BASE（現行正式；程式內名稱 base）。REAL_{opt,p90,carry} × j 0…12 × seed 0…19，每變體 63,820 列。欄 variant,j,seed,signal,exec,exec_idx,picks,inserted。由 k_gen_plans.py（原程式 v7x_lib.make，只讀 import）重產；解壓後位元組與 integrate_o20 當時的 plans_base.csv 相同（CSV sha256 ' + SP.get('M1', {}).get('csv_sha256', '（尚未封存）') + '；見 k_seal_plans.json、k_v7x_plans_check.json）。exec_idx＝成交日在 V7 面板的列號；inserted＝picks 中屬於放回集合 A 者。',
 'v7x_plans_M2.csv.gz': '同上，遮罩 M2（全期統一）。解壓後位元組與 integrate_o20 當時的 plans_M2.csv 相同（CSV sha256 ' + SP.get('M2', {}).get('csv_sha256', '（尚未封存）') + '）。',
 'paper_lists_M1.csv.gz': '紙上 ALLPIT_REAL_{opt,p90,carry} 名單，遮罩 M1_paper（現行正式），每變體 245 個訊號日。欄 mask,variant,signal,exec,picks,inserted（inserted＝picks 中屬於放回集合 A 者）。由 k_paper_regen.py（原程式 o18lib.build）重產；signal／picks 與 o_g6_allpit_list_trace_stage2/out/k_paper_lists.csv.gz 的 M1 列逐列相同，名單 sha 等於 RESULTS_PAPER.lists_sha。',
 'paper_lists_M2.csv.gz': '同上，遮罩 M2。',
 'v7x_navs_jmean_100w.csv.gz': 'integrate_o20/data/ 同名檔的位元組複本。V7X 逐 j 淨值（每欄＝單一 j 的 20 seeds 平均淨值，S-01；檔名的 jmean 指 seeds 平均，不是 13 個 j 的平均）。只有 100 萬一般與 100 萬壓力兩格；欄名 {base|M2}|{REAL_opt|REAL_p90|REAL_carry}|{j}|{normal|stress}|100萬|std，共 2×3×13×2＝156 欄；index 2006-04-10～2026-10-22（5,052 列），各 j 第一個訊號日以前為空。float32、輸出 6 位有效數字。其餘五格（300 萬、1,000 萬、30 萬三格）的逐 j 淨值未保存。',
 'paper_daily_navs.csv.gz': 'integrate_o20/data/ 同名檔的位元組複本。紙上日淨值（float32），欄名 {M0|M1|M2}|{C0|ALLPIT_REAL_opt|ALLPIT_REAL_p90|ALLPIT_REAL_carry}|{normal|stress}，共 20 欄（C0 只有 M0）；2006-04-11～2026-10-12（5,018 列）。紙上沒有 j 與 seed。驗收只用 M1、M2 的 12 欄。',
 'paper_windows.npz': 'integrate_o20/data/ 同名檔的位元組複本。紙上逐窗報酬：鍵 {mask}|{variant}|{cost}|r（策略）與 |r50（0050），各 245 窗；鍵 per＝窗起點月份（2006-04…2026-09）。同樣含 M0 與 C0。',
}
not_saved = [
 dict(item='V7X 逐 j 淨值（20 seeds 平均）：300 萬一般、1,000 萬一般、30 萬標準、30 萬整股、30 萬壓力五格 × 三變體 × 13 j × M1／M2', status=NS,
      note='integrate_o20/run_v7x.py 把全部格寫在 scratchpad 的 navs_v7x_o20.parquet，已刪；研究資料夾只留 100 萬兩格。這五格的逐 j「統計值」（exc、t、exc_Y19on、CAGR_Y19on、maxDD、CAGR）有保存，見本檔 per_j。'),
 dict(item='V7X 每個 (變體, j, seed) 的日淨值：七格 × M1／M2', status=NS, note='從未存檔（run_v7x.py 只在記憶體內平均）。'),
 dict(item='V7X 逐 j 的三段超額（S0612／S1318／S19on）與逐 j 的 t_Y19on、0050 年化', status=NS, note='results.json 只存 j 中位數與 p10。100 萬兩格可由封存淨值重算；其餘五格不行。0050 逐 j 年化可由 per_j 的 CAGR − exc 得到。'),
 dict(item='原引擎的逐筆成交／現金紀錄（ACCEPTANCE 第 4.2 節 Track E 找第一個分歧用）', status=NS, note='正式 REAL 名單 × cons_main 從未存逐筆紀錄。'),
]
rerun = dict(program='integrate_o20/run_v7x.py（需另存 K 複本、改讀 sealed 名單與輸出路徑，並拿掉「只跑一次」與寫回 results.json 的部分；引擎為 cz_v7x.CSim，只讀 import）',
             jobs='1,560 組（780 組 × M1、M2）× 7 格', runtime='原執行 1,419 秒（4 行程；integrate_o20/results.json :: RESULTS_V7X.runtime_s）。本機與其他工作共用 4 核與同一個記憶體上限，用 2 行程估 45～60 分鐘（估計值，未實測）。'
                     '注意：2026-11-14 K 重產名單時（每行程約 1.6GB），同機其他工作的行程被記憶體上限強制終止 12 次；引擎每行程的記憶體用量未量測，重跑前要先排開其他工作或只用 1 行程。',
             disk='逐 j 淨值（七格 × 兩遮罩，546 欄 float32）約 8～15MB；若連每個 seed 的日淨值都存（10,920 條 × 約 5,050 日，float32 原始約 220MB），壓縮後估 130～200MB，超過 sealed/ 100MB 的額度，需總監決定只存部分格或放寬。',
             acceptance='重跑後必須先核對七格 × 三變體的 j 中位數／p10 與 results.json 表最大差 0.0，才可當目標。',
             status='本次依指示未重跑。')
out = dict(
 what='G6 第三階段驗收目標數字（K 保管員整理）。全部取自既有正式結果檔，本檔沒有任何新算的績效數字。',
 built_utc=pd.Timestamp.now('UTC').isoformat(),
 sources={'integrate_o20/results.json': dict(sha256=sha(F20), used='RESULTS_V7X.table／per_j、RESULTS_PAPER.stats／lists_sha、DECISION.D1（核對用）'),
          'integrate_o19/results.json': dict(sha256=sha(F19), used="RESULTS_V7X.table['strict|…']（正式表；只用來核對 M1 與它相同）")},
 conventions=dict(mask='M1＝現行正式（V7X：DM_BASE，results.json 鍵名 base；紙上：M1_paper）。M2＝全期統一。',
                  units='exc、CAGR、maxDD、年化＝百分點／%；t 無單位。',
                  v7x_fields=dict(exc='全期超額（S-02）', t='全期對 0050 t（S-03；欄位 t_0050）', CAGR_Y19on='2019 起年化（主欄起點 2019-01-02 收盤）', exc_Y19on='2019 起超額', maxDD='最大回撤', CAGR='全期年化',
                                  t_Y19on='2019 起 t', E50_CAGR='0050 全期年化', E50_CAGR_Y19on='0050 2019 起年化'),
                  j_median='13 個 j 的中位數（S-04）', j_p10='numpy.percentile(13 個 j, 10)', per_j='j＝0…12 依序的 13 個值',
                  median_j='由 per_j 導出：穩定排序後第 7 個值所屬的 j（非另存欄位）',
                  periods='三段表（起點＝區間前一交易日收盤，N-10）：exc／t／cagr／cagr50／mdd 的 j 中位數與 p10；逐 j 值未保存',
                  worst_of_three='逐格逐指標取三變體最小值（S-04）；由本檔 j_median／j_p10 導出，非另存欄位'),
 v7x=v7x, v7x_worst_of_three=worst, paper=paper, paper_worst_of_three=pworst,
 files=files, not_saved=not_saved, rerun_needed_for_not_saved=rerun, checks=chk)
SEAL.mkdir(exist_ok=True)
(SEAL / 'targets.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(chk, ensure_ascii=False, indent=1)); print('cells', len(MASK) * len(VARS) * len(CELLS), 'size', (SEAL / 'targets.json').stat().st_size)

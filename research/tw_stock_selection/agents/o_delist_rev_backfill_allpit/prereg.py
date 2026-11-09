#!/usr/bin/env python3
"""凍結（只執行一次）：寫 SPEC.json 與 results.json 的 PREREG（sha256、written_utc）。凍結前已完成：補回值表、覆蓋表（槽位＋報酬權重）、
預測 P1～P6、判定規則、「下界能否觸發」檢查、煙霧測試（smoke.json）。python3 -I -B prereg.py"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
RJ = HERE / 'results.json'
assert not RJ.exists(), 'results.json 已存在：PREREG 只寫一次'
SM = json.loads((HERE / 'smoke.json').read_text(encoding='utf-8')); assert all(SM['checks'].values()) and SM['C0_normal_maxabs_vs_main_v2'] == 0
CV = json.loads((HERE / 'coverage_B1.json').read_text(encoding='utf-8'))
EX = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
TOP10 = ['2325', '2311', '1704', '3454', '1507', '6286', '1729', '3474', '6452', '5305']
KY = ['1262', '1592', '6422', '6452']
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o18_revbackfill'
own = ['o18lib.py', 'v7x_lib.py', 'run_paper.py', 'v7x_plans.py', 'v7x_run.py', 'bflib.py', 's1_demand.py', 's2_extract_raw.py', 's2b_extract_taiwanan.py',
       's3_parse.py', 's4_gate.py', 's5_backfill.py', 's6_coverage.py', 's7_smoke_explore.py', 'prereg.py',
       'data/rev_backfill_B1.csv', 'data/backfill_B1_meta.json', 'data/coverage_slots_B1.csv', 'data/demand_cells.csv', 'data/demand_slots.csv', 'data/demand_aff.csv',
       'data/news_vals_P5.parquet', 'raw/news_rev_segments.csv.gz', 'raw/taiwanan_monthly_lastver.csv.gz', 'raw/step0_connectivity.txt',
       'coverage_B1.json', 'gate_P5_T1.json', 'smoke.json', 'explore.json', 'iterations.json', 'demand.json', 'env.json']
ext = [AG / 'independent_audit/ia_core.py', AG / 'main_v2/v2_core.py', AG / 'main_v2/navs_gapfix.parquet', AG / 'o_rev_avail_ds_pit_count/pclib.py',
       AG / 'o_rev_avail_ds_pit_count/avail.py', AG / 'o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv',
       AG / 'o_engine_delist_consideration/cz_engine.py', AG / 'o_engine_delist_consideration/cz_v7x.py', AG / 'o_engine_delist_consideration/events_v2.csv',
       AG / 'o_exec_engine_rewrite_v7x/x_exec.py', AG / 'o_exec_engine_rewrite_v7x/x_data.py', AG / 'o_exec_engine_rewrite_v7x/x_stats.py',
       AG / 'o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet', AG / 'o_v7_listing_exec_engine/data/plans_j7.csv',
       AG / 'o_v4_revenue_patch_official/v4lib.py', AG / 'o_stagger_tranche/st_core.py', Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')]
FS = {f: sha(HERE / f) for f in own}
FS.update({str(p): sha(p) for p in ext})
P_ = lambda fill, lo, rev, dsp, patch: [fill, lo, rev, dsp, patch]
variants = {'C0': dict(kind=P_('base', None, None, False, None)),
            'U13_opt_o17': dict(kind=P_('opt', '2013-01-01', None, False, None)),
            'ALLPIT_opt_o17': dict(kind=P_('opt', '2006-01-01', 'UB', True, None))}
for f in ('opt', 'p90', 'carry'):
    variants[f'U13_REAL_{f}'] = dict(kind=P_(f, '2013-01-01', None, False, 'B1'))
    variants[f'UFULL_REAL_{f}'] = dict(kind=P_(f, '2006-01-01', None, False, 'B1'))
    variants[f'ALLPIT_REAL_{f}'] = dict(kind=P_(f, '2006-01-01', 'UB', True, 'B1'))
variants['ALLPIT_REAL_opt_KYX'] = dict(kind=P_('opt', '2006-01-01', 'UB', True, 'B1'), excl=KY)
for c in TOP10: variants[f'ALLPIT_REAL_opt_X{c}'] = dict(kind=P_('opt', '2006-01-01', 'UB', True, 'B1'), excl=[c])
variants['ALLPIT_REAL_opt_Xtop3'] = dict(kind=P_('opt', '2006-01-01', 'UB', True, 'B1'), excl=['2325', '2311', '1704'])
eng = {'ref': {'freeze': ['normal', 'stress']}, 'o17': {'cons_main': ['normal', 'stress']},
       'main': {'cons_main': ['normal', 'stress'], 'cons_lo': ['normal', 'stress'], 'cons_hi': ['normal', 'stress'], 'freeze': ['normal']},
       'real': {'cons_main': ['normal', 'stress'], 'freeze': ['normal']}, 'sens': {'cons_main': ['normal']}}
eset = {k: ('ref' if k == 'C0' else 'o17' if k.endswith('_o17') else 'main' if k == 'ALLPIT_REAL_opt' else 'sens' if ('_X' in k) else 'real') for k in variants}
cov = dict(slots_2013_2026=CV['all_2013_2026'], slots_2013_2018=CV['s2013_2018'], slots_2019_2026=CV['s2019_2026'], top10=CV['top10'], by_year=CV['by_year'])
PR = dict(
    topic='o_delist_rev_backfill_allpit（o18 題 1）：補回 2013～2018 下市公司實際月營收，用 ALLPIT 口徑重算主策略正式數字（紙上 C0＋可執行 V7X）',
    written_utc=pd.Timestamp.now('UTC').isoformat(), scratch=SCR, backfill_version='B1', news_parser='P5', tw_parser='T1', ky_codes=KY, top10=TOP10,
    s1_note='依 S1：本題不產生新選股訊號，不立「靠歷史回測取得可投資」的題；只做資料勘誤與正式數字重算。',
    coverage_known_at_freeze=cov,
    explore_known_at_freeze=dict(U13_REAL_opt=EX['U13_REAL_opt'], U13_REAL_p90={k: v for k, v in EX['U13_REAL_p90'].items() if k != 'real_slots'},
                                 U13_REAL_carry={k: v for k, v in EX['U13_REAL_carry'].items() if k != 'real_slots'},
                                 real_backfilled_pool_rev_pct=EX['real_backfilled_pool_rev_pct'], rounding=EX['rounding'], months_to_delist=EX['U13_REAL_opt_months_to_delist']),
    smoke=dict(file='smoke.json', sha256=sha(HERE / 'smoke.json'), C0_maxabs=0.0, U13_opt=SM['U13_opt_cons_main'], ALLPIT_opt=SM['ALLPIT_opt_cons_main'],
               V7X_orig='v7x_plans.py 斷言 orig 名單逐筆＝plans_j7；v7x_run.py 斷言 orig×orig 對 navs_oos_R4 ≤ 1e−12（凍結後第一步執行，失敗即停）'),
    paper=dict(variants=variants, engines=eng, engine_set=eset,
               stats='訊號日收盤→下一訊號日收盤窗（245 窗）對 0050 含息；ALL、SEG1 2006～2012、SEG2 2013～2018、SEG3 2019 起、SEL ～2019-04、VAL 2019-05 起、HOLD 2024 起；'
                     '年化、超額、t、窗頻與日頻最大回撤、拿掉最好 3 年 t、對 C0 月差與 t、相關、逐年絕對差'),
    v7x=dict(plan_variants=['orig', 'REAL_opt', 'REAL_p90', 'REAL_carry'],
             engines={'orig': ['orig', 'cons_main'], 'REAL_opt': ['cons_main', 'cons_lo', 'cons_hi', 'orig'], 'REAL_p90': ['cons_main'], 'REAL_carry': ['cons_main']},
             grid='AB_V7_B10 規格、13 j × 20 seeds、K5T、GROW、一般與壓力 × 100 萬／300 萬／1,000 萬',
             definition='見 v7x_lib.py docstring：營收＝R_FIX 但存活者缺口格改成本題補回值（無則缺）；unavail＝MC(seed) ∪ A_UB；處置＝DMT ∪ 官方全歷史；放回＝GAP ∪ GOV。'),
    predictions=dict(
        P1='來源覆蓋：MOPS／archive 仍連不到機率約 70%；前 10 家槽位覆蓋 50～95%，全部約 40 家 35～85%；覆蓋 < 80% 機率約 60%。【凍結前已知：MOPS、archive 等 9 台主機 000；槽位覆蓋 2013～2026 21.1%（報酬權重 22.2%）、2013～2018 19.8%（20.8%）、前 10 家 23.6%（24.0%）→ P1 覆蓋區間不符（更差）】',
        P2='真實營收版放回入選槽 ≤ U_opt（251／227），違反即程式錯誤；預測 30～175 槽，不利端放寬到 210。【凍結前已知（名單層）：U13_REAL_opt 200 槽（真實補回 5＋opt 補缺 195）、p90 133、carry 8；全部 ≤ U_opt】',
        P3='紙上 cons_main 一般成本：U13_REAL 對 C0 年化差 −4.0～0pp（中心 −1.5）；2013～2018 超額 9.71 → −2.0～+9.9pp（中心約 +5）；ALLPIT_REAL 全期年化 19.0～26.0%、t 1.15～2.30、拿掉最好 3 年 t ≤ 1.3、2013～2018 超額 −4.5～+8pp；'
           'ALLPIT_REAL 全期 t ≥ 2.643 → 先查資料。【註：覆蓋只有 21%，U13_REAL_opt 有 195／200 放回槽仍是 opt，預期與 U13 opt（−3.36pp）接近；中心預測改看 opt 補缺版 −3.5～−2.0pp（凍結前補記，原區間照列）】',
        P4='損害集中在 2013～2018；2019 起 |Δ| ≤ 0.8pp。放回槽依入選時距下市分組：≤ 6 個月平均差 < −2%／月；> 12 個月接近 0 或正；若鎖價期反而勝過被擠掉者 → 否定「併購鎖價拖累」、改判排名母體改變。前 3 家（2325、2311、1704）合計貢獻損害 40～80%；必報拿掉最大貢獻股後的數字。',
        P5='V7X ALLPIT_REAL（cons_main）100 萬一般成本：全期超額 j 中位數 +1.57pp → −5.5～+1.6pp；2019 起 −6.31 → −10.5～−5.5pp；壓力、300 萬、1,000 萬都更差；j p10 全部 < 0；G1、G2、G4、G7 仍不過（高信心）；V7X 全期 t ≥ 1.0 或 2019 起超額 > 0 → 先查資料。',
        P6='取代規則能否觸發：ALLPIT opt 版已知 −7.00pp、t −1.17 → 「變動 > 0.3pp 年化或 t > 0.05 → 取代」幾乎確定觸發（不觸發機率 < 5%）。本題資訊量在「取代成多少」與區間寬度，不在會不會觸發。'),
    trigger_check=dict(
        replace_rule='幾乎必定觸發（見 P6）→ 判定文字只寫取代成多少',
        coverage80_rule='凍結前已知覆蓋 21.1%／19.8% < 80% → 必定走「opt 補缺版＋p90／carry 區間」分支',
        G4_note_rule='需要覆蓋 ≥ 80% → 結構上無法觸發「已排除」；G4 第二、三段必定維持「存活者偏誤未排除」並附區間',
        final_grade='覆蓋 < 80% → 最高只能「待觀察」；「可用（勘誤生效）」結構上無法觸發'),
    decision=dict(
        a='覆蓋 ≥ 80% 槽位（同報報酬權重）：新正式數字＝ALLPIT_REAL，未覆蓋用 opt；覆蓋 < 80%：新正式數字取 opt 補缺版（ALLPIT_REAL_opt，cons_main，一般成本；不利端），旁附 ALLPIT_REAL_p90 與 ALLPIT_REAL_carry 區間（opt／p90／carry 三者的最小～最大）。',
        b='下列任一正式數字變動 > 0.3pp 年化或 t 變動 > 0.05 → 取代 main_v2/FINDINGS 與 SOP 引用的正式數字（不只加註），舊數字移附錄並標「含 2013 年後營收存活者偏誤」：'
          'C0 紙上：全期年化、t、三段超額（新 ALLPIT_REAL_opt cons_main 對 C0）；V7X：全期 j 中位數、p10、2019 起、壓力成本、300 萬、1,000 萬（新 REAL_opt cons_main 對現行正式 orig×orig＝navs_oos_R4）。',
        c='G4 第二、三段：覆蓋 ≥ 80% 且區間兩端同號 → 「已用實際營收排除（覆蓋 X%，報酬權重 Y%）」；否則維持「未排除」並附區間。',
        d='和主策略比：ALLPIT_REAL 對 C0 月差 t、相關係數、逐年差；重寫 G1～G8 表；預期仍不可投資；V7X 全期 t ≥ 1.0 或 2019 起超額 > 0 → 先查資料再報。',
        e='最終分級：可用（勘誤生效）＝覆蓋 ≥ 80%、閘門通過、複製一致；待觀察＝覆蓋 < 80% 或複製部分不一致；不可用＝閘門失敗或來源無法驗證。',
        engine_note='freeze 只當對照（o17 教訓：freeze 會把損失誇大一倍）；cons_lo／cons_hi 報界限。'),
    replication=dict(rule='G6：不同子代理（記 agent id 與開始時間），不得讀本研究員的 *.py；從 raw/ 原始檔自寫解析器重建 rev_backfill：issuer-month 值一致率 ≥ 99%；'
                          '重算 ALLPIT_REAL 紙上：年化差 ≤ 0.05pp、t 差 ≤ 0.01；V7X j 中位數差 ≤ 0.05pp；不一致逐筆列出、以原始證據裁定、有疑義取不利方向。',
                     status='本研究員不自我複製（同一子代理不算 G6）；交工作流程的複製員執行。'),
    readonly='live_tracking/main_forward、o_forward_infra_live、o_g8p2a_protocol_addendum：開題與結束 sha256 比對（ro_snapshot_start/end.json）；不 import、不呼叫 G8P-2a 判定器。',
    file_sha256=FS)
spec = json.dumps(PR, ensure_ascii=False, indent=1, sort_keys=True)
(HERE / 'SPEC.json').write_text(spec, encoding='utf-8')
R = dict(PREREG=PR, PREREG_sha256=hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest(), SPEC_sha256=sha(HERE / 'SPEC.json'))
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['SPEC_sha256'], PR['written_utc'])

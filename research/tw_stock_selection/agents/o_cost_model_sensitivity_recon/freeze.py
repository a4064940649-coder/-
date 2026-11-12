#!/usr/bin/env python3
"""第 3 步凍結：把最終規格寫成 results.json['PREREG_FREEZE']（含程式、輸入、引擎輸出檔 sha256 與時間戳）。之後 analyze.py 才能跑。只寫一次。
python3 -I -B freeze.py"""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R and 'PREREG_FREEZE' not in R
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/csr')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = ['csr_lib.py', 'run_engine.py', 'diag.py', 'build_swaps.py', 'explore.py', 'iterate.py', 'analyze.py', 'prereg.py', 'freeze.py', 'explore.json', 'explore_grid.py', 'iterations.json',
         'build_swaps_meta.json']
files += [str(SCR / f) for f in ('navs_center.parquet', 'center_meta.pkl', 'navs_refund.parquet', 'refund_meta.pkl', 'navs_coal.parquet', 'navs_grid.parquet', 'grid_meta.pkl',
                                  'navs_tick.parquet', 'tick_meta.pkl', 'swaps_center.parquet')]
files += [f'../o_turnover_fixedcost_diagnosis/data/{f}' for f in ('plans_real_allseeds.csv.gz', 'shapley_coalitions.csv.gz', 'tradelog_100n_REAL_opt_seed0.csv.gz',
                                                                   'tradelog_30n_REAL_opt_seed0.csv.gz', 'swaps_exitlevel_100n_REAL_opt_seed0.csv.gz')]
FZ = dict(
    written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    iterations_used=IT['n_versions'], refund_main=IT['refund_main'], refund_rule=IT['refund_rule'], selection_mean_abs_U=IT['mean_abs_U'],
    bridge_order='起點 會計 c_acc（o20 B_drag 口徑）→ B1 分母/歸屬/年數（c_B1：Σ C/E_pre，成交日 ∈ (s0,s1]，Y＝(s1−s0)/365.25）→ B2 複利（G＝(1＋g^R)(1−e^{−ℓ})，ℓ＝Σlog(1＋C_d/E_post,d)/Y）→ U 未解釋（(g^R−g^A)−G）→ B4 路徑（g^T−g^R）→ B5 時點交互（逐 j Shapley 成本類−(g^T−g^A)）→ B3 中位數不可加（中位數值函數 Shapley − 逐 j Shapley 中位數）→ 終點 13.67。中位數層主欄；三層（增量中位、逐 j 平均、逐 j 最大）並報；逐 j 增量 120 種排列範圍並報。',
    bridge_periods=['Y19on', 'S0618', 'V1520', 'H2126', 'ALL'], bridge_boot_B=1000,
    bridge_decision='主退款模式、REAL_opt、2019 起、中位數層：|U| ≤ 0.3pp →「對帳完成」；否則 B2 ≥ 2.6pp →「部分對帳」；否則「未對帳」。另兩變體與另一退款模式並列。',
    sensitivity='mo×mc＝{0.5,1,2}²，6 格 × 3 變體 × 13 j seed 0；四角＋中心 20 seeds；mt＝0.5 中心 seed 0。每格逐格取最差（opt/p90/carry）；假設區間＝9 格相對中心格（seed 0）的範圍、四角 20 seeds 範圍；抽樣區間＝正式格中位 j 2019 起超額月份 bootstrap 90% CI。',
    sensitivity_decision='「敏感」＝任一 normal std 格（30/100/300/1,000 萬）9 格中有 2019 起年化偏離中心 > 1.5pp，或任一格使 G1（j p10>0）、G2（t≥3）、G7（300 萬/1,000 萬/壓力 全期超額與 p10 > 0）翻號；否則「穩健」。',
    edges='主欄＝實際成交價（名目加權參考價：W 開盤、O 零股基準價、D 收盤，含順延）；並列＝開盤價（o20）與含成本成交價。平均數主欄、中位數並列；月份區塊 bootstrap（訊號月）有效樣本數與 MDE80＝2.8×SE。',
    quintiles='ĉ（PREREG.quintile_plan.chat 定義）與實際 rtc 兩種分組 × 開盤價／實際成交價兩種 edge；期間 Y19on、S0618、V1520、H2126；最低 1/5 − 最高 1/5 平均淨 gap；B＝2000；p_le0、90% CI、MDE80。',
    predictions_thresholds=R['PREREG']['predictions'],
    annualization='日曆年化：(終點收盤/起點收盤)^(365.25/天數)−1。Y19on、POSTODD、V1520、H2126 的起點＝期間內第一個交易日收盤（x_stats 口徑；2019 起＝2019-01-02 收盤，與正式表、Shapley 組合表相同）；ALL、SEL14 起點＝j 第一個訊號日收盤；S0612、S1318、S0618、PREODD 起點＝期間前最後一個收盤（integrate_o19 pstats 口徑）。終點＝j 終點日（jscan_end_days）。',
    annualization_note='中心格 run_engine 成交彙總（center_meta.summ）在口徑統一前產生，其 Y19on 視窗多含 2019-01-02 一個交易日（當日無換股，下單日比例分母多 1 日，影響 < 0.1%）；其他階段為統一後口徑。',
    seeds20_note='navs_seeds20.parquet 由已鎖定 sha 的 run_engine.py seeds 階段在凍結後產生（四角＋中心 20 seeds，純資料），其 sha256 記在 run_engine_meta.json 與 RESULTS.G_files。',
    files_sha256={f: sha(HERE / f) for f in files},
)
R['PREREG_FREEZE'] = FZ
R['PREREG_FREEZE_sha256'] = hashlib.sha256(json.dumps(FZ, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_FREEZE_sha256'], FZ['written_utc'])

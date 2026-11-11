#!/usr/bin/env python3
"""凍結 PREREG 補充（挑選期迭代完成後、任何挑選期以外統計量之前）→ results.json['PREREG_FREEZE'] 與 sha256。只能執行一次。
python3 -I -B freeze.py"""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; RJ = HERE / 'results.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R and 'PREREG_FREEZE' not in R
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
F = dict(
 written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
 prereg_sha256=R['PREREG_sha256'],
 iterations=dict(file='iterations.json', sha256=sha(HERE / 'iterations.json'), n_versions=IT['n_versions'], selected=IT['selected']),
 files_sha256={f: sha(HERE / f) for f in ('tfc_lib.py', 'gen_plans.py', 'run_logs.py', 'build_tables.py', 'diag_lib.py', 'explore.py', 'iterate.py', 'run_shapley.py',
                                          'data/plans_real_allseeds.csv.gz', 'data/ranks_seed0.csv.gz')},
 frozen_cfg=dict(pairing='B（主欄；每檔 exit 對同日全部 entry 的等權平均；edge＝mean r_entry − r_exit，來回成本＝exit 賣出率＋entries 平均買進率）；並列 A2（邊際）、A0（V7X 成交順序）',
                 price_def='被換出股成交日開盤價（無開盤用前一收盤）', cost='ENG 主欄；H05／H10 假設區間', slot='原始加權貢獻 r/n（o19 定義）主欄；超額版並列',
                 reversal='換股日可觀察：sign(0050 近 21 日報酬) ≠ sign(前 21～126 日報酬)；事後版並列'),
 boundaries=dict(price=['<20', '20~50', '50~100', '100~500', '>=500'], P3_low='<50', P3_high='>=100', exit_rank=['<=10', '11~15', '16~30', '31~50', '>50', 'F_pool', 'F_disp', 'F_notrade', 'F_rev', 'F_other', 'F_na'],
                 n_swaps=['<=2', '3~4', '>=5'], slot_band=[-0.02, 0.02], entry_src=['>50/NA', '11~15', '<=10', '16~30', '31~50']),
 tests=dict(bootstrap='月份區塊（訊號日所在日曆月；13 j 同月合併），B＝2,000，seed 20261112，單尾',
            P3='通過（嚴格）＝成本占比 ≥ 1.5x 且 edge<成本 比率差（<50 減 ≥100）≥ 10pp 且 bootstrap p(差 ≤ 0) < 0.05；並列寬鬆讀法＝只看 bootstrap p<0.05',
            P4='通過（嚴格）＝c1（11～15 名 edge 中位數 < 來回成本中位數）且 c2（11～15 名占 edge<成本換股 ≥ 40%）且 c3（>50 名 edge 中位數 > 來回成本中位數）且 bootstrap p（(edge−成本) 中位數差：>50 減 11～15 ≤ 0）< 0.05；並列寬鬆讀法＝只看 bootstrap p',
            P5='c1：≥5 檔換股日每筆 edge 中位數 < ≤2 檔；c2：≥5 檔換股日反轉比例 > ≤2 檔；bootstrap p（≤2 減 ≥5 的 edge 中位數差 ≤ 0）並報',
            P6='band（貢獻 ∈ [−2%, +2%]）成本總和 > band 紙上貢獻總和',
            P2='30 萬／100 萬（2020-10-26 起，seed 0，逐格最差＝比值最小者）每筆衝擊＋價差占名目（總額比）≥ 0.7 成立；< 0.5 否定；並報逐筆中位數比與跳動點層占比',
            P1='6 因子 64 組合 × 720 順序 Shapley（值函數＝13 j 中位數年化；並報逐 j Shapley 的中位數）；成本類＝X＋F＋K＋I＋C；o19 順序 T→X→F→{K,I,C}；成本類任一順序 < 12pp → 「順序相依、未定」',
            P7='β＝挑選期（窗起點 ≤ 2019-04-30）訊號日窗報酬對 0050 窗報酬的 OLS 斜率（逐 j，seed 0 全開淨值）；β-0050＝每日 β×0050 報酬（現金 0%）；報扣成本前（成本全關、時點開）與扣成本後（全開）2019 起對 β-0050 超額；全期 β 並報'),
 worst_rule='opt／p90／carry × 一般／壓力 逐格取最差：bootstrap p 取最大、比率取最不利、成本取最大；結構判定條件 (3)＝所有格方向一致（不翻號）',
 periods=dict(primary='Y19on（訊號日 2019-01-01 起）', direction_check='S0618（2006～2018）', also=['S0612', 'S1318', 'ALL', 'POSTODD（30 萬主欄）', 'Y19on_xsrc3（排除 2024-02-15～2025-09-23 成交）']),
 accounting='6.22pp 對照：Σ（edge<成本 的換股，B 配對）來回成本元 ÷ 換股日可用淨值，逐 j 年化（÷ 期間年數），13 j 中位數，逐格最差；並報全部成交的會計成本拖累',
 auc=dict(target='edge < 來回成本（B）', features=['被換出股價格', '被換出股排名（強制出場記 999）', '分數差 dscore_B', '當日換股檔數', '來回成本本身（事前可估）'], report='AUC 與 max(AUC, 1−AUC) 方向；Y19on 與 S0618'),
 breakeven=dict(table='價格級距（被換出股）× 帳戶 {30萬,100萬,300萬,1000萬} × 成本 {一般,壓力}：來回成本中位數與平均＝每筆換股需要的事後 edge（% / 持有期≈1 月）；換算「每年需要的毛超額」＝ 年化會計成本拖累',
                representative=['100萬 Y19on <50', '100萬 Y19on >=100', '30萬 POSTODD <50', '30萬 POSTODD >=100']),
 robustness=['逐年 edge<成本 比率與 (edge−成本) 中位數', '分產業（stock_list industry）', '去掉紙上貢獻最大 5 檔股票後重算 P4 bootstrap', '排除 src 3 區段'],
 replication_targets=['2019 起會計成本拖累各子項（稅、手續費、跳動點層、衝擊層、零股價差）', 'Shapley 平均（由 data/shapley_coalitions.csv.gz 重算）與 o19 順序階梯', '兩平表 4 代表格', 'P3 成本占比與比率差', 'P4 各類 edge/成本中位數與 11～15 占比'],
 note='凍結後的任何修正一律為事後版並列，凍結版為主欄。',
)
R['PREREG_FREEZE'] = F
R['PREREG_FREEZE_sha256'] = hashlib.sha256(json.dumps(F, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(F['written_utc'], R['PREREG_FREEZE_sha256'])

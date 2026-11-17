#!/usr/bin/env python3
"""S55 寫 SPEC.json／PREREG.json 與各自的 sha256、code.sha256（在 run.py 執行之前；之後不得修改）。用法：python3 -I -B prereg_write.py"""
import json, hashlib
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s55')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE / 'results.json').exists() and not (HERE / 'PREREG.json').exists(), 'PREREG 已寫過或已有結果，不得重寫'
prep = json.loads((HERE / 'prep_check.json').read_text())
now = pd.Timestamp.now('UTC').isoformat()

SPEC = {
    'topic': 'S55 真正樣本外：把凍結的現行規則往前延伸到 2008-01～2015-06，一次跑完，不調任何參數',
    'written_at': now,
    'frozen_rule': {
        'selection_F1': 's10lib.make_signal_s10(depth=10, mom_lo=0.10, nsel=10, shift=0)：訊號日股池（上市、排除創新板、60 日均成交金額 ≥ 1,000 萬、收盤 ≥ 10、非 91 開頭）內 SURGE 有值且還原收盤 > 60 日均線，依 SURGE 由大到小；前 10 名中 20 日報酬 > 10% 者買進（最多 10 檔）',
        'surge': 's17lib.rev_family(defs=[STD])：D＝月營收−12 個月前；SURGE＝(D−前 24 個月 D 平均)／前 24 個月 D 標準差（24 個都要有值、σ>0）；2013 年 D 一律缺值（IFRS 斷點，凍結規則原樣）',
        'exit_E0': '訊號日不在 L_MA 前 20 名就於次日開盤賣',
        'signal_day_A10': 'cslib.rev_avail_days(pit=True)：次月 10 日（遇假日順延）之後第一個交易日為訊號列，次一交易日開盤成交',
        'allocation_W10_70': 's10lib.simulate_s10(sat_w=0.7, sizing=W10, lock=False, band=0.10, slots=10, cap_pct=0.05, sweep_min=2 萬, min_frac=0.5)；0050 底倉 30%，空位買 0050；本金 100 萬',
        'cost': 'COSTS[strict]：手續費 0.1425%×0.28（最低 20 元）、證交稅 0.3%（ETF 0.1%）、每邊額外滑價 0.3%、跳動點半格',
        'switches': 'cslib.EXRIGHT_FIX＝DEADLINE_FIX＝INS15_MASK＝True（s47lib import 時設定）',
        'engine_entry': 's47lib.sim(P, sg, U, cap=1e6)（÷10、空位 0050 → 原引擎 simulate_s10），與 S47／S52 的現行格相同',
        'not_changed': '引擎與既有研究檔案一律不改；只在執行期改資料位置與樣本窗（見 runtime_overrides）'},
    'runtime_overrides': {
        'cache_dirs': '各層快取 → scratchpad/s55/cache_pre（放好自建面板 panel_cs.npz、0050 開高低收 e50_ohlc.csv、創新板 innov.npz 全 False），不呼叫 git',
        'stlib.START': '2008-01-02（回測起點＝P.t0；原 2015-07-01）',
        's17lib.FIRST_MONTH': '2008-01（原 2016-01；這是樣本窗，不是策略參數）',
        'cslib.END': '不變（2026-10-22）；面板本身到 2015-06-30 為止'},
    'data': {
        'panel': 'build_pre_panel.py：2007-01-02～2015-06-30 上市日 K（Asoul/tsec 證交所原始行情，上市期間依 data_panel；含已下市股）；2015 上半年與官方日 K 收、開、成交金額、漲跌價差 100% 相同（panel_check.json）',
        'revenue': '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv（現行引擎同一檔，2005-01 起）',
        '0050': 'cslib.E50_CSV 含息還原收盤（2003 起）；原始開盤 2007～2014 無 → 引擎退路（開盤／收盤比＝1、跳動點以 100 元計）',
        'twt49u': 'raw/twt49u（2014-01 起）；2014 以前除權息日漲跌價差＝0 → 當日報酬記 0',
        'inventory': 'data_inventory.json'},
    'data_sha256': {f: sha(SCR / 'cache_pre' / f) for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz')},
    'prep_check': {'is_reproduced_bitwise': prep['is']['bitwise_ok'], 'is_vs_s52_monthly_max_abs_diff': prep['is']['vs_s52_monthly'],
                   'oos_signal_months': prep['oos']['n_signal_months'], 'oos_first_signal': prep['oos']['first_signal'], 'oos_last_signal': prep['oos']['last_signal'],
                   'note': 'prep.py oos 只算訊號層計數，沒有跑淨值、沒有看任何 2015-07 以前的報酬'},
    'earliest_start': {
        'revenue_first_month': '2005-01', 'D_first_month': '2006-01（需 12 個月前營收）',
        'sigma24_first_signal_month': '2008-01（SURGE 需要前 24 個 D：2006-01～2007-12）',
        'first_signal_day': prep['oos']['first_signal'] + '（2008-01 營收；次一交易日開盤首次成交）',
        'last_signal_day_before_gap': prep['oos']['last_signal'] + '（2012-12 營收）',
        'ifrs_gap': '2013-01～2015-12 營收月份 SURGE 全缺（2013 年 D 設缺值，σ24 視窗含 2013 就算不出）→ 凍結規則在 2013-02～2015-06 沒有訊號、也不出場（「卡住」期，持股停在 2013-01 的名單）'},
    'monthly': '日淨值取每個日曆月最後一個交易日；第一個月以 2008-01-02 為基底（策略＝100 萬、0050＝當日含息還原收盤）；月超額＝策略月報酬−0050 月報酬（算術差），同 S52',
    'stats': '年化超額＝月超額平均×12（百分點）；資訊比率＝平均／標準差×√12；t：iid 與 Newey-West（Bartlett 落後 6，同 S52）；另報幾何年化、最大回撤、beta、逐年',
    'in_sample_series': 'prep.py is 重現的現行正式淨值（2015-07-01＝100 萬～2026-10-22；與 deadline_engine_fix ALL 逐位相同、與 S52 月淨值差 < 1e-8），只用於合併全期',
}
PREREG = {
    'written_at': now,
    'run_once': 'run.py 只跑一次（results.json 存在就拒跑）；結果不論好壞都照登記判定，不調任何參數、不換窗',
    'windows': {
        'primary': {'from': '2008-01', 'to': '2013-01', 'why': '訊號有效期：第一個訊號月到最後一個訊號（2013-01-11）所建持股的下一個應有訊號日之前（61 個月）'},
        'secondary': {'from': '2008-01', 'to': '2015-06', 'why': '凍結規則照跑全段（含 2013-02～2015-06 卡住期，90 個月）'},
        'merged': '合併全期：primary（61 個月）＋樣本內 2015-07～2026-10（136 個月）＝197 個月；另報 secondary＋樣本內＝226 個月（2008-01～2026-10 連續）'},
    'decision': {
        'rule': 'primary 窗：年化超額 > 0 且 Newey-West t > 1.65 →「支持」；年化超額 > 0 但 t ≤ 1.65 →「方向一致但不顯著」；年化超額 ≤ 0 →「不支持」。secondary 窗同規則，只報是否與 primary 同判定。',
        't_threshold': 1.65,
        'also_report': '合併全期（兩種）的資訊比率與 t（iid、NW6）；樣本內單獨（應≈S52：年化超額 11.68 個百分點、資訊比率 0.721、NW t 2.91）'},
    'known_biases_declared_before_run': [
        '2014 以前除權息日報酬記 0（無 TWT49U 參考價；data_exright_fix 量過修正約 +0.86 個百分點／年）→ 對策略偏保守',
        '2007～2014 0050 用當日還原收盤代替開盤成交 → 影響小、方向不定',
        '漲跌停價固定 10%（2015-06-01 以前實際 7%）→ 可能把實際漲停（買不到）的開盤當成買得到；事後報告受影響筆數（不改結果）',
        '下市股營收在最後交易前約 13 個月就斷（data_inventory.json）→ 下市前一年不可能入選，方向多半對策略有利；事後報告「股池內、站上季線、SURGE 缺值、15 個月內下市」的格數',
        'Asoul 缺部分 2008～2012 從上市下市股的上市期行情（例：英群、華宇、精碟），已確認者都是低價股（多半不在股池）',
        '2015 以前沒有 2019-01／2023-03 那種營收期限特例登錄 → 一律用法定 10 日'],
    'diagnostics_preregistered': ['7% 漲跌停下開盤即漲停的買進筆數、開盤即跌停的賣出筆數', '持有期間遇到的 2014 以前除權息日數', '存活者偏誤格數（如上）', '逐年策略／0050 報酬', '交易筆數、勝率、平均持股、最賺 5 筆占比'],
    'code_sha256_file': 'code.sha256（s55lib.py、run.py、prep.py、build_pre_panel.py、inventory.py、fetch_twse.py）',
}
(HERE / 'SPEC.json').write_text(json.dumps(SPEC, ensure_ascii=False, indent=1))
(HERE / 'SPEC.sha256').write_text(sha(HERE / 'SPEC.json') + '  SPEC.json\n')
(HERE / 'PREREG.json').write_text(json.dumps(PREREG, ensure_ascii=False, indent=1))
(HERE / 'PREREG.sha256').write_text(sha(HERE / 'PREREG.json') + '  PREREG.json\n')
(HERE / 'code.sha256').write_text(''.join(f'{sha(HERE / f)}  {f}\n' for f in ('s55lib.py', 'run.py', 'prep.py', 'build_pre_panel.py', 'inventory.py', 'fetch_twse.py')))
print('SPEC', sha(HERE / 'SPEC.json')); print('PREREG', sha(HERE / 'PREREG.json'))

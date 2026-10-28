#!/usr/bin/env python3
"""凍結：選定版本與最終規格寫進 results.json 的 PREREG（含 sha256 與時間戳）。
sha256 涵蓋本資料夾全部 .py、plan／gate／explore／iterations／validate、data/ 與 raw/ 全部檔案、引擎庫（v4lib、drj_core、v2_core、ia_core、o3_core）、
V4 凍結檔 data_freeze_v4.json 與其列出的面板與輸入檔、o_v4 的 navs_v4.parquet／plan_V4.csv／raw_news_mentions.parquet。
之後 oos.py 斷言每個檔案 sha 不變才計算 2019 年後。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
sel = IT['decision']['selected']
fz = json.loads((AG / 'o_v4_revenue_patch_official' / 'data_freeze_v4.json').read_text(encoding='utf-8'))
files = sorted(HERE.glob('*.py')) + [HERE / x for x in ('plan.json', 'gate.json', 'explore.json', 'iterations.json', 'validate.json')]
files += sorted(p for p in (HERE / 'data').rglob('*') if p.is_file()) + sorted(p for p in (HERE / 'raw').rglob('*') if p.is_file())
files += [AG / 'o_v4_revenue_patch_official' / x for x in ('v4lib.py', 'data_freeze_v4.json', 'navs_v4.parquet', 'plan_V4.csv', 'raw_news_mentions.parquet',
                                                          'yoy_long_R_FIX.parquet', 'yoy_rev5y_FIX.parquet')]
files += [AG / 'o_disposal_rejudge' / 'drj_core.py', AG / 'main_v2' / 'v2_core.py', AG / 'independent_audit' / 'ia_core.py', AG / 'o_infra_v3_rebuild' / 'o3_core.py',
          AG / 'rev_announce' / 'p_delay.json', AG / 'rev_announce' / 'firstseen_2026-09.csv', AG / 'o_infra_v3_rebuild' / 'data_freeze.json']
for grp in ('v3_base', 'panel', 'other'):
    for k, v in fz[grp].items(): files.append(Path(v['path']))
H = {str(f): sha(f) for f in files}
for grp in ('v3_base', 'panel', 'other'):
    for k, v in fz[grp].items(): assert H[v['path']] == v['sha256'], f'{k} 與 data_freeze_v4 不符'
SPEC = {
 'name': f'{sel}（J0／J+1 正式化，V4 資料）',
 'signal_day': {
   'J+1': '錨定月 m 的法定期限 D_m＝m 月 10 日起第一個交易日（加權指數日曆；主管機關延長時用延長後期限：2019-02→2019-02-15、2023-04→2023-04-12）；訊號日＝D_m 之後第 1 個交易日；當天收盤算訊號',
   'J0': '訊號日＝D_m；收盤算訊號，只用收盤前已公告營收',
   'J0N': '訊號日＝D_m；價格腿用 D_m 收盤，營收腿用 D_m 當晚已公告值'}[sel],
 'trade_day': '訊號日的次一面板交易日開盤（成交日）；開盤漲停（tick 價）且未持有 → 跳過遞補',
 'months': 'V4 官方訊號月集合（v2 營收可用條件；2024-01 無面板資料跳過）；訊號日或次一日不是 GOOD 面板日 → 該月續抱',
 'strategy': 'v4lib.V4 全部規格：前 300 流動性池（60 日均成交值 gapfix、≥ 1,000 萬、收盤 ≥ 10）、排除金融 fin2、KY／DR／F-／91xx、處置成交日口徑（公告日 ≤ 訊號日、成交日在處置期間）、R_FIX（< 2021-03-01）／Y5_FIX、y3＝m−1..m−3 年增率平均、0.5×52 週比百分位＋0.5×y3 百分位、正式同分序、n10、P、X',
 'availability_AB': 'fx_core.make_unavail(mode, seed)：' + {'J+1': 'day 模式', 'J0': 'day 模式', 'J0N': f'night 模式，q＝{json.loads((HERE / "data" / "q_night.json").read_text())["q"]:.6f}'}[sel] +
                    '；seed 0～39，各 seed 一張名單一條淨值，日報酬平均成正式 AB 淨值；未公告者用 m−2..m−4',
 'cost': '一般：買 0.1425%×0.6＋0.1%，賣 同＋0.3% 稅；壓力：每邊 0.5%；損益兩平格點單邊滑價 0.10%～0.40%（0.02% 步長）',
 'returns': '簡單報酬；年化起點＝該版本 2006-04 訊號日收盤（J+1＝2006-04-11、J0／J0N＝2006-04-10），終點＝2026-10 錨定月同型訊號日收盤（J+1＝2026-10-13、J0／J0N＝2026-10-12）',
 't': '訊號窗（訊號日收盤→下一訊號日收盤）對 0050 含息月超額 t，窗依錨定月歸段；另報日曆月 t（完整月 2006-05～2026-09）',
 'periods': {'SEL': '2006-04～2018-12', 'VAL': '2019-01～2023-12', 'HOLD': '2024-01～2026-10', 'ALL': '2006-04～2026-10'},
 'decision': '可用（只改定義、不改預期）；若選中版本 AB 全期 t ≤ 官方 AB_V4 t（2.512）− 0.1 → 實盤預期下修（用選中版本 AB 年化：0050＋0.5×超額）',
}
P = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'selected_version': sel,
     'selection_trace': IT['decision'], 'plan_sha256': IT['plan_sha256'], 'spec': SPEC, 'sha256': H,
     'versions_count': {'this_topic_iterations': IT['decision']['n_versions_this_topic'], 'oos_reported_candidates': 3, 'jscan_versions': 13,
                        'note': 'jscan 只是穩健性分布，不參與選版'},
     'contamination': 'o4 o_stagger_tranche 已在 V3 資料上看過 J0／J+1 全期（含 2019 年後）；本題樣本外不是盲測'}
canon = json.dumps(P, ensure_ascii=False, sort_keys=True).encode()
f = HERE / 'results.json'
assert not f.exists() or 'PREREG' not in json.loads(f.read_text(encoding='utf-8')), 'PREREG 已存在，不得覆寫'
f.write_text(json.dumps({'PREREG': P, 'PREREG_sha256': hashlib.sha256(canon).hexdigest()}, ensure_ascii=False, indent=1), encoding='utf-8')
print(hashlib.sha256(canon).hexdigest(), P['created_utc'], sel, len(H))

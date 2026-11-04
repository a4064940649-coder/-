#!/usr/bin/env python3
"""第 0 步：資料與制度閘門（不看任何報酬）。輸出 data_gate.json。
閘門：(a) 窗內有價覆蓋 ≥ 99%（FULL 2006-04～2026-10、SEL、VAL 各自）；(b) 公告日全部早於進場決策點（公告日 ≤ 進場前一日）。
用法：python3 -I -B s0_gate.py"""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import dlib as D

K = D.K
G = {'raw_sha256': {p.name: D.sha(p) for p in sorted((HERE / 'raw').glob('*')) if p.is_file() and p.suffix in ('.csv', '.txt', '.json', '.parquet')}}

# ---- 1. 制度時間表（逐年） ----
U = D.U
UC = U[U['in_D']].copy()
UC['y'] = UC['start'].dt.year
tl = {}
for y, g in UC.groupby('y'):
    tl[int(y)] = {'n_records': int(len(g)), 'regime': g['regime'].value_counts().to_dict(),
                  'match_min': {str(k): int(v) for k, v in g['match_min'].fillna(-1).value_counts().items()},
                  'nth2_share': round(float((g['nth'] >= 2).mean()), 3),
                  'limit': '7%' if y < 2015 else ('7%→10%（2015-06-01）' if y == 2015 else '10%'),
                  'odd_lot': '只有盤後零股（14:30 單次撮合）' if y < 2020 else ('2020-10-26 起盤中零股（3 分鐘）' if y == 2020 else
                             ('盤中零股 3 分鐘；2022-12-19 起 1 分鐘' if y in (2021, 2022) else ('盤中零股 1 分鐘；2024-12-02 起 5 秒' if y in (2023, 2024) else '盤中零股 5 秒'))),
                  'daytrade': '現股當沖未開放' if y < 2014 else ('現股當沖（2014-01-06 起）' if y < 2017 else '現股當沖（2017-04-28 起證交稅減半）'),
                  'prepay': ('只預收期：收足五成或全部款券（連續撮合）' if y <= 2010 else
                             '第一次處置：約 5 分鐘撮合，單筆 ≥ 10 張或當日累計 ≥ 30 張預收；第二次：約 20 分鐘撮合，全部預收') +
                            ('；2026-08-10 新制：5 日、2 分鐘撮合' if y == 2026 else '')}
G['regime_timeline'] = tl
G['regime_sources'] = {
    'disposal_list': 'TWSE rwd punish 封存（o_disposal_rejudge），measure 欄：收足五成款券／收足全部款券（≤2010）、第一次處置（5 分）、第二次處置（20 分）、人工管制撮合（10～60 分）、新制 2 分',
    'oddlot_and_continuous': 'o_exec_capacity_tiers/raw/regulation_sources.json（搜尋摘要，未見原文）',
    'github_search': 'gh search code（2026-11-03）：查無證交所「公布或通知注意交易資訊暨處置作業要點」歷年條文封存 repo；只找到二手片段：'
                     'kiang/bribes_map 判決書（2009 年處置「每日委託買賣數量達一定交易單位時應收取全部買進價金」）、'
                     'Chris-WenYuan/taiwan-stock-analyzer 與 aRthur08701244/Intern-in-CTBC 的鉅亨新聞（2021 第二次處置「約每 20 分鐘撮合一次…收取全部買進價金」）。'
                     '沒有可 add_repo 的條文封存，改用事前登記參數網格（零股每邊 0.2／0.4／0.8%，主欄 0.4%）。',
    'unverified': ['處置期間盤中零股撮合間隔（本題假設與整股相同，以最後一盤價＋零股價差成交）', '新制 2026-08-10 預收門檻（沿用 10／30 張）']}

# ---- 2. 事件層統計（主時點 C1×CL 的可執行事件表；不讀報酬欄） ----
ev, unf, st = D.build_x(dict(entry='C1', exit='CL'))
allx = pd.concat([ev[[c for c in ev.columns if c in unf.columns]], unf], ignore_index=True) if len(unf) else ev
allx = D.sub(allx, '2006-04-01', '2026-10-22')
evf = D.sub(ev, '2006-04-01', '2026-10-22')
lead = (allx['s_i'] - allx['ann_i'])
G['build_stats_C1xCL'] = st
G['event_level'] = {
    'n_candidates_after_rules_FULL': int(len(allx)), 'n_filled_FULL': int(len(evf)),
    'lead_tdays_s_minus_announce': {str(k): int(v) for k, v in lead.value_counts().sort_index().items()},
    'share_announce_le_s_minus_2': round(float((lead >= 2).mean()), 4),
    'share_T2_fundable_C1 (ann ≤ s−3)': round(float((lead >= 3).mean()), 4),
    'share_T2_fundable_O2 (ann ≤ s−2)': round(float((lead >= 2).mean()), 4),
    'second_share': round(float(allx['second'].mean()), 4), 'nth2_share': round(float((allx['nth'] >= 2).mean()), 4),
}
# s 日收盤鎖漲停、一字鎖、e 日鎖跌停
s_lock = []; s_one = []; e_lock = []
for r in allx.itertuples():
    j, s = r.j, r.s_i
    c = K.C[s, j]
    s_lock.append(bool(np.isfinite(c) and c >= K.LIMUP[s, j] - 1e-9))
    h = D.HIGH[s, j]
    s_one.append(bool(np.isfinite(c) and c >= K.LIMUP[s, j] - 1e-9 and np.isfinite(K.O[s, j]) and K.O[s, j] >= K.LIMUP[s, j] - 1e-9))
    e = min(r.e_i, K.T - 1); ce = K.C[e, j]
    e_lock.append(bool(np.isfinite(ce) and ce <= K.LIMDN[e, j] + 1e-9))
allx['s_lock'] = s_lock; allx['e_lock'] = e_lock
G['event_level'].update({'s_close_locked_limitup_share': round(float(np.mean(s_lock)), 4),
                         's_open_and_close_at_limitup_share': round(float(np.mean(s_one)), 4),
                         'e_close_locked_limitdown_share': round(float(np.mean(e_lock)), 4),
                         's_lock_by_era': {k: round(float(v), 4) for k, v in allx.groupby(allx['date'].dt.year < 2014)['s_lock'].mean().rename({True: '2006-2013', False: '2014+'}).items()}})
# ---- 3. 各級距需要零股的比例（以 s 日漲停限價下單；10 槽） ----
tiers = {'30萬': 3e4, '100萬': 1e5, '300萬': 3e5, '1000萬': 1e6}
oddt = {}
for nm, slot in tiers.items():
    N = []; NL = []
    for r in allx.itertuples():
        L = K.LIMUP[r.s_i, r.j]
        A = slot * D.cpi_factor(r.s_i)
        n = int(A // L) if np.isfinite(L) and L > 0 else 0
        N.append(n); NL.append(n // 1000)
    N = np.array(N); NL = np.array(NL)
    oddt[nm] = {'needs_oddlot_share': round(float(((N % 1000) > 0).mean()), 4), 'odd_only_share': round(float((NL == 0).mean()), 4),
                'lots_ge_10_share (第一次處置也要預收)': round(float((NL >= 10).mean()), 4), 'median_lots': float(np.median(NL))}
G['oddlot_by_tier'] = oddt
# ---- 4. 閘門 ----
cov = {}
for nm, (a, b) in (('FULL', D.PER['FULL']), ('SEL', D.PER['SEL']), ('VAL', D.PER['VAL']), ('OOS_o9', D.PER['OOS_o9'])):
    x = D.sub(allx, a, b); tot = 0; okc = 0
    for r in x.itertuples():
        e = min(r.e_i, K.T - 1)
        seg = K.TR[r.s_i:e + 1, r.j]; tot += len(seg); okc += int(seg.sum())
    cov[nm] = {'n_events': int(len(x)), 'cell_coverage': round(okc / tot, 5) if tot else None}
ann_ok = bool((allx['ann_i'] <= allx['i0'] - 1).all())
G['gate'] = {'price_coverage': cov, 'announce_before_decision_all': ann_ok,
             'pass': bool(all(v['cell_coverage'] is not None and v['cell_coverage'] >= 0.99 for v in cov.values()) and ann_ok)}
# ---- 5. 名單交叉核對：official_trades_twse（disposal_v2 官方版 2021-10 起）對 o9 規則 C1×CL 事件 ----
ot = pd.read_csv(HERE / 'raw' / 'official_trades_twse.csv', dtype={'code': str}, parse_dates=['date'])
o9ev, _ = K.build(dict(entry='C1', exit='CL'))
key9 = set(zip(o9ev['code'], o9ev['date']))
keyx = set(zip(ev['code'], ev['date']))
keyo = set(zip(ot['code'], ot['date']))
G['crosscheck_official_trades_twse'] = {'n_official_trades': int(len(ot)),
                                        'official_in_o9_rules': round(len(keyo & key9) / len(keyo), 4),
                                        'official_in_exec_rules': round(len(keyo & keyx) / len(keyo), 4),
                                        'note': '差異來自面板（V7 對共用面板）與 R3 改成交規則；只核對 (code, 進場日)'}
# ---- 6. 前瞻帳本：公告時戳與 LATE ----
fl = pd.read_csv(HERE / 'raw' / 'fwd_disposal_ledger.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
fd = pd.read_csv(HERE / 'raw' / 'fwd_disposal_decisions.csv')
flt = fl[fl['market'] == 'TWSE']
m = flt.merge(D.load_list()[['code', 'start', 'announce']].rename(columns={'announce': 'ann_list'}), on=['code', 'start'], how='left')
G['forward_ledger'] = {'n_rows': int(len(fl)), 'n_twse': int(len(flt)), 'matched_to_list': int(m['ann_list'].notna().sum()),
                       'announce_equal_share': round(float((m['announce'] == m['ann_list'])[m['ann_list'].notna()].mean()), 4),
                       'ts_ok_share': round(float(flt['ts_ok'].mean()), 4) if 'ts_ok' in flt else None,
                       'decisions': fd.to_dict('records'),
                       'clock_kinds': flt['clock_kind'].value_counts().to_dict(),
                       'announce_ts_hour': flt['announce_ts'].str[11:16].value_counts().head(5).to_dict()}
cal = HERE / 'calib5m.json'
if cal.exists():
    G['calib5m'] = json.loads(cal.read_text())['frozen_shares']
json.dump(G, open(HERE / 'data_gate.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: G[k] for k in ('event_level', 'oddlot_by_tier', 'gate', 'crosscheck_official_trades_twse')}, ensure_ascii=False, indent=1, default=str))
print(json.dumps(G['forward_ledger'], ensure_ascii=False, default=str)[:1500])

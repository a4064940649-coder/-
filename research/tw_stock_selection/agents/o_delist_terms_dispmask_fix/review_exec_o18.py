#!/usr/bin/env python3
"""反方審查（成本與可成交）o18 題 2：只讀研究員檔案，不修改。python3 -I -B review_exec_o18.py
檢查：
 1. V7X 20 起結算事件的核實來源：逐字原文 vs WebSearch 摘要；raw/ws_*.txt 是否含研究員註記（非逐字）
 2. 事前可行性 best_lo 與正式 v3_lo 的差（凍結前是否已看到正式下界）
 3. 3063（下界最大貢獻）：以 TPEx 面板 6147 頎邦檢查換股追蹤與空窗期存續股走勢（固定 lc 是否為不利端）
 4. 官方處置清單逐年列數、take_R1、announce 欄是否保留"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; AG = H.parent
R = json.loads((H / 'results.json').read_text())
E = pd.read_csv(H / 'events_v3.csv', dtype={'code': str})
out = {}
# 1
v20 = R['PREREG']['event_list']['V7X_settled_20']; o17 = R['PREREG']['event_list']['V7X_verified_o17']
ver = R['PREREG']['events_v3']['v7x20_verified']
new = [c for c in ver if c not in o17]
kind = dict(zip(E.code, E.src_kind))
out['verified'] = dict(n=len(ver), of=len(v20), new=new, new_kinds={c: kind.get(c) for c in new},
                       verbatim_only=len(o17), verbatim_ratio=len(o17) / len(v20),
                       margin_events=len(ver) - int(np.ceil(0.5 * len(v20))))
ann = {}
for f in sorted((H / 'raw').glob('ws_*.txt')):
    t = f.read_text(encoding='utf-8')
    ann[f.name] = bool(re.search(r'（註：', t))
out['ws_files_with_researcher_note'] = [k for k, v in ann.items() if v]
log = pd.read_csv(H / 'search_log_web.csv', dtype={'event_code': str})
lead = {}
for r in E[E.code.isin(new)].itertuples():
    nums = [x for x in [r.cash, r.ratio_text] if isinstance(x, (int, float)) and np.isfinite(x)]
    q = ' '.join(log[log.event_code == r.code]['query'])
    lead[r.code] = [s for s in re.findall(r'\d+(?:\.\d+)?', q) if not re.fullmatch(r'20\d\d', s)]
out['query_numbers_new_verified'] = lead
# 2
fb = R['PREREG']['feasibility']['best_lo']['exc_med']; lo = R['VERDICT']['a']['lo']
out['feasibility_vs_formal_lo'] = dict(best_lo_pre_freeze=fb, formal_lo=lo, diff_pp=lo - fb,
                                       feas_utc=json.loads((H / 'feasibility.json').read_text())['utc'], prereg_utc=R['PREREG_utc'])
# 3
TP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tpex_panel/close_raw.parquet')
m = pd.read_csv(H / 'raw' / 'market_3063.csv', parse_dates=['date']).set_index('date')['target']
if TP.exists():
    s = pd.read_parquet(TP)['6147']
    j = pd.concat([m, s.reindex(m.index)], axis=1, keys=['t', 's']).dropna()
    rr = np.log(j).diff().dropna().iloc[-20:]
    b = np.polyfit(rr.s, rr.t, 1); res = rr.t - np.polyval(b, rr.s); r2 = 1 - res.var() / rr.t.var()
    ratio = float((j.t / j.s).iloc[-10:].median()); lc = float(m.iloc[-1])
    sp = s.loc['2010-03-19':'2010-06-30'].dropna()
    val = {d: round(ratio * float(sp.loc[d]) / lc - 1, 4) for d in ['2010-04-01', '2010-04-30', '2010-05-31', '2010-06-30'] if d in sp.index}
    out['3063_vs_6147'] = dict(R2_20d=round(float(r2), 3), beta=round(float(b[0]), 3), implied_ratio=round(ratio, 4), lc=lc,
                               swap_value_vs_lc=val, min_vs_lc_to_0630=round(ratio * float(sp.min()) / lc - 1, 4),
                               engine_lo=-0.12, note='面板內 6147 存在（TPEx），研究員稱不在面板僅指 V7X 上市面板')
else:
    out['3063_vs_6147'] = 'TPEx panel 不存在，無法檢查'
# 4
du = pd.read_csv(AG / 'o_rev_avail_ds_pit_count' / 'raw' / 'disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
w = du[(du.start >= '2014-01-01') & (du.start <= '2021-09-23')]
out['disposal'] = dict(cols=du.columns.tolist(), has_announce='announce' in du.columns,
                       rows_by_year={int(k): int(v) for k, v in du.groupby(du.start.dt.year).size().items() if 2013 <= k <= 2022},
                       win_rows=len(w), win_take_R1_false=int((~w.take_R1).sum()), win_second30=int(w.second30.sum()))
# 5 V7X 原始數字（本題外、可成交面）
out['v7x_cells_v3_main'] = dict(normal_100=1.500, stress_100=-5.998, normal_300=-0.238, normal_1000=-3.105, Y19on_normal_100=-6.158)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

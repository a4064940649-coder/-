#!/usr/bin/env python3
"""2026-10-12 訊號日：M1～M4 名單（隔日 10-13 開盤漲停遞補已套用）與主策略 ex-fin 對照。輸出 results_today.json。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ms_core as K

SD = pd.Timestamp('2026-10-12')
TOOL = ['2305', '2347', '2466', '6834', '2408', '6213', '2383', '3653', '1312', '3231']
names = K.SL.set_index('code')['name']
d5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str}); dh = pd.read_csv(K.X / 'rev_hist.csv.gz', dtype={'code': str})
k5 = set(zip(d5['code'], d5['ym'])); dh = dh[[(c, y) not in k5 for c, y in zip(dh['code'], dh['ym'])]]
d = pd.concat([d5[['code', 'ym', 'revenue', 'yoy']], dh[['code', 'ym', 'revenue', 'yoy']]], ignore_index=True)
d['p'] = pd.PeriodIndex(d['ym'], freq='M'); d = d.drop_duplicates(['p', 'code'], keep='last')
Y = d.pivot(index='p', columns='code', values='yoy'); Y = Y.reindex(pd.period_range(Y.index.min(), Y.index.max(), freq='M'))
LV, BS = K.rev_levels_5y(merge_hist=True)
prev = json.loads((K.HERE / 'picks.json').read_text())
t = K.IDX.get_loc(SD); e = t + 1; gu = K.gap_up_open(e)
out = {'signal_day': str(SD.date()), 'exec_day': str(K.IDX[e].date()), 'tool_exfin': TOOL}

def take(cand, held, n=10):
    pick, skip = [], []
    for j in cand:
        if gu[j] and K.COLS[j] not in held: skip.append(K.COLS[j]); continue
        pick.append(j)
        if len(pick) == n: break
    return pick, skip

# 主策略 ex-fin（本引擎重算）
SX = K.Strategy(Y, '2021-01-01', fin_exclude=set(K.COLS[K.FIN]))
cs = SX.cross_section(SD); od = SX.order(cs)[:int(cs['el'].sum())]
heldm = set(prev['r5y|MAIN_exfin'].get('2026-09-11', []))
pm, sk = take(od, heldm)
out['main_exfin_engine'] = [K.COLS[j] for j in pm]; out['main_exfin_engine_skipped_limitup'] = sk
out['main_exfin_engine_vs_tool_overlap'] = len(set(out['main_exfin_engine']) & set(TOOL))
print('main ex-fin（本引擎）', out['main_exfin_engine'], '跳過', sk, '與工具重疊', out['main_exfin_engine_vs_tool_overlap'])
S0 = K.Strategy(Y, '2021-01-01'); cs0 = S0.cross_section(SD); od0 = S0.order(cs0)[:int(cs0['el'].sum())]
p0, _ = take(od0, set(prev['r5y|MAIN'].get('2026-09-11', []))); out['main_orig_engine'] = [K.COLS[j] for j in p0]
print('main 原版（含金融）', out['main_orig_engine'])
for r in ['M1', 'M2', 'M3', 'M4']:
    M = K.Manual(r, Y, LV, BS, [SD])
    cand, f = M.candidates(SD, return_frame=True)
    held = set(prev[f'r5y|{r}'].get('2026-09-11', []))
    pk, sk = take(cand, held)
    R1 = K.r1_for_month(Y, pd.Period(SD, 'M')); R3 = K.r3_for_month(LV, BS, pd.Period(SD, 'M'))
    rows = [{'code': K.COLS[j], 'name': names.get(K.COLS[j], ''), 'close': float(K.Cn[t, j]), 'dist_1y_high_pct': round(float(f['D'][j]) * 100, 2),
             'rev3m_yoy_pct': round(float(R3[j]), 1), 'rev1m_yoy_pct': round(float(R1[j]), 1)} for j in pk]
    ov = len({x['code'] for x in rows} & set(TOOL))
    out[r] = {'n_candidates': int(len(cand)), 'picks': rows, 'skipped_limitup': sk, 'overlap_with_tool_exfin': ov,
              'overlap_with_main_exfin_engine': len({x['code'] for x in rows} & set(out['main_exfin_engine']))}
    print(r, '候選', len(cand), '跳過', sk, '與工具名單重疊', ov)
    for x in rows: print('   ', x)
# 工具名單在網站口徑下的數值
R3 = K.r3_for_month(LV, BS, pd.Period(SD, 'M')); W = K.web_signals()
out['tool_list_web_values'] = [{'code': c, 'name': names.get(c, ''), 'dist_1y_high_pct': round(float(W['D'][t, K.CI[c]]) * 100, 2),
                                'rev3m_yoy_pct': round(float(R3[K.CI[c]]), 1)} for c in TOOL]
for x in out['tool_list_web_values']: print('tool', x)
(K.HERE / 'results_today.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

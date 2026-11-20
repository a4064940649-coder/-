#!/usr/bin/env python3
"""稽核獨立重建（三）：減資／面額變更恢復買賣日事件表，逐筆算「用事件參考價的當日報酬」對「引擎現在記的報酬」。
事件來源：官方日 K 倉庫 data/universe/reduce、parvalue（extract_raw.py 抽出）。引擎現況：cslib.Panel（EXRIGHT_FIX＝True，只讀不改）。
輸出 data/reduce_events_mine.csv、data/reduce_compare_vs_eng.csv、data/reduce_build.json"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; D = HERE / 'data'; AG = HERE.parent
sys.path.insert(0, str(AG / 'o_ext_canslim_xs'))
import cslib
cslib.TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r_fix/cache')
P = cslib.Panel(exright_fix=True)
ev = pd.concat([pd.read_csv(D / 'ev_reduce_tsd.csv', dtype=str).assign(kind='reduce'), pd.read_csv(D / 'ev_parvalue_tsd.csv', dtype=str).assign(kind='parvalue')], ignore_index=True)
ev['date'] = pd.to_datetime(ev['date'])
for c in ('pre_close', 'ref_price', 'open_base'):
    ev[c] = pd.to_numeric(ev[c], errors='coerce')
out = dict(n_raw=len(ev), dup=int(ev.duplicated(['date', 'stock_id']).sum()), file_date_mismatch=int((ev['file'].str[:10] != ev['date'].dt.strftime('%Y-%m-%d')).sum()))
S, E = pd.Timestamp('2015-07-01'), cslib.END
ev = ev[(ev['date'] >= S) & (ev['date'] <= E)]
out['n_in_window'] = len(ev)
di = {d: i for i, d in enumerate(P.IDX)}
rows = []; drop = []
for _, r in ev.iterrows():
    code = r['stock_id']; d = r['date']
    if code not in P.ci or d not in di:
        drop.append(dict(date=str(d.date()), code=code, why='不在面板（非四碼普通股或非交易日）')); continue
    t, j = di[d], P.ci[code]
    if P.MK[t, j] != 1:
        drop.append(dict(date=str(d.date()), code=code, why=f'當天市場別＝{int(P.MK[t, j])}（不是上市）')); continue
    if not P.has[t, j]:
        # 恢復買賣日沒成交：看後面第一個成交日引擎怎麼記
        tt = t + 1
        while tt < P.T and not P.has[tt, j]: tt += 1
        drop.append(dict(date=str(d.date()), code=code, why='當天沒成交', next_trade=str(P.IDX[tt].date()) if tt < P.T else None,
                         next_engine_R=float(P.R[tt, j]) if tt < P.T else None, next_close=float(P.C[tt, j]) if tt < P.T else None, ref=float(r['ref_price']), pre_close=float(r['pre_close'])))
        continue
    tp = t - 1
    while tp >= 0 and not P.has[tp, j]: tp -= 1
    pc = float(P.C[tp, j]) if tp >= 0 else np.nan
    C = float(P.C[t, j]); O = float(P.O[t, j]); ref = float(r['ref_price']); eR = float(P.R[t, j]); tR = C / ref - 1
    # 退還股款：由 參考價＝(前收−每股退款)÷(1−退款÷10) 反推每股退款（假設面額 10 元）；精確持有人報酬＝((1−比例)×收盤＋退款)÷前收−1
    cash = np.nan; exact = np.nan
    if r['kind'] == 'reduce' and r['reason'] == '退還股款' and ref != 10:
        cash = (ref - r['pre_close']) / (ref / 10 - 1)
        if 0 < cash < 10:
            exact = ((1 - cash / 10) * C + cash) / r['pre_close'] - 1
    rows.append(dict(date=str(d.date()), code=code, kind=r['kind'], reason=r['reason'] if isinstance(r['reason'], str) else '', evt_pre_close=float(r['pre_close']), evt_ref=ref, open_base=float(r['open_base']),
                     prev_trade_date=str(P.IDX[tp].date()) if tp >= 0 else None, prev_day_traded=bool(P.has[t - 1, j]), panel_prev_close=pc, pre_close_ok=bool(abs(pc - r['pre_close']) < 1e-9),
                     open=O, close=C, official_chg=None, engine_REF=float(P.REF[t, j]), engine_R=eR, true_R=tR, engine_case=('記 0' if abs(eR) < 1e-12 else '當一般漲跌'),
                     factor_true_over_engine=(1 + tR) / (1 + eR), true_R_beyond_limit=bool(abs(tR) > 0.101), implied_cash_per_share=cash, exact_holder_R=exact,
                     approx_minus_exact_pp=(tR - exact) * 100 if np.isfinite(exact) else np.nan))
m = pd.DataFrame(rows)
m.to_csv(D / 'reduce_events_mine.csv', index=False)
out.update(n_events=len(m), by_kind=m['kind'].value_counts().to_dict(), by_case=m['engine_case'].value_counts().to_dict(), pre_close_mismatch=int((~m['pre_close_ok']).sum()),
           beyond_limit=int(m['true_R_beyond_limit'].sum()), dropped=drop,
           mean_bias_pct={k: round(float((g['factor_true_over_engine'] - 1).mean() * 100), 3) for k, g in m.groupby('engine_case')},
           case_by_prev_traded=pd.crosstab(m['engine_case'], m['prev_day_traded']).to_dict(),
           approx_vs_exact=dict(n=int(m['exact_holder_R'].notna().sum()), mean_pp=float(m['approx_minus_exact_pp'].mean()), max_abs_pp=float(m['approx_minus_exact_pp'].abs().max()),
                                mean_abs_pp=float(m['approx_minus_exact_pp'].abs().mean())))
eng = pd.read_csv(AG / 'eng_audit_fixes_2026-11' / 'reduce' / 'event_cells.csv', dtype={'code': str})
j = m.merge(eng, on=['date', 'code'], how='outer', suffixes=('', '_eng'), indicator=True)
both = j[j['_merge'] == 'both']
out['vs_eng'] = dict(n_mine=len(m), n_eng=len(eng), both=len(both), only_mine=j.loc[j['_merge'] == 'left_only', ['date', 'code']].values.tolist(), only_eng=j.loc[j['_merge'] == 'right_only', ['date', 'code']].values.tolist(),
                     max_abs_true_R_diff=float((both['true_R'] - both['true_R_eng']).abs().max()), max_abs_engine_R_diff=float((both['engine_R'] - both['engine_R_eng']).abs().max()),
                     max_abs_ref_diff=float((both['evt_ref'] - both['evt_ref_eng']).abs().max()), kind_diff=int((both['kind'] != both['kind_eng']).sum()),
                     eng_cases=eng['engine_case'].value_counts().to_dict())
j[['date', 'code', '_merge', 'kind', 'evt_ref', 'evt_ref_eng', 'engine_R', 'engine_R_eng', 'true_R', 'true_R_eng']].to_csv(D / 'reduce_compare_vs_eng.csv', index=False)
json.dump(out, open(D / 'reduce_build.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(m[m['code'] == '8249'].T)

#!/usr/bin/env python3
"""模型迭代（只用挑選期 2021-10～2023-12；只改執行模型、不改事件規則；只看非報酬診斷）。
E1：o9 V7X 層原樣重現（錨點：402／926 筆）。
E2：＋最後一盤成交、參與上限 20%×(V×最後一盤占比)、零股拆單（兩張單各付最低手續費 20 元）。
E3：＋嚴格預收款券與逐日現金帳（現金池 k=2；出場款 T+2）。
E4：＋R3／R5 改成交規則，所有過濾只用 s−1 資訊（＝可執行全模型）。
每輪記錄：成交率、縮單率、零股比例、跳過率、現金帳可行性（已交割現金從不為負）、和前瞻帳本的旗標吻合度。
選版準則（跑之前寫死）：E4 是唯一同時滿足四項可執行要求的版本；E2／E3 只是逐步拆解，不可被選。
若 E4 現金帳出現負現金或成交率 < 50% → 修正實作（最多 3 輪），不改經濟參數。
用法：python3 -I -B s3_iterate.py"""
import sys, json, datetime
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import dlib as D

K = D.K
A, B = D.PER['SEL']
prm = D.default_prm()
IT = {'created': datetime.datetime.utcnow().isoformat() + 'Z', 'script_sha256': D.sha(Path(__file__)), 'dlib_sha256': D.sha(HERE / 'dlib.py'),
      'calib_sha256': D.sha(HERE / 'calib5m.json'), 'versions': []}


def o9_events():
    df, st = K.build(dict(entry='C1', exit='CL'))
    m = D.U.set_index('ordr')['ann_i']
    df['ann_i'] = df['ordr'].map(m).astype(int)
    df['filled'] = True
    return df, st


def diag_exec(df, slots=(3e4, 1e5, 3e5, 1e6)):
    o = {}
    for sl in slots:
        x = D.event_exec(df, sl, prm)
        o[f'slot_{int(sl)}'] = {'n': int(len(x)), 'mean_fill': round(float(x['fill'].mean()), 3), 'capped_share': round(float(x['capped'].mean()), 3),
                                'odd_only_share': round(float((x['nl'] == 0).mean()), 3), 'mean_sell_days': round(float(x['sell_days'].mean()), 2),
                                'zero_fill': int(x['ret_x'].isna().sum())}
    return o


def diag_ledger(ev, unf, prepay):
    out = {}
    for k in (1, 2, 3):
        nav, st, rows = D.sleeve(ev, unf, tier_2026=1e6, prepay=prepay, pool_k=k, t_start=A, t_end=B)
        out[f'k{k}'] = {kk: st[kk] for kk in ('decided', 'entered', 'no_slot', 'cash_skip', 'cash_partial', 'unfilled_lock', 'cash_skip_rate',
                                              'mean_invested', 'mean_cash_share', 'min_settled_cash')}
    return out


# E1
e1, st1 = o9_events()
IT['versions'].append({'id': 'E1', 'desc': 'o9 V7X 層原樣（K.build＋K.exec_cost_event）',
                       'anchor_counts': {'OOS_2006-04_2021-09': int(len(D.sub(e1, '2006-04-01', '2021-09-30'))),
                                         'FULL_2006-04_2026-10': int(len(D.sub(e1, '2006-04-01', '2026-10-22'))), 'SEL': int(len(D.sub(e1, A, B)))},
                       'anchor_expected': {'OOS': 402, 'FULL': 926},
                       'sel_v7x_fill_10萬': round(float(K.exec_cost_event(D.sub(e1, A, B), 1e5)['fill'].mean()), 3)})
IT['versions'][-1]['anchor_ok'] = IT['versions'][-1]['anchor_counts']['OOS_2006-04_2021-09'] == 402 and IT['versions'][-1]['anchor_counts']['FULL_2006-04_2026-10'] == 926
e1s = D.sub(e1, A, B).reset_index(drop=True)
# E2
IT['versions'].append({'id': 'E2', 'desc': 'E1 事件＋最後一盤成交、參與上限 20%×V×最後一盤占比、零股拆單（漲停限價下單、股數＝預算÷限價）',
                       'exec': diag_exec(e1s), 'ledger_none': diag_ledger(e1s, None, 'none')['k2']})
# E3
IT['versions'].append({'id': 'E3', 'desc': 'E2＋嚴格預收款券、T+2 逐日現金帳（現金池 k 槽，主欄 k=2）',
                       'ledger_strict': diag_ledger(e1s, None, 'strict'), 'ledger_threshold': diag_ledger(e1s, None, 'threshold')['k2']})
# E4
e4, u4, st4 = D.build_x(dict(entry='C1', exit='CL'))
e4s = D.sub(e4, A, B).reset_index(drop=True); u4s = D.sub(u4, A, B).reset_index(drop=True)
key1 = set(zip(e1s.code, e1s.date)); key4 = set(zip(e4s.code, e4s.date))
IT['versions'].append({'id': 'E4', 'desc': 'E3＋R3 改「漲停價掛最後一盤、收盤鎖漲停未成交」、R5 改「跌停價掛單、收盤鎖跌停順延」；所有過濾只用 s−1 資訊',
                       'n_filled': int(len(e4s)), 'n_unfilled_lock': int(len(u4s)), 'overlap_with_E1': int(len(key1 & key4)),
                       'only_E4': int(len(key4 - key1)), 'only_E1': int(len(key1 - key4)), 'build_stats_full': st4,
                       'exec': diag_exec(e4s), 'ledger_strict': diag_ledger(e4s, u4s, 'strict')})
# 現金帳可行性檢查：全期主欄（不讀報酬，只讀現金狀態）
nav, stF, rows = D.sleeve(e4, u4, prepay='strict', pool_k=2, t_start='2006-04-01', t_end=B)
IT['versions'][-1]['ledger_feasibility_2006_2023'] = {k: stF[k] for k in ('entered', 'cash_skip', 'cash_skip_rate', 'mean_cash_share', 'min_settled_cash')}
# 9 時點（E4）的成交／跳過診斷（挑選期）
tim = {}
for en in ('O1', 'C1', 'O2'):
    for xk in ('CL1', 'CL', 'OA'):
        e, u, s_ = D.build_x(dict(entry=en, exit=xk))
        es = D.sub(e, A, B); us = D.sub(u, A, B)
        nav, st, _ = D.sleeve(e, u, prepay='strict', pool_k=2, t_start=A, t_end=B)
        tim[f'{en}x{xk}'] = {'n_filled': int(len(es)), 'n_unfilled_lock': int(len(us)), 'entered': st['entered'], 'cash_skip_rate': st['cash_skip_rate'],
                             'no_slot': st['no_slot']}
IT['versions'][-1]['timings_sel'] = tim
# 前瞻帳本吻合度（2026-08 起上市事件：chain_take、second、撮合分鐘）
fl = pd.read_csv(HERE / 'raw' / 'fwd_disposal_ledger.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
fl = fl[(fl.market == 'TWSE') & (fl.start >= '2026-08-10')]
m = fl.merge(D.U[['code', 'start', 'take_R1', 'second30', 'match_min', 'announce']], on=['code', 'start'], how='inner', suffixes=('_fw', '_me'))
IT['forward_ledger_check'] = {'n_twse_newreg_rows': int(len(fl)), 'matched': int(len(m)),
                              'chain_take_agree': round(float((m['chain_take'].astype(bool) == m['take_R1']).mean()), 4),
                              'second_agree': round(float((m['second'].astype(bool) == m['second30']).mean()), 4),
                              'match_min_agree': round(float((m['match_min_fw'] == m['match_min_me']).mean()), 4),
                              'announce_agree': round(float((m['announce_fw'] == m['announce_me']).mean()), 4),
                              'note': '帳本沒有成交價欄，無法比對價格；帳本 announce_ts 一律 14:30（推定），first_seen_ts 為回填，ts_ok=0。'}
IT['selected'] = 'E4'
IT['selection_rule'] = '事前寫死：E4 為可執行全模型；E1～E3 為拆解步驟。只看非報酬診斷。'
IT['n_versions'] = 4
IT['n_rounds_used'] = 3
json.dump(IT, open(HERE / 'iterations.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(IT, ensure_ascii=False, indent=1, default=str)[:7000])

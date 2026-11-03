#!/usr/bin/env python3
"""彙整：G1～G8 逐條、判定、P1～P6 事前預測檢查、可操作規則書（最新訊號日的產業表與 100 萬／1,000 萬下單表）。只讀 results.json，不再跑報酬。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lclib as X
L = X.L
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
PRE = R['PREREG']; assert X.canon(PRE) == R['PREREG_sha256']
S = PRE['chosen']['cfg']; D = PRE['default']['cfg']
O = R['OOS']; M = O['main']; E = R['EXEC']['cells']; PG = R['PLACEBO_G3']
def g(k, a='jmed', b='mean_pp'): return M[k][a][b]
fin = {}
# ---------- G1 ----------
g1 = {}
for nm in (S, D):
    for cell in ('100萬|GROW', '300萬|GROW', '1000萬|GROW', '100萬|FIX', '30萬|FIX'):
        for c in ('normal', 'stress'):
            v = E[f'{nm}|{cell}|{c}']['VAL']
            g1[f'{nm}|{cell}|{c}'] = dict(jmed=v['vs0050_mean_jmed'], p10=v['vs0050_mean_p10'], t=v['vs0050_t_jmed'])
fin['G1'] = dict(rule='可執行 K5T，一般與壓力成本，驗證期對 0050 月超額 j 中位數 > 0 且 p10 > 0', cells=g1,
                 pass_=all(x['jmed'] > 0 and x['p10'] > 0 for k, x in g1.items() if k.startswith(S)))
# ---------- G2 ----------
fin['G2'] = dict(full_2006_2026='不可評估（D1 未過，2006～2013 只作描述）', formal_2014_2026_t=g(f'{S}|normal|vs0050|FORMAL', 'jmed', 't'),
                 desc_2006_2013_t_nonPIT=g(f'{S}|normal|vs0050|DESC', 'jmed', 't'), y19_t=g(f'{S}|normal|vs0050|Y19', 'jmed', 't'),
                 y19_t_exec_100萬GROW=E[f'{S}|100萬|GROW|normal']['Y19']['vs0050_t_jmed'], pass_=False)
# ---------- G3 ----------
fin['G3'] = {fam: dict(p_SPA=PG[f'G3|{fam}'][S]['p_SPA'], DSR=PG[f'G3|{fam}'][S]['DSR'], DSR_LiJi=PG[f'G3|{fam}'][S]['DSR_LiJi'], t=PG[f'G3|{fam}'][S]['t'],
                       N=PG[f'G3|{fam}']['N'], n_windows=PG[f'G3|{fam}']['n_windows'], best_of_40=PG[f'G3|{fam}']['best_of_40'])
             for fam in ('all2334', 'F_main')}
fin['G3']['pass_'] = all(fin['G3'][f]['p_SPA'] < 0.05 and fin['G3'][f]['DSR'] >= 0.95 for f in ('all2334', 'F_main'))
# ---------- G4 ----------
segs = {p: g(f'{S}|normal|vs0050|{p}') for p in ('SEG1', 'SEG2', 'SEG3')}
fin['G4'] = dict(segments_vs0050_pp=segs, seg1_note='2006～2012 為非 PIT 描述段', t_drop_best3_VAL=M[f'{S}|normal|vs0050|VAL']['t_drop3_jmed'],
                 t_drop_best3_FORMAL=M[f'{S}|normal|vs0050|FORMAL']['t_drop3_jmed'], neighborhood=O['neighborhood_flip'],
                 pass_=False)
# ---------- G5～G8 ----------
gate = json.loads((X.DATA / 'gate.json').read_text(encoding='utf-8'))
fin['G5'] = dict(D1=gate['D1_pass'], D1a={k: v for k, v in gate['D1a'].items() if k != 'by_q'}, D1b_med_abs=gate['D1b']['med_abs_log'],
                 notes=['2014 起股數沿用 o8 shares_pit（季內內插有前視，o8 審查已註明）', '產業表 2015-05 以前回溯沿用最早公開快照（約 34 檔下市股人工推定）',
                        '營收沿用 V4 已公告口徑 y3_ab', '2013 以前無股數下市股以 ADV 插補（前 100 中平均 2～10 檔）'], pass_=False)
fin['G6'] = dict(status='待複製員（本題未做）', pass_=False)
g7 = {c: dict(VAL_jmed=E[f'{S}|{c}|normal']['VAL']['vs0050_mean_jmed'], VAL_t=E[f'{S}|{c}|normal']['VAL']['vs0050_t_jmed'],
              Y19_jmed=E[f'{S}|{c}|normal']['Y19']['vs0050_mean_jmed']) for c in ('100萬|GROW', '300萬|GROW', '1000萬|GROW')}
fin['G7'] = dict(cells=g7, pass_=all(v['VAL_jmed'] > 0 and v['VAL_t'] >= 2 for v in g7.values()))
fin['G8'] = dict(status='未開始（凍結 2026-11-03；最早 2027-11 才滿 12 個月）', pass_=False)
# ---------- 判定 ----------
pl = PG[f'placebo|{S}|VAL']
t_val = g(f'{S}|normal|vs0050|VAL', 'jmed', 't')
fin['verdict'] = dict(all_hist_pass=all(fin[k]['pass_'] for k in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7')),
                      watch_condition=dict(val_t_vs0050=t_val, placebo_p_gross=pl['p_gross'], G1=fin['G1']['pass_'],
                                           met=bool(t_val >= 2 and pl['p_gross'] < 0.05 and fin['G1']['pass_'])),
                      supplement_main=dict(combo=O['combo'], corr=O['P6']['corr_jmed'],
                                           met=bool(O['combo']['main70_VAL']['p_boot_improve'] is not None and O['combo']['main70_VAL']['p_boot_improve'] < 0.10 and O['P6']['corr_jmed'] < 0.5)))
fin['verdict']['label'] = '可用（歷史最高標準通過，前瞻驗證中）' if fin['verdict']['all_hist_pass'] else ('待觀察' if fin['verdict']['watch_condition']['met'] else '不可用')
# ---------- 差距 ----------
ir_b = g(f'{S}|normal|vsB|FORMAL', 'jmed', 'IR'); ir_5 = g(f'{S}|normal|vs0050|FORMAL', 'jmed', 'IR')
fin['gap'] = dict(IR_vs0050_FORMAL=ir_5, IR_vsB_FORMAL=ir_b, IR_needed_G2=0.66, IR_needed_Y19=0.72,
                  years_for_t3_at_IR_vsB=(3 / ir_b) ** 2 if ir_b > 0 else None, years_for_t2_at_IR_vsB=(2 / ir_b) ** 2 if ir_b > 0 else None)
# ---------- 事前預測檢查 ----------
mech = O['mech'][f'{S}|FORMAL']
fin['predictions_check'] = dict(
    P1=dict(pred='COMP 產業 IC t ≥ 2', actual=json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))['P1'], held=False),
    P2=dict(pred='VAL 對 B +0.02pp、IR 0.15；對 0050 IR −0.25；TE 1.5～2.5%', actual=dict(vsB=M[f'{S}|normal|vsB|VAL']['jmed'], vs0050=M[f'{S}|normal|vs0050|VAL']['jmed'])),
    P3=dict(REV3_j0_4_minus_j8_12_gross_pp=O['P3_REV3_j0_4_minus_j8_12'], jcurve=O['P3_jcurve_gross_vsB_VAL'], disp_hi=mech['disp_hi_pp'], disp_lo=mech['disp_lo_pp'],
            post_bear=mech['post_bear_pp'], Y19=mech['gross_Y19_pp'], SEG2=mech['gross_SEG2_pp'], semis=mech['semis_contrib_pp']),
    P4=dict(pred='VAL 毛安慰劑百分位約 70', actual=pl),
    P5=dict(pred='拿掉 2021 航運＋2025 AI 後 t < 0.5', t_after=mech['t_drop_2021ship_2025AI'], t_after_top10=mech['t_drop_top10_sectormonths'], t_before=mech['gross_total_t']),
    P6=dict(pred='相關約 0.2；對 0050 β 約 −0.3', actual=O['P6']))
# ---------- 規則書：最新訊號日（j=5）----------
ls = O['last_signal']
d = pd.Timestamp(ls['d']); t = X.IDX.get_loc(d)
U = ls['U']; b = np.array(ls['b']); w = np.array(ls['w_sel']); ind = ls['ind']
px = np.array([L.CFF[t, X.CI[c]] for c in U])
book = []
for cap in (1e6, 1e7):
    rows = []
    for c, wi, bi, p, s in zip(U, w, b, px, ind):
        amt = wi * cap * 0.995; sh = int(amt // p)
        rows.append(dict(code=c, sector=s, w=round(float(wi), 5), b=round(float(bi), 5), price=float(p), amount=round(amt), lots=sh // 1000, odd=sh % 1000))
    book.append(dict(capital=cap, rows=sorted(rows, key=lambda r: -r['w'])))
ST = pd.DataFrame(ls['ST'])
sec_w = pd.DataFrame(dict(sector=ind, w=w, b=b)).groupby('sector').sum()
fin['rulebook'] = dict(signal_day=ls['d'], j=O['last_signal_j'], top_sectors=ls['top'], sector_table=ST[['sector', 'n', 'H52', 'REV3', 'RS', 'NH', 'COMP']].round(4).to_dict('records'),
                       sector_weights=(sec_w.assign(active=lambda z: z.w - z.b) * 100).round(2).reset_index().to_dict('records'), orders=book)
R['FINAL'] = fin
(HERE / 'results.json').write_text(X.dumps(R), encoding='utf-8')
print(json.dumps({k: (v['pass_'] if isinstance(v, dict) and 'pass_' in v else None) for k, v in fin.items() if k.startswith('G')}, ensure_ascii=False))
print(fin['verdict']['label'], fin['verdict']['watch_condition'], fin['gap'])
print('top sectors', ls['top']); print((sec_w.assign(active=lambda z: z.w - z.b) * 100).round(2).sort_values('active').to_string())
print(pd.DataFrame(book[0]['rows']).head(25).to_string())

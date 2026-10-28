#!/usr/bin/env python3
"""步驟 2：迭代（只用挑選期 2019-01～2022-12；所有輸入截到 2023-01 決策日）。
事前網格（寫在跑之前，共 8 版，≤ 12）：
  第 1 輪（x_C＝100%、h＝0）：V1 S1 L3、V2 S1 L6、V3 S1 L12、V4 S1skip L12（12-1）、V5 S2 L6。
  第 2 輪：第 1 輪合格版本中「對同平均權重靜態混合的配對 IR」最高的訊號，配 (x_C 70%, h 0)、(100%, ±2%)、(70%, ±2%)。
  第 3 輪：只在第 2 輪後沒有任何合格版本（轉換 < 2）時才跑；否則不跑。
  挑選準則：配對 IR（年化）；平手（差 < 0.05）取轉換次數少者；轉換 < 2 者不能入選。
主策略腿：AB_V4|normal（navs_v4）；0050：etf_0050.csv。模擬起點 2018-03 決策日（只當起始狀態，2018 視窗不進任何統計）。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cr_lib as L, cr_eval as E

W = pd.read_parquet(L.DATA / 'W_sel.parquet')
END = W['end'].iloc[-1]
mainnav = pd.read_parquet(L.RAW / 'navs_v4.parquet')['AB_V4|normal'].dropna()
mainnav = mainnav[mainnav.index <= END]
e50 = L.e50_series(); e50 = e50[e50.index <= END]
SIM0 = '2018-03-01'

R1 = [dict(id='V1', sig='S1', L=3, xC=1.0, h=0.0), dict(id='V2', sig='S1', L=6, xC=1.0, h=0.0), dict(id='V3', sig='S1', L=12, xC=1.0, h=0.0),
      dict(id='V4', sig='S1skip', L=12, xC=1.0, h=0.0), dict(id='V5', sig='S2', L=6, xC=1.0, h=0.0)]
IT = dict(plan=__doc__, plan_written_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), rounds=[])


def run(spec):
    B = E.build(W, spec, mainnav, e50, SIM0, 'normal', per=L.SEL)
    o = E.evaluate(B, W, mainnav, e50, L.SEL, END, label=spec['id'])
    o['spec'] = spec; o['eligible'] = o['switches'] >= 2
    return o


def brief(o):
    return dict(id=o['spec']['id'], spec=o['spec'], eligible=o['eligible'], switches=o['switches'], wbar=round(o['wbar'], 3),
                IR=round(o['vs_static']['IR_ann'], 3), mean_vs_static_pp=round(o['vs_static']['mean_pp'], 3), t_vs_static=round(o['vs_static']['t'], 2),
                boot_state_p5_p95=[round(o['vs_static']['boot_state']['p5'], 3), round(o['vs_static']['boot_state']['p95'], 3)],
                t_vs_0050=round(o['vs_0050']['t'], 2), mean_vs_0050_pp=round(o['vs_0050']['mean_pp'], 3), CAGR_rot=round(o['CAGR_rot'] * 100, 2),
                CAGR_static=round(o['CAGR_static'] * 100, 2), CAGR_main=round(o['CAGR_main'] * 100, 2), CAGR_0050=round(o['CAGR_0050'] * 100, 2),
                MDD_rot=round(o['MDD_rot'] * 100, 1), whip_ratio=round(o['whipsaw']['ratio_whip_to_gain'], 2))


def pick(res):
    el = [o for o in res if o['eligible']]
    if not el: return None
    el.sort(key=lambda o: -o['vs_static']['IR_ann'])
    best = el[0]
    ties = [o for o in el if best['vs_static']['IR_ann'] - o['vs_static']['IR_ann'] < 0.05]
    ties.sort(key=lambda o: (o['switches'], -o['vs_static']['IR_ann']))
    return ties[0]


res1 = [run(s) for s in R1]
IT['rounds'].append(dict(round=1, results=[brief(o) for o in res1], full=[E.strip(o) for o in res1]))
b1 = pick(res1)
IT['round1_pick'] = b1['spec']['id'] if b1 else None
allres = list(res1)
if b1:
    s0 = b1['spec']
    R2 = [dict(s0, id='V6', xC=0.7, h=0.0), dict(s0, id='V7', xC=1.0, h=0.02), dict(s0, id='V8', xC=0.7, h=0.02)]
    res2 = [run(s) for s in R2]
    IT['rounds'].append(dict(round=2, results=[brief(o) for o in res2], full=[E.strip(o) for o in res2]))
    allres += res2
final = pick(allres)
IT['final_pick'] = final['spec'] if final else None
IT['round3'] = '未執行（第 2 輪後已有合格版本）' if final else '需要'
IT['n_versions'] = len(allres)
IT['P1_note'] = '見 explore.json S1.P1_selection：挑選期 d 的過去累積對下月 d 的係數全部為負（L6 t_HAC −2.0、L12 −1.7），與 P1 預測方向相反。'
L.save_json(IT, L.HERE / 'iterations.json')
for o in allres: print(brief(o))
print('FINAL', IT['final_pick'])

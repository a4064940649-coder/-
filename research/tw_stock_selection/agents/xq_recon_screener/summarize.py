#!/usr/bin/env python3
"""把 diag.json、diag_ky.json 與對帳結論併進 results.json 的 summary。python3 -I -B summarize.py（先跑 recon.py、diag.py、diag_ky.py）。"""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
R = json.loads((H / 'results.json').read_text())
XQ = dict(n=874, win=0.4680, n_win=409, n_loss=465, mean=0.0261, twr=22.8005, max=1.1836, min=-0.4164, max_consec_win=0.3012, max_consec_loss=-0.2760,
          max_runup=51.4948, maxdd=-0.4888)
fp = lambda s, f: R['scen'][s]['fees'][f]
R['summary'] = dict(
    xq=XQ,
    fingerprint=dict(
        xq_max_trade='XQ 最大單筆 +118.36% ＝ 研究中心 2492 華新科 2018-04-12 買進、H21（買進那根算第 1 根、第 22 根開盤賣）、單邊 0.6% 的 +118.38%；'
                     '0.2% 是 +120.13%、不扣費 +121.01%、H22 是 +121.45%（且 H22 另有 1325 恆大 +126.16% 超過 XQ 最大值）',
        all_scope_out_of_range='上市＋上櫃 H21 0.6% 有 3 筆超過 +118.36%（3064 2016-02 +131.8%、2743 2023-05 +120.0%、6217 2026-02 +148.5%）與 1 筆低於 −41.64%（6150 2018-04 −55.5%），只上市沒有',
        twr_is_sum='XQ 時間加權報酬 2280.05% ÷ 874 筆 ＝ 2.6088%，四捨五入正是平均報酬率 2.61%：XQ 的時間加權報酬＝每筆報酬依持有期數加權的加總，持有期全部 21 期時就等於「筆數 × 平均報酬」'),
    most_likely='只上市（XQ「上市」或「普通股全部」裡實際只有上市股進場）、還原值（以最大單筆無法區分還原／原始）、最大持有 21 期＝open-to-open 21 根、單邊 0.6%',
    research_TW_f06=dict(n=fp('TW_adj_H21_strict', 'f06')['n'], n_open=fp('TW_adj_H21_strict', 'f06')['n_open_end'], win=fp('TW_adj_H21_strict', 'f06')['win'],
                         mean=fp('TW_adj_H21_strict', 'f06')['mean'], sum=fp('TW_adj_H21_strict', 'f06')['sum'], max=fp('TW_adj_H21_strict', 'f06')['max'],
                         min=fp('TW_adj_H21_strict', 'f06')['min'], curve_a=fp('TW_adj_H21_strict', 'f06')['curve_a']),
    diag=json.loads((H / 'diag.json').read_text()),
    diag_ky=json.loads((H / 'diag_ky.json').read_text()),
)
(H / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
print('ok')

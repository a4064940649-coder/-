# -*- coding: utf-8 -*-
"""反方審查（資料與統計）o20：只讀研究員輸出，不改其檔案。python3 -I -B review_stat_o20.py
1) 對齊實驗逐 j 拆解：每一步讀法改變（V0->V1 資料 vrank、V1->V2 E23、V2->V3 停牌順延）逐 j 影響的符號與大小
2) 紙上 M1->M2：逐窗差、集中度、拿掉最大貢獻窗後的差
3) PREREG P2 門檻檢核
"""
import json, numpy as np, pandas as pd
D = '/home/user/-/research/tw_stock_selection/agents/o_g6_allpit_list_trace_stage2/'
B = json.load(open(D + 'out/b9_align.json'))
out = {}
print('== 1) 對齊步驟逐 j 影響（exc，pp；正值＝改成原引擎讀法讓複製員變好）')
for cell in ['N1000', 'S100', 'O30']:
    P = {v: np.array(B[f'{cell}|{v}']['exc']['perj']) for v in ['V0', 'V1', 'V2', 'V3']}
    med = {v: B[f'{cell}|{v}']['exc']['diff_of_med'] for v in P}
    r = {}
    for a, b, nm in [('V0', 'V1', 'c_vrank'), ('V1', 'V2', 'b_E23'), ('V2', 'V3', 'a_suspdefer'), ('V0', 'V3', 'all')]:
        dj = P[b] - P[a]
        r[nm] = dict(med_step=round(med[b] - med[a], 4), perj_mean=round(dj.mean(), 4), perj_median=round(float(np.median(dj)), 4),
                     n_pos=int((dj > 1e-3).sum()), n_neg=int((dj < -1e-3).sum()), perj=[round(x, 4) for x in dj])
        print(cell, nm, r[nm])
    r['V3_residual_perj_maxabs'] = round(float(np.abs(P['V3']).max()), 4)
    out[cell] = r

print('\n== 2) 紙上 M1->M2 逐窗差（opt normal）')
Z = np.load(D + 'out/k_paper_windows.npz')
for v in ['ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry']:
    for c in ['normal', 'stress']:
        r1 = Z[f'M1|{v}|{c}|r']; r2 = Z[f'M2|{v}|{c}|r']; r50 = Z[f'M2|{v}|{c}|r50']; per = Z[f'M2|{v}|{c}|per']
        n = len(r1); d = r2 - r1; nz = np.where(np.abs(d) > 1e-12)[0]
        ann = lambda r: (np.prod(1 + r) ** (12 / n) - 1) * 100
        tt = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        base = ann(r2) - ann(r1)
        # 拿掉 |貢獻| 最大的一窗（用 M1 值取代 M2）
        o = nz[np.argsort(-np.abs(d[nz]))]
        r2b = r2.copy(); r2b[o[0]] = r1[o[0]]
        rec = dict(n=n, n_diff_windows=len(nz), ann_diff=round(base, 4), t_M1=round(tt(r1 - r50), 4), t_M2=round(tt(r2 - r50), 4),
                   top=[(str(per[i]), round(d[i] * 100, 3)) for i in o[:6]], ann_diff_drop_top1=round(ann(r2b) - ann(r1), 4))
        out[f'paper|{v}|{c}'] = rec
        print(v, c, rec)

print('\n== 3) P2 門檻：opt 22.06 vs M2')
S = json.load(open(D + 'out/p2_paper_compare.json'))['stats']
m1 = S['M1|ALLPIT_REAL_opt|normal']; m2 = S['M2|ALLPIT_REAL_opt|normal']
p2 = dict(ann_diff=round(m2['ann_ia'] - m1['ann_ia'], 4), t_diff=round(m2['t'] - m1['t'], 4), pass_prereg=bool(abs(m2['ann_ia'] - 22.06) <= 0.1 and abs(m2['t'] - 1.59) <= 0.05))
for v in ['p90', 'carry']:
    p2[v] = round(S[f'M2|ALLPIT_REAL_{v}|normal']['ann_ia'] - S[f'M1|ALLPIT_REAL_{v}|normal']['ann_ia'], 4)
out['P2'] = p2; print(p2)
json.dump(out, open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_stat_o20_out.json', 'w'), ensure_ascii=False, indent=1)

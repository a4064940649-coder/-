#!/usr/bin/env python3
"""main_picks_today.py（現行，只執行不修改；第 1 個參數可指定 scratchpad 的 v2 patch 複本、第 2 個參數為輸出檔名）與 v2（gapfix、正式次排序、跳動點漲停）逐月名單比較。
最近 13 個訊號日（2025-10～2026-10）× n10／n30 × 全市場／--ex-fin。工具的 --held＝v2 前一期持股（讓「已持有不擋漲停」一致）。
輸出 results_tool_compare.json。"""
import sys, json, subprocess, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v2_core as V
K = V.K
TOOL = Path(sys.argv[1]) if len(sys.argv) > 1 else V.HERE.parents[1] / 'main_picks_today.py'
OUTF = sys.argv[2] if len(sys.argv) > 2 else 'results_tool_compare.json'
DATA = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data'
T0 = time.time()
YL, Y5, Y5f = V.load_revenue()

def parse(out):
    lines = out.splitlines(); picks = []; ranks = {}
    for i, l in enumerate(lines):
        if '實際買進名單' in l or '【前 ' in l:
            for m in lines[i + 2:]:
                f = m.split()
                if not f or not f[0].isdigit(): break
                picks.append(f[1])
            break
    for l in lines:
        f = l.split()
        if len(f) >= 7 and f[0].isdigit() and len(f[1]) == 4:
            try: ranks[f[1]] = (int(f[0]), float(f[-2]) if False else None)
            except ValueError: pass
    skipped = [l for l in lines if '漲停' in l and '未買' not in l and l.strip() and l.split()[0].isdigit()]
    return picks, skipped

res = {}
for univ in ('ALL', 'EXFIN'):
    S = V.V2('gapfix', YL, Y5f, latest=True, exfin=(univ == 'EXFIN'))
    Ss = V.V2('strict', YL, Y5f, latest=True, exfin=(univ == 'EXFIN'))
    for n in (10, 30):
        pl = S.plan(n, 'official'); pls = Ss.plan(n, 'code')
        bysig = {p[3]: p for p in pl}; bysig_s = {p[3]: p for p in pls}
        sigs = [d for d in S.sig_days if d >= pd.Timestamp('2025-10-01')]
        prev = {p[3]: q for p, q in zip(pl[1:], pl[:-1])}
        for d in sigs:
            held = [K.COLS[j] for j in prev[d][1]] if d in prev else []
            cmd = ['python3', '-I', '-B', str(TOOL), '--data', DATA, '--date', str(d.date()), '--n', str(n), '--held', ','.join(held)]
            if univ == 'EXFIN': cmd.append('--ex-fin')
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
            tp, sk = parse(out)
            vp = [K.COLS[j] for j in bysig[d][1]]
            sp = [K.COLS[j] for j in bysig_s[d][1]] if d in bysig_s else None
            cs = S.cross_section(d)
            od = list(S.order(cs, 'official')[:int(cs['el'].sum())])
            def info(c):
                j = K.CI[c]
                return {'v2_rank': (od.index(j) + 1) if j in od else None, 'score': None if not np.isfinite(cs['sc'][j]) else round(float(cs['sc'][j]), 6),
                        's52_gapfix': None if not np.isfinite(cs['s52'][j]) else round(float(cs['s52'][j]), 4), 'y3': None if not np.isfinite(cs['y3'][j]) else round(float(cs['y3'][j]), 2),
                        'el': bool(cs['el'][j]), 's52_strict': None if not np.isfinite(Ss.S52[cs['t'], j]) else round(float(Ss.S52[cs['t'], j]), 4),
                        'gap_up_tick': bool(K.gap_up_open(cs['e'])[j])}
            only_v2 = [c for c in vp if c not in tp]; only_tool = [c for c in tp if c not in vp]
            key = f'{univ}|n{n}|{d.date()}'
            res[key] = {'v2': vp, 'tool': tp, 'same_set': set(vp) == set(tp), 'only_v2': {c: info(c) for c in only_v2}, 'only_tool': {c: info(c) for c in only_tool if c in K.CI},
                        'tool_skipped_lines': sk, 'v2_strict_codetie_same_as_tool': (set(sp) == set(tp)) if sp else None, 'pool_v2': int(cs['el'].sum())}
            print(f'{time.time() - T0:5.0f}s', key, 'same' if res[key]['same_set'] else f'DIFF v2+{only_v2} tool+{only_tool}', flush=True)
(V.HERE / OUTF).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('done')

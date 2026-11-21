#!/usr/bin/env python3
"""S89 掛鉤機械測試（登記前；只印機械性質，不印任何報酬／淨值）：T5_5、T5_5h、NM10 各跑標準路徑一次，檢查
(1) 每個觸發事件的 g 符合門檻且發生在進場列＋n−1；(2) 全賣格：被標記的部位在次一個可賣列出場（why＝tstop）或被現行出場升級；
(3) 減碼格：每個事件要嘛成交約一半、要嘛因不足一張取消；之後該筆交易仍照現行規則出場；(4) 淨值全部有限且為正。"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s89lib as L
P, U, RT = L.load()
S = L.signal(P, U, RT, 0)
out = {}
for c in ('T5_5', 'T5_5h', 'NM10'):
    cell = L.CELLS[c]
    nav, lg, st = L.sim(P, S, U, cell=c)
    ev = st.get('ts_events', []); fl = st.get('ts_fills', [])
    ok1 = all((t - e == cell['n'] - 1) and ((g < 1) if cell['strict'] else (g <= 1 - cell['x'])) and abs(g - L.gpath(P, j, e, t)) < 1e-9 for t, j, e, g, h in ev)
    key = {(int(r.j), int(r.ent_t)): r for r in lg.itertuples()}
    ok2 = True; n_next = 0
    if not cell['half']:
        for t, j, e, g, h in ev:
            r = key[(j, e)]
            ds = L.sell_open(P, j, t + 1)
            good = (r.why == 'tstop' and r.exit_t >= ds and (r.exit_t == ds or P.AMT[ds, j] * 0.05 < r.shares * P.O[ds, j])) or (ds is None and r.why == 'open_end') or r.why in ('rule', 'delist')
            ok2 = ok2 and good; n_next += int(r.why == 'tstop' and r.exit_t == t + 1)
    ok3 = True
    if cell['half']:
        fk = {(j, e): (t, fr) for t, j, e, fr, _ in fl}
        for t, j, e, g, h in ev:
            if (j, e) in fk:
                ok3 = ok3 and fk[(j, e)][0] == L.sell_open(P, j, t + 1) and 0.3 <= fk[(j, e)][1] <= 0.51
            ok3 = ok3 and key[(j, e)].why in ('rule', 'open_end', 'delist')
    out[c] = dict(n_events=len(ev), n_tstop_trades=int((lg['why'] == 'tstop').sum()), n_sold_next_day=n_next, half_sold=int(st.get('half_sold', 0)), half_skip_lot=int(st.get('half_skip_lot', 0)),
                  n_trades=int(len(lg)), ok_trigger=bool(ok1), ok_full_exit=bool(ok2), ok_half=bool(ok3), nav_ok=bool(np.isfinite(nav.to_numpy()).all() and (nav > 0).all()))
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(HERE / 'out' / 'hooktest.json', 'w'), ensure_ascii=False, indent=1)
assert all(v['ok_trigger'] and v['ok_full_exit'] and v['ok_half'] and v['nav_ok'] for v in out.values())

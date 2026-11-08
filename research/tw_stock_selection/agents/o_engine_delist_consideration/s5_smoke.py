#!/usr/bin/env python3
"""步驟四：煙霧測試（凍結前；不算任何登記變體的報酬或差異，C0 除外＝P1 登記前已知）。
(a) cz_engine 舊三口徑與 ia_core.simulate 逐日差＝0（C0、U_opt、U_opt_F2 × 一般／壓力 × freeze／nofreeze／dzero）；
(b) C0 freeze 與 main_v2/navs_gapfix.parquet 逐日差＝0；
(c) C0 的 consideration（main／lo／hi）與 freeze 逐日差＝0（P1）；
(d) 手算對帳 3 個事件（現金 2325、換股＋現金 3697、股份轉換 2311）：單一股票合成名單，比對引擎結算值與公開對價（差 > 0.5% 斷言失敗）。
python3 -I -B s5_smoke.py   輸出 smoke.json"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cz_lib as Z
import cz_engine as CE
K = CE.K
E = pd.read_csv(HERE / 'events_v2.csv', dtype={'code': str, 'successor': str})
PL = pickle.loads((Z.WORK / 'plans_ia.pkl').read_bytes())
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
out = dict(a={}, b={}, c={}, d={})
for k in ('C0', 'U_opt', 'U_opt_F2'):
    v = PL[k]; plan = [(p[0], p[1]) for p in v['plan']]
    for c, (bb, ss) in COST.items():
        for mode, kw in (('freeze', {}), ('nofreeze', dict(freeze=False)), ('dzero', dict(delist_zero=True))):
            a, _ = K.simulate(plan, bb, ss, v['t0'], v['t1'], **kw)
            b, _ = CE.simulate(plan, bb, ss, v['t0'], v['t1'], mode=mode)
            out['a'][f'{k}|{c}|{mode}'] = float(np.max(np.abs(a.to_numpy() - b.to_numpy())))
assert max(out['a'].values()) == 0.0, out['a']
OFF = pd.read_parquet(Z.AG / 'main_v2' / 'navs_gapfix.parquet')
v = PL['C0']; plan = [(p[0], p[1]) for p in v['plan']]
for c, (bb, ss) in COST.items():
    a, _ = CE.simulate(plan, bb, ss, v['t0'], v['t1'], mode='freeze')
    ref = OFF[f'EXFIN|n10|{c}'].dropna()
    out['b'][c] = float(np.nanmax(np.abs(a.reindex(ref.index).to_numpy() - ref.to_numpy())))
assert max(out['b'].values()) < 1e-12, out['b']
for bd in ('main', 'lo', 'hi'):
    st = CE.build_settlement(E, bd)
    for c, (bb, ss) in COST.items():
        a, _ = CE.simulate(plan, bb, ss, v['t0'], v['t1'], mode='freeze')
        b, lg = CE.simulate(plan, bb, ss, v['t0'], v['t1'], mode='consideration', settle=st)
        out['c'][f'{bd}|{c}'] = dict(maxabs=float(np.max(np.abs(a.to_numpy() - b.to_numpy()))), n_settled=len(lg['settled']))
assert all(x['maxabs'] == 0.0 and x['n_settled'] == 0 for x in out['c'].values()), out['c']

# (d) 手算對帳
st = CE.build_settlement(E, 'main')
bb, ss = COST['normal']
def single(code, buy_before):
    j = K.CI[code]
    rd = K.rebalance_days('2006-01-01')
    rb = [d for d in rd if d <= pd.Timestamp(buy_before)][-1]
    t_b = K.IDX.get_loc(rb)
    later = [K.IDX.get_loc(d) for d in rd if d > rb][:4]
    other = K.CI['2330']
    plan = [(t_b, np.array([j]))] + [(t, np.array([other])) for t in later]
    nav, lg = CE.simulate(plan, bb, ss, t_b, later[-1], mode='consideration', settle=st)
    return j, nav, lg
rows = {}
for code, buy_before, check in (('2325', '2018-04-16', 'cash'), ('3697', '2014-01-14', 'mixed'), ('2311', '2018-04-16', 'conversion')):
    j, nav, lg = single(code, buy_before)
    ev = [e for e in lg['settled'] if e['code'] == code]; assert len(ev) == 1, (code, ev)
    ev = ev[0]; r = E.set_index('code').loc[code]
    lc = float(r.last_close); V = ev['value_lastclose']; sh = V / lc
    parts = ev['parts']; res = dict(type=ev['type'], value_lastclose=V, shares=sh, parts=parts)
    if check == 'cash':
        got = [p for p in parts if p[0] == 'cash'][0][2] / sh
        res['settled_per_share'] = got; res['public'] = float(r.cash); res['err'] = got / float(r.cash) - 1
    else:
        sp = [p for p in parts if p[0] == 'stock'][0]
        sj = K.CI[sp[2]]; t_s = K.IDX.get_loc(pd.Timestamp(sp[1]))
        px = K.On[t_s, sj] if t_s in [x[0] for x in []] else None
        # 換股：結算股數 × 當日價（開盤或收盤，引擎記錄的值）÷ 股數 ÷ 換股比例 → 應等於存續股當日價
        q_val = sp[3]; implied_px = q_val / (sh * float(r.ratio))
        res['succ'] = sp[2]; res['succ_date'] = sp[1]
        res['succ_open'] = float(K.On[t_s, sj]); res['succ_close'] = float(K.Cn[t_s, sj])
        ref = res['succ_open'] if abs(implied_px / res['succ_open'] - 1) < abs(implied_px / res['succ_close'] - 1) else res['succ_close']
        res['implied_succ_px'] = implied_px; res['err'] = implied_px / ref - 1
        if check == 'mixed':
            got = [p for p in parts if p[0] == 'cash'][0][2] / sh; res['cash_err'] = got / float(r.cash) - 1
            assert abs(res['cash_err']) < 0.005
    assert abs(res['err']) < 0.005, (code, res)
    rows[code] = res
out['d'] = rows
(HERE / 'smoke.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, default=str)[:3000])

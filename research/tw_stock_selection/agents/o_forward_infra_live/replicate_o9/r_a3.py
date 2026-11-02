"""A3：V7X 原引擎（V7 面板 x_data）與前瞻引擎（history 截斷資料）同名單比較：官方日、J0、J6 × 兩窗（2026-08、2026-09 錨定）× 一般／壓力／上限 10%。"""
import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live/replicate_o9')
import numpy as np, pandas as pd
import rlib as L, r_engine as RE

ASOF = pd.Timestamp('2026-11-02')
Dm, M = RE.build_D(ASOF); XF = RE.load_engine(Dm)
data = M['data']
sys.modules.pop('x_data')
sys.path.insert(0, str(RE.V7X))
import x_data as DO  # 原 V7 面板資料層（讀既有快取）
XO = L.load_mod('x_exec_orig', RE.V7X / 'x_exec.py')
assert XO.D is DO and XF.D is Dm


def sched(line, y, m):
    return L.official_day(y, m) if line == 'OFF' else L.j_days(y, m)[int(line[1:])]


def lst(line, sd, j):
    if line == 'OFF': return L.official_list(sd)[0]
    return L.j_list(sd, j)[0]


rows = []
for (ya, ma), (yb, mb) in [((2026, 8), (2026, 9)), ((2026, 9), (2026, 10))]:
    for line in ['OFF', 'J0', 'J6']:
        sd0, sd1 = sched(line, ya, ma), sched(line, yb, mb)
        order = lst(line, sd0, 0 if line == 'OFF' else int(line[1:]))
        ex = L.MP.execute(data, order, sd0, 10, set())
        for nm, cfgx, cap in [('normal', {}, None), ('stress', dict(cost='stress'), None), ('cap10', {}, 0.10)]:
            r = {}
            for tag, X, D in [('fwd', XF, Dm), ('orig', XO, DO)]:
                pk = np.array([D.CI[c] for c in ex['picks']])
                i0, i1, ie = D.TPOS[sd0], D.TPOS[sd1], D.TPOS[ex['next_day']]
                old = X.CAP
                if cap: X.CAP = cap
                cf = dict(RE.BASE_CFG, **cfgx)
                if tag == 'orig': cf['tier'] = 1e6 / DO.cpi_scale(2006)   # 原引擎起始現金＝tier×cpi(2006)，換算成同為 100 萬
                s = X.Sim([(ie, pk, i0)], cf); nav = s.run(t1=i1); X.CAP = old
                r[tag] = 100 * (nav.iloc[-1] - 1)
            rows.append(dict(anchor=f'{ya}-{ma:02d}', line=line, cfg=nm, sd0=str(sd0.date()), sd1=str(sd1.date()),
                             fwd=r['fwd'], orig=r['orig'], diff_pp=r['fwd'] - r['orig']))
            print(rows[-1])
df = pd.DataFrame(rows)
print('cells', len(df), 'max |diff| pp', df.diff_pp.abs().max())
(L.HERE / 'out_a3.json').write_text(json.dumps(dict(rows=rows, max_abs_diff_pp=float(df.diff_pp.abs().max())), indent=1))

#!/usr/bin/env python3
"""A3：同一份名單（重播凍結的官方日、J0、J6 2026-09／10 窗），V7X 原引擎（V7 凍結面板，至 2026-10-22）對 前瞻引擎（xlive，history.csv.gz 截斷到 10-22），
每窗年化前報酬差 ≤ 0.10pp。兩邊都是 GROW、起始現金 100 萬、V7X 正式設定（F1～F4、K5T、rank_right、stress_odd_extra）。
另報：壓力成本、參與上限 10%；以及舊版 K5T-lite（o_forward_sequential_v2 的 cost_side）在同名單上的同窗報酬（o8 指出的偏樂觀量）。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(AG.parent))
import fwd_data as FD, xlive as XL  # noqa
import main_picks_today as MP  # noqa
sys.path.insert(0, str(XL.V7X))
import x_data as XD  # noqa: E402（原 V7X 資料層，只讀）
OUT = HERE / 'replay/seq'
END = pd.Timestamp('2026-10-22')


def plans(line):
    L = []
    for m in ('202609', '202610'):
        f = OUT / 'paper_lists' / f'{line}_{m}.exec.json'
        ex = json.loads(f.read_text())
        L.append((pd.Timestamp(ex['signal']), pd.Timestamp(ex['exec']), ex['picks']))
    return L


def windows(nav, L):
    res = []
    for k, (sd, nd, pk) in enumerate(L):
        end = L[k + 1][0] if k + 1 < len(L) else END
        s = nav.dropna()
        a = s[s.index <= sd].iloc[-1]; b = s[s.index <= end].iloc[-1]
        res.append((str(sd.date()), str(end.date()), float(b / a - 1)))
    return res


def run(XE, D, L, var, tier):
    pl = [(D.TPOS[nd], np.array([D.CI[c] for c in pk], int), D.TPOS[sd]) for sd, nd, pk in L]
    cfg = dict(XL.CFG_V7X, tier=tier, cost='stress' if var == 'stress' else 'normal')
    cap0 = XE.CAP
    if var == 'cap10': XE.CAP = 0.10
    try:
        s = XE.Sim(pl, cfg); nav = s.run(t1=int(np.searchsorted(D.DATES, np.datetime64(END), side='right') - 1))
    finally:
        XE.CAP = cap0
    return nav, s.acc


def lite(Ld, L, data):
    """o_forward_sequential_v2 的 K5T-lite（每邊 0.1425%×0.28＋max(0.1%, 一跳/開盤)；賣出加 0.3%），等權、開盤換股。"""
    sys.path.insert(0, str(AG / 'o_forward_sequential_v2'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('mf_old', AG / 'o_forward_sequential_v2/make_forward.py')
    mf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mf)
    px = dict(R=data['R'], O=data['O'], C=data['C'], ADJ=data['ADJ'], cal=data['cal'][data['cal'] <= END])
    nav, _ = mf.run_line([(sd, nd, pk) for sd, nd, pk in L], px, L[0][0], END)
    return nav


def main():
    XE0 = XL.load_engine(XD, 'x_exec_orig')
    src = FD.Source()
    adir = HERE / 'work' / f'a3_asof_{END:%Y%m%d}'
    src.build_asof(END, adir)
    FD.assert_dir_asof(adir, END)
    pnl = src.panel_pre(HERE / 'work')
    data = MP.load_prices(adir, END, pnl)
    dsp = MP.load_disposal(adir)
    e50 = pd.read_csv(adir / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
    import shutil; shutil.rmtree(adir)
    dm = XL.build_module(data, dsp, e50, start=pd.Timestamp('2026-03-01'))
    XE1 = XL.load_engine(dm, 'x_exec_live_a3')
    rows = []
    for line in ('official', 'j0', 'j6'):
        L = plans(line)
        miss = [c for _, _, pk in L for c in pk if c not in XD.CI or c not in dm.CI]
        assert not miss, miss
        lw = windows(lite(None, L, data), L)
        for var in ('normal', 'stress', 'cap10'):
            n0, a0 = run(XE0, XD, L, var, 1_000_000 / XD.cpi_scale(2006))
            n1, a1 = run(XE1, dm, L, var, 1_000_000)
            w0, w1 = windows(n0, L), windows(n1, L)
            for k in range(len(L)):
                rows.append(dict(line=line, var=var, start=w0[k][0], end=w0[k][1], v7x_ret=w0[k][2], live_ret=w1[k][2],
                                 diff_pp=(w1[k][2] - w0[k][2]) * 100, lite_ret=lw[k][2] if var == 'normal' else np.nan,
                                 lite_minus_v7x_pp=(lw[k][2] - w0[k][2]) * 100 if var == 'normal' else np.nan,
                                 v7x_n_lim_block=a0['n_lim_block'], live_n_lim_block=a1['n_lim_block'],
                                 v7x_n_disp_sells=a0['n_disp_sells'], live_n_disp_sells=a1['n_disp_sells'],
                                 v7x_n_cap_hit=a0['n_cap_hit'], live_n_cap_hit=a1['n_cap_hit']))
    df = pd.DataFrame(rows)
    FD.write_csv(df, HERE / 'out' / 'a3_v7x_compare.csv', gz=False)
    main_ = df[df['var'] == 'normal']
    res = dict(max_abs_diff_pp_normal=float(main_.diff_pp.abs().max()), max_abs_diff_pp_all=float(df.diff_pp.abs().max()),
               pass_normal=bool(main_.diff_pp.abs().max() <= 0.10), pass_all=bool(df.diff_pp.abs().max() <= 0.10),
               lite_minus_v7x_pp=main_.lite_minus_v7x_pp.round(4).tolist(), rows=len(df))
    print(json.dumps(res, ensure_ascii=False))
    (HERE / 'out' / 'a3_summary.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

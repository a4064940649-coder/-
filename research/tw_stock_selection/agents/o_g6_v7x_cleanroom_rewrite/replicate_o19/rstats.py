# -*- coding: utf-8 -*-
"""統計：20 seeds 日報酬平均 → 一條淨值；年化、t、2019 起、三段、p10。
python3 -I -B rstats.py CELL [mode]
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rengine as RE

SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o19'
ENDS = json.load(open(RE.AG + '/o_v7_listing_exec_engine/data/jscan_end_days.json'))


def first_on_or_after(dates, s):
    return int(np.searchsorted(dates, np.datetime64(s), 'left'))


def last_before(dates, s):
    return int(np.searchsorted(dates, np.datetime64(s), 'left')) - 1


def ann(nav, a, b, dates):
    days = (dates[b] - dates[a]).astype(int)
    return (nav[b] / nav[a]) ** (365.25 / days) - 1


def cell_stats(npz, dates, E, plans, y19base='2019-01-01'):
    out = {}
    z = np.load(npz)
    js = sorted(int(k[4:]) for k in z.files if k.startswith('eq_j'))
    for j in js:
        eqs = z[f'eq_j{j}']; s0 = int(z[f's0_j{j}'])
        end = RE_dpos(dates, ENDS[f'AB_V7_B10|J{j}'])
        r = eqs[:, s0 + 1:] / eqs[:, s0:-1] - 1
        rm = r.mean(0)
        nav = np.full(len(dates), np.nan); nav[s0] = 1.0
        nav[s0 + 1:] = np.cumprod(1 + rm)
        sig = [RE_dpos(dates, s) for s in plans[j]]
        wins = list(zip(sig, sig[1:] + [end]))
        wins = [(a, b) for a, b in wins if a < end]
        ex = np.array([(nav[b] / nav[a] - 1) - (E[b] / E[a] - 1) for a, b in wins])
        t = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))
        y0 = first_on_or_after(dates, y19base)
        ex19 = np.array([(nav[b] / nav[a] - 1) - (E[b] / E[a] - 1) for a, b in wins if dates[a] >= np.datetime64('2019-01-01')])
        t19 = ex19.mean() / (ex19.std(ddof=1) / np.sqrt(len(ex19)))
        e12 = last_before(dates, '2013-01-01'); b13 = first_on_or_after(dates, '2013-01-01'); e18 = last_before(dates, '2019-01-01')
        A = ann(nav, s0, end, dates); A5 = ann(E, s0, end, dates)
        Y = ann(nav, y0, end, dates); Y5 = ann(E, y0, end, dates)
        mdd = np.min(nav[s0:end + 1] / np.maximum.accumulate(nav[s0:end + 1]) - 1)
        out[j] = dict(ann=A, ann0050=A5, exc=A - A5, t=t, n_win=len(ex), ann_y19=Y, ann0050_y19=Y5, exc_y19=Y - Y5, t_y19=t19,
                      exc_s1=ann(nav, s0, e12, dates) - ann(E, s0, e12, dates),
                      exc_s2=ann(nav, b13, e18, dates) - ann(E, b13, e18, dates),
                      mdd=mdd, mean_ex=ex.mean())
    keys = list(out[js[0]].keys())
    summ = {k + '_med': float(np.median([out[j][k] for j in js])) for k in keys}
    summ['exc_p10'] = float(np.percentile([out[j]['exc'] for j in js], 10))
    summ['exc_y19_p10'] = float(np.percentile([out[j]['exc_y19'] for j in js], 10))
    return summ, out


_dp = {}
def RE_dpos(dates, s):
    return int(np.searchsorted(dates, np.datetime64(s), 'left'))


def load_plans():
    p = pd.read_csv(RE.AG + '/o_v7_listing_exec_engine/data/plans_j7.csv', dtype={'picks': str})
    p = p[(p.strat == 'AB_V7_B10') & (p.seed == 0)]
    return {j: sorted(g.signal.tolist()) for j, g in p.groupby('j')}


if __name__ == '__main__':
    cell = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else 'strict'
    tag = os.environ.get('RTAG', '')
    z = np.load(f'{SCR}/rdata_{mode}.npz')
    dates = z['dates'].astype('datetime64[D]'); E = z['E'].astype(float)
    plans = load_plans()
    s, per = cell_stats(f'{SCR}/navs_{cell}_{mode}{tag}.npz', dates, E, plans)
    for base in ['2019-01-01']:
        pass
    print(json.dumps({k: round(v * (100 if not k.startswith('t') else 1), 3) for k, v in s.items()}, ensure_ascii=False))
    json.dump(dict(summary=s, per_j={str(k): v for k, v in per.items()}), open(f'{SCR}/stats_{cell}_{mode}{tag}.json', 'w'), default=float, indent=1)

# 批次：多個名單（j×seed）× 設定 → 每 j 的 seed 平均淨值與統計
import numpy as np, pandas as pd, json, os, sys, time, pickle
from multiprocessing import Pool
import xdata, xengine, xstats
_W = None
def _init():
    global _W
    _W = xengine.World()
def _one(args):
    key, plan, end, cfg = args
    sim = xengine.Sim(_W, plan, end, cfg)
    nav = sim.run()
    extra = None
    if cfg.get('record'):
        extra = {'pnl': sim.pnl, 'eq_open': sim.eq_open, 'n_trades': len(sim.trades), 'turn': float(sim.turn.sum())}
    return key, nav, extra

def end_for(W, plan_sig_dates_last, j=None):
    return None

def run_grid(jobs, procs=4):
    """jobs: list of (key, plan, end, cfg)；key=(cell, j, seed)"""
    out = {}
    with Pool(procs, initializer=_init) as p:
        for key, nav, extra in p.imap_unordered(_one, jobs, chunksize=2):
            out[key] = (nav, extra)
    return out

def summarize(res, plans_sig, ends, dates, E):
    """res[(cell,j,seed)] → {cell: {'per_j':..., 'sum':...}}"""
    cells = sorted(set(k[0] for k in res))
    o = {}
    for c in cells:
        js = sorted(set(k[1] for k in res if k[0] == c))
        per_j = {}; navs = {}
        for j in js:
            ks = sorted(k for k in res if k[0] == c and k[1] == j)
            nav = xstats.seed_avg([res[k][0] for k in ks])
            navs[j] = nav
            per_j[j] = xstats.metrics(nav, plans_sig[j], ends[j], E, dates)
        o[c] = {'per_j': per_j, 'sum': xstats.jsummary(per_j), 'navs': navs}
    return o

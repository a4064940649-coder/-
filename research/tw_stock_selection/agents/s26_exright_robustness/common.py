#!/usr/bin/env python3
"""S26 共用：匯入各題 lib（不改它們），把所有快取改到 scratchpad/s26/cache，並在 load 之前設定 cslib.EXRIGHT_FIX。
用法：import common; common.setup(fix=True); 之後 import 各題 lib，再呼叫 common.redirect() 把快取路徑全部改掉。
一律 python3 -I -B。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s26')
CACHE = SCR / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
T0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def addpath(folder):
    p = str(AG / folder)
    if p not in sys.path:
        sys.path.insert(0, p)


def redirect():
    """所有已載入模組裡指向舊 scratchpad 的模組層級 Path 變數，一律改成 CACHE（只改記憶體中的變數，不改檔案）。"""
    n = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or not hasattr(mod, '__file__') or not mod.__file__ or str(AG) not in str(mod.__file__):
            continue
        for k, v in list(vars(mod).items()):
            if isinstance(v, Path) and 'scratchpad' in str(v) and k not in ('SCR',):
                setattr(mod, k, CACHE)
                n += 1
            elif isinstance(v, Path) and k == 'SCR' and 'scratchpad' in str(v):
                setattr(mod, k, SCR)
    return n


def set_fix(fix):
    addpath('o_ext_canslim_xs')
    import cslib
    cslib.EXRIGHT_FIX = bool(fix)
    return cslib


def log(*a):
    print(f'[{time.time()-T0:.0f}s {rss()}MB]', *a, flush=True)


def dump(name, obj):
    json.dump(obj, open(OUT / name, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def med(x):
    return float(np.median(np.asarray(list(x), float)))

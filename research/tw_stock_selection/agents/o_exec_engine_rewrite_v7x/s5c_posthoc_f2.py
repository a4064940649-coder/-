#!/usr/bin/env python3
"""事後（PREREG 之後）檢查：F2 第二部分「處置賣出款隔日才可用」的實作偏差。
凍結版（R2～R4）在換股求解時把處置賣出款當 0，結果所有入選股的目標金額一起縮小，處置賣出款隔天入帳後沒有任何委託會用到它，
現金閒置到下一個換股日（約一個月）。這比規格文字「款項隔日才可用」更嚴：真實交易者會在隔天用這筆錢補買。
本腳本不改 x_exec.py，只用子類別把求解改回「含處置賣出款」，開盤時現金不足的買單照原規則整筆順延、隔天以入帳現金補買。
Runs：R2c＝R1＋F2(上限＋款項隔日可用、隔日補買)；R4c＝R2c＋F3(0.5)＋F4。標為「事後修正」，不取代凍結數字。"""
import sys, json, inspect, textwrap
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x_data as D, x_exec as X, x_run as XR

src = textwrap.dedent(inspect.getsource(X.Sim.rebalance))
key = "if c.get('f2') and dmask[i] and d < 0: continue"
assert src.count(key) == 1
src = src.replace(key, "pass  # 事後修正：求解含處置賣出款；款項隔日入帳，開盤現金不足的買單照規則順延")
ns = {}
exec(src, X.__dict__, ns)


class SimF2c(X.Sim):
    pass


SimF2c.rebalance = ns['rebalance']
X.SimF2c = SimF2c
_orig = X.run_one


def run_one_c(pl, cfg):
    if cfg.get('f2c'):
        s = SimF2c(pl, cfg); nav = s.run()
        pa = np.array(s.part) if s.part else np.array([np.nan])
        s.acc['part_median'] = float(np.nanmedian(pa)); s.acc['part_p90'] = float(np.nanpercentile(pa, 90)) if np.isfinite(pa).any() else float('nan')
        s.acc['part_gt10'] = float(np.mean(pa > 0.10)) if np.isfinite(pa).any() else float('nan'); s.acc['n_part'] = int(np.isfinite(pa).sum())
        s.acc['part_mean'] = float(np.nanmean(pa))
        return nav, s.acc
    return _orig(pl, cfg)


X.run_one = run_one_c

if __name__ == '__main__':
    R = json.loads((HERE / 'results.json').read_text()); base = dict(R['PREREG']['orig_cfg'])
    for name, ex in (('R2c', dict(f1=True, f2=True, f2c=True)), ('R4c', dict(f1=True, f2=True, f2c=True, f3=0.5, f4=True))):
        if (HERE / 'out' / f'navs_oos_{name}.parquet').exists(): print('skip', name); continue
        cfg = dict(base); cfg.update(ex)
        XR.run_grid('oos_' + name, js=range(13), seeds=range(20), extra=cfg)

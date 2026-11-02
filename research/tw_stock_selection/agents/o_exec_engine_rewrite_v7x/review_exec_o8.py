#!/usr/bin/env python3
"""反方審查（成本與可成交角度）o8：不改研究員檔案，只 import 其引擎做子類別反事實。
A. 重現：R4 正式設定逐 (j,seed) 重跑，seed 平均淨值對 navs_oos_R4.parquet。
B. 反事實 1（ODDLIM）：2020-10-26 前盤後零股以收盤成交，收盤漲停（買）/跌停（賣）時當日不成交、順延。
   2020-10-26 後盤中零股：收盤或開盤鎖漲跌停時同樣不成交（開盤已由 lim_block 擋，這裡再加收盤鎖死）。
C. 反事實 2（PRECOL）：買進處置股須預收款：只能用當日開始前的現金（同日賣出款不可用），不足部分順延。
D. 統計：處置股買進占比、收盤漲停零股買占比、每年需下單日數、手續費／稅占權益。
執行：python3 -I -B review_exec_o8.py"""
import sys, json, math, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x_data as D, x_exec as X, x_stats as S

R4CFG = dict(stress_odd_extra=True, rank_right=True, f1=True, f2=True, f3=0.5, f4=True, model='K5T', cost='normal')
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
CELLS = [('30萬', 'FIX'), ('30萬', 'GROW'), ('100萬', 'FIX'), ('300萬', 'GROW'), ('1000萬', 'FIX'), ('1000萬', 'GROW')]


class Rv(X.Sim):
    def __init__(self, pl, cfg):
        super().__init__(pl, cfg)
        self.log = dict(odd_buy_v=0.0, odd_buy_lockup_v=0.0, odd_sell_v=0.0, odd_sell_lockdn_v=0.0,
                        buy_v=0.0, disp_buy_v=0.0, n_disp_buy=0, precol_short_v=0.0, order_days=set(), n_odd_lock=0)
        self.precash = None

    def _locked(self, t, j, side):
        c = D.C[t, j]
        if side > 0: return bool(np.isfinite(X.LIMUP[t, j]) and c >= X.LIMUP[t, j] - 1e-6)
        return bool(np.isfinite(X.LIMDN[t, j]) and c <= X.LIMDN[t, j] + 1e-6)

    def fill_odd(self, t, j, q, side, cashlim=None):
        if q <= 0: return 0
        lk = self._locked(t, j, side)
        if self.c.get('oddlim') and lk and (t < X.I_ODD or not self.c.get('oddlim_pre')):
            self.log['n_odd_lock'] += 1; return 0
        if side > 0 and self.c.get('precol') and D.DISP[t, j]:
            px = self.oddpx(t, j); v = q * px * (1 + self.oddc(t, j, 1)) * 1.002
            if self.precash is None or v > self.precash:
                self.log['precol_short_v'] += v; return 0
        got = super().fill_odd(t, j, q, side, cashlim)
        if got:
            v = got * self.oddpx(t, j); self.log['order_days'].add(t)
            if side > 0:
                self.log['odd_buy_v'] += v; self.log['buy_v'] += v
                if lk: self.log['odd_buy_lockup_v'] += v
                if D.DISP[t, j]: self.log['disp_buy_v'] += v; self.log['n_disp_buy'] += 1
                if self.c.get('precol') and D.DISP[t, j]: self.precash -= v * 1.002
            else:
                self.log['odd_sell_v'] += v
                if lk: self.log['odd_sell_lockdn_v'] += v
        return got

    def fill_whole(self, t, j, L, side):
        if L <= 0: return 0
        if side > 0 and self.c.get('precol') and D.DISP[t, j]:
            o = D.O[t, j]; per = 1000.0 * o * 1.012
            Lp = int(max(self.precash or 0.0, 0.0) // per)
            if Lp < L:
                self.log['precol_short_v'] += (L - Lp) * per
                L = Lp
            if L <= 0: return 0
        got = super().fill_whole(t, j, L, side)
        if got:
            v = got * D.O[t, j]; self.log['order_days'].add(t)
            if side > 0:
                self.log['buy_v'] += v
                if D.DISP[t, j]: self.log['disp_buy_v'] += v; self.log['n_disp_buy'] += 1
                if self.c.get('precol') and D.DISP[t, j]: self.precash -= v * 1.012
        return got

    def fill_disp_sell(self, t, j, n):
        got = super().fill_disp_sell(t, j, n)
        if got: self.log['order_days'].add(t)
        return got

    def rebalance(self, t, pk, tr):
        self.precash = max(self.cash, 0.0)
        return super().rebalance(t, pk, tr)

    def deferred(self, t, tr):
        self.precash = max(self.cash, 0.0)
        return super().deferred(t, tr)


def _job(args):
    key, pl = args
    out = {}
    for tn, md in CELLS:
        for var in ('base', 'oddlim', 'precol', 'both'):
            cfg = dict(R4CFG, tier=TIERS[tn], mode=md)
            if var in ('oddlim', 'both'): cfg['oddlim'] = True
            if var in ('precol', 'both'): cfg['precol'] = True
            s = Rv(pl, cfg); nav = s.run()
            lg = dict(s.log); lg['n_order_days'] = len(lg.pop('order_days'))
            lg.update(fee=s.acc['fee'], tax=s.acc['tax'], n_orders=s.acc['n_orders'], deferred_days=s.acc['deferred_days'])
            out[(tn, md, var)] = (nav, lg)
    return key, out


def main():
    t0 = time.time()
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    jobs = sorted(P.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    navs = {}; logs = {}
    with Pool(4) as pool:
        for key, out in pool.imap_unordered(_job, jobs, chunksize=2):
            for ck, (n, lg) in out.items():
                navs.setdefault((key[0],) + ck, []).append(n); logs.setdefault(ck, []).append(lg)
    print('sim done', round(time.time() - t0), 's', flush=True)
    R4 = pd.read_parquet(HERE / 'out' / 'navs_oos_R4.parquet')
    res = {'repro_maxreldiff': 0.0, 'cells': {}}
    for tn, md in CELLS:
        per = {v: [] for v in ('base', 'oddlim', 'precol', 'both')}
        for j in range(13):
            for var in per:
                nav = S.avg_nav(navs[(j, tn, md, var)])
                per[var].append(S.stats(nav, 'AB_V7_B10', j))
                if var == 'base':
                    ref = R4[f'B10|J{j}|K5T|normal|{tn}|{md}'].dropna()
                    a = nav.reindex(ref.index)
                    res['repro_maxreldiff'] = max(res['repro_maxreldiff'], float(np.nanmax(np.abs(a / ref - 1))))
        cell = {}
        for var, L in per.items():
            sm = S.jsumm(L)
            cell[var] = {k: round(sm[k + '_med'], 3) for k in ('CAGR', 'exc', 't_0050', 'exc_Y19on', 'exc_SEL')}
        for var in ('oddlim', 'precol', 'both'):
            d = [p['CAGR'] - b['CAGR'] for p, b in zip(per[var], per['base'])]
            cell['delta_' + var + '_jmed_pp'] = round(float(np.median(d)), 3)
            cell['delta_' + var + '_jmin_pp'] = round(float(np.min(d)), 3)
        lg = logs[(tn, md, 'base')]
        sm = lambda k: float(np.sum([x[k] for x in lg]))
        nseed = len(lg)
        cell['share_disp_buy_of_buys_pct'] = round(100 * sm('disp_buy_v') / sm('buy_v'), 3)
        cell['n_disp_buy_per_run'] = round(sm('n_disp_buy') / nseed, 2)
        cell['share_oddbuy_close_lockup_pct'] = round(100 * sm('odd_buy_lockup_v') / max(sm('odd_buy_v'), 1), 3)
        cell['share_oddsell_close_lockdn_pct'] = round(100 * sm('odd_sell_lockdn_v') / max(sm('odd_sell_v'), 1), 3)
        cell['order_days_per_year'] = round(sm('n_order_days') / nseed / 20.5, 1)
        cell['deferred_days_per_year'] = round(sm('deferred_days') / nseed / 20.5, 1)
        cell['orders_per_year'] = round(sm('n_orders') / nseed / 20.5, 1)
        cell['fee_over_tax'] = round(sm('fee') / sm('tax'), 3)
        lgp = logs[(tn, md, 'precol')]
        cell['precol_short_v_per_run'] = round(float(np.mean([x['precol_short_v'] for x in lgp])), 0)
        res['cells'][f'{tn}|{md}'] = cell
        print(tn, md, json.dumps(cell, ensure_ascii=False), flush=True)
    print('repro_maxreldiff', res['repro_maxreldiff'])
    Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_exec_o8.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))


def _job_pre(args):
    key, pl = args
    out = {}
    for tn, md in CELLS[:3]:
        cfg = dict(R4CFG, tier=TIERS[tn], mode=md, oddlim=True, oddlim_pre=True)
        s = Rv(pl, cfg); out[(tn, md)] = s.run()
    return key, out


def main_pre():
    """E. ODDLIM 只用在 2020-10-26 前（盤後零股以收盤成交）；之後盤中零股已由開盤 lim_block 處理。"""
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    jobs = sorted(P.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    navs = {}
    with Pool(4) as pool:
        for key, out in pool.imap_unordered(_job_pre, jobs, chunksize=2):
            for ck, n in out.items(): navs.setdefault((key[0],) + ck, []).append(n)
    R4 = pd.read_parquet(HERE / 'out' / 'navs_oos_R4.parquet')
    for tn, md in CELLS[:3]:
        pp, bb = [], []
        for j in range(13):
            pp.append(S.stats(S.avg_nav(navs[(j, tn, md)]), 'AB_V7_B10', j))
            bb.append(S.stats(R4[f'B10|J{j}|K5T|normal|{tn}|{md}'], 'AB_V7_B10', j))
        sm = S.jsumm(pp)
        d = {k: round(sm[k + '_med'], 3) for k in ('CAGR', 'exc', 't_0050', 'exc_Y19on', 'exc_SEL', 'CAGR_PREODD', 'CAGR_POSTODD')}
        d['delta_jmed_pp'] = round(float(np.median([p['CAGR'] - b['CAGR'] for p, b in zip(pp, bb)])), 3)
        d['delta_PREODD_jmed_pp'] = round(float(np.median([p['CAGR_PREODD'] - b['CAGR_PREODD'] for p, b in zip(pp, bb)])), 3)
        print('ODDLIM_PRE', tn, md, json.dumps(d, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['pre']: main_pre()
    else: main()

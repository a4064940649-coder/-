#!/usr/bin/env python3
"""main_v2 主回測（一個滾動窗版本一個行程）：python3 -I -B run.py <strict|gapfix|lenient>
輸出 run_<win>.json、navs_<win>.parquet。"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v2_core as V
K = V.K

WIN = sys.argv[1]
assert WIN in ('strict', 'gapfix', 'lenient')
assert json.loads((V.HERE / 'results.json').read_text(encoding='utf-8')).get('PREREG_sha256'), '先執行 prereg.py'
import hashlib
assert hashlib.md5((K.P / 'value.parquet').read_bytes()).hexdigest().startswith('8272e253'), 'value.parquet 不是修正版'

T0 = time.time()
YL, Y5, Y5f = V.load_revenue()
G1, GAPMEAN, NFLAG1 = V.exdiv_override()
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
TIES = ['official'] + [f'rand:{i}' for i in range(30)]
NMC = 40
import os
if os.environ.get('V2_SMOKE'):
    TIES = TIES[:2]; NMC = 1; WIN_OUT = 'smoke_'
else:
    WIN_OUT = ''
res = {'win': WIN, 'exdiv_gap_mean': GAPMEAN, 'n_flag1_days': NFLAG1, 'grid': {}, 'official': {}, 'exdiv': {}, 'latest': {}, 'AB': {}, 'lists_1012': {}}
navs = {}

def r2(o):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}

def segstats(nav, ew, sd, segs, yearly=False):
    return {k: r2(V.seg_stats(nav, ew, a, b, sd, yearly=yearly, ad=ad)) for k, (a, b, ad) in segs.items()}

def log(*a):
    print(f'[{WIN} {time.time() - T0:6.0f}s]', *a, flush=True)

def pick_info(S, pl, d, n):
    """訊號日 d 的前 40 名（分數、52 週比、營收）與實際成交名單。"""
    cs = S.cross_section(d)
    od = S.order(cs, 'official')[:int(cs['el'].sum())]
    gu = K.gap_up_open(cs['e'])
    top = [{'code': K.COLS[j], 'score': round(float(cs['sc'][j]), 6), 's52': round(float(cs['s52'][j]), 4), 'yoy3': round(float(cs['y3'][j]), 2),
            'gap_up_block': bool(gu[j])} for j in od[:40]]
    exe = [p for p in pl if p[3] == d]
    picks = [K.COLS[j] for j in exe[0][1]] if exe else None
    return {'signal': str(d.date()), 'exec': str(K.IDX[cs['e']].date()), 'pool': int(cs['el'].sum()), 'top40': top, 'picks': picks}

for univ in ('ALL', 'EXFIN'):
    exfin = univ == 'EXFIN'
    # ---------- 主格點：同分 31 組 × n × 成本 ----------
    S = V.V2(WIN, YL, Y5, exfin=exfin)
    t0, t1, sd = S.bounds()
    ew = None
    res['grid'][univ] = {}
    for n in (10, 30):
        res['grid'][univ][n] = {c: {} for c in COST}
        for tie in TIES:
            pl = S.plan(n, tie)
            if ew is None:
                ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
                ew_ex, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1, ret_override=G1)
                navs[f'{univ}|EW'] = ew
            for c, (bb, ss) in COST.items():
                nav, lg = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
                off = tie == 'official'
                res['grid'][univ][n][c][tie] = segstats(nav, ew, sd, V.SEGS, yearly=off)
                if off:
                    navs[f'{univ}|n{n}|{c}'] = nav
                    o = res['official'].setdefault(univ, {}).setdefault(n, {})
                    o[c] = {'turnover': lg['turnover'], 'blocked': S.nblk, 'n_frozen': len(lg['frozen']), 'frozen': lg['frozen'][:20],
                            'n_signals': len(pl), 'first_exec': str(K.IDX[pl[0][0]].date()), 'last_exec': str(K.IDX[pl[-1][0]].date())}
                    if c == 'normal':
                        o['contrib'] = {k: V.top_contrib(pl, t1, a, b) for k, (a, b, ad) in V.SEGS.items()}
                        o['fin_slots_share'] = float(np.mean([np.mean([V.FINMASK[j] for j in p[1]]) for p in pl]))
                        # 除權息日近似（策略與等權同時套用）
                        nav_x, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1, ret_override=G1)
                        res['exdiv'].setdefault(univ, {})[n] = segstats(nav_x, ew_ex, sd, V.SEGS)
                        res['lists_1012'].setdefault('hold_' + univ, {})[n] = [K.COLS[j] for j in pl[-1][1]]
                        res['lists_1012'].setdefault('hold_sig_' + univ, str(pl[-1][3].date()))
                        # 不執行 10 月換股、9 月持股抱到最新資料日
                        nav_h, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, len(K.IDX) - 1)
                        ew_h, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, len(K.IDX) - 1)
                        res['latest'].setdefault(univ, {}).setdefault(n, {})['hold_sep'] = segstats(nav_h, ew_h, sd + [K.IDX[-1]], V.SEGS_LATEST)
            if tie == 'official' or tie.endswith(':29'):
                log(univ, n, tie, {k: (v['CAGR'], v['t_0050'], v['t_EW']) for k, v in res['grid'][univ][n]['normal'][tie].items()})
    res['official'][univ]['pool_mean'] = float(np.mean([S.cross_section(d)['el'].sum() for d in S.sig_days]))

    # ---------- 到最新資料日：補 2026-09 營收，執行 10-12 訊號 ----------
    SL_ = V.V2(WIN, YL, Y5f, latest=True, exfin=exfin)
    t0l, t1l, sdl = SL_.bounds()
    d1012 = pd.Timestamp('2026-10-12'); assert d1012 in SL_.sig_days
    for n in (10, 30):
        pl = SL_.plan(n, 'official')
        ewl, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0l, t1l)
        for c, (bb, ss) in COST.items():
            nav, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0l, t1l)
            res['latest'][univ][n][f'exec_oct|{c}'] = segstats(nav, ewl, sdl, V.SEGS_LATEST)
            if c == 'normal': navs[f'{univ}|n{n}|latest'] = nav
        res['lists_1012'].setdefault(univ, {})[n] = pick_info(SL_, pl, d1012, n)
        # 同分隨機下 10-12 名單變動
        alt = []
        for tie in TIES[1:]:
            p2 = SL_.plan(n, tie); alt.append([K.COLS[j] for j in p2[-1][1]])
        res['lists_1012'][univ][n]['rand_tie_lists_distinct'] = len({tuple(sorted(x)) for x in alt})
        log('latest', univ, n, res['latest'][univ][n]['exec_oct|normal']['S3']['CAGR'], res['lists_1012'][univ][n]['picks'])

    # ---------- AB：只用已公告營收（MC 40 次）＋最壞 ----------
    res['AB'][univ] = {}
    for n in (10, 30):
        rets = {c: [] for c in COST}; ewr = []; cg = {c: [] for c in COST}; nold = []
        for seed in range(NMC):
            Sa = V.V2(WIN, YL, Y5, exfin=exfin, unavail=V.make_unavail('mc', seed))
            ta, tb, sda = Sa.bounds()
            pl = Sa.plan(n, 'official')
            ewa, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, ta, tb)
            ewr.append(ewa.pct_change())
            nold.append(np.mean([Sa.cross_section(d)['nold'] for d in Sa.sig_days]))
            for c, (bb, ss) in COST.items():
                nav, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, ta, tb)
                rets[c].append(nav.pct_change())
                cg[c].append({k: V.seg_stats(nav, ewa, a, b, sda, ad=ad)['CAGR'] for k, (a, b, ad) in V.SEGS.items()})
        assert ta == t0 and tb == t1
        ew_m = (1 + pd.concat(ewr, axis=1).mean(axis=1).fillna(0)).cumprod()
        out = {'mean_unavail_in_top300': float(np.mean(nold))}
        for c in COST:
            nav_m = (1 + pd.concat(rets[c], axis=1).mean(axis=1).fillna(0)).cumprod()
            out[c] = segstats(nav_m, ew_m, sd, V.SEGS, yearly=(c == 'normal'))
            out[c + '_mc_p05_p95'] = {k: [round(float(np.percentile([x[k] for x in cg[c]], q)), 2) for q in (5, 50, 95)] for k in V.SEGS}
            if c == 'normal': navs[f'AB_{univ}|n{n}|mc'] = nav_m
        Sw = V.V2(WIN, YL, Y5, exfin=exfin, unavail=V.make_unavail('worst'))
        pl = Sw.plan(n, 'official')
        eww, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
        for c, (bb, ss) in COST.items():
            nav, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
            out['worst|' + c] = segstats(nav, eww, sd, V.SEGS)
        res['AB'][univ][n] = out
        log('AB', univ, n, {k: (v['CAGR'], v['t_0050']) for k, v in out['normal'].items()}, out['normal_mc_p05_p95']['S3'])

    # AB 到最新（只跑正式窗 gapfix，MC 40 次；10-12 訊號用 firstseen 分類）
    if WIN == 'gapfix':
        for n in (10, 30):
            rets = []; ewr = []; lists = []
            for seed in range(NMC):
                Sa = V.V2(WIN, YL, Y5f, latest=True, exfin=exfin, unavail=V.make_unavail('mc', seed))
                ta, tb, sda = Sa.bounds()
                pl = Sa.plan(n, 'official')
                ewa, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, ta, tb); ewr.append(ewa.pct_change())
                bb, ss = COST['normal']
                nav, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, ta, tb); rets.append(nav.pct_change())
                lists.append(tuple(K.COLS[j] for j in pl[-1][1]))
            nav_m = (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
            ew_m = (1 + pd.concat(ewr, axis=1).mean(axis=1).fillna(0)).cumprod()
            res['AB'][univ][n]['latest_exec_oct|normal'] = segstats(nav_m, ew_m, sda, V.SEGS_LATEST)
            res['AB'][univ][n]['oct_list_distinct'] = len(set(lists))
            res['lists_1012'].setdefault('AB_' + univ, {})[n] = {'signal': '2026-10-12', 'picks': list(lists[0]), 'n_distinct_over_mc': len(set(lists))}
            res['lists_1012']['AB_' + univ][n]['info'] = pick_info(Sa, pl, pd.Timestamp('2026-10-12'), n)

(V.HERE / f'run_{WIN_OUT}{WIN}.json').write_text(json.dumps(res, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
pd.DataFrame(navs).to_parquet(V.HERE / f'navs_{WIN_OUT}{WIN}.parquet')
log('done')

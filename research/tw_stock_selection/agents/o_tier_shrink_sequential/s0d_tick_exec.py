#!/usr/bin/env python3
"""步驟 0d（輸入建置，不含任何 α 估計）：第二個衝擊模型「每邊一個跳動單位」的可執行淨值。
 1) 先驗證 ecore_tick.py 的 impact='sqrt'（o5 κ 模型）逐日重現 o_exec_capacity_tiers navs_main 的 seed 0／1（相對差 < 1e-9）。
 2) 跑 40 seeds × 4 級距 × FIX／GROW × {一般：手續費 A＋1 跳；壓力：手續費 B＋零股壓力＋1 跳}，取 40 seeds 日報酬平均
    （與 navs_abavg 的 main|…|A、stress|…|B 同口徑），輸出 data/navs_tick_abavg.parquet。
用法：python3 -I -B s0d_tick_exec.py
"""
import sys, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import ecore_tick as E

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
MODEL = json.loads((E.DATA / 'model_spec.json').read_text(encoding='utf-8'))
TIERS = list(E.TIERS); VERS = ('FIX', 'GROW'); SEEDS = list(range(40))
_D = {}
def _init(): _D['D'] = E.Data()

def job(spec):
    D = _D['D']
    m = dict(MODEL); m.update(spec.get('model', {}))
    nav, info = E.simulate(D, spec['seed'], tier=E.TIERS[spec['tier']], version=spec['ver'], fee=spec['fee'], model=E.ExecModel(m),
                           flags=spec.get('flags'))
    return spec, nav.to_numpy(), {k: float(v) for k, v in info['acc'].items() if not k.startswith('_')}

def main():
    D = E.Data(); IDXN = D.IDX[D.t0:D.t1 + 1]
    # 1) 重現檢查
    chk = pd.read_parquet(E.DATA / 'navs_main_check_slice.parquet')
    specs = [dict(name='chk', seed=s, tier=t, ver=v, fee='A') for s, tl in ((0, TIERS), (1, ['30萬', '1000萬'])) for t in tl for v in VERS]
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        res = list(ex.map(job, specs))
    rep = {}
    for sp, nav, _ in res:
        c = f"{sp['tier']}|{sp['ver']}|A|s{sp['seed']:02d}"
        ref = chk[c].reindex(IDXN).to_numpy()
        d = float(np.nanmax(np.abs(nav / ref - 1)))
        rep[c] = d
        assert d < 1e-9, (c, d)
    log('sqrt 模型重現 navs_main：最大相對差', max(rep.values()))
    # 2) tick 模型
    specs = []
    for s in SEEDS:
        for t in TIERS:
            for v in VERS:
                specs.append(dict(name='tick', seed=s, tier=t, ver=v, fee='A', flags={'impact': 'tick'}))
                specs.append(dict(name='tickstress', seed=s, tier=t, ver=v, fee='B', flags={'impact': 'tick'}, model={'odd_stress': True}))
    log('jobs', len(specs))
    out = {}; acc = {}
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        for i, (sp, nav, a) in enumerate(ex.map(job, specs, chunksize=4)):
            k = f"{sp['name']}|{sp['tier']}|{sp['ver']}|{sp['fee']}"
            out.setdefault(k, {})[sp['seed']] = nav; acc.setdefault(k, []).append(a)
            if i % 100 == 0: log('done', i)
    N = {}
    for k, g in out.items():
        R = pd.DataFrame({s: pd.Series(g[s], index=IDXN).pct_change() for s in SEEDS})
        N[k] = (1 + R.mean(axis=1).fillna(0)).cumprod()
    N = pd.DataFrame(N)
    N.to_parquet(HERE / 'data' / 'navs_tick_abavg.parquet')
    yrs = (IDXN[-1] - IDXN[0]).days / 365.25
    summ = {k: {'annual_cost_pct': {kk[4:]: float(np.mean([x[kk] for x in v])) / yrs * 100 for kk in v[0] if kk.startswith('pct_')},
                'n_cap_hit_mean': float(np.mean([x['n_cap_hit'] for x in v]))} for k, v in acc.items()}
    (HERE / 'data' / 'tick_exec_meta.json').write_text(json.dumps({'repro_sqrt_maxrel': rep, 'cost_breakdown': summ,
                                                                   'tick_rule': 'TWSE 六級升降單位；買＝開盤＋1 跳、賣＝開盤−1 跳'},
                                                                  ensure_ascii=False, indent=1), encoding='utf-8')
    log('saved', N.shape)

if __name__ == '__main__':
    main()

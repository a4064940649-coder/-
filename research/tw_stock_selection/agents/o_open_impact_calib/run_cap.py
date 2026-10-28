#!/usr/bin/env python3
"""步驟 5：校準模型下重出容量表（PREREG 之後一次跑完，不改規格）。必須先有 results.json 的 PREREG（雜湊吻合、所列檔案未被改過）。
模擬組（全部手續費 A 為底；FRICTIONS＝漲跌停順延＋零股賣成本下限 0＋2% 現金＋2015 前手續費 B；Z＝零股量上限 part 1.0）：
  G1 主欄：{AB, B10} × κ{point, p10, p90} × 4 級距 × {GROW, FIX} × seed 0～39，M3V＋FRICTIONS＋Z。
  G2 壓力：{AB, B10} × M3V p95 × 基礎滑價 0.4%（每邊＋0.3%）× 全期手續費 B × 零股壓力 × FRICTIONS（不含 feemix）＋Z。
  G3 結構敏感度（AB、非判定欄）：M2、M1、M3V_VAL、O5（o5 原式）＋FRICTIONS＋Z；M3V＋FRICTIONS 但 Z part 0.2；M3V＋FRICTIONS 無 Z。
  G4 執行替代（AB、point；紙上追蹤上限）：E1、E2。
  G5 換股日分布：{ABj0..12, B10j0..12} × seed 0～19 × 4 級距 × {GROW, FIX}，M3V point＋FRICTIONS＋Z。
輸出：data/cap_navs.parquet（每個模擬的淨值 float32）、data/cap_acc.json（成本與執行計數）、data/cap_rows.parquet（G1 point 的整股開盤委託明細）。
環境變數 CAP_DRYRUN=1：只跑 2 seeds × 少數格做冒煙，輸出到 scratchpad、不印數字（PREREG 前檢查流程用）。
用法：python3 -I -B run_cap.py
"""
import sys, os, json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import ecore6 as E

DRY = os.environ.get('CAP_DRYRUN') == '1'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/cap_dry')
OUTD = E.DATA if not DRY else SCR
if not DRY:
    res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：先跑 prereg.py'
    canon = json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()
    assert hashlib.sha256(canon).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
    assert not (E.DATA / 'cap_navs.parquet').exists(), '樣本外已跑過（一次跑完）'
    for f, h in res['PREREG']['file_sha256'].items():
        assert E.sha(HERE / f) == h, f'{f} 與 PREREG 雜湊不符'
else:
    SCR.mkdir(parents=True, exist_ok=True)
SPEC = json.loads((E.DATA / 'model_calib.json').read_text(encoding='utf-8'))
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
TIERS = list(E.TIERS); VERS = ('GROW', 'FIX')
SEEDS = list(range(40)) if not DRY else [0, 1]
JSEEDS = list(range(20)) if not DRY else [0]
JS = list(range(13)) if not DRY else [0, 5]
FZ = dict(E.FRICTIONS, zpart=1.0)

def specs():
    L = []
    for st in ('AB', 'B10'):
        for kq in ('point', 'p10', 'p90'):
            for ti in TIERS:
                for ve in VERS:
                    for s in SEEDS:
                        L.append(dict(g='G1', name=f'{st}|M3V_{kq}', strat=st, seed=s, tier=ti, ver=ve, fee='A', impact='M3V', q=kq, opts=FZ,
                                      collect=(kq == 'point')))
        for ti in TIERS:
            for ve in VERS:
                for s in SEEDS:
                    L.append(dict(g='G2', name=f'{st}|STRESS', strat=st, seed=s, tier=ti, ver=ve, fee='B', impact='M3V', q='p95',
                                  opts=dict(lim=True, odd0=True, cash=0.02, feemix=False, zpart=1.0, odd_stress=True, base=0.004)))
    sens = {'M2': dict(impact='M2', q=None, opts=FZ), 'M1': dict(impact='M1', q=None, opts=FZ), 'M3V_VAL': dict(impact='M3V', q='VAL', opts=FZ),
            'O5': dict(impact='O5', q=None, opts=FZ), 'Z02': dict(impact='M3V', q='point', opts=dict(E.FRICTIONS, zpart=0.2)),
            'NOZ': dict(impact='M3V', q='point', opts=dict(E.FRICTIONS)),
            'E1': dict(impact='M3V', q='point', opts=dict(FZ, e1=True)), 'E2': dict(impact='M3V', q='point', opts=dict(FZ, e2=True))}
    for k, v in sens.items():
        for ti in TIERS:
            for ve in VERS:
                for s in SEEDS:
                    L.append(dict(g='G4' if k in ('E1', 'E2') else 'G3', name=f'AB|{k}', strat='AB', seed=s, tier=ti, ver=ve, fee='A', **v))
    for st in ('AB', 'B10'):
        for j in JS:
            for ti in TIERS:
                for ve in VERS:
                    for s in JSEEDS:
                        L.append(dict(g='G5', name=f'{st}j{j}|M3V_point', strat=f'{st}j{j}', seed=s, tier=ti, ver=ve, fee='A', impact='M3V', q='point', opts=FZ))
    if DRY:
        L = [x for x in L if x['tier'] in ('30萬', '1000萬')]
    return L

_W = {}
def _init(): _W['D'] = E.Data(load_j=True)

def job(sp):
    D = _W['D']
    if sp['q'] == 'VAL': M = E.Model(SPEC, impact='M3V', params=SPEC['sens']['M3V_VAL'])
    elif sp['impact'] == 'M3V': M = E.Model(SPEC, impact='M3V', q=sp['q'])
    else: M = E.Model(SPEC, impact=sp['impact'])
    t0, t1, _ = D.bounds(sp['strat'])
    nav, acc, rows = E.Sim(D, M, sp['opts']).run(D.plans[(sp['strat'], sp['seed'])], t0, t1, E.TIERS[sp['tier']], sp['ver'], sp['fee'], collect=sp.get('collect', False))
    tv = np.zeros(len(nav), np.float32)
    for t, v in acc['_tv'].items(): tv[t - t0] = v
    a = {k: float(v) for k, v in acc.items() if not k.startswith('_')}
    return sp, nav, tv, a, rows

if __name__ == '__main__':
    L = specs(); log('jobs', len(L))
    navs, tvs, accs, rows = {}, {}, {}, []
    with ProcessPoolExecutor(4, initializer=_init) as ex:
        for i, (sp, nav, tv, a, rw) in enumerate(ex.map(job, L, chunksize=4)):
            key = f"{sp['name']}|{sp['tier']}|{sp['ver']}|s{sp['seed']:02d}"
            navs[key] = nav.astype(np.float64); tvs[key] = tv; accs[key] = a
            if rw: rows += [(sp['strat'], sp['tier'], sp['ver'], sp['seed']) + tuple(r) for r in rw]
            if i % 500 == 0: log('done', i)
    df = pd.concat({k: v for k, v in navs.items()}, axis=1)
    df.to_parquet(OUTD / 'cap_navs.parquet')
    pd.concat({k: pd.Series(v, index=navs[k].index) for k, v in tvs.items()}, axis=1).to_parquet(OUTD / 'cap_tv.parquet')
    json.dump(accs, open(OUTD / 'cap_acc.json', 'w'), ensure_ascii=False)
    pd.DataFrame(rows, columns=['strat', 'tier', 'ver', 'seed', 't', 'j', 'side', 'n', 'lot', 'q_over_oa', 'impact_val', 'Q']).to_parquet(OUTD / 'cap_rows.parquet')
    log('sims done', len(L))

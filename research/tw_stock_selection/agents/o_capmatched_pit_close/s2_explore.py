#!/usr/bin/env python3
"""探索（PREREG 前；只看股數與市值名次的描述統計，不算任何報酬）。輸出 explore.json。
python3 -I -B s2_explore.py"""
import sys, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

def worker(name):
    import pcore as P
    T = P.setup(name)
    L = P.L
    S = L._MC['S']
    out = {}
    def at(ds): return L.IDX.get_loc(L.IDX[L.IDX >= pd.Timestamp(ds)][0])
    t19, t21 = at('2019-06-03'), at('2021-10-07')
    fin = np.isfinite(S[t19])
    out['same_shares_2019-06_vs_2021-10'] = round(float((fin & (S[t19] == S[t21])).sum() / max(fin.sum(), 1)), 4)
    out['n_with_shares_2019-06'] = int(fin.sum())
    # 每年 1 月官方日：市值前 100／前 300 名單、4763／2317 名次、下市股進入前 300 的檔數
    days = L.official_days('2014-01-01', '2026-10-31')
    rows = []
    for d in days:
        cs = T.cross_section(d)
        rk = cs['rank']
        rows.append(dict(d=str(d.date()), r4763=int(rk[L.CI['4763']]) if rk[L.CI['4763']] < 10 ** 8 else None,
                         r2317=int(rk[L.CI['2317']]) if rk[L.CI['2317']] < 10 ** 8 else None,
                         n_capuni=int(cs['capuni'].sum()), n_elig=int(cs['elig'].sum()),
                         top100=[L.COLS[j] for j in np.where(rk <= 100)[0]], top300=[L.COLS[j] for j in np.where(rk <= 300)[0]],
                         n_delisted_later_top300=int(((rk <= 300) & (L.LAST_TRADE < len(L.IDX) - 1)).sum()),
                         n_delisted_later_elig=int((cs['elig'] & (L.LAST_TRADE < len(L.IDX) - 1)).sum())))
    out['monthly'] = rows
    if P.DELIST_INFO: out['delist_patch'] = P.DELIST_INFO
    return out

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(json.dumps(worker(sys.argv[1]), ensure_ascii=False)); sys.exit()
    res = {}
    for nm in ['o7', 'i', 'ii', 'a', 'b']:
        r = subprocess.run([sys.executable, '-I', '-B', str(HERE / 's2_explore.py'), nm], capture_output=True, text=True, check=True)
        res[nm] = json.loads(r.stdout.strip().splitlines()[-1])
    # 比較：每月 top100／top300 與 o7 的重疊；4763 名次路徑
    E = {'note': '只用股數與收盤價（市值名次），沒有任何報酬。比較 o7 回推股數與 PIT 各版本。', 'by_config': {}}
    base = {r['d']: r for r in res['o7']['monthly']}
    for nm, r in res.items():
        mm = r['monthly']
        ov100 = [len(set(x['top100']) & set(base[x['d']]['top100'])) for x in mm]
        ov300 = [len(set(x['top300']) & set(base[x['d']]['top300'])) for x in mm]
        dd = pd.Series([x['d'] for x in mm])
        post = (dd >= '2019-05-01').to_numpy()
        E['by_config'][nm] = dict(
            same_shares_2019_06_vs_2021_10=r['same_shares_2019-06_vs_2021-10'], n_with_shares_2019_06=r['n_with_shares_2019-06'],
            top100_overlap_with_o7_mean_post2019_05=float(np.mean(np.array(ov100)[post])), top100_overlap_min_post2019_05=int(np.min(np.array(ov100)[post])),
            top300_overlap_with_o7_mean_post2019_05=float(np.mean(np.array(ov300)[post])),
            top100_overlap_with_o7_mean_2014_2019_04=float(np.mean(np.array(ov100)[~post])),
            r4763_path={x['d']: x['r4763'] for x in mm if x['d'][5:7] in ('01', '07')},
            r2317_2018={x['d']: x['r2317'] for x in mm if x['d'][:4] == '2018'},
            delisted_later_in_top300_mean_post2019_05=float(np.mean([x['n_delisted_later_top300'] for x, p in zip(mm, post) if p])),
            delisted_later_elig_mean_post2019_05=float(np.mean([x['n_delisted_later_elig'] for x, p in zip(mm, post) if p])),
            delisted_later_elig_by_year={y: float(np.mean([x['n_delisted_later_elig'] for x in mm if x['d'][:4] == y])) for y in sorted(set(x['d'][:4] for x in mm))},
            n_elig_mean_post2019_05=float(np.mean([x['n_elig'] for x, p in zip(mm, post) if p])),
            delist_patch=r.get('delist_patch'))
        # 4763 在前 10 名的月份數
        E['by_config'][nm]['months_4763_top10_post2019_05'] = int(sum(1 for x, p in zip(mm, post) if p and x['r4763'] is not None and x['r4763'] <= 10))
        E['by_config'][nm]['months_4763_top100_post2019_05'] = int(sum(1 for x, p in zip(mm, post) if p and x['r4763'] is not None and x['r4763'] <= 100))
    (HERE / 'explore.json').write_text(json.dumps(E, ensure_ascii=False, indent=1), encoding='utf-8')
    for nm, v in E['by_config'].items():
        print(nm, {k: v[k] for k in v if not isinstance(v[k], dict)})

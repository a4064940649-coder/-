#!/usr/bin/env python3
"""步驟三：迭代（只用挑選期）。每個版本＝一組規格解讀；B10、j 0～12、seeds 0～19、32 格（K5T／TICK × 一般／壓力 × 4 級距 × FIX／GROW），
模擬到 2019-04-30 為止，只比 SEL 年化（起點→2018-12-28）的 j 中位數與原表（navs_exec7 截到同一天）之差。不看 2019 年以後任何數字。
用法：python3 -I -B s3_iterate.py <版本名> '<extra cfg json>' '<說明>'"""
import sys, json, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_data as D, x_run as XR

HERE = Path(__file__).resolve().parent
T1 = int(np.searchsorted(D.DATES, np.datetime64('2019-04-30'), side='right')) - 1


def orig_sel():
    NX = pd.read_parquet(D.O7 / 'navs_exec7.parquet')
    NX = NX[[c for c in NX.columns if c.startswith('B10|J')]].loc[:'2019-04-30']
    return XR.table(NX, sel_only=True)


if __name__ == '__main__':
    name, extra, note = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
    f = HERE / 'out' / f'navs_iter_{name}.parquet'
    if f.exists() and '--reuse' in sys.argv: df = pd.read_parquet(f)
    else:
        costs = tuple(x.split('=')[1] for x in sys.argv if x.startswith('--costs=')) or ('normal', 'stress')
        df, acc = XR.run_grid('iter_' + name, js=range(13), seeds=range(20), extra=extra, t1=T1, cl=XR.cells(costs=costs))
        base = [x.split('=')[1] for x in sys.argv if x.startswith('--base=')]
        if base:   # 本版不影響的格子沿用 base 版（解讀開關只作用在壓力成本時）
            B = pd.read_parquet(HERE / 'out' / f'navs_iter_{base[0]}.parquet')
            df = pd.concat([df, B[[c for c in B.columns if c not in df.columns]]], axis=1).sort_index(axis=1)
            df.to_parquet(f)
    mine = XR.table(df, sel_only=True); ref = orig_sel()
    rows = {}
    for k in ref:
        rows[k] = dict(orig=ref[k]['CAGR_SEL_med'], v7x=mine[k]['CAGR_SEL_med'], diff=mine[k]['CAGR_SEL_med'] - ref[k]['CAGR_SEL_med'],
                       max_abs_diff_by_j=float(np.max(np.abs(np.array(mine[k]['CAGR_by_j']) - np.array(ref[k]['CAGR_by_j'])))) if 'CAGR_by_j' in ref[k] else None)
    d = np.array([abs(r['diff']) for r in rows.values()])
    dn = np.array([abs(r['diff']) for k, r in rows.items() if '|normal|' in k])
    rec = dict(version=name, time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), extra_cfg=extra, note=note,
               period='SEL 年化（起點→2018-12-28），模擬到 2019-04-30',
               max_abs_diff_pp_all32=float(d.max()), max_abs_diff_pp_normal16=float(dn.max()), mean_abs_diff_pp=float(d.mean()),
               n_cells_over_0p10=int((d > 0.10).sum()), cells=rows)
    p = HERE / 'iterations.json'
    L = json.loads(p.read_text()) if p.exists() else []
    L = [x for x in L if x['version'] != name] + [rec]
    p.write_text(json.dumps(L, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != 'cells'}, ensure_ascii=False))
    for k, r in sorted(rows.items(), key=lambda kv: -abs(kv[1]['diff']))[:12]:
        print(k, {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in r.items()})

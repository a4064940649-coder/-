#!/usr/bin/env python3
"""步驟 9：make_forward.py 乾跑驗收（事前寫死的三項）。
1. 2026-10-12 官方日名單逐檔（含順序）＝選股SOP.md「2026-10-12 訊號的名單」。
2. 2026-09、2026-10 重跑：2026-09 官方日＝V7 plan_V7_B10（2026-09-11）；J 名單對 V7 plans_j7（AB_V7_B10、seeds 0～19）依訊號日比對；
   2026-09 各 J 窗報酬對 V7（navs_jscan_v7 零衝擊 AB 平均、navs_exec7 K5T 100 萬 FIX）對帳。
3. 冪等：同一資料連跑兩次，輸出檔（runs.jsonl 以外）sha256 完全相同。
結果寫 results.json 的 DRYRUN。用法：python3 -I -B s9_dryrun.py"""
import sys, json, re, shutil, subprocess, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fslib as F

OUT = F.HERE / 'forward_dry'
SOP = F.AG.parent.parent.parent / '選股SOP.md'


def run_mf(out):
    r = subprocess.run([sys.executable, '-I', '-B', str(F.HERE / 'make_forward.py'), '--start', '2026-08', '--out', str(out), '--asof', '2026-11-02'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout.strip().splitlines()[-1]


def shas(d):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob('*')) if p.is_file() and p.name != 'runs.jsonl'}


def sop_list():
    t = SOP.read_text(encoding='utf-8')
    sec = t.split('## 2026-10-12 訊號的名單')[1].split('\n## ')[0]
    return re.findall(r'^\| *\d+ *\| *(\d{4}) *\|', sec, flags=re.M)


def picks(f):
    d = pd.read_csv(f, dtype={'code': str}); b = d[d.bought == 1].sort_values('rank')
    return list(b.code), b.signal.iloc[0], b.exec.iloc[0]


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    l1 = run_mf(OUT); s1 = shas(OUT)
    l2 = run_mf(OUT); s2 = shas(OUT)
    out = {'run1': json.loads(l1), 'idempotent': s1 == s2, 'n_files': len(s1)}
    # 1. SOP
    sop = sop_list(); off10, sig10, ex10 = picks(OUT / 'paper_lists/official_202610.csv')
    out['sop'] = dict(sop=sop, ours=off10, signal=sig10, exact_order=(sop == off10), same_set=(set(sop) == set(off10)))
    # 2. V7
    pv = pd.read_csv(F.RAW / 'v7/plan_V7_B10.csv'); v7off = pv[pv.signal == '2026-09-11'].picks.iloc[0].split(',')
    off09, sig09, _ = picks(OUT / 'paper_lists/official_202609.csv')
    out['v7_official_2026-09'] = dict(v7=v7off, ours=off09, exact_order=(v7off == off09))
    pj = pd.read_csv(F.RAW / 'v7/plans_j7.csv', dtype={'picks': str})
    pj = pj[(pj.strat == 'AB_V7_B10') & (pj.seed < 20)]
    jrows = []
    for m in ('202608', '202609', '202610'):
        for j in range(13):
            f = OUT / f'paper_lists/j{j}_{m}.csv'
            if not f.exists(): continue
            ours, sig, ex = picks(f)
            v = pj[pj.signal == sig]
            if not len(v):
                jrows.append(dict(anchor=m, j=j, signal=sig, v7_j=None, n_seeds=0, share_exact_set=None, best_overlap=None)); continue
            sets = [set(p.split(',')) for p in v.picks]
            jrows.append(dict(anchor=m, j=j, signal=sig, v7_j=sorted(v.j.unique().tolist()), n_seeds=len(sets),
                              share_exact_set=float(np.mean([s == set(ours) for s in sets])), best_overlap=int(max(len(s & set(ours)) for s in sets)),
                              median_overlap=float(np.median([len(s & set(ours)) for s in sets]))))
    out['v7_jlists'] = jrows
    # 窗報酬對帳（2026-09 錨定月）
    fr = pd.read_csv(OUT / 'forward_returns.csv')
    js = pd.read_parquet(F.RAW / 'v7/navs_jscan_v7.parquet'); xe = F.exec_navs()
    rec = []
    for j in range(13):
        r = fr[(fr.line == f'J{j}') & (fr.anchor == '2026-09')]
        if not len(r): continue
        a, b = pd.Timestamp(r.start.iloc[0]), pd.Timestamp(r.end.iloc[0])
        row = dict(j=j, start=str(a.date()), end=str(b.date()), ours=float(r.ret.iloc[0]) * 100)
        # V7 的 J 索引：依訊號日找（V7 2026-09 的 J7 落在不存在的 2026-09-20，j≥7 平移一格）
        v = pj[(pj.signal == str(a.date())) & (pj.seed == 0)]
        if len(v) and b <= js.index[-1]:
            vj = int(v.j.iloc[0])
            n1 = js[f'AB_V7_B10|normal|J{vj}']; n2 = xe[F.cell_col(vj, '100萬', 'FIX')]
            row.update(v7_j=vj, v7_zero=float(F._at(n1, b) / F._at(n1, a) - 1) * 100, v7_exec=float(F._at(n2, b) / F._at(n2, a) - 1) * 100)
        rec.append(row)
    R = pd.DataFrame(rec)
    if 'v7_zero' in R:
        R['d_zero'] = R.ours - R.v7_zero; R['d_exec'] = R.ours - R.v7_exec
    out['window_recon_2026-09'] = R.round(4).to_dict('records')
    out['window_recon_summary'] = dict(mean_abs_d_zero=float(R.d_zero.abs().mean()), mean_abs_d_exec=float(R.d_exec.abs().mean()),
                                       max_abs_d_exec=float(R.d_exec.abs().max()), n=int(R.v7_zero.notna().sum())) if 'v7_zero' in R else None
    tx = pd.read_csv(F.RAW / 'finlab/data/raw/taiex_twii.csv', parse_dates=['date'])
    out['v7_calendar_defect'] = [str(d.date()) for d in tx.date if d.weekday() >= 5 and d.year >= 2019]
    out['pass'] = bool(out['idempotent'] and out['sop']['exact_order'] and out['v7_official_2026-09']['exact_order'])
    out['utc'] = F.now_utc()
    F.save_key('DRYRUN', out)
    print(json.dumps({k: out[k] for k in ('idempotent', 'sop', 'v7_official_2026-09', 'pass', 'window_recon_summary', 'v7_calendar_defect')}, ensure_ascii=False, indent=0))
    print(pd.DataFrame(jrows).to_string()); print(R.round(3).to_string())


if __name__ == '__main__':
    main()

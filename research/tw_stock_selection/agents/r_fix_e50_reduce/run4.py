#!/usr/bin/env python3
"""稽核獨立複製：把正式管線 run_official.py 複製到暫存區（輸出全部改寫到暫存區，不覆蓋正式檔），跑 現行／只做 A／只做 B／A＋B（＋穩健性 Bp）。
用法：python3 -I -B run4.py [modes...]（預設 base A B AB）；加 --no-paths 則不跑 20 條路徑。
現行版必須重現：2019 起 48.76%、381 筆、與 xq_scripts/site/navs_official.csv.gz 逐日差 0、結果檔與 official_nav/out/res_official.json 相同，否則停。
輸出：out/<mode>/res.json、navs.csv.gz、trades_*.csv（本資料夾）。"""
import sys, json, subprocess, shutil, re, time
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SITE = Path('/home/user/-/xq_scripts/site')
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r_fix')
CACHE = SCR / 'cache'
modes = [a for a in sys.argv[1:] if not a.startswith('--')] or ['base', 'A', 'B', 'AB']
nopaths = '--no-paths' in sys.argv
src0 = (AG / 'official_nav' / 'run_official.py').read_text(encoding='utf-8')


def rep(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b)


for mode in modes:
    W = SCR / mode; (W / 'site').mkdir(parents=True, exist_ok=True); (W / 'out').mkdir(exist_ok=True)
    s = src0
    s = rep(s, "HERE = Path(__file__).resolve().parent", f"HERE = Path({str(AG / 'official_nav')!r})")
    s = rep(s, "SITE = Path('/home/user/-/xq_scripts/site')", f"SITE = Path({str(W / 'site')!r})")
    s = rep(s, "OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)", f"OUT = Path({str(W / 'out')!r})")
    s = rep(s, "TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/offfix/cache')", f"TMP = Path({str(CACHE)!r})")
    s = rep(s, "cslib.INS15_MASK = True\n", f"cslib.INS15_MASK = True\nsys.path.insert(0, {str(HERE)!r}); import rfix; rfix.pre(cslib, {mode!r})\n")
    s = rep(s, "P = s19lib.load()\n", f"P = s19lib.load()\nrfix.post(P, {mode!r}, TMP)\n")
    s = rep(s, "                navs[sid] = nav\n", "                navs[sid] = nav\n                lg.to_csv(OUT / f'trades_{sid}.csv', index=False)\n")
    s = rep(s, "json.dump(res, open(OUT / 'res_official.json', 'w')", "res['rfix'] = {k: v for k, v in rfix.INFO.items() if k != 'cells'}\njson.dump(res, open(OUT / 'res_official.json', 'w')")
    if mode != 'base':
        s = rep(s, "assert acc['pass'], '驗收失敗'", "pass")
    (W / 'run_copy.py').write_text(s, encoding='utf-8')
    t0 = time.time()
    r = subprocess.run([sys.executable, '-I', '-B', str(W / 'run_copy.py')] + (['--no-paths'] if nopaths else []), capture_output=True, text=True,
                       env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': '/root'})
    print(mode, 'rc', r.returncode, round(time.time() - t0), 's', r.stdout.strip().split('\n')[-1], flush=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-3000:]); sys.exit(1)
    O = HERE / 'out' / mode; O.mkdir(parents=True, exist_ok=True)
    shutil.copy(W / 'out' / 'res_official.json', O / 'res.json'); shutil.copy(W / 'site' / 'navs_official.csv.gz', O / 'navs.csv.gz')
    for f in (W / 'out').glob('trades_*.csv'):
        shutil.copy(f, O / f.name)
    if mode == 'base':
        a = pd.read_csv(SITE / 'navs_official.csv.gz'); b = pd.read_csv(O / 'navs.csv.gz')
        same = a.shape == b.shape and list(a.columns) == list(b.columns)
        num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
        dmax = float((a[num] - b[num]).abs().max().max()) if same else float('nan')
        res = json.loads((O / 'res.json').read_text()); ref = json.loads((AG / 'official_nav' / 'out' / 'res_official.json').read_text())
        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if not re.search(r'sec|rss|time|elapsed|clock|peak_mb|exright_info|rfix', k, re.I)}
            return [strip(v) for v in o] if isinstance(o, list) else o
        e0 = res['strats']['E0|W10_70']['std']
        chk = dict(nav_diff_max=dmax, same_shape=same, res_same=(nopaths or strip(res) == strip(ref)), cagr19=e0['cagr_since2019'], cagr_full=e0['cagr_full'], n_closed=e0['n_trades_closed'],
                   exright_info=res['exright_info'], exright_info_official=ref['exright_info'])
        chk['pass'] = bool(same and dmax == 0 and chk['res_same'] and round(e0['cagr_since2019'] * 100, 2) == 48.76 and e0['n_trades_closed'] == 381)
        json.dump(chk, open(O / 'baseline_check.json', 'w'), ensure_ascii=False, indent=1)
        print('baseline', chk, flush=True)
        if not chk['pass']:
            print('現行版沒有重現，停'); sys.exit(2)
    shutil.rmtree(W, ignore_errors=True)

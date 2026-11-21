#!/usr/bin/env python3
"""開關全關的回歸檢查（做法同 audit_daily/daily_check.py，但不寫 log.csv、不碰正式檔）。
把正式管線 official_nav/run_official.py 複製到暫存區、輸出改寫到暫存區，重跑後核對：
  2019 起 48.76%、全期 34.40%、已出場 381 筆、與 xq_scripts/site/navs_official.csv.gz 逐日差 0、20 條路徑中位數 46.40%、
  結果檔與 official_nav/out/res_official.json 相同（去掉執行時間等欄位）。
用法：python3 -I -B verify_baseline.py → verify/baseline_check.json；不通過時結束碼 2。"""
import sys, json, subprocess, time, re, shutil, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SITE = Path('/home/user/-/xq_scripts/site')
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/v2/baseline')
(TMP / 'site').mkdir(parents=True, exist_ok=True); (TMP / 'out').mkdir(exist_ok=True)
src = (AG / 'official_nav' / 'run_official.py').read_text(encoding='utf-8')
for a, b in (("HERE = Path(__file__).resolve().parent", f"HERE = Path({str(AG / 'official_nav')!r})"),
             ("SITE = Path('/home/user/-/xq_scripts/site')", f"SITE = Path({str(TMP / 'site')!r})"),
             ("OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)", f"OUT = Path({str(TMP / 'out')!r})")):
    assert src.count(a) == 1, a
    src = src.replace(a, b)
(TMP / 'run_copy.py').write_text(src, encoding='utf-8')
t0 = time.time()
r = subprocess.run([sys.executable, '-I', '-B', str(TMP / 'run_copy.py')], capture_output=True, text=True,
                   env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': '/root'})
chk = dict(rc=r.returncode, secs=round(time.time() - t0), cslib_sha256=hashlib.sha256((AG / 'o_ext_canslim_xs' / 'cslib.py').read_bytes()).hexdigest())
if r.returncode == 0:
    a = pd.read_csv(SITE / 'navs_official.csv.gz'); b = pd.read_csv(TMP / 'site' / 'navs_official.csv.gz')
    same = a.shape == b.shape and list(a.columns) == list(b.columns)
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    dmax = float((a[num] - b[num]).abs().max().max()) if same else float('nan')
    res = json.loads((TMP / 'out' / 'res_official.json').read_text()); ref = json.loads((AG / 'official_nav' / 'out' / 'res_official.json').read_text())
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if not re.search(r'sec|rss|time|elapsed|clock|peak_mb|exright_info', k, re.I)}
        return [strip(v) for v in o] if isinstance(o, list) else o
    e0 = res['strats']['E0|W10_70']
    chk.update(nav_same_shape=same, nav_diff_max=dmax, nav_frames_equal=bool(a.equals(b)), res_same=strip(res) == strip(ref),
               cagr19_pct=round(e0['std']['cagr_since2019'] * 100, 2), cagr_full_pct=round(e0['std']['cagr_full'] * 100, 2), maxdd_pct=round(e0['std']['maxdd'] * 100, 2),
               n_closed=e0['std']['n_trades_closed'], paths_median_pct=round(e0['paths_summary']['cagr19_median'] * 100, 2), peak_mb=res['peak_mb'], exright_info=res['exright_info'])
    chk['pass'] = bool(same and dmax == 0 and chk['res_same'] and chk['cagr19_pct'] == 48.76 and chk['cagr_full_pct'] == 34.40 and chk['n_closed'] == 381 and chk['paths_median_pct'] == 46.40)
else:
    chk.update(err=(r.stderr or r.stdout)[-1500:]); chk['pass'] = False
(HERE / 'verify').mkdir(exist_ok=True)
json.dump(chk, open(HERE / 'verify' / 'baseline_check.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(chk, ensure_ascii=False))
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if chk['pass'] else 2)

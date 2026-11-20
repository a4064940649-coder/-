#!/usr/bin/env python3
"""每日回測回歸檢查（數據稽核部）。用法：python3 -I -B daily_check.py <今天 YYYY-MM-DD>
把正式管線 official_nav/run_official.py 複製到暫存區、輸出改寫到暫存區（不覆蓋正式檔），重跑後核對：
  2019 起年化 48.76%、全期 34.40%、已出場 381 筆、與 xq_scripts/site/navs_official.csv.gz 逐日差 0、20 條路徑中位數 46.40%；
  網站 status 總資產＝live 最後一筆、除權息警告、資料最後一天。結果附加到 log.csv。"""
import sys, json, subprocess, time, re, csv, shutil
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
SITE = Path('/home/user/-/xq_scripts/site')
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/audit_daily')
today = sys.argv[1]
TMP.mkdir(parents=True, exist_ok=True); (TMP / 'site').mkdir(exist_ok=True); (TMP / 'out').mkdir(exist_ok=True)
src = (AG / 'official_nav' / 'run_official.py').read_text(encoding='utf-8')
src = src.replace("HERE = Path(__file__).resolve().parent", f"HERE = Path({str(AG / 'official_nav')!r})")
src = src.replace("SITE = Path('/home/user/-/xq_scripts/site')", f"SITE = Path({str(TMP / 'site')!r})")
src = src.replace("OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)", f"OUT = Path({str(TMP / 'out')!r})")
assert str(TMP / 'site') in src and str(TMP / 'out') in src
# 程式若會讀 SITE 底下的既有檔，先複製一份唯讀副本
for f in SITE.glob('navs_*.csv.gz'):
    if f.name != 'navs_official.csv.gz':
        shutil.copy(f, TMP / 'site' / f.name)
(TMP / 'run_copy.py').write_text(src, encoding='utf-8')
t0 = time.time()
r = subprocess.run([sys.executable, '-I', '-B', str(TMP / 'run_copy.py')], capture_output=True, text=True,
                   env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': '/root'})
secs = round(time.time() - t0)
note = []
row = dict(date=today, official_2019_cagr='', trades='', nav_diff_max='', paths_median='', site_total_ok='', exright_ok='', last_day='', result='FAIL', note='')
ok = r.returncode == 0
if not ok:
    note.append('run_official rc=%d %s' % (r.returncode, (r.stderr or r.stdout)[-300:].replace('\n', ' ')))
else:
    a = pd.read_csv(SITE / 'navs_official.csv.gz'); b = pd.read_csv(TMP / 'site' / 'navs_official.csv.gz')
    same_shape = a.shape == b.shape and list(a.columns) == list(b.columns)
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    dmax = float((a[num] - b[num]).abs().max().max()) if same_shape else float('nan')
    row['nav_diff_max'] = dmax
    res = json.loads((TMP / 'out' / 'res_official.json').read_text(encoding='utf-8'))
    ref = json.loads((AG / 'official_nav' / 'out' / 'res_official.json').read_text(encoding='utf-8'))
    txt, txt0 = json.dumps(res, ensure_ascii=False, sort_keys=True), json.dumps(ref, ensure_ascii=False, sort_keys=True)
    def strip(o):                                   # 去掉執行時間、記憶體等每次會變的欄位
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if not re.search(r'sec|rss|time|elapsed|clock|peak_mb|exright_info', k, re.I)}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    res_same = strip(res) == strip(ref)
    e0 = res['strats']['E0|W10_70']
    c19, cfull, pmed = e0['std']['cagr_since2019'], e0['std']['cagr_full'], e0['paths_summary']['cagr19_median']
    row['official_2019_cagr'] = round(c19 * 100, 2)
    row['paths_median'] = round(pmed * 100, 2)
    row['trades'] = '381' if re.search(r'\b381\b', txt) else '?'
    nums_ok = round(c19 * 100, 2) == 48.76 and round(cfull * 100, 2) == 34.40 and round(pmed * 100, 2) == 46.40 and row['trades'] == '381'
    note.append('除權息封存到 %s' % res.get('exright_info', {}).get('twt49u_last'))   # 封存每月補檔會讓 exright_info 變，不列入比對；淨值逐日差才是判準
    if not nums_ok: note.append('驗收數字不符 %.4f %.4f %.4f' % (c19, cfull, pmed))
    if not res_same:
        a_, b_ = strip(res), strip(ref)
        note.append('不同的頂層鍵：' + ','.join(k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k)))
    ok = same_shape and dmax == 0 and res_same and nums_ok
    if not same_shape: note.append('淨值檔形狀不同')
    if not res_same: note.append('res_official.json 與存檔不同')
st = json.loads((SITE / 'data' / 'status.json').read_text(encoding='utf-8'))['rows'][0]
lv = json.loads((SITE / 'data' / 'live.json').read_text(encoding='utf-8'))['rows'][-1]
row['site_total_ok'] = int(st['目前總資產'] == lv['策略'])
row['last_day'] = st['資料最後一天']
warn = [v for k, v in st.items() if '除權息警告' in k]
row['exright_ok'] = '' if not warn else int(str(warn[0]) in ('False', 'false', '無', ''))
if not row['site_total_ok']: ok = False; note.append('status 總資產≠live 最後一筆')
row['result'] = 'PASS' if ok else 'FAIL'
row['note'] = ('%ds ' % secs) + '；'.join(note)
log = HERE / 'log.csv'
new = not log.exists()
with open(log, 'a', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new: w.writeheader()
    w.writerow(row)
print(json.dumps(row, ensure_ascii=False))
shutil.rmtree(TMP, ignore_errors=True)

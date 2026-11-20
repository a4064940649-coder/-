#!/usr/bin/env python3
"""假設試算：正式管線（official_nav/run_official.py）原樣重跑，只把 0050 含息序列換成自建版（e50/e50_tr_official.csv）。
做法同 audit_daily/daily_check.py：把 run_official.py 複製到暫存區、輸出改寫到暫存區，不覆蓋任何正式檔；另外在 load() 之後插入一段換 P.E50。
先跑一次「不換」的對照（應與正式檔逐日差 0），再跑「換」的。結果寫 e50/whatif_official.json、e50/navs_official_e50fix.csv.gz（候選檔，不是正式檔）。
python3 -I -B run_official_e50_whatif.py"""
import sys, json, subprocess, shutil, re
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SITE = Path('/home/user/-/xq_scripts/site')
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/eng_e50_whatif')
INJECT = '''
# ==== 假設試算插入：0050 含息改用自建版（2015-07-01 以前沿用原序列、在接點等比例銜接；開盤÷收盤比值不變） ====
_new = pd.read_csv(%r, parse_dates=['date']).set_index('date')['tr_close']
_old = np.array(P.E50, dtype=float, copy=True); _idx = pd.DatetimeIndex(P.IDX)
_n = _new.reindex(_idx).to_numpy(); _k = int(np.flatnonzero(np.isfinite(_n))[0])
assert np.isfinite(_n[_k:]).all(), '自建序列沒有涵蓋到面板最後一天'
_e = _n.copy(); _e[:_k] = _old[:_k] * (_n[_k] / _old[_k])
P.E50O = np.asarray(P.E50O, float) / _old * _e
P.E50 = _e
print('E50 已換成自建版；接點', _idx[_k].date(), '期末 新÷舊−1 =', float((_e[-1] / _e[_k]) / (_old[-1] / _old[_k]) - 1), flush=True)
'''


def run(tag, inject):
    d = TMP / tag; (d / 'site').mkdir(parents=True, exist_ok=True); (d / 'out').mkdir(exist_ok=True)
    src = (AG / 'official_nav' / 'run_official.py').read_text(encoding='utf-8')
    for a, b in (("HERE = Path(__file__).resolve().parent", f"HERE = Path({str(AG / 'official_nav')!r})"),
                 ("SITE = Path('/home/user/-/xq_scripts/site')", f"SITE = Path({str(d / 'site')!r})"),
                 ("OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)", f"OUT = Path({str(d / 'out')!r})")):
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    if inject:
        a = "P = s19lib.load()\n"; assert src.count(a) == 1
        src = src.replace(a, a + INJECT % str(HERE / 'e50' / 'e50_tr_official.csv'))
    for f in SITE.glob('navs_*.csv.gz'):
        if f.name != 'navs_official.csv.gz':
            shutil.copy(f, d / 'site' / f.name)
    (d / 'run_copy.py').write_text(src, encoding='utf-8')
    r = subprocess.run([sys.executable, '-I', '-B', str(d / 'run_copy.py')], capture_output=True, text=True,
                       env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': '/root'})
    print(tag, 'rc', r.returncode, r.stdout[-400:], r.stderr[-300:])
    return d, r.returncode


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not re.search(r'sec|rss|time|elapsed|clock|peak_mb', k, re.I)}
    return [strip(v) for v in o] if isinstance(o, list) else o


off = pd.read_csv(SITE / 'navs_official.csv.gz', parse_dates=['date']).set_index('date')
ref = json.loads((AG / 'official_nav/out/res_official.json').read_text())
R = {}
d0, rc0 = run('control', False)
c = pd.read_csv(d0 / 'site/navs_official.csv.gz', parse_dates=['date']).set_index('date')
res0 = json.loads((d0 / 'out/res_official.json').read_text())
R['control'] = dict(rc=rc0, nav_max_abs_diff_vs_official=float((c - off).abs().max().max()), res_json_same=strip(res0) == strip(ref))
d1, rc1 = run('e50fix', True)
n = pd.read_csv(d1 / 'site/navs_official.csv.gz', parse_dates=['date']).set_index('date')
res1 = json.loads((d1 / 'out/res_official.json').read_text())
shutil.copy(d1 / 'site/navs_official.csv.gz', HERE / 'e50/navs_official_e50fix.csv.gz')
(HERE / 'e50/res_official_e50fix.json').write_text(json.dumps(res1, ensure_ascii=False, indent=1))
R['e50fix'] = dict(rc=rc1, note='rc≠0 只代表 run_official.py 結尾「與 48.76% 差 ≤ 0.02」的驗收斷言不過（數字改了本來就會不過）；輸出在斷言之前已寫完')
cols = list(off.columns); R['columns'] = {}
for k in cols:
    rel = (n[k] / off[k] - 1) * 100
    R['columns'][k] = dict(end_now=float(off[k].iloc[-1]), end_new=float(n[k].iloc[-1]), end_diff_pct=float(rel.iloc[-1]), max_abs_diff_pct=float(rel.abs().max()),
                           first_day_diff_gt_0p001pct=(str(rel[rel.abs() > 1e-3].index[0].date()) if (rel.abs() > 1e-3).any() else None),
                           max_abs_diff_pct_before_2025=float(rel.loc[:'2024-12-31'].abs().max()))


def flat(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f'{p}/{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f'{p}/{i}')
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield p, o


a, b = dict(flat(strip(ref))), dict(flat(strip(res1)))
ch = {k: (a[k], b.get(k)) for k in a if b.get(k) != a[k]}
R['res_official_json'] = dict(n_numeric_fields=len(a), n_changed=len(ch))
e0a, e0b = ref['strats']['E0|W10_70'], res1['strats']['E0|W10_70']
R['E0_W10_70'] = {k: dict(now=e0a['std'][k], new=e0b['std'][k]) for k in e0a['std'] if isinstance(e0a['std'][k], (int, float)) and e0a['std'][k] != e0b['std'][k]}
R['E0_paths_summary'] = dict(now=e0a.get('paths_summary'), new=e0b.get('paths_summary'))
R['changed_fields_sample'] = {k: v for k, v in list(sorted(ch.items(), key=lambda kv: -abs((kv[1][1] or 0) - kv[1][0])))[:40]}
(HERE / 'e50/whatif_official.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in R.items() if k != 'changed_fields_sample'}, ensure_ascii=False, indent=1))
shutil.rmtree(TMP, ignore_errors=True)

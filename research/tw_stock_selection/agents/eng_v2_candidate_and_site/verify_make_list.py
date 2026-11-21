#!/usr/bin/env python3
"""名單產生器同步檢查：make_list.py（線上，不動）對 make_list_v2.py（加 --reduce-fix，預設關）。
同一組輸入各跑三次（現行、v2 不加開關、v2 加 --reduce-fix），兩種情境：空倉 100 萬、策略帳戶（上月回測持股，底倉 1,995 股）。
  (1) 現行 對 v2 不加開關：.csv／.json／.md 三個輸出檔逐位元相同。
  (2) 現行 對 v2 加開關：前 20 名、F1、換股計畫逐欄比，列出差異。
輸出 verify/make_list_check.json。用法：python3 -I -B verify_make_list.py"""
import sys, json, subprocess, shutil, csv, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
LS = AG / 's18_live_list' / 'live_surge'
RAW = '/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data'
TSD = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsd'
W = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/v2/ml'); W.mkdir(parents=True, exist_ok=True)
MONTH = '2026-11'; CUT = '2026-11-11T14:00:00Z'
# 策略帳戶情境的輸入：由已發布的 lists/2026-11_backtest_holdings_exright.json 的計畫還原（賣出＋續抱股數、0050 換股前股數）
pub = json.loads((AG / 's18_live_list' / 'lists' / '2026-11_backtest_holdings_exright.json').read_text())
with open(W / 'hold.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['代號', '股數'])
    for r in pub['plan']['sells'] + pub['plan']['keeps']:
        w.writerow([r['code'], r['shares']])
    w.writerow(['0050', int(pub['plan']['e50']['shares_before'])])
CASES = {'空倉100萬': ['--total', '1000000'], '策略帳戶（上月回測持股）': ['--holdings', str(W / 'hold.csv'), '--cash', '0', '--base-shares', '1995']}
RUNS = {'現行': ('make_list.py', []), 'v2關': ('make_list_v2.py', []), 'v2開': ('make_list_v2.py', ['--reduce-fix'])}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = dict(month=MONTH, rev_cutoff=CUT, make_list_sha256=sha(LS / 'make_list.py'), make_list_v2_sha256=sha(LS / 'make_list_v2.py'), cases={})


def diff(a, b, path=''):
    """逐欄比兩個 JSON 物件，回傳差異清單。"""
    d = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                d.append(dict(path=f'{path}.{k}', 現行=a.get(k, '（無）'), v2開=b.get(k, '（無）')))
            else:
                d += diff(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            d += diff(x, y, f'{path}[{i}]')
    elif a != b:
        d.append(dict(path=path, 現行=a, v2開=b))
    return d


for cn, extra in CASES.items():
    J = {}; files = {}
    for rn, (prog, flag) in RUNS.items():
        pre = W / f'{len(out["cases"])}_{rn}'
        cmd = [sys.executable, '-I', '-B', str(LS / prog), 'list', '--month', MONTH, '--repo', RAW, '--repo', TSD, '--rev-cutoff', CUT, '--out-prefix', str(pre), '--quiet'] + extra + flag
        r = subprocess.run(cmd, capture_output=True, text=True, env={'GIT_NO_LAZY_FETCH': '1', 'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': '/root'})
        assert r.returncode == 0, (rn, r.stderr[-800:], r.stdout[-300:])
        files[rn] = {e: sha(str(pre) + e) for e in ('.csv', '.json', '.md')}
        J[rn] = json.loads(Path(str(pre) + '.json').read_text())
    a, c = J['現行'], J['v2開']
    codes = lambda L: [r['code'] for r in L]
    R = dict(off_identical_files={e: files['現行'][e] == files['v2關'][e] for e in files['現行']},
             sig_date=a['sig_date'], exec_date=a['exec_date'], data_last_day=a['data_last_day'],
             top20_codes_now=codes(a['top20']), top20_codes_same=codes(a['top20']) == codes(c['top20']),
             f1_codes_now=codes(a['f1']), f1_codes_same=codes(a['f1']) == codes(c['f1']),
             top20_field_diffs=diff(a['top20'], c['top20'], 'top20'), f1_field_diffs=diff(a['f1'], c['f1'], 'f1'),
             plan_diffs=diff(a['plan'], c['plan'], 'plan'),
             other_top_level_diffs=diff({k: v for k, v in a.items() if k not in ('top20', 'f1', 'plan', 'meta')}, {k: v for k, v in c.items() if k not in ('top20', 'f1', 'plan', 'meta')}),
             meta_diffs_except_reduce=diff({k: v for k, v in a['meta'].items()}, {k: v for k, v in c['meta'].items() if k != 'reduce'}, 'meta'),
             reduce_meta={k: v for k, v in c['meta']['reduce'].items() if k != 'cells'}, reduce_cells_in_window=c['meta']['reduce']['cells'],
             sells=codes(a['plan']['sells']), keeps=codes(a['plan']['keeps']), buys=codes(a['plan']['buys']))
    R['off_pass'] = all(R['off_identical_files'].values())
    R['on_changed'] = bool(R['top20_field_diffs'] or R['f1_field_diffs'] or R['plan_diffs'] or R['other_top_level_diffs'])
    out['cases'][cn] = R
# 和已發布的 2026-11 名單（11/13 產生）比名次（輸入的日 K／除權息封存之後有更新，只比代號）
p1 = json.loads((AG / 's18_live_list' / 'lists' / '2026-11.json').read_text())
out['vs_published_2026-11'] = dict(top20_same=[r['code'] for r in p1['top20']] == out['cases']['空倉100萬']['top20_codes_now'], f1_same=[r['code'] for r in p1['f1']] == out['cases']['空倉100萬']['f1_codes_now'])
out['pass_off_identical'] = all(c['off_pass'] for c in out['cases'].values())
(HERE / 'verify' / 'make_list_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for cn, R in out['cases'].items():
    print(cn, '關＝現行逐位元相同：', R['off_identical_files'], '｜開：前20代號同', R['top20_codes_same'], 'F1同', R['f1_codes_same'], '欄位差', len(R['top20_field_diffs']), len(R['f1_field_diffs']), '計畫差', len(R['plan_diffs']),
          '其他差', len(R['other_top_level_diffs']), 'meta差', len(R['meta_diffs_except_reduce']), '視窗內事件格', R['reduce_meta']['n_reduce_parvalue'], '名單相關', len(R['reduce_meta']['cells_in_top20_f1_holdings']))
print(out['vs_published_2026-11'], 'pass_off', out['pass_off_identical'])
shutil.rmtree(W, ignore_errors=True)
sys.exit(0 if out['pass_off_identical'] else 2)

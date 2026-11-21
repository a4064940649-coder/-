#!/usr/bin/env python3
"""產生兩份候選用資料封存（2026-11-21，工程）。只新增檔案，不改任何既有檔。
用法：python3 -I -B build_raw_archives.py [輸出根目錄（預設 research/tw_stock_selection/raw）] [tsd 路徑]
  加 --check：產生到指定的輸出根目錄後，和 agents/r_fix_e50_reduce/data/ 的稽核對照檔逐筆比對，結果寫本資料夾 verify/archives_check_vs_audit.json。

(1) <輸出>/e50_tr_official/etf_0050_tr.csv（date,adj）＋ etf_0050_events.csv ＋ build_info.json ＋ SHA256SUMS
    0050 含息序列＝官方日 K 原始收盤 × 事件日累乘倍數（只列有成交的日子；停牌日不列，引擎讀檔時會沿用前一天）。
    日 K：既有封存 raw/jb7k6295r7-bot_tw-stock-data（git HEAD）為主，它沒有的日子用官方日 K 倉庫 tsd（git HEAD）。
    事件：除息＝TWT49U 封存（raw/twt49u）與 tsd 的 data/universe/exright 聯集（同一天以 tsd 為準），倍數＝前收 ÷（前收 − 息值）；
          分割＝tsd 的 data/universe/etfsplit，倍數＝前收 ÷ 參考價 取最接近的整數。
    算法與稽核 agents/r_fix_e50_reduce/build_e50.py 相同（稽核已和工程 eng_audit_fixes_2026-11/build_e50_tr.py 互相核對）。
(2) <輸出>/twse_reduce/twse_reduce_events.csv（date,code,prev_close,ref_price,reason）＋ build_info.json ＋ SHA256SUMS
    上市減資恢復買賣（tsd data/universe/reduce）＋面額變更恢復買賣（data/universe/parvalue；原檔 reason 空白，這裡填「面額變更」）。
    同一天同一代號只留一列（去重複列；順序＝先減資後面額變更，留最後一筆，同稽核 rfix.events）。
不連網：git 一律 GIT_NO_LAZY_FETCH=1；本機沒有的 blob 記在 build_info.json 的 unreadable。"""
import sys, os, io, csv, json, subprocess, hashlib, glob, gzip
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # research/tw_stock_selection
args = [a for a in sys.argv[1:] if not a.startswith('--')]
OUT = Path(args[0]) if args else ROOT / 'raw'
TSD = args[1] if len(args) > 1 else '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsd'
RAW = str(ROOT / 'raw' / 'jb7k6295r7-bot_tw-stock-data')
TWT49U = ROOT / 'raw' / 'twt49u' / 'twse-exright'
AUD = ROOT / 'agents' / 'r_fix_e50_reduce' / 'data'
ENV = dict(os.environ, GIT_NO_LAZY_FETCH='1')


def git(repo, *a):
    return subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True, check=True, env=ENV).stdout


def blobs(repo, sub):
    """逐檔回傳 (檔名, 文字或 None＝本機沒有這個 blob)。"""
    items = []
    for l in git(repo, 'ls-tree', 'HEAD', sub + '/').strip().split('\n'):
        meta, path = l.split('\t'); f = path.rsplit('/', 1)[1]
        if f.endswith('.csv') and not f.startswith('_'):
            items.append((meta.split()[2], f))
    have = set(git(repo, 'cat-file', '--batch-all-objects', '--batch-check=%(objectname)', '--unordered').split())
    p = subprocess.Popen(['git', '-C', repo, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=ENV)
    for sha, f in items:
        if sha not in have:
            yield f, None; continue
        p.stdin.write((sha + '\n').encode()); p.stdin.flush()
        hdr = p.stdout.readline().decode().split()
        n = int(hdr[2]); data = p.stdout.read(n); p.stdout.read(1)
        yield f, data
    p.stdin.close(); p.wait()


def sha_dir(d, names):
    with open(d / 'SHA256SUMS', 'w') as fh:
        for n in sorted(names):
            fh.write(f'{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n')


def num(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


heads = dict(tsd=git(TSD, 'rev-parse', 'HEAD').strip(), raw=git(RAW, 'rev-parse', 'HEAD').strip())

# ================================================================ (1) 0050 含息序列
px = {}; info1 = dict(heads=heads, daily={})
for name, repo in (('raw', RAW), ('tsd', TSD)):
    n = 0; miss = []; got = 0; diff = 0
    for f, b in blobs(repo, 'data/universe/daily'):
        n += 1
        if b is None:
            miss.append(f); continue
        d = f[:10]
        i = b.find(('\n' + d + '_0050,').encode())
        if i < 0:
            continue
        j = b.find(b'\n', i + 1); line = b[i + 1:j if j > 0 else None].decode('utf-8-sig')
        hdr = b[:b.find(b'\n')].decode('utf-8-sig').strip().split(',')
        r = dict(zip(hdr, next(csv.reader([line]))))
        assert r['stock_id'] == '0050' and r['date'] == d, (f, r)
        got += 1
        if d in px:                                       # 兩個倉庫都有：以既有封存為主，記錄收盤是否不同
            diff += int(num(r['close']) != px[d][1]); continue
        px[d] = (num(r['open']), num(r['close']))
    info1['daily'][name] = dict(files=n, unreadable=len(miss), rows_0050=got, close_diff_vs_raw=diff)
    print('日K', name, info1['daily'][name], flush=True)
days = sorted(d for d, (o, c) in px.items() if c == c and c > 0)
ev = {}                                                    # date → (種類, 前收, 參考價, 息值, 來源)
tw_n = 0
for f in sorted(glob.glob(str(TWT49U / '*' / '*.csv.gz'))):
    with gzip.open(f, 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['code'] == '0050' and r['date'] >= '2015-01-01':
                ev[r['date']] = ('息', float(r['prev_close']), float(r['ref_price']), float(r['value']), 'TWT49U封存'); tw_n += 1
unread = {}
ex_n = 0
for f, b in blobs(TSD, 'data/universe/exright'):
    if b is None:
        unread.setdefault('exright', []).append(f); continue
    for r in csv.DictReader(io.StringIO(b.decode('utf-8-sig'))):
        if r.get('stock_id') == '0050':
            old = ev.get(r['date'])
            new = ('息', float(r['pre_close']), float(r['ref_price']), float(r['value']), '兩者相同' if old and old[1:4] == (float(r['pre_close']), float(r['ref_price']), float(r['value'])) else 'tsd事件檔')
            ev[r['date']] = new; ex_n += 1
for f, b in blobs(TSD, 'data/universe/etfsplit'):
    if b is None:
        unread.setdefault('etfsplit', []).append(f); continue
    for r in csv.DictReader(io.StringIO(b.decode('utf-8-sig'))):
        if r.get('stock_id') == '0050':
            assert r['date'] not in ev, '同一天有除息又有分割：' + r['date']
            ev[r['date']] = ('分割', float(r['pre_close']), float(r['ref_price']), float('nan'), 'tsd事件檔')
dayset = set(days); cum = 1.0; rows = []; evrows = []; prev_close = None
for d in days:
    f = 1.0
    if d in ev:
        k, pc, ref, val, src = ev[d]
        f = float(round(pc / ref)) if k == '分割' else pc / (pc - val)
        evrows.append(dict(date=d, kind=k, prev_close=pc, ref_price=ref, value='' if val != val else val, factor=repr(f), source=src,
                           prev_close_matches_daily=int(prev_close is not None and abs(pc - prev_close) < 1e-9)))
    cum *= f
    rows.append((d, px[d][0], px[d][1], f, px[d][1] * cum))
    prev_close = px[d][1]
skipped = sorted(d for d in ev if d not in dayset)
assert not skipped, f'事件日 0050 沒有成交，倍數會漏掉：{skipped}'
assert all(e['prev_close_matches_daily'] for e in evrows), '事件前收和日 K 前一成交日收盤不符'
D1 = OUT / 'e50_tr_official'; D1.mkdir(parents=True, exist_ok=True)
with open(D1 / 'etf_0050_tr.csv', 'w', newline='') as fh:
    w = csv.writer(fh, lineterminator='\n'); w.writerow(['date', 'adj'])
    for r in rows:
        w.writerow([r[0], repr(r[4])])
with open(D1 / 'etf_0050_events.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(evrows[0]), lineterminator='\n'); w.writeheader(); w.writerows(evrows)
info1.update(first=days[0], last=days[-1], n_days=len(days), n_events=len(evrows), n_dividend=sum(e['kind'] == '息' for e in evrows), n_split=sum(e['kind'] == '分割' for e in evrows),
             twt49u_rows_0050=tw_n, tsd_exright_rows_0050=ex_n, event_files_unreadable={k: len(v) for k, v in unread.items()}, event_files_unreadable_list=unread,
             no_trade_days_between=[d for d, (o, c) in sorted(px.items()) if not (c == c and c > 0)], cum_factor_end=cum,
             program='research/tw_stock_selection/agents/eng_v2_candidate_and_site/build_raw_archives.py')
(D1 / 'build_info.json').write_text(json.dumps(info1, ensure_ascii=False, indent=1))
sha_dir(D1, ['etf_0050_tr.csv', 'etf_0050_events.csv'])
print('0050', {k: v for k, v in info1.items() if k not in ('event_files_unreadable_list',)}, flush=True)

# ================================================================ (2) 減資／面額變更事件
ev2 = {}; info2 = dict(heads=heads); nraw = {}
for kind, label in (('reduce', None), ('parvalue', '面額變更')):
    un = []; n = 0; k = 0
    for f, b in blobs(TSD, 'data/universe/' + kind):
        n += 1
        if b is None:
            un.append(f); continue
        for r in csv.DictReader(io.StringIO(b.decode('utf-8-sig'))):
            k += 1
            ev2[(r['date'], r['stock_id'].strip())] = (r['pre_close'], r['ref_price'], label or (r.get('reason') or '').strip())   # 同鍵留最後一筆
    nraw[kind] = dict(files=n, unreadable=un, rows=k)
keys = sorted(ev2)
D2 = OUT / 'twse_reduce'; D2.mkdir(parents=True, exist_ok=True)
with open(D2 / 'twse_reduce_events.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh, lineterminator='\n'); w.writerow(['date', 'code', 'prev_close', 'ref_price', 'reason'])
    for d, c in keys:
        w.writerow([d, c, *ev2[(d, c)]])
info2.update(source=nraw, n_rows_raw=sum(v['rows'] for v in nraw.values()), n_rows=len(keys), n_dropped_duplicates=sum(v['rows'] for v in nraw.values()) - len(keys),
             first=keys[0][0], last=keys[-1][0], by_reason={}, program='research/tw_stock_selection/agents/eng_v2_candidate_and_site/build_raw_archives.py')
for kk in keys:
    info2['by_reason'][ev2[kk][2]] = info2['by_reason'].get(ev2[kk][2], 0) + 1
(D2 / 'build_info.json').write_text(json.dumps(info2, ensure_ascii=False, indent=1))
sha_dir(D2, ['twse_reduce_events.csv'])
print('減資', info2, flush=True)

# ================================================================ 和稽核對照檔逐筆比對
if '--check' in sys.argv:
    chk = {}
    a = {r['date']: float(r['adj']) for r in csv.DictReader(open(AUD / 'e50_tr_mine.csv'))}
    m = {r[0]: r[4] for r in rows}
    chk['e50'] = dict(n_mine=len(m), n_audit=len(a), only_mine=sorted(set(m) - set(a)), only_audit=sorted(set(a) - set(m)),
                      n_value_diff=sum(1 for d in m if d in a and m[d] != a[d]), max_abs_diff=max((abs(m[d] - a[d]) for d in m if d in a), default=None))
    # 讀回寫出的檔再比一次（確認寫檔沒有掉精度）
    back = {r['date']: float(r['adj']) for r in csv.DictReader(open(D1 / 'etf_0050_tr.csv'))}
    chk['e50']['file_roundtrip_same'] = back == m
    au = {}
    for fn in ('ev_reduce_tsd.csv', 'ev_parvalue_tsd.csv'):
        for r in csv.DictReader(open(AUD / fn, encoding='utf-8')):
            au[(r['date'], r['stock_id'])] = (r['pre_close'], r['ref_price'], '面額變更' if 'parvalue' in fn else (r.get('reason') or '').strip())
    chk['reduce'] = dict(n_mine=len(ev2), n_audit=len(au), only_mine=sorted(set(ev2) - set(au)), only_audit=sorted(set(au) - set(ev2)),
                         n_field_diff=sum(1 for k in ev2 if k in au and ev2[k] != au[k]))
    chk['pass'] = bool(chk['e50']['n_value_diff'] == 0 and not chk['e50']['only_mine'] and not chk['e50']['only_audit'] and chk['e50']['file_roundtrip_same']
                       and chk['reduce']['n_field_diff'] == 0 and not chk['reduce']['only_mine'] and not chk['reduce']['only_audit'])
    (HERE / 'verify').mkdir(exist_ok=True); (HERE / 'verify' / 'archives_check_vs_audit.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
    print('比對', chk)
    sys.exit(0 if chk['pass'] else 3)

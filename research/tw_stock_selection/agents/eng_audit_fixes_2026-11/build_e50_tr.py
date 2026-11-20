#!/usr/bin/env python3
"""自建 0050 含息序列（2015-07-01 起）：官方日 K 原始價＋上市除權息事件（exright）＋ETF 分割（etfsplit）。
不用 jonatw、不用倉庫的 data/adj；只讀兩個官方日 K 倉庫的 git HEAD（raw 封存＋tsd），不連網。

作法（和還原價的標準算法相同，股利在除息日以參考價再投入）：
  事件日倍數＝事件檔的 前收 ÷ 參考價（除息：前收 ÷（前收 − 息值）；分割：前收 ÷ 分割後參考價，取最接近的整數倍）。
  含息淨值＝原始收盤 × 累乘倍數。停牌日（2025-06-11～17 分割換發）沒有成交：收盤沿用前一成交日。
  事件的「前收」會和日 K 前一成交日收盤核對；事件日不是交易日（壞資料）不採用並記錄。
輸出：e50/e50_tr_official.csv（date,open,close,factor,cum_factor,tr_close,tr_open,event）、e50/e50_events.json
用法：python3 -I -B build_e50_tr.py <raw 倉庫> <tsd> <輸出資料夾>
"""
import sys, os, io, csv, json, subprocess
from pathlib import Path

ENV = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='protocol.allow', GIT_CONFIG_VALUE_0='never')
START = '2015-07-01'


def git(repo, *a, inp=None):
    return subprocess.run(['git', '-C', str(repo), *a], input=inp, capture_output=True, env=ENV)


def ls(repo, sub):
    out = {}
    for ln in git(repo, 'ls-tree', '-r', 'HEAD', sub).stdout.decode().splitlines():
        meta, path = ln.split('\t', 1); out[path.rsplit('/', 1)[1]] = meta.split()[2]
    return out


def local(repo):
    return set(x.split()[0] for x in git(repo, 'cat-file', '--batch-check', '--batch-all-objects').stdout.decode().splitlines())


def stream(repo, shas):
    """逐一讀 blob（串流，不一次載入）。"""
    p = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=ENV)
    for s in shas:
        p.stdin.write((s + '\n').encode()); p.stdin.flush()
        hd = p.stdout.readline().decode().split()
        assert len(hd) == 3 and hd[1] == 'blob', hd
        b = p.stdout.read(int(hd[2])); p.stdout.read(1)
        yield s, b
    p.stdin.close(); p.wait()


def main(raw, tsd, outd):
    outd = Path(outd); outd.mkdir(parents=True, exist_ok=True)
    # ---- 日 K：同一天 raw 與 tsd 都有時用 tsd（較新）；blob 相同就無差別
    plan = {}; src_n = {}; blob_diff = []
    for nm, repo in (('raw', raw), ('tsd', tsd)):
        have = local(repo)
        for f, sha in ls(repo, 'data/universe/daily').items():
            d = f[:10]
            if not (f.endswith('.csv') and d >= START and sha in have):
                continue
            if d in plan and plan[d][1] != sha:
                blob_diff.append(d)
            plan[d] = (repo, sha, nm)
    days = sorted(plan)
    px = {}
    for repo in (raw, tsd):
        want = [(d, plan[d][1]) for d in days if plan[d][0] == repo]
        for (d, _), (_, b) in zip(want, stream(repo, [s for _, s in want])):
            i = b.find(('\n' + d + '_0050,').encode())
            if i < 0:
                px[d] = None; continue
            j = b.find(b'\n', i + 1); line = b[i + 1:j if j > 0 else None].decode()
            hdr = b[:b.find(b'\n')].decode().split(',')
            r = dict(zip(hdr, next(csv.reader([line]))))
            f = lambda k: float(r[k]) if r.get(k, '') not in ('', '--', None) else float('nan')
            px[d] = dict(open=f('open'), close=f('close'), change=f('change'), volume=f('volume'))
    for d in days:
        src_n[plan[d][2]] = src_n.get(plan[d][2], 0) + 1
    # ---- 事件
    ev = {}; notes = []; unread = []
    for kind, sub in (('exright', 'data/universe/exright'), ('etfsplit', 'data/universe/etfsplit')):
        have = local(tsd); files = {f: s for f, s in ls(tsd, sub).items() if f.endswith('.csv') and f[:4].isdigit() and f[:10] >= START}
        unread += [f'{kind}/{f}' for f, s in files.items() if s not in have]
        ok = sorted((f, s) for f, s in files.items() if s in have)
        for (f, _), (_, b) in zip(ok, stream(tsd, [s for _, s in ok])):
            for r in csv.DictReader(io.StringIO(b.decode('utf-8'))):
                if r['stock_id'].strip() == '0050':
                    ev.setdefault(r['date'], []).append(dict(kind=kind, pre_close=float(r['pre_close']), ref_price=float(r['ref_price']),
                                                             value=float(r['value']) if r.get('value') else None, label=r.get('kind') or r.get('reason')))
    # ---- 串起來
    rows = []; cum = 1.0; last_close = None; used = []
    for d in days:
        p = px[d]; traded = p is not None and p['close'] == p['close'] and p['close'] > 0
        fac = 1.0; label = ''
        for e in ev.get(d, []):
            f = e['pre_close'] / e['ref_price']
            if e['kind'] == 'etfsplit':
                f = float(round(f))
            chk = last_close is not None and abs(e['pre_close'] - last_close) < 0.011
            used.append(dict(date=d, kind=e['kind'], label=e['label'], pre_close=e['pre_close'], ref_price=e['ref_price'], value=e['value'], factor=f,
                             pre_close_matches_daily=bool(chk), daily_prev_close=last_close))
            if not chk:
                notes.append(f'{d} {e["kind"]}：事件前收 {e["pre_close"]} ≠ 日 K 前一成交日收盤 {last_close}')
            fac *= f; label += e['label'] or e['kind']
        cum *= fac
        if traded:
            last_close = p['close']
        o = p['open'] if traded and p['open'] == p['open'] and p['open'] > 0 else float('nan')
        rows.append((d, o, last_close, fac, cum, last_close * cum, (o * cum) if o == o else float('nan'), label, int(traded)))
    not_trading = sorted(d for d in ev if d not in px and d >= START)
    for d in not_trading:
        notes.append(f'{d}：事件日不是交易日，不採用：{ev[d]}')
    with open(outd / 'e50_tr_official.csv', 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n'); w.writerow(['date', 'open', 'close', 'factor', 'cum_factor', 'tr_close', 'tr_open', 'event', 'traded'])
        for r in rows:
            w.writerow([r[0], '' if r[1] != r[1] else r[1], r[2], repr(r[3]), repr(r[4]), repr(r[5]), '' if r[6] != r[6] else repr(r[6]), r[7], r[8]])
    info = dict(first=days[0], last=days[-1], n_days=len(days), daily_source_counts=src_n, daily_blob_diff_days=blob_diff,
                heads={nm: git(r, 'rev-parse', 'HEAD').stdout.decode().strip() for nm, r in (('raw', raw), ('tsd', tsd))},
                n_events_used=len(used), events=used, events_on_non_trading_days=not_trading, event_files_unreadable=unread,
                days_no_trade=[r[0] for r in rows if not r[8]], notes=notes)
    (outd / 'e50_events.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != 'events'}, ensure_ascii=False, indent=1))
    for u in used:
        print(u)


if __name__ == '__main__':
    main(*sys.argv[1:])

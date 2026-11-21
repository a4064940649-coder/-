#!/usr/bin/env python3
"""兩來源交叉核對（離線；用已存的官方快照，不連網）。
用法：python3 -I -B cross_check.py --tsd <GitHub 日 K 倉庫> --out <結果.json> [--snap-root DIR] [--list-json <名單 json>]... [--exright 起:迄]...
  月營收：官方最後一份快照（上市、上櫃）對 GitHub 倉庫同月份彙總表，逐公司比當月營收與其他數字欄；
          給 --list-json 時，另外對照那份名單當時用的彙總表版本（meta.rev.supplement 記的 blob）。
  除權息：每個 --exright 期間要先有同起迄的官方快照；對封存 raw/twt49u 與 GitHub 倉庫事件檔逐筆比。"""
import argparse, csv, io, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_official as F, ss_lib as L          # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tsd', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--snap-root', default=str(F.SNAP_DEFAULT)); ap.add_argument('--list-json', action='append', default=[])
    ap.add_argument('--exright', action='append', default=[]); ap.add_argument('--twt49u-dir', default=str(HERE.parent.parent / 'raw' / 'twt49u' / 'twse-exright'))
    a = ap.parse_args()
    hd, ht = L.head(a.tsd)
    R = dict(checked_at_utc=F.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), tsd_head=hd, tsd_head_time=ht, revenue={}, exright={})
    for mk in ('twse', 'tpex'):
        rec, snap = L.latest_snapshot(a.snap_root, 'revenue', mk)
        if rec is None:
            R['revenue'][mk] = '沒有官方快照'; continue
        rows, info = F.parse_revenue(F.gz_read(snap)); per = next(iter(info['periods']))
        off = {r['stock_id']: r for r in rows}
        o = dict(month=per, official_snapshot=snap.name, official_sha256=rec['sha256'], official_fetched_utc=rec['fetched_utc'], report_dates=info['report_dates'])
        repo, src = L.repo_revenue(a.tsd, per, mk)
        o['vs_repo_head'] = dict(source=src, **L.cmp_revenue(repo, off)) if repo is not None else src
        if mk == 'twse':
            for lj in a.list_json:
                m = json.load(open(lj, encoding='utf-8'))['meta']['rev'].get('supplement', {}).get(per)
                if not m:
                    continue
                b = L.blob(a.tsd, m['blob'])
                if b is None:
                    o.setdefault('vs_list', {})[lj] = f"名單用的版本（blob {m['blob'][:12]}）本機讀不到"; continue
                old = {r['stock_id'].strip(): r for r in csv.DictReader(io.StringIO(b.decode('utf-8-sig')))}
                c = L.cmp_revenue(old, off)
                used_ex = sorted(set(m.get('excluded_after_cutoff', [])) | set(m.get('excluded_never_seen', [])))
                o.setdefault('vs_list', {})[lj] = dict(list_blob=m['blob'], list_source=m['source'], list_cutoff=m.get('cutoff'), list_n_used=m.get('n_used'),
                                                       list_excluded_by_cutoff=used_ex, **{k: c[k] for k in ('n_repo', 'n_official', 'n_both', 'only_repo', 'only_official', 'revenue_diff', 'n_other_col_diff')})
        R['revenue'][mk] = o
    for w in a.exright:
        s, e = w.split(':')
        rec, snap = L.latest_snapshot(a.snap_root, 'exright', f'twt49u_{s}_{e}')
        if rec is None:
            R['exright'][w] = '沒有這個期間的官方快照'; continue
        rows, info = F.parse_twt49u(F.gz_read(snap)); off = [dict(zip(L.EX_COLS, r)) for r in rows]
        arc, last = L.archive_exright(a.twt49u_dir, s, e); rep, bad = L.repo_exright(a.tsd, s, e)
        R['exright'][w] = dict(official_snapshot=snap.name, official_sha256=rec['sha256'], official_fetched_utc=rec['fetched_utc'], n_official=len(off),
                               n_official_4digit=sum(1 for r in off if L.IS4.match(r['code'])), archive_last_event=last,
                               archive_vs_official=L.cmp_events(arc, off), repo_vs_official=L.cmp_events(rep, off), repo_unreadable_days=bad)
    json.dump(R, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'已寫 {a.out}')


if __name__ == '__main__':
    main()

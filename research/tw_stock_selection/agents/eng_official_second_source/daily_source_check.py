#!/usr/bin/env python3
"""每日例行多加的一步：官方第二來源快照＋和 GitHub 倉庫比對，結果加一列到 agents/audit_daily/source_check_log.csv。

用法（python3 -I -B）：
  daily_source_check.py --tsd <GitHub 日 K 倉庫> [--today YYYY-MM-DD] [--snap-root DIR] [--log CSV] [--no-fetch]
做的事（官方請求共 3 次：上市營收、上櫃營收、近 30 日 TWT49U 各 1 次；低頻率規則寫在 fetch_official.py）：
  1. 抓官方月營收（內容和上次相同就不重存，只記「未變」），和 GitHub 倉庫同一個月的彙總表逐公司比當月營收。
  2. 抓官方近 30 日除權息，和 GitHub 倉庫事件檔逐筆比（比到倉庫最新日 K 那天為止），並檢查封存 raw/twt49u 有沒有漏。
  3. 加一列紀錄。任何來源連不上、被擋、解析失敗都只寫進紀錄；這支程式永遠回 0，不影響其他例行。
紀錄欄位：date,checked_utc,rev_period_official,n_twse_repo,n_twse_official,n_tpex_repo,n_tpex_official,rev_mismatch_twse,rev_mismatch_tpex,
          exright_window,n_ex_repo,n_ex_official,exright_mismatch,exright_archive_missing,official_status,report_date,snapshot,note
  rev_mismatch＝只有一邊有的家數＋當月營收數字不同的家數；空白＝沒得比（原因在 note）。
"""
import argparse, csv, datetime as dt, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_official as F, ss_lib as L          # noqa: E402

AG = HERE.parent
TPE = dt.timezone(dt.timedelta(hours=8))
HDR = ['date', 'checked_utc', 'rev_period_official', 'n_twse_repo', 'n_twse_official', 'n_tpex_repo', 'n_tpex_official', 'rev_mismatch_twse',
       'rev_mismatch_tpex', 'exright_window', 'n_ex_repo', 'n_ex_official', 'exright_mismatch', 'exright_archive_missing', 'official_status',
       'report_date', 'snapshot', 'note']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tsd', required=True); ap.add_argument('--today'); ap.add_argument('--no-fetch', action='store_true')
    ap.add_argument('--snap-root', default=str(F.SNAP_DEFAULT)); ap.add_argument('--log', default=str(AG / 'audit_daily' / 'source_check_log.csv'))
    ap.add_argument('--twt49u-dir', default=str(AG.parent / 'raw' / 'twt49u' / 'twse-exright'))
    a = ap.parse_args()
    today = a.today or dt.datetime.now(TPE).strftime('%Y-%m-%d')
    row = {k: '' for k in HDR}; row.update(date=today, checked_utc=F.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))
    note = []; status = []; snaps = []
    pol = F.Polite(a.snap_root)
    # ---- 1. 月營收
    try:
        rev = {} if a.no_fetch else F.do_revenue(a.snap_root, ('twse', 'tpex'), pol)
        periods = set(); reports = set()
        for mk in ('twse', 'tpex'):
            o = rev.get(mk)
            if o is not None:
                status.append(f"{mk}:{o['result']}/{o['status']}")
                if o['result'] != 'ok':
                    note.append(f"官方{mk}營收沒抓到：{o['note']}"[:150]); continue
                snaps.append(f"{mk}:{'新' if o['new_snapshot'] else '未變'}")
            rec, snap = L.latest_snapshot(a.snap_root, 'revenue', mk)
            if rec is None or (o is None and not a.no_fetch):
                continue
            rows, info = F.parse_revenue(F.gz_read(snap))
            if len(info['periods']) != 1:
                note.append(f"官方{mk}同時有多個資料年月 {info['periods']}，沒比"); row[f'n_{mk}_official'] = len(rows); continue
            per = next(iter(info['periods'])); periods.add(per); reports |= set(info['report_dates'])
            row[f'n_{mk}_official'] = len(rows)
            off = {r['stock_id']: dict(r, name=r['name']) for r in rows}
            repo, why = L.repo_revenue(a.tsd, per, mk)
            if repo is None:
                row[f'n_{mk}_repo'] = 0; note.append(f'GitHub 沒得比（{why}）'); continue
            c = L.cmp_revenue(repo, off)
            row[f'n_{mk}_repo'] = c['n_repo']
            nbad = len(c['only_repo']) + len(c['only_official']) + len(c['revenue_diff'])
            row[f'rev_mismatch_{mk}'] = nbad
            if nbad:
                note.append(f"{mk} 只有GitHub {[x.split()[0] for x in c['only_repo']][:8]} 只有官方 {[x.split()[0] for x in c['only_official']][:8]} "
                            f"數字不同 {[(x['code'], x['repo'], x['official']) for x in c['revenue_diff']][:5]}")
        row['rev_period_official'] = '|'.join(sorted(periods)); row['report_date'] = '|'.join(sorted(reports))
        if 1 <= int(today[8:10]) <= 15 and any(row[f'rev_mismatch_{m}'] not in ('', 0) for m in ('twse', 'tpex')):
            note.append('每月 1～15 日是申報期間，兩邊更新時間不同，家數差多半只是先後')
    except Exception as e:                                   # 任何狀況都只記錄
        note.append(f'營收比對出錯：{type(e).__name__}: {e}'[:200]); traceback.print_exc()
    # ---- 2. 除權息（近 30 日）
    try:
        s = (dt.date.fromisoformat(today) - dt.timedelta(days=30)).isoformat(); e = today
        row['exright_window'] = f'{s}~{e}'
        offx = None
        if not a.no_fetch:
            o = F.do_exright(a.snap_root, s, e, pol)
            status.append(f"twt49u:{o['result']}/{o['status']}")
            if o['result'] == 'ok':
                offx = L.csv_exright(o['csv']); snaps.append(f"twt49u:{'新' if o['new_snapshot'] else '未變'}")
            else:
                note.append(f"官方 TWT49U 沒抓到：{o['note']}"[:150])
        lastk = L.last_daily(a.tsd) or e
        if a.no_fetch:                                       # 不連網：拿最後一份涵蓋得到的官方快照來比（視窗可能較短）
            rec, snap, ws, we = L.covering_snapshot(a.snap_root, 'exright', 'twt49u', min(e, lastk))
            if rec is not None:
                rows, _ = F.parse_twt49u(F.gz_read(snap)); s = max(s, ws)
                offx = [dict(zip(L.EX_COLS, r)) for r in rows if r[0] >= s]
                row['exright_window'] = f'{s}~{min(e, we)}'; note.append(f"除權息用 {rec['fetched_utc']} 的官方快照")
        rep, bad = L.repo_exright(a.tsd, s, e); row['n_ex_repo'] = len(rep)
        arc, arc_last = L.archive_exright(a.twt49u_dir, s, e)
        if bad:
            note.append(f'GitHub 除權息事件檔本機讀不到 {len(bad)} 天')
        miss = set(L.cmp_events(arc, rep)['only_b'])
        if offx is not None:
            row['n_ex_official'] = len(offx)
            lim = min(e, lastk)
            c = L.cmp_events([r for r in rep if r['date'] <= lim], [r for r in offx if r['date'] <= lim])
            row['exright_mismatch'] = len(c['only_a']) + len(c['only_b']) + len(c['diff'])
            if row['exright_mismatch']:
                note.append(f"除權息 只有GitHub {c['only_a'][:5]} 只有官方 {c['only_b'][:5]} 內容不同 {[d['key'] for d in c['diff']][:5]}")
            newer = [r for r in offx if r['date'] > lim]
            if newer:
                note.append(f'官方比 GitHub 新 {len(newer)} 筆（{lim} 之後，倉庫明早才會有）')
            miss |= set(L.cmp_events(arc, [r for r in offx if r['date'] <= lim])['only_b'])
        row['exright_archive_missing'] = len(miss)
        if miss:
            note.append(f'封存 raw/twt49u 還沒有 {len(miss)} 筆（最後事件日 {arc_last}）；出名單前要重產月檔')
    except Exception as e:
        note.append(f'除權息比對出錯：{type(e).__name__}: {e}'[:200]); traceback.print_exc()
    row['official_status'] = ' '.join(status) or ('未連網' if a.no_fetch else ''); row['snapshot'] = ' '.join(snaps); row['note'] = '；'.join(note)
    try:
        F.append_csv(a.log, HDR, row)
    except OSError as e:
        print(f'寫不進紀錄檔 {a.log}：{e}')
    print(' '.join(f'{k}={row[k]}' for k in HDR if row[k] != ''))
    sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:                               # 連參數以外的意外也不讓例行中斷
        print(f'daily_source_check 意外中止：{type(e).__name__}: {e}'); sys.exit(0)

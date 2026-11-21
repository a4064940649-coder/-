#!/usr/bin/env python3
"""換股日總檢查：輸出「可以出名單／還不能，缺什麼」。不改任何正式檔；只寫官方快照資料夾與 --json 指定的檔。

用法（python3 -I -B）：
  rebalance_readiness.py --signal-date 2026-12-11 --tsd <GitHub 日 K 倉庫> [--no-fetch] [--holdings 持股.csv]
                         [--rev-cutoff 2026-12-11T14:00:00Z] [--today YYYY-MM-DD] [--json out.json] [--work 暫存資料夾]
檢查項目：
  (a) GitHub 倉庫最新日 K 日期 ≥ 訊號日。
  (b) 當月營收（訊號月份的上一個月）：兩來源各幾家、互相差幾家；名單產生器口徑的涵蓋率（重用 check_revenue_ready.py）；
      名單候選（股池內、站上季線）有沒有缺營收的。
  (c) 除權息：封存 raw/twt49u 是否含兩來源到訊號日為止的每一筆；GitHub 倉庫與官方 TWT49U 逐筆是否一致。
  (d) 減資事件封存 raw/twse_reduce 是否含兩來源到訊號日為止的每一筆。
  (e) 每日回測檢查（agents/audit_daily/log.csv 最後一列）是今天且 PASS。
  (f) 兩來源營收不一致（數字不同或只有一邊有）的公司若排在站上季線名單前 30 名內，列出來請人工確認；
      並比較「用 GitHub 營收」和「用官方營收」各算一次的前 20 名與 F1 是否相同——不同就判「還不能」，
      人工確認後加 --ack-mismatch 才放行。
連網：預設向官方請求上市營收、上櫃營收各 1 次；TWT49U、TWTAUU 只有在「今天還沒有涵蓋訊號日的快照」時才各請求 1 次
      （證交所主站很敏感，一天最多 4 次；--refetch-events＝今天抓過也再抓一次）。全部遵守 fetch_official.py 的低頻率規則。
      --no-fetch＝不連網，用快照資料夾裡最後一次成功的結果（會標明抓取時間）。官方連不上只會讓該項變成「只有單一來源」的提醒，不會當掉。
結束碼：0＝可以出名單；1＝還不能；提醒事項（WARN）不影響結束碼但一定要讀。
"""
import argparse, csv, datetime as dt, importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_official as F, ss_lib as L          # noqa: E402

AG = HERE.parent
RAW = AG.parent / 'raw'
ML = AG / 's18_live_list' / 'live_surge' / 'make_list.py'
CHK = AG / 'eng_audit_fixes_2026-11' / 'check_revenue_ready.py'
TPE = dt.timezone(dt.timedelta(hours=8))
NEAR_N = 30


def prev_month(m):
    y, mo = int(m[:4]), int(m[5:7])
    return f'{y - 1}-12' if mo == 1 else f'{y}-{mo - 1:02d}'


def dminus(d, n):
    return (dt.date.fromisoformat(d) - dt.timedelta(days=n)).isoformat()


def run_check_ready(month, tsd, cutoff, holdings, manual_csv, out_json):
    cmd = [sys.executable, '-I', '-B', str(CHK), '--month', month, '--tsd', tsd, '--json', out_json]
    if cutoff and not manual_csv:
        cmd += ['--cutoff', cutoff]
    if holdings:
        cmd += ['--holdings', holdings]
    if manual_csv:
        cmd += ['--manual-csv', manual_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    j = json.load(open(out_json, encoding='utf-8')) if os.path.exists(out_json) else {}
    return r.returncode, j, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''


def rank_table(ml, repos, month, cutoff, holdings_codes):
    """用名單產生器（唯讀 import）算訊號日全部股票的欄位；回傳 (res, {代號: 列}, 該月有營收數字的代號集合)。"""
    D = ml.Data(repos, rev_cutoff=cutoff)
    res = ml.compute(D, month, allow_partial=True, exright=True, extra_codes=holdings_codes, deadline=True, reduce=True)
    rows = {}
    if res.get('status') == '有訊號':
        for j, c in enumerate(res['_codes']):
            rows[str(c)] = res['_row'](j)
    have = set()
    Rv = getattr(D, '_Rv', None)
    if Rv is not None:
        import pandas as pd
        p = pd.Period(prev_month(month), freq='M')
        if p in Rv.index:
            s = Rv.loc[p]; have = set(s[s.notna()].index.astype(str))
    return res, rows, have, D.rev_info


def main():
    ap = argparse.ArgumentParser(description='換股日總檢查')
    ap.add_argument('--signal-date', required=True); ap.add_argument('--tsd', required=True)
    ap.add_argument('--snap-root', default=str(F.SNAP_DEFAULT)); ap.add_argument('--no-fetch', action='store_true'); ap.add_argument('--refetch-events', action='store_true'); ap.add_argument('--ack-mismatch', action='store_true')
    ap.add_argument('--holdings'); ap.add_argument('--rev-cutoff'); ap.add_argument('--today'); ap.add_argument('--json'); ap.add_argument('--work')
    ap.add_argument('--raw-repo', default=str(RAW / 'jb7k6295r7-bot_tw-stock-data'))
    ap.add_argument('--twt49u-dir', default=str(RAW / 'twt49u' / 'twse-exright'))
    ap.add_argument('--reduce-csv', default=str(RAW / 'twse_reduce' / 'twse_reduce_events.csv'))
    ap.add_argument('--audit-log', default=str(AG / 'audit_daily' / 'log.csv'))
    a = ap.parse_args()
    sig = a.signal_date; dt.date.fromisoformat(sig)
    month = sig[:7]; rev_m = prev_month(month)
    today = a.today or dt.datetime.now(TPE).strftime('%Y-%m-%d')
    cutoff = a.rev_cutoff or f'{sig}T14:00:00Z'
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix='readiness_'))
    work.mkdir(parents=True, exist_ok=True)
    R = dict(signal_date=sig, signal_month=month, rev_month=rev_m, today=today, checked_at_utc=F.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
             rev_cutoff=cutoff, items={}, fail=[], warn=[])
    I, FAIL, WARN = R['items'], R['fail'], R['warn']
    hd, ht = L.head(a.tsd)
    if hd is None:
        print(f'【還不能出名單】讀不到 GitHub 日 K 倉庫：{a.tsd}'); sys.exit(1)
    R['tsd_head'] = hd; R['tsd_head_time'] = ht

    # ---------------- (a) 日 K
    lastk = L.last_daily(a.tsd)
    I['a_日K'] = dict(last_day=lastk, ok=bool(lastk and lastk >= sig))
    if not I['a_日K']['ok']:
        FAIL.append(f'(a) GitHub 倉庫日 K 最新到 {lastk}，還沒到訊號日 {sig}（倉庫 commit 時間 {ht}）')

    # ---------------- 官方抓取（各 1 次）
    pol = F.Polite(a.snap_root); off = {}
    ex_s, ex_e = dminus(sig, 35), sig
    rd_s, rd_e = dminus(sig, 120), sig
    fetched_today = lambda rec: bool(rec) and dt.datetime.strptime(rec['fetched_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc).astimezone(TPE).strftime('%Y-%m-%d') == today
    if not a.no_fetch:
        off['rev'] = F.do_revenue(a.snap_root, ('twse', 'tpex'), pol)
        # 證交所主站很敏感：今天已經有涵蓋訊號日的快照（例如早上每日例行抓的）就不再請求，除非 --refetch-events
        if a.refetch_events or not fetched_today(L.covering_snapshot(a.snap_root, 'exright', 'twt49u', sig)[0]):
            off['ex'] = F.do_exright(a.snap_root, ex_s, ex_e, pol)
        if a.refetch_events or not fetched_today(L.covering_snapshot(a.snap_root, 'reduce', 'twtauu', sig)[0]):
            off['rd'] = F.do_reduce(a.snap_root, rd_s, rd_e, pol)
    I['官方端點'] = dict(mode='本次不連網，用最後一次成功的快照' if a.no_fetch else '本次各請求 1 次',
                      status={k: (dict((m, f"{o['result']}/{o['status']}") for m, o in v.items()) if k == 'rev' else f"{v['result']}/{v['status']}") for k, v in off.items()},
                      budget_today=pol.budget())

    # ---------------- (b) 月營收
    b = dict(rev_month=rev_m); I['b_營收'] = b
    offcsv = {}
    for mk in ('twse', 'tpex'):
        rec, snap = L.latest_snapshot(a.snap_root, 'revenue', mk)
        o = dict(snapshot=None)
        if rec is not None and snap.exists():
            rows, info = F.parse_revenue(F.gz_read(snap))
            o = dict(snapshot=snap.name, fetched_utc=rec['fetched_utc'], periods=info['periods'], report_dates=info['report_dates'], n=len(rows))
            if set(info['periods']) == {rev_m}:
                offcsv[mk] = str(F.write_revenue_csv(rows, mk, work / 'official_csv')[rev_m])
        b[f'official_{mk}'] = o
    repo_tw, why_tw = L.repo_revenue(a.tsd, rev_m, 'twse')
    b['repo_twse'] = dict(n=len(repo_tw) if repo_tw else 0, source=why_tw)
    have_repo = repo_tw is not None; have_off = 'twse' in offcsv
    if not have_off:
        o = b['official_twse']
        WARN.append(f"(b) 官方上市營收不能當第二來源：" + ('沒有成功的快照' if not o.get('snapshot') else f"官方端點目前的資料年月是 {o['periods']}（抓取 {o['fetched_utc']}），不是 {rev_m}"))
    cmp_tw = None
    if have_repo and have_off:
        cmp_tw = L.cmp_revenue(repo_tw, L.csv_revenue(offcsv['twse']), only4=True)
        b['兩來源比對_上市四碼'] = {k: v for k, v in cmp_tw.items() if k not in ('other_col_diff',)}
        n_bad = len(cmp_tw['only_repo']) + len(cmp_tw['only_official']) + len(cmp_tw['revenue_diff'])
        if n_bad:
            WARN.append(f"(b) 上市營收兩來源不一致 {n_bad} 家：只有 GitHub 有 {len(cmp_tw['only_repo'])}、只有官方有 {len(cmp_tw['only_official'])}、數字不同 {len(cmp_tw['revenue_diff'])}（明細見下）")
    # 涵蓋率（名單產生器口徑）：兩個來源各算一次
    for tag, ok, mcsv in (('repo', have_repo, None), ('official', have_off, offcsv.get('twse'))):
        if not ok:
            b[f'coverage_{tag}'] = None; continue
        rc, j, last = run_check_ready(rev_m, a.tsd, cutoff, a.holdings, mcsv, str(work / f'check_{tag}.json'))
        b[f'coverage_{tag}'] = dict(exit_code=rc, coverage_A=(j.get('coverage_A') or {}).get('value'), n_missing_A=(j.get('coverage_A') or {}).get('n_missing'),
                                    n_4digit=j.get('n_4digit'), cutoff=(j.get('cutoff') or {}).get('n_used'), holdings=j.get('holdings'), verdict=last, why_not=j.get('why_not'))
    cr, co = b.get('coverage_repo'), b.get('coverage_official')
    repo_ready = bool(cr and cr['exit_code'] == 0); off_ready = bool(co and co['exit_code'] == 0)
    b['repo_ready'] = repo_ready; b['official_ready'] = off_ready
    if not repo_ready and not off_ready:
        FAIL.append(f"(b) {rev_m} 營收兩個來源都還不能用：GitHub＝{(cr or {}).get('verdict') or why_tw}；官方＝{(co or {}).get('verdict') or '沒有這個月的快照'}")
    elif not repo_ready:
        WARN.append(f"(b) GitHub 倉庫的 {rev_m} 營收還不能用（{(cr or {}).get('verdict') or why_tw}）；官方快照可用 → 走備援：official_to_repo.py {rev_m} <新資料夾> 後把它排在最後一個 --repo")

    # ---------------- (b)(f) 名單候選與前 20 名附近
    f = {}; I['f_名單試算'] = f
    if I['a_日K']['ok'] and (repo_ready or off_ready):
        spec = importlib.util.spec_from_file_location('make_list', ML); ml = importlib.util.module_from_spec(spec); spec.loader.exec_module(ml)
        hold = [c for c in (ml.read_holdings(a.holdings) if a.holdings else {}) if c != '0050']
        tabs = {}
        if have_repo:
            tabs['repo'] = rank_table(ml, [a.raw_repo, a.tsd], month, cutoff, hold)
        if have_off:
            small = work / 'official_repo'
            if not small.exists():
                subprocess.run([sys.executable, '-I', '-B', str(HERE / 'official_to_repo.py'), rev_m, str(small), '--snap-root', a.snap_root], capture_output=True, text=True)
            if (small / '.git').exists():
                tabs['official'] = rank_table(ml, [a.raw_repo, a.tsd, str(small)], month, cutoff, hold)
        for tag, (res, rows, have, info) in tabs.items():
            t = dict(status=res.get('status'), reason=res.get('reason'), sig_date=res.get('sig_date'), exec_date=res.get('exec_date'),
                     n_univ=res.get('n_univ'), n_univ_surge=res.get('n_univ_surge'), n_ma=res.get('n_ma'))
            if res.get('sig_date') and res['sig_date'] != sig:
                msg = f"(a) 名單產生器算出的訊號日是 {res['sig_date']}，和 --signal-date {sig} 不同；請確認日期"
                if msg not in FAIL:
                    FAIL.append(msg)
            if rows:
                cand = [r for r in rows.values() if r['in_univ'] and r['above_ma60']]
                miss = sorted(r['code'] for r in cand if r['code'] not in have)
                nosurge = sorted(r['code'] for r in cand if r['code'] in have and r['surge'] is None)
                t.update(n_candidates=len(cand), candidates_missing_revenue=[f"{c} {rows[c]['name']}" for c in miss],
                         n_candidates_no_surge_history=len(nosurge),
                         top20=[r['code'] for r in res['top20']], f1=[r['code'] for r in res['f1']],
                         top_near=[f"{r['rank_ma']} {r['code']} {r['name']} {r['surge']:.3f}" for r in sorted((x for x in rows.values() if x['rank_ma'] and x['rank_ma'] <= NEAR_N), key=lambda x: x['rank_ma'])])
                if miss:
                    WARN.append(f"(b) 用{'GitHub' if tag == 'repo' else '官方'}營收時，站上季線的候選 {len(cand)} 檔中有 {len(miss)} 檔沒有 {rev_m} 營收（會被當成沒有分數；金融保險股申報期限是 15 日，訊號日還沒公布屬正常，其他產業請查原因）：{'、'.join(t['candidates_missing_revenue'][:30])}")
            f[tag] = t
        if 'repo' in tabs and 'official' in tabs and tabs['repo'][1] and tabs['official'][1]:
            ra, rb = tabs['repo'][0], tabs['official'][0]
            f['兩來源名單相同'] = dict(top20=[r['code'] for r in ra['top20']] == [r['code'] for r in rb['top20']], f1=[r['code'] for r in ra['f1']] == [r['code'] for r in rb['f1']])
            if not all(f['兩來源名單相同'].values()):
                msg = '(f) 用 GitHub 營收和用官方營收算出的前 20 名或 F1 不同（見 f_名單試算）；請人工到公開資訊觀測站確認差在哪幾家'
                if a.ack_mismatch:
                    WARN.append(msg + '（已加 --ack-mismatch：人工確認過，放行）')
                else:
                    FAIL.append(msg + '；確認後加 --ack-mismatch 再跑一次才放行')
        # 不一致的公司是否在前 30 名附近（任一來源的排名）
        if cmp_tw is not None:
            bad = {x['code']: f"數字不同 GitHub {x['repo']}／官方 {x['official']}" for x in cmp_tw['revenue_diff']}
            bad.update({x.split()[0]: '只有 GitHub 有' for x in cmp_tw['only_repo']}); bad.update({x.split()[0]: '只有官方有' for x in cmp_tw['only_official']})
            near = []
            for c, whyc in sorted(bad.items()):
                for tag, (res, rows, have, info) in tabs.items():
                    r = rows.get(c)
                    if r and r['rank_ma'] and r['rank_ma'] <= NEAR_N:
                        near.append(f"{c} {r['name']}：{whyc}；用{'GitHub' if tag == 'repo' else '官方'}營收排第 {r['rank_ma']} 名")
            f['不一致且在前30名內'] = near
            if near:
                WARN.append('(f) 兩來源營收不一致、又排在前 30 名內，請人工到公開資訊觀測站確認：' + '；'.join(near))
    else:
        f['skipped'] = '日 K 還沒到訊號日或兩來源營收都不能用，名單試算先不做'

    # ---------------- (c) 除權息
    c = dict(window=f'{ex_s}～{ex_e}', compared_window=f'{ex_s}～{ex_e}'); I['c_除權息'] = c
    arc, arc_last = L.archive_exright(a.twt49u_dir, ex_s, ex_e)
    rep, bad_days = L.repo_exright(a.tsd, ex_s, ex_e)
    c.update(archive_last_event=arc_last, n_archive=len(arc), n_repo=len(rep), repo_unreadable_days=bad_days)
    orec, osnap, os_, oe_ = L.covering_snapshot(a.snap_root, 'exright', 'twt49u', sig)
    offx = None
    if orec is not None and osnap.exists():
        rows, info = F.parse_twt49u(F.gz_read(osnap))
        cs = max(os_, ex_s)                              # 官方快照的視窗可能比這次要查的短：只比重疊的日子
        offx = [dict(zip(L.EX_COLS, r)) for r in rows if cs <= r[0] <= sig]
        arc = [r for r in arc if r['date'] >= cs]; rep = [r for r in rep if r['date'] >= cs]
        c.update(n_official=len(offx), official_fetched_utc=orec['fetched_utc'], official_window=f'{os_}～{oe_}', compared_window=f'{cs}～{sig}',
                 n_archive=len(arc), n_repo=len(rep))
        if not fetched_today(orec):
            WARN.append(f"(c) 官方 TWT49U 快照不是今天抓的（抓取 {orec['fetched_utc']}，視窗 {os_}～{oe_}）")
    m1 = L.cmp_events(arc, rep); c['封存_vs_GitHub'] = m1
    if m1['only_b'] or m1['diff']:
        FAIL.append(f"(c) 除權息封存少了或不同於 GitHub 倉庫 {len(m1['only_b']) + len(m1['diff'])} 筆（{(m1['only_b'] + [d['key'] for d in m1['diff']])[:8]}）→ 先重產 TWT49U 月檔（convert_exright.py build）")
    if m1['only_a']:
        WARN.append(f"(c) 封存有、GitHub 倉庫沒有 {len(m1['only_a'])} 筆：{m1['only_a'][:8]}")
    if bad_days:
        WARN.append(f'(c) GitHub 倉庫除權息事件檔本機讀不到 {len(bad_days)} 天：{bad_days[:8]}')
    if offx is None:
        WARN.append('(c) 沒有官方 TWT49U 這個期間的快照：除權息只有單一來源（GitHub 倉庫）')
    else:
        lim = min(sig, lastk or sig)
        m2 = L.cmp_events([r for r in rep if r['date'] <= lim], [r for r in offx if r['date'] <= lim]); c['GitHub_vs_官方'] = dict(through=lim, **m2)
        m3 = L.cmp_events(arc, offx); c['封存_vs_官方'] = m3
        if m2['only_a'] or m2['only_b'] or m2['diff']:
            FAIL.append(f"(c) 除權息 GitHub 倉庫與官方不一致（到 {lim}）：只有 GitHub {m2['only_a'][:8]}、只有官方 {m2['only_b'][:8]}、內容不同 {[d['key'] for d in m2['diff']][:8]}")
        if m3['only_b'] or m3['diff']:
            FAIL.append(f"(c) 除權息封存少了或不同於官方 {len(m3['only_b']) + len(m3['diff'])} 筆：{(m3['only_b'] + [d['key'] for d in m3['diff']])[:8]} → 先重產 TWT49U 月檔")
    if not I['a_日K']['ok']:
        c['note'] = 'GitHub 倉庫日 K 還沒到訊號日，倉庫的除權息事件也可能還沒到；等 (a) 過了再看一次'

    # ---------------- (d) 減資
    d = dict(window=f'{rd_s}～{rd_e}'); I['d_減資'] = d
    with open(a.reduce_csv, newline='', encoding='utf-8') as fh:
        allr = list(csv.DictReader(fh))
    norm = lambda r, k1='prev_close': dict(date=r['date'], code=r.get('code', r.get('stock_id', '')).strip(), prev_close=L._trim(r.get(k1, '')), ref_price=L._trim(r['ref_price']))
    arcr = [norm(r) for r in allr if rd_s <= r['date'] <= rd_e]
    d.update(archive_last_event=max((r['date'] for r in allr), default=''), n_archive=len(arcr))
    repr_ = []; rbad = []
    for sub in ('data/universe/reduce', 'data/universe/parvalue'):
        for p, sha in sorted(L.ls(a.tsd, sub).items()):
            day = p.rsplit('/', 1)[1][:10]
            if len(day) == 10 and day[4] == '-' and rd_s <= day <= rd_e and p.endswith('.csv'):
                bb = L.blob(a.tsd, sha)
                if bb is None:
                    rbad.append(day); continue
                import io
                repr_ += [norm(r, 'pre_close') for r in csv.DictReader(io.StringIO(bb.decode('utf-8-sig')))]
    cols = ['date', 'code', 'prev_close', 'ref_price']
    m4 = L.cmp_events(arcr, repr_, cols=cols); d.update(n_repo=len(repr_), repo_unreadable_days=rbad, 封存_vs_GitHub=m4)
    if m4['only_b'] or m4['diff']:
        FAIL.append(f"(d) 減資事件封存少了或不同於 GitHub 倉庫：{(m4['only_b'] + [x['key'] for x in m4['diff']])[:8]} → 先重產（agents/eng_v2_candidate_and_site/build_raw_archives.py）")
    rrec, rsnap, rs_, re_ = L.covering_snapshot(a.snap_root, 'reduce', 'twtauu', sig)
    if rrec is not None and rsnap.exists():
        rows, info = F.parse_twtauu(F.gz_read(rsnap))
        cs = max(rs_, rd_s)
        offr = [dict(date=r[0], code=r[1], prev_close=r[2], ref_price=r[3]) for r in rows if cs <= r[0] <= sig]
        m5 = L.cmp_events([r for r in arcr if r['date'] >= cs], offr, cols=cols)
        d.update(n_official=len(offr), official_fetched_utc=rrec['fetched_utc'], official_window=f'{rs_}～{re_}', 封存_vs_官方=m5)
        if not fetched_today(rrec):
            WARN.append(f"(d) 官方 TWTAUU 快照不是今天抓的（抓取 {rrec['fetched_utc']}）")
        if m5['only_b'] or m5['diff']:
            FAIL.append(f"(d) 減資事件封存少了或不同於官方 TWTAUU：{(m5['only_b'] + [x['key'] for x in m5['diff']])[:8]} → 先重產減資封存")
        if m5['only_a']:
            d['note'] = f"封存有、官方 TWTAUU 沒有 {len(m5['only_a'])} 筆（面額變更不在 TWTAUU，屬正常；請對照 reason 欄）：{m5['only_a'][:8]}"
    else:
        WARN.append('(d) 沒有官方 TWTAUU 這個期間的快照：減資事件只有單一來源（GitHub 倉庫）')

    # ---------------- (e) 每日回測檢查
    e = {}; I['e_每日回測檢查'] = e
    try:
        with open(a.audit_log, newline='', encoding='utf-8') as fh:
            last = list(csv.DictReader(fh))[-1]
        e.update(date=last['date'], result=last['result'], note=last.get('note', ''))
        if last['date'] != today or last['result'] != 'PASS':
            FAIL.append(f"(e) 每日回測檢查最後一列是 {last['date']} {last['result']}，不是今天（{today}）PASS → 先跑 audit_daily/daily_check.py {today}")
    except (OSError, IndexError, KeyError) as ex:
        FAIL.append(f'(e) 讀不到每日回測檢查紀錄：{ex}')

    # ---------------- 判定與輸出
    R['ready'] = not FAIL
    R['use_official_backup'] = bool(R['ready'] and not repo_ready and off_ready)
    out = [f'== 換股日總檢查：訊號日 {sig}（{month} 名單，用 {rev_m} 營收）；檢查時間 {R["checked_at_utc"]}（今天 {today}）==',
           f'GitHub 倉庫 HEAD {hd[:12]}，commit 時間 {ht}',
           f"(a) 日 K 最新 {lastk}：{'OK' if I['a_日K']['ok'] else '還沒到訊號日'}"]
    ot, op = b['official_twse'], b['official_tpex']
    out.append(f"(b) {rev_m} 營收：GitHub 上市 {b['repo_twse']['n']} 家（{'可用' if repo_ready else '不可用'}）；"
               f"官方上市快照 {ot.get('periods')} 出表 {ot.get('report_dates')} 抓取 {ot.get('fetched_utc')}（{'可用' if off_ready else '不可用'}）；官方上櫃 {op.get('periods')}")
    for tag in ('repo', 'official'):
        cv = b.get(f'coverage_{tag}')
        if cv:
            out.append(f"    {'GitHub' if tag == 'repo' else '官方  '}：四碼 {cv['n_4digit']} 家，涵蓋率 A {cv['coverage_A']}（門檻 0.95），{cv['verdict']}")
    if cmp_tw is not None:
        out.append(f"    兩來源（上市四碼）：都有 {cmp_tw['n_both']} 家；只有 GitHub {cmp_tw['only_repo']}；只有官方 {cmp_tw['only_official']}；數字不同 {cmp_tw['revenue_diff']}")
    for tag, t in f.items():
        if isinstance(t, dict) and 'status' in t:
            out.append(f"(f) 用{'GitHub' if tag == 'repo' else '官方'}營收試算：{t['status']} {t.get('reason') or ''}，股池有分數 {t.get('n_univ_surge')}、站上季線 {t.get('n_ma')}；"
                       f"候選缺營收 {len(t.get('candidates_missing_revenue', []))} 檔；F1 {t.get('f1')}")
    if '兩來源名單相同' in f:
        out.append(f"    兩來源算出的前 20 名相同：{f['兩來源名單相同']['top20']}；F1 相同：{f['兩來源名單相同']['f1']}；不一致且在前 {NEAR_N} 名內：{f.get('不一致且在前30名內') or '無'}")
    if 'skipped' in f:
        out.append(f"(f) {f['skipped']}")
    out.append(f"(c) 除權息 {c['compared_window']}：封存 {c['n_archive']} 筆（最後事件日 {c['archive_last_event']}）、GitHub {c['n_repo']} 筆、官方 {c.get('n_official', '無快照')} 筆；"
               f"封存對 GitHub 相同 {m1['n_same']}" + (f"；GitHub 對官方相同 {c['GitHub_vs_官方']['n_same']}（到 {c['GitHub_vs_官方']['through']}）" if 'GitHub_vs_官方' in c else ''))
    out.append(f"(d) 減資 {d['window']}：封存 {d['n_archive']} 筆（最後事件日 {d['archive_last_event']}）、GitHub {d['n_repo']} 筆、官方 {d.get('n_official', '無快照')} 筆" + (f"；{d['note']}" if d.get('note') else ''))
    out.append(f"(e) 每日回測檢查：{e.get('date')} {e.get('result')}")
    out.append(f"官方端點：{I['官方端點']['mode']} {json.dumps(I['官方端點']['status'], ensure_ascii=False)}；今天已用請求數 {json.dumps({k: v['used'] for k, v in I['官方端點']['budget_today'].items()})}")
    for w in WARN:
        out.append('提醒：' + w)
    for x in FAIL:
        out.append('缺：' + x)
    if R['ready']:
        out.append('【可以出名單】' + ('（GitHub 營收還沒到，改用官方備援：先跑 official_to_repo.py，把輸出資料夾排在最後一個 --repo）' if R['use_official_backup'] else '') + ('；有提醒事項，請先讀完' if WARN else ''))
    else:
        out.append(f'【還不能出名單】缺 {len(FAIL)} 項（見上面「缺：」）')
    print('\n'.join(out))
    if a.json:
        json.dump(R, open(a.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    sys.exit(0 if R['ready'] else 1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""月營收到齊檢查（12/11 出名單當天先跑；只讀、不連網、不改任何檔）。

用法（python3 -I -B）：
  check_revenue_ready.py --month 2026-11 --tsd <官方日K倉庫> [--cutoff 2026-12-11T14:00:00Z]
                         [--holdings <持股 csv>] [--rev-csv <tw_revenue.csv>] [--manual-csv <人工下載轉好的彙總表>] [--json <輸出>]
  --month＝「營收月份」（12 月名單用 2026-11 營收；注意和 make_list.py 的 --month＝訊號月份差一個月）。

它回報：
  1. 彙總表 data/mops/revenue_hist/<月>_twse.csv 在倉庫 HEAD 是否存在、本機讀不讀得到、欄位對不對。
  2. 上市涵蓋率（兩種口徑）：
     A＝名單產生器口徑：上一個月營收有數字的上市公司，本月也有的比例（make_list.py 的 REV_MIN_COVER＝95% 門檻就是這個；
        分母取 tw_revenue.csv 最後一個月的上市公司，和 make_list.py 第 195、219 行相同）。
     B＝對最新一天日 K 的上市四碼普通股（market＝twse）。
  3. 缺哪些公司（A、B 各列；有給 --holdings 時另外標出持股是否缺）。
  4. 檔案最後更新時間：倉庫 HEAD commit 時間、首見紀錄 data/meta/revenue_seen/<月>/ 的第一份與最後一份快照時間（UTC 與台北）。
  5. 有給 --cutoff 時：截止前已看到幾家（名單產生器 --rev-cutoff 的時點檢查同一套規則）。
結束碼：0＝可以出名單；1＝還不行（涵蓋率 A < 95% 或持股缺營收）；2＝檔案不存在或讀不到。
--manual-csv：人工備援演練用——檢查人工下載後轉成同欄位的 csv（至少要有 stock_id、name、當月營收），不讀倉庫的彙總表。
"""
import sys, os, io, re, csv, json, argparse, subprocess, datetime as dt

ENV = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='protocol.allow', GIT_CONFIG_VALUE_0='never')
REV_DEFAULT = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
REV_MIN_COVER = 0.95
NEED_COLS = ('stock_id', 'name', '當月營收')
IS4 = re.compile(r'^[1-9]\d{3}$')
TPE = dt.timezone(dt.timedelta(hours=8))


def git(tsd, *a):
    return subprocess.run(['git', '-C', tsd, *a], capture_output=True, env=ENV)


def ls(tsd, sub):
    r = git(tsd, 'ls-tree', '-r', 'HEAD', sub)
    out = {}
    for ln in r.stdout.decode().splitlines():
        meta, path = ln.split('\t', 1); out[path] = meta.split()[2]
    return out


def blob(tsd, sha):
    """本機沒有內容回 None（不向遠端抓）。"""
    r = git(tsd, 'cat-file', 'blob', sha)
    return r.stdout if r.returncode == 0 else None


def fnum(s):
    try:
        return float(str(s).replace(',', '').strip())
    except ValueError:
        return float('nan')


def prev_month(m):
    y, mo = int(m[:4]), int(m[5:7])
    return f'{y - 1}-12' if mo == 1 else f'{y}-{mo - 1:02d}'


def read_hist(b):
    rows = list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
    cols = list(rows[0].keys()) if rows else []
    return rows, cols


def main():
    ap = argparse.ArgumentParser(description='月營收到齊檢查')
    ap.add_argument('--month', required=True); ap.add_argument('--tsd', required=True)
    ap.add_argument('--cutoff'); ap.add_argument('--holdings'); ap.add_argument('--rev-csv', default=REV_DEFAULT)
    ap.add_argument('--manual-csv'); ap.add_argument('--json'); ap.add_argument('--max-list', type=int, default=60)
    a = ap.parse_args()
    if not re.fullmatch(r'\d{4}-\d{2}', a.month):
        sys.exit('--month 要寫 YYYY-MM（營收月份）')
    R = dict(month=a.month, tsd=a.tsd, checked_at_utc=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), problems=[])
    P = R['problems']
    h = git(a.tsd, 'log', '-1', '--format=%H %cI')
    if h.returncode != 0:
        print(f'讀不到官方日 K 倉庫：{a.tsd}'); sys.exit(2)
    R['tsd_head'], R['tsd_head_time'] = h.stdout.decode().split()
    # ---- 1. 彙總表
    path = f'data/mops/revenue_hist/{a.month}_twse.csv'
    if a.manual_csv:
        R['source'] = f'人工檔 {a.manual_csv}'
        R['file_mtime_taipei'] = dt.datetime.fromtimestamp(os.path.getmtime(a.manual_csv), TPE).strftime('%Y-%m-%d %H:%M:%S')
        b = open(a.manual_csv, 'rb').read()
    else:
        files = ls(a.tsd, 'data/mops/revenue_hist')
        have = sorted(re.match(r'.*/(\d{4}-\d{2})_twse\.csv$', p).group(1) for p in files if re.match(r'.*/\d{4}-\d{2}_twse\.csv$', p))
        R['hist_last_month_in_repo'] = have[-1] if have else None
        if path not in files:
            R['status'] = '檔案不存在'
            print(f'【還不能出名單】倉庫 HEAD（{R["tsd_head"][:12]}，{R["tsd_head_time"]}）沒有 {path}；倉庫最新的彙總表月份＝{R["hist_last_month_in_repo"]}。'
                  f'\n→ 先更新官方日 K 倉庫；還是沒有就走人工備援（見 plan.json 第 3 項）。')
            if a.json:
                json.dump(R, open(a.json, 'w'), ensure_ascii=False, indent=1)
            sys.exit(2)
        R['source'] = f'{a.tsd}@{R["tsd_head"][:12]}:{path}'; R['blob'] = files[path]
        b = blob(a.tsd, files[path])
        if b is None:
            R['status'] = '檔案在倉庫清單裡，但本機沒有內容（部分複製）'
            print(f'【還不能出名單】{path} 本機讀不到內容（倉庫是部分複製，需要連網補抓這個檔）。'); sys.exit(2)
    rows, cols = read_hist(b)
    R['columns'] = cols; R['n_rows'] = len(rows)
    miss_cols = [c for c in NEED_COLS if c not in cols]
    if miss_cols:
        R['status'] = f'欄位不對，缺 {miss_cols}'; print(f'【還不能出名單】彙總表缺欄位 {miss_cols}；實際欄位：{cols}'); sys.exit(2)
    per = sorted({r.get('period', '') for r in rows} - {''})
    if per and per != [a.month]:
        P.append(f'period 欄不是 {a.month}：{per}')
    r4 = [r for r in rows if IS4.match(r['stock_id'].strip())]
    rev = {}
    for r in r4:
        rev[r['stock_id'].strip()] = fnum(r['當月營收'])
    name = {r['stock_id'].strip(): r['name'].strip() for r in r4}
    pos = {c for c, v in rev.items() if v == v and v > 0}
    nonpos = sorted(c for c in rev if c not in pos)
    R.update(n_4digit=len(rev), n_dup_codes=len(r4) - len(rev), n_revenue_positive=len(pos), nonpositive_or_blank=[f'{c} {name.get(c, "")}' for c in nonpos])
    # ---- 5. 截止時點（同 make_list.Data._seen_codes）
    used = set(rev)
    seen = ls(a.tsd, f'data/meta/revenue_seen/{a.month}') if not a.manual_csv else {}
    snaps = []
    for p in sorted(seen):
        m = re.match(r'^(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})', p.rsplit('/', 1)[1])
        if m:
            snaps.append((dt.datetime.strptime(m.group(1) + m.group(2) + m.group(3) + m.group(4), '%Y-%m-%d%H%M%S').replace(tzinfo=dt.timezone.utc), p, seen[p]))
    if snaps:
        f = lambda t: f"{t.strftime('%Y-%m-%d %H:%M:%S')} UTC（台北 {t.astimezone(TPE).strftime('%m-%d %H:%M')}）"
        R['seen_snapshots'] = dict(n=len(snaps), first=f(snaps[0][0]), last=f(snaps[-1][0]))
    elif not a.manual_csv:
        P.append('倉庫沒有這個月的首見紀錄（data/meta/revenue_seen），無法做時點檢查')
    if a.cutoff and snaps:
        cut = dt.datetime.fromisoformat(a.cutoff.replace('Z', '+00:00'))
        cut = cut if cut.tzinfo else cut.replace(tzinfo=dt.timezone.utc)
        ok, late, unread = set(), set(), 0
        for t, p, sha in snaps:
            bb = blob(a.tsd, sha)
            if bb is None:
                unread += 1; continue
            cs = {r['stock_id'].strip() for r in csv.DictReader(io.StringIO(bb.decode('utf-8-sig')))}
            (ok if t <= cut else late).update(cs)
        late -= ok
        ex_late = sorted(c for c in rev if c not in ok and c in late); ex_never = sorted(c for c in rev if c not in ok and c not in late)
        used = {c for c in rev if c in ok}
        R['cutoff'] = dict(cutoff=a.cutoff, n_used=len(used), excluded_after_cutoff=ex_late, excluded_never_seen=ex_never, snapshots_unreadable=unread)
        if unread:
            P.append(f'有 {unread} 份首見快照本機讀不到內容，截止檢查可能把公司誤判成沒看到')
    # ---- 2. 涵蓋率 A（名單產生器口徑）
    pm = prev_month(a.month); prevA = None; srcA = None; base_last = None
    try:
        last = {}
        with open(a.rev_csv, newline='') as fh:
            rd = csv.reader(fh); hd = next(rd); im, ik, ic, iv = hd.index('month'), hd.index('market'), hd.index('code'), hd.index('revenue_k_twd')
            for r in rd:
                if r[ik] == 'sii':
                    last.setdefault(r[im][:7], set()).add(r[ic])
        base_last = max(last); R['rev_csv_last_month'] = base_last
        prevA = last[base_last]; srcA = f'tw_revenue.csv {base_last} 上市 {len(prevA)} 家（make_list.py 的分母）'
        if base_last >= a.month:
            P.append(f'tw_revenue.csv 已經到 {base_last}，名單產生器不會用彙總表補 {a.month}')
    except (OSError, ValueError) as e:
        P.append(f'讀不到營收主檔 {a.rev_csv}：{e}')
    if prevA is not None:
        covA = len(used & prevA) / max(len(prevA), 1); missA = sorted(prevA - used)
        R['coverage_A'] = dict(value=round(covA, 4), denominator=srcA, n_missing=len(missA), missing=missA, threshold=REV_MIN_COVER)
    # 上一個月彙總表（倉庫裡有才比；12 月名單時＝2026-10 彙總表）
    if not a.manual_csv:
        pp = f'data/mops/revenue_hist/{pm}_twse.csv'
        if pp in files and (bb := blob(a.tsd, files[pp])) is not None:
            pr, _ = read_hist(bb); pc = {r['stock_id'].strip() for r in pr if IS4.match(r['stock_id'].strip())}
            R['vs_prev_month_hist'] = dict(prev_month=pm, n_prev=len(pc), coverage=round(len(used & pc) / max(len(pc), 1), 4),
                                           missing=sorted(pc - used), new=sorted(used - pc))
    # ---- 2. 涵蓋率 B（對最新日 K 的上市四碼）
    dl = ls(a.tsd, 'data/universe/daily'); days = sorted(p for p in dl if re.search(r'/\d{4}-\d{2}-\d{2}\.csv$', p))
    uni = None
    for p in reversed(days[-10:]):
        bb = blob(a.tsd, dl[p])
        if bb is not None:
            uni = {}
            for r in csv.DictReader(io.StringIO(bb.decode('utf-8-sig'))):
                if r['market'] == 'twse' and IS4.match(r['stock_id']):
                    uni[r['stock_id']] = r['name'].strip()
            R['latest_price_day'] = p.rsplit('/', 1)[1][:10]; break
    if uni:
        missB = sorted(set(uni) - used)
        R['coverage_B'] = dict(value=round(len(set(uni) & used) / len(uni), 4), denominator=f'{R["latest_price_day"]} 日 K 上市四碼 {len(uni)} 檔',
                               n_missing=len(missB), missing=[f'{c} {uni[c]}' for c in missB])
        name.update({c: n for c, n in uni.items() if c not in name})
    else:
        P.append('最近 10 個日 K 檔本機都讀不到，沒算涵蓋率 B')
    # ---- 3. 持股
    if a.holdings:
        hc = []
        for r in csv.reader(open(a.holdings, newline='', encoding='utf-8-sig')):
            if r and IS4.match(r[0].strip()):
                hc.append(r[0].strip())
        hm = [c for c in hc if c not in used or c not in pos]
        R['holdings'] = dict(codes=hc, missing_revenue=hm)
    # ---- 判定
    ready = True; why = []
    if 'coverage_A' not in R:
        ready = False; why.append('算不出涵蓋率 A')
    elif R['coverage_A']['value'] < REV_MIN_COVER:
        ready = False; why.append(f"涵蓋率 A {R['coverage_A']['value']:.1%} < {REV_MIN_COVER:.0%}（名單產生器會拒絕出名單）")
    if a.holdings and R['holdings']['missing_revenue']:
        ready = False; why.append(f"持股缺營收：{R['holdings']['missing_revenue']}（沒有 SURGE 分數會被當成跌出名單，先確認是不是還沒公布）")
    R['ready'] = ready; R['why_not'] = why
    # ---- 輸出
    L = [f"== 月營收到齊檢查：{a.month} 營收（給 {a.month[:4]}/{int(a.month[5:7]) % 12 + 1:02d} 月名單用）==",
         f"來源：{R['source']}",
         f"倉庫 HEAD：{R['tsd_head'][:12]}，commit 時間 {R['tsd_head_time']}"]
    if a.manual_csv:
        L.append(f"人工檔最後修改（台北）：{R['file_mtime_taipei']}")
    if 'seen_snapshots' in R:
        s = R['seen_snapshots']; L.append(f"首見紀錄 {s['n']} 份：第一份 {s['first']}；最後一份 {s['last']}  ← 彙總表內容最後一次有變動的時間")
    L.append(f"格式：{len(cols)} 欄（{'、'.join(cols[:6])}…），{R['n_rows']} 列，四碼普通股 {R['n_4digit']} 家，營收 > 0 的 {R['n_revenue_positive']} 家")
    if nonpos:
        L.append(f"營收 ≤ 0 或空白（引擎當缺值）{len(nonpos)} 家：{'、'.join(R['nonpositive_or_blank'][:a.max_list])}")
    if 'cutoff' in R:
        c = R['cutoff']; L.append(f"截止 {c['cutoff']} 前已看到 {c['n_used']} 家；截止後才出現 {len(c['excluded_after_cutoff'])} 家 {c['excluded_after_cutoff'][:20]}；首見紀錄沒有 {len(c['excluded_never_seen'])} 家 {c['excluded_never_seen'][:20]}")
    if 'coverage_A' in R:
        c = R['coverage_A']; L.append(f"上市涵蓋率 A（名單產生器口徑，門檻 95%）：{c['value']:.2%}；分母＝{c['denominator']}；缺 {c['n_missing']} 家")
        if c['missing']:
            L.append('  缺：' + '、'.join(f"{x} {name.get(x, '')}".strip() for x in c['missing'][:a.max_list]) + (' …' if c['n_missing'] > a.max_list else ''))
    if 'vs_prev_month_hist' in R:
        c = R['vs_prev_month_hist']; L.append(f"對上月彙總表（{c['prev_month']}，{c['n_prev']} 家）：{c['coverage']:.2%}；上月有本月沒有 {len(c['missing'])} 家 {c['missing'][:20]}；本月新出現 {len(c['new'])} 家 {c['new'][:20]}")
    if 'coverage_B' in R:
        c = R['coverage_B']; L.append(f"上市涵蓋率 B（對 {c['denominator']}）：{c['value']:.2%}；缺 {c['n_missing']} 家")
        if c['missing']:
            L.append('  缺：' + '、'.join(c['missing'][:a.max_list]) + (' …（完整清單用 --json）' if c['n_missing'] > a.max_list else ''))
    if a.holdings:
        L.append(f"持股 {R['holdings']['codes']}：缺營收 {R['holdings']['missing_revenue'] or '無'}")
    for p in P:
        L.append(f'注意：{p}')
    L.append('【可以出名單】' if ready else '【還不能出名單】' + '；'.join(why))
    print('\n'.join(L))
    if a.json:
        json.dump(R, open(a.json, 'w'), ensure_ascii=False, indent=1)
    sys.exit(0 if ready else 1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""月營收人工備援：把人工下載的「上市公司每月營業收入彙總表」csv 轉成名單產生器讀得懂的小倉庫。

名單產生器（make_list.py 第 188～199 行）只會從 --repo 給的 git 倉庫 HEAD 讀 data/mops/revenue_hist/YYYY-MM_twse.csv，
同月份「後面給的倉庫覆蓋前面的」。所以人工備援＝做一個只有這一個檔的小 git 倉庫，排在最後一個 --repo。

用法（python3 -I -B）：
  manual_revenue_to_repo.py <人工下載 csv> <營收月份 YYYY-MM> <輸出資料夾（新的、空的）>
接受的欄名（自動對應，編碼 utf-8／utf-8-sig／big5 都試）：
  代號：stock_id／公司代號；名稱：name／公司名稱；當月營收：當月營收／營業收入-當月營收（單位：千元，和公開資訊觀測站彙總表相同）。
  有「資料年月」（民國 YYYMM，例如 11511）或 period 欄時會檢查月份是否相符，不符就停。
輸出：<輸出資料夾>/data/mops/revenue_hist/<月>_twse.csv（欄位 stock_id,name,period,market,當月營收）＋一個 commit。
注意：小倉庫沒有 data/meta/revenue_seen，名單產生器的 --rev-cutoff 時點檢查對這個月不會做（meta 會寫「未做時點檢查」）——人工檔請在訊號日截止時間前下載。
"""
import sys, os, io, csv, re, subprocess
from pathlib import Path

ALIAS = dict(stock_id=('stock_id', '公司代號', '代號'), name=('name', '公司名稱', '名稱'), rev=('當月營收', '營業收入-當月營收'))


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, month, out = sys.argv[1:]
    if not re.fullmatch(r'\d{4}-\d{2}', month):
        sys.exit('月份要寫 YYYY-MM')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f'{out} 不是空的；請給新的資料夾')
    raw = open(src, 'rb').read(); txt = None
    for enc in ('utf-8-sig', 'big5', 'cp950'):
        try:
            txt = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    if txt is None:
        sys.exit('檔案編碼認不得')
    rows = list(csv.DictReader(io.StringIO(txt)))
    if not rows:
        sys.exit('檔案沒有資料列')
    cols = {k.strip(): k for k in rows[0]}
    col = {}
    for want, names in ALIAS.items():
        hit = [cols[n] for n in names if n in cols]
        if not hit:
            sys.exit(f'找不到欄位 {names}；實際欄位：{list(cols)}')
        col[want] = hit[0]
    roc = f'{int(month[:4]) - 1911}{int(month[5:7])}'; roc2 = f'{int(month[:4]) - 1911}{month[5:7]}'
    for pc in ('資料年月', 'period'):
        if pc in cols:
            vals = {r[cols[pc]].strip() for r in rows} - {''}
            okv = {month, roc, roc2, month.replace('-', '/'), f'{int(month[:4]) - 1911}/{int(month[5:7])}', f'{int(month[:4]) - 1911}/{month[5:7]}'}
            if not vals <= okv:
                sys.exit(f'「{pc}」欄是 {sorted(vals)[:5]}，不是 {month}（民國 {roc2}）；下載錯月份？')
    seen = {}; bad = []
    for r in rows:
        c = r[col['stock_id']].strip()
        v = r[col['rev']].replace(',', '').strip()
        if not re.fullmatch(r'[1-9]\d{3}', c):
            continue
        try:
            float(v) if v else None
        except ValueError:
            bad.append((c, v)); v = ''
        seen[c] = (r[col['name']].strip(), v)
    if bad:
        print(f'注意：{len(bad)} 家營收不是數字，寫成空白（引擎當缺值）：{bad[:10]}')
    if len(seen) < 800:
        sys.exit(f'四碼公司只有 {len(seen)} 家，不像完整的上市彙總表（平常 1,050～1,100 家）；不產生')
    p = out / 'data/mops/revenue_hist' / f'{month}_twse.csv'; p.parent.mkdir(parents=True)
    with open(p, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n'); w.writerow(['stock_id', 'name', 'period', 'market', '當月營收'])
        for c in sorted(seen):
            w.writerow([c, seen[c][0], month, 'twse', seen[c][1]])
    env = dict(os.environ, GIT_AUTHOR_NAME='manual', GIT_AUTHOR_EMAIL='manual@local', GIT_COMMITTER_NAME='manual', GIT_COMMITTER_EMAIL='manual@local')
    for cmd in (['init', '-q'], ['add', '-A'], ['commit', '-q', '-m', f'manual revenue {month}（來源檔 {os.path.basename(src)}）']):
        subprocess.run(['git', '-C', str(out), *cmd], check=True, env=env)
    print(f'已寫 {p}（{len(seen)} 家）。出名單時把 --repo {out} 排在最後一個。先跑：check_revenue_ready.py --month {month} --tsd <官方日K倉庫> --manual-csv {p}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""備援：把官方月營收快照轉成名單產生器讀得懂的小 git 倉庫（接在 eng_audit_fixes_2026-11/manual_revenue_to_repo.py 前面）。

名單產生器只會從 --repo 的 git HEAD 讀 data/mops/revenue_hist/<月>_twse.csv，同月份後面的倉庫覆蓋前面的。
GitHub 倉庫當晚沒更新營收時：先跑 fetch_official.py revenue（1 次請求），再跑這支，把輸出資料夾排在最後一個 --repo。

用法（python3 -I -B；不連網）：
  official_to_repo.py <營收月份 YYYY-MM> <輸出資料夾（新的、空的）> [--snap-root DIR]
做的事：找 index.csv 裡上市（twse）最後一次成功的快照 → 驗 sha256 → 重新解析 → 確認「資料年月」就是要的月份
       （不是就停，結束碼 4：官方端點還沒切到這個月，不能拿上個月頂替）→ 轉成同格式 csv → 呼叫 manual_revenue_to_repo.py 建小倉庫。
另外寫 <輸出資料夾>.source.json：快照檔名、sha256、抓取時間（UTC）、家數——抓取時間就是這份資料的可得時點。
注意：小倉庫沒有 data/meta/revenue_seen，名單產生器的 --rev-cutoff 對這個月不做時點檢查；快照的抓取時間必須早於訊號日截止時間（台北 22:00）。
"""
import argparse, hashlib, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_official as F, ss_lib as L          # noqa: E402
MANUAL = HERE.parent / 'eng_audit_fixes_2026-11' / 'manual_revenue_to_repo.py'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('month'); ap.add_argument('out'); ap.add_argument('--snap-root', default=str(F.SNAP_DEFAULT))
    a = ap.parse_args()
    rec, snap = L.latest_snapshot(a.snap_root, 'revenue', 'twse')
    if rec is None or not snap.exists():
        print('還沒有成功的上市月營收官方快照；先跑 fetch_official.py revenue'); sys.exit(2)
    body = F.gz_read(snap)
    if hashlib.sha256(body).hexdigest() != rec['sha256']:
        print(f'快照 {snap} 的 sha256 和 index.csv 不符，不使用'); sys.exit(2)
    rows, info = F.parse_revenue(body)
    if set(info['periods']) != {a.month}:
        print(f"官方快照（抓取 {rec['fetched_utc']}）的資料年月是 {info['periods']}，不是 {a.month}；官方端點還沒切到這個月，不能當備援。"); sys.exit(4)
    with tempfile.TemporaryDirectory() as td:
        f = F.write_revenue_csv(rows, 'twse', td)[a.month]
        r = subprocess.run([sys.executable, '-I', '-B', str(MANUAL), str(f), a.month, a.out])
    if r.returncode != 0:
        sys.exit(r.returncode)
    src = dict(month=a.month, snapshot=str(snap), sha256=rec['sha256'], fetched_utc=rec['fetched_utc'], url=rec['url'], n_rows=len(rows),
               report_dates=info['report_dates'], note='抓取時間＝這份資料的可得時點；小倉庫沒有首見紀錄，名單產生器對這個月不做 --rev-cutoff 時點檢查')
    json.dump(src, open(str(Path(a.out)) + '.source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"官方快照 {snap.name}（抓取 {rec['fetched_utc']}，出表日期 {info['report_dates']}）→ {a.out}")


if __name__ == '__main__':
    main()

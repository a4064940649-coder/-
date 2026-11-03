#!/usr/bin/env python3
"""訊號日當晚／隔天早上人工核對用（2026-11-03 新增，不在 PREREG 登記；只讀資料、只寫 live_tracking/main_forward/checks/）。

做三件事：
  1. 讀凍結快照 paper_lists/<line>_<YYYYMM>.meta.json：frozen_ts、exec_open_ts、ts_ok、data_head、rev_cutoff。
  2. 用「凍結當時的同一個上游 commit」（meta 的 data_head；或 --commit 指定）git show 取出資料，
     執行 main_picks_today.py --ex-fin（a. 原始資料；b. 處置換成 disposal_5y ∪ warnings_hist，即產生器口徑）。
  3. 比較官方日快照前 10 名與 b（目標 10/10 同順序）；a 與 b 的差＝disposal_5y 過期而漏排的處置股。
用法：
  python3 -I -B check_official.py --signal 2026-11-11            # 官方日（預設 line＝official）
  python3 -I -B check_official.py --signal 2026-11-10 --line j0  # 只看 J0 快照時戳（J 線沒有 main_picks_today 對照）
  python3 -I -B check_official.py --make-union <data 目錄> <輸出目錄> [<官方月營收根目錄>]
      # 只建「處置聯集」資料夾（fallback_manual.sh 用）；給第三個參數時，rev_hist 另外補上「現在之前抓到」的官方月營收（BF-5）
BF-5（2026-11-03）：快照 meta 有 official_fill（產生器用官方月營收補過值）時，main_picks_today 對照也用同一個使用期限
  t_off 之前抓到的官方月營收補值，口徑和產生器相同。
"""
import argparse, json, shutil, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent / 'live_tracking/main_forward'
TZ = 'Asia/Taipei'


def make_union(D, U, official=None, t_cut=None):
    import a2_pit as A
    U = Path(U)
    if U.exists(): shutil.rmtree(U)
    A.union_dir(Path(D).resolve(), U)
    info = None
    if official:                       # BF-5：rev_hist 換成「上游＋官方補值」的實體檔（原本是 symlink）
        t_cut = pd.Timestamp.now(tz=TZ) if t_cut is None else pd.Timestamp(t_cut)
        (U / 'extras/rev_hist.csv.gz').unlink()
        info = A.FD.fill_dir_official(Path(D).resolve(), U / 'extras/rev_hist.csv.gz', official, t_cut)
        (U / 'OFFICIAL_FILL.json').write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str))
    return U, info


def main():
    if sys.argv[1:2] == ['--make-union']:
        U, info = make_union(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
        print(U)
        if info is not None:
            print(f'官方月營收補值：{info["n_filled"]} 列（抓取完成早於 {info["t_off"]}；各月 {info.get("filled_by_ym", {})}；'
                  f'可用的官方列 {info["n_official_rows"]}）', file=sys.stderr)
        return
    import a2_pit as A
    ap = argparse.ArgumentParser()
    ap.add_argument('--signal', required=True)
    ap.add_argument('--line', default='official')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--commit', help='預設＝快照 meta 的 data_head（凍結當時的資料）')
    ap.add_argument('--work', default=str(HERE / 'work/check_official'))
    ap.add_argument('--official-rev', default=str(HERE / 'raw/official_rev'), help='BF-5：官方月營收抓取根目錄（快照有 official_fill 時用）')
    a = ap.parse_args()
    sd = pd.Timestamp(a.signal); ym = f'{sd:%Y%m}'
    out = Path(a.out); pl = out / 'paper_lists'
    meta_p = pl / f'{a.line}_{ym}.meta.json'
    now = pd.Timestamp.now(tz=TZ)
    rep = dict(checked_at=now.isoformat(), signal=str(sd.date()), line=a.line)
    if not meta_p.exists():
        rep['status'] = 'NO_SNAPSHOT'
        print(f'【沒有快照】{meta_p} 不存在：產生器還沒在資料到齊後執行，或資料過期（回傳碼 2）。'
              f'請看 o_forward_infra_live/logs/ 最新一份；訊號日當晚已過 23:00 仍沒有 → 跑 fallback_manual.sh。')
    else:
        mt = json.loads(meta_p.read_text())
        tab = pd.read_csv(pl / f'{a.line}_{ym}.csv', dtype={'code': str}).sort_values('rank')
        ft, ot = pd.Timestamp(mt['frozen_ts']), pd.Timestamp(mt['exec_open_ts'])
        rep.update(frozen_ts=mt['frozen_ts'], exec_open_ts=mt['exec_open_ts'], ts_ok_meta=mt['ts_ok'], ts_ok_recomputed=bool(ft < ot),
                   clock_kind=mt['clock_kind'], wall_utc=mt.get('wall_utc'), data_head=mt['data_head'], rev_cutoff=mt.get('rev_cutoff'),
                   freeze_on_signal_day=mt['freeze_on_signal_day'], snapshot_top10=list(tab.code[:10]))
        f = pl / f'{a.line}_{ym}.csv'
        rep['snapshot_sha256_ok'] = (A.FD.sha(f) == mt['snapshot_sha256'])
        rr = [json.loads(l) for l in (out / 'runs.jsonl').read_text().splitlines() if l.strip()]
        fr = [r for r in rr if f'{a.line}_{ym}' in (r.get('frozen_now') or [])]
        rep['run_that_froze'] = {k: fr[0].get(k) for k in ('clock', 'utc', 'data_head', 'asof', 'status')} if fr else None
        print(f'快照 {a.line}_{ym}：凍結 {ft.tz_convert(TZ):%Y-%m-%d %H:%M:%S}（台北），成交日開盤 {ot.tz_convert(TZ):%Y-%m-%d %H:%M}，'
              f'ts_ok＝{rep["ts_ok_recomputed"]}（meta {mt["ts_ok"]}），clock_kind＝{mt["clock_kind"]}，檔案 sha 一致＝{rep["snapshot_sha256_ok"]}')
        print(f'  資料 head：{mt["data_head"]}；營收截斷日：{mt.get("rev_cutoff")}')
        print(f'  快照前 10 名：{" ".join(tab.code[:10])}')
        if a.line == 'official':
            commit = a.commit or (mt['data_head'].split() or [None])[0]
            if not commit:
                raise SystemExit('快照沒有記錄 data_head（不是從 git 資料夾執行）；請用 --commit 指定凍結當時的上游 commit')
            W = Path(a.work) / f'{ym}_{commit[:8]}'
            if W.exists(): shutil.rmtree(W)
            A.extract(commit, W / 'data')
            of = mt.get('official_fill') or {}
            if of.get('n_filled'):
                _, finfo = make_union(W / 'data', W / 'data_union', a.official_rev, of['t_off'])
                rep['official_fill_check'] = dict(snapshot_n_filled=of['n_filled'], recheck_n_filled=finfo['n_filled'], t_off=of['t_off'])
                print(f'  官方月營收補值：快照 {of["n_filled"]} 列；對照資料夾 {finfo["n_filled"]} 列（t_off {of["t_off"]}）')
            else:
                A.union_dir(W / 'data', W / 'data_union')
            raw, uni = A.mp_cli(W / 'data', str(sd.date())), A.mp_cli(W / 'data_union', str(sd.date()))
            rep.update(commit=commit, mp_raw=raw, mp_union=uni, cmp_union=A.cmp(rep['snapshot_top10'], uni['top10']),
                       cmp_raw=A.cmp(rep['snapshot_top10'], raw['top10']))
            rep['status'] = 'PASS' if rep['cmp_union']['same_order'] and rep['ts_ok_recomputed'] else 'FAIL'
            print(f'  main_picks_today --ex-fin（同一 commit {commit[:8]}，處置聯集）：{" ".join(uni["top10"])}（{uni["section"]}）')
            print(f'    → 與快照重疊 {rep["cmp_union"]["overlap"]}/10、同順序 {rep["cmp_union"]["same_order"]}')
            print(f'  main_picks_today --ex-fin（原始資料，處置只讀 disposal_5y）：{" ".join(raw["top10"])}')
            if raw['top10'] != uni['top10']:
                print(f'    → 原始資料多出 {raw["top10"] and [c for c in raw["top10"] if c not in uni["top10"]]}：'
                      'disposal_5y 過期漏排的處置股，以處置聯集（產生器）為準，並到證交所處置公告核對')
            shutil.rmtree(W, ignore_errors=True)
        else:
            rep['status'] = 'PASS' if rep['ts_ok_recomputed'] else 'FAIL'
        print(f'結果：{rep["status"]}')
    d = out / 'checks'; d.mkdir(exist_ok=True)
    (d / f'{a.line}_{ym}_check_{now:%Y%m%dT%H%M%S}.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()

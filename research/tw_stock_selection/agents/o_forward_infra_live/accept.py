#!/usr/bin/env python3
"""驗收 A1、A2、A4、A5、A6（A3 見 a3_v7x_compare.py）。結果寫 out/accept_<項目>.json。
用法：python3 -I -B accept.py a1|a2|a4|a5|a6|stale  [--out replay/seq]"""
import argparse, json, re, shutil, subprocess, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TW = HERE.parent.parent
sys.path.insert(0, str(HERE))
import fwd_data as FD  # noqa
OUTD = HERE / 'out'; OUTD.mkdir(exist_ok=True)
A1_DAYS = ['2026-09-10', '2026-09-11', '2026-10-12', '2026-10-13']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, res):
    (OUTD / f'accept_{name}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k in ('pass', 'summary')}, ensure_ascii=False, default=str))


def a1(out):
    runs = [json.loads(l) for l in (out / 'runs.jsonl').read_text().splitlines()]
    res = dict(runs=[])
    ok = True
    for r in runs:
        if r.get('status') != 'OK': continue
        asof = pd.Timestamp(r['asof'])
        mx = {k: v for k, v in r['loaded_max_dates'].items() if v and k != 'revenue_max_ym'}
        dx = {k: v for k, v in r['dir_check'].items() if v and not str(v).startswith('ym')}
        bad = [k for k, v in {**mx, **dx}.items() if pd.Timestamp(v) > asof]
        ok &= not bad
        res['runs'].append(dict(asof=r['asof'], clock=r['clock'], frozen_now=r['frozen_now'], max_loaded=max(mx.values()), bad=bad))
    metas = []
    for f in sorted((out / 'paper_lists').glob('*.meta.json')):
        m = json.loads(f.read_text())
        metas.append(dict(file=f.name, signal=m['signal'], freeze_asof=m['freeze_asof'], frozen_ts=m['frozen_ts'], exec_open_ts=m['exec_open_ts'],
                          ts_ok=m['ts_ok'], on_signal_day=m['freeze_on_signal_day'], view_max=max(v for k, v in m['view_max_dates'].items() if v and k != 'revenue_max_ym')))
    M = pd.DataFrame(metas)
    # 成交日實際＝凍結時估計的成交日
    exok = []
    for f in sorted((out / 'paper_lists').glob('*.exec.json')):
        e = json.loads(f.read_text()); m = json.loads(f.with_name(f.name.replace('.exec.json', '.meta.json')).read_text())
        exok.append(e['exec'] == m['exec_day_est'])
    focus = M[M.signal.isin(A1_DAYS)]
    res.update(n_snapshots=len(M), all_ts_ok=bool(M.ts_ok.all()), all_on_signal_day=bool(M.on_signal_day.all()),
               all_view_max_le_signal=bool((M.view_max <= M.signal).all()), exec_day_est_correct=f'{sum(exok)}/{len(exok)}',
               a1_days=focus.to_dict('records'))
    res['pass'] = bool(ok and res['all_ts_ok'] and res['all_on_signal_day'] and res['all_view_max_le_signal'] and all(exok)
                       and set(A1_DAYS) <= set(M.signal))
    res['summary'] = f"{len(M)} 份快照全部在訊號日當晚（模擬 06:49）寫入、早於成交日 09:00；載入資料最大日期 ≤ asof：{ok}；成交日估計正確 {sum(exok)}/{len(exok)}"
    save('a1', res)


def parse_mp(txt):
    sec = txt.split('【前 10 名】')[1].split('【備取')[0]
    return [m.group(2) for m in re.finditer(r'^\s*(\d+)\s+(\d{4})\s', sec, flags=re.M)]


def a2(out):
    rows = []
    src = FD.Source(); pnl = src.panel_pre(HERE / 'work')
    for f in sorted((out / 'paper_lists').glob('official_*.csv')):
        tab = pd.read_csv(f, dtype={'code': str}).sort_values('rank')
        sd = tab.signal.iloc[0]; adir = HERE / 'work' / f'a2_asof_{pd.Timestamp(sd):%Y%m%d}'
        src.build_asof(sd, adir)
        cmd = [sys.executable, '-I', str(TW / 'main_picks_today.py'), '--data', str(adir), '--date', sd, '--n', '10', '--ex-fin', '--panel', str(pnl)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        shutil.rmtree(adir)
        mp = parse_mp(p.stdout)
        gen = list(tab.code[:10])
        rows.append(dict(signal=sd, generator=gen, main_picks_today=mp, same_order=gen == mp, n_match=sum(a == b for a, b in zip(gen, mp))))
    j0 = []
    for f in sorted((out / 'paper_lists').glob('j0_*.meta.json')):
        m = json.loads(f.read_text()); i = m['j0_ann_info']
        j0.append(dict(anchor=m['anchor'], signal=m['signal'], n_ann_missing_pool=i['n_ann_missing_pool'], n_top10_changed=i['n_top10_changed'],
                       top10_new=i['top10_new_rule'], top10_old=i['top10_old_rule'],
                       dropped=[c for c in i['top10_old_rule'] if c not in i['top10_new_rule']], added=[c for c in i['top10_new_rule'] if c not in i['top10_old_rule']]))
    res = dict(official=rows, j0_ann_missing=j0, pass_=None)
    res['pass'] = bool(rows) and all(r['same_order'] for r in rows)
    res['summary'] = '；'.join(f"{r['signal']} {r['n_match']}/10" for r in rows)
    save('a2', res)


def a4(out):
    """同一 asof 重跑：輸出 sha256 相同；資料修訂時快照不覆寫、寫 list_diffs.csv。"""
    runs = [json.loads(l) for l in (out / 'runs.jsonl').read_text().splitlines()]
    last = [r for r in runs if r.get('status') == 'OK'][-1]
    cp = HERE / 'replay/a4_rerun'
    if cp.exists(): shutil.rmtree(cp)
    shutil.copytree(out, cp)
    before = {p.relative_to(cp).as_posix(): sha(p) for p in sorted(cp.rglob('*')) if p.is_file() and p.name != 'runs.jsonl'}
    cmd = [sys.executable, '-I', '-B', str(HERE / 'make_forward.py'), '--replay', '--sim-clock', last['clock'], '--asof', last['asof'],
           '--start', last['months'][0], '--disp-start', last['disp_start'], '--out', str(cp), '--recheck', '5']
    p = subprocess.run(cmd, capture_output=True, text=True); assert p.returncode == 0, p.stderr[-2000:]
    after = {p_.relative_to(cp).as_posix(): sha(p_) for p_ in sorted(cp.rglob('*')) if p_.is_file() and p_.name != 'runs.jsonl'}
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    # 第二次重跑（不同時鐘）也要相同
    cmd2 = list(cmd); cmd2[cmd2.index('--sim-clock') + 1] = '2026-11-03T07:30+08:00'
    p2 = subprocess.run(cmd2, capture_output=True, text=True); assert p2.returncode == 0, p2.stderr[-2000:]
    after2 = {p_.relative_to(cp).as_posix(): sha(p_) for p_ in sorted(cp.rglob('*')) if p_.is_file() and p_.name != 'runs.jsonl'}
    changed2 = sorted(k for k in set(before) | set(after2) if before.get(k) != after2.get(k))
    # 資料修訂測試：把最新凍結官方名單第 1 名的營收年增率改成 −90%，重跑 → 快照不變、list_diffs.csv 有紀錄
    tamper = HERE / 'replay/a4_tamper_src'
    if tamper.exists(): shutil.rmtree(tamper)
    for f in FD.SRC.rglob('*'):                       # 省磁碟：未修改的檔案用符號連結
        if f.is_file():
            t = tamper / f.relative_to(FD.SRC); t.parent.mkdir(parents=True, exist_ok=True)
            if f.name in ('rev_hist.csv.gz', 'rev_5y.csv.gz'): shutil.copyfile(f, t)
            else: t.symlink_to(f)
    offs = sorted((cp / 'paper_lists').glob('official_*.csv'))
    lastoff = offs[-1]; tab = pd.read_csv(lastoff, dtype={'code': str}).sort_values('rank')
    top = tab.code.iloc[0]; ym = tab.rev_month.iloc[0]
    rh = pd.read_csv(tamper / 'extras/rev_hist.csv.gz', dtype={'code': str}); r5 = pd.read_csv(tamper / 'extras/rev_5y.csv.gz', dtype={'code': str})
    for df, fn in ((rh, 'rev_hist'), (r5, 'rev_5y')):
        df.loc[(df.code == top) & (df.ym == ym), 'yoy'] = -90.0
        (tamper / f'extras/{fn}.csv.gz').unlink(); FD.write_csv(df, tamper / f'extras/{fn}.csv.gz')
    snap_sha = sha(lastoff)
    cmd3 = list(cmd) + ['--src', str(tamper), '--recheck', '40']
    p3 = subprocess.run(cmd3, capture_output=True, text=True); assert p3.returncode == 0, p3.stderr[-2000:]
    ld = pd.read_csv(cp / 'list_diffs.csv', dtype=str) if (cp / 'list_diffs.csv').exists() else pd.DataFrame()
    shutil.rmtree(tamper)
    res = dict(n_files=len(before), changed_same_clock=changed, changed_other_clock=changed2, tamper_code=top, tamper_ym=ym,
               snapshot_unchanged=sha(lastoff) == snap_sha, n_list_diffs=len(ld), list_diffs=ld.to_dict('records')[:5])
    res['pass'] = (not changed) and (not changed2) and res['snapshot_unchanged'] and len(ld) > 0
    res['summary'] = f"同 asof 重跑 {len(before)} 個檔案 sha256 不變：{not changed and not changed2}；資料修訂後快照不覆寫：{res['snapshot_unchanged']}，list_diffs {len(ld)} 列"
    save('a4', res)


def a5(out):
    L = pd.read_csv(out / 'disposal_ledger.csv', dtype=str)
    first_run = json.loads((out / 'runs.jsonl').read_text().splitlines()[0])['asof']
    L['announce_d'] = pd.to_datetime(L['announce'])
    new = L[L['announce_d'] >= first_run].copy()
    for c in ('announce_ts', 'entry_deadline_ts', 'first_seen_ts'): new[c] = pd.to_datetime(new[c], utc=True)
    new['chk'] = (new['first_seen_ts'] >= new['announce_ts']) & (new['first_seen_ts'] <= new['entry_deadline_ts'])
    D = pd.read_csv(out / 'disposal_decisions.csv', dtype=str)
    T = pd.read_csv(out / 'trades.csv', dtype={'code': str})
    buys = T[(T.kind == 'paper_disp') & (T.side == 'buy')]
    okmap = dict(zip(L.event_id, L.ts_ok))
    traded_ok = all(okmap.get(e) == '1' for e in buys.event_id)
    by = new.groupby(['market', 'regime']).size().to_dict()
    res = dict(n_events_in_replay=len(new), n_ts_ok=int(new.chk.sum()), by_market_regime={f'{k[0]}_{k[1]}': int(v) for k, v in by.items()},
               n_first_only_twse=int(((new.market == 'TWSE') & (new.chain_take == '1') & (new.second == '0')).sum()),
               n_take_twse=int(((new.market == 'TWSE') & (new.chain_take == '1')).sum()),
               decisions=D.groupby(['line', 'decision']).size().reset_index().rename(columns={0: 'n'}).to_dict('records'),
               all_traded_events_ts_ok=traded_ok, n_buys=len(buys),
               lines_present=sorted(set(D.line)))
    res['pass'] = bool(len(new) and new.chk.all() and traded_ok and {'DISP_TWSE', 'DISP_TWSE_FIRST', 'DISP_TPEX'} <= set(D.line)
                       and any(k.endswith('new5') for k in res['by_market_regime']))
    res['summary'] = f"重播期間公告 {len(new)} 筆，時戳合格 {int(new.chk.sum())}；紙上買進 {len(buys)} 筆全部時戳合格：{traded_ok}"
    save('a5', res)


def a6(out):
    FR = pd.read_csv(out / 'forward_returns.csv')
    T = pd.read_csv(out / 'trades.csv', dtype={'code': str})
    need = ['MAIN_OFFICIAL', 'JMED', 'E50', 'DISP_TWSE', 'MIX_C70', 'MIX_D80']
    have = sorted(set(FR.line))
    tcols = ['fill_px', 'ticks_from_open', 'share_of_open_vol']
    pm = T[T.kind.isin(['paper_whole', 'paper_odd'])]
    chk = pm[['fill_px', 'open_px']].notna().all(axis=1).mean() if len(pm) else 0
    # 紙上整股成交價＝開盤×(1＋K5T 衝擊)：距開盤跳數 ≥ 0（買）
    res = dict(lines=have, missing=[l for l in need if l not in have], trade_cols_ok=all(c in T.columns for c in tcols),
               n_paper_fills=len(pm), frac_with_open=float(chk), ticks_from_open_median_whole=float(pm[pm.kind == 'paper_whole'].ticks_from_open.median()) if len(pm) else None,
               share_open_vol_median=float(pm.share_of_open_vol.median()) if len(pm) else None,
               complete_windows=FR[FR.status == 'complete'].groupby('line').size().to_dict())
    res['pass'] = not res['missing'] and res['trade_cols_ok'] and len(pm) > 0 and chk == 1.0
    res['summary'] = f"forward_returns 線 {len(have)} 條，缺 {res['missing']}；trades 紙上成交 {len(pm)} 筆"
    save('a6', res)


def stale(out):
    """模擬 11-10／11-11：資料尚未更新（來源只到最後交易日）時，產生器必須拒絕凍結、回傳 2。"""
    cp = HERE / 'replay/stale_test'
    if cp.exists(): shutil.rmtree(cp)
    shutil.copytree(out, cp)
    before = sorted(p.name for p in (cp / 'paper_lists').glob('*'))
    rs = []
    for asof, clk in (('2026-11-10', '2026-11-11T06:49+08:00'), ('2026-11-11', '2026-11-12T06:49+08:00')):
        cmd = [sys.executable, '-I', '-B', str(HERE / 'make_forward.py'), '--replay', '--sim-clock', clk, '--asof', asof, '--start', '2026-09', '--out', str(cp)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        rs.append(dict(asof=asof, rc=p.returncode, out=p.stdout.strip()[-400:]))
    after = sorted(p.name for p in (cp / 'paper_lists').glob('*'))
    res = dict(runs=rs, no_new_snapshot=before == after)
    res['pass'] = all(r['rc'] == 2 for r in rs) and res['no_new_snapshot']
    res['summary'] = f"asof 11-10／11-11（資料只到 {FD.Source().hcal[-1].date()}）：回傳碼 {[r['rc'] for r in rs]}，未寫入快照：{res['no_new_snapshot']}"
    save('stale', res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('what'); ap.add_argument('--out', default=str(HERE / 'replay/seq'))
    a = ap.parse_args()
    globals()[a.what](Path(a.out))

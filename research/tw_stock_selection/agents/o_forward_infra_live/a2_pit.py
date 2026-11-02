#!/usr/bin/env python3
"""A2 重做（2026-11-03，DIRECTOR_REPORT o9 第 4 節 3(e)）：用上游 git 歷史的真實快照，不用自己截斷的資料夾。

每個 (訊號日, commit)：
  1. git show <commit>:data/... 取出當時的資料（raw/screener 稀疏 clone，完整歷史；共用 clone 是 shallow，沒有 10 月以前的 commit）。
  2. main_picks_today.py --ex-fin（命令列原樣執行，--date＝訊號日）：
     a. 原始快照（處置只讀 disposal_5y，2026-10-05 起停更）；
     b. 同一快照，但 extras/disposal_5y 換成 disposal_5y ∪ warnings_hist（flag＝處置）——前瞻產生器的處置口徑。
  3. 前瞻產生器 make_forward.py --replay --src <快照> --asof <訊號日> --sim-clock <commit 時間>（新規則 BF-1）。
  4. 舊規則（營收截到訊號日，o9 審查推翻的版本）：同一程式碼 View(rev_cutoff＝None)＋rank_official（in-process）。
比較前 10 名（官方日，成交日開盤資料還不存在 → 未套 M2）。結果寫 out/accept_a2_pit.json；快照在 scratch，跑完刪除。
用法：python3 -I -B a2_pit.py [--scratch DIR] [--keep]
"""
import argparse, json, re, shutil, subprocess, sys, hashlib
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
TW = HERE.parent.parent
REPO = HERE / 'raw/screener'
sys.path.insert(0, str(HERE))
import fwd_data as FD  # noqa: E402
import make_forward as MF  # noqa: E402
FILES = ['history.csv.gz', 'stock_list.csv', 'extras/rev_5y.csv.gz', 'extras/rev_hist.csv.gz', 'extras/disposal_5y.csv.gz',
         'extras/warnings_hist.csv.gz', 'extras/exdiv_5y.csv.gz', 'extras/attention_5y.csv.gz', 'extras/tw_holidays.json', 'extras/updated.json']
# 2026-10-12 訊號日當晚到成交日開盤前的每一個「資料有變」的上游 commit；2026-09 錨定月上游 repo 還不存在（2026-09-29 建立），
# 只能用「MP 所需檔案全部到齊的第一個 commit」b44e5a7（2026-10-08 00:05），這是事後回填，不是 PIT。
CASES = [
    dict(sd='2026-10-12', commit='83f11fc', pit=True, label='10-12 17:40 每日資料'),
    dict(sd='2026-10-12', commit='4531dc6', pit=True, label='10-12 22:38（午夜前最後一個）'),
    dict(sd='2026-10-12', commit='4686735', pit=True, label='10-13 03:18 每日資料重跑'),
    dict(sd='2026-10-12', commit='50bda15', pit=True, label='10-13 03:48（06:49 排程前最後一個；審查用的版本）'),
    dict(sd='2026-10-12', commit='84c7943', pit=True, label='10-13 08:47（成交日開盤前最後一個）'),
    dict(sd='2026-09-11', commit='b44e5a7', pit=False, label='最早可得快照 10-08 00:05（事後回填，非 PIT）', clock='2026-09-14T06:49+08:00'),
]


def git(*a, binary=False):
    r = subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode().strip()


def extract(commit, d):
    d.mkdir(parents=True, exist_ok=True)
    got = {}
    for f in FILES:
        t = d / f; t.parent.mkdir(parents=True, exist_ok=True)
        try:
            t.write_bytes(git('show', f'{commit}:data/{f}', binary=True)); got[f] = FD.sha(t)
        except subprocess.CalledProcessError:
            got[f] = None
    return got


def union_dir(D, U):
    """同一快照，只把 extras/disposal_5y 換成 disposal_5y ∪ warnings_hist（處置），去重規則同 fwd_data.Source.disposal。"""
    (U / 'extras').mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if (D / f).exists() and f != 'extras/disposal_5y.csv.gz':
            (U / f).symlink_to((D / f).resolve())
    d5 = pd.read_csv(D / 'extras/disposal_5y.csv.gz', dtype={'code': str})
    w = pd.read_csv(D / 'extras/warnings_hist.csv.gz', dtype={'code': str})
    w = w[w['flag'] == '處置'][['date', 'code', 'flag', 'start', 'end']]
    u = pd.concat([d5[['date', 'code', 'flag', 'start', 'end']].assign(_s=0), w.assign(_s=1)]).dropna(subset=['start', 'end'])
    u = u.sort_values(['code', 'start', 'end', 'date', '_s']).drop_duplicates(['code', 'start', 'end']).drop(columns='_s')
    FD.write_csv(u.sort_values(['date', 'code', 'start']), U / 'extras/disposal_5y.csv.gz')


def mp_cli(D, sd):
    p = subprocess.run([sys.executable, '-I', str(TW / 'main_picks_today.py'), '--data', str(D), '--date', sd, '--n', '10', '--ex-fin'],
                       capture_output=True, text=True)
    txt = p.stdout
    m = re.search(r'【(前 10 名|M2 實際買進名單[^】]*)】(.*?)\n\n', txt, flags=re.S)
    codes = [x.group(2) for x in re.finditer(r'^\s*(\d+)\s+(\d{4})\s', m.group(2), flags=re.M)] if m else []
    head = [l for l in txt.splitlines()[:12] if l.startswith(('查詢日', '注意', '資料來源', '營收月份', '股池'))]
    return dict(rc=p.returncode, section=m.group(1) if m else None, top10=codes, head=head, err=p.stderr[-800:] if p.returncode else '')


def generator(D, sd, clock, tag, scr):
    out, work = scr / f'out_{tag}', scr / f'work_{tag}'
    m = sd[:7]
    p = subprocess.run([sys.executable, '-I', '-B', str(HERE / 'make_forward.py'), '--replay', '--sim-clock', clock, '--asof', sd,
                        '--src', str(D), '--start', m, '--disp-start', sd, '--out', str(out), '--work', str(work), '--recheck', '1'],
                       capture_output=True, text=True)
    if p.returncode: return dict(rc=p.returncode, err=p.stderr[-1500:])
    mt = json.loads((out / f'paper_lists/official_{m.replace("-", "")}.meta.json').read_text())
    tab = pd.read_csv(out / f'paper_lists/official_{m.replace("-", "")}.csv', dtype={'code': str})
    return dict(rc=0, top10=list(tab.code[:10]), ranks11_15=list(tab.code[10:15]), rev_cutoff=mt.get('rev_cutoff'), ts_ok=mt['ts_ok'],
                frozen_ts=mt['frozen_ts'], exec_open_ts=mt['exec_open_ts'], n_eligible=mt['n_eligible'],
                names={c: n for c, n in zip(tab.code[:15], tab.name[:15])})


def old_rule(D, sd, scr, tag):
    src = FD.Source(D, FD.E50_SRC, FD.TAIEX_SRC, FD.PANEL_SRC)
    v = MF.View(src, pd.Timestamp(sd), scr / f'work_old_{tag}')
    order, _, _ = MF.rank_official(v, pd.Timestamp(sd))
    return list(order.index[:10])


def cmp(a, b):
    return dict(overlap=len(set(a) & set(b)), same_order=a == b, only_left=[c for c in a if c not in b], only_right=[c for c in b if c not in a])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', default='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/a2_pit')
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    scr = Path(a.scratch)
    if scr.exists(): shutil.rmtree(scr)
    res = dict(run_utc=pd.Timestamp.now(tz='UTC').isoformat(), repo_head=git('rev-parse', 'HEAD'),
               make_forward_sha256=FD.sha(HERE / 'make_forward.py'), fwd_data_sha256=FD.sha(HERE / 'fwd_data.py'),
               mp_sha256=FD.sha(TW / 'main_picks_today.py'), cases=[])
    for c in CASES:
        tag = f"{c['sd'].replace('-', '')}_{c['commit']}"
        D = scr / tag / 'data'; U = scr / tag / 'data_union'
        files = extract(c['commit'], D)
        union_dir(D, U)
        ctime = git('log', '-1', '--format=%cI', c['commit'])
        clock = c.get('clock') or pd.Timestamp(ctime).tz_convert('Asia/Taipei').isoformat()
        r = dict(c, commit_time=ctime, sim_clock=clock, files_sha256=files)
        r['mp_raw'] = mp_cli(D, c['sd'])
        r['mp_union'] = mp_cli(U, c['sd'])
        r['gen_new'] = generator(D, c['sd'], clock, tag, scr)
        r['gen_old_rule'] = old_rule(D, c['sd'], scr, tag)
        g = r['gen_new'].get('top10', [])
        r['cmp_gen_new_vs_mp_union'] = cmp(g, r['mp_union']['top10'])
        r['cmp_gen_new_vs_mp_raw'] = cmp(g, r['mp_raw']['top10'])
        r['cmp_gen_old_vs_mp_union'] = cmp(r['gen_old_rule'], r['mp_union']['top10'])
        rh = pd.read_csv(D / 'extras/rev_hist.csv.gz', dtype={'code': str})
        pm1 = str(pd.Period(c['sd'], 'M') - 1)
        rr = rh[rh.ym == pm1]
        r['rev_m1_ann_after_sd'] = int((pd.to_datetime(rr.ann_date, errors='coerce') > pd.Timestamp(c['sd'])).sum())
        res['cases'].append(r)
        print(json.dumps({k: r[k] for k in ('sd', 'commit', 'label', 'cmp_gen_new_vs_mp_union', 'cmp_gen_new_vs_mp_raw', 'cmp_gen_old_vs_mp_union')},
                         ensure_ascii=False), flush=True)
        if not a.keep: shutil.rmtree(scr / tag, ignore_errors=True)
        for p in scr.glob(f'work*_{tag}'): shutil.rmtree(p, ignore_errors=True)
    pit = [r for r in res['cases'] if r['pit']]
    res['summary'] = dict(
        pit_cases=len(pit),
        gen_new_vs_mp_union_10of10_same_order=sum(r['cmp_gen_new_vs_mp_union']['same_order'] for r in pit),
        gen_old_vs_mp_union_overlap=[r['cmp_gen_old_vs_mp_union']['overlap'] for r in pit],
        gen_new_vs_mp_raw_overlap=[r['cmp_gen_new_vs_mp_raw']['overlap'] for r in pit])
    res['pass'] = bool(pit) and all(r['cmp_gen_new_vs_mp_union']['same_order'] for r in pit)
    (HERE / 'out/accept_a2_pit.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res['summary'], ensure_ascii=False), 'pass', res['pass'])
    if not a.keep: shutil.rmtree(scr, ignore_errors=True)


if __name__ == '__main__':
    main()

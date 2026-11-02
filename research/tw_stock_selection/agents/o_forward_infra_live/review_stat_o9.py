#!/usr/bin/env python3
"""反方審查（資料與統計，o9）：用上游 git 歷史的真實 PIT 快照，檢驗截斷重播是否等於「當時真的看得到的資料」。
只讀研究員檔案；下載的快照放 scratchpad，跑完刪除。用法：python3 -I -B review_stat_o9.py
"""
import sys, subprocess, shutil, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TW = HERE.parent.parent
sys.path.insert(0, str(TW)); import main_picks_today as MP; sys.path.pop(0)
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_o9_pit')
REPO = 'https://raw.githubusercontent.com/stevenhsiehsheng-ctrl/tw-stock-screener'
FILES = ['history.csv.gz', 'stock_list.csv', 'extras/rev_5y.csv.gz', 'extras/rev_hist.csv.gz', 'extras/disposal_5y.csv.gz',
         'extras/exdiv_5y.csv.gz', 'extras/attention_5y.csv.gz', 'extras/warnings_hist.csv.gz']
CUR = HERE / 'raw/screener/data'


def fetch(commit, d):
    d = Path(d)
    for f in FILES:
        t = d / f; t.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(['curl', '-sS', '-f', '-o', str(t), f'{REPO}/{commit}/data/{f}'], capture_output=True, text=True)
        if r.returncode != 0: print('fetch fail', f, r.stderr.strip()); t.unlink(missing_ok=True)
    return d


def union_disp(D, asof):
    d5 = pd.read_csv(D / 'extras/disposal_5y.csv.gz', dtype={'code': str})
    p = D / 'extras/warnings_hist.csv.gz'
    w = pd.read_csv(p, dtype={'code': str}) if p.exists() else d5.iloc[:0]
    w = w[w['flag'] == '處置'][['date', 'code', 'flag', 'start', 'end']]
    u = pd.concat([d5[['date', 'code', 'flag', 'start', 'end']], w]).dropna(subset=['start', 'end'])
    u = u.sort_values(['code', 'start', 'end', 'date']).drop_duplicates(['code', 'start', 'end'])
    u = u[pd.to_datetime(u['date']) <= asof]
    for c in ('date', 'start', 'end'): u[c] = pd.to_datetime(u[c])
    return u


def lists(D, sd):
    data = MP.load_prices(D, sd, MP.PANEL)
    _, Y3, _ = MP.load_revenue(D)
    d5 = MP.load_disposal(D)
    o_sop, df, m, n = MP.score_on(data, Y3, d5, sd, ex_fin=True)
    o_uni, _, _, n2 = MP.score_on(data, Y3, union_disp(D, sd), sd, ex_fin=True)
    return list(o_sop.index[:10]), list(o_uni.index[:10]), n, n2, data


def main():
    rep = {}
    sd = pd.Timestamp('2026-10-12')
    # 10-13 06:49 台北之前最後一個上游 commit（git log --until=2026-10-12T22:49Z -1）
    commit = '50bda15'
    if SCR.exists(): shutil.rmtree(SCR)
    D = fetch(commit, SCR / commit)
    sop, uni, n_sop, n_uni, data_pit = lists(D, sd)
    replay = json.loads((HERE / 'replay/seq/paper_lists/official_202610.meta.json').read_text())['top10']
    rep['pit_1012'] = dict(commit=commit, top10_disposal5y_only=sop, top10_union=uni, n_el=[n_sop, n_uni], replay_official=replay,
                           union_eq_replay_order=uni == replay, union_vs_replay_overlap=len(set(uni) & set(replay)),
                           has_2466_6213_in_sop=[c in sop for c in ('2466', '6213')])
    # 營收公告日：快照當時 vs 現在
    rh_p = pd.read_csv(D / 'extras/rev_hist.csv.gz', dtype={'code': str}); rh_c = pd.read_csv(CUR / 'extras/rev_hist.csv.gz', dtype={'code': str})
    r5_p = pd.read_csv(D / 'extras/rev_5y.csv.gz', dtype={'code': str})
    sp = set(r5_p[r5_p.ym == '2026-09'].code) | set(rh_p[rh_p.ym == '2026-09'].code)
    cc = rh_c[(rh_c.ym == '2026-09')].copy(); cc['ann_date'] = pd.to_datetime(cc['ann_date'], errors='coerce')
    cur_le = set(cc[cc.ann_date <= sd].code)
    rep['rev_2026_09_at_1012'] = dict(n_present_in_pit_snapshot=len(sp), n_current_ann_le_1012=len(cur_le),
                                      current_says_announced_but_absent_in_pit=len(cur_le - sp), present_in_pit_but_current_ann_after=len(sp - cur_le),
                                      examples_absent=sorted(cur_le - sp)[:10], examples_late=sorted(sp - cur_le)[:10])
    # 價格修訂：日期 ≤ 10-12 的列
    hp = pd.read_csv(D / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    hc = pd.read_csv(CUR / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    lo = max(hp.date.min(), hc.date.min())
    a = hp[(hp.date >= lo) & (hp.date <= sd)]; b = hc[(hc.date >= lo) & (hc.date <= sd)]
    mg = a.merge(b, on=['date', 'code'], how='outer', suffixes=('_p', '_c'), indicator=True)
    num = [c for c in hp.columns if c not in ('date', 'code') and pd.api.types.is_numeric_dtype(hp[c])]
    both = mg[mg._merge == 'both']
    rev = {c: int((~np.isclose(both[c + '_p'], both[c + '_c'], equal_nan=True)).sum()) for c in num if c + '_c' in both}
    rep['history_le_1012'] = dict(pit_start=str(hp.date.min().date()), cur_start=str(hc.date.min().date()), rows_pit=len(a), rows_cur=len(b),
                                  only_pit=int((mg._merge == 'left_only').sum()), only_cur=int((mg._merge == 'right_only').sum()), revised_cells=rev,
                                  codes_only_pit=sorted(set(mg[mg._merge == 'left_only'].code))[:15])
    # 處置：10-07 公告在 10-12 晚上的快照裡看得到嗎
    w = pd.read_csv(D / 'extras/warnings_hist.csv.gz', dtype={'code': str})
    rep['warnings_hist_pit_2466_6213'] = w[(w.code.isin(['2466', '6213'])) & (w.flag == '處置') & (w.date >= '2026-10-01')].to_dict('records')
    shutil.rmtree(SCR / commit)
    # 9 月重播：上游 repo 的第一個 commit
    r = subprocess.run(['git', '-C', str(HERE / 'raw/screener'), 'log', '--reverse', '--format=%h %cI %s'], capture_output=True, text=True)
    rep['upstream_first_commits'] = r.stdout.splitlines()[:3]
    r = subprocess.run(['git', '-C', str(HERE / 'raw/screener'), 'log', '--diff-filter=A', '--format=%h %cI', '--', 'data/extras/warnings_hist.csv.gz'],
                       capture_output=True, text=True)
    rep['warnings_hist_created'] = r.stdout.strip()
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


def ann_pit():
    """10-12 晚上快照裡 2026-09 營收的 ann_date 狀態 vs 現在；以及 PIT 與重播官方名單差異的來源。"""
    D = SCR / 'ann'; D.mkdir(parents=True, exist_ok=True)
    t = D / 'rev_hist.csv.gz'
    subprocess.run(['curl', '-sS', '-f', '-o', str(t), f'{REPO}/50bda15/data/extras/rev_hist.csv.gz'], check=True)
    p = pd.read_csv(t, dtype={'code': str}); c = pd.read_csv(CUR / 'extras/rev_hist.csv.gz', dtype={'code': str})
    p = p[p.ym == '2026-09'][['code', 'ann_date']]; c = c[c.ym == '2026-09'][['code', 'ann_date']]
    m = p.merge(c, on='code', how='outer', suffixes=('_pit', '_cur'))
    late = m[pd.to_datetime(m.ann_date_cur, errors='coerce') > '2026-10-12']
    out = dict(n_pit_rows=len(p), pit_ann_nonnull=int(p.ann_date.notna().sum()), pit_ann_max=str(p.ann_date.max()),
               n_cur_after_1012=len(late), late_pit_ann_values=late.ann_date_pit.fillna('NaN').value_counts().head(5).to_dict(),
               late_cur_ann_values=late.ann_date_cur.value_counts().head(5).to_dict(),
               changed=int((m.ann_date_pit.notna() & m.ann_date_cur.notna() & (m.ann_date_pit != m.ann_date_cur)).sum()))
    for code in ('2395', '6196', '6691'):
        out[f'code_{code}'] = m[m.code == code].to_dict('records')
    shutil.rmtree(D)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    ann_pit() if sys.argv[1:] == ['ann'] else main()

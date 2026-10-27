#!/usr/bin/env python3
"""rev_announce 第 1 步：營收公告時點資料。
- tw-stock-screener（scratchpad 的 blobless clone，只讀）git 歷史：每檔 2026-09 營收第一次被抓到的 commit 時間。
- 各產業在訊號日（2026-10-12）收盤前未公告的比例、10 日後才公告的比例；前 300 名股池的情形。
- 2006～2026 每個營收月的法定期限 D 與訊號日 S（10 日遇假日 → D＝S）。
- 產生 B 版用的延後機率 p（依 PREREG）。輸出 results.json 的 ANN、firstseen_2026-09.csv。
"""
import json, hashlib, subprocess, io, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / 'revenue_longhistory'))
import lhlib as L  # noqa: E402

CLONE = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/ra_clone/scr')
SCR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
BN = Path('/home/user/niklavsd/bottleneck-signals')
TZ = 'Asia/Taipei'
INS15 = ['2880', '2881', '2882', '2883', '2885', '2886', '2887', '2891', '2892', '5880', '2816', '2832', '2850', '2851', '2852', '2867', '2905', '2207']

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256']


def git(*a, repo=CLONE, binary=False):
    r = subprocess.run(['git', '-C', str(repo), *a], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode()


def fin_set(codes, sl):
    return set(sl.loc[sl['industry'] == '金融保險業', 'code']) | {c for c in codes if c[:2] == '28' or c in ('5876', '5880')}


def main():
    out = {}
    sl = pd.read_csv(SCR / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
    # ---------- 1. git 歷史：第一次看到 2026-09 營收 ----------
    log = [l.split() for l in git('log', '--reverse', '--format=%H %cI', '--', 'data/extras/rev_hist.csv.gz').strip().splitlines()]
    first = {}; commits = []
    for h, t in log:
        raw = git('show', f'{h}:data/extras/rev_hist.csv.gz', binary=True)
        d = pd.read_csv(io.BytesIO(raw), compression='gzip', dtype=str)
        s = set(d.loc[d['ym'] == '2026-09', 'code'])
        ts = pd.Timestamp(t).tz_convert(TZ)
        commits.append({'commit': h[:7], 'time_taipei': str(ts), 'n_2026_09': len(s), 'has_ann_col': 'ann_date' in d.columns})
        for c in s:
            first.setdefault(c, ts)
    fin = pd.read_csv(SCR / 'extras' / 'rev_hist.csv.gz', dtype=str)
    fin = fin[fin['ym'] == '2026-09'].set_index('code')
    fs = pd.DataFrame({'first_seen': pd.Series(first)})
    fs['ann_date'] = fin['ann_date'].reindex(fs.index)
    fs['industry'] = sl['industry'].reindex(fs.index).fillna('（stock_list 無）')
    fs['market'] = sl['market'].reindex(fs.index)
    sig_close = pd.Timestamp('2026-10-12 13:30', tz=TZ)
    amb_end = pd.Timestamp('2026-10-12 15:40', tz=TZ)
    fs['cat'] = np.where(fs['first_seen'] < sig_close, 'A_before', np.where(fs['first_seen'] <= amb_end, 'B_ambiguous', 'C_after'))
    fs['after10'] = fs['ann_date'].fillna('2026-10-01') > '2026-10-10'
    fs['ins15'] = fs.index.isin(INS15) | (fs.index == '2884')
    FIN = fin_set(list(fs.index) + list(L.COLS), sl.reset_index())
    fs['fin'] = fs.index.isin(FIN)
    fs.sort_values('first_seen').to_csv(HERE / 'firstseen_2026-09.csv', encoding='utf-8')
    # ann_date 與 git 第一次出現日的一致性
    lab = fs.dropna(subset=['ann_date'])
    out['ann_date_vs_git'] = {'n_with_ann': int(len(lab)),
                              'ann_eq_taipei_date_of_first_commit': round(float((lab['ann_date'] == lab['first_seen'].dt.strftime('%Y-%m-%d')).mean()), 4),
                              'n_present_before_ann_field': int(fs['ann_date'].isna().sum())}
    out['commits_rev_hist'] = commits
    out['first_seen_distribution_taipei_date'] = fs['first_seen'].dt.strftime('%m-%d').value_counts().sort_index().to_dict()
    # ---------- 2. 產業統計 ----------
    def tab(df):
        g = df.groupby('industry')
        t = pd.DataFrame({'n': g.size(), 'share_after_10th': g['after10'].mean().round(3),
                          'share_not_before_signal_close': (g['cat'].apply(lambda s: (s != 'A_before').mean())).round(3),
                          'share_C_certain_after': (g['cat'].apply(lambda s: (s == 'C_after').mean())).round(3)})
        return t.sort_values('share_not_before_signal_close', ascending=False)
    t_all = tab(fs)
    out['by_industry_all'] = t_all.to_dict(orient='index')
    out['overall'] = {'n': int(len(fs)), 'share_after_10th': round(float(fs['after10'].mean()), 3),
                      'share_not_before_signal_close': round(float((fs['cat'] != 'A_before').mean()), 3),
                      'share_C_certain_after': round(float((fs['cat'] == 'C_after').mean()), 3),
                      'nonfin': {'share_not_before_signal_close': round(float((fs.loc[~fs.fin, 'cat'] != 'A_before').mean()), 3),
                                 'share_after_10th': round(float(fs.loc[~fs.fin, 'after10'].mean()), 3)},
                      'fin': {'n': int(fs.fin.sum()), 'share_not_before_signal_close': round(float((fs.loc[fs.fin, 'cat'] != 'A_before').mean()), 3),
                              'share_after_10th': round(float(fs.loc[fs.fin, 'after10'].mean()), 3)},
                      'fin_ins15': {'n': int((fs.fin & fs.ins15).sum()), 'share_not_before_signal_close': round(float((fs.loc[fs.fin & fs.ins15, 'cat'] != 'A_before').mean()), 3)},
                      'fin_10day_rule': {'n': int((fs.fin & ~fs.ins15).sum()), 'share_not_before_signal_close': round(float((fs.loc[fs.fin & ~fs.ins15, 'cat'] != 'A_before').mean()), 3)}}
    # 前 300 名股池（2026-10-12 訊號日）
    Y = L.yoy_long(); S = L.Setup(Y, start='2005-01-01')
    d0 = pd.Timestamp('2026-10-12'); assert d0 in S.DATES
    el = (S.t300 & S.s52.notna() & ~S.inD & S.yoy3.notna()).loc[d0]
    pool = list(el[el].index)
    fp = fs.reindex(pool)
    missing = [c for c in pool if c not in fs.index]
    fp = fp.dropna(subset=['first_seen'])
    out['pool300_2026-10-12'] = {'n_pool': len(pool), 'n_in_rev_hist': int(len(fp)), 'missing_in_rev_hist': missing,
                                 'n_not_before_close': int((fp['cat'] != 'A_before').sum()),
                                 'n_B_ambiguous': int((fp['cat'] == 'B_ambiguous').sum()), 'n_C_after': int((fp['cat'] == 'C_after').sum()),
                                 'not_before_close_codes': {c: [str(fp.loc[c, 'industry']), str(fp.loc[c, 'first_seen'])] for c in fp.index[fp['cat'] != 'A_before']},
                                 'by_industry': tab(fp).to_dict(orient='index')}
    # ---------- 3. 延後機率 p（PREREG） ----------
    est = fs[~fs.ins15 & ~fs.fin]
    p_all = float((est['cat'] != 'A_before').mean())
    g = est.groupby('industry')['cat']
    p_ind = {k: (float((v != 'A_before').mean()) if len(v) >= 15 else p_all) for k, v in g}
    p_ind['金融保險業'] = 1.0
    out['p_delay'] = {'p_overall_nonfin_non15': round(p_all, 4), 'p_by_industry': {k: round(v, 4) for k, v in sorted(p_ind.items(), key=lambda x: -x[1])},
                      'n_by_industry': {k: int(len(v)) for k, v in g}}
    # ---------- 4. 法定期限 vs 訊號日（全期） ----------
    T = pd.DatetimeIndex(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']).sort_values()
    def first_td(ts):
        k = T.searchsorted(ts); return T[k] if k < len(T) else pd.NaT
    rows = []
    for d in S.dates:
        m = pd.Period(d, 'M') - 1
        if m < pd.Period('2006-01', 'M'): continue
        ten = pd.Timestamp(d.year, d.month, 10)
        D = first_td(ten); D15 = first_td(pd.Timestamp(d.year, d.month, 15))
        special = {'2019-01': '延至 2019-02-15（春節）', '2023-03': '延至 2023-04-12（清明連假）', '2020-01': '受疫情影響公司可延至 2020-02-15'}.get(str(m))
        if str(m) in ('2019-01', '2023-03'): case = 'D>S'
        elif str(m) == '2020-01': case = 'D=S'
        else: case = 'D<S' if D < d else ('D=S' if D == d else 'D>S')
        rows.append({'rev_month': str(m), 'signal_day': str(d.date()), 'tenth_weekday': ten.day_name(), 'tenth_is_trading': bool(ten in T),
                     'deadline_D': str(D.date()), 'case': case, 'D15': str(D15.date()), 'ins15_case': ('D>S' if D15 > d else ('D=S' if D15 == d else 'D<S')) if m >= pd.Period('2026-01', 'M') else None,
                     'special': special})
    cal = pd.DataFrame(rows)
    cal.to_csv(HERE / 'deadline_calendar.csv', index=False, encoding='utf-8')
    out['calendar_summary'] = {'n_months': int(len(cal)), 'case_counts': cal['case'].value_counts().to_dict(),
                               'share_D_eq_S': round(float((cal['case'] == 'D=S').mean()), 3),
                               'D_eq_S_by_signal_month': cal[cal['case'] == 'D=S']['signal_day'].str[5:7].value_counts().sort_index().to_dict(),
                               'tenth_weekday_when_D_eq_S': cal[cal['case'] == 'D=S']['tenth_weekday'].value_counts().to_dict(),
                               'P1_2006_2020_share_D_eq_S': round(float((cal[cal.signal_day < '2021-01-01']['case'] == 'D=S').mean()), 3),
                               'P2_2021_2026_share_D_eq_S': round(float((cal[cal.signal_day >= '2021-01-01']['case'] == 'D=S').mean()), 3),
                               'ins15_2026': cal.dropna(subset=['ins15_case'])[['rev_month', 'signal_day', 'D15', 'ins15_case']].to_dict(orient='records')}
    # ---------- 5. bottleneck-signals git 歷史 ----------
    bl = [l.split() for l in git('log', '--format=%h %cI', '--', 'data/processed/tw_revenue.csv', repo=BN).strip().splitlines()]
    bo = []
    for h, t in bl:
        raw = git('show', f'{h}:data/processed/tw_revenue.csv', repo=BN, binary=True)
        d = pd.read_csv(io.BytesIO(raw), dtype=str, usecols=['month'])
        bo.append({'commit': h, 'time': t, 'last_month': d['month'].max(), 'n_last_month': int((d['month'] == d['month'].max()).sum())})
    out['bottleneck_history'] = {'is_shallow': git('rev-parse', '--is-shallow-repository', repo=BN).strip(),
                                 'n_commits_all': int(git('rev-list', '--count', '--all', repo=BN).strip()), 'tw_revenue_commits': bo}
    res['ANN'] = out
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    (HERE / 'p_delay.json').write_text(json.dumps({'p_by_industry': p_ind, 'p_overall': p_all, 'INS15': INS15}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('ann_date_vs_git', 'overall', 'calendar_summary')}, ensure_ascii=False, indent=1, default=str)[:6000])
    print(t_all.to_string())
    print(json.dumps(out['pool300_2026-10-12'], ensure_ascii=False, default=str)[:3000])
    print(json.dumps(out['p_delay'], ensure_ascii=False)[:2000])
    print(bo)

if __name__ == '__main__':
    main()

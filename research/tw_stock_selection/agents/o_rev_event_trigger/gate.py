#!/usr/bin/env python3
"""步驟 1：資料閘門（看任何報酬之前）。
覆蓋率＝2011～2018 年 V4 官方訊號月的前 300 流動性池（排除金融 fin2、KY／DR）股月中，
能取得 PIT 合法、精確到日的逐檔公告時間戳的比例。門檻 ≥ 80%（扣掉來源本身缺口後的上界）。
來源嘗試：
  (a) 鉅亨網新聞（o_v4 凍結的 raw_news_mentions.parquet，文章日期＝公告時間上限；原始 clone 在 o_v4 raw/ctbc，2013-10～2021-09）
  (b) web.archive.org MOPS t21sc03 快照、(c) MOPS／重大訊息封存：本 session 連線測試結果寫進 raw/access_log.json
  (d) rev_announce deadline_calendar.csv／p_delay.json：法定期限與機率，不是逐檔時間戳
  (e) tw-stock-screener rev_hist ann_date：只有 2026-09（＋6 筆 2026-08），與 2011～2018 沒有重疊
輸出：data/ann_news.csv（逐檔×營收月的最早新聞日期、嚴格同日旗標），gate.json。不讀任何報酬。
"""
import sys, json, re, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import fx_core as F
L, K, V = F.L, F.K, F.V

st = F.setup()
fz = st['fz']
NEWS = F.V4DIR / 'raw_news_mentions.parquet'
assert F.sha(NEWS) == fz['this_topic_files']['raw_news_mentions.parquet']['sha256'], 'raw_news_mentions 與 V4 凍結不符'
M = pd.read_parquet(NEWS)
M['date'] = pd.to_datetime(M['date'])
# 營收月 ym 的新聞必須在 ym 的次月（公告月）內；同日嚴格旗標：片段含「今(X)日／今日／今天」且 X＝文章日
def strict_flag(snip, day):
    m = re.search(r'今\s*[\(（]\s*(\d{1,2})\s*[\)）]\s*日', snip)
    if m: return int(m.group(1)) == day
    return bool(re.search(r'今日|今天|今\s*\(\s*\)\s*日', snip))
M['ann_month'] = (pd.PeriodIndex(M['ym'], freq='M') + 1).astype(str)
M = M[M['date'].dt.to_period('M').astype(str) == M['ann_month']].copy()
M['strict'] = [strict_flag(s, d.day) for s, d in zip(M['snip'], M['date'])]
g = M.groupby(['code', 'ym'])
A = pd.DataFrame({'first_news': g['date'].min(),
                  'first_strict': M[M.strict].groupby(['code', 'ym'])['date'].min(),
                  'n_mentions': g.size()}).reset_index().rename(columns={'ym': 'rev_month'})
A = A.sort_values(['rev_month', 'code']).reset_index(drop=True)
(F.DATA).mkdir(exist_ok=True)
A.to_csv(F.DATA / 'ann_news.csv', index=False, date_format='%Y-%m-%d')

# ---------------- 池內股月（2011～2018 官方訊號月）----------------
S0 = L.V4(st['YFX'], st['Y5F'], st['DMT'])
rows = []
for d in S0.sig_days:
    if not (pd.Timestamp('2011-01-01') <= d <= pd.Timestamp('2018-12-31')): continue
    cs = S0.cross_section(d)
    pool = cs['top'] & ~L.FIN2MASK & ~L.FORMASK
    m = pd.Period(d, 'M'); rm = str(m - 1); D = F.deadline(m)
    for j in np.where(pool)[0]:
        rows.append((str(d.date()), rm, K.COLS[j], D))
PM = pd.DataFrame(rows, columns=['signal', 'rev_month', 'code', 'deadline'])
key = A.set_index(['code', 'rev_month'])
PM = PM.join(key, on=['code', 'rev_month'])
# 「精確到日」：嚴格同日旗標（片段寫「今(X)日」且 X＝文章日）；寬鬆：任何公告月內、且不晚於期限後 2 個交易日的新聞
def td_after(D, k): return F.TCAL[F.TCAL.get_loc(D) + k]
PM['lim2'] = [td_after(D, 2) for D in PM['deadline']]
PM['cov_strict'] = PM['first_strict'].notna() & (PM['first_strict'] <= PM['lim2'])
PM['cov_loose'] = PM['first_news'].notna() & (PM['first_news'] <= PM['lim2'])
SRC_MONTHS = set(A.groupby('rev_month').size().loc[lambda x: x >= 20].index)   # 來源本身有資料的營收月（該月 ≥ 20 檔有營收新聞）
PM['src_window'] = PM['rev_month'].isin(SRC_MONTHS)
PM['year'] = PM['signal'].str[:4]
by_year = PM.groupby('year').agg(n=('code', 'size'), strict=('cov_strict', 'mean'), loose=('cov_loose', 'mean'), src=('src_window', 'mean')).round(4)
cov = dict(n_stock_months=int(len(PM)), strict=float(PM.cov_strict.mean()), loose=float(PM.cov_loose.mean()),
           upper_bound_after_source_gap=float(PM.src_window.mean()),
           loose_within_source_window=float(PM[PM.src_window].cov_loose.mean()),
           strict_within_source_window=float(PM[PM.src_window].cov_strict.mean()))

# ---------------- 其他來源的連線測試（不可信資料不下載；只記狀態碼）----------------
urls = {'web.archive.org': 'https://web.archive.org/web/2015*/mops.twse.com.tw/nas/t21/sii/t21sc03_104_1_0.html',
        'archive.org_api': 'https://archive.org/wayback/available?url=mops.twse.com.tw/nas/t21/sii/t21sc03_104_1_0.html',
        'mops': 'https://mops.twse.com.tw/nas/t21/sii/t21sc03_104_1_0.html', 'mopsov': 'https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_104_1_0.html',
        'finmind': 'https://api.finmindtrade.com/api/v4/data', 'twse': 'https://www.twse.com.tw/'}
acc = {}
for k, u in urls.items():
    try:
        r = subprocess.run(['curl', '-sS', '-m', '20', '-o', '/dev/null', '-w', '%{http_code}', u], capture_output=True, text=True, timeout=40)
        acc[k] = {'url': u, 'http_code': r.stdout.strip(), 'stderr': r.stderr.strip()[:200]}
    except Exception as e:
        acc[k] = {'url': u, 'error': str(e)[:200]}
acc['github_code_search'] = {'queries': ['"營收公告日" OR "營收發布日" extension:csv', 't21sc03 出表日期', 'repo search: monthly revenue announcement date taiwan mops'],
                             'result': '0 筆資料檔；只找到即時爬蟲程式（抓的是更正後現值，沒有歷史公告時間），沒有 2011～2018 的逐檔公告時間戳封存'}
(HERE / 'raw').mkdir(exist_ok=True)
(HERE / 'raw' / 'access_log.json').write_text(json.dumps(acc, ensure_ascii=False, indent=1), encoding='utf-8')

# ---------------- (e) 2026 ann_date 精度校準：與新聞沒有重疊期 ----------------
rh = pd.read_csv(K.X / 'rev_hist.csv.gz', dtype={'code': str})
ann_months = sorted(rh.loc[rh['ann_date'].notna(), 'ym'].unique().tolist()) if 'ann_date' in rh.columns else []
precision = {'rev_hist_ann_date_months': ann_months, 'overlap_with_news_months': sorted(set(ann_months) & set(A.rev_month)),
             'agreement_rate': None, 'note': '新聞只到 2021-09、ann_date 只有 2026-08／09，重疊 0 個月，無法計算一致率',
             'news_internal': {'share_first_news_equals_first_strict': float((A.first_news == A.first_strict)[A.first_strict.notna()].mean()),
                               'n_with_strict': int(A.first_strict.notna().sum()), 'n_code_months': int(len(A))}}

gate_pass_cov = cov['strict'] >= 0.80 and cov['upper_bound_after_source_gap'] >= 0.80
out = {'coverage': cov, 'by_year': by_year.reset_index().to_dict('records'), 'precision': precision, 'access': acc,
       'threshold': 0.80, 'gate_coverage_pass': bool(gate_pass_cov),
       'gate_breakeven_part': '覆蓋閘門未過，不計算（需要報酬）' if not gate_pass_cov else '待算',
       'decision': '閘門不通過 → 改做後備題「J0／J+1 固定換股日正式化與前視清理」' if not gate_pass_cov else '閘門通過',
       'files': {'data/ann_news.csv': F.sha(F.DATA / 'ann_news.csv'), 'raw/access_log.json': F.sha(HERE / 'raw' / 'access_log.json')},
       'inputs': {'raw_news_mentions.parquet': F.sha(NEWS)}}
(HERE / 'gate.json').write_text(json.dumps(F.jsonable(out), ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(F.jsonable({'coverage': cov, 'by_year': out['by_year'], 'gate': out['decision']}), ensure_ascii=False, indent=1))

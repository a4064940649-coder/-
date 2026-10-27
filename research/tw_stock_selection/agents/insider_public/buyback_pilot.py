#!/usr/bin/env python3
"""庫藏股（董事會決議買回）公告事件研究——小樣本試做。

資料（唯讀，只讀 CSV）：/home/user/shinbutou/nccu_bigdata/databases/detailed_info/<code>_<ROC年>.csv
  政大課程專案爬下的 MOPS 重大訊息全文，59 檔大型股 × 民國 106–109 年（2017–2020）。不是全市場。

事前固定：
* 事件：主旨含「董事會…決議/通過…買回」，排除子公司、證券化、債券、減資、註銷、特別股、訴訟、更正、補充；同一公司同一天只算一次。
* 進場：發言日期之後第一個交易日開盤（不論盤中或盤後公告，一律隔日開盤）；開盤漲停視為買不到。
* 持有 5／20／60 交易日；AR = 報酬 − 0.5% − 同日進場上市流動池等權報酬（池：60 日均值 ≥ 1,000 萬、價 ≥ 10、4 碼普通股）。
* t：按進場月份分群。挑選期 2017–2018、驗證期 2019–2020（資料長度切半）。
* 執行率：解析決議公告第 11 項「申報前三年內買回公司股份之情形」中的「執行情形(%)」，以及「期間屆滿／執行完畢」公告的預定 vs 已買回股數。
"""
import sys, json, math, glob, re
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'exec_rebuild'))
import exlib as L  # noqa: E402

SRC = '/home/user/shinbutou/nccu_bigdata/databases/detailed_info'
POS = re.compile(r'董事會.*(決議|通過).*買回|決議買回')
NEG = re.compile('子公司|證券化|債|減資|註銷|特別股|訴訟|更正|補充')

def roc(s):
    y, m, d = [int(x) for x in str(s).strip().split('/')]
    return pd.Timestamp(y + 1911, m, d)

def num(pat, text):
    m = re.search(pat, text)
    return float(m.group(1).replace(',', '')) if m else None

events, past_exec, done = [], [], []
for f in sorted(glob.glob(f'{SRC}/*_1*.csv')):
    code = Path(f).name.split('_')[0]
    d = pd.read_csv(f)
    for _, r in d.iterrows():
        s = str(r['主旨']).replace('\n', ''); t = str(r['說明'])
        if POS.search(s) and not NEG.search(s):
            tt = t.replace('\n', '')
            events.append({'code': code, 'date': roc(r['發言日期']), 'time': str(r['發言時間']).strip(), 'subject': s[:40],
                           'purpose': (re.search(r'買回股份目的:([^0-9]{2,20}?)\d\.', tt) or [None, None])[1],
                           'planned_shares': num(r'預定買回之數量\(股\):([\d,]+)', tt),
                           'planned_pct': num(r'占公司已發行股份總數之比率\(%\):([\d.]+)', tt)})
            for m in re.finditer(r'預定買回股數\(股\)[:：]\s*([\d,]+).{0,40}?實際已\s*買回股數\(股\)[:：]\s*([\d,]+).{0,60}?%\)[:：]\s*([\d.]+)', tt):
                past_exec.append({'code': code, 'planned': float(m.group(1).replace(',', '')), 'bought': float(m.group(2).replace(',', '')), 'pct': float(m.group(3))})
        if re.search('期間屆滿|執行完畢|執行情形|執行結果', s) and '買回' in s and not NEG.search(s):
            tt = t.replace('\n', '')
            p = num(r'原預定買回之數量\(股\)[:：]([\d,]+)', tt); b = num(r'已買回股份數量\(股\)[:：]([\d,]+)', tt)
            done.append({'code': code, 'date': str(roc(r['發言日期']).date()), 'planned': p, 'bought': b,
                         'rate': round(b / p, 3) if p and b is not None else None})

ev = pd.DataFrame(events).drop_duplicates(['code', 'date']).sort_values('date').reset_index(drop=True)
# past_exec 會被同公司多次公告重複列出 → 依 (code, planned, bought) 去重
pe = pd.DataFrame(past_exec).drop_duplicates(['code', 'planned', 'bought']) if past_exec else pd.DataFrame()
dn = pd.DataFrame(done)

IDX, COLS, CI = L.IDX, L.COLS, L.CI
AO, AC = L.AO, L.AC
pool_all = (L.v60_all >= 1e7) & (L.C >= 10)
common = np.array([bool(re.fullmatch(r'[1-9]\d{3}', c)) for c in COLS])
HS = [5, 20, 60]
rows = []
for _, e in ev.iterrows():
    if e['code'] not in CI: continue
    i = IDX.searchsorted(e['date'], side='right')
    j = CI[e['code']]
    if i >= len(IDX) or not np.isfinite(AO[i, j]) or L.GAP_UP_OPEN[i, j]: continue
    dd = IDX[i - 1]
    pm = pool_all.loc[dd].to_numpy() & common & np.isfinite(AO[i])
    row = {'code': e['code'], 'ann': str(e['date'].date()), 'entry': str(IDX[i].date()), 'm': IDX[i].strftime('%Y-%m'),
           'purpose': e['purpose'], 'planned_pct': e['planned_pct'],
           'pre20_ret_pct': round(float((AC[i - 1, j] / AC[i - 21, j] - 1) * 100), 2)}
    for h in HS:
        k = i + h - 1
        if k >= len(IDX): continue
        r = AC[k] / AO[i] - 1
        row[f'ar{h}'] = float(r[j] - 0.005 - np.nanmean(np.where(pm, r, np.nan)))
    rows.append(row)
R = pd.DataFrame(rows)

def stat(x, col):
    x = x.dropna(subset=[col])
    if len(x) == 0: return {'n': 0}
    m = x.groupby('m')[col].mean()
    t = float(m.mean() / (m.std(ddof=1) / math.sqrt(len(m)))) if len(m) > 2 and m.std() > 0 else float('nan')
    return {'n': int(len(x)), 'months': int(len(m)), 'mean_ar_pct': round(float(x[col].mean() * 100), 2),
            'median_ar_pct': round(float(x[col].median() * 100), 2), 'hit': round(float((x[col] > 0).mean()), 3), 't_month': round(t, 2)}

out = {'sample': 'MOPS 重大訊息全文（59 檔大型股，2017–2020）', 'n_events': int(len(ev)), 'n_tradable': int(len(R)),
       'events': R.round(4).to_dict('records')}
for h in HS:
    c = f'ar{h}'
    out[f'h{h}'] = {'all': stat(R, c), 'select_2017_2018': stat(R[R['ann'] < '2019-01-01'], c), 'valid_2019_2020': stat(R[R['ann'] >= '2019-01-01'], c)}
out['pre20_ret_pct_mean'] = round(float(R['pre20_ret_pct'].mean()), 2)
if len(pe):
    out['past3y_exec_from_resolutions'] = {'n_programs': int(len(pe)), 'mean_pct': round(float(pe['pct'].mean()), 1),
                                           'median_pct': round(float(pe['pct'].median()), 1),
                                           'share_lt50': round(float((pe['pct'] < 50).mean()), 3), 'share_100': round(float((pe['pct'] >= 99.9).mean()), 3)}
out['completion_announcements'] = dn.to_dict('records')
dc = dn.drop_duplicates(['code', 'date']); dc = dc[dc['planned'].notna()]
out['completion_stats'] = {'n_programs': int(len(dc)), 'mean_rate': round(float(dc['rate'].mean()), 3), 'median_rate': round(float(dc['rate'].median()), 3),
                           'share_full': round(float((dc['rate'] >= 0.999).mean()), 3), 'share_lt50pct': round(float((dc['rate'] < 0.5).mean()), 3),
                           'note': '依 code+date 去重；只含 59 檔大型股樣本中有「期間屆滿/執行完畢」公告者'}
(HERE / 'results_buyback.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(R.to_string())
print(json.dumps({k: v for k, v in out.items() if k not in ('events',)}, ensure_ascii=False, indent=1, default=str))

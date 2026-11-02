#!/usr/bin/env python3
"""第 0 步 D：否決閘門（事前門檻）。
(1) EPS 與本機兩份交叉核對資料一致 ≥ 98%（hawan0228 FinMind 格式 113 檔 2015 起；sinotrade 2023Q1 起 AI 整理，只核對）。
(2) 公告日推斷 ±1 交易日 ≥ 90%（對上市 BWIBBU 財報季標記；s0c 已算）。
(3) 每季流動性股池覆蓋率 ≥ 80%（A 日股池 ∩ 有 SUE 的事件 ÷ 法定期限日股池檔數）。
(4) 季營收（財報）與三個月月營收加總一致（±2%）的比例，說明 REVSUE 只用季報前已公開的資訊。
輸出 data/quality.json；不到門檻的季別標為缺口。"""
import json, sys, re, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pcore import DATA, deadline, load_panel, FIN_IND

fin = pd.read_parquet(DATA / 'fin_long.parquet')
fin = fin[fin['code'].str.fullmatch(r'\d{4}')]
key = fin.set_index(['code', 'y', 'q'])
out = {}
# (1a) hawan0228
rows = []
for f in glob.glob('/home/user/hawan0228/fintech_ai/data/external/raw/financials/*_financial.csv'):
    d = pd.read_csv(f, dtype={'stock_id': str}, encoding='utf-8-sig')
    d = d[d['type'] == 'EPS']
    for _, r in d.iterrows():
        dt = pd.Timestamp(r['date']); y, q = dt.year, (dt.month - 1) // 3 + 1
        if (r['stock_id'], y, q) in key.index:
            mine = key.loc[(r['stock_id'], y, q), 'eps_s']
            rows.append((r['stock_id'], y, q, float(r['value']), float(mine)))
h = pd.DataFrame(rows, columns=['code', 'y', 'q', 'theirs', 'mine']).dropna()
h['ok'] = (h['theirs'] - h['mine']).abs() <= np.maximum(0.011, 0.01 * h['theirs'].abs())
out['eps_vs_hawan'] = {'n': int(len(h)), 'agree': float(h['ok'].mean()), 'agree_ex_q4': float(h.loc[h['q'] != 4, 'ok'].mean()),
                       'agree_by_q': h.groupby('q')['ok'].mean().round(4).to_dict()}
# (1b) sinotrade：年度 EPS（近 3 年）與近 4 季單季 EPS
rows = []
for f in glob.glob('/home/user/sinotrade/shioaji-data/v1/tw/financials/*.md'):
    code = Path(f).stem
    if not re.fullmatch(r'\d{4}', code):
        continue
    txt = Path(f).read_text(encoding='utf-8')
    for sec in re.findall(r'\| 項目 \|([^\n]+)\n(?:[^\n]*\n)*?\| EPS（元） \|([^\n]+)', txt):
        hdr = [x.strip() for x in sec[0].split('|') if x.strip()]
        val = [x.strip() for x in sec[1].split('|') if x.strip() != '']
        for hh, vv in zip(hdr, val):
            try:
                v = float(vv.replace(',', ''))
            except ValueError:
                continue
            if re.fullmatch(r'\d{4}', hh):
                y = int(hh)
                if (code, y, 4) in key.index:
                    rows.append((code, 'annual', y, 4, v, float(key.loc[(code, y, 4), 'eps_ytd'])))
            elif re.fullmatch(r'\d{4}Q\d', hh):
                y, q = int(hh[:4]), int(hh[-1])
                if (code, y, q) in key.index:
                    # 對方單季＝累計差；我方 Q2/Q3 用公告單季值、Q4 用累計差
                    rows.append((code, 'quarter', y, q, v, float(key.loc[(code, y, q), 'eps_s'])))
                    # 同口徑（累計差）比對：對方方法＝本季累計 − 上季累計
                    ytd = float(key.loc[(code, y, q), 'eps_ytd'])
                    prev = float(key.loc[(code, y, q - 1), 'eps_ytd']) if (q > 1 and (code, y, q - 1) in key.index) else (0.0 if q == 1 else np.nan)
                    rows.append((code, 'quarter_ytddiff', y, q, v, ytd - prev))
s = pd.DataFrame(rows, columns=['code', 'kind', 'y', 'q', 'theirs', 'mine']).dropna()
s['ok'] = (s['theirs'] - s['mine']).abs() <= np.maximum(0.011, 0.01 * s['theirs'].abs())
out['eps_vs_sinotrade'] = {k: {'n': int(len(g)), 'agree': float(g['ok'].mean())} for k, g in s.groupby('kind')}
ss = s[s['kind'] != 'quarter']
out['eps_vs_sinotrade']['all_like_for_like'] = {'n': int(len(ss)), 'agree': float(ss['ok'].mean())}
# (3) 股池覆蓋率
E = pd.read_parquet(DATA / 'events.parquet')
P = load_panel(); IDX = pd.DatetimeIndex(P['IDX']); COLS = np.array(P['COLS'])
V60, C = P['V60'], pd.DataFrame(P['C']).ffill().to_numpy()
ind = fin.sort_values(['y', 'q']).groupby('code')['industry'].last()
ok_code = np.array([c.isdigit() and len(c) == 4 and not c.startswith('0') and not c.startswith('91') and ind.get(c, 'ci') not in FIN_IND for c in COLS])
cov = {}
for k, g in E.groupby('k'):
    y, q = k // 4, k % 4 + 1
    di = min(IDX.searchsorted(deadline(y, q)), len(IDX) - 1)
    pool = set(COLS[(V60[di] >= 1e7) & (C[di] >= 10) & ok_code])
    have = set(g.loc[g['SUEP'].notna() | g['SUE'].notna(), 'code'])
    cov[f'{y}Q{q}'] = round(len(pool & have) / max(len(pool), 1), 4)
out['pool_coverage'] = cov
out['pool_coverage_fail'] = [k for k, v in cov.items() if v < 0.8]
# (4) 季營收 vs 月營收加總
rv = pd.read_parquet(DATA / 'rev_long.parquet')
rv['p'] = pd.PeriodIndex(rv['ym'], freq='M'); rv['y'] = rv['p'].dt.year; rv['q'] = (rv['p'].dt.month - 1) // 3 + 1
qs = rv.groupby(['code', 'y', 'q']).agg(rev=('rev', 'sum'), n=('rev', 'count')).reset_index()
qs = qs[qs['n'] == 3]
m = qs.merge(fin[['code', 'y', 'q', 'rev_s']], on=['code', 'y', 'q'])
m = m[(m['rev_s'] > 0) & (m['rev'] > 0)]
m['ratio'] = m['rev_s'] / 1000 / m['rev']   # 財報元 vs 月營收千元
out['rev_stmt_vs_monthly'] = {'n': int(len(m)), 'within2pct': float(((m['ratio'] - 1).abs() <= 0.02).mean()),
                              'within5pct': float(((m['ratio'] - 1).abs() <= 0.05).mean()), 'median_ratio': float(m['ratio'].median())}
av = json.load(open(DATA / 'avail_dates_summary.json'))
out['dates_validation'] = av['validation_vs_fsq']
out['gates'] = {'eps_ge_98': bool(min(out['eps_vs_hawan']['agree'], out['eps_vs_sinotrade']['all_like_for_like']['agree']) >= 0.98),
                'dates_ge_90': bool(av['validation_vs_fsq']['A_ttm']['within1'] >= 0.9 and av['validation_vs_fsq']['A_bv']['within1'] >= 0.9),
                'coverage_ge_80_all': len(out['pool_coverage_fail']) == 0,
                'pre_ifrs_2006_2012': 'NOT AVAILABLE（fin_hist 從 2013Q1 起；SUE 需 8 季標準差，第一個 SUE⊥ 事件季為 2016Q1）'}
json.dump(out, open(DATA / 'quality.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ['eps_vs_hawan', 'eps_vs_sinotrade', 'pool_coverage_fail', 'rev_stmt_vs_monthly', 'gates']}, ensure_ascii=False, indent=0))
print('coverage', cov)

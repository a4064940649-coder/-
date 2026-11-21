#!/usr/bin/env python3
"""第 1 項：資產負債表（bs_hist 簡明表）＋財報細項（fs_xbrl）＋研發（rd_hist）＋財報公告日（filing_dates.csv）。
用法：python3 -I -B build_fundamentals.py <臨時clone> <輸出資料夾> <fin_long.parquet(交叉檢查用)>
"""
import sys, io, os, json, csv, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsdlib
import pandas as pd, numpy as np

c, out, finlong = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
head = tsdlib.head(c)
STD = {'資產總計': 'total_assets', '資產總額': 'total_assets', '負債總計': 'total_liabilities', '負債總額': 'total_liabilities',
       '權益總計': 'equity_total', '權益總額': 'equity_total', '歸屬於母公司業主之權益合計': 'equity_parent', '歸屬於母公司業主權益合計': 'equity_parent',
       '股本': 'share_capital', '流動資產': 'current_assets', '非流動資產': 'noncurrent_assets', '流動負債': 'current_liabilities',
       '非流動負債': 'noncurrent_liabilities', '現金及約當現金': 'cash', '資本公積': 'capital_surplus', '保留盈餘': 'retained_earnings',
       '保留盈餘（或累積虧損）': 'retained_earnings', '其他權益': 'other_equity', '庫藏股票': 'treasury_stock', '非控制權益': 'minority_interest',
       '每股參考淨值': 'bvps_ref', '應付公司債': 'bonds_payable', '應付債券': 'bonds_payable', '應付金融債券': 'bank_debentures_payable',
       '短期債務': 'short_term_debt', '其他借款': 'other_borrowings', '應收款項－淨額': 'receivables_net', '應收款項': 'receivables_net'}
man = dict(upstream_repo=tsdlib.URL, upstream_commit=head, files={})

# ---- bs_hist → 長表
e = tsdlib.ls(c, 'data/mops/bs_hist'); tsdlib.fetch_shas(c, [s for _, s in e])
rows = []; nfile = collections.Counter(); unmapped = collections.Counter()
for p, b in tsdlib.iter_blobs(c, e):
    nm = p.rsplit('/', 1)[1][:-4]; per, tpl, mk = nm.split('_')
    rd = list(csv.reader(io.StringIO(b.decode('utf-8-sig')))); h = rd[0]
    assert h[:4] == ['stock_id', 'name', 'period', 'market'], h[:4]
    for r in rd[1:]:
        assert len(r) == len(h) and r[2] == per and r[3] == mk, (p, r[:4])
        for k, v in zip(h[4:], r[4:]):
            v = v.strip().replace(',', '')
            if v == '':
                continue
            if k not in STD: unmapped[k] += 1
            rows.append((per, r[0].strip(), mk, tpl, k, STD.get(k), float(v)))
    nfile[tpl + '_' + mk] += 1
bs = pd.DataFrame(rows, columns=['period', 'code', 'market', 'template', 'account', 'item', 'value'])
dup = bs.duplicated(['period', 'code', 'account']).sum()
bs = bs.sort_values(['period', 'code', 'template', 'account']).reset_index(drop=True)
for col in ['period', 'market', 'template', 'account', 'item']: bs[col] = bs[col].astype('category')
bs.to_parquet(f'{out}/bs_hist_long.parquet', compression='zstd', index=False)
cov = bs[bs.item == 'total_assets'].groupby('period', observed=True).code.nunique()
man['files']['bs_hist_long.parquet'] = dict(
    source_dir='data/mops/bs_hist', source_files=len(e), files_by_type=dict(nfile), rows=len(bs), periods=[min(bs.period.astype(str)), max(bs.period.astype(str))],
    n_periods=int(bs.period.nunique()), n_codes=int(bs.code.nunique()), duplicate_period_code_account=int(dup),
    companies_with_total_assets_per_period=dict(min=int(cov.min()), max=int(cov.max()), first=int(cov.iloc[0]), last=int(cov.iloc[-1])),
    columns=dict(period='季別 2015Q1', code='公司代號', market='twse／tpex', template='業別表：ci 一般業、basi 銀行、bd 證券、fh 金控、ins 保險、other 其他',
                 account='官方簡明表的原始科目名（中文）', item='標準化英文代稱（沒對應到的是空值）', value='數值；金額單位＝新台幣千元；每股參考淨值＝元；股數欄＝股'),
    item_map=STD, note='空白格（公司沒有該科目）不產生列。一般業（ci）簡明表只有流動／非流動資產與負債、總額、股本與權益各項；沒有現金、應收、存貨、借款的細項。')

# ---- fs_xbrl（細項：存貨、應收、現金…；單位元）
e = [x for x in tsdlib.ls(c, 'data/meta/fs_xbrl') if x[0].endswith('.csv')]; tsdlib.fetch_shas(c, [s for _, s in e])
fx = pd.concat([pd.read_csv(io.BytesIO(b), dtype={'stock_id': str}) for p, b in tsdlib.iter_blobs(c, e)], ignore_index=True)
fx = fx.rename(columns={'stock_id': 'code'}).sort_values(['period', 'code']).reset_index(drop=True)
fx.to_parquet(f'{out}/fs_xbrl.parquet', compression='zstd', index=False)
valcols = [x for x in fx.columns if x not in ('period', 'code', 'industry_template', 'report', 'period_end', 'source_file')]
man['files']['fs_xbrl.parquet'] = dict(source_dir='data/meta/fs_xbrl（與 curated/fs_xbrl 同內容）', source_files=len(e), rows=len(fx),
    periods=[fx.period.min(), fx.period.max()], n_periods=int(fx.period.nunique()), n_codes=int(fx.code.nunique()),
    duplicate_period_code=int(fx.duplicated(['period', 'code']).sum()), columns=list(fx.columns),
    non_null_counts={k: int(fx[k].notna().sum()) for k in valcols},
    note='一家一季一列；金額單位＝新台幣元；_ytd 欄是年初至今累計；空白＝XBRL 沒有該科目，不是 0。含未上市櫃的公開發行公司。上游說明見倉庫 curated/fs_xbrl/README.md。')

# ---- rd_hist
e = tsdlib.ls(c, 'data/mops/rd_hist'); tsdlib.fetch_shas(c, [s for _, s in e])
rdh = pd.concat([pd.read_csv(io.BytesIO(b), dtype={'stock_id': str, 'alt': str}) for p, b in tsdlib.iter_blobs(c, e)], ignore_index=True)
rdh = rdh.rename(columns={'stock_id': 'code'}).sort_values(['period', 'code']).reset_index(drop=True)
rdh.to_parquet(f'{out}/rd_hist.parquet', compression='zstd', index=False)
man['files']['rd_hist.parquet'] = dict(source_dir='data/mops/rd_hist', source_files=len(e), rows=len(rdh), periods=[rdh.period.min(), rdh.period.max()],
    n_periods=int(rdh.period.nunique()), n_codes=int(rdh.code.nunique()), duplicate_period_code=int(rdh.duplicated(['period', 'code']).sum()),
    columns=list(rdh.columns), rows_with_rd_q=int(rdh.rd_q.notna().sum()), rows_with_rd_ytd=int(rdh.rd_ytd.notna().sum()),
    note='rd_q／rd_ytd＝研發費用單季／累計；rev_q／rev_ytd＝營收；單位＝新台幣元；report cr 合併、ir 個體。')

# ---- filing_dates
e = tsdlib.ls(c, 'data/meta/filing_dates.csv'); tsdlib.fetch_shas(c, [s for _, s in e])
(_, b), = list(tsdlib.iter_blobs(c, e))
fd = pd.read_csv(io.BytesIO(b), dtype=str)
fd = fd.rename(columns={'stock_id': 'code'})
fd['year'] = fd.year.astype('int16'); fd['season'] = fd.season.astype('int8'); fd['bytes'] = pd.to_numeric(fd.bytes, errors='coerce').astype('Int64')
bad_ts = int(pd.to_datetime(fd.uploaded_at, errors='coerce').isna().sum())
fd['uploaded_at'] = pd.to_datetime(fd.uploaded_at, errors='coerce')
fd.to_parquet(f'{out}/filing_dates.parquet', compression='zstd', index=False)
man['files']['filing_dates.parquet'] = dict(source_dir='data/meta/filing_dates.csv', rows=len(fd), years=[int(fd.year.min()), int(fd.year.max())],
    n_codes=int(fd.code.nunique()), columns=list(fd.columns), uploaded_at_unparseable=bad_ts,
    doc_type_counts=fd.doc_type.value_counts().head(15).to_dict(), corrected_counts=fd.corrected.value_counts().to_dict(),
    uploaded_at_range=[str(fd.uploaded_at.min()), str(fd.uploaded_at.max())],
    note='一份上傳文件一列；uploaded_at＝上傳時戳（台北時間，到秒）；corrected＝是否為更正版。同一家同一季可有多列（合併／個體、更正重傳）。')

# ---- 交叉檢查
fl = pd.read_parquet(finlong, columns=['code', 'period', 'assets', 'equity_parent', 'industry'])
ta = bs[bs.item == 'total_assets'][['period', 'code', 'template', 'value']].astype({'period': str, 'template': str})
m = ta.merge(fl, on=['period', 'code'])
m = m[m.assets.notna()]
m['ok'] = (m.value - (m.assets / 1000).round()).abs() <= 1
bad = m[~m.ok].assign(rel=lambda d: (d.value * 1000 / d.assets - 1))
x1 = dict(compared=len(m), match_within_1k=int(m.ok.sum()), mismatch=int((~m.ok).sum()), bs_rows_without_fin_long=int(len(ta) - len(ta.merge(fl[fl.assets.notna()], on=['period', 'code']))),
          by_template={k: [int(g.ok.sum()), len(g)] for k, g in m.groupby('template')},
          mismatch_abs_rel_gt_1pct=int((bad.rel.abs() > 0.01).sum()), mismatch_sample=bad.sort_values('rel', key=abs, ascending=False).head(25)[['period', 'code', 'template', 'value', 'assets', 'rel']].to_dict('records'),
          note='bs_hist 千元 對 fin_long 元／1000 四捨五入，容許差 1 千元。fin_long 來自同倉庫 fin_hist（XBRL 整包），bs_hist 來自 MOPS 簡明表：同一官方來源的兩條管道，不是完全獨立。')
fxa = fx[['period', 'code', 'total_assets', 'cash', 'inventories', 'accounts_receivable']]
m2 = ta.merge(fxa, on=['period', 'code']); m2 = m2[m2.total_assets.notna()]
ok2 = (m2.value - (m2.total_assets / 1000).round()).abs() <= 1
x2 = dict(compared=len(m2), match_within_1k=int(ok2.sum()), mismatch=int((~ok2).sum()),
          mismatch_keys=m2[~ok2][['period', 'code']].astype(str).agg('_'.join, axis=1).tolist()[:60])
# 金融業 bs_hist 的現金 對 xbrl 現金
ca = bs[bs.item == 'cash'][['period', 'code', 'value']].astype({'period': str}).merge(fxa, on=['period', 'code']); ca = ca[ca.cash.notna()]
ok3 = (ca.value - (ca.cash / 1000).round()).abs() <= 1
x3 = dict(compared=len(ca), match_within_1k=int(ok3.sum()))
# 研發 對 fs（fin_long 無研發）：rd_hist 的 rev_ytd 對 fs_xbrl revenue_ytd
r2 = rdh[['period', 'code', 'rev_ytd']].merge(fx[['period', 'code', 'revenue_ytd']], on=['period', 'code']).dropna()
x4 = dict(compared=len(r2), equal=int((r2.rev_ytd == r2.revenue_ytd).sum()))
man['cross_checks'] = dict(bs_total_assets_vs_fin_long=x1, bs_total_assets_vs_fs_xbrl=x2, bs_cash_financials_vs_fs_xbrl=x3, rd_hist_rev_ytd_vs_fs_xbrl_revenue_ytd=x4)
man['unmapped_accounts_top'] = dict(unmapped.most_common(40))
for f in man['files']:
    man['files'][f]['bytes'] = os.path.getsize(f'{out}/{f}'); man['files'][f]['sha256'] = tsdlib.sha256_file(f'{out}/{f}')
json.dump(man, open(f'{out}/_build_info.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in man.items() if k != 'unmapped_accounts_top'}, ensure_ascii=False, indent=1, default=str)[:9000])

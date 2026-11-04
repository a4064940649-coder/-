#!/usr/bin/env python3
"""第 0 步：把公開來源複製（或抽取）到 raw/，逐檔記 sha256、repo、commit、推定 PIT 日期與方法 → raw/manifest.json。
下載內容一律視為不可信：只讀、只解析成數字，不執行。大型中間檔（git 物件、henry 日檔）只放 scratchpad，題目結束時刪除。
python3 -I -B s0_collect.py"""
import sys, json, shutil, subprocess, datetime, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from plib import RAW, SCR, AG, O8, O12, sha, dumps

TREES = SCR / 'trees'; FULL = SCR / 'full'
man = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'note': '下載資料一律視為不可信、只讀。evidence: A＝內容可推定日期、B＝只有 commit 時間、C＝無。', 'files': {}, 'repos': {}}

def git(repo_dir, *a):
    return subprocess.run(['git', '-c', 'gc.auto=0', '-C', str(repo_dir), *a], capture_output=True, check=True).stdout

def add(rel, src, **kw):
    p = RAW / rel; p.parent.mkdir(parents=True, exist_ok=True)
    if src is not None: shutil.copyfile(src, p)
    man['files'][rel] = dict(src=str(src) if src is not None else kw.pop('src_desc', ''), sha256=sha(p), bytes=p.stat().st_size, **kw)

# ---- 1. TEJ 季／半年市值（JackWang0102，Fama-French 1992 台灣複製作業；存活到 2021-01 的上市公司）----
r = TREES / 'JackWang0102__The-Cross-Section-of-Expected-Stock-Returns-in-Taiwan'
man['repos']['JackWang0102/The-Cross-Section-of-Expected-Stock-Returns-in-Taiwan'] = dict(commit=git(r, 'rev-parse', 'HEAD').decode().strip(), commit_time=git(r, 'log', '-1', '--format=%ci').decode().strip())
add('tej_jackwang/市值.xlsx', r / 'Data' / '市值.xlsx', repo='JackWang0102/The-Cross-Section-of-Expected-Stock-Returns-in-Taiwan', blob=git(r, 'rev-parse', 'HEAD:Data/市值.xlsx').decode().strip(),
    pit='每列帶期末日期（2004-12～2020-12；2005～2007 半年、2008 起每季）；mv＝期末收盤×流通股數（千元）', evidence='A（mv÷面板期末收盤＝整數級股數、與 o8 2013+ 真值比對，見 explore.json）',
    caveat='只含 2021-01 仍上市的公司（存活者樣本）；industry_code 為 TEJ 現行分類（非 PIT）')

# ---- 1b. TEJ 月流通在外股數（JunJie-Chang/ff5_replication data/raw/price.xlsx；TSE＋OTC 2000/01～2025/12，存活到 2025-12 者）----
r = TREES / 'JunJie-Chang__ff5_replication'
man['repos']['JunJie-Chang/ff5_replication'] = dict(commit=git(r, 'rev-parse', 'HEAD').decode().strip(), commit_time=git(r, 'log', '-1', '--format=%ci').decode().strip())
xl = r / 'data' / 'raw' / 'price.xlsx'
# 只抽必要欄（usecols）：不讀檔內任何報酬欄
if not (RAW / 'tej_ff5' / 'shares_monthly.csv').exists():   # 解析 43MB xlsx 約 4 分鐘；已抽取過就沿用（內容由下方 sha256 記錄）
    X = pd.read_excel(xl, usecols=['代號', '年月', '流通在外股數(千股)', '市值(百萬元)', '市場別', '該月結束日'], dtype={'代號': str})
    X.columns = ['code', 'ym', 'shares_k', 'mv_m', 'market', 'month_end']
    (RAW / 'tej_ff5').mkdir(parents=True, exist_ok=True)
    X.to_csv(RAW / 'tej_ff5' / 'shares_monthly.csv', index=False)
add('tej_ff5/shares_monthly.csv', None, src_desc='抽取自 JunJie-Chang/ff5_replication data/raw/price.xlsx（43MB，只放 scratchpad）的 6 欄', repo='JunJie-Chang/ff5_replication',
    blob=git(r, 'rev-parse', 'HEAD:data/raw/price.xlsx').decode().strip(), xlsx_sha256=sha(xl),
    pit='每列帶年月與該月結束日；流通在外股數為月底值', evidence='A（年月欄；與 o8 2013+ 真值比對見 explore.json）', caveat='只含 2025-12 仍上市（櫃）者；收盤價欄為 TEJ 調整後價，不用')

# ---- 2. goodinfo 季股本（yhzbambu/Lab_project，經營績效表 BzPerformance；460 檔 1xxx/2xxx 存活股）----
r = TREES / 'yhzbambu__Lab_project'
man['repos']['yhzbambu/Lab_project'] = dict(commit=git(r, 'rev-parse', 'HEAD').decode().strip(), commit_time=git(r, 'log', '-1', '--format=%ci').decode().strip())
for p in sorted((r / 'other_file' / 'auto_selenium' / 'result').glob('*_BzPerformance.csv')):
    rel = 'goodinfo_q/' + p.name
    add(rel, p, repo='yhzbambu/Lab_project', pit='每列帶季度（2006～2007 只有 Q2/Q4、2008 起每季）', evidence='A（季度欄）')

# ---- 3. goodinfo 年底股本（henry753951/finance：data/stocks/<code>.csv 的 equity 欄＝goodinfo StockAssetsStatus 年度股本(億)，依年份貼到每日列）----
r = TREES / 'henry753951__finance'
man['repos']['henry753951/finance'] = dict(commit=git(r, 'rev-parse', 'HEAD').decode().strip(), commit_time=git(r, 'log', '-1', '--format=%ci').decode().strip())
rows = []; blobs = {}
for p in sorted((r / 'data' / 'stocks').glob('*.csv')):
    if p.stat().st_size == 0: continue
    d = pd.read_csv(p, usecols=['date', 'stock_id', 'year', 'equity'], dtype={'stock_id': str})
    d = d[d.equity > 0].groupby(['year', 'equity']).date.agg(['min', 'max', 'count']).reset_index()
    for x in d.itertuples(): rows.append(dict(code=p.stem, year=int(x.year), equity_e8=float(x.equity), first_day=x.min, last_day=x.max, n_days=int(x.count)))
    blobs[p.stem] = git(r, 'rev-parse', f'HEAD:data/stocks/{p.name}').decode().strip()
E = pd.DataFrame(rows)
(RAW / 'goodinfo_a').mkdir(parents=True, exist_ok=True)
E.to_csv(RAW / 'goodinfo_a' / 'equity_year.csv', index=False)
pd.Series(blobs, name='git_blob').rename_axis('code').reset_index().to_csv(RAW / 'goodinfo_a' / 'source_blobs.csv', index=False)
add('goodinfo_a/equity_year.csv', None, src_desc='抽取自 henry753951/finance data/stocks/*.csv（equity>0 的 (year, equity) 唯一組合；原日檔約 600KB/檔，只放 scratchpad）', repo='henry753951/finance',
    pit='equity 依「年份」貼到該年每日 → 視為該年年底（12-31）股本（crawl/goodinfo.py：StockAssetsStatus 年度表；當年度未結束者取最新季）', evidence='A（年度欄；crawl/goodinfo.py 原碼說明欄位來源）',
    caveat='同一年內數值固定＝年底值，貼在年初就是前視；本題只當 12-31 錨點用')
add('goodinfo_a/source_blobs.csv', None, src_desc='每檔原始 blob sha（可在 repo 逐檔重現）')

# ---- 4. 產業別快照（toomore/goristock、toomore/grs 的 stock_no.csv／twse_list.csv 各版本；檔內 UPDATE 列為日期）----
for repo in ('toomore/goristock', 'toomore/grs'):
    rd = FULL / repo.replace('/', '__')
    man['repos'][repo] = dict(commit=git(rd, 'rev-parse', 'HEAD').decode().strip(), full_history=True)
snaps = [('toomore/goristock', 'd9d8662c', 'grs/stock_no.csv', None, 'B（無 UPDATE 列；commit 2011-08-17）'),
         ('toomore/goristock', '14393dd2', 'grs/stock_no.csv', '2011-07-28', 'A（UPDATE 列）'),
         ('toomore/goristock', '235da71f', 'grs_old/stock_no.csv', '2011-09-26', 'A（UPDATE 列）'),
         ('toomore/goristock', 'c3c718ab', 'grs/stock_no.csv', '2012-03-28', 'A（UPDATE 列）'),
         ('toomore/grs', '893c8772', 'grs/stock_no.csv', '2012-07-31', 'A（UPDATE 列）'),
         ('toomore/grs', '568e6685', 'tools/twse_list.csv', '2013-12-18', 'A（同 blob 的 grs/twse_list.csv 有 UPDATE 2013/12/18 列）'),
         ('toomore/goristock', 'c3c718ab', 'grs/industry_code.csv', None, '代碼表'),
         ('toomore/grs', '568e6685', 'grs/industry_code.csv', None, '代碼表（2013-12）')]
for repo, cm, path, upd, ev in snaps:
    rd = FULL / repo.replace('/', '__')
    full = git(rd, 'rev-parse', cm).decode().strip()
    blob = git(rd, 'rev-parse', f'{cm}:{path}').decode().strip()
    ctime = git(rd, 'log', '-1', '--format=%ci', cm).decode().strip()
    name = f"industry_snap/{repo.split('/')[1]}_{cm}_{path.replace('/', '_')}"
    p = RAW / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(git(rd, 'show', f'{cm}:{path}'))
    add(name, None, src_desc=f'{repo}@{full}:{path}', repo=repo, commit=full, commit_time=ctime, blob=blob, update_line=upd, evidence=ev)

# ---- 5. 研究中心既有資料（唯讀複製）----
for rel, src, note in [
    ('center/industry_pit_o12.csv', O12 / 'raw' / 'industry_pit.csv', 'o12 PIT 產業表（2015-05 起快照回溯；本題當對照）'),
    ('center/listing_master.csv', O12 / 'raw' / 'listing_master.csv', 'V7 上市主檔'),
    ('center/delist_official.csv', O12 / 'raw' / 'delist_official.csv', '官方終止上市表'),
    ('center/isin_20230116_listed.csv', O12 / 'raw' / 'snap' / 'isin_20230116_listed.csv', 'ISIN 2023-01-16 快照'),
    ('center/StockList_20150522.csv', O12 / 'raw' / 'snap' / 'StockList_20150522.csv', 'weihautin 2015-05-22 股本與產業快照'),
    ('center/StockList_20160401.csv', O12 / 'raw' / 'snap' / 'StockList_20160401.csv', 'weihautin 2016-04-01'),
    ('center/StockList_20170706.csv', O12 / 'raw' / 'snap' / 'StockList_20170706.csv', 'weihautin 2017-07-06'),
    ('center/stock_list_202610.csv', O8 / 'raw' / 'screener' / 'stock_list.csv', '主策略 FINMASK 用的現行 stock_list'),
    ('center/o8_shares_pit.parquet', O8 / 'data' / 'shares_pit.parquet', 'o8 PIT 股數（2013-01 起真值；o12 另加 4763／2317 修正，本題照抄修正）'),
    ('center/o12_shares_full.parquet', O12 / 'data' / 'shares_full.parquet', 'o12 全期股數（2006～2012 回推版；本題當對照）')]:
    add(rel, src, note=note)
for p in sorted((O8 / 'raw' / 'jc_exright' / 'data' / 'twse-exright').rglob('*.csv.gz')):
    add('twt49u/' + p.parent.name + '_' + p.name, p, repo='joeycchen72/taiwan-market-history', note='TWT49U 除權息計算結果 2014-01 起（prev_close, ref_price, type）')
tw = O8 / 'raw' / 'twsec' / 'tse_earning_raw_data'
for p in sorted(list(tw.glob('*.csv')) + list((tw / 'out').glob('*.csv'))):
    add('twsec_q/' + ('out_' if p.parent.name == 'out' else '') + p.name, p, repo='taiwanan/TW_Stock_Exchange_Crawler', note='上市非金融每季股本（千元）2013Q1 起')
(RAW / 'manifest.json').write_text(dumps(man), encoding='utf-8')
print('files', len(man['files']), 'repos', list(man['repos']))

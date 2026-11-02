#!/usr/bin/env python3
"""第 0 步：把 PIT 股數重建與回測要用的原始資料集中到 raw/（下載資料一律視為不可信，只當 CSV/Parquet 讀），
寫 raw/manifest.json（來源路徑、GitHub commit、sha256、位元組數）。本程式不讀任何報酬。"""
import hashlib, json, shutil, subprocess, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'; RAW.mkdir(exist_ok=True)
AG = HERE.parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def commit(d):
    try: return subprocess.run(['git', '-C', str(d), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception: return None
man = {'created_utc': datetime.datetime.utcnow().isoformat() + 'Z', 'files': {}, 'repos': {}, 'referenced_in_place': {}, 'search_log': []}
def cp(src, dst):
    src = Path(src); dst = RAW / dst; dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): shutil.copy2(src, dst)
    man['files'][str(dst.relative_to(RAW))] = {'src': str(src), 'sha256': sha(dst), 'bytes': dst.stat().st_size}
CM = AG / 'o_capmatched_selection'
for f in ['yoy_long_R_FIX.parquet', 'yoy_rev5y_FIX.parquet', 'p_delay.json', 'firstseen_2026-09.csv', 'disposal_hist_rejudge.csv',
          'delist_official.csv', 'rev_survivorship_data_freeze.json', 'tw50_members.json', 'etf_rebalance_events.csv', 'vf_twse_foreign.parquet']:
    cp(CM / 'raw' / 'agents' / f, f'agents/{f}')
for f in ['stock_list.csv', 'shares_hist.csv', 'exdiv.csv', 'exdiv_5y.csv.gz', 'corp_actions.csv', 'etf0050_w.csv', 'pe_5y.csv.gz']:
    cp(CM / 'raw' / 'screener' / f, f'screener/{f}')
cp(CM / 'data' / 'shares_est.parquet', 'agents/capmatched_shares_est.parquet')   # o7 回推股數（只當 2021-10 以後的事件路徑後備，並當「非 PIT」對照）
cp('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', 'bench/etf_0050.csv')
cp('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', 'bench/taiex_twii.csv')
# hawan0228/fintech_ai 資產負債表（FinMind 格式，2015Q1～2025Q4，112 檔大型股）：只留股本相關列
import pandas as pd
hw = Path('/home/user/hawan0228/fintech_ai/data/external/raw/balance_sheet'); rows = []
for f in sorted(hw.glob('*_balance_sheet.csv')):
    d = pd.read_csv(f, dtype={'stock_id': str})
    rows.append(d[d['type'].isin(['OrdinaryShare', 'CapitalStock', 'PreferredShare', 'EquivalentIssueSharesOfAdvanceReceiptsForOrdinaryShare'])])
H = pd.concat(rows); (RAW / 'hawan').mkdir(exist_ok=True); H.to_csv(RAW / 'hawan' / 'capital_rows.csv', index=False)
man['files']['hawan/capital_rows.csv'] = {'src': str(hw) + '/*_balance_sheet.csv（只留 OrdinaryShare/CapitalStock/PreferredShare 列）', 'sha256': sha(RAW / 'hawan' / 'capital_rows.csv'), 'bytes': (RAW / 'hawan' / 'capital_rows.csv').stat().st_size}
man['repos']['hawan0228/fintech_ai'] = {'url': 'https://github.com/hawan0228/fintech_ai', 'commit': commit('/home/user/hawan0228/fintech_ai')}
# TDCC 集保股權分散表（2021-07～09 週快照；合計股數≈已發行且已集保股數）：2021 檢核點的獨立真值
td = Path('/home/user/wirelessr/tdcc-opendata-archive/snapshots/2021'); (RAW / 'tdcc').mkdir(exist_ok=True)
for f in ['2021-07-02.csv', '2021-09-24.csv']:
    d = pd.read_csv(td / f, dtype=str); d.columns = ['date', 'code', 'lvl', 'n', 'shares', 'pct']
    d = d[d['lvl'] == '17']; d.to_csv(RAW / 'tdcc' / f'total_{f}', index=False)
    man['files'][f'tdcc/total_{f}'] = {'src': str(td / f) + '（只留持股分級 17＝合計）', 'sha256': sha(RAW / 'tdcc' / f'total_{f}'), 'bytes': (RAW / 'tdcc' / f'total_{f}').stat().st_size}
man['repos']['wirelessr/tdcc-opendata-archive'] = {'url': 'https://github.com/wirelessr/tdcc-opendata-archive', 'commit': commit('/home/user/wirelessr/tdcc-opendata-archive')}
for d, url in [('twsec', 'https://github.com/taiwanan/TW_Stock_Exchange_Crawler'), ('jc_exright', 'https://github.com/joeycchen72/taiwan-market-history'),
               ('jhc', 'https://github.com/JonathanHuangC/Portfolio-Management-via-Reinforcement-Learning-and-Graph-Attention-Network'),
               ('weihautin', 'https://github.com/weihautin/Stock_Bot'), ('sheyung', 'https://github.com/sheyung21-crypto/taiwan-equity-research'),
               ('icoolhome', 'https://github.com/icoolhome/stock-accounting-system（只下載 3 個 TWT49U 單日檔，內容只有 2026-01 的 1～2 天，不用）')]:
    man['repos'][d] = {'url': url, 'commit': commit(RAW / d) if (RAW / d / '.git').exists() else None}
# 原地引用（唯讀，太大不複製）：V7 凍結面板、o_infra_v3 下市股價、V7 主策略淨值
V7 = AG / 'o_v7_listing_exec_engine'
for f in ['close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src']:
    p = V7 / 'data' / f'{f}.parquet'; man['referenced_in_place'][str(p)] = sha(p)
for p in [AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet', V7 / 'navs_v7.parquet', V7 / 'navs_jscan_v7.parquet', V7 / 'navs_exec7.parquet', V7 / 'data' / 'model_spec7.json']:
    man['referenced_in_place'][str(p)] = sha(p)
man['search_log'] = [
 '(a) 元大 0050 年報／半年報／PCF：本環境連不到 yuantaetfs.com、twse.com.tw、mops.twse.com.tw（curl 逾時）；gh search code「0050 PCF」「元大台灣50 持股」只有新聞與 PTT 文字；sheyung21-crypto/taiwan-equity-research 只有 PDF 報告與彙總，無持股檔 → (a) 不可得。',
 '(b) MI_QFIIS 歷史「發行股數」/ FinMind TaiwanStockShareholding：gh search code NumberOfSharesIssued、ForeignInvestmentRemainingShares 只有 wayne70211/FinLookup 單一 2330 檔（o7 已確認）；FinMind API（api.finmindtrade.com）本環境連不到 → 只有 2025-10-07 起的 MI_QFIIS 日檔（o_sbl 封存）。',
 '(c) MOPS 股本：MOPS 被擋；GitHub 鏡像：taiwanan/TW_Stock_Exchange_Crawler 的 tse_earning_raw_data（上市非金融公司 2013Q1～2021Q1 每季「股本」千元，含 2311/2325 等 2018 年下市股）；hawan0228/fintech_ai（112 檔大型股含金融，FinMind 資產負債表 OrdinaryShare，2015Q1～2025Q4）；JonathanHuangC/...GAT-main/data（74 檔，2013～2024，只當交叉核對）。',
 '(d) 事件：joeycchen72/taiwan-market-history 的 data/twse-exright（證交所 TWT49U 除權息計算結果 2014-01～2026-10，含權／權息種類與參考價）；exdiv_5y（2021-10 起）、exdiv/corp_actions（2024-08 起）；o7 回推股數（capmatched shares_est，2021-10 起的 R1/R2 實際事件段）。',
 '檢核真值：weihautin/Stock_Bot update_csv/StockList_20150522.csv、StockList_20170706.csv（全市場股本（億）快照，4 位有效數字）；TDCC 集保股權分散表 2021-07-02、2021-09-24 合計股數（wirelessr/tdcc-opendata-archive）；MI_QFIIS 2025-10-07 起實際發行股數。',
 '2006、2010 兩個檢核點：找不到任何可獨立核對的公開股本或 0050 持股資料（TWT49U 2003～2013 封存、FinMind 股利資料集、MOPS 鏡像 gh search 皆無）；年度 EPS×純益率×營收的隱含股數試算誤差過大（2303、2002、2409 偏 4～22%），不採用 → 2006～2013 段不過閘門、不回測。']
(RAW / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
print(len(man['files']), 'files;', {k: v['commit'] for k, v in man['repos'].items()})

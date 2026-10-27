#!/usr/bin/env python3
"""把 GitHub 鏡像（不可信資料）彙整成 raw/ 下的精簡 parquet，並記錄來源 commit、檔案清單與 sha256。

用法：python3 -I -B fetch_consolidate.py <BlackBoxOP_tw_stock clone> <voidful/tw-institutional-stocker clone>
只用 pandas 讀檔（不執行來源 repo 的任何程式）。
"""
import sys, json, hashlib, subprocess
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'; RAW.mkdir(exist_ok=True)
BB = Path(sys.argv[1]); VF = Path(sys.argv[2])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def head(repo):
    return subprocess.run(['git', '-C', str(repo), 'log', '-1', '--format=%H %cI'], capture_output=True, text=True).stdout.strip()


man = {'BlackBoxOP/tw_stock': {'commit': head(BB), 'files': {}}, 'voidful/tw-institutional-stocker': {'commit': head(VF), 'files': {}}}


def stack(folder, cols, key):
    fs = sorted(Path(folder).glob('*.parquet'))
    out = []
    for f in fs:
        man['BlackBoxOP/tw_stock']['files'][f'{key}/{f.name}'] = sha(f)
        d = pd.read_parquet(f)
        d = d[[c for c in cols if c in d.columns]].copy()
        out.append(d)
    X = pd.concat(out, ignore_index=True)
    X['Date'] = pd.to_datetime(X['Date']).dt.strftime('%Y-%m-%d')
    X['Ticker'] = X['Ticker'].astype(str)
    X = X[X['Ticker'].str.fullmatch(r'\d{4}|0050|0056|006208')]      # 只留普通股（4 位數代碼）與少數 ETF，去掉權證
    return X.reset_index(drop=True)


sbl = stack(BB / 'data/margin/sbl', ['Date', 'Ticker', 'Name', 'Margin_Short_Balance', 'Margin_Short_Limit', 'SBL_Prev_Balance',
                                       'SBL_Daily_Sell', 'SBL_Daily_Return', 'SBL_Daily_Adjust', 'SBL_Balance', 'SBL_Net_Change',
                                       'SBL_Next_Day_Limit'], 'data/margin/sbl')
sbl.to_parquet(RAW / 'bb_sbl_twt93u.parquet', index=False)
mg = stack(BB / 'data/margin', ['Date', 'Ticker', 'Market', 'Margin_Balance', 'Margin_Prev_Balance', 'Short_Sell', 'Short_Buy',
                                 'Short_Stock_Repay', 'Short_Prev_Balance', 'Short_Balance', 'Short_Limit'], 'data/margin')
mg.to_parquet(RAW / 'bb_margin.parquet', index=False)
ins = stack(BB / 'data/institutional', ['Date', 'Ticker', 'Market', 'Foreign_Net', 'Foreign_Dealer_Net', 'Trust_Net', 'Dealer_Net'],
            'data/institutional')
ins.to_parquet(RAW / 'bb_inst.parquet', index=False)

for f in ['data/twse_foreign.csv', 'data/twse_flows.csv']:
    p = VF / f
    man['voidful/tw-institutional-stocker']['files'][f] = sha(p)
    d = pd.read_csv(p, dtype={'code': str})
    d.to_parquet(RAW / ('vf_' + Path(f).stem + '.parquet'), index=False)

for f in sorted(RAW.glob('*.parquet')):
    man.setdefault('consolidated', {})[f.name] = sha(f)
(RAW / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
print({k: (len(v['files']) if 'files' in v else v) for k, v in man.items()})

"""S22 資料盤點：三大法人逐日買賣超（官方 T86 類）在研究中心的涵蓋、PIT 性質與可用切分。
只讀資料、不算任何報酬。輸出 data_check.json。一律 python3 -I -B。"""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # research/tw_stock_selection
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
SRC = {
    'inst_5y': X / 'inst_5y.csv.gz',
    'inst_hist': X / 'inst_hist.csv.gz',
    'bb_inst': ROOT / 'agents/o_sbl_foreign_level/raw/bb_inst.parquet',
    'voidful_twse_flows': Path('/home/user/voidful/tw-institutional-stocker/data/twse_flows.csv'),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


close = pd.read_parquet(ROOT / 'data_panel/panel/close_raw.parquet')
cal = close.index
common = [c for c in close.columns if len(c) == 4 and c.isdigit() and not c.startswith('00')]

out = {'written_at': pd.Timestamp.now().isoformat(timespec='seconds'), 'sources': {}}

def summarize(name, df, dcol, ccol):
    df = df[[dcol, ccol]].copy()
    df[dcol] = pd.to_datetime(df[dcol])
    dts = np.sort(df[dcol].unique())
    first, last = pd.Timestamp(dts[0]), pd.Timestamp(dts[-1])
    pcal = cal[(cal >= first) & (cal <= min(last, cal[-1]))]
    missing = [str(d.date()) for d in pcal if d not in set(pd.DatetimeIndex(dts))]
    # 覆蓋率：每日「面板有收盤的上市普通股」有出現在來源的比例（每 60 日取樣）
    g = df.groupby(dcol)[ccol].apply(set)
    covs = []
    for d in pcal[::60]:
        if d in g.index:
            have = close.loc[d, common].dropna().index
            covs.append(float(np.mean([c in g[d] for c in have])))
    by_year = df.groupby(df[dcol].dt.year)[dcol].nunique().to_dict()
    return {'first': str(first.date()), 'last': str(last.date()), 'rows': int(len(df)),
            'trading_days': int(len(dts)), 'panel_days_missing_in_range': missing[:20],
            'n_panel_days_missing': len(missing),
            'listed_common_coverage_median': round(float(np.median(covs)), 4) if covs else None,
            'days_by_year': {int(k): int(v) for k, v in by_year.items()}}

i5 = pd.read_csv(SRC['inst_5y'], dtype={'code': str}, usecols=['date', 'code', 'foreign', 'trust'])
ih = pd.read_csv(SRC['inst_hist'], dtype={'code': str}, usecols=['date', 'code', 'foreign', 'trust'])
bb = pd.read_parquet(SRC['bb_inst'], columns=['Date', 'Ticker', 'Market', 'Foreign_Net', 'Trust_Net'])
bb = bb[bb.Market == 'TWSE']
for name, df, d, c in [('inst_5y', i5, 'date', 'code'), ('inst_hist', ih, 'date', 'code'), ('bb_inst', bb, 'Date', 'Ticker')]:
    out['sources'][name] = summarize(name, df, d, c)
    out['sources'][name]['path'] = str(SRC[name]); out['sources'][name]['sha256'] = sha(SRC[name])
vf = pd.read_csv(SRC['voidful_twse_flows'], usecols=['date'])
out['sources']['voidful_twse_flows'] = {'path': str(SRC['voidful_twse_flows']), 'first': vf.date.min(), 'last': vf.date.max(),
                                        'note': '只用來確認沒有更早資料'}
# 重疊一致率（inst_5y vs inst_hist；inst_5y vs bb 上市）
m = i5.merge(ih, on=['date', 'code'], suffixes=('_a', '_b'))
out['agreement'] = {'inst_5y_vs_inst_hist': {'n': int(len(m)),
                    'trust_exact': round(float((np.abs(m.trust_a - m.trust_b) < 1e-6).mean()), 5),
                    'foreign_exact': round(float((np.abs(m.foreign_a - m.foreign_b) < 1e-6).mean()), 5)}}
bb2 = bb.rename(columns={'Date': 'date', 'Ticker': 'code'}); bb2['date'] = bb2['date'].astype(str).str[:10]
m2 = i5.merge(bb2, on=['date', 'code'])
out['agreement']['inst_5y(張)_vs_bb_inst(股)'] = {'n': int(len(m2)),
    'trust_exact': round(float((np.abs(m2.trust * 1000 - m2.Trust_Net) <= 1).mean()), 5),
    'foreign_exact': round(float((np.abs(m2.foreign * 1000 - m2.Foreign_Net) <= 1).mean()), 5)}
del m, m2

# 可用訊號月份：訊號日（每月 10 日後第一個交易日，A10）前需有 ≥ 20 個法人交易日
first_inst = pd.Timestamp(out['sources']['inst_5y']['first'])
last_inst = pd.Timestamp(out['sources']['bb_inst']['last'])
inst_days = cal[(cal >= first_inst)]
months = []
for mo in pd.period_range('2016-02', '2026-11', freq='M'):
    d10 = pd.Timestamp(mo.year, mo.month, 10)
    after = cal[cal > d10]
    if len(after) == 0:
        continue
    sig = after[0]
    n_before = int(((inst_days <= sig)).sum())
    if n_before >= 20 and sig <= min(last_inst, cal[-1]):
        months.append(str(mo))
out['usable_signal_months'] = {'first': months[0], 'last': months[-1], 'n': len(months),
                               'years': round(len(months) / 12, 2)}
out['split_feasibility'] = {
    'S20_split（訓練訊號日 2016-02～2021-12）': '不可行：法人資料 2021-10-08 才開始，訓練期 71 個訊號月中只有 2 個月（2021-11、2021-12）有 20 日回看',
    'requirement': '訓練、驗證各至少 3 年＝至少 72 個訊號月',
    'available_months': len(months),
    'feasible': len(months) >= 72,
    'best_possible': f'{len(months)} 個月：最多 3 年＋{len(months)-36} 個月，或兩段各 {len(months)//2} 個月（約 {len(months)/24:.1f} 年）',
}
out['pit'] = ('T86（三大法人買賣超日報）由證交所於每個交易日收盤後（約 15:00～16:30）公布當日資料；'
              '研究中心現行訊號日為 A10（每月 10 日後第一個交易日）收盤後算訊號、次日開盤成交，'
              '所以用「訊號日當天（含）以前」的法人資料是 PIT。資料本身沒有回補修訂欄位，視同發布值。')
out['other_paths_checked'] = [
    'www.twse.com.tw T86 端點（rwd 與舊版）：代理 403，連不上',
    'api.finmindtrade.com：連不上',
    'GitHub 鏡像：voidful/tw-institutional-stocker（2025-09-11 起）、BlackBoxOP/tw_stock（2024-01 起，o_sbl_foreign_level/raw/bb_inst.parquet）',
    'o_sbl_foreign_level FINDINGS：已用 code search 約 1,000 筆、逐一檢查 18 個候選 repo，沒有 2021-10 前的全市場逐股 T86',
    'w1_trust_first_buy_small、w12_foreign_*、o_concentration_regime 同樣記錄法人資料 2021-10 才開始',
    '本機所有 csv／csv.gz／parquet 欄名掃描（trust／foreign／投信／外資）：只有上述四個來源及其複本',
]
out['verdict'] = ('資料不足' if not out['split_feasibility']['feasible'] else '可做')
out['verdict_reason'] = (f"三大法人逐股逐日資料最早 {out['sources']['inst_5y']['first']}，"
                         f"可用訊號月 {months[0]}～{months[-1]} 共 {len(months)} 個月（{len(months)/12:.1f} 年），"
                         '不足「訓練、驗證各至少 3 年」（72 個月）；依題目「資料不足就判資料不足並停，不准硬湊」，'
                         '不寫 PREREG、不算任何報酬。')
(HERE / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: out[k] for k in ('usable_signal_months', 'split_feasibility', 'verdict', 'agreement')}, ensure_ascii=False, indent=1))
for k, v in out['sources'].items():
    print(k, v.get('first'), v.get('last'), v.get('trading_days'), v.get('n_panel_days_missing'), v.get('listed_common_coverage_median'))

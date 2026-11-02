#!/usr/bin/env python3
"""第 5 步：統計與成本兩道審查（只讀 trades_all.csv.gz、navs_fixcorrected.parquet、results.json、replicate/rep_results.json）。
→ results.json 的 REVIEW（含 G6 比對）。另附未事前登記的當沖制度分期診斷（假說生成用）。"""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import core as K

res = json.loads(K.RES_F.read_text(encoding='utf-8'))
A = pd.read_csv(HERE / 'trades_all.csv.gz', dtype={'code': str}, parse_dates=['date', 'exit_date'])
O = K.sub(A, *K.PER['OOS_2006-04_2021-09'])
RV = {'stat': {}, 'cost': {}}
# --- 統計審查 ---
st = RV['stat']
st['t_by_entry_date_cluster_OOS'] = K.tstat(O.groupby('date')['net'].mean())
st['t_by_month_OOS'] = K.S(O)['t_month']
m = O.groupby(O['date'].dt.to_period('M'))['net'].mean().to_numpy()
rng = np.random.default_rng(7); bs = []
for _ in range(5000):
    # 月份區塊 bootstrap（區塊長 3 個事件月）
    n = len(m); out = []
    while len(out) < n:
        s0 = rng.integers(0, n); out.extend(m[s0:s0 + 3])
    bs.append(np.mean(out[:n]))
st['OOS_month_mean_block_boot_90CI_pct'] = [round(float(np.percentile(bs, 5) * 100), 2), round(float(np.percentile(bs, 95) * 100), 2)]
st['OOS_month_weighted_mean_pct'] = round(float(m.mean() * 100), 2)
st['OOS_share_of_sum_from_top10_events'] = round(float(O['net'].nlargest(10).sum() / O['net'].sum()), 3)
st['OOS_survivor_window_2019-05_2021-09'] = K.S(K.sub(O, '2019-05-16', '2021-09-30'))
st['OOS_pre_survivor_window'] = K.S(K.sub(O, '2006-04-01', '2019-05-15'))
nav = pd.read_parquet(HERE / 'navs_fixcorrected.parquet')['C1xCL|100萬|FIX|normal']
ex = (K.monthly(nav) - K.monthly(K.e50_nav(nav.index))).dropna()
def nw_t(x, L=6):
    x = np.asarray(x); n = len(x); e = x - x.mean(); s = (e @ e) / n
    for l in range(1, L + 1): s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / math.sqrt(s / n))
st['focus_FIX_corrected_t_iid'] = K.tstat(ex); st['focus_FIX_corrected_t_NW6'] = round(nw_t(ex.to_numpy()), 2)
st['focus_FIX_corrected_t_NW6_2019+'] = round(nw_t(ex[ex.index.year >= 2019].to_numpy()), 2)
# t 提升拆解（G2 用的衛星對 0050）：均值 vs 波動
for nm, a, b in (('2006-2015', 2006, 2015), ('2016-2026', 2016, 2026)):
    x = ex[(ex.index.year >= a) & (ex.index.year <= b)]
    st[f'focus_ex_{nm}'] = {'mean_pp': round(float(x.mean() * 100), 3), 'sd_pp': round(float(x.std() * 100), 3), 't': K.tstat(x)}
# G6
rep = json.loads((HERE / 'replicate' / 'rep_results.json').read_text())
mine = res['RESULTS']['P1']['net_flat_0.8_1.2']['OOS_2006-04_2021-09']
st['G6'] = {'researcher': {'mean_pct': mine['mean_pct'], 't_month': mine['t_month'], 'n': mine['n']}, 'replicator': rep['OOS']['net'],
            'diff_pp': round(abs(rep['OOS']['net']['mean_pct'] - mine['mean_pct']), 3), 'diff_t': round(abs(rep['OOS']['net']['t_month'] - mine['t_month']), 3)}
st['G6']['pass_numeric'] = st['G6']['diff_pp'] <= 0.1 and st['G6']['diff_t'] <= 0.2
# --- 成本審查 ---
cs = RV['cost']
cs['cost_x_mean_pct_by_seg'] = {s: round(float(g['cost_x'].mean() * 100), 2) for s, g in A.groupby('seg')}
cs['flat_cost_mean_pct_by_seg'] = {s: round(float(g['cost'].mean() * 100), 2) for s, g in A.groupby('seg')}
cs['share_fill_lt1_slot10萬'] = round(float((A['fill10'] < 0.999).mean()), 3)
U = K.universe()
for kap in (0.25, 1.0):
    K.KAPPA = kap
    X = K.exec_cost_event(K.sub(A.assign(ordr=A['ordr']), *K.PER['OOS_2006-04_2021-09']), 1e5)
    cs[f'OOS_net_x_kappa{kap}'] = K.S(X, 'net_x')
K.KAPPA = K.SPEC7['kappa']
for slot in (3e5, 1e6):
    X = K.exec_cost_event(O, slot)
    cs[f'OOS_net_x_slot{int(slot/1e4)}萬'] = dict(K.S(X, 'net_x'), fill_mean=round(float(X['fill'].mean()), 3))
cs['note_prepay'] = '主欄假設 0050 於公告日（處置前一交易日）收盤贖回，公告在盤後，實務上做不到；嚴格版（2 槽現金緩衝）見 POSTHOC_FIX sleeves cashbuffer2'
# --- 未事前登記：當沖制度分期（假說生成） ---
eras = {'2006-04_2013-12（無現股當沖）': ('2006-04-01', '2013-12-31'), '2014-01_2017-04（現股當沖開放、稅 0.3%）': ('2014-01-01', '2017-04-27'),
        '2017-04-28_2021-09（當沖稅減半 0.15%）': ('2017-04-28', '2021-09-30'), '2021-10_2026-10': ('2021-10-01', '2026-10-22')}
RV['posthoc_daytrade_eras'] = {k: {'net': K.S(K.sub(A, *v)), 'gross': K.S(K.sub(A, *v), 'gross')} for k, v in eras.items()}
res['REVIEW'] = RV
K.RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(RV, ensure_ascii=False, default=str)[:5000])

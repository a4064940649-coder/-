#!/usr/bin/env python3
"""R97：彙整 results.json（讀本資料夾 out_*.json、analysis.json、extra_checks.json、paths20_*.json；S97 的數字只取自其 results.json／FINDINGS.md）。python3 -I -B make_results.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
J = lambda f: json.loads((H / f).read_text())
b, l, lh, A, X, pb, pl = J('out_base.json'), J('out_lim7.json'), J('out_lim7h.json'), J('analysis.json'), J('extra_checks.json'), J('paths20_base.json'), J('paths20_lim7.json')
s97 = json.loads((H.parent / 's97_limit_lock_execution' / 'results.json').read_text())['cont']['scen']
sb, sl = s97['BASE'], s97['A_i']
nav = {m: pd.read_csv(H / f'nav_{m}.csv.gz', index_col=0, parse_dates=True)['nav'] for m in ('base', 'lim7')}
def c630(n):
    a = pd.Timestamp('2015-06-30'); e = n.index[-1]
    return float((n.iloc[-1] / n[a]) ** (1 / ((e - a).days / 365.25)) - 1)
p = lambda x: round(x * 100, 4)
def row(name, mine_b, mine_l, s_b, s_l, unit='pct'):
    return dict(item=name, r97_base=mine_b, r97_lim7=mine_l, s97_base=s_b, s97_lim7=s_l, diff_lim7_r97_minus_s97=round(mine_l - s_l, 4), within_0_3=bool(abs(mine_l - s_l) <= 0.3) if unit == 'pct' else bool(mine_l == s_l))
cmp = [row('全期年化 %', p(b['cagr']), p(l['cagr']), sb['cagr_full_pct'], sl['cagr_full_pct']),
       row('2008～2015 年中年化 %（2008-01-02→2015-06-30）', p(b['cagr_2008_2015H1']), p(l['cagr_2008_2015H1']), sb['cagr_2008_2015H1_pct'], sl['cagr_2008_2015H1_pct']),
       row('2015-07 起年化 %（S97 定義：基底 2015-06-30 收盤）', p(c630(nav['base'])), p(c630(nav['lim7'])), sb['cagr_since2015_07_pct'], sl['cagr_since2015_07_pct']),
       row('2015-07 起年化 %（v2 正式定義：基底 2015-07-01 收盤；S97 無此口徑，沿用上列比）', p(b['cagr_since2015_07']), p(l['cagr_since2015_07']), sb['cagr_since2015_07_pct'], sl['cagr_since2015_07_pct']),
       row('2019 起年化 %（基底 2018-12-28）', p(b['cagr_since2019']), p(l['cagr_since2019']), sb['cagr_since2019_pct'], sl['cagr_since2019_pct']),
       row('最大回撤 %', p(b['maxdd']['maxdd']), p(l['maxdd']['maxdd']), sb['maxdd_pct'], sl['maxdd_pct']),
       row('期末金額（100 萬起）', round(b['final_nav']), round(l['final_nav']), sb['final_nav'], sl['final_nav'], 'n'),
       row('已出場筆數', b['n_closed'], l['n_closed'], sb['n_closed'], sl['n_closed'], 'n'),
       row('全部筆數（含期末未平倉 8）', b['n_all'], l['n_all'], sb['n_all'], sl['n_all'], 'n'),
       row('開盤鎖漲停不買次數（全期）', b['st']['blk_limup'], l['st']['blk_limup'], sb['blk_limup'], sl['blk_limup'], 'n'),
       row('開盤鎖跌停順延賣次數（全期）', b['st']['sell_defer_limdn'], l['st']['sell_defer_limdn'], sb['sell_defer_limdn'], sl['sell_defer_limdn'], 'n')]
yearly = [dict(year=x['year'], base_pct=round(x['strat'] * 100, 2), lim7_pct=round(y['strat'] * 100, 2), diff_pp=round((y['strat'] - x['strat']) * 100, 2), e0050_pct=round(x['e0050'] * 100, 2),
               base_win=bool(x['strat'] > x['e0050']), lim7_win=bool(y['strat'] > y['e0050'])) for x, y in zip(b['yearly'], l['yearly'])]
pd_ = lambda k: [(pl['paths'][q][k] - pb['paths'][q][k]) * 100 for q in pb['paths']]
out = dict(
    id='R97', date='2026-11-23', 結論='重現成功：S97 的 7% 修正數字全部重現（標準路徑期末金額、筆數逐項相同）；修正幅度在路徑雜訊內。',
    基準重現=dict(target=dict(cagr=0.235577, final_nav=53392744, n_all=632, maxdd=-0.4921), got=dict(cagr=b['cagr'], final_nav=b['final_nav'], n_all=b['n_all'], n_closed=b['n_closed'], maxdd=b['maxdd']),
              nav_same_index=b['accept']['same_index'], nav_max_abs_diff_vs_v2=b['accept']['max_abs_diff_vs_v2_nav'], trades_same_as_v2=b['accept']['trades_same'],
              my_10pct_limit_formula_vs_engine=b['info']['my10_vs_engine'], passed=bool(b['accept']['max_abs_diff_vs_v2_nav'] == 0 and b['accept']['trades_same'])),
    實作=dict(判定位置=['cslib.py 第 208～209 行：Panel.__init__ 算 LIMUP／LIMDN，固定 REF×1.10／REF×0.90（全期間同一個 10%）',
                    's10lib.py simulate_s10：買進「o >= P.LIMUP[t, j] - 1e-9 → blk_limup、不買不遞補」；賣出「o <= P.LIMDN[t, j] + 1e-9 → sell_defer_limdn、順延」；rebalance() 減碼也跳過開盤 ≤ 跌停的持股',
                    's75lib.py 沒有重算漲跌停（只有 limit7_diag 做事後統計、不改結果）'],
            改法='外掛：run_r97.py 載入面板後，只在記憶體覆寫 P.LIMUP[:2084]、P.LIMDN[:2084]（2084＝2015-06-01 的列號；之前各列用 7%，2015-06-01 起原值不動）；沒有改任何既有檔。',
            公式='漲停＝floor(REF×1.07÷tick + 1e-9)×tick；跌停＝ceil(REF×0.93÷tick − 1e-9)×tick；tick 取「未取整的漲跌停價」所在級距（<10：0.01；10～<50：0.05；50～<100：0.1；100～<500：0.5；500～<1000：1；≥1000：5）',
            與引擎寫法差異='級距、取整方向、1e-9 容差與 cslib 完全相同（自寫 10% 版對引擎 LIMUP／LIMDN 全面板逐格差 0）；唯一差別是 2015-06-01 前的百分比。',
            參考價='REF＝收盤−漲跌價差（cslib 第 199～207 行）。2014 起除權息日用 TWT49U 參考價（EXRIGHT_FIX）；減資復牌事件檔 2015-08-24 起才有（REDUCE_FIX）；2014 以前除權息日原始檔漲跌價差＝0 → REF＝當日收盤（不是除權息參考價），這一段沒有事件參考價可用。',
            lim7_cells_changed=l['info']['cells_changed']),
    比對=cmp, 全部在0_3pp內且結論相同=bool(all(r['within_0_3'] for r in cmp if '正式定義' not in r['item'])),
    修正後=dict(cagr=l['cagr'], cagr_2008_2015H1=l['cagr_2008_2015H1'], cagr_since2015_07_base0701=l['cagr_since2015_07'], cagr_since2015_07_base0630=c630(nav['lim7']), cagr_since2019=l['cagr_since2019'],
             maxdd=l['maxdd'], final_nav=l['final_nav'], n_closed=l['n_closed'], n_all=l['n_all'], win_rate=l['win_rate'], years_win=l['years_win'], years_lose=l['years_lose'],
             cagr_0050=l['cagr_0050'], final_nav_0050=l['final_nav_0050']),
    基準=dict(cagr=b['cagr'], cagr_2008_2015H1=b['cagr_2008_2015H1'], cagr_since2015_07_base0701=b['cagr_since2015_07'], cagr_since2015_07_base0630=c630(nav['base']), cagr_since2019=b['cagr_since2019'],
            maxdd=b['maxdd'], final_nav=b['final_nav'], n_closed=b['n_closed'], n_all=b['n_all'], win_rate=b['win_rate'], years_win=b['years_win'], years_lose=b['years_lose']),
    逐年=yearly, 勝負年數=dict(base=[b['years_win'], b['years_lose']], lim7=[l['years_win'], l['years_lose']], 輸的年份_base=[y['year'] for y in yearly if not y['base_win']], 輸的年份_lim7=[y['year'] for y in yearly if not y['lim7_win']]),
    受影響交易=dict(直接_名單開盤等於7pct漲停=A['locked7'], 直接摘要=A['locked7_summary'], 用原始前收獨立重算=X['list_pre_cut_by_raw_prev_close'],
               交易清單差異=A['trade_diff'], 賣出=dict(sell_defer_limdn_base=b['st']['sell_defer_limdn'], sell_defer_limdn_lim7=l['st']['sell_defer_limdn'], 說明='標準路徑 2015-06-01 前 198 筆賣出，沒有任何一筆開盤 ≤ 7% 跌停價；賣出端 0 筆受影響')),
    paths20=dict(base_median=pb['median'], lim7_median=pl['median'], base_range=[pb['cagr_full_min'], pb['cagr_full_max']], lim7_range=[pl['cagr_full_min'], pl['cagr_full_max']],
                 paired_lim7_minus_base_pp={k: dict(median=float(np.median(pd_(k))), min=min(pd_(k)), max=max(pd_(k)), n_neg=int(sum(v < 0 for v in pd_(k))), n_pos=int(sum(v > 0 for v in pd_(k)))) for k in ('cagr_full', 'cagr_2008_2015H1', 'maxdd')},
                 blk_limup_total=[sum(q['blk_limup'] for q in pb['paths'].values()), sum(q['blk_limup'] for q in pl['paths'].values())],
                 sell_defer_limdn_total=[sum(q['sell_defer_limdn'] for q in pb['paths'].values()), sum(q['sell_defer_limdn'] for q in pl['paths'].values())]),
    制度切換日_資料驗證=A['regime_from_data'],
    跌幅減半_2008_10=dict(資料=X['oct2008'], 欄位='日期、檔數、最低<−3.6% 檔數、收盤在 −3.3%～−3.6% 檔數、收盤<−6.6% 檔數', 結論='2008-10-13～2008-10-24 共 10 個交易日，全市場沒有任何一檔最低價低於前收 −3.6%（10-24 有 1 檔例外）→ 當時跌停是 3.5%，引擎沒有這條規則',
                    lim7h_情境='run_r97.py lim7h：這 10 天跌停改 3.5% → 標準路徑逐日與 lim7 完全相同；20 條路徑未跑', lim7h_final=lh['final_nav'], lim7h_rows=lh['info'].get('halved_rows')),
    除權息日成交=dict(說明='2015-06-01 前成交落在原始檔 xflag＝1（除權息／減資）日的筆數；這些日子引擎 REF＝當日收盤', rows=A['fills_scan_lim7']['on_xflag_day'], data_panel補格代號成交=dict(n=X['fills_from_data_panel_fill']['n'], n_inside_7pct=X['fills_from_data_panel_fill']['n_inside_7pct'], n_at_limit=X['fills_from_data_panel_fill']['n_at_limit'])),
    resources=dict(maxrss_mb=max(b['maxrss_mb'], l['maxrss_mb'], lh['maxrss_mb'], pb['maxrss_mb'], pl['maxrss_mb']), runtime_s=dict(base=b['runtime_s'], lim7=l['runtime_s'])))
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
for r in cmp: print(r)
for y in yearly: print(y)
print(out['全部在0_3pp內且結論相同'], out['resources'])

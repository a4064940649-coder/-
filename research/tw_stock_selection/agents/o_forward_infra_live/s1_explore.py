#!/usr/bin/env python3
"""探索（只描述，不改規格）：時間線、資料覆蓋、處置名單來源一致性、history 單獨 vs 面板拼接的名單一致性。寫 explore.json。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TW = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(TW))
import fwd_data as FD  # noqa
import main_picks_today as MP  # noqa

src = FD.Source()
res = {}
# 1. 時間線（2026-11）
cal_nov = [d for d in pd.date_range('2026-11-02', '2026-12-31') if src.is_trading(d)]
cal_nov = pd.DatetimeIndex(cal_nov)
D = cal_nov[cal_nov >= '2026-11-10'][0]; k = cal_nov.get_loc(D)
J = [str(cal_nov[k + j].date()) for j in range(13)]
Ddec = cal_nov[cal_nov >= '2026-12-10'][0]; kd = cal_nov.get_loc(Ddec)
res['timeline_2026_11'] = dict(D_m_J0=str(D.date()), official_J1=J[1], exec_official=J[2], J=J, J12=J[12],
                               dec_D=str(Ddec.date()), dec_J12=str(cal_nov[kd + 12].date()),
                               jmed_final_after=str(cal_nov[kd + 12].date()) + ' 收盤（11 月錨定的 13 個 J 窗全部完成）',
                               holidays_used=sorted(str(h.date()) for h in src.holidays if h >= pd.Timestamp('2026-11-01')),
                               note='tw_holidays.json 2026 年 11～12 月只列 12-25；若有颱風假等臨時休市，J 日期以實際交易日為準（產生器只用已發生的交易日）')
# 2. 資料覆蓋
rh = src.rh.copy()
cov = rh.groupby('ym').agg(n=('code', 'size'), ann=('ann_date', lambda s: int(s.notna().sum()))).tail(6)
res['rev_ann_date_coverage'] = {k_: dict(n=int(v.n), with_ann=int(v.ann)) for k_, v in cov.iterrows()}
a = rh.dropna(subset=['ann_date']).copy(); a['ann'] = pd.to_datetime(a['ann_date'])
a['dl'] = [src.deadline(y) for y in a['ym']]
a['late'] = a['ann'] > a['dl']
res['rev_ann_vs_deadline'] = dict(n=int(len(a)), late=int(a['late'].sum()), late_share=float(a['late'].mean()),
                                  late_examples=a[a['late']][['code', 'ym', 'ann_date']].head(10).values.tolist())
res['history'] = dict(start=str(src.hcal[0].date()), end=str(src.hcal[-1].date()), n_days=len(src.hcal),
                      note='上游只保留約 260 個交易日（滾動）；2026-11-10 起訊號日在 history 內已有 ≥252 日 → 單獨使用模式')
d5 = src.d5.copy(); d5['date'] = pd.to_datetime(d5['date'])
w = src.wh[src.wh.flag == '處置'].copy(); w['date'] = pd.to_datetime(w['date'])
res['disposal_sources'] = dict(disposal_5y_last=str(d5['date'].max().date()), warnings_hist_first=str(w['date'].min().date()),
                               warnings_hist_last=str(w['date'].max().date()))
lo, hi = w['date'].min(), d5['date'].max()
k5 = set(zip(d5[(d5.date >= lo) & (d5.date <= hi)].code, d5[(d5.date >= lo) & (d5.date <= hi)].start))
kw = set(zip(w[(w.date >= lo) & (w.date <= hi)].code, w[(w.date >= lo) & (w.date <= hi)].start))
res['disposal_overlap'] = dict(window=[str(lo.date()), str(hi.date())], n_disposal_5y=len(k5), n_warnings_hist=len(kw), both=len(k5 & kw),
                               only_5y=sorted(map(str, k5 - kw))[:10], only_wh=sorted(map(str, kw - k5))[:10])
mg = src.disposal(); mg['start'] = pd.to_datetime(mg['start']); mg['end'] = pd.to_datetime(mg['end'])
lst = src.lst.set_index('code')
mg['market'] = mg.code.map(lst['market'])
rec = mg[mg.start >= '2026-08-10']
nd = [int(np.busday_count(s.date(), (e + pd.Timedelta(days=1)).date())) for s, e in zip(rec.start, rec.end)]
res['disposal_new_regime'] = dict(n=len(rec), by_market=rec.market.value_counts().to_dict(), busdays_median=float(np.median(nd)) if nd else None,
                                  busdays_dist=pd.Series(nd).value_counts().sort_index().to_dict())
# 3. 拼接用 panel_pre（< history 起日）vs 完整研究面板（至 2026-10-22）：同一訊號日名單是否相同
asof = src.hcal[-1]
adir = HERE / 'work' / 'explore_asof'
src.build_asof(asof, adir); pnl = src.panel_pre(HERE / 'work')
dh = MP.load_prices(adir, asof, FD.PANEL_SRC)            # 完整研究面板（含 asof 前全部日期；只供比對）
ds_ = MP.load_prices(adir, asof, pnl)                    # panel_pre（同 make_forward）
Y, Y3, _ = MP.load_revenue(adir); dsp = MP.load_disposal(adir)
sd = MP.signal_day(dh['cal'], asof)
o1, _, _, n1 = MP.score_on(dh, Y3, dsp, sd, ex_fin=True)
o2, _, _, n2 = MP.score_on(ds_, Y3, dsp, sd, ex_fin=True)
res['panel_pre_vs_full_panel'] = dict(signal=str(sd.date()), mode_full=dh['mode'], mode_pre=ds_['mode'], n_el=[n1, n2],
                                     top10_full=list(o1.index[:10]), top10_pre=list(o2.index[:10]), same=list(o1.index[:10]) == list(o2.index[:10]),
                                     top30_same=list(o1.index[:30]) == list(o2.index[:30]),
                                     note='2026-11-10 起訊號日在 history 內已有 ≥252 日，改用 history 單獨模式；此比對只涵蓋重播期')
import shutil; shutil.rmtree(adir)
# 4. 產生器瑕疵與修正
res['o8_flaws'] = [
    dict(id='F-A', flaw='訊號日當晚實跑不寫入名單（MP.execute 隔日無開盤價 → continue）', fix='快照只需訊號日資料：凍結完整排序；成交日再依凍結排序套 M2'),
    dict(id='F-B', flaw='--asof 不截斷價格（Store 永遠讀全部 history）', fix='fwd_data.build_asof 截斷全部輸入＋assert；每個訊號日用截斷到該日的資料算名單'),
    dict(id='F-C', flaw='J0 遇 ann_date 缺值視為已公告（前視）', fix='J0 缺值＝未公告（沿用 m−2）；J≥1 缺值＝法定期限已過視為已公告；m−1 不存在一律沿用 m−2'),
    dict(id='F-D', flaw='決策線成本 K5T-lite（無零股、現金、參與上限、處置賣出），每窗偏樂觀 0.3～0.6pp', fix='逐字載入 V7X x_exec.py（xlive.py 資料模組）'),
    dict(id='F-E（本題新發現）', flaw='disposal_5y 停在 2026-10-05，處置排除與處置衛星會漏', fix='disposal_5y ∪ warnings_hist（處置，2026-06-22 起每日）'),
    dict(id='F-F（本題新發現）', flaw='本地 tw-stock-screener 停在 10-27 且工作樹有未提交的修改，git pull 會中止', fix='本題 raw/screener 自建稀疏 clone，每日 fetch＋checkout；不動共用 clone'),
    dict(id='F-G（本題新發現）', flaw='trades.csv 舊設計只放實際成交；若混入紙上成交，s8_forward_calib.py 會把紙上衝擊當觀測', fix='trades.csv 加 kind 欄（paper_*／real）；校準只用 kind=real（calib_real.py）'),
]
res['paper_lines_inventory'] = dict(
    tracked_here=['MAIN_OFFICIAL', 'J0..J12', 'JMED', 'E50', 'DISP_TWSE', 'DISP_TWSE_FIRST', 'DISP_TPEX', 'MIX_C70', 'MIX_D80'],
    not_tracked_here={'CAPM_F1／C50／C70（o7 capmatched）': 'o7 判不可用（機制部分支持）；o9 題 4 PIT 重驗中，由 live_tracking/o7_capmatched 自己的腳本記錄',
                      'REGIME_V5（o7 輪動）': 'o7 判不可用（機制不支持）；只檢驗機制，由 live_tracking/o7_regime 記錄',
                      'R_ROLL': 'o8：時間戳覆蓋 33.6%，閘門沒過，只做前瞻描述（o_rev_freshness_attention/forward）',
                      'R2': 'o8 判不可用', 'T1／T2（0050 成分內傾斜）': 'o8 判不可用，未登記前瞻腳本',
                      'M_E50_30、M_2330_30': '舊產生器的混合線；本題依題目只保留 C_E50_70（＝MIX_C70）與 MIX_D80'})
(HERE / 'explore.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k_: res[k_] for k_ in ('timeline_2026_11', 'disposal_overlap', 'panel_pre_vs_full_panel', 'rev_ann_vs_deadline')}, ensure_ascii=False, default=str)[:3000])

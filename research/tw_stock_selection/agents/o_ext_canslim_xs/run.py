#!/usr/bin/env python3
"""o_ext_canslim_xs：CAN SLIM 台股版（XQ/XS 腳本）忠實重做與回測。
用法（一律 python3 -I -B）：
  python3 -I -B run.py gate      # 資料閘門（不算任何報酬）→ results.json['GATE']
  python3 -I -B run.py prereg    # 事前登記（不算任何報酬）→ results.json['PREREG']（含 sha256 與時間）
  python3 -I -B run.py main      # 全部回測（需先有 PREREG，且 sha256 必須與程式內的登記內容一致）
中間檔：scratchpad/ext_canslim/panel_cs.npz（可刪，會自動重建，約 40 秒）。"""
import json, sys, time, datetime, gzip
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import cslib as cs

RES = HERE / 'results.json'
START = '2017-03-31'     # 第一個訊號日（FY2016 年報法定期限；此日起 EpsCAGR 才對全體公司可算）
A2019 = '2019-01-01'
SEGS = [('2017-03-31', '2018-12-31'), ('2019-01-01', '2023-12-31'), ('2024-01-01', '2026-10-22')]
NB = [('nb_eps15', dict(MinEpsQYoY=15.0)), ('nb_eps25', dict(MinEpsQYoY=25.0)), ('nb_rev15', dict(MinRevYoY3=15.0)), ('nb_rev25', dict(MinRevYoY3=25.0)),
      ('nb_high5', dict(MaxFromHigh=5.0)), ('nb_high15', dict(MaxFromHigh=15.0)), ('nb_trail8', dict(TrailPct=8.0)), ('nb_trail10', dict(TrailPct=10.0)),
      ('nb_mkt120', dict(MarketMA=120))]
ABL = ['C', 'A_cagr', 'A_rev', 'N', 'S', 'L', 'M']

PREREG = {
 'title': 'o_ext_canslim_xs 事前登記（寫於任何報酬計算之前）',
 'source': 'raw/canslim_entry.xs、raw/canslim_exit.xs、raw/SOURCE.txt（原文不改）',
 'data': {
  'price': '官方日 K（jb7k6295r7-bot/tw-stock-data commit 107f9073 的 data/universe/daily，2015-01-05～2026-10-30，上市＋上櫃、含日後下市股、含成交股數與逐日市場別）；只留四碼數字非 0 開頭代號；截到 2026-10-22（0050／加權指數檔最後一天）',
  'eps': 'o_pead fin_long.parquet（MOPS XBRL 單季，2013Q1 起）＋ avail_dates.parquet（公告日 A；缺者用法定期限 5/15、8/14、11/14、次年 3/31）',
  'revenue': 'o_pead rev_long.parquet 的 yoy_pct（MOPS 月營收彙總表當時值，2015-01 起）',
  'index': 'finlab-txd taiex_twii.csv 收盤；0050 含息：etf_0050.csv 的 adj',
  'disposal': 'o_disposal_list_pit_audit_pkl18/data/universe_announce.csv（只有上市）'},
 'signal_rules': {
  'bars': '以市場交易日曆為 K 棒；無成交日沿用前一收盤（高＝收、量＝0）。主欄用未還原原始價（XQ 預設是否還原：需查證）；還原價列為變體 ADJ',
  'C': 'EpsQYoY＝(Q0/Q4−1)×100，Q0＝訊號日已可得的最新一季單季 EPS、Q4＝去年同季；Q4≤0 或缺值 → −999；門檻 ≥18',
  'A_cagr': 'EpsCAGR＝((Y0/Y3)^(1/3)−1)×100，Y0＝已可得的最新年度 EPS（第四季累計 EPS；缺則四季單季加總）、Y3＝往前第 3 年；兩者都 >0 才算；門檻 ≥15',
  'A_rev': 'RevYoY3＝已公告的最近 3 個月月營收年增率平均，缺任一月 → −999；門檻 ≥20',
  'N': 'FromHigh＝(收盤/近 240 日最高「最高價」−1)×100 ≥ −10，且已有 ≥240 根日 K',
  'S': '近 20 日平均成交量（張＝成交股數/1000）≥ 500',
  'L': 'StockRet＝收盤/120 日前收盤−1；IndexRet＝加權指數同期；StockRet > IndexRet 且 StockRet > 0',
  'M': '加權指數收盤 ≥ 其 60 日均線',
  'PIT': '季報：公告日 A ≤ 訊號日；月營收 m：次月 10 日（非交易日順延）之後的第一個交易日起可用',
  'nonPIT': '季報：季末日起可用（年報＝12/31 起）；月營收：當月最後一日起可用。只用來量化前視灌水',
  'exit': '收盤 ≤ 近 20 日最高收盤×(1−6%)；或 8 日均線向下穿越 20 日均線（今日快<慢 且 昨日快≥慢）。自進場當日收盤起檢查；成立後次日開盤賣'},
 'start': '第一個訊號日 2017-03-31（FY2016 年報法定期限；EpsCAGR 需要 FY2016 與 FY2013，fin_long 自 2013Q1 起）。2014-11～2017-03 無法照腳本算 A 條件，不回測、不用未來資料補。非 PIT 版用同一起點以便比較',
 'post_caliber': {
  'trade': '訊號日 t 盤後入選且當時未持有 → t+1 開盤買（無開盤價則放棄該次訊號）；出場訊號日 k → k+1 開盤賣（無開盤價則順延）；賣出當日收盤即可再入選；期末未平倉以最後收盤結算並標記',
  'ret': '每筆報酬主欄＝賣出開盤/買進開盤−1（未還原，貼文口徑）；另報含息報酬',
  'curve': '時間加權（本題定義）＝每日「持有中各筆的含息日報酬等權平均」連乘，無持股日報酬 0；另報固定每筆金額帳戶：需要資金＝同時持有筆數最大值×每筆金額，帳戶報酬＝總損益/需要資金（不複利）；閒置比例＝1−平均持有筆數/最大持有筆數',
  'cost': '扣成本版：買 0.1425%×0.28＋半跳；賣 0.1425%×0.28＋0.3%＋半跳',
  'combos': '只上市／上市＋上櫃 × PIT／非 PIT × 扣／不扣成本'},
 'executable': {
  'MAIN': '只上市（訊號日當天市場別＝twse）、PIT、原始價、預設參數、100 萬、同時最多 10 檔、一般成本、空位放現金（利率 0）',
  'ranking': '訊號多於空位時依訊號日的「超越大盤%」（StockRet−IndexRet，腳本輸出欄 5）由大到小；同分依代號由小到大。理由：腳本唯一每日更新、無極端值的連續強度欄位（L）。事前寫死，不看報酬',
  'sizing': '每檔目標＝成交日開盤估算的總權益/檔數；不得超過現金；2020-10-26 前只買整張（買不起一張就跳過換下一名），之後可零股；單筆 ≤ 當日成交值 5%（買超過就縮量；賣超過就分日）',
  'fills': '買＝開盤×(1＋滑價)，賣＝開盤×(1−滑價)，滑價＝0.5×跳動點/開盤價；開盤 ≥ 漲停價不買（該次訊號作廢）；開盤 ≤ 跌停價或無開盤價不賣、順延；漲跌停價＝參考價×1.1／0.9 依跳動點取整，參考價＝收盤−漲跌價差',
  'cost': '一般：手續費 0.1425%×0.28 每邊、每筆最低 20 元（整股與零股各一筆）、賣出證交稅 0.3%；壓力：每邊滑價再加 0.3%；零成本：無費用無滑價（回測上緣，實現不了）',
  'dividends': '持股市值以含息日報酬（收盤/參考價−1）滾動，等同除權息零成本再投入',
  'delist': '持股自官方日 K 消失（之後不再出現）→ 以最後收盤價結清，另報筆數',
  'disposal': '買進日落在上市官方處置期間的筆數與報酬另列，不排除',
  'variants': ['S20（20 檔）', 'C30／C300／C1000（30 萬、300 萬、1,000 萬）', 'STRESS', 'ZERO', 'E50（閒置資金放 0050；買 0.0399%＋0.03%、賣 0.0399%＋0.1%＋0.03% 的近似換手成本）',
               'NONPIT', 'ALL（上市＋上櫃）', 'ALL_NONPIT', 'ADJ（還原價訊號）', 'RK_EPS（依 EpsQYoY 排序）', 'RK_LIQ（依 20 日均量排序）',
               '參數鄰域 9 個（一次改一個）：' + '、'.join(n for n, _ in NB), '消融 7 個（一次拿掉一個條件）：' + '、'.join(ABL),
               'J 偏移：起始訊號日往後 0、5、…、60 個交易日共 13 條，同規則（不算新策略），報 2019 起超額的中位數與 p10']},
 'stats': {
  'primary': '2019-01-01～2026-10-22 年化（日曆年化）對 0050 含息同期；放第一',
  'full': '2017-03-31～2026-10-22',
  'segments': '2017-03-31～2018-12-31、2019～2023、2024～2026-10-22；逐年',
  't': '日曆月報酬（月底對月底）策略−0050 的平均 ÷ (標準差 ÷ √n)',
  'drop_best3': '拿掉年度超額最高的 3 個日曆年後重算月超額 t',
  'turnover': '單邊周轉＝買進金額 ÷ 平均淨值 ÷ 年數；成本拖累＝ZERO 年化 − MAIN 年化（pp），另報費用＋稅＋滑價 ÷ 平均淨值 ÷ 年數',
  'overlap': '主策略近似重建（每月 11 日後第一個交易日、上市非金融、60 日均值 ≥1,000 萬、收盤 ≥10、非處置、均值前 300，0.5×52 週比百分位＋0.5×營收 3 月年增百分位，前 10 檔）；重疊率＝CAN SLIM 持股日落在主策略當期前 10（另報前 30）的比例；報酬相關＝MAIN 月報酬與 main_v2 navs_gapfix「EXFIN|n10|normal」月報酬（及兩者對 0050 月超額）的相關'},
 'gates': {
  'G1': 'MAIN 2019 起與全期超額 > 0，STRESS 也要；13 條 J 偏移的 2019 起超額 p10 > 0',
  'G2': '全期月超額 t ≥ 3.0 且 2019 起 t ≥ 2.0',
  'G3': '族群 N＝2,573＋本題新增；主欄 t 未達 Bonferroni 門檻即判不過；本題不重建全族群報酬矩陣，不跑完整 SPA（若主欄 t ≥ 2 才需要另跑）',
  'G4': '三個分段超額都 > 0；拿掉最好 3 年 t ≥ 2.0；9 個鄰域參數的 2019 起超額不翻號',
  'G5': '全部 PIT、含下市股、無前視', 'G6': '另一位研究員只看規格重做，關鍵數字一致', 'G7': '100 萬、300 萬、1,000 萬的 2019 起超額都 > 0',
  'G8': '凍結後紙上追蹤滿 12 個月超額 > 0',
  'verdict': '任一條不過＝不可投資'},
 'family': {'N_before': 2573,
            'new': ['MAIN', 'S20', 'C30', 'C300', 'C1000', 'STRESS', 'ZERO', 'E50', 'NONPIT', 'ALL', 'ALL_NONPIT', 'ADJ', 'RK_EPS', 'RK_LIQ']
                   + [n for n, _ in NB] + ['abl_' + a for a in ABL] + ['POST_TWSE_PIT', 'POST_TWSE_NONPIT', 'POST_ALL_PIT', 'POST_ALL_NONPIT']},
 'not_tested': '貼文建議的「更嚴格大盤濾網／市場寬度」不測（新規則，不在本題）',
}


def prereg_sha():
    return cs.sha_bytes(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode('utf-8'))


def load_res():
    return json.load(open(RES)) if RES.exists() else {}


def save_res(r):
    json.dump(r, open(RES, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def rnd(o, k=6):
    if isinstance(o, dict):
        return {kk: rnd(v, k) for kk, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rnd(v, k) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), k)
    if isinstance(o, np.integer):
        return int(o)
    return o


# =============================================================== gate
def stage_gate():
    import glob, os
    P = cs.Panel(); g = {}
    g['panel'] = dict(days=P.T, codes=P.N, first=str(P.IDX[0].date()), last=str(P.IDX[-1].date()),
                      codes_ever_twse=int((P.MK == 1).any(0).sum()), codes_ever_tpex=int((P.MK == 2).any(0).sum()),
                      codes_both=int(((P.MK == 1).any(0) & (P.MK == 2).any(0)).sum()), codes_gone_by_last_day=int((P.MK[-1] == 0).sum()),
                      bad_ret_cells_gt50pct=P.bad_ret, twii_missing_days=P.twii_missing, e50_missing_days=P.e50_missing,
                      stocks_per_day_median_by_year={int(k): float(v) for k, v in pd.Series(P.listed.sum(1), index=P.IDX).groupby(P.IDX.year).median().items()},
                      panel_sha256=cs.sha_file(cs.TMP / 'panel_cs.npz'))
    yr = P.IDX.year.to_numpy()
    def agree(path, name):
        x = pd.read_parquet(path).reindex(P.IDX); common = [c for c in x.columns if c in P.ci]; jj = [P.ci[c] for c in common]
        a = np.where(P.has[:, jj], P.C[:, jj], np.nan); b = x[common].to_numpy(); m = np.isfinite(a) & np.isfinite(b); d = np.abs(a - b) / b
        g[name] = {int(y): round(float((d[m & (yr == y)[:, None]] < 1e-3).mean()), 5) for y in range(2015, 2027)}
    agree(str(cs.SCR / 'panel/close_raw.parquet'), 'close_agree_vs_scratch_panel_by_year')
    agree(str(cs.AG / 'o_v6_price_panel_official/data/close_raw.parquet'), 'close_agree_vs_V6_official_by_year')
    # 成交股數對 TWSE 個股日成交（tsec_all，到 2019-05）
    rng = np.random.default_rng(1); files = [f for f in sorted(glob.glob(str(cs.SCR / 'tsec_all/*.csv'))) if os.path.basename(f)[:-4] in P.ci]
    tot = {1: 0, 2: 0}; ok = {1: 0, 2: 0}; med = {1: [], 2: []}
    for f in rng.choice(files, 150, replace=False):
        c = os.path.basename(f)[:-4]
        try:
            x = pd.read_csv(f, header=None, usecols=[0, 1], names=['d', 'vol'], dtype=str)
        except Exception:
            continue
        x = x[x['d'].str.match(r'^10[4-8]/')]
        if len(x) == 0:
            continue
        dt = [pd.Timestamp(int(s.split('/')[0]) + 1911, int(s.split('/')[1]), int(s.split('/')[2])) for s in x['d']]
        w = pd.Series(pd.to_numeric(x['vol'].str.replace(',', ''), errors='coerce').to_numpy(), index=dt)
        w = w[~w.index.duplicated()]
        v = pd.Series(P.VOL[:, P.ci[c]], index=P.IDX).reindex(w.index); mk = pd.Series(P.MK[:, P.ci[c]], index=P.IDX).reindex(w.index)
        for k in (1, 2):
            mm = (w > 0) & (v > 0) & (mk == k)
            if mm.sum():
                tot[k] += int(mm.sum()); ok[k] += int(((v[mm] / w[mm] - 1).abs() < 1e-9).sum()); med[k].append(float((v[mm] / w[mm] - 1).median()))
    g['volume_vs_tsec_all'] = dict(files=150, period='2015-01～2019-05', twse=dict(cells=tot[1], exact_agree=round(ok[1] / max(tot[1], 1), 5)),
                                   tpex=dict(cells=tot[2], exact_agree=round(ok[2] / max(tot[2], 1), 5), median_rel_diff=float(np.median(med[2])) if med[2] else None,
                                             note='上櫃官方日 K 的成交股數以千股為單位（不含零股等），與對照來源差約千分之幾'))
    vol20 = cs.roll_mean(P.VOLF / 1000, 20); app = cs.roll_mean(np.where(P.has, P.AMT / P.C / 1000, 0.0), 20)
    mm = np.isfinite(vol20) & (vol20 > 0) & np.isfinite(app); r = app[mm] / vol20[mm] - 1
    g['approx_lots_value_over_close'] = dict(note='本題用真實成交股數；此處只量化「成交值÷收盤÷1000」近似的誤差', median=float(np.median(r)), p5=float(np.quantile(r, .05)), p95=float(np.quantile(r, .95)),
                                             S500_disagree=float(((vol20 >= 500) != (app >= 500))[mm].mean()))
    f = pd.read_parquet(cs.PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'eps_s', 'eps_ytd'])
    f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')]
    s4 = f.pivot(index=['code', 'y'], columns='q', values='eps_s').sum(axis=1, skipna=False); a4 = f[f['q'] == 4].set_index(['code', 'y'])['eps_ytd']
    d = (s4 - a4).dropna()
    g['annual_eps_sum4_vs_ytdQ4'] = dict(n=int(len(d)), within_0p011=float((d.abs() <= 0.011).mean()), within_0p05=float((d.abs() <= 0.05).mean()))
    g['fin_first_period'] = str(f.sort_values(['y', 'q']).iloc[0][['y', 'q']].tolist()); g['pead_quality_json'] = json.load(open(cs.PEAD / 'quality.json')).get('eps_vs_hawan')
    for pit in (True, False):
        e = cs.load_eps(P, pit); k = 'PIT' if pit else 'nonPIT'
        cy = pd.Series(((e['EpsY0'] > 0) & (e['EpsY3'] > 0) & P.listed).sum(1), index=P.IDX)
        any_y = pd.Series((np.isfinite(e['EpsY0']) & np.isfinite(e['EpsY3']) & P.listed).sum(1), index=P.IDX)
        r3 = cs.load_rev3(P, pit); c3 = pd.Series((np.isfinite(r3) & P.listed).sum(1), index=P.IDX)
        cq = pd.Series((np.isfinite(e['EpsQ0']) & np.isfinite(e['EpsQ4']) & P.listed).sum(1), index=P.IDX)
        days = ['2016-12-30', '2017-01-03', '2017-02-15', '2017-03-15', '2017-03-31', '2017-04-05', '2018-04-10', '2022-04-11', '2026-10-22']
        g['coverage_' + k] = dict(info=e['info'], first_day_Y0_and_Y3_available=str(any_y[any_y > 0].index[0].date()),
                                  n_Y0Y3_both_positive={d: int(cy.asof(pd.Timestamp(d))) for d in days},
                                  n_Q0Q4_available={d: int(cq.asof(pd.Timestamp(d))) for d in days}, n_rev3_available={d: int(c3.asof(pd.Timestamp(d))) for d in days})
    # 月營收年增率對另一來源（tw_revenue.csv 自行計算）
    r = pd.read_parquet(cs.PEAD / 'rev_long.parquet', columns=['code', 'ym', 'yoy_pct'])
    w = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}, parse_dates=['month'])
    w['ym'] = w['month'].dt.strftime('%Y-%m'); w = w.drop_duplicates(['code', 'ym']).set_index(['code', 'ym'])['revenue_k_twd']
    prev = w.copy(); prev.index = pd.MultiIndex.from_arrays([prev.index.get_level_values(0), (pd.PeriodIndex(prev.index.get_level_values(1), freq='M') + 12).strftime('%Y-%m')])
    yy = ((w / prev.reindex(w.index) - 1) * 100).rename('yoy2').reset_index()
    m = r.merge(yy, on=['code', 'ym'], how='inner').replace([np.inf, -np.inf], np.nan).dropna()
    g['rev_yoy_vs_tw_revenue_csv'] = dict(n=int(len(m)), within_1pp=float(((m['yoy_pct'] - m['yoy2']).abs() <= 1).mean()), within_5pp=float(((m['yoy_pct'] - m['yoy2']).abs() <= 5).mean()),
                                          rev_long_months=[r['ym'].min(), r['ym'].max()])
    # 0050 與加權指數
    e = pd.read_csv(cs.E50_CSV, parse_dates=['date']).set_index('date')['adj']; a, b = e.asof(pd.Timestamp('2014-11-03')), e.asof(pd.Timestamp('2026-10-22'))
    yrs = (pd.Timestamp('2026-10-22') - pd.Timestamp('2014-11-03')).days / 365.25
    g['e50_2014-11-03_2026-10-22'] = dict(total=float(b / a - 1), cagr=float((b / a) ** (1 / yrs) - 1), years=yrs, post_648p44_cagr_12y=float(7.4844 ** (1 / 12) - 1),
                                          post_648p44_cagr_same_days=float(7.4844 ** (1 / yrs) - 1))
    dz = cs.disposal_mask(P); g['disposal'] = dict(stock_days=int(dz.sum()), codes=int(dz.any(0).sum()), note='只有上市；上櫃處置資料不足')
    res = load_res(); res['GATE'] = rnd(g); save_res(res)
    print(json.dumps(rnd(g, 4), ensure_ascii=False, indent=1)[:6000])


def stage_prereg():
    res = load_res()
    assert 'GATE' in res, '先跑 gate'
    assert 'POST' not in res and 'EXEC' not in res, '已有回測結果，不得重寫 PREREG'
    res['PREREG'] = dict(spec=PREREG, sha256=prereg_sha(), written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                         cslib_sha256=cs.sha_file(HERE / 'cslib.py'), note='此時尚未計算任何交易報酬或淨值；只做過資料閘門（覆蓋率、來源一致率）')
    save_res(res); print(res['PREREG']['sha256'], res['PREREG']['written_utc'])



# =============================================================== main
def trade_stats(P, tr, t0, net=False):
    """貼文口徑逐筆統計。主欄報酬＝未還原價（賣出開盤/買進開盤−1）；net=True 再扣成本。"""
    O = P.O
    oe = O[tr['ent_t'].to_numpy(), tr['j'].to_numpy()]; closed = tr['open_end'].to_numpy() == 0
    ox = np.where(closed, O[tr['exit_t'].to_numpy(), tr['j'].to_numpy()], P.CF[tr['exit_t'].to_numpy(), tr['j'].to_numpy()])
    cb = cs.FEE + 0.5 * cs.tick(oe) / oe; cx = cs.FEE + cs.TAX + 0.5 * cs.tick(ox) / ox
    if not net:
        cb = np.zeros(len(tr)); cx = np.zeros(len(tr))
    rp = (1 + tr['ret_px'].to_numpy()) * (1 - cx) / (1 + cb) - 1
    rt = (1 + tr['ret_tr'].to_numpy()) * (1 - cx) / (1 + cb) - 1
    def blk(r):
        w = r > 0; aw = float(r[w].mean()) if w.any() else float('nan'); al = float(-r[~w].mean()) if (~w).any() else float('nan'); pw = float(w.mean())
        return dict(n=int(len(r)), n_win=int(w.sum()), n_loss=int((~w).sum()), win_rate=pw, mean=float(r.mean()), median=float(np.median(r)), max=float(r.max()), min=float(r.min()),
                    avg_win=aw, avg_loss=al, payoff=aw / al if al else float('nan'), expectancy=pw * aw - (1 - pw) * al)
    out = dict(px=blk(rp), tr=blk(rt))
    hd = (tr['exit_t'] - tr['ent_t']).to_numpy(); out['hold_days_mean'] = float(hd.mean()); out['hold_days_median'] = float(np.median(hd))
    out['n_open_end'] = int((~closed).sum()); out['n_hold_1day'] = int((hd <= 1).sum())
    out['n_by_entry_year'] = {int(k): int(v) for k, v in tr.groupby(tr['ent_date'].dt.year).size().items()}
    out['n_artifact_px_vs_tr_gt15pp'] = int((np.abs(tr['ret_px'] - tr['ret_tr']) > 0.15).sum())
    # 權益曲線
    cv = cs.post_curve(P, tr, t0, (cb, cx))
    twr = pd.Series(cv['twr'], index=P.IDX[t0:]); yrs = (P.IDX[-1] - P.IDX[t0]).days / 365.25
    no = cv['n_open'][t0 + 1:]
    out['curve'] = dict(twr_total=float(twr.iloc[-1] - 1), twr_cagr=float(twr.iloc[-1] ** (1 / yrs) - 1), twr_maxdd=float((twr / twr.cummax() - 1).min()),
                        twr_peak_total=float(twr.max() - 1), twr_peak_date=str(twr.idxmax().date()), twr_trough_before_peak=float(twr[:twr.idxmax()].min() - 1),
                        max_concurrent=int(no.max()), avg_concurrent=float(no.mean()), median_concurrent=float(np.median(no)), days_zero_pos=float((no == 0).mean()),
                        idle_ratio=float(1 - no.mean() / no.max()), sum_pnl_units=float(cv['pnl'].sum()),
                        fixed_account_total=float(cv['pnl'].sum() / no.max()), fixed_account_cagr=float((1 + cv['pnl'].sum() / no.max()) ** (1 / yrs) - 1), years=yrs)
    out['_twr'] = twr; out['_ret_px'] = rp; out['_ret_tr'] = rt
    # 出場拆解
    c = tr[closed]; r = rp[closed]
    kind = np.where((c['x_trail'] == 1) & (c['x_cross'] == 1), 'both', np.where(c['x_trail'] == 1, 'trail', 'cross'))
    out['exit_decomp'] = {k: dict(n=int((kind == k).sum()), share=float((kind == k).mean()), mean_px=float(r[kind == k].mean()), win_rate=float((r[kind == k] > 0).mean()),
                                  mean_tr=float(rt[closed][kind == k].mean()), hold_days=float(hd[closed][kind == k].mean())) for k in ('trail', 'cross', 'both')}
    # 依出場月份（2026-05～10）
    xm = tr['exit_date'].dt.to_period('M'); mm = {}
    for m in pd.period_range('2026-01', '2026-10', freq='M'):
        k = (xm == m).to_numpy() & closed
        if k.any():
            mm[str(m)] = dict(n=int(k.sum()), mean_px=float(rp[k].mean()), win_rate=float((rp[k] > 0).mean()), mean_tr=float(rt[k].mean()))
    out['by_exit_month_2026'] = mm
    # 依持有交易日數分組（含息報酬）
    hb = {}
    for lo, hi in ((1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 40), (41, 9999)):
        k = closed & (hd >= lo) & (hd <= hi)
        if k.any():
            hb['%d-%d' % (lo, hi)] = dict(n=int(k.sum()), share=float(k.sum() / closed.sum()), mean_tr=float(rt[k].mean()), win_rate=float((rt[k] > 0).mean()))
    out['by_hold_days'] = hb
    # 大盤濾網狀態（前一日收盤是否在 60 日線上）下的持有日與每持有日平均報酬
    ic = P.IDXC; ma = pd.Series(ic).rolling(60).mean().to_numpy(); mok = ~(ic < ma); mprev = np.concatenate([[True], mok[:-1]])
    no_ = cv['n_open'][t0 + 1:]; sr = (cv['dr'] * cv['n_open'])[t0 + 1:]; mp = mprev[t0 + 1:]
    out['market_filter_state'] = dict(pos_days_total=float(no_.sum()), pos_days_share_when_below_ma=float(no_[~mp].sum() / no_.sum()),
                                      mean_ret_per_pos_day_above=float(sr[mp].sum() / max(no_[mp].sum(), 1)), mean_ret_per_pos_day_below=float(sr[~mp].sum() / max(no_[~mp].sum(), 1)),
                                      days_below_share=float((~mp).mean()), sum_ret_below=float(sr[~mp].sum()), sum_ret_above=float(sr[mp].sum()))
    xy = tr['exit_date'].dt.year
    out['by_exit_year'] = {int(y): dict(n=int((xy == y).sum()), mean_px=float(rp[(xy == y).to_numpy()].mean()), win_rate=float((rp[(xy == y).to_numpy()] > 0).mean())) for y in sorted(xy.unique())}
    return out


def main_approx(P, rev3_pit, disp, t0):
    """主策略近似重建（非正式 V7X 名單）：回傳 (T,N) 布林：前 10、前 30 名的持有期間。"""
    f = pd.read_parquet(cs.PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'industry'])
    ind = f.sort_values(['y', 'q']).groupby('code')['industry'].last()
    okc = np.array([(not c.startswith('91')) and ind.get(c, 'ci') not in {'bd', 'basi', 'ins', 'fh', 'mim'} for c in P.COLS])
    V60 = cs.roll_mean(np.where(P.has, P.AMT, np.nan), 60, 50); R52 = P.G / cs.roll_max(P.G, 252, 240)
    s = pd.Series(P.IDX, index=P.IDX)
    days = [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([P.IDX.year, P.IDX.month]) if (g.dt.day >= 11).any()]
    top10 = np.zeros((P.T, P.N), bool); top30 = np.zeros((P.T, P.N), bool)
    for n, d in enumerate(days):
        t = P.IDX.get_loc(d); t2 = P.IDX.get_loc(days[n + 1]) if n + 1 < len(days) else P.T - 1
        if t < 300:
            continue
        pool = okc & (P.MK[t] == 1) & (V60[t] >= 1e7) & (P.CF[t] >= 10) & np.isfinite(R52[t]) & ~disp[t]
        w = np.where(pool)[0]; w = w[np.argsort(-V60[t, w])][:300]
        y3 = rev3_pit[t, w]; ok = np.isfinite(y3); w = w[ok]; y3 = y3[ok]
        if len(w) < 50:
            continue
        sc = 0.5 * pd.Series(R52[t, w]).rank(pct=True).to_numpy() + 0.5 * pd.Series(y3).rank(pct=True).to_numpy()
        order = w[np.argsort(-sc, kind='stable')]
        top10[t + 1:t2 + 1, order[:10]] = True; top30[t + 1:t2 + 1, order[:30]] = True
    return top10, top30


def stage_main():
    T0 = time.time()
    res = load_res()
    assert 'PREREG' in res and res['PREREG']['sha256'] == prereg_sha(), 'PREREG 不存在或內容與登記不符'
    P = cs.Panel(); t0 = int(P.IDX.get_loc(pd.Timestamp(START)))
    bench = pd.Series(P.E50, index=P.IDX)
    U = {'TWSE': P.MK == 1, 'ALL': P.MK > 0}
    eps = {True: cs.load_eps(P, True), False: cs.load_eps(P, False)}; rev = {True: cs.load_rev3(P, True), False: cs.load_rev3(P, False)}
    disp = cs.disposal_mask(P)
    cache = {}
    def sigset(pit=True, over=None, adj=False, drop=()):
        prm = dict(cs.DEF); prm.update(over or {})
        key = (pit, json.dumps(over or {}, sort_keys=True), adj)
        if key not in cache:
            if len(cache) > 2:
                for k in [k for k in cache if k != (True, '{}', False)][:1]:
                    del cache[k]
            cond, fld = cs.entry_fields(P, eps[pit], rev[pit], prm, adj); tr_, cr_ = cs.exit_signal(P, prm, adj)
            cache[key] = (cond, fld, tr_, cr_)
        cond, fld, tr_, cr_ = cache[key]
        return cs.entry_signal(cond, drop), fld, tr_, cr_, cond

    # ---------------- 訊號檔與訊號統計
    sig, fld, trail, cross, cond = sigset(True)
    tt, jj = np.where(sig[t0:] & U['ALL'][t0:]); tt = tt + t0
    S = pd.DataFrame({'date': P.IDX[tt].strftime('%Y-%m-%d'), 'code': P.COLS[jj], 'mkt': np.where(P.MK[tt, jj] == 1, 'twse', 'tpex'),
                      'EpsQYoY': np.round(fld['EpsQYoY'][tt, jj], 1), 'EpsCAGR': np.round(fld['EpsCAGR'][tt, jj], 1), 'RevYoY3': np.round(fld['RevYoY3'][tt, jj], 1),
                      'FromHigh': np.round(fld['FromHigh'][tt, jj], 2), 'RS': np.round(fld['RS'][tt, jj], 2), 'AvgQty': np.round(fld['AvgQty'][tt, jj], 0)})
    S.to_csv(HERE / 'signals.csv.gz', index=False, compression='gzip')
    SIG = {}
    for un in ('TWSE', 'ALL'):
        m = sig & U[un]; d = m[t0:].sum(1)
        SIG[un] = dict(stock_days=int(d.sum()), distinct_codes=int(m[t0:].any(0).sum()), days_with_signal=float((d > 0).mean()), mean_per_day=float(d.mean()), max_per_day=int(d.max()),
                       by_year={int(k): int(v) for k, v in pd.Series(d, index=P.IDX[t0:]).groupby(P.IDX[t0:].year).sum().items()})
        # 各條件單獨通過率（訊號日平均，上市或全體在榜股票中）
        base = cond['base'] & U[un]
        SIG[un]['pass_rate_each'] = {k: float((cond[k] & base)[t0:].sum() / base[t0:].sum()) for k in ('C', 'A_cagr', 'A_rev', 'N', 'S', 'L', 'M')}
        both = sig & U[un] & (trail | cross)
        SIG[un]['share_signal_days_with_exit_true_same_day'] = float(both[t0:].sum() / max(m[t0:].sum(), 1))
    res['SIGNALS'] = rnd(SIG)

    # ---------------- 貼文口徑
    POST = {}; alltr = []
    for un in ('TWSE', 'ALL'):
        for pit in (True, False):
            sg, _, tr_, cr_, _ = sigset(pit)
            tr = cs.post_trades(P, sg, tr_, cr_, t0, U[un])
            nm = 'POST_%s_%s' % (un, 'PIT' if pit else 'NONPIT')
            for net in (False, True):
                o = trade_stats(P, tr, t0, net)
                twr = o.pop('_twr'); rp = o.pop('_ret_px'); rt = o.pop('_ret_tr')
                ws = cs.window_stats(twr, bench); w19 = cs.window_stats(twr, bench, A2019)
                o['twr_vs_0050'] = dict(full=ws, from2019=w19)
                POST[nm + ('_net' if net else '_gross')] = o
                if not net:
                    x = tr[['code', 'sig_date', 'ent_date', 'exit_date', 'ret_px', 'ret_tr', 'x_trail', 'x_cross', 'open_end']].copy(); x.insert(0, 'variant', nm)
                    x['mkt'] = np.where(P.MK[tr['sig_t'].to_numpy(), tr['j'].to_numpy()] == 1, 'twse', 'tpex'); alltr.append(x)
    # ---- 事後對帳（看過結果後才加，不是策略變體）：用貼文的最大單筆獲利／虧損反推來回成本約 0.40%，再以含息（分割已還原）報酬重算
    RECON = {}
    for x in alltr:
        nm = x['variant'].iloc[0]; trn = x
        for c in (0.0040, 0.0041):
            r = (1 + trn['ret_tr'].to_numpy()) * (1 - c) - 1; cl = trn['open_end'].to_numpy() == 0
            xm = pd.to_datetime(trn['exit_date']).dt.to_period('M')
            o = dict(n=int(len(r)), n_win=int((r > 0).sum()), n_loss=int((r <= 0).sum()), win_rate=float((r > 0).mean()), mean=float(r.mean()), max=float(r.max()), min=float(r.min()),
                     by_exit_month_2026={str(m): dict(n=int(((xm == m).to_numpy() & cl).sum()), mean=float(r[(xm == m).to_numpy() & cl].mean()), win_rate=float((r[(xm == m).to_numpy() & cl] > 0).mean()))
                                         for m in pd.period_range('2026-05', '2026-10', freq='M')})
            RECON['%s_c%.2f%%' % (nm, c * 100)] = o
    for un in ('TWSE', 'ALL'):
        for pit in (True, False):
            sg, _, tr_, cr_, _ = sigset(pit)
            tr = cs.post_trades(P, sg, tr_, cr_, t0, U[un]); n_ = len(tr)
            cv = cs.post_curve(P, tr, t0, (np.full(n_, 0.0005), np.full(n_, 0.0035)))
            twr = pd.Series(cv['twr'], index=P.IDX[t0:]); yy = (P.IDX[-1] - P.IDX[t0]).days / 365.25
            dd = twr / twr.cummax() - 1
            RECON['POST_%s_%s_curve_c0.40%%' % (un, 'PIT' if pit else 'NONPIT')] = dict(
                twr_total=float(twr.iloc[-1] - 1), twr_cagr=float(twr.iloc[-1] ** (1 / yy) - 1), twr_maxdd=float(dd.min()), maxdd_date=str(dd.idxmin().date()),
                peak_date=str(twr.idxmax().date()), peak_total=float(twr.max() - 1), dd_from_peak_to_end=float(twr.iloc[-1] / twr.max() - 1),
                twr_total_12y_if_same_cagr=float(twr.iloc[-1] ** (12 / yy) - 1))
    RECON['implied_cost'] = dict(max_win_trade='5508（上櫃）2024-02-29 開盤 105.5 → 2024-05-08 開盤 218.5，毛報酬 +107.109%；貼文 +106.29% → 隱含來回成本 1−2.0629/2.07109＝0.395%',
                                 max_loss_trade='3665（上市）2026-05-12 開盤 3000 → 2026-05-14 開盤 2250，毛報酬 −25.000%；貼文 −25.31% → 隱含來回成本 1−0.7469/0.75＝0.413%',
                                 note='兩筆隱含成本都約 0.40%（例如證交稅 0.3%＋手續費每邊約 0.05%）；XQ 的實際成本設定需查證')
    res['RECON_POSTHOC'] = rnd(RECON)
    yrs = (P.IDX[-1] - P.IDX[t0]).days / 365.25
    POST['_claim'] = dict(n=3271, n_win=1139, n_loss=2132, win_rate=0.3482, twr_total=6.4844, max_win=1.0629, max_loss=-0.2531, max_runup=10.8766, max_dd=-0.2923,
                          period='2014/11/01～2026/11/01', trades_per_year=3271 / 12.0, our_years=yrs,
                          consistency='若最大區間虧損 −29.23% 就是 2026 年中高點到期末的跌幅：高點淨值＝7.4844/(1−0.2923)＝10.576（累積 +957.6%）；最大區間獲利 +1,087.66% 若是「低點到高點」，低點淨值＝10.576/11.8766＝0.890（期初後曾跌 11.0%）。這只是自洽性推算，XQ 的定義需查證')
    res['POST'] = rnd(POST)

    # ---------------- 可執行
    EXEC = {}; navs = {}
    def run(name, sg, rk, tr_, cr_, un='TWSE', keep_log=False, ts=None, **kw):
        nav, lg, st = cs.simulate(P, sg, rk, tr_, cr_, U[un], ts if ts is not None else t0, disp=disp, **kw)
        o = dict(full=cs.window_stats(nav, bench), from2019=cs.window_stats(nav, bench, A2019),
                 segments={a + '~' + b: cs.window_stats(nav, bench, a, b) for a, b in SEGS}, drop_best3=cs.drop_best3_t(nav, bench),
                 drop_best3_from2019=cs.drop_best3_t(nav, bench, A2019)['t'])
        y = (nav.index[-1] - nav.index[0]).days / 365.25; an = float(nav.mean())
        st2 = dict(st); st2['turnover_oneway_per_year'] = st['buy_val'] / an / y; st2['cost_acct_pct_per_year'] = (st['fee'] + st['tax'] + st['slip'] + st['e50_cost']) / an / y * 100
        st2['fee_pct'] = st['fee'] / an / y * 100; st2['tax_pct'] = st['tax'] / an / y * 100; st2['slip_pct'] = st['slip'] / an / y * 100
        st2['minfee_order_share'] = st['minfee_orders'] / max(st['orders'], 1)
        st2['blk_limup_share_of_attempts'] = st['blk_limup'] / max(st['n_buy'] + st['blk_limup'] + st['blk_lot'] + st['blk_noopen'] + st['blk_nocash'], 1)
        if len(lg):
            r = lg['ret'].to_numpy(); w = r > 0
            st2['trades'] = dict(n=int(len(r)), win_rate=float(w.mean()), mean=float(r.mean()), avg_win=float(r[w].mean()) if w.any() else None,
                                 avg_loss=float(-r[~w].mean()) if (~w).any() else None, max=float(r.max()), min=float(r.min()),
                                 hold_days=float((lg['exit_t'] - lg['ent_t']).mean()),
                                 by_why={k: dict(n=int(len(g)), mean=float(g['ret'].mean()), win_rate=float((g['ret'] > 0).mean())) for k, g in lg.groupby('why')},
                                 disp_buys=dict(n=int(lg['disp'].sum()), mean=float(lg.loc[lg['disp'] == 1, 'ret'].mean()) if lg['disp'].sum() else None))
        o['exec'] = st2; EXEC[name] = o; navs[name] = nav
        return (nav, lg) if keep_log else nav

    sig, fld, trail, cross, cond = sigset(True)
    RS = fld['RS']
    nav_main, lg_main = run('MAIN', sig, RS, trail, cross, keep_log=True)
    run('ZERO', sig, RS, trail, cross, cost='zero'); run('STRESS', sig, RS, trail, cross, cost='stress')
    run('S20', sig, RS, trail, cross, slots=20); run('S20_ZERO', sig, RS, trail, cross, slots=20, cost='zero'); run('S20_STRESS', sig, RS, trail, cross, slots=20, cost='stress')
    for nm, cap in (('C30', 3e5), ('C300', 3e6), ('C1000', 1e7)):
        run(nm, sig, RS, trail, cross, capital=cap)
        run(nm + '_STRESS', sig, RS, trail, cross, capital=cap, cost='stress')
    run('E50', sig, RS, trail, cross, idle='e50')
    run('RK_EPS', sig, fld['EpsQYoY'], trail, cross); run('RK_LIQ', sig, fld['AvgQty'], trail, cross)
    run('ALL', sig, RS, trail, cross, un='ALL'); run('ALL_ZERO', sig, RS, trail, cross, un='ALL', cost='zero')
    # J 偏移
    J = []
    for k in range(13):
        nv, _, _ = cs.simulate(P, sig, RS, trail, cross, U['TWSE'], t0 + 5 * k, disp=disp)
        w = cs.window_stats(nv, bench, A2019); J.append(dict(j=k, start=str(P.IDX[t0 + 5 * k].date()), cagr=w['cagr'], excess_pp=w['excess_pp'], t=w['t_ex_m'], maxdd=w['maxdd']))
    ex = np.array([x['excess_pp'] for x in J])
    res_J = dict(runs=J, median_excess_pp=float(np.median(ex)), p10_excess_pp=float(np.quantile(ex, 0.10)), min=float(ex.min()), max=float(ex.max()))
    # 消融
    ABLR = {}
    for a in ABL:
        sg = cs.entry_signal(cond, (a,))
        run('abl_' + a, sg, RS, trail, cross)
        trp = cs.post_trades(P, sg, trail, cross, t0, U['TWSE']); o = trade_stats(P, trp, t0, False); o.pop('_twr'); o.pop('_ret_px'); o.pop('_ret_tr')
        ABLR['abl_' + a] = dict(post_n=o['px']['n'], post_win=o['px']['win_rate'], post_mean_px=o['px']['mean'], post_twr_cagr=o['curve']['twr_cagr'],
                                post_max_conc=o['curve']['max_concurrent'], signal_stock_days=int((sg & U['TWSE'])[t0:].sum()))
    # 主策略重疊（用預設訊號、PIT）
    top10, top30 = main_approx(P, rev[True], disp, t0)
    # 用 MAIN 成交紀錄重建每日持股
    held = np.zeros((P.T, P.N), bool)
    for j, e, x in zip(lg_main['j'], lg_main['ent_t'], lg_main['exit_t']):
        held[e:x, j] = True
    hd = held[t0:]; n_h = hd.sum()
    mn = pd.read_parquet(cs.AG / 'main_v2' / 'navs_gapfix.parquet')['EXFIN|n10|normal'].dropna()
    mm_ = cs.monthly(nav_main); mo = cs.monthly(mn.reindex(nav_main.index).ffill().dropna()); mb = cs.monthly(bench.reindex(nav_main.index))
    ix = mm_.index.intersection(mo.index)[1:]
    OV = dict(canslim_holding_days=int(n_h), share_in_main_top10=float((hd & top10[t0:]).sum() / max(n_h, 1)), share_in_main_top30=float((hd & top30[t0:]).sum() / max(n_h, 1)),
              main_top10_days_held_by_canslim=float((hd & top10[t0:]).sum() / max(top10[t0:].sum(), 1)),
              distinct_codes_canslim=int(hd.any(0).sum()), distinct_codes_also_ever_main_top10=int((hd.any(0) & top10[t0:].any(0)).sum()),
              corr_monthly_ret=float(np.corrcoef(mm_[ix], mo[ix])[0, 1]), corr_monthly_excess_vs_0050=float(np.corrcoef(mm_[ix] - mb[ix], mo[ix] - mb[ix])[0, 1]), n_months=int(len(ix)),
              corr_monthly_ret_from2019=float(np.corrcoef(mm_[ix][ix >= pd.Period('2019-01')], mo[ix][ix >= pd.Period('2019-01')])[0, 1]),
              note='主策略名單為近似重建（非正式 V7X 名單）；報酬相關用 main_v2 navs_gapfix 的 EXFIN|n10|normal')
    res['ENGINE_CHECK'] = dict(nav_end=float(nav_main.iloc[-1]), capital_plus_sum_trade_pnl=float(1e6 + (lg_main['proceeds'] - lg_main['cost']).sum()),
                               note='期末淨值應等於起始資金加上逐筆（賣出淨收入−買進總成本）的總和')
    x = lg_main[['code', 'ent_date', 'exit_date', 'ret', 'why', 'shares', 'disp', 'cost', 'proceeds']].copy()
    x['sig_date'] = P.IDX[lg_main['sig_t'].to_numpy()]; x.insert(0, 'variant', 'EXEC_MAIN'); alltr.append(x)
    # 非 PIT、還原價、鄰域
    sg, f2, tr_, cr_, _ = sigset(False)
    run('NONPIT', sg, f2['RS'], tr_, cr_); run('NONPIT_ZERO', sg, f2['RS'], tr_, cr_, cost='zero'); run('ALL_NONPIT', sg, f2['RS'], tr_, cr_, un='ALL')
    run('NONPIT_S20', sg, f2['RS'], tr_, cr_, slots=20)
    sg, f2, tr_, cr_, _ = sigset(True, adj=True)
    run('ADJ', sg, f2['RS'], tr_, cr_)
    trp = cs.post_trades(P, sg, tr_, cr_, t0, U['TWSE']); o = trade_stats(P, trp, t0, False); o.pop('_twr'); o.pop('_ret_px'); o.pop('_ret_tr')
    res['POST_ADJ_TWSE_PIT_gross'] = rnd(dict(px=o['px'], tr=o['tr'], curve=o['curve'], exit_decomp=o['exit_decomp']))
    NBR = {}
    for nm, over in NB:
        sg, f2, tr_, cr_, _ = sigset(True, over)
        run(nm, sg, f2['RS'], tr_, cr_)
        NBR[nm] = dict(over=over, from2019_excess_pp=EXEC[nm]['from2019']['excess_pp'], from2019_t=EXEC[nm]['from2019']['t_ex_m'], full_excess_pp=EXEC[nm]['full']['excess_pp'],
                       full_t=EXEC[nm]['full']['t_ex_m'], n_buy=EXEC[nm]['exec']['n_buy'])
    res['EXEC'] = rnd(EXEC); res['J_OFFSETS'] = rnd(res_J); res['ABLATION_POST'] = rnd(ABLR); res['NEIGHBOR'] = rnd(NBR); res['OVERLAP'] = rnd(OV)
    # 成本拖累
    res['COST_DRAG'] = rnd(dict(
        main_full_pp=(EXEC['ZERO']['full']['cagr'] - EXEC['MAIN']['full']['cagr']) * 100, main_from2019_pp=(EXEC['ZERO']['from2019']['cagr'] - EXEC['MAIN']['from2019']['cagr']) * 100,
        stress_full_pp=(EXEC['ZERO']['full']['cagr'] - EXEC['STRESS']['full']['cagr']) * 100,
        s20_full_pp=(EXEC['S20_ZERO']['full']['cagr'] - EXEC['S20']['full']['cagr']) * 100,
        pit_vs_nonpit=dict(main_full=EXEC['MAIN']['full']['cagr'], nonpit_full=EXEC['NONPIT']['full']['cagr'], diff_full_pp=(EXEC['NONPIT']['full']['cagr'] - EXEC['MAIN']['full']['cagr']) * 100,
                           main_2019=EXEC['MAIN']['from2019']['cagr'], nonpit_2019=EXEC['NONPIT']['from2019']['cagr'],
                           diff_2019_pp=(EXEC['NONPIT']['from2019']['cagr'] - EXEC['MAIN']['from2019']['cagr']) * 100)))
    # 族群
    from statistics import NormalDist
    fam = PREREG['family']; n_new = len(fam['new']); N = fam['N_before'] + n_new
    res['FAMILY'] = dict(N_before=fam['N_before'], n_new=n_new, N_after=N, bonferroni_two_sided_t=round(NormalDist().inv_cdf(1 - 0.025 / N), 3), names=fam['new'],
                         extra_cost_or_path_runs_not_counted=['S20_ZERO', 'S20_STRESS', 'C30_STRESS', 'C300_STRESS', 'C1000_STRESS', 'ALL_ZERO', 'NONPIT_ZERO', 'NONPIT_S20', 'J 偏移 12 條', 'POST 扣成本版 4 條', 'POST_ADJ'])
    T_ = pd.concat(alltr, ignore_index=True)
    for c in ('sig_date', 'ent_date', 'exit_date'):
        T_[c] = pd.to_datetime(T_[c]).dt.strftime('%Y-%m-%d')
    for c in ('ret_px', 'ret_tr', 'ret'):
        T_[c] = T_[c].round(5)
    T_.to_csv(HERE / 'trades.csv.gz', index=False, compression='gzip')
    pd.DataFrame({k: v for k, v in navs.items()}).round(2).to_csv(HERE / 'navs.csv.gz', compression='gzip')
    res['RUN'] = dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), seconds=round(time.time() - T0, 1), start_signal_day=START,
                      first_buy_day=str(P.IDX[t0 + 1].date()), last_day=str(P.IDX[-1].date()), cslib_sha256=cs.sha_file(HERE / 'cslib.py'), run_sha256=cs.sha_file(HERE / 'run.py'))
    save_res(res)
    print('done', round(time.time() - T0, 1), 's')

if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'main'
    if st == 'gate':
        stage_gate()
    elif st == 'prereg':
        stage_prereg()
    else:
        stage_main()

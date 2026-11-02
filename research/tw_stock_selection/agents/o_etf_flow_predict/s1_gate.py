"""第 0 步資料閘門（不看報酬）：成分名單完整率（≥90%）與規模月資料覆蓋（≥90%；整體 <70% → 資料不足）。
審核次數的分母依各 ETF 規則推算；名單來源：etf_rebalance/events.csv、o_index_deletion_flow/hd_etf_transcribed.tsv、本題新聞封存擷取（s0_parse_news.py）。"""
import sys, json, hashlib, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import eflib as E
R = E.reviews(); ev = E.events()
hd = pd.read_csv(E.AG / 'o_index_deletion_flow/hd_etf_transcribed.tsv', sep='\t')
def cnt(etf):
    r = R[R.etf == etf]; return len(r), int((r.verify == 'news').sum()), int((r.verify == 'wiki').sum()), int((r.verify == 'partial').sum()), int((r.verify == 'none_assumed').sum())
# 分母：2006-01～2026-10 期間應有的審核次數
DEN = {
 'TW50': ('季度（3/6/9/12），2006～2026-09', 83),
 '0056': ('2007～2012 半年（6/12）＋2013 起季度（新聞 2013-03 季度審核為證）至 2026-09', 12 + 55),
 '00878': ('半年（5/11）2020-11～2026-05', 12),
 '00919': ('半年（5/12）2022-12～2026-06', 8),
 '00713': ('半年／季，2017 起；只抄到 2025-12 一次', 18),
 '00918': ('季度，2022-12 起；只抄到 1 次', 15),
 '00929': ('半年／季，2023-06 起；只抄到 1～2 次', 7),
 '00940': ('半年（5/11），2024-03 掛牌；只抄到 2025-11 一次', 5),
}
gate = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'rule': '成分名單完整率 ≥90%、規模月資料覆蓋 ≥90%；整體規模覆蓋 <70% → 資料不足', 'etf': {}}
for etf, (desc, den) in DEN.items():
    if etf in ('TW50', '0056', '00878', '00919'):
        n, nn, nw, npart, nnone = cnt(etf)
        complete = nn + nw + nnone
    else:
        n = int(hd[hd.etf.astype(str) == etf].review.nunique()); npart = 0; complete = n
    extra = 0
    if etf == '0056': extra = 1   # 2013-03 由新聞封存完整擷取（5 入 5 出）；2015-03、2018-12 只有部分名單，不計
    gate['etf'][etf] = dict(den_desc=desc, reviews_expected=den, reviews_with_list=n + extra, reviews_complete=complete + extra,
                            list_completeness=round((complete + extra) / den, 3), aum_monthly_coverage=0.0,
                            aum_points=('5 個新聞規模點（2017-12、2021-10、2022-10、2026-03、2026-07）' if etf == '0056' else '未取得'),
                            pass_list=(complete + extra) / den >= 0.9, pass_aum=False)
gate['etf']['0056']['note'] = '2022-12 起 9 次審核名稱完整率約 92%（2024-06 刪除 5 檔只查到 2 檔），但沒有任何成分快照，無法重建成分 → precision 不可算'
gate['etf']['TW50']['note'] = '2015-09 起有完整成分（o8 tw50_members.json，每步 50 檔核對）；2006～2015-06 無名單 → 全期完整率不到 90%；2015-09～2026-09 子期完整'
gate['etf']['TW50']['sub_2015_2026_completeness'] = 1.0
gate['aum_sources_tried'] = {
 'SITCA 投信投顧公會月報（www.sitca.org.tw）': 'DNS 不在允許網域（WebFetch ENOTFOUND）',
 'TWSE ETF 專區（www.twse.com.tw）': 'proxy CONNECT 403',
 'FinMind API（TaiwanStockHoldingSharesPer 含 ETF 集保股數）': 'proxy CONNECT 403',
 'TDCC 集保週資料封存（wirelessr/tdcc-opendata-archive）': '只有 2021-07～09 與 2026-04 起 39 週，覆蓋 < 10%',
 'GitHub code search（受益權單位數、ETF 規模、0056 holdings、TaiwanStockHoldingSharesPer 等 20 組關鍵字）': '只找到爬蟲程式與單日快照（nuts-has-been-taken 前 5 大持股、erichuang2013 兩個月 SITCA xls），沒有歷史封存',
 'z-Wind/stockTools extraData': '只有指數與基金價格（臺灣高股息報酬指數日線 2007 起），沒有成分與規模',
}
gate['overall_aum_coverage'] = 0.0
gate['decision'] = '整體規模月資料覆蓋 0% < 70% → 依事前失敗條件結案為「資料不足」；P1（不需規模）仍照計畫做，供下一輪決策。'
s = json.dumps(gate, ensure_ascii=False, indent=1, sort_keys=True)
gate['sha256'] = hashlib.sha256(s.encode()).hexdigest()
json.dump(gate, open(E.HERE / 'gate.json', 'w'), ensure_ascii=False, indent=1)
for k, v in gate['etf'].items(): print(k, v['reviews_complete'], '/', v['reviews_expected'], v['list_completeness'], v['pass_list'], v['pass_aum'])
print(gate['sha256'])

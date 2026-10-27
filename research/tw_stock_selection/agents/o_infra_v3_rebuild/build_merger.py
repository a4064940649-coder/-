#!/usr/bin/env python3
"""第 0 步 (a)-2：併購／下市對價表 merger_consideration.csv（不看任何策略報酬）。
python3 -I -B build_merger.py

涵蓋範圍（事前定）：
  (1) probe_dead2.json 的 _all_dead_codes：R_DEL 營收＋價格修補＋缺營收界限（pfix 0.5／1.0）×全市場／排除金融×n10／n30／等權
      任何版本中「持有時永久停止交易」的 28 檔；
  (2) 2019-05-17～2024 終止上市的 4 碼股票（價格修補的 32 檔＋1701）。
欄位：type＝cash（現金對價）／swap（換股、控股公司轉換、現金＋換股混合）／failure（經營惡化、財報不實、淨值為負等下市）／unknown；
  cash＝每股現金對價（元）；ratio、successor＝換股比例與存續公司代碼（僅供參考，引擎對 swap 一律用最後收盤近似）；
  verified＝news（中央社等新聞鏡像 aRthur08701244/Intern-in-CTBC 2013～2020 文字中找到對價）／knowledge（研究員知識，未經公開文件核對）。
  last_trade、last_close 由面板（含價格修補）計算。pay_date 未知者留空（引擎用終止上市日＋14 天）。
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import o3_core as O
K = O.K
O.apply_patch(require_freeze=False)
lt = O.last_trade_idx()
D = pd.read_csv(HERE / 'delist_official.csv', dtype=str).set_index('code')

# code: (type, cash, ratio, successor, verified, note)
T = {
 '1704': ('cash', 56.0, None, None, 'news', 'KKR／李長榮投資公開收購＋合併；新聞「榮化走高接近收購價每股 56 元」'),
 '1716': ('swap', None, 1.0, '3705', 'knowledge', '轉換為永信投控（股份轉換 1:1）'),
 '1902': ('cash', 20.8, None, None, 'news', '新誠資產管理收購；新聞「台紙公布的收購價 20.8 元」'),
 '2311': ('swap', None, 0.5, '3711', 'knowledge', '共組日月光投控：1 股日月光換 0.5 股日月光投控'),
 '2315': ('swap', None, 1.0, '3706', 'knowledge', '轉換為神達投控（股份轉換）；新聞提到「神達投資控股」'),
 '2325': ('cash', 51.2, None, None, 'news', '日月光投控以現金 51.2 元取得矽品全部股份；新聞「矽品…51.2」'),
 '2343': ('swap', None, None, '6214', 'knowledge', '與精誠合併（換股）'),
 '2394': ('swap', None, None, '2317', 'knowledge', '鴻海換股合併普立爾'),
 '2403': ('swap', None, None, '3702', 'knowledge', '大聯大換股收購友尚'),
 '2418': ('failure', None, None, None, 'knowledge', '財報不實、停止買賣後終止上市'),
 '2446': ('swap', None, None, '3037', 'knowledge', '欣興換股合併全懋'),
 '2473': ('cash', None, None, None, 'knowledge', 'Synopsys 現金收購（金額未核對 → 引擎以最後收盤近似）'),
 '2823': ('swap', None, None, '2891', 'knowledge', '中信金股份轉換取得中壽'),
 '2833': ('swap', None, None, '2891', 'knowledge', '中信金股份轉換取得台灣人壽'),
 '2847': ('swap', None, None, '2885', 'knowledge', '元大金以現金＋股份併購大眾銀（新聞：斥資 565.5 億元）；混合對價以最後收盤近似'),
 '2854': ('swap', None, None, '2885', 'knowledge', '元大金換股併購寶來證'),
 '3009': ('swap', None, None, '3481', 'knowledge', '與群創合併（群創存續，更名奇美電）'),
 '3012': ('swap', None, None, '2409', 'knowledge', '友達換股合併廣輝'),
 '3063': ('unknown', None, None, None, 'knowledge', '2010 年併購下市，對價型態未核對'),
 '3214': ('swap', None, None, '2448', 'knowledge', '晶電換股合併元砷'),
 '3474': ('cash', 30.0, None, None, 'news', '美光現金收購，新聞「收購價維持 30 元」'),
 '3534': ('swap', None, None, '2454', 'knowledge', '聯發科以現金＋股份合併雷凌'),
 '3614': ('unknown', None, None, None, 'knowledge', '2010 年合併下市，對價型態未核對'),
 '3697': ('swap', None, None, '2454', 'knowledge', '聯發科以股份＋現金合併晨星'),
 '5280': ('swap', None, None, '3545', 'knowledge', '與旭曜合併（旭曜存續、更名敦泰，上櫃）'),
 '5854': ('swap', None, 1.0, '5880', 'knowledge', '轉換為合庫金控（股份轉換 1:1）'),
 '6004': ('swap', None, None, '2885', 'knowledge', '元京證與復華證合併為元大證（元大金體系）'),
 '6286': ('cash', 195.0, None, None, 'news', '聯發科分兩階段每股 195 元現金收購；新聞「每股195元分兩階段收購」'),
 # 2019-05-17～2024 終止上市（價格修補對象；目前都不在持有中下市，僅供完整性）
 '3579': ('failure', None, None, None, 'knowledge', '太陽能矽晶圓，終止上市'),
 '1262': ('failure', None, None, None, 'knowledge', 'KY 股，終止上市'),
 '9157': ('failure', None, None, None, 'knowledge', 'DR，終止上市'),
 '5259': ('failure', None, None, None, 'knowledge', '終止上市'),
 '4144': ('failure', None, None, None, 'knowledge', 'KY 股，終止上市'),
 '8497': ('unknown', None, None, None, 'knowledge', '終止上市，原因未核對'),
 '2499': ('failure', None, None, None, 'knowledge', '終止上市'),
 '5305': ('cash', None, None, None, 'knowledge', 'Diodes 現金收購（金額未核對）'),
 '3698': ('swap', None, None, '3714', 'knowledge', '與晶電共組富采投控'),
 '2448': ('swap', None, None, '3714', 'knowledge', '與隆達共組富采投控'),
 '5264': ('cash', None, None, None, 'knowledge', 'KY 股下市（現金收購，金額未核對）'),
 '4725': ('cash', None, None, None, 'knowledge', '現金收購（金額未核對）'),
 '6452': ('failure', None, None, None, 'knowledge', 'KY 股，停止買賣後終止上市'),
 '6131': ('failure', None, None, None, 'knowledge', '終止上市'),
 '1724': ('cash', None, None, None, 'knowledge', '現金收購（金額未核對）'),
 '2456': ('swap', None, None, '2327', 'knowledge', '國巨以股份＋現金收購奇力新'),
 '1592': ('failure', None, None, None, 'knowledge', 'KY 股，終止上市'),
 '8427': ('cash', None, None, None, 'knowledge', 'KY 股下市（金額未核對）'),
 '9188': ('failure', None, None, None, 'knowledge', 'DR，終止上市'),
 '1507': ('cash', None, None, None, 'knowledge', '日立現金收購永大（金額未核對）'),
 '4141': ('cash', None, None, None, 'knowledge', 'KY 股下市（金額未核對）'),
 '2936': ('cash', None, None, None, 'knowledge', 'KY 股下市（金額未核對）'),
 '6172': ('unknown', None, None, None, 'knowledge', '終止上市，原因未核對'),
 '2841': ('unknown', None, None, None, 'knowledge', '終止上市，原因未核對'),
 '6251': ('swap', None, 1.0, '3715', 'knowledge', '轉換為定穎投控'),
 '6289': ('swap', None, None, None, 'knowledge', '合併下市（存續公司未核對）'),
 '3383': ('swap', None, None, None, 'knowledge', '合併下市（存續公司未核對）'),
 '8480': ('cash', None, None, None, 'knowledge', 'KY 股下市（金額未核對）'),
 '3536': ('failure', None, None, None, 'knowledge', '終止上市'),
 '3682': ('swap', None, None, '4904', 'knowledge', '遠傳以換股合併亞太電'),
 '1701': ('unknown', None, None, None, 'knowledge', '2024-09 終止上市，原因未核對'),
}
rows = []
for c, (typ, cash, ratio, succ, ver, note) in T.items():
    j = K.CI.get(c)
    ltd = K.IDX[lt[j]] if j is not None and lt[j] >= 0 else pd.NaT
    lc = float(K.Cn[lt[j], j]) if j is not None and lt[j] >= 0 else np.nan
    dd = pd.Timestamp(D.loc[c, 'date']) if c in D.index else pd.NaT
    rows.append({'code': c, 'name': D.loc[c, 'name'] if c in D.index else '', 'announce_date': '', 'delist_date': str(dd.date()) if pd.notna(dd) else '',
                 'last_trade': str(ltd.date()) if pd.notna(ltd) else '', 'last_close': lc, 'pay_date': '', 'type': typ, 'cash': cash,
                 'cash_over_last_close': (cash / lc) if cash and lc else np.nan, 'ratio': ratio, 'successor': succ or '', 'verified': ver, 'note': note})
E = pd.DataFrame(rows).sort_values('delist_date')
E.to_csv(HERE / 'merger_consideration.csv', index=False)
print(E[['code', 'name', 'delist_date', 'last_trade', 'last_close', 'type', 'cash', 'cash_over_last_close', 'verified']].to_string())
print('sha256', hashlib.sha256((HERE / 'merger_consideration.csv').read_bytes()).hexdigest())

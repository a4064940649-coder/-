"""建立 ETF 成分股異動事件名單（人工整理自新聞／維基百科搜尋摘要），輸出 events.csv 與 reviews.csv。

日期規則（事前固定）：
* 臺灣50、臺灣高股息（FTSE/TWSE）：E＝審核月第三個星期五（被動資金收盤換股日；非交易日取前一交易日）；
  A＝E 前 14 天的星期五（公告日，盤後公告；非交易日取前一交易日）；C＝E 前 25 天的星期一（審核資料截止日，
  「審核生效日四週前的週一收盤資料」）。多筆新聞公告日已逐一核對（見 reviews 的 src）。
* 00878（MSCI）：E＝5／11 月 MSCI 半年度調整生效日（月最後營業日收盤；2024-11、2025-11 依新聞）；
  A＝MSCI 公告日（美國時間，台灣隔日開盤前得知）；日期依 MSCI 例行時程推定，未逐一查證。
* 00919（臺灣指數公司）：A＝審核基準日（5 月第 17、12 月第 7 個交易日）收盤後公告；E＝其後第 5 個交易日收盤。
verify 欄：news＝新聞明文；wiki＝只見於維基百科表格（經搜尋引擎摘要，未能直接開啟）；partial＝名單不完整；
none_assumed＝搜尋不到該季異動、維基表格無該季，推定無變動。
"""
import csv
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
WIKI = 'https://zh.wikipedia.org/zh-tw/臺灣50指數'

# (review, adds, dels, verify, src, note)
TW50 = [
    ('2015-03', [], [], 'none_assumed', WIKI, ''),
    ('2015-06', [], [], 'news', 'https://www.twse.com.tw/zh/news/newsDetail/?id=16660', '104/6/5 公告，無增刪'),
    ('2015-09', ['1476'], ['2498'], 'wiki', WIKI, ''),
    ('2015-12', [], [], 'none_assumed', WIKI, ''),
    ('2016-03', [], [], 'none_assumed', WIKI, ''),
    ('2016-06', ['2324'], ['2227'], 'news', 'https://tw.stock.yahoo.com/news/臺灣指數系列於2016-06-20調整成分股-011809154.html', ''),
    ('2016-09', [], [], 'none_assumed', WIKI, ''),
    ('2016-12', [], [], 'news', 'https://www.cna.com.tw/news/afe/201612020311.aspx', '定期審核無變動；中壽替補華亞科為下市遞補，非定期事件，不列入'),
    ('2017-03', [], [], 'news', 'https://www.chinatimes.com/realtimenews/20170303006063-260410', ''),
    ('2017-06', ['2633'], ['1476'], 'news', 'https://www.cna.com.tw/news/afe/201706020459.aspx', ''),
    ('2017-09', [], [], 'news', 'https://www.cna.com.tw/news/afe/201709010250.aspx', ''),
    ('2017-12', [], [], 'none_assumed', WIKI, ''),
    ('2018-03', ['5871'], ['2207'], 'news', 'https://www.chinatimes.com/realtimenews/20180302003302-260410', ''),
    ('2018-06', ['2492'], ['2324'], 'news', 'https://www.chinatimes.com/realtimenews/20180601003479-260410', '國巨替補矽品（4 月下市遞補）不列入'),
    ('2018-09', [], [], 'none_assumed', WIKI, ''),
    ('2018-12', ['5876'], ['2492'], 'wiki', WIKI, ''),
    ('2019-03', ['2207'], ['2354'], 'news', 'https://www.ctee.com.tw/news/20190304700029-430298', ''),
    ('2019-06', ['9910'], ['3481'], 'news', 'https://www.cna.com.tw/news/afe/201906060309.aspx', ''),
    ('2019-09', ['2888'], ['2409'], 'wiki', WIKI, ''),
    ('2019-12', [], [], 'news', 'http://hn28082251.blogspot.com/2019/12/2019-12-0050-0056.html', '部落格轉述公告：無增刪'),
    ('2020-03', [], [], 'none_assumed', WIKI, ''),
    ('2020-06', ['6669'], ['9904'], 'news', 'https://news.cnyes.com/news/id/4486864', ''),
    ('2020-09', ['2379', '6415'], ['2888', '2823'], 'news', 'https://taiwanindex.com.tw/news/169', '中壽 2823 面板 2019-05 後缺（存活者偏誤）'),
    ('2020-12', ['3034'], ['2301'], 'news', 'https://news.cnyes.com/news/id/4547643', ''),
    ('2021-03', ['1590', '8046'], ['2883', '2890'], 'news', 'https://ctee.com.tw/news/stock/432182.html', ''),
    ('2021-06', ['2603', '2615', '2609', '2409'], ['2633', '6669', '2105', '2474'], 'news', 'https://taiwanindex.com.tw/news/181', ''),
    ('2021-09', ['8454'], ['1102'], 'news', 'https://wantrich.chinatimes.com/news/20210903900800-420101', ''),
    ('2021-12', ['3037'], ['1402'], 'news', 'https://www.cna.com.tw/news/afe/202112030344.aspx', ''),
    ('2022-03', ['2883'], ['4938'], 'news', 'https://www.ctee.com.tw/news/20220304701041-439803', ''),
    ('2022-06', ['6770'], ['8454'], 'wiki', WIKI, ''),
    ('2022-09', ['2890'], ['6770'], 'news', 'https://www.cna.com.tw/news/afe/202209020282.aspx', ''),
    ('2022-12', ['1402'], ['2409'], 'news', 'https://wantrich.chinatimes.com/news/20221202900801-420101', '刪友達見維基'),
    ('2023-03', ['1605'], ['8046'], 'wiki', WIKI, ''),
    ('2023-06', ['4938'], ['6415'], 'news', 'https://udn.com/news/story/123006/7209164', ''),
    ('2023-09', ['2345', '2301', '3231', '6669'], ['2615', '2609', '1402', '1605'], 'news', 'https://www.cna.com.tw/news/afe/202309010331.aspx', ''),
    ('2023-12', [], [], 'news', 'https://www.ctee.com.tw/news/20231202700015-439901', '無增刪'),
    ('2024-03', ['3661'], ['9910'], 'wiki', WIKI, '公告前 20 日窗口跨面板 2024-01-02～02-14 缺口'),
    ('2024-06', ['3017'], ['2801'], 'news', 'https://money.udn.com/money/story/5618/8790092', ''),
    ('2024-09', ['6446'], ['2408'], 'news', 'https://tw.news.yahoo.com/台灣50指數成分股調整-納入藥華藥-刪除南亞科-台灣高股息指數成分股刪除新唐-113459281.html', '藥華藥 2024-03 才在面板（轉上市）'),
    ('2024-12', ['2609'], ['1590'], 'news', 'https://www.businesstoday.com.tw/article/category/183013/post/202412070006/', ''),
    ('2025-03', ['2615'], ['1326'], 'wiki', WIKI, ''),
    ('2025-06', ['2383'], ['3037'], 'news', 'https://news.cnyes.com/news/id/6010992', ''),
    ('2025-09', ['6919', '2059'], ['6446', '1101'], 'news', 'https://www.esunsec.com.tw/article/post/403', '康霈 6919 面板 2025-09-24 才有'),
    ('2025-12', ['2408', '3665', '3653', '2360'], ['2609', '4938', '5871', '5876'], 'news', 'https://www.cna.com.tw/news/afe/202512050295.aspx', '南亞科延後生效（udn 9211710）'),
    ('2026-03', ['2368', '7769', '2449', '3037', '2344'], ['3665', '3034', '2912', '2379', '2615'], 'news', 'https://wantrich.chinatimes.com/news/20260307900230-420101', '鴻勁 7769 2025-11 才上市'),
    ('2026-06', ['8046', '3443', '3665', '4958'], ['6919', '2002', '1301', '2207'], 'news', 'https://www.chinatimes.com/realtimenews/20260608000991-260410', 'E=6/18（6/19 端午）'),
    ('2026-09', ['6446'], ['3661'], 'news', 'https://udn.com/news/story/7251/9734823', ''),
]

D56 = [
    ('2022-12', ['2542', '2915', '1303', '2891', '2890', '6239', '2474', '1301', '1402', '2885', '4958', '2317', '2606', '6271', '2637', '2383',
                 '3017', '2337', '2379', '2027', '3665', '3711', '3189', '8046', '1802', '2809'],
     ['2002', '2014', '2441', '1101', '2609'], 'news', 'https://wealth.businessweekly.com.tw/m/GArticle.aspx?id=ARTL003009408', '30→50 檔擴編，屬一次性'),
    ('2023-06', ['1477', '2353', '2886', '6669', '1907'], ['3017', '2603', '2337', '1802', '6271'], 'news',
     'https://tw.stock.yahoo.com/news/0050、0056指數成分股調整結果一次看！6月16日尾盤生效-103348373.html', ''),
    ('2023-12', ['2105', '3005', '1605', '6285', '9917'], ['3189', '2606', '6669', '1907'], 'news',
     'https://www.btcc.com/zh-TW/academy/financial-investment/0056-constituent-stocks', ''),
    ('2024-06', ['5871', '3532', '4919', '1319', '9941'], ['2409', '3665'], 'partial', 'https://finance.ettoday.net/news/2753892', '刪除 5 檔只查到 2 檔'),
    ('2024-09', [], ['4919'], 'news', 'https://tw.stock.yahoo.com/news/0056換股結果出爐-50名老兵僅有-它-被砍-9-121500869.html', '季度只刪不增'),
    ('2024-12', ['2618', '2603', '2542', '2880', '4763', '5876', '1504', '2609'], ['1402', '5522', '3532', '2809', '2915', '9917', '1605'], 'news',
     'https://finance.ettoday.net/news/2868990', ''),
    ('2025-06', ['2883', '9904', '3023', '2887', '1216'], ['2383', '2542', '4763', '1303', '2609'], 'news', 'https://finance.ettoday.net/news/2973896', ''),
    ('2025-12', ['2884', '2645', '3045'], ['5871', '1504', '4958'], 'news', 'https://news.cnyes.com/news/id/6263201', ''),
    ('2026-06', ['1513', '4904', '1303', '2408', '2344'], ['1477', '6176', '1319', '2006'], 'news', 'https://www.ctee.com.tw/news/20260605701573-430403', 'E=6/18'),
]

# 00878：(review, A, E, adds, dels, verify, src)
D878 = [
    ('2022-05', '2022-05-12', '2022-05-31', ['1605', '5876', '8069', '2474', '3702'], ['2409', '2354', '2912', '5347'], 'news',
     'https://www.businesstoday.com.tw/article/category/183008/post/202205310044/'),
    ('2023-11', '2023-11-14', '2023-11-30', ['1216', '2303', '5347'], ['2353', '2883', '2002', '2408'], 'news', 'https://www.ctee.com.tw/news/20231122700719-430201'),
    ('2024-05', '2024-05-14', '2024-05-31', ['3034', '2412', '2105'], ['1303', '2347', '2357'], 'news',
     'https://www.businesstoday.com.tw/article/category/183013/post/202406030025/'),
    ('2024-11', '2024-11-06', '2024-11-25', ['2357'], ['2474'], 'news', 'https://udn.com/news/story/123006/8386207'),
    ('2025-05', '2025-05-13', '2025-05-30', ['5871', '2474', '2615'], ['1101', '2890', '2892'], 'news', 'https://wantrich.chinatimes.com/news/20250603900474-420101'),
    ('2025-11', '2025-11-05', '2025-11-24', ['2449', '3293', '4958'], ['2377', '2615', '3702'], 'news',
     'https://tw.stock.yahoo.com/news/00878成分股調整出爐！3進3出踢萬海-納入鈊象、京元電子-063252670.html'),
    ('2026-05', '2026-05-12', '2026-05-29', ['2603', '2618', '2880', '2887', '2912'], ['1402', '2324', '2474', '2449'], 'news',
     'https://www.chinatimes.com/realtimenews/20260602003248-260410'),
]

# 00919：(review, A, E, adds, dels, verify, src)
D919 = [
    ('2024-12', '2024-12-10', '2024-12-17', ['1504', '1722', '2211', '2886', '2891', '3005', '5483', '5904'],
     ['2352', '2382', '2609', '2615', '3702', '4961', '8046', '9802'], 'news', 'https://www.moneyweekly.com.tw/_Article?AID=134297'),
    ('2025-06', '2025-05-26', '2025-06-03', ['1477', '2371'], ['2348', '2454'], 'partial', 'https://www.cna.com.tw/news/afe/202506030315.aspx'),
    ('2025-12', '2025-12-09', '2025-12-16', ['1215', '2027', '2211', '2357', '2887'], ['1442', '2371', '2393', '2439', '8299'], 'partial',
     'https://tw.stock.yahoo.com/news/群益00919成分股調整名單出爐-新增卜蜂等8檔-094500409.html'),
    ('2026-06', '2026-05-26', '2026-06-02', ['1210', '1319', '1402', '2105', '2347', '2379', '2382', '2542', '2597', '2855', '2883', '3293', '4938',
                                             '5871', '6890', '8454', '9910', '9941'],
     ['1215', '2027', '2303', '2404', '2451', '2609', '3036', '3211', '3702'], 'partial', 'https://money.udn.com/money/story/5618/9541228'),
]


def third_friday(y, m):
    d = pd.Timestamp(y, m, 1)
    off = (4 - d.weekday()) % 7
    return d + pd.Timedelta(days=off + 14)


def on_or_before(idx, t):
    p = idx.searchsorted(t, side='right') - 1
    return idx[p]


def main():
    idx = pd.read_parquet(PANEL / 'ret.parquet', columns=['2330']).index
    sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str).set_index('code')
    name = lambda c: sl['name'].get(c, '')
    mkt = lambda c: sl['market'].get(c, '')
    ev, rv = [], []

    def ftse_dates(review):
        y, m = map(int, review.split('-'))
        tf = third_friday(y, m)
        E = on_or_before(idx, tf)
        A = on_or_before(idx, tf - pd.Timedelta(days=14))
        C = on_or_before(idx, tf - pd.Timedelta(days=25))
        return A, E, C

    for etf, rows in [('TW50', TW50), ('0056', D56)]:
        for review, adds, dels, ver, src, note in rows:
            A, E, C = ftse_dates(review)
            rv.append(dict(etf=etf, review=review, n_add=len(adds), n_del=len(dels), ann=A.date(), eff=E.date(), cutoff=C.date(), verify=ver, src=src, note=note))
            for act, lst in [('add', adds), ('del', dels)]:
                for c in lst:
                    ev.append(dict(etf=etf, review=review, code=c, name=name(c), market=mkt(c), action=act, ann=A.date(), eff=E.date(), cutoff=C.date(),
                                   verify=ver, date_rule='FTSE 第三個星期五規則', src=src, note=note))
    for etf, rows, rule in [('00878', D878, 'MSCI 例行時程（推定）'), ('00919', D919, 'TIP 基準日+5 交易日')]:
        for review, a, e, adds, dels, ver, src in rows:
            A, E = on_or_before(idx, pd.Timestamp(a)), on_or_before(idx, pd.Timestamp(e))
            rv.append(dict(etf=etf, review=review, n_add=len(adds), n_del=len(dels), ann=A.date(), eff=E.date(), cutoff='', verify=ver, src=src, note=''))
            for act, lst in [('add', adds), ('del', dels)]:
                for c in lst:
                    ev.append(dict(etf=etf, review=review, code=c, name=name(c), market=mkt(c), action=act, ann=A.date(), eff=E.date(), cutoff='',
                                   verify=ver, date_rule=rule, src=src, note=''))
    pd.DataFrame(ev).to_csv(HERE / 'events.csv', index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(rv).to_csv(HERE / 'reviews.csv', index=False)
    print(len(ev), 'events;', pd.DataFrame(ev).groupby(['etf', 'action']).size().to_dict())


if __name__ == '__main__':
    main()

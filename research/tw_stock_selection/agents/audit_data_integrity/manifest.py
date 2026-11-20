#!/usr/bin/env python3
"""A 資料清冊：正式回測（official_nav/run_official.py）與每日網站更新實際讀到的資料來源。python3 -I -B manifest.py → data_manifest.json、tables/manifest_table.csv。只讀。"""
import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent))
import json, time, subprocess
import pandas as pd
from common import *                                    # noqa: E402

meta = json.loads((TMP / 'off_meta.json').read_text())
P = {k: json.loads((HERE / 'parts' / f'{k}.json').read_text()) for k in ('prices', 'exright', 'revenue_delist', 'e50_chain')}
TOP = Path('/home/user/-')
tracked = set(subprocess.run(['git', '-C', str(TOP), 'ls-files'], capture_output=True, text=True, env=ENV).stdout.split('\n'))
def where(p):
    p = Path(p); s = str(p)
    if s.startswith('/tmp/'):
        return '暫存區（/tmp，重開機或例行清理會消失）'
    try:
        rel = str(p.relative_to(TOP))
    except ValueError:
        return '容器磁碟、主倉庫之外（容器回收就消失；靠 data_sources.txt 重新 clone）'
    if p.is_file():
        return '主倉庫內、已進版本控制' if rel in tracked else '主倉庫內、未進版本控制（.gitignore 或未加入；容器回收就消失）'
    n = sum(1 for t in tracked if t.startswith(rel + '/'))
    return f'主倉庫內、已進版本控制（{n} 個檔）' if n else '主倉庫內、未進版本控制（.gitignore；容器回收就消失）'
def finfo(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha_file(p)) if p.exists() else dict(path=str(p), exists=False)
def repo_info(repo, sub=None):
    h = git(repo, 'log', '-1', '--format=%H %cI').stdout.split()
    d = dict(repo=str(repo), remote=git(repo, 'config', '--get', 'remote.origin.url').stdout.strip(), commit=h[0] if h else None, commit_time=h[1] if len(h) > 1 else None,
             partial_clone=git(repo, 'config', '--get', 'remote.origin.partialclonefilter').stdout.strip() or None,
             shallow=(Path(repo) / '.git/shallow').exists())
    if sub:
        d['path_in_repo'] = sub; d['tree_sha'] = git(repo, 'rev-parse', f'HEAD:{sub}').stdout.strip()
    return d
ds = {l.split()[0]: l.split()[1:] for l in (RS / 'data_sources.txt').read_text().split('\n') if l and not l.startswith('#')}
pr, ex, rv, ec = P['prices'], P['exright'], P['revenue_delist'], P['e50_chain']
by = pr['B2_coverage']['by_year']
M = []
def add(**k):
    k['location_class'] = where(k['path']); M.append(k)

add(id='D01', name='官方日 K（上市＋上櫃，逐日一檔）— 封存版', used_by=['正式回測：cslib.build_panel／stlib.build_panel／s4lib.innov_panel（經 git cat-file）', '名單產生器 make_list.py 預設 --repo'],
    path=str(RAW_REPO), source='https://github.com/jb7k6295r7-bot/tw-stock-data（公開 GitHub 倉庫；倉庫內 data/universe/_coverage_README.md 說明每日管線，原始交易所網址檔內未記載）',
    repo=repo_info(RAW_REPO, 'data/universe/daily'), coverage=[meta['daily']['raw_first'], meta['daily']['raw_last']], count=f"{meta['daily']['raw_n']} 個日檔（本機 blob 全有）",
    update='不更新（2026-11-02 clone 後凍結）；引擎另以 cslib.END＝2026-10-22 截斷', gaps=['比每日更新版少 14 個交易日（2026-11-02～11-19）', '同倉庫的 data/universe/exright 等事件檔 blob 本機一個都沒有（1,787 個除權息檔讀不到）'],
    single_point=True, single_point_note='2015 起個股日 K 只有這一個上游；上游倉庫若刪除或改寫歷史，無法重新取得同一份', persist_advice='把 data/universe/daily（約 2,900 檔）連同 commit 編號打包成 tar.zst 放進主倉庫 raw/（或 Git LFS／物件儲存），並寫 SHA256SUMS；目前靠 .gitignore 排除，容器回收後只能重新 clone')
add(id='D02', name='官方日 K — 每日更新版（tsd）', used_by=['每日網站更新：build_site_data.py／live_tr.py／ledger.py／monthly_rebalance.py 的 --tsd', 'make_list.py 第二個 --repo（新日子與新月份營收）'],
    path=str(TSD), source='同 D01（同一個 GitHub 倉庫的較新 clone）', repo=repo_info(TSD, 'data/universe/daily'), coverage=[meta['daily']['tsd_first'], meta['daily']['tsd_last']],
    count=f"{meta['daily']['tsd_n']} 個日檔；本機只有 {meta['daily']['tsd_n'] - meta['daily']['tsd_missing_local']} 個有 blob（其餘 {meta['daily']['tsd_missing_local']} 個要連網才讀得到）",
    update='每日排程更新（目前 HEAD＝2026-11-20 05:32 台北的提交；淺層 clone 只有 1 個 commit，上游每日提交時刻無法由本機驗證；排程定義不在本地檔案內）', gaps=['shallow＋blob:none：歷史日檔本機沒有內容，離線時只能靠 D01 補', f"與 D01 重疊的 {meta['daily']['n_common']} 個日檔 blob 完全相同（上游沒有改寫歷史）"],
    single_point=True, single_point_note='網站每日更新與每月名單的最新股價只有這一個來源', persist_advice='搬到主倉庫之外的固定路徑（例如 /home/user/-/research/tw_stock_selection/raw/tsd_live，加進 .gitignore 並在 restore_env.sh 重建），不要放 /tmp；或直接讓 D01 每日 fetch 取代 tsd')
add(id='D03', name='正式引擎面板快取 panel_cs.npz／e50_ohlc.csv／innov.npz', used_by=['run_official.py（cslib.TMP＝scratchpad/offfix/cache）', 'S55／S75／R75 複製用 scratchpad/eng/cache'],
    path=str(ENG_CACHE), files=[finfo(ENG_CACHE / f) for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz')], source='由 D01 衍生（可重建）', coverage=['2015-01-05', '2026-10-30'],
    count='2,880 列 × 2,142 欄', update='不存在時自動重建（約 1～2 分鐘）', gaps=['run_official.py 指定的 scratchpad/offfix/cache 目前不存在（已被清掉；下次執行會重建）', '本稽核從 D01 重建後與 eng/cache/panel_cs.npz 逐格相同（8 個陣列差 0 格）'],
    single_point=False, persist_advice='可重建，不必搬；但建議把 panel_cs.npz 的 sha256（434f6422…）寫進正式結果檔，重建後核對')
add(id='D04', name='月營收長表 tw_revenue.csv', used_by=['正式回測：s19lib.rev_std／s2lib.rev_tables（stlib.REV_CSV）', 'make_list.py --rev-csv（歷史月份）'],
    path=str(REV_CSV), files=[finfo(REV_CSV)], source=f"https://github.com/NiklavsD/bottleneck-signals commit {ds.get('niklavsd/bottleneck-signals', ['?'])[0]}（data_sources.txt）；s55 data_inventory 記載內容為公開資訊觀測站 t21sc03 彙總頁解析；檔內未記載網址",
    coverage=[rv['B5_revenue']['first'], rv['B5_revenue']['last']], count=f"{rv['B5_revenue']['n_rows']:,} 列、{rv['B5_revenue']['n_codes']:,} 個代號（上市 sii {rv['B5_revenue']['by_market']['sii']:,}、上櫃 otc {rv['B5_revenue']['by_market']['otc']:,}）",
    update='不更新（上游 2026-10-17 的版本；之後的月份靠 D05）', gaps=['只到 2026-09', '沒有 KY／外國企業（上市股涵蓋率 2013 起約 88%～91%）', '下市公司營收提早斷、2013 起只有現存公司', '2018-10 上櫃只有 126 家（缺洞）', '已含事後更正值'],
    single_point=True, single_point_note='歷史月營收只有這一份；在主倉庫之外（/home/user/niklavsd/…），容器回收後靠 data_sources.txt 重新 clone 上游', persist_advice='複製一份到主倉庫 raw/revenue/tw_revenue_<commit>.csv.gz（約 15MB 壓縮後更小）並寫 sha256')
add(id='D05', name='官方倉庫月營收彙總表 data/mops/revenue_hist', used_by=['make_list.py：tw_revenue.csv 沒有的新月份（目前 2026-10；12/11 名單需要 2026-10 與 2026-11）', 'revenue_seen（--rev-cutoff 時點檢查）'],
    path=str(TSD) + ' : data/mops/revenue_hist', source='同 D01 倉庫（檔內欄位為公開資訊觀測站彙總表；網址檔內未記載）', repo=repo_info(TSD, 'data/mops/revenue_hist'),
    coverage=['2015-01', '2026-10'], count=f"425 個檔（上市／上櫃／興櫃各月）；tsd 本機只有 1 個有 blob、D01 封存有全部但 2026-08～10 是舊版",
    update='上游每日更新（當月公告期間逐日增加公司）', gaps=['D01 封存的 2026-10 上市檔只有 3 家（11-02 當時）', '2026-08、2026-09 上市檔上游在 11-02 之後改過，本機沒有新版 blob（無法驗證差異）'],
    single_point=True, single_point_note='最新一個月的營收（決定當月名單）只有這個來源；官方直接抓取（o_forward 的 rev_official_fetch）14 次全失敗', persist_advice='每月訊號日名單定稿後，把當月 YYYY-MM_twse.csv 的 blob 存一份到 s18_live_list/lists/ 或 raw/revenue_hist_snapshots/（含 sha256 與抓取時間）')
add(id='D06', name='證交所 TWT49U 除權除息計算結果表（封存）', used_by=['正式回測 EXRIGHT_FIX（cslib.load_twt49u）', 'make_list.py --exright-fix', 'live_tr.py 備援'],
    path=str(TWT49U.parent), source='https://github.com/joeycchen72/taiwan-market-history commit fb838a00719a31b5e7591b17070bd5a3b616c6c7（raw/twt49u/README.md）', files=[finfo(TWT49U.parent / 'SHA256SUMS'), finfo(TWT49U.parent / 'README.md')],
    coverage=[ex['twt49u']['first'], ex['twt49u']['last']], count=f"{ex['twt49u']['n_files']} 個月檔、{ex['twt49u']['n_rows']:,} 列（四碼 {ex['twt49u']['n_rows_4digit']:,}）；SHA256SUMS 全部相符",
    update='人工每月補（README 規定）；目前停在 2026-10-29', gaps=['只有上市', '2014-01-16 以前沒有', '2026-10-30 起未補（已落後 15 個交易日、6 個上市四碼事件）'], single_point=False,
    single_point_note='2015-07-02 起可用 D07（官方倉庫 exright）交叉；2014-01～2015-07-01 只有這一份', persist_advice='已在主倉庫版本控制內，位置正確；需要的是每月補檔')
add(id='D07', name='官方倉庫日期軸事件檔（exright／otcexright／etfsplit／parvalue／reduce…）', used_by=['網站 live.json／holdings 含息口徑（live_tr.py）', 'monthly_rebalance.py 現金股利'],
    path=str(TSD) + ' : data/universe/{exright,otcexright,etfsplit,parvalue,reduce,otcparvalue,otcreduce}', source='同 D01 倉庫', repo=repo_info(TSD, 'data/universe/exright'),
    coverage=[ex['repo_exright']['tsd']['first'], ex['repo_exright']['tsd']['last']], count=f"exright {ex['repo_exright']['tsd']['n_files']} 檔（本機讀得到 {ex['repo_exright']['tsd']['n_files'] - ex['repo_exright']['tsd']['missing_local']}）",
    update='上游每日', gaps=['上市 exright 有 42 天本機沒有 blob（2015-01-05～07-01 共 40 天、2019-09-25、2026-11-20）', '上櫃 otcexright 717 天本機沒有 blob', 'D01 封存版這些事件檔一個 blob 都沒有'],
    single_point=False, single_point_note='上市可用 D06 備援（但 D06 停在 10-29）；ETF 分割、減資、面額變更、上櫃除權息沒有第二來源', persist_advice='同 D02（跟著 tsd 搬到固定路徑）；每月把當月用到的事件檔存快照')
add(id='D08', name='0050 含息還原價 etf_0050.csv', used_by=['正式回測基準 P.E50（cslib.E50_CSV）', 'make_profit.py（網站回測淨值的 0050 線）'],
    path=str(E50_CSV), files=[finfo(E50_CSV)], source=f"https://github.com/jonatw/finlab-txd commit {ds.get('jonatw/finlab-txd', ['?'])[0]}（data_sources.txt）；原始資料商檔內未記載",
    coverage=[ec['B7_0050']['engine_file']['first'], ec['B7_0050']['engine_file']['last']], count=f"{ec['B7_0050']['engine_file']['n']} 列，只有 date、adj 兩欄",
    update='不更新（凍結在 2026-10-22＝cslib.END）；每日更新的副本在 D10', gaps=['只有還原價，沒有原始價與配息明細', '多 3 列非交易日（2026-07-10、2026-09-20 週日且數值異常 −7.15%、2026-10-23），缺 2025-06-11～17 分割停牌 5 天'],
    single_point=True, single_point_note='2015 以前的 0050 含息報酬只有這一份（2015 起可用 D01 的 0050 原始價＋D06 配息重建）', persist_advice='複製到主倉庫 raw/benchmark/etf_0050_<commit>.csv 並寫 sha256')
add(id='D09', name='加權指數 taiex_twii.csv', used_by=['cslib.Panel 讀入 P.IDXC（正式 E0 訊號不使用，載入後即刪除）'], path=str(TWII_CSV), files=[finfo(TWII_CSV)],
    source='同 D08 倉庫（jonatw/finlab-txd）', coverage=['1999-01-05', '2026-10-22'], count='6,866 列', update='不更新', gaps=['多 1 列非交易日 2026-09-20（週日）'], single_point=False, persist_advice='同 D08')
add(id='D10', name='前瞻追蹤每日資料（o_forward_infra_live/raw：screener、finlab、official_rev）', used_by=['daily_step0.sh → make_forward.py（前瞻名單、處置帳本；不是網站名單）'],
    path=str(FWD / 'raw'), source='https://github.com/stevenhsiehsheng-ctrl/tw-stock-screener、https://github.com/jonatw/finlab-txd（daily_step0.sh 內記載）；官方月營收 openapi.twse.com.tw、www.tpex.org.tw、mopsov.twse.com.tw',
    repo=dict(screener=repo_info(FWD / 'raw/screener'), finlab=repo_info(FWD / 'raw/finlab_repo')), coverage=['—', ec['B8_daily_chain']['twii_forward_copy_last']],
    count=f"official_rev {ec['B8_daily_chain']['official_rev_fetch_summary']['n_attempts']} 次抓取目錄", update='每日 06:49（台北）git fetch；每月 9～20 日抓官方月營收',
    gaps=[f"官方月營收抓取 {ec['B8_daily_chain']['official_rev_fetch_summary']['n_attempts']} 次全部失敗（代理 403）", '紀錄檔缺 2026-11-11、11-13 兩天'], single_point=False,
    persist_advice='.gitignore 排除、容器回收後由 daily_step0.sh 自動重建（已有機制）；不必搬')
add(id='D11', name='2015 以前上市日 K（Asoul/tsec 衍生長表）', used_by=['S55／S75／R75 的 2007～2014 面板（不是 2015 起的正式數字）'], path=str(PRE), files=[finfo(PRE), finfo(PRE.parent / 'asoul_source_manifest.json')],
    source='Asoul/tsec（證交所原始行情的本機副本；asoul_source_manifest.json 記載來源目錄與逐檔 sha256，網址檔內未記載）', coverage=['2007-01-02', '2015-06-30'], count='1,531,951 列、926 個代號',
    update='不更新', gaps=['來源目錄 scratchpad/tsec_all（313MB、1,921 檔）在暫存區', '除權息日只有 X 記號、沒有參考價', '9 檔 2007～2013 下市股整檔沒有'], single_point=True,
    single_point_note='2007～2014 的個股日 K 只有這一份（data_panel 是補格用，同源頭）', persist_advice='衍生長表已在主倉庫；原始 tsec_all 若要保留可重現性，打包（約 60～80MB）放進 raw/pre2015/')
add(id='D12', name='舊研究面板 data_panel/panel/*.parquet', used_by=['S75 pre 段補格（72,224 格）', 'daily_step0／restore_env.sh 還原 scratchpad/panel'], path=str(DP), files=[finfo(DP / f) for f in ('close_raw.parquet', 'open.parquet', 'high.parquet', 'value.parquet', 'ret.parquet')],
    source='研究中心自建（build_twse_panel.py；meta.json）', coverage=['2004-01-02', '2026-10-22'], count='5,586 個交易日、1,249 個代號', update='不更新',
    gaps=['缺 2024-01-02～02-14 共 25 個交易日（官方面板有）', '多 1 個非交易日 2018-01-13'], single_point=False, persist_advice='已在主倉庫版本控制內')
add(id='D13', name='營收申報期限例外表／保險業 15 日名單', used_by=['正式回測 DEADLINE_FIX／INS15_MASK', 'make_list.py --deadline-fix'], path=str(RS / 'raw/rev_deadline_exceptions'),
    files=[finfo(RS / 'raw/rev_deadline_exceptions' / f) for f in ('deadline_exceptions.csv', 'ins15_list.csv')], source='新聞整理（README 列出 ETtoday、中央社、工商時報、聯合報網址；主管機關原文讀不到）',
    coverage=['2019-01', '現在'], count='例外 3 列（採用 2 列）、名單 19 檔', update='人工，每年依金管會公告', gaps=['依新聞而非原始公告'], single_point=True, single_point_note='人工維護的小表，沒有第二來源', persist_advice='已在主倉庫版本控制內')
add(id='D14', name='網站狀態與已發布淨值', used_by=['每日網站更新（build_site_data.py）', 'make_profit.py'], path=str(SITE),
    files=[finfo(SITE / f) for f in ('navs_official.csv.gz', 'profit.json', 'state/segments.csv', 'state/holdings_after_2026-11.csv', 'state/cash_after_2026-11.txt')] +
          [finfo(AG / 's18_live_list/lists' / f) for f in ('2026-11_exright.csv', '2026-11_exright.json', '2026-11_backtest_holdings_exright.csv')],
    source='研究中心自產', coverage=['2015-07-01', '2026-10-22（navs）／2026-11-19（網站每日資料）'], count='navs_official 2,760 列', update='每日（data/*.json）、每月（state、lists）',
    gaps=[], single_point=False, persist_advice='已在主倉庫（請確認 state/ 與 lists/ 每次換股後有提交）')
add(id='D15', name='下市清單 delist_official.csv', used_by=['不在正式引擎內；存活者偏誤研究與本稽核 B6 使用'], path=str(DELIST), files=[finfo(DELIST)],
    source='o_infra_v3_rebuild/raw/songdoyouho/delisted_stocks.csv（第三方封存；原始網址檔內未記載）', coverage=['2001-01-20', '2024-09-02'], count='256 列（2007 起四碼 133 檔）', update='不更新',
    gaps=['2024-09-02 之後沒有（面板推得其後至少 7 檔離開上市）', '只有上市'], single_point=True, persist_advice='已在主倉庫；需要補到最新')
add(id='D16', name='產業分類 stock_list.csv', used_by=['s4lib.industry_map（正式 E0 淨值不使用；產業相關研究與網站說明用）'], path='/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv',
    files=[finfo('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv')], source=f"https://github.com/stevenhsiehsheng-ctrl/tw-stock-screener commit {ds.get('stevenhsiehsheng-ctrl/tw-stock-screener', ['?'])[0]}",
    coverage=['現行分類（非歷史時點）', '—'], count='—', update='不更新', gaps=['現行分類，不是當時分類'], single_point=False, persist_advice='不影響正式數字，維持現狀')
out = dict(written_at=pd.Timestamp.now(tz='UTC').isoformat(), purpose='正式回測（official_nav/run_official.py）與每日網站更新實際用到的資料來源清冊（audit_data_integrity）',
           engine_settings=dict(cslib_END='2026-10-22', stlib_START='2015-07-01', first_signal_month='2016-01', flags=dict(EXRIGHT_FIX=True, DEADLINE_FIX=True, INS15_MASK=True),
                                universe='上市（MK＝1）、四碼非 0 開頭、非 91 開頭、非創新板、60 日均額 ≥ 1,000 萬、收盤 ≥ 10 元', git_no_lazy_fetch='git 2.43.0-1ubuntu7.4 有支援 GIT_NO_LAZY_FETCH（實測會印 lazy fetching disabled）'),
           items=M,
           single_points=[m['id'] + ' ' + m['name'] for m in M if m.get('single_point')],
           in_scratch=[m['id'] + ' ' + m['name'] for m in M if m['location_class'].startswith('暫存區')] + ['D11 的原始來源 scratchpad/tsec_all', 'run_official.py 的面板快取 scratchpad/offfix/cache（目前不存在）'],
           not_version_controlled=[m['id'] + ' ' + m['name'] for m in M if '未進版本控制' in m['location_class'] or '主倉庫之外' in m['location_class']])
jdump(out, HERE / 'data_manifest.json')
tb = pd.DataFrame([dict(編號=m['id'], 資料=m['name'], 用途='；'.join(m['used_by']), 路徑=m['path'], 來源=m['source'], 起=m['coverage'][0], 迄=m['coverage'][1], 筆數=m['count'], 更新=m['update'],
                        位置=m['location_class'], 單點依賴='是' if m.get('single_point') else '否', 已知缺口='；'.join(m['gaps']), 建議=m['persist_advice']) for m in M])
tb.to_csv(HERE / 'tables/manifest_table.csv', index=False)
for m in M:
    print(m['id'], m['name'], '|', m['location_class'], '| 單點' if m.get('single_point') else '', '|', m['coverage'])
print(out['in_scratch']); print(out['not_version_controlled'])

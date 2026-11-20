#!/usr/bin/env python3
"""寫 PREREG.json＋PREREG.sha256＋code.sha256（只寫一次；已存在就拒絕）。寫之前只跑過 accept.py（現行規則重現＋特徵可用性），沒有算過任何新規則的績效。"""
import json, hashlib, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trials as TR
assert not (HERE / 'PREREG.json').exists()
T = TR.build(); NB = TR.neighbors(T)
acc = json.loads((HERE / 'accept.json').read_text()); assert acc['pass']
pre = dict(
 task='S79 營收訊號＋本機可回溯到 2008 的其他資訊：2008–2015 訓練 → 凍結 → 2016–2026 驗證一次',
 written_before='任何新規則的績效計算（只跑過 accept.py：現行規則訊號／淨值重現、特徵有值比例）',
 data_inventory=dict(
  usable_back_to_2008={
   '日 K（開高低收、成交金額、漲跌價差、上市別）': 'S75 連續面板 2007-01-02～2026-10-22（2015 前 Asoul/tsec＋data_panel 補格；2015 起官方日 K）。價量衍生特徵（3／6／12 個月報酬、距 52 週高點、波動、成交值、股價）都由此算；面板從 2007-01 起，240 日特徵在 2008-01 第一個訊號日（2008-02-13）股池內 98% 有值',
   '月營收': 'niklavsd tw_revenue.csv 2005-01 起（事後更正值；缺 KY 與多數下市股；2013 IFRS 口徑斷點用 S71 B2 水準校正）→ 年增率、3 個月累計年增率、連續年增月數、近 12 月年增為正月數、營收創 12 個月新高、前一月 SURGE',
   '0050 含息／加權指數': 'jonatw finlab-txd（2008 起有；本題只當基準）',
   '產業分類': 'niklavsd tw_industry.csv：只有「現在」的分類一份（32 類），不是當時分類＝有前視疑慮；2008 第一個訊號日股池只有 87.6% 對得到（對不到的多為後來下市），沒有分類的公司各自一類。F8 的結果一律加註',
  },
  not_available_back_to_2008={
   '三大法人買賣超／融資融券／借券／本益比淨值比': '官方倉庫 data/universe/inst、margin、sbl、per 檔名只從 2015-01-05 起，且本機是 blob:none 部分 clone，內容一個都讀不到；voidful twse_flows.csv 只有 2025-09 起 → 沒有',
   '財報（EPS、毛利、ROE）': 'data/mops/fin_hist 只有 2013Q1 起、fs_hist 2015Q1 起 → 2008–2012 沒有',
   '股本／市值': '只有現在的股本（非當時值），2015 前不能用（S60 已註明）→ 不做市值因子；以「60 日均成交值」「股價」代替',
   '集保股權分散': 'tdcc_hist 2017 起 → 沒有',
   '處置／注意股、當沖比': '2015 前沒有',
  },
  prior_work_not_repeated='S17（SURGE 定義家族含年增率尺度）、S20（持續成長）、S21／S31／S56（產業共振／中性／上限）、S32／S53（波動）、S33（52 週高點）、S46（動能門檻家族）、S12（規模分層）、S22／S28（法人／融資確認）、S70（獲利確認）都是 2015-07 以後的樣本，判定皆為不改預設；本題不重做它們，只在 2008–2015 訓練、2016–2026 驗證的新切法下測同類附加資訊，結論時互相對照',
 ),
 base_rule='現行：SURGE（B2 口徑＝V_BRIDGE）；股池上市非創新板；名單＝還原收盤 > 60 日線依 SURGE 排名；前 10 名且 RET20 > 10% 進場；訊號列不在前 20 名出場；A10 換股日；W10_70；嚴成本；空位買 0050。引擎＝s75lib（s47lib.sim），連續帳戶自 2008-01-02 起，100 萬',
 search_space=dict(
  how_modified='每個試驗只在現行規則上加一種附加資訊（s79lib.make_signal）：list_gates＝進名單的額外條件（進場與續抱同一張名單，沒過等於不在名單）；pool_gates＝前 10 名的「已在漲」條件（取代或加在 RET20>10% 上）；score＝排名分數改成 SURGE 與其他特徵的百分位加權／產業內相對；ind_cap＝同月同產業買進上限',
  stage1_trials=[dict(id=t['id'], fam=t['fam'], sub=t['sub'], grid=t['grid'], spec=t['spec'], neighbors=NB[t['id']]) for t in T],
  stage1_n=len(T),
  stage2='第一階段合格（見 selection）的家族中，各取 J_adj 最高的一格，依 J_adj 取前 4 個家族，做全部兩兩／三個／四個的合併（最多 11 個）：list_gates 相接；pool_gates＝有改動者的條件聯集（都沒改就用預設）；非 SURGE 的 score 最多一個（兩個成員都改 score 的合併跳過）；ind_cap 取較小者。合格家族 < 2 就沒有第二階段',
  families={'F0': '現行', 'F1': '價格動能 3／6／12 個月', 'F2': '距 52 週高點', 'F3': '低波動', 'F4': '成交值／流動性', 'F5': '股價位階', 'F6': '營收年增率與 SURGE 合成', 'F7': '營收連續成長', 'F8': '產業內相對（現行分類，前視疑慮）', 'F9': 'SURGE×價量合成排名', 'F11': '過熱迴避'},
  max_trials_total=len(T) + 11, hard_cap=300,
 ),
 train=dict(window=['2008-01', '2015-12'], note='連續帳戶 2008-01-02 起跑；訓練程式把每條淨值截到 2015-12-31 才算統計，之後的數字不看、不存'),
 paths='20 條＝本金 60／80／100／150／200 萬 × 訊號位移 +0～+3 個交易日（同 S75）',
 selection=dict(
  per_trial_stats={'ex_year': '2008～2015 每個日曆年：20 條路徑「策略年報酬 − 0050 年報酬」的中位數（個百分點）',
                   'J': '(ex_year 的中位數 ＋ ex_year 的最差值) ÷ 2',
                   'cagr_ex': '20 條路徑「訓練窗年化 − 0050 年化」的中位數（個百分點）',
                   'trades': '標準路徑（100 萬、+0）訓練窗內進場且已平倉的筆數',
                   'J_adj': 'min(J, 鄰格 J 的平均)；沒有鄰格的單格＝J − 1.0；第二階段合併格＝J − 1.0'},
  eligible='cagr_ex ≥ 現行的 cagr_ex ＋ 2.0，且 trades ≥ 現行的 50%，且 J_adj ≥ 現行的 J ＋ 1.0',
  winner='合格者中 J_adj 最高的第一階段單格；第二階段合併格要比它的 J_adj 高 ≥ 1.0 才取代（寧簡）。同分取 id 字典序較前者',
  if_none='沒有合格者 → FROZEN＝現行規則（結論：加其他資訊沒有比現行好），驗證段仍跑一次全體分佈',
 ),
 validation=dict(window=['2016-01', '2026-10'], run_once=True, account='同一個連續帳戶（2008-01-02 起），取驗證窗的統計；不回頭改',
  pass_criteria={'P1 維持現行水準': '新規則驗證窗 20 路徑年化中位數 ≥ 現行 − 2.0 個百分點',
                 'P2 回撤': '驗證窗最大跌幅不比現行深超過 5 個百分點（標準路徑）',
                 'P3 對 0050': '驗證窗月超額年化 > 0 且 Newey-West(6) t ≥ 1.65（標準路徑）',
                 'T1 前段明顯贏 0050': '訓練窗月超額年化 ≥ 5 個百分點且 NW t ≥ 2.0（標準路徑；這是挑過的樣本內數字，另報多重比較）'},
  verdict='P1、P2、P3、T1 全過 →「值得獨立複製的候選規則」；否則「沒有比現行好，維持現行」',
  post_freeze_distribution='凍結後把全部試驗（第一＋第二階段）在驗證窗的年化、對現行的差距列出分佈：新規則排第幾、現行排第幾、訓練 J_adj 與驗證超額的等級相關（判讀是規則好還是後段什麼都好）',
 ),
 multiple_testing='試驗總數＝第一階段 %d ＋第二階段 ≤ 11。報告：訓練窗最佳者的 t 與「N 次獨立試驗下最大 t 的期望值 ≈ sqrt(2 ln N)」「Bonferroni 單尾 5%% 門檻」對照；驗證段只有 1 個凍結規則，不需校正，但與現行規則高度相關。資料窺探累計 N 依實際試驗數增加' % len(T),
 known_limits=['2014-01-16 前除權息日報酬記 0', '營收為事後更正值', '缺 KY 與下市股營收（存活者偏誤，偏樂觀）', '2008–2009 股價單一來源', '2015-06 前實際漲跌停 7%', '2020-10 前只能買整張', '產業分類為現行分類', 'B2 口徑是 2026 年才訂的',
               '連續帳戶：新規則若前段賺較多，2016 年的本金較大，整股時代買得起較多高價股（R75 (d)）→ 驗證窗比較略偏向前段較好的規則'],
 accept=acc,
)
s = json.dumps(pre, ensure_ascii=False, indent=1)
(HERE / 'PREREG.json').write_text(s)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
(HERE / 'PREREG.sha256').write_text(sha(HERE / 'PREREG.json') + '  PREREG.json\n')
(HERE / 'code.sha256').write_text(''.join(f'{sha(HERE / f)}  {f}\n' for f in ('s79lib.py', 'trials.py', 'train.py', 'validate.py')))
print(sha(HERE / 'PREREG.json'), len(T))

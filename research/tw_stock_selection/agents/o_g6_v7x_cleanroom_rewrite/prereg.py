# R2 凍結：iterations.json＋PREREG（程式 sha、interp_register sha、P1～P5、判定規則、o18 教訓）寫入 results.json
import json, hashlib, datetime, os
HERE = os.path.dirname(os.path.abspath(__file__))
sha = lambda f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest()
dev = {v: open(os.path.join(HERE, 'logs', f'k_dev_{v}.txt')).read() for v in ['v1', 'v2', 'v3']}
it = {'versions_tried': 3, 'limit': 5, 'rule': '只允許不變量與開發切片（2006-04～2008-12、j0～2、seeds0～2、100 萬一般；K 只回年化差）；每次修改引用規格原文',
      'rounds': [
        {'v': 'v1', 'change': '初版（interp_register v1 全部項目）', 'dev_ann_diff_pp': dev['v1'], 'invariants': 'logs/invariants_v1.json（round(S) 賣量造成 −0.05 股 → E33 修正後才跑 dev）'},
        {'v': 'v2', 'change': 'j 日改以加權指數交易日計數再對到面板日；次一面板日非 GOOD 的月份跳過（引文：o_v6 FINDINGS 449 行「m 月 10 日起第一個加權指數交易日之後的第 j 個交易日」；o_rev_freshness_attention FINDINGS 29 行「次一交易日不是 GOOD 日的月份跳過」）', 'dev_ann_diff_pp': dev['v2'], 'note': '開發切片內無影響（只影響 2018-01）'},
        {'v': 'v3', 'change': '52 週比改用自算還原價（G 連乘）；合格股 < 50 的換股日跳過；開盤漲停容忍 1e−6（引文：o4 FINDINGS 366、375、377 行）；新增紙上零衝擊引擎 xpaper.py（o4 FINDINGS 379～382 行）與右尾 P&L 記錄', 'dev_ann_diff_pp': dev['v3'], 'note': '切片差不變；J0 差 2.4pp 推定為 AB 亂數（原 20 seeds vs 本端 3 seeds、RNG 不同），J1／J2 ±0.5pp 未追（凍結前不得看名單）'}],
      'selected': 'v3', 'selection_rule': '最後一版；沒有依切片差調參數（三版差異只來自引文修正）'}
json.dump(it, open(os.path.join(HERE, 'iterations.json'), 'w'), ensure_ascii=False, indent=1)
code = ['xdata.py', 'xengine.py', 'xlists.py', 'xstats.py', 'xrun.py', 'xpaper.py', 'run_formal.py', 'audit_run.py', 'ast_scan.py']
FROZEN = open(os.path.join(HERE, 'frozen_spec_text.txt')).read()
pre = {
 'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'code_sha256': {f: sha(f) for f in code},
 'interp_register_sha256': sha('interp_register.json'), 'iterations_sha256': sha('iterations.json'), 'explore_sha256': sha('explore.json'),
 'frozen_spec': FROZEN,
 'predictions': {
  'P1': '引擎正確：Track E（原名單餵本引擎）6 格（100 萬一般全期 +0.86／t 0.45、j p10 −2.25、2019 起 −6.22（23.27% 對 29.49%）、300 萬 −1.00、1,000 萬 −3.32、壓力 −6.28）逐格取 opt／p90／carry 最差後與原正式數字差 ≤ 0.1pp、t ≤ 0.05；18 個底層數字逐一比對。紙上 22.22%／t 1.64 一致機率 ≥ 80%；可執行 18 格至少 1 格分歧機率約 40%。',
  'P2': '分歧集中在 1,000 萬（K5T 非線性）、壓力（零股壓力加碼）、carry；月份落在下市事件月、A_UB 回退日、開盤漲停遞補日、src 3 期間、2015-06-01 前 7% 段；分歧主要落在 2013～2018 一般月份 → P2 否定。',
  'P3': '30 萬整股主欄全期超額比 100 萬低 ≥ 1.5pp 且 < 0；2006～2020-10 平均投入比率 ≤ 75%；2021～2026 與 100 萬差 ≤ 1pp；30 萬 ≥ 100 萬 → 否定。',
  'P4': '2019 起落差：真實成本類（稅、手續費、滑價、K5T）合計 ≤ 8pp；成交時點（收盤→開盤）＋漲停遞補 ≥ 6pp；成本類 > 12pp 或時點類 < 3pp → 否定。',
  'P5': '紙上貢獻前 5% 持股槽可執行捕捉率 ≤ 50%；2019 起更低、2006～2012 較高；全期 ≥ 80% → 否定。'},
 'subjective_prob': {'P1_paper': 0.8, 'P1_any_exec_divergence': 0.4, 'P3': 0.6, 'P4': 0.4, 'P5': 0.5},
 'decision_rules': {
  'threshold': '關鍵數字 |Δ| > 0.1pp 或 |Δt| > 0.05 → 分歧；另報逐年 |Δ| 合計',
  'divergence': '找出第一個不同的日期、股票、股數、價格，以原始資料（官方開收盤、整股規則、2020-10-26 盤中零股、預收與分盤、跳動點表）裁定；裁定 J 寫、V 覆核；不平均',
  'outcomes': '原引擎錯 → 取代正式數字並附逐項歸因；重寫錯 → 修正後只重跑受影響格，記凍結後修正；規格不足 → 兩版並列、不利端為正式、註「規格缺口」；未解前正式數字旁註「引擎未獨立重寫驗證」',
  'G6': '通過（引擎＋名單）＝6 格、紙上、30 萬全在門檻內（或分歧經 J 裁定、V 覆核已解決）且無未解規格缺口影響正式格，且 A／K／J／V 為不同子代理；本題四角色由同一子代理扮演 → 依 E00 最多判「部分進展」',
  'replace': '原引擎錯誤且改變 > 0.1pp 或 t > 0.05 → 取代正式數字',
  'main_strategy': '本題是主策略本身的量測；G1～G5、G7、G8 不因本題變好；判定預期「不可投資」；30 萬寫進 G7；G3 N 維持 2,573'},
 'o18_lessons': ['1 不利端逐格取最差', '2 凍結前不算正式統計量（開發切片只回年化差；不變量只回結構量）', '3 事後新增狀態與嚴格版並列、嚴格版為主欄', '4 raw/ 只放未修改原始檔、註記另放；查詢詞不含目標數字（本題無外部查詢）', '5 假設區間（opt／p90／carry）與抽樣區間（月份 block bootstrap、±1.96SE 年化）並報', '6 取代正式數字附歸因拆解', '7 紙上附成本兩平點（t<1、超額歸零）', '8 可執行測到 30 萬（整股主欄＋盤後零股敏感度）', '9 G6 狀態在複製完成後重新核對'],
 'DSR_SPA_note': '本題不產生新選股訊號，G3 族群 N 維持 2,573'
}
s = json.dumps(pre, ensure_ascii=False, sort_keys=True)
R = {'PREREG': pre, 'PREREG_sha256': hashlib.sha256(s.encode()).hexdigest()}
assert not os.path.exists(os.path.join(HERE, 'results.json'))
json.dump(R, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1)
print(R['PREREG_sha256'], pre['written_utc'])

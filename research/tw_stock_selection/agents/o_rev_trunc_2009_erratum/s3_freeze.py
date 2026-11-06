#!/usr/bin/env python3
"""第 3 步：凍結 ERRATUM2_SPEC.json（在算任何名次之前）。寫入 sha256 與 UTC，並把 PREREG 記進 results.json。
用法：python3 -I -B s3_freeze.py（已存在就拒絕覆寫）"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
P = HERE / 'ERRATUM2_SPEC.json'
assert not P.exists(), 'ERRATUM2_SPEC 已凍結，不得覆寫'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ex = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
sl = json.loads((HERE / 'search_log.json').read_text(encoding='utf-8'))
spec = {
 'title': '主策略營收面板 2009-11 截斷勘誤：2006～2013 名單的存活者缺口上界計數（o_rev_trunc_2009_erratum）',
 'no_returns': '只計數名單槽位，不計算任何報酬、淨值或 t 值',
 'base': {
   'strategy': 'main_v2/v2_core.V2(sigmode=gapfix, YL, Y5, exfin=True)，plan(10, tie=official)，含開盤漲停遞補（held 集合沿用 V2.plan）',
   'signal_days': '2006-01-01～2013-12-31 的 V2.sig_days（93 個；2006-04-11～2013-12-11）',
   'baseline_check': '重建名單必須與 o_g8_protocol_fin_erratum/ERRATUM.json 的 lists_C0 逐日逐檔相同（程式斷言），否則中止',
   'v2_core_sha256': sha(AG / 'main_v2/v2_core.py'), 'ia_core_sha256': sha(AG / 'independent_audit/ia_core.py'),
   'erlib_sha256': sha(HERE / 'erlib.py'), 'tw_revenue_sha256': sha('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'),
   'erratum_json_sha256': sha(AG / 'o_g8_protocol_fin_erratum/ERRATUM.json'),
   'listing_master_sha256': sha(AG / 'o_v7_listing_exec_engine/listing_master.csv')},
 'affected_set': {
   'pool': '換股日 d：前 300 流動性（60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10，依 V60 排序取前 300）∩ 52 週分數有值 ∩ 非處置 ∩ 隔日有開盤成交 ∩ 非 C0 金融（V.FINMASK）',
   'missing': '該池內近 3 月營收年增率 y3（m−1～m−3 三個月年增率平均，需 m−13～m−15 基期）為缺值的股票',
   'cause': '對 m−1～m−3 與 m−13～m−15 六個月中「沒有營收水準」（或基期 ≤ 0）的月份 x 分類（erlib.cause）：'
            'FOREIGN＝代號 91xx 或任一名稱含 KY／DR／以 F- 開頭；NEVER_DOM＝國內股但 tw_revenue 從未出現；'
            'x > 營收最後月份 且 面板最後交易月落在 2009-12～2012-12 → TRUNC；x ≥ 2013-01 且營收最後月份 ≥ 2012-12 → IFRS13；其他 x > 最後月份 → PRE12；'
            'x < 營收第一個月份 → NEWLIST；介於第一與最後月份之間 → HOLE；水準存在但基期 ≤ 0 → BASE_LE0。'
            '一個股月有多種原因時依 TRUNC > IFRS13 > PRE12 > HOLE > NEWLIST > BASE_LE0 取第一個。',
   'governing_causes': ['TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEVER_DOM'],
   'excluded_causes': {'FOREIGN': 'SOP 明文排除 KY／DR（股池定義，不是存活者偏誤）', 'NEWLIST': '新上市缺基期（主策略設計上不算，存活與下市股一視同仁）',
                       'BASE_LE0': '基期 ≤ 0（設計上不算）'},
   'descriptive_ALL': '另報 which=ALL（所有缺值原因都放回）的 U_opt，只描述'},
 'U_opt': {
   'definition': '每個換股日：受影響股（governing 原因）加入合格集合 el′＝el ∪ A；52 週分數百分位在 el′ 內重算（pandas rank pct，average）；'
                 '營收百分位：A 內股票設為 1.0，其他股票＝其 y3 在「el′ 內非 A 股」的 average rank ÷ |el′|；合成分數＝0.5×兩者；'
                 '同分次序沿用 official（分數高→y3 高（A 股 y3 視為 +∞）→欄位順序小）；之後照 V2.plan 選 10 檔（開盤漲停遞補、held 集合）。',
   'count': 'U_opt 槽位＝Σ_d |新名單 ∩ A(d)|；另報 changed＝Σ_d |原名單 − 新名單|（含 held 連鎖效應）',
   'archive_override': '補回的封存值若通過一致性（重疊月份 ≤ 2009-11 逐檔誤差 ≤ 0.5%、代碼名稱相符；否則整份來源一票否決）就以真值取代樂觀假設；'
                       f"本次凍結時（{sl['conclusion']}）沒有任何通過的值，故 U_opt 全部用樂觀假設"},
 'denominators': {'D1_2006_2012': '2006-01-01～2012-12-31 訊號日 × 10（G4 第一段；81 日 → 810 槽位）',
                  'D2_2006_2013': '93 日 × 10＝930 槽位', 'governing_share': 'max(U_opt_2006_2012 / 810, U_opt_2006_2013 / 930)'},
 'rule': {'le_5pct': '只在 main_v2/FINDINGS 與報告第 0 節加註；o14 金融勘誤文字改成「以現行截斷營收面板為準，上界 X%」',
          'gt_5pct': '主策略 G4 第一段改標「未證實」；G5 第一段標「存活者偏誤未排除」',
          'threshold': 0.05, 'strict': '比例 > 0.05 才算超過（= 0.05 算 ≤）',
          'no_override': 'U_p90、U_carry、方向描述、子樣本都不得推翻 U_opt 判定'},
 'descriptive': {'U_p90': 'A 股營收百分位設為 0.9（y3 同分次序以池內 y3 的 p90 代入）',
                 'U_carry': 'A 股 y3＝往回 1～24 個月內最近一個有值的 y3，與其他股一起照正式規則排名；找不到者不放回',
                 'subsamples': ['分年', '按原因', '受影響股進榜當日 52 週分數百分位 ≥ 0.8 與否', '2006～2009 對 2010～2012', '金融（C0 已排除，計數為 0 只是確認）'],
                 'who': '被擠出的原名單股、進榜的受影響股與其下市日與原因（listing_master 下市日；原因只描述：併購、轉金控、終止上市）',
                 '5854_6012': '更新 o14「5854 從未入選、6012 一次」：報告 U_opt 下兩檔的入選次數'},
 'replication': '複製員只看本規格，自寫原因分類、排序與計數（只 import v2_core 的股池與 V2.sig_days／S52／V60）；槽位清單逐筆相同才算一致；不一致又無法解釋時取較大的 U',
 'explore_summary': {'P1': {k: ex['P1'][k] for k in ('n_domestic_last_trade_2009_12_2012_12', 'n_last_rev_le_2009_11', 'share_le_2009_11')},
                     'SURV_stockmonths': ex['P2']},
 'search_summary': {'n_queries': sl['n_queries'], 'n_repos_inspected': sl['n_repos_inspected'], 'conclusion': sl['conclusion']},
}
spec['frozen_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
txt = json.dumps(spec, ensure_ascii=False, indent=1)
P.write_text(txt, encoding='utf-8')
h = sha(P)
R = HERE / 'results.json'
res = json.loads(R.read_text(encoding='utf-8')) if R.exists() else {}
res['PREREG'] = {'file': 'ERRATUM2_SPEC.json', 'sha256': h, 'utc': spec['frozen_utc'], 'note': '凍結時尚未計算任何名次（explore 只看覆蓋）'}
R.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(h, spec['frozen_utc'])

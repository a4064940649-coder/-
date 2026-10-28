#!/usr/bin/env python3
"""步驟 4：凍結。寫 results.json 的 PREREG（全部腳本、引擎庫、raw/ 與 data/ 輸入、探索與迭代輸出的 sha256＋時戳），
並寫前瞻登記 live_tracking/o7_capmatched/SPEC.json。斷言此時還沒有任何 2023 年以後的策略計算（results_oos.json 不存在）。"""
import sys, json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
LIVE = AG / 'live_tracking' / 'o7_capmatched'
assert not (HERE / 'results_oos.json').exists(), '已有樣本外輸出，不能事後登記'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
files = sorted([p for p in HERE.rglob('*') if p.is_file() and p.name not in ('results.json',) and '__pycache__' not in p.parts
                and not p.name.startswith('results_smoke') and not p.name.startswith('navs_smoke')])
sha_files = {str(p.relative_to(HERE)): sha(p) for p in files}
ext = {'../live_tracking/o7_capmatched/o7_paper.py': sha(LIVE / 'o7_paper.py'), '../../main_picks_today.py': sha(AG.parent / 'main_picks_today.py')}
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
FROZEN = it['round1']['chosen']
gate = json.loads((HERE / 'results_gate.json').read_text(encoding='utf-8'))

FROZEN_SPEC = f"""
o_capmatched_selection 凍結規格（PREREG）— 市值匹配選股：0050 自由流通骨架＋層內「52 週高點×營收」替換

一、主版本 {FROZEN}（F1 層替換，K＝5、N＝20、衛星市值^0.5 加權）
- 換股日：每月第一個日 ≥ 11 的面板交易日 d；d 收盤算訊號，d 的次一交易日 e 開盤成交；持有到下一個 e。
- 市值：發行股數（data/shares_est.parquet，s2_capgate.py 回推：錨＝MI_QFIIS 2025-10-07～2026-10-21 實值；之前依 R1 exdiv.csv 配股／corp_actions 減資、
  R2 exdiv_5y 權與權息（權息現金以 pe_5y 殖利率估）、R3 2021-10 前除權息日只認減資型（f＜0.8）、E5 停牌後恢復跳升 > 25% 視為減資）× 前值補收盤。
  市值名次：有市值、近 6 日有成交、非 DR（91xx）的上市股，由大到小。
- 0050 複製權重 R：e 日開盤適用的臺灣50 成分（2015-09-18 起用 etf_rebalance 公開名單倒推的 tw50_members.json；生效日收盤後換新；
  2015-09 以前為規則重建：審核基準日市值名次，非成分 ≤ 40 納入、成分 > 60 剔除、補足 50），權重 ∝ 市值 × φ（自由流通係數，
  2026-10-09 0050 實際權重 ÷ 回推市值權重，台積電＝1，快照外的股票用中位數 0.764）。已下市成分 2311、2325、3474 用複製誤差最小的固定股數，
  2823、2888 用「同日最小成分市值」。
- 衛星合格：市值名次 ≤ 300、非金融保險（stock_list 金融保險業、28xx、5876、5880、已下市金融名稱）、非 KY／DR、60 日均成交值 ≥ 1,000 萬（gapfix 窗）、
  收盤 ≥ 10、52 週比（gapfix）有值、近 3 月營收年增率有值（V4：R_FIX ＜2021-03-01、Y5_FIX 之後；期限日＝訊號月 10 日起第一個加權指數交易日
  ［2019-02、2023-04 用 15 日］≥ 訊號日 → 用 m−2..m−4；2026-09 用 firstseen；2026-01～08 INS15 一律視為未公告）、
  官方處置名單（公告日 ≤ d 且 e 落在處置期間 → 排除）、e 日有開盤成交。
- 分數＝0.5×52 週比百分位＋0.5×營收 3 月年增率百分位（在合格集合內）；同分依營收高、再代碼序。
- 權重：H＝市值前 5 大成分（必含台積電）∪ 全部金融成分，持有＝R；不在 H 的成分若 R > 5pp，至少保留 R−5pp；
  剩餘權重 B 給「市值名次 > 5、不在 H」的合格股分數最高 20 檔，權重 ∝ 市值^0.5，單檔主動權重（相對 R）≤ 5pp，觸頂則往下加檔；
  開盤漲停（跳動點漲停價）且未持有的衛星跳過、由下一名遞補。其餘成分權重＝0。
- 約束：台積電持有＝R（主動 0，±2pp 內）；|單檔主動| ≤ 5pp。
- 成本：一般＝買 0.0855%＋0.1%、賣 0.0855%＋0.3%＋0.1%（滑價）；壓力＝每邊 0.5%。日頻引擎（cmlib.simulate_w）：凍結停牌持股、二分法解成本後總值、
  買進日開盤→收盤；日報酬用 exfix（面板 flag 1 除權息日的 ret＝收盤/開盤×(1＋當日非除權息股開盤跳空中位數)−1）；面板口徑（ret＝0）為敏感度。
- 核心＋衛星：C50＝0050 50%＋本題 50%、C70＝0050 70%＋本題 30%，每個換股日收盤再平衡，轉移金額付 0.0855%＋0.1%。

二、期間與統計
- 挑選期 SEL＝訊號窗起點 2019-01～2022-12（48 窗）；驗證期 VAL＝2023-01～2026-10-12（面板 2024-01 缺，2 月換股日為 2024-02-15）；
  POST2019＝2019-01～2026-10-12；結構前 PRE＝2006-04～2018-12（市值與成分為回推，只看方向）。
- 對 0050（finlab 含息）的訊號窗月超額：均值、t、追蹤誤差（×√12）、IR＝均值×12／TE；年化、最大回撤（日頻）。

三、預測與判定（事前，照總監計畫）
- P1：各市值層內分數前 30% 減該層市值加權（扣一般成本：前 30% 籃子單邊周轉×來回成本）。51～300 名層 ≥ +0.3pp／月、t ≥ 1.5；11～50 > 0；2～10 名 |t| < 1。
  【揭露】挑選期已算出：51～300 −0.01pp（t −0.04）、11～50 −0.12pp、2～10 −0.19pp → P1 在挑選期已不成立，判定 (c) 已確定不成立。
- P2：凍結版挑選期 ≥ +0.25pp、TE ≤ 10%、IR ≥ 0.3；驗證期超額 > 0、IR ≥ 0.2。
- P3：TE ＜ 10.3%（主策略 20.6% 的一半）；最大回撤不比 0050 深 10pp 以上。
- P4：本題−主策略（AB_V4）的月差：PRE 平均 ＜ 0、POST2019 ＞ 0；對（0050−同池等權）的 β 在 0.6～1.0。
- P5：台積電主動權重貢獻 ＜ 20%；去台積電骨架對去台積電複製的超額方向不變。
- P6（探索）：股價 ≥ 500 的層內殘差在 2020-10-26 後相對 ＜ 500 擴大。
- P7：C50／C70 兩期超額 > 0；IR 不低於本題單獨的 0.8 倍。
- 判定（凍結版與 C50、C70 各自）：待觀察需 (a) SEL IR ≥ 0.3 且 t ≥ 1.5；(b) VAL 超額 > 0、t ≥ 1.0，且 POST2019 IR > 主策略 AB_V4 同期 IR；
  (c) 51～300 層殘差兩期都 > 0；(d) 台積電貢獻 ＜ 20%；(e) 壓力成本、j 中位數下 VAL 超額 > 0，且 VAL 拿掉最好 3 個月 > 0。
  任一不成立 → 不可用（未達待觀察）。另列硬性不可用觸發：VAL 超額 ≤ 0；51～300 層兩期都 ≤ 0；台積電貢獻 ≥ 50%。歷史資料不得判可用。

四、穩健性（全部報告）：12 版本全部的 VAL；換股日 j＝0～12（D_m 之後第 j 個交易日，每個 j 重跑換股與成本）的 t 中位數與 p10；
  拿掉最好 1／3／5 個月、第一年、最後一窗；排除 src 3 期間的窗；壓力成本；單邊滑價兩平點；可執行版 30 萬／100 萬／300 萬／1,000 萬 × FIX／GROW ×
  o5 衝擊（0.1%＋0.5σ√(Q/OA)）／一個跳動單位，手續費 2.8 折最低 20 元、零股價差；台積電貢獻與去台積電骨架；P4 結構前對照；逐年；產業與單股主動貢獻；
  面板口徑。有效檢定數＝12 版本 × 13 j＝156。

五、前瞻登記：凍結版＋C50＋C70 寫進 live_tracking/o7_capmatched/（SPEC.json、o7_paper.py），2026-11 換股日起每月紙上淨值；
  判定改用 o_tier_shrink_sequential 登記的序貫規則（該題尚未登記時，暫用 SPEC.json 內的保守序貫規則）。
"""
PRED = {'P1': '51～300 層 ≥ +0.3pp／月、t ≥ 1.5；11～50 > 0；2～10 |t| < 1（挑選期已不成立）',
        'P2': '挑選期 ≥ +0.25pp、TE ≤ 10%、IR ≥ 0.3；驗證期 > 0、IR ≥ 0.2', 'P3': 'TE < 10.3%；MDD 不比 0050 深 10pp 以上',
        'P4': 'PRE 平均 < 0、POST2019 > 0；β 0.6～1.0', 'P5': '台積電貢獻 < 20%；去台積電骨架方向不變', 'P6': '探索', 'P7': 'C50／C70 兩期 > 0；IR ≥ 0.8×單獨'}
PREREG = {'topic': 'o_capmatched_selection', 'frozen_version': FROZEN, 'combos': {'C50': 0.5, 'C70': 0.7}, 'frozen_spec': FROZEN_SPEC, 'predictions': PRED,
          'selection_result': it['round1']['results'][FROZEN]['normal'], 'n_versions': len(it['round1']['versions']), 'rounds': it['rounds_run'],
          'gate_summary': {'gate_i_post2019_min_agree': gate['gate_i']['post2019_min_agree'], 'gate_ii_spearman_top50': gate['gate_ii']['spearman_top50'],
                           'gate_iii_TE_2019_2022': gate['gate_iii']['exfix_FA|2019-01~2022-12']['TE_month_ann_pct'], 'skeleton_float_adjusted': gate['skeleton_float_adjusted'],
                           'pass_all': gate['gate_pass_all']},
          'disclosures': [
              '閘門 (iii) 的口徑（exfix、自由流通 FA）是在看過「固定股數、面板口徑」的速算（TE 3.07%）之後才寫成選擇規則的；FA 係數用 2026-10-09 快照（未來資訊用於資料建構）。',
              's2 閘門算過 0050 複製（不含選股）在 2023～2026 的追蹤誤差 1.66%（資料檢查）；s1 贏家描述用了 2020-08～2026-09 的 ETF 與主策略報酬（只定性）。',
              'R3 配股偵測門檻用 2021-10～2024-08 的真值樣本選（不偵測配股誤差最小）。',
              'P1 與判定 (c) 在挑選期已確定不成立（51～300 層 −0.01pp），凍結前已知；照總監門檻，歷史判定上限為「不可用（未達待觀察）」，前瞻登記照做。',
              '2019 年後的大型股資料都已被前面各題看過（seen ledger 見 explore.json STEP0），驗證期不是盲測。',
              '冒煙測試 results_smoke.json 只用 ≤ 2023-01 換股日的資料。'],
          'sha256_files': sha_files, 'sha256_external': ext,
          'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
blob = json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()
res = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(blob).hexdigest()}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
SPEC = {'spec': {'title': 'o7 市值匹配選股（0050 自由流通骨架＋層內 52 週高點×營收替換）前瞻紙上追蹤',
                 'origin': f"agents/o_capmatched_selection（PREREG sha256 {res['PREREG_sha256']}）",
                 'frozen_version': FROZEN, 'combos': {'C50': '0050 50%＋本題 50%', 'C70': '0050 70%＋本題 30%'},
                 'start_signal_on_or_after': '2026-11-01',
                 'run': 'python3 -I -B research/tw_stock_selection/agents/live_tracking/o7_capmatched/o7_paper.py --data /home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data（每月換股日收盤後；資料先 git pull）',
                 'outputs': ['targets.tsv（訊號日、成交日、代碼、目標權重、0050 複製權重、成分快照日）', 'paper_nav.tsv（成交日開盤→下一成交日開盤：F、REP、0050、C50、C70）'],
                 'why': '2019～2026 大型股報酬都已被看過，2026-11 起才是乾淨樣本。',
                 'sequential_rule': {'primary': '依 agents/o_tier_shrink_sequential 登記的序貫判定規則（本輪 C 題）；以該題 PREREG 為準。',
                                     'fallback_until_registered': {'metric': 'F、C50、C70 各自對 0050 的成交窗月超額（扣一般成本）',
                                                                   'looks': '每 12 個月檢視一次（2027-11、2028-11、2029-11…）',
                                                                   'upgrade_to_usable': '累積 t ≥ 2.8（第 12 月）／2.4（第 24 月）／2.1（第 36 月起），且 IR > 主策略同期 IR',
                                                                   'stop_as_unusable': '第 24 月起累積平均 ≤ 0 且 t ≤ −1.0；或任一年 TE > 10%',
                                                                   'otherwise': '續追蹤（待觀察）'}},
                 'sha256': {'o7_paper.py': ext['../live_tracking/o7_capmatched/o7_paper.py']}},
        'frozen_utc': PREREG['timestamp_utc']}
(LIVE / 'SPEC.json').write_text(json.dumps(SPEC, ensure_ascii=False, indent=1), encoding='utf-8')
print('PREREG', res['PREREG_sha256'], PREREG['timestamp_utc'], len(sha_files), 'files')

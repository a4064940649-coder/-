#!/usr/bin/env python3
"""題 B 規格凍結（ERRATUM_SPEC）：在產生任何名單之前寫死分類器 C0／C1／C1B／C2、v2_core 版本、輸出欄位與判定規則。
只產出名單與計數，不產出任何報酬。用法：python3 -I -B s6_erratum_spec.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g8lib as G

AG = G.AG
O13 = AG / 'o_pit_shares_industry_2006_2013'
SNAP = O13 / 'raw/industry_snap'
man = json.loads((O13 / 'raw/manifest.json').read_text(encoding='utf-8'))['files']
snaps = {}
for rel, meta in man.items():
    if rel.startswith('industry_snap/') and 'industry_code' not in rel:
        eff = meta.get('update_line') or meta['commit_time'][:10]
        if 'twse_list' in rel: eff = '2013-12-18'
        snaps[rel] = dict(effective_date=eff, commit_time=meta.get('commit_time'), evidence=meta.get('evidence'), sha256=meta['sha256'],
                          sha256_now=G.sha(O13 / 'raw' / rel))
spec = dict(
    utc=G.now_utc(),
    purpose='主策略 2006-01～2013-12 金融排除 PIT 勘誤：只重建名單與計數，不重算任何報酬',
    base=dict(code='agents/main_v2/v2_core.py（只 import）＋independent_audit/ia_core.py（只 import）',
              v2_core_sha256=G.sha(AG / 'main_v2/v2_core.py'), ia_core_sha256=G.sha(AG / 'independent_audit/ia_core.py'),
              config="V2('gapfix', YL, Y5, exfin=True, unavail=None)；plan(10, tie='official')（同分：營收 3 月年增率高 → 代碼小；開盤漲停且上月未持有 → 遞補）",
              window='訊號日 2006-01-01～2013-12-31', revenue='長期營收 tw_revenue.csv（v2_core.load_revenue 的 YL；2021-03 前）'),
    classifiers=dict(
        C0='主策略現行旗標＝v2_core.FINMASK：stock_list.csv（2026）產業＝金融保險業 ∪ 代號前 2 碼 28 ∪ {5876, 5880}',
        C1='PIT 只往回看：訊號日 d 取生效日 ≤ d 的最新 dated 產業快照（toomore/goristock、toomore/grs；生效日＝檔內 UPDATE 列日期，無 UPDATE 列者＝commit 日；twse_list＝2013-12-18）；'
           '快照內有該股 → 金融＝TSE 產業別 17（金融保險）；d 早於最早快照（2011-07-28）或快照內無該股 → 無法判定，沿用 C0',
        C1B='敏感度（非 PIT，往前套用）：同 C1，但 d 早於最早快照時改用最早快照（2011-07-28）；只用來估計 2011-08 以前的可能異動量',
        C2='敏感度：代號前 2 碼 ∈ {28, 58, 60}，或任一來源名稱（stock_list 2026、上列快照）含 金控／金融控股／銀行／商銀／保險／產險／人壽／壽險／證券／票券／期貨 或以 銀／金／證／保／壽 結尾；'
           '名稱規則排除非金融誤配 {2049 上銀, 2947 振宇五金, 3049 精金, 4576 大銀微系統, 7819 精誠金融, 9917 中保科, 9925 新保}（排除清單在跑名單前依名稱寫死）'),
    snapshots=snaps,
    outputs=dict(
        per_signal=['signal', 'n_eligible', 'C0_picks', 'Ck_picks（k＝C1, C1B, C2）', 'changed_slots_k'],
        s5854='5854 合庫：2006-01～2011-11 在 C0 股池（合格集合）的月數、名次（C0 完整排序中的位置）分布、進入 C0 前 10 名（實際買進名單）的月數 k 與日期、逐年分布',
        s2841='2841 台開：C0 下被排除；把 2841 放回（只改這一檔）時進入前 10 名的月數與日期；C1／C1B 下的實際異動',
        und_861='o13 D2 口徑（前 300 流動性池股月；快照取「之前 731 日內最新，否則之後 731 日內最早」）無法判定且 C0＝金融的股月：股票清單、股月數；另報 C1 嚴格口徑（只往回看）的同一清單',
        slots='每個訊號日 10 個槽位；某分類器下 C0 名單中某檔不在該分類器名單 → 該槽位異動；總槽位＝10×訊號日數'),
    rules=dict(
        R6='5854 進 C0 前 10 名的月數 k ≥ 3 → 依 o13 事前規則，在 main_v2/FINDINGS 與報告第 0 節加註「2006～2011 段含非 PIT 金融股」（由主研究員執行；本題只產出文字）',
        R8='異動槽位比例取 max(C1, C2)（C1B 只報不判）；> 5% → G4 第一段（2006～2012）改標「含非 PIT 金融分類，未證實」',
        no_returns='任何程式不得計算報酬、淨值或 t 值'),
    predictions=dict(P6='k ∈ [0, 6]，集中在 2006～2007 與 2009 反彈段', P7='C1 或 C2 異動槽位 ≤ 2%（≤ 20 個）；2841 擠進前 10 的月數 ≤ 5', P8='異動 > 5% 則否定「2006～2013 不受金融分類影響」'),
)
sh = G.register('ERRATUM_SPEC', spec)
print('ERRATUM_SPEC', sh)

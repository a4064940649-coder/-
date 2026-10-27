#!/usr/bin/env python3
"""事前設定（看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '溫水煮青蛙動能（Frog-in-the-Pan, Da-Gurun-Warachka 2014）：12-1 動能前 20% 中資訊離散度 ID 最低的一半',
 'sources': ['https://www3.nd.edu/~zda/Frog.pdf',
             'https://alphaarchitect.com/2015/11/frog-in-the-pan-identifying-the-highest-quality-momentum-stocks/'],
 'source_access_note': '規則依任務敘述逐條轉成程式；教學沒寫明的細節用論文／Alpha Architect 最常見寫法，於下方固定，看結果後不改。',
 'type': '選股型（月訊號、3 批重疊持有 3 個月）：組合報告（年化、日頻最大回撤、對 0050 日曆月 t、對同池等權日曆月 t、逐年、周轉）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才算任何 rolling／位置回看。只用 adj（含息還原）。',
   'daily_return': 'r_t = adj_t / adj_(t−1) − 1（剔除稀疏日後的相鄰交易日；任一端缺值則缺）；當日 flag ∈ {3,4,5} 的 r_t 設為缺值（不計入 PRET、%pos、%neg）。',
   'window': '換股訊號日 d（位置 i）：取 r 的位置 i−251 … i−21（即 adj 位置 i−252 到 i−21 之間的 231 個日報酬，跳過最近一個月）。有效日報酬 >= 200 才有訊號。',
   'PRET': 'PRET = Π(1 + r_t) − 1（同一窗、有效日）。',
   'ID': '%pos = #(r>0)/#有效日，%neg = #(r<0)/#有效日（0 報酬日計入分母、不計入分子）；ID = sign(PRET) × (%neg − %pos)。ID 越低＝越「連續」（溫水）。',
 },
 'universe': '上市流動股：4 碼數字代碼、排除 91xx（TDR）；判斷日 d：60 日均成交值（剔除稀疏日後滾動、上市後停牌日記 0、需滿 60 日）>= NT$1,000 萬、原始收盤 >= 10 元、d 的 flag ∈ {0,1,2}、d 有 adj、d 不在處置期間（處置名單 2021-09 起才有）、d+1 有開盤價與 adj、PRET 與 ID 都有值。',
 'selection': {
   'PRIMARY_FIP20_LOW_H3': '股池內依 PRET 由高到低排序取前 floor(0.2×N) 檔（同值依代碼順序）＝ M 組；M 組內依 ID 由低到高（同 ID 時 PRET 高者優先）排序，取前 ceil(|M|/2) 檔，等權。',
   'blocked': 'd+1 開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）且該批次未持有 → 買不到，由同一排序的下一名（M 組內 ID 次低者）遞補，維持目標檔數；該批次已持有者照常保留。',
 },
 'execution': {
   'calendar': '每月最後一個交易日 d（剔除稀疏日後的日曆）收盤算訊號；d+1（隔月第一個交易日）開盤成交；最後一個訊號日 2026-09-30，持有到面板最後一天 2026-10-22 收盤。訊號需 252 日歷史，第一個有效訊號日約 2005-01 月底。',
   'holding_3m_3tranches': '三個批次（各 1/3 起始資金，之後各自複利、不在批次間再平衡）。批次 k 在訊號月序號 m ≡ k (mod 3) 時換股（其餘月份不動）；第一個訊號月三個批次同時建倉（暖身，批次 1、2 第一輪只持有 1、2 個月）。',
   'holding_1m': '變體 (a)：每月全部換股、持有 1 個月。',
   'engine_cost': 'exec_rebuild/exlib 的價格陣列（還原開盤 AO、還原收盤 AC）與 cost_rates：主要＝買 0.1425%×0.6＋滑價 0.1%；賣 0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%，即 BRIEF「0.5%」情境加滑價）；壓力＝買 0.5%＋賣 0.5%（來回 1%）；另報毛報酬。換股時把漂移後持股調回等權，買賣金額分別乘費率；持有期間停牌／下市以最後成交價計，下次換股時賣出。',
   'empty': '股池 < 50 檔該月不換股（沿用持股）。',
 },
 'variants': {
   'PRIMARY': 'FIP20_LOW_H3：PRET 前 20% × ID 低半、持有 3 個月、3 批重疊',
   'all_reported': [
     'FIP20_LOW_H1 (a)：同 PRIMARY 選股，月換股持有 1 個月',
     'FIP10_LOW_H3 (b)：PRET 前 10%（floor(0.1×N)）× ID 低半，3 個月 3 批',
     'FIP20_HIGH_H3 (c) 對照：PRET 前 20% 中 ID 高的一半（ID 由高到低排序取 ceil(|M|/2)，遞補往低 ID 走），3 個月 3 批',
     'FIP20_LOW_H3_MA200 (d)：PRIMARY 加大盤濾網——每個訊號日 d 收盤加權指數（taiex_twii close，價格指數）> 其 200 日均線才持股；否則所有批次在 d+1 開盤全部賣出持現金（現金報酬 0）；濾網恢復當月，原本持現金的批次與當月輪到的批次都用當月名單建倉（之後照原本 3 個月輪序換股）。',
     '診斷 MOM20_H3：PRET 前 20% 全部等權，3 個月 3 批（用來看 ID 篩選相對純 12-1 動能的增量）',
     '診斷 LOW−HIGH 毛價差：PRIMARY 毛報酬 − (c) 毛報酬，月 t',
     '動能家族直接比較（momentum_family/mlib.py 相同股池、日曆、成本、M2 成交）：mlib 日曆（每月第一個 >= 11 日交易日）、前 300 流動股池、月換股；FIP_MLIB_n30 = mlib P2（12-1 動能）前 20%（約 60 檔）中 ID 最低的 30 檔（遞補依 ID 次低、再依 P2）；對照 P2_MLIB_n30 = P2 前 30 檔（重跑 mlib.select）；統計用 mlib.stats（換股日窗 t，對 mlib 共同股池等權）。',
   ],
   'note': '非 PRIMARY 只作穩健性與機制參考，不得用來改判。',
 },
 'benchmarks': {
   'EW': '同池等權：同一訊號日、同一股池（PRIMARY 的股池），同一引擎 d+1 開盤月換股、不扣成本（毛）——比扣成本的策略嚴格；另報 EW 扣成本年化。',
   '0050': '0050 含息（finlab-txd etf_0050.csv）',
 },
 'stats': '年化（實際天數）、日頻最大回撤、波動、對 0050 與同池等權的日曆月超額報酬 t（btlib.tstat）、逐年報酬、周轉；挑選期與驗證期分開，另報 2024–2025 與 2024-01～2026-10。',
 'windows': {'select': '2005-01-01~2019-04-30（實際從第一次建倉起）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'extra_reported': ['valid_survivor_2019-05_2023-12（存活者面板）', 'valid_2024-01_2026-10']},
 'verdict_rule': 'PRIMARY（FIP20_LOW_H3，滑價 0.1%）在挑選期與驗證期都要：扣成本後對同池等權（毛）月超額平均 > 0 且 t >= 2，且淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。',
 'known_limits': ['只有上市股、無上櫃', '2019–2023 面板只有存活者（動能策略受存活者偏差影響方向：剔除了下跌後下市者，對高動能組影響應小於對低價／虧損股）',
                  '2024-01-02~2024-02-14 無資料（跨缺口的日報酬是多日報酬，照常計 1 天）',
                  'src 3 期間（2024-02-15~2025-09-23）收盤為 Yahoo 5 分 K 13:20 近似，日報酬正負號可能失真，影響 %pos／%neg',
                  '無市值資料，股池以成交值與股價篩選'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

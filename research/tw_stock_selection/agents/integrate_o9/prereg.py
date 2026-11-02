#!/usr/bin/env python3
"""integrate_o9 事前登記：把 o9 唯一通過審查（待觀察）的 o_capmatched_pit_close（0050 骨架＋52 週×營收傾斜，F1_K5_N20_sqrt，PIT 版本 a／b）
和主策略（排除金融 n10，AB_V7）整合。寫 results.json 的 PREREG 與 PREREG_sha256；之後 run.py 只跑一次。python3 -I -B prereg.py"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
CM = AG / 'o_capmatched_pit_close'
V7 = AG / 'o_v7_listing_exec_engine'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

FILES = [CM / f'out/run_{c}_{p}.npz' for c in ('a', 'b') for p in ('MAIN', 'PRE')] + [CM / 'out/run_a_MAIN_stress.npz',
         CM / 'out/exec_a.json', CM / 'out/exec_b.json', CM / 's6_exec.py', CM / 'pcore.py', CM / 'data/PREREG.json', CM / 'results.json',
         V7 / 'navs_jscan_v7.parquet', V7 / 'navs_v7.parquet', V7 / 'navs_exec7.parquet',
         Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'), HERE / 'cm_exec.py', HERE / 'run.py',
         V7 / 'data/plans_j7.csv'] + [HERE / f'out/cm_exec_{c}{x}' for c in ('a', 'b') for x in ('.npz', '_check.json')]

PREREG = dict(
    written_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    topic='o9 整合：o_capmatched_pit_close（以下稱 C）× 主策略（排除金融 n10，AB_V7，以下稱 M）',
    legs=dict(
        M='主策略排除金融 n10 正式資料版 V7：紙上 navs_jscan_v7.parquet 的 AB_V7|{normal,stress}|J0..J12（逐 j 日淨值）；官方日 navs_v7.parquet AB_V7|normal。'
          '可執行版：navs_exec7.parquet AB|Jj|K5T|{normal,stress}|{100萬,300萬,1000萬}|{FIX,GROW}（V7 原可執行表；V7X 只有 B10，依 o8 凍結判定不取代）。',
        C='o_capmatched_pit_close 凍結版 F1_K5_N20_sqrt；版本 (a)（PIT 股數全格＋修正＋下市股）為主欄，版本 (b)（股數落後 63 日）為敏感度；'
          '判定一律取 (a)(b) 較不利者。紙上：out/run_{a,b}_{MAIN,PRE}.npz 的逐窗報酬 ra（一般成本）；壓力只有 run_a_MAIN_stress。'
          '可執行：cm_exec.py 唯讀 import C 的 s6_exec.run_exec（K5T 與 K5T＋每邊 0.5%），逐 j 逐窗輸出報酬；已在 PREREG 前執行（只產生 C 腿本身，不做混合），重現 exec_{a,b}.json 差 0，輸出雜湊登記於下。',
        E='0050 含息（etf_0050.csv），C 檔案內的 r5（已核對與 etf_0050.csv 窗報酬差 0）。'),
    windows='一律用 C 的窗：13 個 j，每窗＝訊號日 sd 收盤到下一訊號日收盤（最後一窗到 de）。M 的窗報酬取 M 的日淨值在這些日期（前值填補）的比值。'
            'PRE＝2014-01～2019-04 錨定月（終點 2019-05 同 j 日），MAIN＝2019-05～2026-10，ALL＝PRE 與 MAIN 串接（2014-01～2026-10）。'
            '描述性子期：MAIN_2024＝MAIN 中 sd ≥ 2024-01-01 的窗。C 沒有 2014 年前資料，所以 2006～2013 全部不可評估。',
    combos=dict(
        MIX='第二條腿固定比例混合（主策略比例 w）：w＝0.8（80/20）、0.5（50/50）、0.2（20/80）。三個比例全部報告，不挑；比例照任務指示與 integrate_o7 慣例，事前寫死，不看保留期。'
            '袖子法（與 integrate_o4～o8 sleeve 同一公式，改成窗頻）：每窗末（C 的訊號日收盤）調回比例；移轉金額 m 的成本：一般 m×(0.4855%＋0.1855%)，壓力 m×(0.5%＋0.5%)。'
            '可執行混合：兩腿用同一級距的可執行淨值（腿實際只有帳戶×w，衝擊高估，偏保守），移轉成本同上。',
        REF='對照：M 單獨（w＝1）、C 單獨（w＝0）、0050；以及 M/0050 同比例混合（0050 腿賣出多 0.1%，同 integrate_o8），用來分離「C 取代 0050 當核心」的效果。',
        FILTER_BONUS='不適用：C 的分數就是主策略的分數（0.5×52 週比百分位＋0.5×近 3 月營收年增率百分位），當主策略濾網或加分項等於恆等變換；'
                     'C 唯一不同的成分是 0050 自由流通權重骨架，只能以第二條腿的形式加入。照實寫「不適用」，不另造變體。'),
    stats='每個組合 × C 版本 (a)/(b) × 期間（PRE、MAIN、ALL、MAIN_2024）：年化（j 中位數，另報官方日 OFF）、窗頻最大回撤（j 中位數）、對 0050 月均超額與 t（j 中位數、p10）、'
          '對 M 單獨月差與 t（j 中位數、p10）、相關係數（corr(M,C) 窗報酬、corr(M−0050, C−0050)、corr(組合, M)）。ALL 另報拿掉最好 3 個日曆年後對 0050 的 t（G4 用）。'
          '可執行（只有 MAIN）：K5T 一般與壓力 × 100 萬／300 萬／1,000 萬 × FIX／GROW：對 0050 月均超額與 t（j 中位數、p10）、對 M 可執行單獨的月差與 t、年化。'
          'M 可執行壓力用 navs_exec7 的 stress（每邊多 0.4%＋其他保守設定），C 用 K5T＋每邊 0.5%，兩者並非完全同口徑，照實註記。',
    decision=dict(
        per_mix='每個 MIX 比例逐一判定，每個統計量取 (a)(b) 較不利者：'
                '(1) ALL 2014～2026 對 0050 t 的 j 中位數 ≥ 2.0；(2) MAIN 對 0050 t 的 j 中位數 ≥ 2.0；(3) PRE 對 0050 月均超額 j 中位數 > 0；'
                '(4) 可執行 K5T 一般成本 6 格（3 級距 × FIX／GROW）對 0050 月均超額的 j 中位數與 p10 都 > 0；(5) 可執行壓力 100 萬 FIX 與 GROW 對 0050 月均超額 j 中位數 > 0。'
                '「可用（建議取代主策略單獨）」：(1)～(5) 全過，且 ALL 與 MAIN 對 0050 t 都高於 M 單獨的對應值。'
                '「待觀察」：(1) 過但 (2)～(5) 有任一不過；或 (1)～(5) 全過但沒有高於 M 單獨。「不可用」：(1) 不過。',
        sop='只有「可用」才建議改 SOP；其餘 SOP 不變（M 單獨仍是正式策略；C 繼續當紙上前瞻追蹤線）。',
        gates='最高標準 G1～G8 對每個組合逐條列。G2 要求 2006～2026 全期 t ≥ 3.0，C 沒有 2014 年前資料，判「未通過（無法評估）」；'
              'G3 本研究不重做 SPA（不新增族群矩陣），沿用各腿既有結果並判未通過或無法評估；G6 本整合無獨立複製；G8 尚未開始。因此任何組合最多只能寫「待觀察」。'),
    predictions='研究員事前預測：MAIN 對 0050 t（j 中位數）80/20 約 0.5、50/50 約 1.2、20/80 約 1.8；ALL 對 0050 t 隨 C 比例增加而上升，20/80 約 2.4、80/20 約 1.6。'
                '所有比例對 M 單獨的月差為負（C 對 M −0.7～−0.8pp／月），t 約 −0.9～−1.5。可執行 K5T 壓力格與 p10 條件會不過，預測三個比例都是「待觀察」或「不可用」，沒有「可用」。',
    honesty='寫 PREREG 前我已讀過 C 的 headline、results.json 的 post（C 對 0050、C 對 AB_V4 月差 −0.83pp、t −0.87；PRE −0.67pp、t −1.51）與 exec_a.json，'
            '也讀過 integrate_o8 FINDINGS（M/0050 混合全部不可用）。PREREG 前只做資料管線核對（窗日期對齊、r5 與 etf_0050.csv 一致、M 與 C 的訊號日在 MAIN 的 J8～J12 各有 1 個不同日 2021-03），'
            '沒有計算任何混合績效。2014～2026 的資料在 C 與 M 研究中都已看過，不是樣本外。',
    tests='判定用檢定：3 個比例 × 5 條件（每個取 (a)(b) 較不利），全部報告。其他數字全部是描述性。',
    sha256={str(f): sha(f) for f in FILES},
)
if __name__ == '__main__':
    RJ = HERE / 'results.json'
    assert not RJ.exists(), 'results.json 已存在：PREREG 只寫一次'
    h = hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    RJ.write_text(json.dumps(dict(PREREG=PREREG, PREREG_sha256=h), ensure_ascii=False, indent=1), encoding='utf-8')
    print(h, PREREG['written_utc'])

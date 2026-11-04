#!/usr/bin/env python3
"""integrate_o14 事前登記（只寫一次）：o14 唯一通過審查（待觀察）的是 o_g8_protocol_fin_erratum（以下稱 G8E），
它有兩個產品：(1) G8 前瞻序貫檢定協定 G8P-1（G8_PROTOCOL.json）；(2) 2006～2013 金融排除 PIT 勘誤（C1／C2／C1B 分類）。
兩者都不是選股訊號、沒有自己的報酬序列。本整合把它們套到主策略（排除金融 n10）上。用法：python3 -I -B prereg.py"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
G8E = AG / 'o_g8_protocol_fin_erratum'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

FILES = [HERE / 'run.py', G8E / 'G8_PROTOCOL.json', G8E / 'g8lib.py', G8E / 'ERRATUM.json', G8E / 'results.json',
         AG / 'main_v2' / 'v2_core.py', AG / 'main_v2' / 'navs_gapfix.parquet', AG / 'independent_audit' / 'ia_core.py',
         AG / 'o_forward_sequential_v2' / 'fslib.py', AG / 'o_forward_sequential_v2' / 'raw/v7/plans_j7.csv',
         AG / 'o_forward_sequential_v2' / 'raw/v7/navs_jscan_v7.parquet', AG / 'o_forward_sequential_v2' / 'raw/v7/jscan_end_days.json',
         AG / 'o_forward_sequential_v2' / 'raw/finlab/data/raw/etf_0050.csv',
         AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet']

PREREG = dict(
    written_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    topic='o14 整合：o_g8_protocol_fin_erratum（G8E：G8 前瞻協定 G8P-1＋金融排除 PIT 勘誤）× 主策略（52 週高＋營收 3 月年增、前 300 池、月換 10 檔、排除金融；以下稱 M）',
    what_is_integrated=('G8E 沒有個股分數也沒有報酬序列，所以「加分項」不適用；「第二條腿」只能是協定本身指定的 0050（協定的描述線 M_E50_30＝主 70／0050 30、C_E50_70＝主 30／0050 70）。'
                        '可以整合的有三件：A 濾網＝把勘誤的 PIT 金融分類（C1；敏感度 C2、C1B）換進 M 的金融排除（只改 2006-01～2013-12 訊號日，2014 起沿用現行 C0，與勘誤範圍相同）；'
                        'B 第二條腿＝M 與 0050 固定比例混合；C 協定歷史套用＝把凍結的 G8P-1 規則原樣套到 M 與各混合的歷史 JMED 序列（滾動 24 個月起點，描述用）。'
                        '注意：G8E 題目本身依總監規格「只計數，不重算報酬」；本整合的 A 塊是整合階段的影響量化，不取代主策略正式數字，也不回寫 G8E。'),
    legs=dict(
        M_v2='main_v2 v2_core V2(gapfix, exfin=True, unavail=None) plan(10, official)，ia_core.simulate，一般成本 rates(0.001)、壓力 rates(stress)；C0 必須逐日重現 main_v2/navs_gapfix.parquet 的 EXFIN|n10|normal（|差| < 1e-9，否則停）。訊號日窗＝S.bounds() 的訊號日收盤到下一訊號日收盤。這是主策略正式數字（年化約 28%）的引擎。',
        M_v7='V7 紙上（含下市股）：navs_jscan_v7.parquet 的 AB_V7|{normal,stress}|J0..J12，窗＝plans_j7 strat AB_V7（seed 0）的各 j 訊號日（fslib.window_excess 錨定月窗）。',
        M_x='V7X 可執行（G8P-1 決策線的引擎）：navs_oos_R4.parquet 的 B10|Jj|K5T|{normal,stress}|{100萬,300萬,1000萬}|GROW，窗＝plans_j7 AB_V7_B10 各 j 訊號日（fslib.cell_jmatrix）。',
        E='0050 含息 etf_0050.csv（fslib.e50）。'),
    periods='ALL＝全部窗（2006-04～2026-10）；SEG1＝錨定月 2006-04～2012-12；SEG2＝2013～2018；SEG3＝2019-01 起；SEL＝～2019-04；VAL＝2019-05 起；HOLD＝2024-01 起（描述）。窗依起點所屬錨定月（M_v2 依訊號日月份）歸期。',
    A_filter=dict(
        variants={'C0': '現行 FINMASK（對照，＝M）', 'C1': 'PIT 只往回看（勘誤主口徑；2011-07-28 前無快照 → 沿用 C0）', 'C2': '代號 28／58／60＋名稱關鍵字（勘誤敏感度）',
                  'C1B': '非 PIT 往前套最早快照（只報不判）'},
        code='遮罩函式依 ERRATUM_SPEC 原樣重寫（快照 sha256 對 ERRATUM_SPEC 逐檔斷言；C2 關鍵字與排除清單照抄）；名單異動數必須重現 ERRATUM：C1 2／930、C2 4／930、C1B 8／930，否則停。',
        stats='各變體 × 成本（一般／壓力）× 期間：年化（訊號日窗幾何）、日頻最大回撤、對 0050 月（窗）超額 t、對 C0 窗差平均與 t、與 C0 窗報酬相關。',
        decision='勘誤「影響可忽略」＝C1 與 C2 在一般成本下：|ALL 年化差| ≤ 0.3pp、|ALL 對 0050 t 差| ≤ 0.05、SEG1 對 0050 幾何超額不變號。成立 → 維持 G8E 結論（不加註、不改標 G4），只在報告記錄量化結果；不成立 → 建議主策略正式數字旁加註量化差。'
                 '任何結果都不建議改 SOP：現行名單用的是現行分類，對當下而言就是 PIT。'),
    B_mix=dict(
        W_LIST=[1.0, 0.8, 0.7, 0.5, 0.3],
        why='w＝M 比例。0.8、0.5 是任務指定例；0.7＝協定描述線 M_E50_30；0.3＝協定描述線 C_E50_70（＝前瞻產生器 MIX_C70）；1.0＝M 單獨對照；0050 單獨為基準。全部事前寫死、全部報告，不看保留期挑比例。',
        rebalance='每窗起點調回 w，成本照前瞻產生器 make_forward.mix：漂移量 |w′−w|×0.5%（w′＝上一窗末的 M 實際權重，第一窗 0）；壓力成本口徑把移轉成本加倍為 1.0%（比產生器保守）。',
        stats='每個 w × 腿（M_v2 一般／壓力；M_v7 一般／壓力 13 j；M_x 一般／壓力 × 3 級距 13 j）× 期間：年化（幾何，窗數÷12 年）、窗頻最大回撤、對 0050 窗超額 t、'
              '對 M 單獨（同腿同 j）窗差 t、相關（混合 vs M、混合 vs 0050、混合超額 vs M 超額）。13 j 的腿報 j 中位數與 p10。ALL 另報拿掉最好 3 個日曆年（依 M 單獨對 0050 超額）後 t。',
        decision=dict(
            c1='M_v7 紙上一般成本 ALL 對 0050 t 的 j 中位數 ≥ 2.0',
            c2='M_x 可執行 100 萬一般成本 SEG3（2019 起）幾何年化超額 j 中位數 > 0',
            c3='M_x 可執行一般成本 3 級距 ALL 幾何年化超額的 j 中位數與 p10 都 > 0',
            c4='M_x 可執行壓力成本 100 萬 ALL 幾何年化超額 j 中位數 > 0',
            c5='M_v7 紙上一般 ALL t 中位數高於 M 單獨',
            verdict='「可用（建議 SOP 改用此混合）」＝c1～c5 全過；「待觀察」＝c1 過但其他有不過；「不可用」＝c1 不過。'),
        sop='只有「可用」才建議改 SOP。G6（無獨立複製）、G8（未開始）必不過，任何組合最多「待觀察」，不可能寫「可投資」或「歷史最高標準通過」。'),
    C_protocol=dict(
        series='決策線＝M_x 100 萬 GROW 一般成本的 JMED（每錨定月 13 個 j 窗超額中位數，有值 j ≥ 7）；混合 w 的 JMED＝每個 j 先算混合窗超額再取中位數。另以 M_v7 紙上 JMED 描述。',
        rule='G8_PROTOCOL.json（sha256 799f8b84… 斷言）的 params 原樣：σ_d、c、b、c_d、up_looks {12,24}、下修 12／18／24、refit 1.5σ_d；用 g8lib.run_rule（只讀 import，sha 對 protocol.code_sha256 斷言）與 oc() 的合併程序（字面 G8 未過 → 前瞻失敗、升級路徑關閉）。'
             '混合 w 的 σ_d 改為 w·σ_d（混合超額約為 w 倍 M 超額；協定本身只登記 JMED，這是描述用的換算，不是協定的一部分）。',
        starts='起點＝每個錨定月 s，只要 s 起有 24 個連續有效月（沒有作廢；歷史資料無時戳作廢）。一個起點＝一條 24 個月的假前瞻路徑。路徑高度重疊，只描述，不做顯著性。',
        report='各 w × 線：起點數；24 個月內 FORWARD_FAILED（字面 G8 未過或下修）、S1_EXCEPTION_EVIDENCE（升級）、NO_REJECTION_24 的比例；字面 G8（前 12 個月和 > 0）通過率；依起點年分 2006～2012／2013～2018／2019～2024 分報；'
               '最後一個可完整 24 個月的起點與最後 12 個月（字面 G8 視窗）的結果。對照：協定 OC（α＝0 時 24 個月內合併失敗約 0.52、升級約 0.02～0.04）。',
        decision='描述性，不改協定、不改判定。若歷史 M 單獨的升級比例 < 協定 OC α＝+1 的 11～17%，在 FINDINGS 寫明「以歷史實際報酬看，24 個月 S1 翻案路徑大概率走不通」。'),
    predictions=('事前預測：A：C1／C2 差 < 0.1pp／年、t 差 < 0.02，影響可忽略。B：混合超額約 w 倍，t 略降；M_v7 紙上 ALL t（M 約 1.6～1.7 的 j 中位數）× 各 w 都 < 2.0 → c1 不過 → 全部「不可用」；'
                 '最大回撤隨 0050 比例增加從約 −65% 縮到約 −58%。C：M 單獨的歷史起點中，FORWARD_FAILED 約 45～60%，升級 < 10%；2019 起起點失敗比例更高。'),
    honesty=('寫 PREREG 前我已讀過：G8E 的 results.json（EXPLORE、PREREG、OC_FINAL、ERRATUM_SPEC、ERRATUM、G8_FREEZE）、G8_PROTOCOL.json、s7_erratum.py、g8lib.run_rule／oc；'
             'integrate_o11 的 prereg 與判定結果（V7X 上 0050 混合全部不可用，M 單獨紙上 t 在族群格點上 1.52）；main_v2 FINDINGS 的正式數字；V7 FINDINGS（AB_V7 t 2.60，j 中位數 t 1.69）。'
             '寫 PREREG 前沒有計算任何本整合的組合績效。2006～2026 資料在主策略各研究中都已看過，沒有樣本外。'),
    tests='判定用：A 一條規則（2 變體）；B 4 個比例 × 5 條件；C 描述。',
    sha256={str(f): sha(f) for f in FILES},
)
if __name__ == '__main__':
    RJ = HERE / 'results.json'
    assert not RJ.exists(), 'results.json 已存在：PREREG 只寫一次'
    h = hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    RJ.write_text(json.dumps(dict(PREREG=PREREG, PREREG_sha256=h), ensure_ascii=False, indent=1), encoding='utf-8')
    print(h, PREREG['written_utc'])

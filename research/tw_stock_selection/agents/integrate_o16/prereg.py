#!/usr/bin/env python3
"""integrate_o16 事前登記（只寫一次）。o16 通過審查的兩題都不產生選股訊號、沒有自己的報酬序列：
  (1) o_g8p2a_protocol_addendum（G8P-2a，待觀察）：G8 前瞻序貫協定 v2a（G8_PROTOCOL_v2a.json sha256 13972450…，選定 A1）。
  (2) o_finmask_delisted_fin_erratum（FM，可用）：金融遮罩 PIT 勘誤計數（M＝1.73% ≤ 5% → 只加註）。
本整合把它們套到主策略（排除金融 n10）上。用法：python3 -I -B prereg.py"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
FM = AG / 'o_finmask_delisted_fin_erratum'
GA = AG / 'o_g8p2a_protocol_addendum'
G2 = AG / 'o_g8p2_protocol_refreeze'
ER = AG / 'o_rev_trunc_2009_erratum'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

FILES = [HERE / 'run.py', HERE / 'smoke' / 'smoke.out',
         FM / 'ERRATUM_SPEC_FM.json', FM / 'FM_COUNTS.json', FM / 'results.json', FM / 'ERRATA_TEXT.json', FM / 'fclass.py', FM / 'flib.py',
         GA / 'G8_PROTOCOL_v2a.json', GA / 'FREEZE_MANIFEST_v2a.json', GA / 'g8_evaluate_v2a.py', GA / 'results.json',
         G2 / 'G8_PROTOCOL_v2.json', G2 / 'results.json',
         ER / 'erlib.py', ER / 'ERRATUM2.json', AG / 'o_g8_protocol_fin_erratum' / 'ERRATUM.json', AG / 'integrate_o14' / 'results.json',
         AG / 'main_v2' / 'v2_core.py', AG / 'main_v2' / 'navs_gapfix.parquet', AG / 'independent_audit' / 'ia_core.py',
         AG / 'o_forward_sequential_v2' / 'fslib.py', AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet',
         Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')]

PREREG = dict(
    written_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    topic='o16 整合：G8P-2a 前瞻協定補遺 ＋ FM 金融遮罩 PIT 勘誤 × 主策略（52 週高 0.5＋營收 3 月年增 0.5、前 300 池、月換 10 檔、排除金融；以下稱 C0）',
    what_is_integrated=('兩題都沒有個股分數也沒有報酬序列，依 S1 本輪不產生新選股訊號。可整合的只有兩塊：'
                        'A 濾網（敏感度）＝把 FM 的 PIT 金融遮罩當成主策略的濾網：2006～2013 訊號日把 v2_core.FINMASK 換成 F1（PIT 主口徑）／F2（靜態敏感度）／F3（F1∪F2，描述），'
                        '2014 起維持 C0 遮罩；同樣套到 ER2 的 U_opt 名單（U_opt_F1／F2／F3，與 U_opt 比）。名單必須逐筆等於 FM_COUNTS.json 的 lists。'
                        'FM 本身依規格只計數、不算報酬；本塊是整合階段的報酬敏感度，不取代主策略正式數字，也不回寫 FM。'
                        'C 協定歷史套用＝獨立重寫 G8P-2a 選定版本 A1（z^A＝S_n／(c·max(σ_d,s_n)·√n)、b_a 2.27963、c_d 1.20985）與 G8P-2 V1（z＝S_n／(max(c·σ_d,s_n)·√n)、b 1.93376），'
                        '以 ORDER 取較不利者合併（歷史無 REAL_CALIB、無作廢 → C1＝C2，回溯／只往後兩模式相同），套到 V7X 可執行歷史線，對照 G8P-2a 自報 HIST_DEMO 與 G8P-2 HIST_DEMO。'),
    B_mix=('不做。理由：(i) 兩題都沒有可當第二條腿的報酬序列（PIT 遮罩版主策略與 C0 只差 4～5 個槽位，當第二條腿等於自己混自己）；'
           '(ii) 主策略×0050 固定比例混合在本輪「不再開」清單；(iii) integrate_o14 已事前登記並跑過 w＝0.8／0.7／0.5／0.3，全部「不可用」。FINDINGS 直接引用 o14 數字。'),
    legs=dict(
        C0='main_v2 v2_core V2(gapfix, exfin=True) plan(10, official)（經 FM flib.VM，遮罩＝C0），ia_core.simulate；必須逐日重現 main_v2/navs_gapfix.parquet EXFIN|n10|{normal,stress}（|差| < 1e-9，否則停）。',
        F=('fclass.Classifier（ERRATUM_SPEC_FM.json 凍結的關鍵字、排除清單＝空）；F1(d)＝f1_value(f1(c,d), C0)；F2＝靜態 f2；F3＝F1∪F2；只在 2006-01-01～2013-12-31 的訊號日替換。'
           '核對：8 個名單 2006～2013 逐日等於 FM_COUNTS.lists；2014 起全部等於 C0；|名單△|/2 槽位數等於 FM blocks（C0|F1 5、C0|F2 4、U_opt|F1 12、U_opt|F2 14）；U_opt 放回槽位清單等於 ERRATUM2（162 筆）。不符 → 停。'),
        engines=('四口徑：freeze 一般（正式引擎，主口徑）、freeze 壓力、dzero 一般（凍結且之後再無成交的持股歸零，下市最壞）、nofreeze 一般（換股日無成交者以最後收盤出清，近似收購／換股對價）。'
                 '依 o15 方法教訓，共用引擎碰到下市股要同時報凍結與對價兩口徑（5854 合庫、6004 元京證、6012 金鼎證都是合併下市）。'),
        X='V7X 可執行 navs_oos_R4.parquet B10|Jj|K5T|{normal,stress}|{100萬,300萬,1000萬}|GROW，窗＝plans_j7 AB_V7_B10 各 j 訊號日（fslib.window_excess）。決策線＝逐 J 窗 min(一般, 壓力) 超額（pp），有效 J ≥ 7 取中位數；歷史沒有 REAL_CALIB，adj＝0。',
        E='0050 含息 etf_0050.csv（fslib.e50）。'),
    periods='ALL＝全部訊號日窗（2006-04～2026-09 起點，245 窗）；SEG1＝2006～2012；SEG2＝2013～2018；SEG3＝2019 起；SEL＝～2019-04；VAL＝2019-05 起；HOLD＝2024 起；Y0613、Y14ON 描述。窗依訊號日月份歸期。',
    A_filter=dict(
        variants=['C0', 'C0_F1', 'C0_F2', 'C0_F3（描述）', 'U_opt', 'U_opt_F1', 'U_opt_F2', 'U_opt_F3（描述）'],
        stats='各變體 × 四口徑 × 期間：年化（窗幾何）、窗頻與日頻最大回撤、對 0050 窗超額 t、對基準（C0；U_opt_F* 另對 U_opt）窗差平均、t、年化差、相關、拿掉最好 3 個日曆年後 t、2006～2013 逐年。'
              '描述：每個異動槽位，移出股與移入股的訊號日收盤→下一訊號日收盤紙上報酬差。',
        decision=('三級（判定對：C0_F1、C0_F2 對 C0；U_opt_F1、U_opt_F2 對 U_opt；一般成本 freeze 為主口徑）：'
                  'R1 可忽略＝四對都 |ALL 年化差| ≤ 0.3pp、|ALL 對 0050 t 差| ≤ 0.05、SEG1 對 0050 幾何超額不變號（門檻沿用 integrate_o14／o15）；'
                  'R2 方向不變＝不符 R1，但四對在四口徑下 SEG1 超額都不變號；R3 可能翻號＝其他。'
                  '任何一級都不推翻 FM 的事前計數判定（M＝1.73% ≤ 5% → 只加註，G5 第一段不新增項目）；R1／R2 只決定加註文字，R3 只產生「建議下一輪以 PIT 遮罩重算主策略正式數字」。不改 SOP。F3 只描述。')),
    C_protocol=dict(
        rule=('獨立重寫（不 import refimpl／g8alib／g8_evaluate_v2a）。24 個有效月、無作廢路徑：第 12 月 字面 G8（S_12 > 0）→ 下修 z ≤ −c_d → 升級 z ≥ b·√2（升級後待 24 月確認、不再下修）；'
              '第 18 月 下修；第 24 月 下修 → 升級 z ≥ b；A1 與 V1 各自判定後依 ORDER（FAILED < NO_REJ_24 < VALIDATING < PENDING < S1）取較不利者。'),
        lines='V7X 100 萬 GROW 較差成本（決策線）、100 萬一般成本；300 萬、1,000 萬只描述（S1 只適用 100 萬 GROW）。',
        starts='每個錨定月起點一條 24 個月路徑（223 條），高度重疊，只描述；依挑選（2006～2017-04）／驗證（2017-05～2021）／保留（2022 起）起點與 2006～12／2013～18／2019～24 起點年分報。',
        extra='描述：z^A_24 與 z^V_24 分布（最大值、p90、≥ 門檻數）；A1 單獨失敗而 V1 未失敗的起點數；升級所需 24 個月平均月超額（σ_d 分母下，A1 與 V1）。',
        consistency=('一致性（一票否決）：決策線合併狀態的 S1／FAILED／NO_REJ 比例、字面通過比例、三期分報、最後起點狀態、n_v2a_better_than_v2 必須與 G8P-2a HIST_DEMO 完全相同；'
                     'V1 單獨必須與 G8P-2a HIST_DEMO.v2 及 G8P-2 HIST_DEMO.V1_worse 相同。不一致 → C 塊標「不一致，待查」，不下結論。'),
        decision='描述性，不改協定。若決策線合併 S1 比例 < G8P-2a OC 在 α＝+1 的檢定力最低值（OC_FINAL power_min 0.01017），寫「以歷史實際報酬看，G8P-2a 下 24 個月 S1 翻案路徑走不通」。'),
    predictions=('事前預測（寫在看任何變體報酬之前）：'
                 'A：C0_F1（換掉 4119、2002、1319、6196 及 9914，換入 2505、9939、2204、2841×2）與 C0_F2（換掉 6012、4119、2002、1319）全期年化差都在 ±0.15pp、t 差在 ±0.03 內，SEG1 超額差在 ±1pp 內；'
                 'U_opt_F1／F2 主要是把 10 個 5854 合庫槽位換成非金融股，相對 U_opt 年化差在 ±0.3pp 內、方向偏正（銀行股動能與營收成長低於換入股）；'
                 '四口徑結論相同 → 判 R1。若 U_opt_F* 超過 0.3pp，最多到 R2。'
                 'C：獨立重寫與兩份 HIST_DEMO 完全一致；決策線合併 S1 0%、FAILED 65.5%（與 V1 相同，因為 A1 分母較大、下修較少，失敗取聯集後仍由 V1 主導）；z^A_24 最大值遠低於 b_a；300 萬、1,000 萬更差。'),
    honesty=('寫 PREREG 前我已讀過：FM 的 ERRATUM_SPEC_FM、results.json、FM_COUNTS.json（8 個名單、異動槽位清單與代號，未看任何報酬）、flib.py、fclass.py、s4_count.py；'
             'G8P-2a 的 results.json（含 HIST_DEMO：v2a 與 v2 都 S1 0%、FAILED 65.5%）、refimpl.py、s6_hist_demo.py、協定 params；'
             'integrate_o15 的 PREREG、run.py、FINDINGS 與結果（C0 全期 28.28%、t 2.643、SEG1 13.17pp；U_opt 數字）；main_v2 FINDINGS 正式數字。'
             '登記前的煙霧測試（smoke/smoke.out）只核對 C0 重現（差 0，C0 報酬早已公開）、8 個名單與 FM_COUNTS 逐筆相同、2014 起與 C0 相同、槽位數、窗數 245、各歷史線均值與 sd（已在 G8P-2 公開）；'
             '沒有算任何 F 變體或 U_opt 變體的報酬、t 值或協定結果。2006～2026 資料在主策略各研究都已看過，沒有樣本外。'
             '本 PREREG 沒有外部錨定（子代理不得 git commit）；請研究中心 commit 時以 written_utc 與 RUN_UTC 的先後為準。'),
    tests='判定用：A 一條三級規則（4 個判定對 × 4 口徑）；C 描述（一致性一票否決）。',
    sha256={str(f): sha(f) for f in FILES},
)
if __name__ == '__main__':
    RJ = HERE / 'results.json'
    assert not RJ.exists(), 'results.json 已存在：PREREG 只寫一次'
    h = hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    RJ.write_text(json.dumps(dict(PREREG=PREREG, PREREG_sha256=h), ensure_ascii=False, indent=1), encoding='utf-8')
    print(h, PREREG['written_utc'])

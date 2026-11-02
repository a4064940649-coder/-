#!/usr/bin/env python3
"""步驟 3：凍結 PREREG（之後才可以碰驗證期與保留期的報酬）。
選定迭代版本：I3_buffer（j 中位數決策線、設計目標誤升級 4%／檢定力 55%、σ×1.00）。
迭代的機械選版（最低誤下修）會選 I5（官方日代理線，誤下修 0.102 對 0.122），但 PREREG_DESIGN 已把 j 中位數寫成決策線、
o7 教訓 1 規定判定一律用 j 中位數；這裡照設計登記取 I3，並在 iterations.json 記錄這個覆寫（多重檢定揭露）。
用法：python3 -I -B s3_prereg.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fslib as F

CONFIG = dict(version='I3_buffer', target_up=0.04, target_pow=0.55, sig_mult=1.0, alpha_min=0.25, up_P=0.975, down_P=0.80,
              decision_line='jmed', main_cell='100萬|FIX', cost='normal', policy_x0=0.30, H_c=36)
FILES = ['fslib.py', 'cvlib.py', 'seq2.py', 'prior.py', 's2b_cv_sel.py', 's4_cv_final.py', 's5_oc_final.py', 's6_pseudo_forward.py',
         's7_alloc.py', 's8_robust.py', 'vendor_o7/tlib.py', 'vendor_o7/hmethods.py', 'vendor_o7/seqlib.py']

TEXT = """凍結規格（o_forward_sequential_v2）
一、決策線：主策略 B10 可執行 K5T（V7 navs_exec7；V7X 交件後只換輸入）一般成本，13 個 j（J0＝D_m、J_j＝D_m 後第 j 個交易日）各自
    「訊號日收盤→下一訊號日收盤」窗超額（對 0050 含息 pp），錨定月對齊後逐月取中位數（≥7 個 j）。主格 100 萬 FIX；其他 7 格同法另報。
二、CV（最終）：族群 2,319 條 × V4 格點 245 窗；m_c ≥ 5 的 31 群；切分＝時間切半 122／123 兩方向＋擴張估計窗滾動起點 {84,120,157,192}×48 窗；
    tag=sel（估計期群內 α̂ 最大者）。版本 V1～V7（hmethods.py）。選版＝sel 合併 logpd 最大；差 < 1 nat 同分取主策略 B10 事後均值低者。
    s＝選定版本在 half、roll 兩方案各自合併 OLS 斜率（y 對事後均值）的較小者，截 [0,1]；k＝1.0 起每 0.1 第一個使 sel 80% 涵蓋 ≥ 0.80 者。
三、先驗：選定版本、全期族群擬合、焦點＝決策線對齊族群窗（缺 0）、群 114 → θ'＝s·μ＋k(θ−μ)。
四、σ_design＝max(HAC_L6 長期 sd 全期, 同 2019-05 起) × 1.00。
五、規則族 R1～R4（seq2.py），第 12 個月前不動作、先觸發吸收：
    R1 6 個月一次：升級 P(α>0.25)≥0.975、下修 P(α<0)≥0.80（常態概似、σ_design）。
    R2 6 個月一次：升級 z_n ≥ c_u·√(36/n)；下修 z_n ≤ −c_d。
    R3 CUSUM（k_ref −0.5pp）C_t ≤ −h 下修（t≥12）＋R1 升級。
    R4 12、24、36、48、60 月：升級 z ≥ c_u4；下修 z ≤ −c_d4。
    校準：iid 常態 200,000 路徑、seed 20261102、120 個月，分位數精確解：α＝0 時 36 月內誤升級 4%（邊際）、α＝−1 時 36 月內下修 55%（邊際）。
六、OC：D1 iid 常態 σ_design；D2 決策線 2019-05 起殘差 6 個月環狀區塊 bootstrap（不縮放）；D3 t4（單位變異）× 狀態乘數 0.7／1.4
    （高波動穩態 0.347；高→低 0.06、低→高 0.0319）× σ_design。α ∈ {−1,−0.5,0,0.5,1}，每 DGP×α 40,000 路徑、60 個月、seed 20261105。
    凍結條件：三 DGP 都要 α＝−1 時 36 月內下修 ≥ 50%、α＝0 時 36 月內誤升級 ≤ 5%。通過者取三 DGP 最大誤下修最低者；同分取檢視次數少者。
    選定規則誤下修 > 30% 或無規則通過 → H_c ∈ {42,48,54,60,72,84,96,120} 依序重校準（R2、R4，20,000 路徑、120 個月），第一個「三 DGP
    檢定力 ≥ 50%、誤升級 ≤ 5%、誤下修 ≤ 30%」者為最短可行期限。不論是否需要都報告。
七、refit：前瞻 ≥ 12 個月且前瞻 sd > 1.5σ_design → σ ← 前瞻 sd，重算（同程序）；升級的 S_n 邊界只能變嚴。唯一允許的 refit。
八、偽前瞻（驗證期 2019-05 起、保留期 2024-01 起）：先驗分別用 CV_SEL＋SEL 窗、CV_PRE2024＋<2024 窗；σ 分別為 SEL HAC、
    max(<2024 HAC, 2019-05～2023-12 HAC)；規則＝第六節選定者。政策 P30（起始 30%、升級→100%、下修→0%）、P100（起始 100%、下修→0%），
    決策從第 n＋2 月生效。對手：0050、同池等權、主策略 j 中位數可執行（一般／壓力）、主策略官方日零衝擊。
九、配置格：K5T、4 級距×FIX／GROW×一般／壓力；CV_FINAL 先驗套各格 j 中位數；成對平穩 bootstrap L6、240 月、20,000 路徑、seed 20261106；
    x 0～100%，x*＝argmax E[log W]／年，平坦區間＝差 ≤ 0.25%／年。
十、判定：規則「可用」＝第六節三 DGP 全過、複製員 OC 差 ±2pp 以內、make_forward 乾跑通過（2026-10-12 逐檔重現 SOP、2026-09 對 V7 名單）；
    否則不可用，並寫明需要多少個月。研究員判定上限：方法「可用」；任何策略含義「待觀察」。
"""

if __name__ == '__main__':
    F.check_prereg('PREREG_DESIGN', files=False)
    res = F.load_res()
    assert 'CV_SEL' in res and 'ITER' in res
    obj = dict(config=CONFIG, text=TEXT, design_sha256=res['PREREG_DESIGN_sha256'],
               iteration_choice=dict(mechanical=res['ITER']['chosen'], frozen='I3_buffer',
                                     reason='PREREG_DESIGN 決策線＝j 中位數；o7 教訓 1；機械選版差 0.017 誤下修，屬同一量級'),
               file_hashes={f: F.sha(F.HERE / f) for f in FILES}, utc=F.now_utc())
    h = F.register('PREREG', obj)
    it = json.loads((F.HERE / 'iterations.json').read_text(encoding='utf-8'))
    it['frozen'] = dict(version='I3_buffer', mechanical_choice=res['ITER']['chosen'], override_reason=obj['iteration_choice']['reason'], prereg_sha256=h)
    (F.HERE / 'iterations.json').write_text(json.dumps(it, ensure_ascii=False, indent=1, default=F._js), encoding='utf-8')
    print('PREREG', h)

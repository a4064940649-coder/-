#!/usr/bin/env python3
"""數據稽核部 2026-11-21：把 patches_risk.json（make_risk_rows.py 產生，build_site_data.py 的風險表）加上其他檔的修改建議，合成 patches.json。
每筆：檔案（相對倉庫根目錄）、原文片段（在檔案中剛好出現一次；本程式逐筆檢查，並在記憶體中試套用）、建議改成、理由。不改任何既有檔。
python3 -I -B make_patches.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/user/-')
R = json.loads((HERE / 'recalc_results.json').read_text(encoding='utf-8'))
B = R['v2_p4']
S75 = json.loads((HERE / 's75' / 'res_v2.json').read_text(encoding='utf-8'))['versions']['V_BRIDGE']
f = lambda x, nd=1: ('−' if x < 0 else '') + f'{abs(x):.{nd}f}'
P = json.loads((HERE / 'patches_risk.json').read_text(encoding='utf-8'))
s42, s27 = B['S42'], B['S27']
A27 = R['v1_p4']['S27']
EX = [
    dict(檔案='research/tw_stock_selection/agents/s75_continuous_2008_2026/chart_template.html',
         原文片段='策略全期每年的中位數是 22.8%（21.8%～24.3%）',
         建議改成=f"策略全期每年的中位數是 {S75['paths20']['cagr_full_median']*100:.1f}%（{S75['paths20']['cagr_full_min']*100:.1f}%～{S75['paths20']['cagr_full_max']*100:.1f}%）",
         理由='2008～2026 利潤圖樣板裡唯一寫死、新口徑下會變的數字（20 條路徑全期年化中位數 22.83%→22.87%，最小最大四捨五入後不變）；其餘數字由 make_chart.py 從淨值檔帶入。圖下方其他提醒文字不動。'),
    dict(檔案='xq_scripts/README.md',
         原文片段='下面各項打折是以修正前的 48.76% 為基準算的，沒有隨 2026-11-21 改版重算，差異預期 ≤0.1 個百分點。',
         建議改成=(f"下面表格是以修正前的 48.76% 為基準算的；2026-11-21 數據稽核部已用新口徑重算（`research/tw_stock_selection/agents/audit_v2_recalc/`，同一支程式先逐位重現表上的數字）："
                 f"樂觀 {f(s42['optimistic'])}%（回撤 {f(s42['optimistic_maxdd'])}%）、中性 {f(s42['neutral'])}%（回撤 {f(s42['neutral_maxdd'])}%，壞時 {f(s42['neutral_maxdd_p10'])}%）、"
                 f"打一半 {f(s42['half'])}%（回撤 {f(s42['half_maxdd'])}%）、保守 {f(s42['conservative'])}%（回撤 {f(s42['conservative_maxdd'])}%）、0050 同期 {f(s42['bench_since2019'])}%（回撤 {f(s42['bench_maxdd'])}%），和表上最多差 0.2 個百分點。"),
         理由='README 第 31 節 S42 表前的註記原本寫「沒有隨改版重算」；現在已重算，把新數字寫在註記裡（表格本身是修正前的紀錄，不動）。'),
    dict(檔案='research/tw_stock_selection/agents/eng_switch_v2_2026-11-21/FINDINGS.md',
         原文片段='原檔沒動，未經稽核複製）',
         建議改成='原檔沒動；2026-11-21 數據稽核部已用 S75 原引擎獨立重跑：淨值逐日差 ≤0.01 元、交易逐筆相同，見 agents/audit_v2_recalc/s75/）',
         理由='S75 新口徑已由稽核獨立核對，狀態更新。'),
    dict(檔案='research/tw_stock_selection/agents/eng_switch_v2_2026-11-21/FINDINGS.md',
         原文片段='- 風險表由 S26、S27、S42、S46、S62、S64、R0 推出的 20 列未重算（已加註）；兩列回撤註「待重跑」。',
         建議改成='- 風險表由 S26、S27、S42、S39、S46、S62、S64 推出的各列：2026-11-21 數據稽核部已用新口徑重算（agents/audit_v2_recalc/risk_rows_new.json；舊口徑先逐位重現），build_site_data.py 的補丁在同資料夾 patches.json，套用後要重建並重傳網站 risk 資料集。R0 那一列（0050 也加滑價）是清潔室引擎的敏感度，無法重算，沿用舊數字。',
         理由='「還沒做」第一項的狀態更新。'),
    dict(檔案='research/tw_stock_selection/agents/eng_switch_v2_2026-11-21/FINDINGS.md',
         原文片段='門檻與誤報率沒重算。',
         建議改成=(f"門檻與誤報率 2026-11-21 由數據稽核部重跑 S27：門檻不變（−8／−26／月底 −29.9%／−24%），策略有效時 5 年誤報 A {f(A27['false_alarm_5y']['A'], 2)}→{f(s27['false_alarm_5y']['A'], 2)}%、"
                 f"B {f(A27['false_alarm_5y']['B'], 2)}→{f(s27['false_alarm_5y']['B'], 2)}%、C {f(A27['false_alarm_5y']['C'], 2)}→{f(s27['false_alarm_5y']['C'], 2)}%、E {f(A27['false_alarm_5y']['E'], 2)}→{f(s27['false_alarm_5y']['E'], 2)}%，"
                 f"兩條以上（紅燈）{f(A27['red_false_alarm_5y'], 2)}→{f(s27['red_false_alarm_5y'], 2)}%（網站「約 7%」要改成「約 6%」）。"),
         理由='「還沒做」第二項的狀態更新。'),
]
s7, s9, s46, s62, s64 = B['S7'], B['S9'], B['S46'], B['S62'], B['S64']
A_ = '| 每月都準時下單的可預期 | 46.7% | **46.6%** |'
B_ = '差的情況 **35.9%**（網站風險表「參數挑選後可預期」「參數挑選_差的情況」）。'
C_ = '- 完全照做的 20 條中位數是 46.4%（回撤 −33.4%）。'
EX += [
    dict(檔案='xq_scripts/README.md', 原文片段=A_,
         建議改成=A_ + (f"\n\n2026-11-21 新口徑（再加減資復牌修正與 0050 官方含息序列）由數據稽核部重算（`research/tw_stock_selection/agents/audit_v2_recalc/`）：晚 1 天少 {f(s7['late1_cost_5cap_median_pp'])}、晚 2 天少 {f(s7['late2_cost_5cap_median_pp'])} 個百分點、"
                       f"晚 2 天回撤 {f(s7['late2_maxdd'])}%；有時晚 0～4 天中位數／差的 10%／回撤中位數 {f(s9['late_median'])}%／{f(s9['late_p10'])}%／{f(s9['late_dd_median'])}%；100 條回撤差的 10% {f(s9['late_dd_p10'])}%；準時可預期 {f(s9['ontime_intercept'])}%。上表是 2026-11-17 的紀錄，不改。"),
         理由='README 晚下單表（2026-11-17 版）後面補一句新口徑的數字；網站風險表這幾列改成新數字後，文件要對得起來（晚 2 天 6.2→6.3）。表格本身是歷史紀錄，不動。'),
    dict(檔案='xq_scripts/README.md', 原文片段=B_,
         建議改成=B_ + f"2026-11-21 新口徑由數據稽核部重算：仍排第 {s46['current_rank']}、比家族中位數高 {f(s46['current_minus_median_pp'], 2)} 個百分點，900 條路徑中位數 {f(s46['all_paths_median'])}%、差的情況 {f(s46['all_paths_p10'])}%（網站已改用這組）。",
         理由='README S46 段落補新口徑數字（40.4→40.5、35.9→36.0、6.25→6.19），和網站風險表一致。'),
    dict(檔案='xq_scripts/README.md', 原文片段=C_,
         建議改成=C_ + (f"（這一段與上表是修正前的數字。2026-11-21 新口徑由數據稽核部重算：照做 {f(s64['baseline_median'])}%（回撤 {f(s64['baseline_maxdd'])}%）；逐條相減的中位數——只買一半 {f(-s64['PICK_RAND']['d19_pp'])}、回落 20% 停手 {f(-s64['STOP_20_6M']['d19_pp'])}、"
                       f"漏四分之一 {f(-s64['SKIP_25']['d19_pp'])}（漏一成 {f(-s64['SKIP_10']['d19_pp'])}、漏一半 {f(-s64['SKIP_50']['d19_pp'])}）、開盤漲 3% 不買 {f(-s64['GAP_3']['d19_pp'])}（5% 才不買 {f(-s64['GAP_5']['d19_pp'])}）；"
                       f"10 萬零股 {f(s62['B_基準|10萬'])}%、比 0050（{f(s62['bench_since2019'])}%）多 {f(s62['B_基準|10萬|excess_pp'])}；10 萬只買整張 {f(s62['D_只買整張|10萬'])}%、多 {f(s62['D_只買整張|10萬|excess_pp'])}。網站風險表已改用這組；組合情境與稅沒有另外列。）"),
         理由='README 散戶照做須知（S64／S62）補新口徑數字（6.0→6.1、4.8→4.9、1.6→1.7、14.7→14.8；開盤漲 5% 才不買 0.2→0.4），和網站風險表一致。'),
]
ALL = P + EX
txt = {}
for p in ALL:
    fp = ROOT / p['檔案']
    if p['檔案'] not in txt:
        txt[p['檔案']] = fp.read_text(encoding='utf-8')
    assert txt[p['檔案']].count(p['原文片段']) == 1, (p['檔案'], p['原文片段'][:50], txt[p['檔案']].count(p['原文片段']))
    assert p['原文片段'] != p['建議改成']
    txt[p['檔案']] = txt[p['檔案']].replace(p['原文片段'], p['建議改成'])
# 原文片段彼此不得互相包含（套用順序無關）
for i, a in enumerate(ALL):
    for j, b in enumerate(ALL):
        if i != j and a['檔案'] == b['檔案']:
            assert a['原文片段'] not in b['原文片段'] and a['原文片段'] not in b['建議改成'], (i, j)
json.dump(ALL, open(HERE / 'patches.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
import collections
print(len(ALL), dict(collections.Counter(p['檔案'] for p in ALL)))

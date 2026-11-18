# S69：由 results.json 產生白話判讀表與事前預期對照（只讀 results.json；不重算）
# 用法：python3 -I -B make_plain_table.py
import json
from pathlib import Path
H = Path(__file__).resolve().parent
R = json.loads((H / 'results.json').read_text(encoding='utf-8'))
B = {b['look']: b for b in R['boundaries']}
P = R['power']['D1_main']['table']; S = R['sequential_sim']['D1_main']['by_delta']
pc = lambda v: f'{100 * v:.0f}%'
em = lambda k: f"{B[k]['efficacy']['m_pp_at_te_main']:+.0f}"
fm = lambda k: f"{B[k]['futility']['m_pp_at_te_main']:+.0f}"
et = lambda k: f"{B[k]['efficacy']['t']:.2f}"
ftt = lambda k: f"{B[k]['futility']['t3']:.2f}"
rng = lambda y: f"{P[str(y)]['0']['obs_p05_pp']:+.0f}～{P[str(y)]['0']['obs_p95_pp']:+.0f}"
rows = [
 dict(when='第 1～2 年（2027-10、2028-10）', see=f"幾乎任何數字。策略其實沒用時，1 年累積年化超額有九成落在 {rng(1)} 個百分點、2 年 {rng(2)}；1 年有 {pc(P['1']['0']['p_obs_ge5'])} 機率看起來 ≥ +5。策略真有 +12 時，1 年仍有 {pc(P['1']['12']['p_obs_lt0'])} 機率是負的。",
      means='幾乎不代表什麼（運氣遠大於實力）', action=f"續做；不因好看加碼、不因難看停用。例外：t ≥ {et(1)}／{et(2)}（約年化 {em(1)}／{em(2)} 以上，實際上不會發生）才算有效；t3 ≤ {ftt(1)}／{ftt(2)}（約年化 {fm(1)}／{fm(2)} 以下）正式判定無效、停用改 0050。"),
 dict(when='第 3～4 年（2029-10、2030-10）', see=f"A：累積年化超額約 ≥ {em(3)}（第 3 年，t ≥ {et(3)}）／{em(4)}（第 4 年，t ≥ {et(4)}）。B：約 ≤ {fm(3)}／{fm(4)}（t3 ≤ {ftt(3)}／{ftt(4)}）。C：介於中間。",
      means='A 判定有效（極少見）；B 顯著低於「每年 +3」＝正式無效；C 還看不出來', action='A 續做。B 停用改 0050。C：≥ +3 續做；0～+3 續做、降低期望；< 0 續做、降低期望，並看 S27 警戒線 A／E 是否觸發。'),
 dict(when='第 5 年（2031-10）', see=f"A：≥ 約 {em(5)}（t ≥ {et(5)}）。B：累積年化超額 < 0。C：0～+3。D：+3 以上但未達 A。",
      means=f"A 判定有效。B 滿 5 年還輸 0050：沒用時有 {pc(P['5']['0']['p_obs_lt0'])} 會這樣，真有 +5 時 {pc(P['5']['5']['p_obs_lt0'])}、真有 +12 時 {pc(P['5']['12']['p_obs_lt0'])}（不是統計顯著，但不值得再花工夫）。C、D 還看不出來。",
      action='A 續做。B 建議停用改 0050（紙上紀錄照做，之後越過有效界線可恢復）。C 續做、降低期望。D 續做。'),
 dict(when='第 6～9 年（2032-10～2035-10）', see=f"A：≥ 約 {em(6)}／{em(7)}／{em(8)}／{em(9)}（t ≥ {et(6)}／{et(7)}／{et(8)}／{et(9)}）。B：累積年化超額 < 0。C：其餘。",
      means='A 判定有效。B 務實停用線。C 還看不出來', action='A 續做。B 建議停用改 0050。C：≥ +3 續做；0～+3 續做、降低期望。'),
 dict(when='第 10 年（2036-10，最後一次）', see=f"A：≥ 約 {em(10)}（t ≥ {et(10)}）。B：+3 以上但未達 A。C：< +3。",
      means='A 判定有效。B 十年仍未證實、點估計尚可。C 十年無法證實且點估計不足', action='A 續做。B 擁有者決定（當成「可能略好於 0050、但沒被證明」）。C 停用改 0050。'),
]
oc = {d: dict(efficacy=S[d]['p_efficacy_first'], stop_total=S[d]['p_any_stop_incl_terminal'], futility_formal=S[d]['p_futility_formal_first'], pragmatic_stop=S[d]['p_pragmatic_stop_first'],
              terminal_stop=S[d]['p_terminal_stop_m_lt3'], owner_decides_at_10=S[d]['p_terminal_owner_decides_m_ge3'], decision_year_median=S[d]['decision_year_median'],
              efficacy_year_median=S[d]['efficacy_year_median_given_efficacy']) for d in S}
E = json.loads((H / 'PREREG.json').read_text(encoding='utf-8'))['pre_stated_expectations']
chk = dict(
 E1=dict(expected=E['E1'], actual=dict(power_d5_10y=P['10']['5']['p_sig_1sided5'], power_d12_10y=P['10']['12']['p_sig_1sided5'], luck_ge5_1y=P['1']['0']['p_obs_ge5'], luck_ge5_10y=P['10']['0']['p_obs_ge5'],
                                       normal_power_d5_10y=R['power']['D5_normal']['table']['10']['5']['p_sig_1sided5']), verdict='部分不符：D1 下 +5 的 10 年檢定力 17%，低於預期 25～35%（常態假設才有 25%）；運氣機率略低於預期'),
 E2=dict(expected=E['E2'], actual=dict(sim_size=R['size_contingency']['sim_size_efficacy_D1_after'], eff_year_median_d12=S['12']['efficacy_year_median_given_efficacy']), verdict='不符：模擬型一誤差 1.1%，比預期 4～5.5% 保守得多（正偏態＋負自相關）；判定年份符合'),
 E3=dict(expected=E['E3'], actual=dict(fut_formal_ever_d0=S['0']['p_futility_formal_ever'], prag_first_d5=S['5']['p_pragmatic_stop_first'], size_fut_before=R['size_contingency']['sim_size_futility_D1_before']), verdict='符合；但無效界線原始誤差 21% > 11%，依事前條款把界線乘 1.328'),
 E4=dict(expected=E['E4'], actual=dict(D2=R['power']['D2_te56_scaled']['table']['10']['5']['p_sig_1sided5'], D3=R['power']['D3_s55_resid']['table']['10']['5']['p_sig_1sided5']), verdict='符合'),
 E5=dict(expected=E['E5'], actual=R['bayes']['main|te_main']['table']['5']['-5']['p_gt0'], verdict='符合'))
out = dict(title='S69 白話判讀表（門檻的年化數字以追蹤誤差 16.2 個百分點換算，實際判定以 t／t3 為準）', start='2026-11-12 起算；界線凍結於 PLAN_FROZEN.json，不得更改，要改必須另題登記並保留舊版',
           plan_frozen_sha256=R['plan_frozen_sha256'], rows=rows, outcome_probabilities_D1_main=oc, expectations_check=chk)
(H / 'plain_table.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
for r in rows: print(r['when'], '｜', r['see'], '｜', r['means'], '｜', r['action'])
print(json.dumps(oc, ensure_ascii=False))

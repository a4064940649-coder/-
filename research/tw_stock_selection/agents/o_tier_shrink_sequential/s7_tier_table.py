#!/usr/bin/env python3
"""步驟 3b：配置表 tier_table.json（由 RESULTS 機械產生，不再計算任何新數字）＋一頁使用者說明。
用法：python3 -I -B s7_tier_table.py [--key RESULTS]
"""
import sys, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tlib as T

TIERS = ['30萬', '100萬', '300萬', '1000萬']; MODES = ['FIX', 'GROW']
MODE_ZH = {'FIX': '獲利不滾入（每月把帳戶調回原金額）', 'GROW': '獲利滾入'}

def build(R):
    rows = []
    for t in TIERS:
        for m in MODES:
            for cost in ('normal', 'stress'):
                k = f'EXEC|{t}|{m}|{cost}'
                tr = R['tier_rule'][k]
                cK, cT = R['cells_posterior'][k + '|KAPPA'], R['cells_posterior'][k + '|TICK']
                aK, aT = R['allocation'][k + '|KAPPA'], R['allocation'][k + '|TICK']
                def at(a, H, x): return a[H]['grid'][str(x)] if str(x) in a[H]['grid'] else a[H]['grid'][x]
                xs = min(tr['xstar_KAPPA_20y'], tr['xstar_TICK_20y'])
                rows.append({'tier': t, 'mode': m, 'cost': cost,
                             'x_star_write': tr['write'], 'x_star_is_number': tr['is_number'],
                             'x_star_KAPPA_20y': tr['xstar_KAPPA_20y'], 'x_star_TICK_20y': tr['xstar_TICK_20y'],
                             'x_star_KAPPA_10y': tr['xstar_KAPPA_10y'], 'x_star_TICK_10y': tr['xstar_TICK_10y'], 'diff_pp': tr['diff_pp'],
                             'post_mean_pp_KAPPA': cK['mean'], 'post_mean_pp_TICK': cT['mean'], 'P_pos_KAPPA': cK['p_pos'], 'P_pos_TICK': cT['p_pos'],
                             'q10_q90_KAPPA': [cK['q10'], cK['q90']], 'q10_q90_TICK': [cT['q10'], cT['q90']],
                             'win10_x100': [at(aK, '10y', 100)['P_win'], at(aT, '10y', 100)['P_win']],
                             'win20_x100': [at(aK, '20y', 100)['P_win'], at(aT, '20y', 100)['P_win']],
                             'Plt075_10y_x100': [at(aK, '10y', 100)['P_ratio_lt_0.75'], at(aT, '10y', 100)['P_ratio_lt_0.75']],
                             'Plt075_20y_x100': [at(aK, '20y', 100)['P_ratio_lt_0.75'], at(aT, '20y', 100)['P_ratio_lt_0.75']],
                             'at_xstar_written': {'x': xs, 'win10': [at(aK, '10y', xs)['P_win'], at(aT, '10y', xs)['P_win']],
                                                  'win20': [at(aK, '20y', xs)['P_win'], at(aT, '20y', xs)['P_win']],
                                                  'Plt075_10y': [at(aK, '10y', xs)['P_ratio_lt_0.75'], at(aT, '10y', xs)['P_ratio_lt_0.75']]},
                             'order_note': '成對數字依序為 [KAPPA（o5 κ 衝擊）, TICK（每邊一跳）]'})
    ref = {}
    for nm in ('AB', 'AB_J', 'B10', 'B10_J', 'AB_stress_J', 'B10_stress_J'):
        c = R['cells_posterior'][nm]; a = R['allocation'][nm]
        ref[nm] = {'post_mean_pp': c['mean'], 'P_pos': c['p_pos'], 'post_over_raw': c.get('post_over_raw'), 'x_star_20y': a['20y']['xstar'],
                   'x_star_10y': a['10y']['xstar'], 'win10_x100': a['10y']['grid']['100' if '100' in a['10y']['grid'] else 100]['P_win']}
    return rows, ref

def user_note(rows, R):
    def row(t, m, c): return next(r for r in rows if r['tier'] == t and r['mode'] == m and r['cost'] == c)
    lines = ['主策略放幾成？（其餘放 0050；研究中心 o_tier_shrink_sequential，2026-10-28；只改配置註記，不改選股）', '']
    for t in TIERS:
        for m in MODES:
            n, s = row(t, m, 'normal'), row(t, m, 'stress')
            lines.append(f'- {t}、{MODE_ZH[m]}：一般成本 {n["x_star_write"]}；成本偏高時 {s["x_star_write"]}。'
                         f' 全押主策略 10 年贏 0050 的機率約 {min(n["win10_x100"]):.0%}～{max(n["win10_x100"]):.0%}。')
    lines += ['', '怎麼讀：',
              '- 「數字」表示兩種交易成本模型的答案差不到 30 個百分點，寫的是較保守的那個；「範圍」表示成本模型還沒校準，答案取決於你實際的成交成本。',
              '- 這張表的意思是「長期複利期望最高的比例」，不是「保證贏 0050 的比例」。主策略近 8 年和 0050 統計上分不出高下。',
              '- 不管放多少，中途某段 5 年輸 0050 都很常見；比例請依你能承受的落後幅度往下調，不要往上調。',
              '- 2026-11 起的實際成交會每月更新這張表的前提（前瞻序貫規則），規則觸發「下修」時改用「成本偏高」那一欄。']
    if R['sop_conditional_0050_primary']['all_GROW_P_lt_0.5']:
        lines.append('- 條件式註記：大資金、獲利滾入時，預設配置應以 0050 為主。')
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--key', default='RESULTS'); a = ap.parse_args()
    T.check_block('PREREG')
    res = T.load_res(); R = res[a.key]
    rows, ref = build(R)
    tt = {'source_key': a.key, 'selected_version': R['selected_version'], 'rule': '兩衝擊模型 x*（20 年對數財富最大）差 ≤ 30pp 寫較小值，否則寫範圍',
          'rows': rows, 'reference_cells': ref, 'sop_conditional_0050_primary': R['sop_conditional_0050_primary'],
          'user_note_zh': user_note(rows, R), 'utc': T.now_utc()}
    out = HERE / ('tier_table.json' if a.key == 'RESULTS' else f'tier_table_{a.key.lower()}.json')
    out.write_text(json.dumps(tt, ensure_ascii=False, indent=1, default=T._js), encoding='utf-8')
    print(tt['user_note_zh'])
    for r in rows:
        print(r['tier'], r['mode'], r['cost'], r['x_star_write'], r['x_star_KAPPA_20y'], r['x_star_TICK_20y'], [round(x, 3) for x in (r['P_pos_KAPPA'], r['P_pos_TICK'])],
              [round(x, 3) for x in r['win10_x100']])

if __name__ == '__main__':
    main()

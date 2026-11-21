#!/usr/bin/env python3
"""數據稽核部 2026-11-21：由 recalc_results.json（extract.py 的輸出）產生
  risk_rows_new.json（網站風險表各列：舊值、新值、差、是否重現舊值、重算方式、要改的說明文字）
  patches.json（對 xq_scripts/site/build_site_data.py 與其他檔的修改建議；每筆「原文片段」在檔案中剛好出現一次；本程式會在暫存副本上試套用並檢查）。
不改任何既有檔。python3 -I -B make_risk_rows.py"""
import json, re, ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
BSD = Path('/home/user/-/xq_scripts/site/build_site_data.py')
R = json.loads((HERE / 'recalc_results.json').read_text(encoding='utf-8'))
A, B = R['v1_p4'], R['v2_p4']                     # 舊（改版前三個開關）、新（五個開關），百分比 4 位小數
ACC = R['acceptance']
src = BSD.read_text(encoding='utf-8')
i0, i1 = src.index('RISK = ['), src.index('dump("risk"')
ROWS = {}
for line in src[i0:i1].split('\n'):
    m = re.match(r'^\s+\("([^"]+)", ([^,]+), "([^"]*)", "(.*)", "([^"]*)"\),$', line)
    if m:
        ROWS[m.group(1)] = dict(line=line.strip(), value=m.group(2), unit=m.group(3), desc=m.group(4), source=m.group(5))
assert len(ROWS) == 36


def f(x, nd=1, sign=False):
    """數字 → 網站文字用的字串（負號用 −；sign=True 時正數加 +）。"""
    s = f'{abs(x):.{nd}f}'
    if float(s) == 0:
        return s
    return ('−' if x < 0 else ('+' if sign else '')) + s


def pyv(x, nd=1):
    v = round(x, nd)
    return repr(int(v)) if nd == 0 else repr(v)


N1 = '（基準為修正前 48.76，未隨 2026-11-21 改版重算，差異預期 ≤0.1）'
N1B = '（基準為修正前 48.76，未隨 2026-11-21 改版重算，差異預期 ≤0.1；0050 改用官方含息序列後 2019 起是 28.82%，這裡的 28.9% 是舊序列）'
N2 = '（基準為修正前（單一路徑回撤 −32.1 那一版），未隨 2026-11-21 改版重算；這次單一路徑回撤變淺 0.5、20 條中位數變淺 0.3 個百分點，本列可能跟著略變淺，待重跑）'
N3 = '警戒線與誤報率出自 S27 模擬，基準為修正前 48.76，未隨改版重算'
SRC_ADD = '；2026-11-21 新口徑重算：agents/audit_v2_recalc/recalc_results.json（數據稽核部，舊口徑先逐位重現）'
HOW_RESYNC = '複製 official_nav/resync/harness.py（原研究程式不改、在暫存區執行）：舊口徑＝三個開關、舊 0050 序列；新口徑＝再開 REDUCE_FIX 與 0050 官方含息序列。'
HOW_SELF = '原研究程式自己開三個開關；在暫存區原樣執行（舊口徑）與只預先加開 REDUCE_FIX、0050 官方含息序列後執行（新口徑＝五個開關）。'
ok_resync = lambda k: ACC.get(f'{k}_v1_vs_resync_new') == []
ok_self = lambda k: (ACC.get(f'{k}_v1_results_vs_original', {}).get('n_diff') == 0) or (k == 'S64' and ACC.get('S64_v1_scenarios_baseline_bench_vs_original', {}).get('n_diff') == 0)
OUT = []


def row(name, old, new, nd, reproduced, how, reps, new_value=None, note='', extra=None, src_add=SRC_ADD, in_scope=True):
    """reps：[(說明裡的原片段, 新片段)]，每個原片段必須剛好出現一次。"""
    r = ROWS[name]; d = r['desc']
    for a, b in reps:
        assert d.count(a) == 1, (name, a, d.count(a))
        d = d.replace(a, b)
    val = r['value'] if new_value is None else new_value
    s2 = r['source'] + (src_add if src_add and src_add not in r['source'] else '')
    newline = f'("{name}", {val}, "{r["unit"]}", "{d}", "{s2}"),'
    assert '"' not in d and '"' not in s2
    OUT.append(dict(項目=name, 單位=r['unit'], 網站現值=r['value'], 舊值=old, 新值=new, 差=(None if (old is None or new is None) else round(new - old, 2)),
                    網站新值=val, 網站數值有變=bool(val != r['value']), 是否重現舊值=reproduced, 重算方式=how, 備註=note, 其他數字=extra or {},
                    原說明=r['desc'], 新說明=d, 原出處=r['source'], 新出處=s2, 原程式行=r['line'], 新程式行=newline, 屬於待重算的列=in_scope))


s7a, s7b, s9a, s9b, s27a, s27b, s42a, s42b = A['S7'], B['S7'], A['S9'], B['S9'], A['S27'], B['S27'], A['S42'], B['S42']
DONE = lambda old, new, nd=2, more='': f'（2026-11-21 新口徑重算：{f(new, nd)}；修正前 {f(old, nd)}{more}）'
# ---- S9／S7（S26 口徑）
row('回撤_壞的時候', s9a['late_dd_p10'], s9b['late_dd_p10'], 1, ok_resync('S9'), HOW_RESYNC + 'S9 run.py：A10～A15 × 20 條＝100 條路徑最大回撤的差的 10%。',
    [(N2, DONE(s9a['late_dd_p10'], s9b['late_dd_p10'], 2, '，四捨五入後不變' if round(s9a['late_dd_p10'], 1) == round(s9b['late_dd_p10'], 1) else ''))],
    new_value=pyv(s9b['late_dd_p10']), extra=dict(中位數舊=s9a['late_dd_median'], 中位數新=s9b['late_dd_median']))
row('晚1天少', s7a['late1_cost_5cap_median_pp'], s7b['late1_cost_5cap_median_pp'], 1, ok_resync('S7'), HOW_RESYNC + 'S7 run.py＋posthoc.py：W10_70L 晚 1 天，5 種資金中位數。',
    [(N1, DONE(s7a['late1_cost_5cap_median_pp'], s7b['late1_cost_5cap_median_pp']))], new_value=pyv(s7b['late1_cost_5cap_median_pp']),
    extra=dict(一百萬單一路徑舊=s7a['late1_cost_100w_pp'], 一百萬單一路徑新=s7b['late1_cost_100w_pp']))
row('晚2天少', s7a['late2_cost_5cap_median_pp'], s7b['late2_cost_5cap_median_pp'], 1, ok_resync('S7'), HOW_RESYNC + 'S7 run.py＋posthoc.py：W10_70L 晚 2 天，5 種資金中位數。',
    [(N1, DONE(s7a['late2_cost_5cap_median_pp'], s7b['late2_cost_5cap_median_pp']))], new_value=pyv(s7b['late2_cost_5cap_median_pp']),
    extra=dict(一百萬單一路徑舊=s7a['late2_cost_100w_pp'], 一百萬單一路徑新=s7b['late2_cost_100w_pp']))
row('晚2天回撤', s7a['late2_maxdd'], s7b['late2_maxdd'], 1, ok_resync('S7'), HOW_RESYNC + 'S7 run.py：W10_70L 晚 2 天、100 萬單一路徑的最大回撤。',
    [(N2, DONE(s7a['late2_maxdd'], s7b['late2_maxdd'], 2, '，重跑後不變：這條路徑最深的那一段不受這次兩項修正影響'))], new_value=pyv(s7b['late2_maxdd']))
row('準時可預期', s9a['ontime_intercept'], s9b['ontime_intercept'], 1, ok_resync('S9'), HOW_RESYNC + 'S9 run.py：100 條路徑對「平均晚幾天」迴歸的截距。',
    [(N1, DONE(s9a['ontime_intercept'], s9b['ontime_intercept']))], new_value=pyv(s9b['ontime_intercept']))
row('有時晚可預期', s9a['late_median'], s9b['late_median'], 0, ok_resync('S9'), HOW_RESYNC + 'S9 run.py：A10～A15 × 20 條＝100 條路徑 2019 起年化的中位數與差的 10%。',
    [('100 條路徑中位數 42.9；差的 10% 為 40.2；', f'100 條路徑中位數 {f(s9b["late_median"])}；差的 10% 為 {f(s9b["late_p10"])}；2026-11-21 新口徑重算，修正前 {f(s9a["late_median"])}／{f(s9a["late_p10"])}；'),
     ('除權息＋期限修正後，2026-11-17 重算；期限修正前 43.2／40.4）' + N1, '更早只修除權息時 43.2／40.4）')],
    new_value=pyv(s9b['late_median'], 0), extra=dict(差的10pct舊=s9a['late_p10'], 差的10pct新=s9b['late_p10']))
# ---- S27 四條警戒線（現值已於改版當天由正式淨值檔重算；這裡重跑整個 S27：現值、門檻、誤報率）
fa_a, fa_b = s27a['false_alarm_5y'], s27b['false_alarm_5y']
th_same = s27a['thresholds'] == s27b['thresholds']
assert th_same, '門檻變了：說明文字要另外改'
S27HOW = HOW_RESYNC + 'S27 build.py（20 條路徑月底淨值）＋analyze.py（成對區塊重抽 10,000 次，亂數種子固定）：現值、門檻（目標 5 年誤報 ≤10% 的格點）、誤報率全部重跑。'
for nm, key, cur_site in (('警戒A_3年超額', 'A', 'A_now'), ('警戒E_相對前高', 'E', 'E_now'), ('警戒B_1年超額', 'B', 'B_now'), ('警戒C_月底回撤', 'C', 'C_now')):
    assert round(s27b[cur_site], 1) == float(ROWS[nm]['value']), (nm, s27b[cur_site], ROWS[nm]['value'])   # 現值應等於網站已更新的值
    reps = [(N3, f'警戒線門檻與誤報率 2026-11-21 用新口徑重跑 S27 模擬：門檻不變，策略有效時 5 年誤報 {f(fa_b[key], 2)}%（修正前 {f(fa_a[key], 2)}%）')]
    if nm == '警戒B_1年超額':
        reps.append(('（5 年內 97%）', f'（5 年內 {f(s27b["any_12m_behind_5y"], 0)}%）'))
    if nm == '警戒A_3年超額':
        assert f(fa_b['A'], 0) == '8'
    if nm == '警戒E_相對前高':
        assert f(fa_b['E'], 0) == '9'
    row(nm, s27a[cur_site], s27b[cur_site], 1, ok_resync('S27'), S27HOW, reps,
        note='現值與網站已更新的值相同（網站當天已由正式淨值檔重算）；門檻不變；誤報率小降。',
        extra=dict(門檻=s27b['thresholds'][key], 五年誤報舊=fa_a[key], 五年誤報新=fa_b[key], 日內回撤現值新=s27b['C_now_daily'] if key == 'C' else None))
row('紅燈規則', s27a['red_false_alarm_5y'], s27b['red_false_alarm_5y'], 0, ok_resync('S27'), S27HOW,
    [('（策略有效時 5 年誤報約 7%）', f'（策略有效時 5 年誤報約 {f(s27b["red_false_alarm_5y"], 0)}%：2026-11-21 新口徑重跑 S27 模擬 {f(s27b["red_false_alarm_5y"])}%，修正前 {f(s27a["red_false_alarm_5y"])}%）')],
    note='本列原本沒有「未重算」註記，但說明裡的誤報率出自同一個 S27 模擬，四捨五入後 7%→6%，一併更正。數值欄（2 條）不變。', in_scope=False,
    src_add='；誤報率 2026-11-21 新口徑重算：agents/audit_v2_recalc/recalc_results.json')
row('正常範圍_1年', s27a['range1y_median'], s27b['range1y_median'], 0, ok_resync('S27'), S27HOW,
    [('四捨五入不變' + N1, f'四捨五入不變；2026-11-21 新口徑重跑 S27 模擬：中位數 {f(s27b["range1y_median"])}%、P10～P90 {f(s27b["range1y_p10_p90"][0])}%～{f(s27b["range1y_p10_p90"][1])}%、輸 0050 機率 {f(s27b["p_lose_1y"])}%（修正前 {f(s27a["range1y_median"])}%、{f(s27a["range1y_p10_p90"][0])}%～{f(s27a["range1y_p10_p90"][1])}%、{f(s27a["p_lose_1y"])}%），文字不必改')],
    new_value=pyv(s27b['range1y_median'], 0), extra=dict(P10_P90舊=s27a['range1y_p10_p90'], P10_P90新=s27b['range1y_p10_p90'], 輸0050機率舊=s27a['p_lose_1y'], 輸0050機率新=s27b['p_lose_1y']))
# ---- S42
S42HOW = HOW_RESYNC + 'S42 run.py：首次公布營收＋期限前視＋有時晚 0～4 天＋每邊滑價 0.4%／0.5% 的 100 條路徑。'
row('實盤可預期_中性', s42a['neutral'], s42b['neutral'], 1, ok_resync('S42'), S42HOW,
    [('最大回撤約 −35%（壞時 −37.5%）。除權息＋期限修正後，2026-11-17 重算：數字不變（S42 原本就已含期限修正）' + N1,
      f'最大回撤約 {f(s42b["neutral_maxdd"], 0)}%（壞時 {f(s42b["neutral_maxdd_p10"])}%）。2026-11-21 新口徑重算：{f(s42b["neutral"], 2)}%、回撤 {f(s42b["neutral_maxdd"])}%／壞時 {f(s42b["neutral_maxdd_p10"])}%；修正前 {f(s42a["neutral"], 2)}%、{f(s42a["neutral_maxdd"])}%／{f(s42a["neutral_maxdd_p10"])}%')],
    new_value=pyv(s42b['neutral']), extra=dict(回撤中位數舊=s42a['neutral_maxdd'], 回撤中位數新=s42b['neutral_maxdd'], 壞時回撤舊=s42a['neutral_maxdd_p10'], 壞時回撤新=s42b['neutral_maxdd_p10']))
row('實盤可預期_保守', s42a['half'], s42b['half'], 1, ok_resync('S42'), S42HOW,
    [('最差 10% 加滑價 0.5% 約 34.0%、回撤約 −40%。', f'最差 10% 加滑價 0.5% 約 {f(s42b["conservative"])}%、回撤約 {f(s42b["conservative_maxdd"], 0)}%。'),
     ('（除權息＋期限修正後，2026-11-17 重算：數字不變；期限修正前寫 48.99%）' + N1,
      f'（2026-11-21 新口徑重算：{f(s42b["half"], 2)}%、最差 10% {f(s42b["conservative"], 2)}%；修正前 {f(s42a["half"], 2)}%、{f(s42a["conservative"], 2)}%）')],
    new_value=pyv(s42b['half']), extra=dict(最差10pct舊=s42a['conservative'], 最差10pct新=s42b['conservative'], 最差回撤舊=s42a['conservative_maxdd'], 最差回撤新=s42b['conservative_maxdd']))
# ---- S39（營收更正）：原數字是「只修除權息」時算的；五個開關下的效果取自 S42 鏈（同一套 S39 首次公布營收函式）
fa, fb = A['S42FPM'], B['S42FPM']
s39_ok = ok_self('S39')
s39o = A.get('S39', {})
row('營收更正高估', fa['d_window_pp'], fb['d_window_pp'], 1, s39_ok,
    'S39 原程式（只開除權息修正）在暫存區原樣重跑＝重現網站舊值（−0.78／20 條中位數 −0.93）；新口徑的值取自 S42 鏈（同一個 S39 首次公布營收函式，五個開關全開）的標準路徑與 20 條路徑；「舊值」欄＝同一條 S42 鏈在三個開關下的值。',
    [('年化約低 0.8（20 條中位數 −0.93，最多約 −5）', f'年化約低 {f(abs(fb["d_window_pp"]))}（20 條中位數 {f(fb["paths20_window_diff_median_pp"], 2)}，最多約 {f(fb["paths20_window_diff_min_pp"], 0)}）'),
     (N1, f'（2026-11-21 新口徑重算：{f(fb["d_window_pp"], 2)}／中位數 {f(fb["paths20_window_diff_median_pp"], 2)}；只修除權息時 −0.78／−0.93，除權息＋期限修正時 {f(fa["d_window_pp"], 2)}／{f(fa["paths20_window_diff_median_pp"], 2)}）')],
    new_value=pyv(fb['d_window_pp']),
    note='網站舊值 −0.8／−0.93 的口徑是「只修除權息」（S39 原研究）；S39 原程式不讀期限開關，所以新口徑用 S42 鏈的同一步驟。',
    extra=dict(S39原程式重跑_標準路徑=s39o.get('d_window_pp'), S39原程式重跑_20條=s39o.get('paths_pairs_FPM'), 三開關=fa, 五開關=fb,
               S39原程式加開兩項_標準路徑=B.get('S39', {}).get('d_window_pp'), S39原程式加開兩項_20條=B.get('S39', {}).get('paths_pairs_FPM')),
    src_add='；2026-11-21 新口徑：agents/audit_v2_recalc/recalc_results.json（S42 鏈的首次公布營收步驟，數據稽核部）')
# ---- S46
a46, b46 = A['S46'], B['S46']
S46HOW = HOW_SELF + 'S46 run.py：45 格 × 20 條＝900 條路徑。'
assert b46['current_rank'] == 1 and b46['n_all_paths'] == 900
row('參數挑選後可預期', a46['all_paths_median'], b46['all_paths_median'], 1, ok_self('S46'), S46HOW,
    [('高出家族中位數 6.25 個百分點（S46，R46 重現）', f'高出家族中位數 {f(b46["current_minus_median_pp"], 2)} 個百分點（S46；R46 重現的是修正前的 6.25）'),
     ('家族 900 條路徑中位數 40.4%，差的情況 35.9%。', f'家族 900 條路徑中位數 {f(b46["all_paths_median"])}%，差的情況 {f(b46["all_paths_p10"])}%。'),
     (N1, f'（2026-11-21 新口徑重算：{f(b46["all_paths_median"], 2)}%／{f(b46["all_paths_p10"], 2)}%、高出 {f(b46["current_minus_median_pp"], 2)}；修正前 {f(a46["all_paths_median"], 2)}%／{f(a46["all_paths_p10"], 2)}%、高出 {f(a46["current_minus_median_pp"], 2)}）')],
    new_value=pyv(b46['all_paths_median']), extra=dict(現行格名次=b46['current_rank'], 家族中位數舊=a46['family_median'], 家族中位數新=b46['family_median'], 高出舊=a46['current_minus_median_pp'], 高出新=b46['current_minus_median_pp'],
                                                    現行格20條中位數新=b46['current_paths_median'], 現行格單一路徑新=b46['current_main_since2019']))
row('參數挑選_差的情況', a46['all_paths_p10'], b46['all_paths_p10'], 1, ok_self('S46'), S46HOW,
    [(N1, DONE(a46['all_paths_p10'], b46['all_paths_p10'], 2))], new_value=pyv(b46['all_paths_p10']))
# ---- S64
a64, b64 = A.get('S64', {}), B.get('S64', {})
S64HOW = HOW_SELF + 'S64 run.py：20 條路徑、各情境（隨機情境的亂數種子固定）。舊口徑重跑的情境、照做基準、0050 全部逐位相同；整份結果檔只差一個稅務診斷用的計數（除權息封存後來補檔，不影響情境）。'
base_a, base_b = a64.get('baseline_median'), b64.get('baseline_median')
N64 = lambda k: f'（2026-11-21 新口徑重算：逐條相減中位數 {f(-b64[k]["d19_pp"], 2)}、20 條中位數 {f(base_b, 2)}% → {f(b64[k]["since2019"], 2)}%；修正前 {f(-a64[k]["d19_pp"], 2)}、{f(base_a, 2)}% → {f(a64[k]["since2019"], 2)}%）'


def r64(name, k, reps):
    row(name, -a64[k]['d19_pp'], -b64[k]['d19_pp'], 1, ok_self('S64'), S64HOW, reps + [('R64 已重現' + N1, 'R64 重現的是修正前的數字' + N64(k))], new_value=pyv(-b64[k]['d19_pp']),
        extra=dict(情境20條中位數舊=a64[k]['since2019'], 情境20條中位數新=b64[k]['since2019'], 照做20條中位數舊=base_a, 照做20條中位數新=base_b,
                   直接相減舊=round(base_a - a64[k]['since2019'], 4), 直接相減新=round(base_b - b64[k]['since2019'], 4), 回撤逐條差中位數舊=a64[k]['ddd_full_pp'], 回撤逐條差中位數新=b64[k]['ddd_full_pp']))


if 'S64' in B and 'S64' in A:
    k = 'PICK_RAND'
    r64('照做_只買一半少', k, [('每年少賺約 6.0 個百分點（20 條路徑逐條相減後取中位數；20 條中位數本身是 46.4% → 40.1%，直接相減是 6.3，兩種算法不同；獨立重現時換一組隨機抽法是 5.8）',
                                f'每年少賺約 {f(-b64[k]["d19_pp"])} 個百分點（20 條路徑逐條相減後取中位數；20 條中位數本身是 {f(base_b)}% → {f(b64[k]["since2019"])}%，直接相減是 {f(base_b - b64[k]["since2019"])}，兩種算法不同；修正前口徑下獨立重現換一組隨機抽法是 5.8）')])
    k = 'STOP_20_6M'
    r64('照做_回落20%停手少', k, [('每年少賺約 4.8 個百分點（20 條路徑逐條相減後取中位數；20 條中位數本身是 46.4% → 41.5%，直接相減是 4.9，兩種算法不同），而且最大回撤沒有變淺（重現值還略深 0.4 個百分點）。',
                                     f'每年少賺約 {f(-b64[k]["d19_pp"])} 個百分點（20 條路徑逐條相減後取中位數；20 條中位數本身是 {f(base_b)}% → {f(b64[k]["since2019"])}%，直接相減是 {f(base_b - b64[k]["since2019"])}，兩種算法不同），而且最大回撤沒有變淺（逐條差中位數 {f(b64[k]["ddd_full_pp"], 1, True)} 個百分點，負＝更深）。')])
    k = 'SKIP_25'
    r64('照做_漏做四分之一少', k, [('每年少賺約 2.1 個百分點（20 條路徑逐條相減後取中位數；20 條中位數本身是 46.4% → 43.8%，直接相減是 2.6，兩種算法不同）；漏一成約少 0.7，漏一半約少 6.1。',
                                      f'每年少賺約 {f(-b64[k]["d19_pp"])} 個百分點（20 條路徑逐條相減後取中位數；20 條中位數本身是 {f(base_b)}% → {f(b64[k]["since2019"])}%，直接相減是 {f(base_b - b64[k]["since2019"])}，兩種算法不同）；漏一成約少 {f(-b64["SKIP_10"]["d19_pp"])}，漏一半約少 {f(-b64["SKIP_50"]["d19_pp"])}。')])
    k = 'GAP_3'
    r64('照做_開盤漲3%不買少', k, [('每年少賺約 1.6 個百分點（20 條路徑逐條相減後取中位數；20 條中位數本身是 46.4% → 45.6%，直接相減是 0.8，兩種算法不同）；門檻放到 5% 只少約 0.2。',
                                      f'每年少賺約 {f(-b64[k]["d19_pp"])} 個百分點（20 條路徑逐條相減後取中位數；20 條中位數本身是 {f(base_b)}% → {f(b64[k]["since2019"])}%，直接相減是 {f(base_b - b64[k]["since2019"])}，兩種算法不同）；門檻放到 5% 只少約 {f(-b64["GAP_5"]["d19_pp"])}。')])
# ---- S62
a62, b62 = A['S62'], B['S62']
S62HOW = HOW_SELF + 'S62 run.py：36 格摩擦 × 6 個本金 × 20 條路徑；0050 用同一條引擎內的序列（新口徑＝官方含息序列）。'
kB, kD = 'B_基準|10萬', 'D_只買整張|10萬'
N62 = lambda k: f'（2026-11-21 新口徑重算：超額 {f(b62[k + "|excess_pp"], 2)}＝{f(b62[k], 2)}% − 0050 {f(b62["bench_since2019"], 2)}%；修正前 {f(a62[k + "|excess_pp"], 2)}＝{f(a62[k], 2)}% − {f(a62["bench_since2019"], 2)}%（舊 0050 序列））'
row('小資金_10萬零股超額', a62[kB + '|excess_pp'], b62[kB + '|excess_pp'], 1, ok_self('S62'), S62HOW,
    [('每年仍比 0050 多約 14.7 個百分點（43.6% 對 0050 的 28.9%）', f'每年仍比 0050 多約 {f(b62[kB + "|excess_pp"])} 個百分點（{f(b62[kB])}% 對 0050 的 {f(b62["bench_since2019"])}%）'), ('R64 已重現' + N1B, 'R64 重現的是修正前的數字' + N62(kB))],
    new_value=pyv(b62[kB + '|excess_pp']), extra=dict(策略舊=a62[kB], 策略新=b62[kB], 基準舊=a62['bench_since2019'], 基準新=b62['bench_since2019']))
row('小資金_10萬只買整張超額', a62[kD + '|excess_pp'], b62[kD + '|excess_pp'], 1, ok_self('S62'), S62HOW,
    [('每年只比 0050 多約 0.5 個百分點（29.4% 對 28.9%）', f'每年只比 0050 多約 {f(b62[kD + "|excess_pp"])} 個百分點（{f(b62[kD])}% 對 {f(b62["bench_since2019"])}%）'), ('R64 已重現' + N1B, 'R64 重現的是修正前的數字' + N62(kD))],
    new_value=pyv(b62[kD + '|excess_pp']), extra=dict(策略舊=a62[kD], 策略新=b62[kD], 基準舊=a62['bench_since2019'], 基準新=b62['bench_since2019']))
# ---- R0：不重算
row('成本_0050也加滑價少', None, None, 1, None, '沒有重算：這一列是清潔室重寫引擎（R0）自己的敏感度；R0 引擎自己處理減資復牌（不讀減資事件檔 raw/twse_reduce）、0050 用舊序列（integrate_o6/data/etf_0050.csv），正式引擎也沒有「0050 買賣加個股滑價」這個開關，所以無法用正式引擎的新口徑重算。',
    [('（基準為修正前 48.76，未隨 2026-11-21 改版重算，差異預期 ≤0.1）', '（基準為修正前 48.76；2026-11-21 數據稽核部重算風險表時確認：這是 R0 清潔室引擎的敏感度，R0 自己處理減資復牌（不讀減資事件檔）、0050 用舊序列，正式引擎也沒有「0050 買賣加個股滑價」的開關，所以無法用新口徑重算，數字沿用）')],
    note='數值不動，只把註記改成「無法重算」。', src_add='')

# ---- 輸出
todo = [k for k, v in ROWS.items() if ('未隨 2026-11-21 改版重算' in v['desc'] or '待重跑' in v['desc'] or '未隨改版重算' in v['desc'])]
done = {r['項目'] for r in OUT}
missing = [k for k in todo if k not in done]
META_OLD = '其餘各列（S26、S27 模擬、S42、S46、S62、S64、R0）基準仍是修正前 48.76%，沒有隨這次改版重算，各列說明有註明。'
META_NEW = '其餘各列（S26、S27 模擬、S42、S39、S46、S62、S64）已由數據稽核部在 2026-11-21 用新口徑重算（agents/audit_v2_recalc/；同一支程式先逐位重現修正前的值，再換新口徑），各列說明寫了新舊數字；只有 R0 那一列（0050 也加滑價）是清潔室引擎的敏感度，無法用正式引擎重算，沿用舊數字。'
patches = []
for r in OUT:
    if r['原程式行'] != r['新程式行']:
        patches.append(dict(檔案='xq_scripts/site/build_site_data.py', 原文片段=r['原程式行'], 建議改成=r['新程式行'],
                            理由=f"風險表「{r['項目']}」用 2026-11-21 新口徑重算（舊 {r['舊值']} → 新 {r['新值']}）；{r['備註'] or '更新數值與說明、拿掉「未重算」註記'}"))
if not missing:
    patches.append(dict(檔案='xq_scripts/site/build_site_data.py', 原文片段=META_OLD, 建議改成=META_NEW, 理由='風險表總說明：原本寫「其餘各列沒有隨改版重算」，現在已重算（R0 除外）'))
# 試套用（暫存副本）並檢查
new_src = src
for p in patches:
    assert new_src.count(p['原文片段']) == 1, p['原文片段'][:60]
    new_src = new_src.replace(p['原文片段'], p['建議改成'])
j0, j1 = new_src.index('RISK = ['), new_src.index('dump("risk"')
risk_new = ast.literal_eval(new_src[j0 + len('RISK = '):j1].strip())
assert len(risk_new) == 36 and all(len(t) == 5 for t in risk_new)
ast.parse(new_src)
left = [t[0] for t in risk_new if ('未隨 2026-11-21 改版重算' in t[3] or '待重跑' in t[3] or '未隨改版重算' in t[3])]
json.dump(dict(說明='網站風險表（xq_scripts/site/build_site_data.py 的 RISK）待重算各列的新舊對照；舊＝改版前（三個開關、舊 0050 序列），新＝2026-11-21 新口徑（五個開關）。數字是百分比或百分點，4 位小數。',
               還沒重算的列=missing, 套用補丁後仍帶未重算註記的列=left, 驗收=ACC, rows=OUT), open(HERE / 'risk_rows_new.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(patches, open(HERE / 'patches_risk.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('rows', len(OUT), 'patches', len(patches), 'missing', missing, 'left', left)
for r in OUT:
    print(r['項目'], '|', r['網站現值'], '→', r['網站新值'], '|', r['舊值'], '→', r['新值'], '| 重現', r['是否重現舊值'])

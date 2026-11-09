#!/usr/bin/env python3
"""integrate_o18 事前登記（只寫一次）。o18 通過審查的兩題（都「待觀察」，都不產生選股訊號，依 S1）：
  題 1 o_delist_rev_backfill_allpit：補回 2013～2018 下市公司實際月營收（覆蓋 19.8%），ALLPIT_REAL 取代主策略正式數字（紙上 22.22%、t 1.64；V7X +1.04pp）。
  題 2 o_delist_terms_dispmask_fix：下市對價條款補核（consideration_v3）＋處置遮罩官方化（union／replace）。
本整合把兩題的勘誤一起套到主策略（濾網口徑）：ALLPIT_REAL 名單（已含官方處置 union）× v3 對價引擎，紙上與 V7X 可執行都跑。
用法：python3 -I -B prereg.py            （正式：寫 results.json 的 PREREG）
      python3 -I -B prereg.py --dry DIR  （乾跑用假 PREREG：所有變體名單代入 C0、V7X 名單代入 orig 的 seed 0；只檢查程式路徑）"""
import sys, json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
T1 = AG / 'o_delist_rev_backfill_allpit'; T2 = AG / 'o_delist_terms_dispmask_fix'
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/integrate_o18'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
R1 = json.loads((T1 / 'results.json').read_text(encoding='utf-8'))
PV = R1['PREREG']['paper']['variants']
SM = json.loads((HERE / 'smoke.json').read_text(encoding='utf-8'))
VC = json.loads((HERE / 'v7x_plans_check.json').read_text(encoding='utf-8'))
VH = json.loads((HERE / 'v7x_held.json').read_text(encoding='utf-8'))
assert all(SM['lists_sha_match_T1'].values()) and max(SM['C0_reproduce_maxabs'].values()) == 0
assert all(v['maxabs'] == 0 for v in SM['ALLPIT_REAL_opt_cons_main_reproduce'].values())
assert VC['checks']['orig_eq_plans_j7'] and all(VC['checks']['counts_eq_T1'].values()) and VC['checks']['ends_eq_T1']
own = ['ilib.py', 'smoke.py', 'gen_v7x_plans.py', 'v7x_held.py', 'run_paper.py', 'run_v7x.py', 'decide.py', 'prereg.py', 'ro_snap.py',
       'smoke.json', 'v7x_plans_check.json', 'v7x_held.json', 'ro_snapshot_start.json']
ext = [T1 / 'o18lib.py', T1 / 'v7x_lib.py', T1 / 'data/rev_backfill_B1.csv', T2 / 'cz3.py', T2 / 'events_v3.csv', T2 / 'dispmask_official.py',
       AG / 'o_engine_delist_consideration/cz_engine.py', AG / 'o_engine_delist_consideration/cz_v7x.py', AG / 'o_engine_delist_consideration/events_v2.csv',
       AG / 'o_rev_avail_ds_pit_count/pclib.py', AG / 'o_rev_avail_ds_pit_count/avail.py', AG / 'o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv',
       AG / 'independent_audit/ia_core.py', AG / 'main_v2/v2_core.py', AG / 'main_v2/navs_gapfix.parquet',
       AG / 'o_exec_engine_rewrite_v7x/x_exec.py', AG / 'o_exec_engine_rewrite_v7x/x_data.py', AG / 'o_exec_engine_rewrite_v7x/x_stats.py',
       AG / 'o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet', AG / 'o_v7_listing_exec_engine/data/plans_j7.csv',
       AG / 'o_forward_sequential_v2/fslib.py']
FS = {f: sha(HERE / f) for f in own}; FS.update({str(p): sha(p) for p in ext})
REF_SHA = {'T1_results.json': sha(T1 / 'results.json'), 'T2_results.json': sha(T2 / 'results.json')}   # 只記錄（研究中心之後可能追加複製結果），不斷言

variants = {'C0': dict(kind=PV['C0']['kind'], role='主策略 C0（現行紙上正式；main_v2 gapfix、排除金融 n10）'),
            'ALLPIT_REAL_opt': dict(kind=PV['ALLPIT_REAL_opt']['kind'], role='題 1 新正式名單（未覆蓋槽照 opt 放回；不利端）＋A_UB＋官方處置 union'),
            'ALLPIT_REAL_p90': dict(kind=PV['ALLPIT_REAL_p90']['kind'], role='區間：未覆蓋槽照 p90'),
            'ALLPIT_REAL_carry': dict(kind=PV['ALLPIT_REAL_carry']['kind'], role='區間：未覆蓋槽沿用最後已知營收'),
            'ALLPIT_REAL_opt_KYX': dict(kind=PV['ALLPIT_REAL_opt_KYX']['kind'], excl=PV['ALLPIT_REAL_opt_KYX']['excl'], role='敏感度：KY 四檔不放回'),
            'ALLPIT_REAL_opt_DSPR': dict(kind=PV['ALLPIT_REAL_opt']['kind'], dsp_mode='replace', role='敏感度：處置遮罩用題 2 replace 模式（2014～2021-09-23 只用官方清單；2006～2013 不加官方）')}
engines = {'C0': {'freeze': ['normal', 'stress'], 'v3_main': ['normal']},
           'ALLPIT_REAL_opt': {'cons_main': ['normal', 'stress'], 'v3_main': ['normal', 'stress'], 'v3_lo': ['normal', 'stress'], 'v3_hi': ['normal', 'stress'], 'freeze': ['normal']},
           'ALLPIT_REAL_p90': {'cons_main': ['normal', 'stress'], 'v3_main': ['normal', 'stress'], 'v3_lo': ['normal'], 'v3_hi': ['normal']},
           'ALLPIT_REAL_carry': {'cons_main': ['normal', 'stress'], 'v3_main': ['normal', 'stress'], 'v3_lo': ['normal'], 'v3_hi': ['normal']},
           'ALLPIT_REAL_opt_KYX': {'v3_main': ['normal', 'stress']},
           'ALLPIT_REAL_opt_DSPR': {'v3_main': ['normal', 'stress']}}
HP = {k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in SM['held_events_paper'].items()}
HV = {k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in VH.items()}
PR = dict(
    written_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
    topic='integrate_o18：o18 兩題勘誤合併套到主策略（52 週高 0.5＋營收 3 月年增 0.5、前 300 池、月換 10 檔、排除金融）——ALLPIT_REAL 名單 × consideration_v3 對價引擎；紙上 ia＋V7X 可執行',
    s1=('依 S1，兩題都沒有個股分數也沒有自己的報酬序列；本整合不產生新選股訊號、不改主策略規則、不立靠歷史回測取得可投資的題。'
        'G3 族群不增加（勘誤變體是主策略本身的資料敏感度）。'),
    combos=dict(
        I1_filter=('「勘誤濾網」組合（主）：題 1 ALLPIT_REAL_opt 名單（2013～2018 補回實際營收、未覆蓋照 opt 放回、A_UB 營收回退、官方處置 union）× 題 2 consideration_v3 結算規則（v3_main；界限 v3_lo／v3_hi）。'
                   '紙上＝ALLPIT_REAL_opt × v3_main；V7X＝REAL_opt × v3_main。這是「o18 兩題一起生效」後的主策略數字候選。'),
        I2_interval='ALLPIT_REAL_p90、ALLPIT_REAL_carry × v3_main（紙上；V7X 同）：未覆蓋槽的放回假設區間。',
        I3_sens='KY 四檔不放回（KYX）；處置 replace 模式（DSPR）。只描述，不決定正式數字。',
        I4_score_bonus='加分項組合：不適用。兩題都沒有分數可加。',
        I5_second_leg=('第二條腿固定比例混合（80/20、50/50）：依事前登記不做。兩題都沒有獨立報酬序列；勘誤版與 C0 的相關約 0.95，混合等於自己混自己；'
                       '和 0050 固定比例混合在「不再開」清單。主策略與 0050 的混合數字引用 integrate_o14（全部不可用），不重跑。')),
    paper=dict(variants=variants, engines=engines,
               stats=('o18lib.win()／stats_pair()：訊號日收盤→下一訊號日收盤窗（245 窗）對 0050 含息；ALL、SEG1 2006～2012、SEG2 2013～2018、SEG3 2019 起、SEL ～2019-04、VAL 2019-05 起、HOLD 2024 起；'
                      '年化、超額、t、窗頻與日頻最大回撤、拿掉最好 3 年 t、對參照月差與 t、相關（對參照、對 0050）、逐年絕對差。'
                      '參照：C0 freeze（同成本）；ALLPIT_REAL_opt cons_main（題 1 新正式，同成本）；同名單 cons_main（引擎差）；KYX／DSPR 另對 ALLPIT_REAL_opt v3_main。'),
               checks='C0 freeze 逐日重現 navs_gapfix（< 1e−12）；ALLPIT_REAL_{opt,p90,carry} × cons_main 的 ALL/SEG1/SEG2/SEG3/HOLD 年化、超額、t、月差重現題 1 RESULTS_PAPER（< 1e−9）；名單 sha＝smoke.json。'),
    v7x=dict(plans_file=VC['file'], plans_sha256=VC['sha256'],
             plans_note='gen_v7x_plans.py 用題 1 v7x_lib.make 重建（題 1 名單檔已清除）；orig 逐筆＝plans_j7（seed < 20）；四變體的列數與平均放回槽數＝題 1 RESULTS_V7X_PLANS.counts；ends 相同。',
             engines={'REAL_opt': ['cons_main', 'v3_main', 'v3_lo', 'v3_hi'], 'REAL_p90': ['v3_main'], 'REAL_carry': ['v3_main']},
             grid='K5T × GROW × {一般, 壓力} × {100萬, 300萬, 1000萬}；20 seeds 日報酬平均成一條淨值；x_stats.stats／jsumm（13 j 中位數與 p10）；分段 S0612／S1318／S19on 用題 1 v7x_run 的 segx。',
             reference='現行正式＝orig × orig（直接讀 navs_oos_R4.parquet，同一段統計程式）；題 1 新正式＝REAL_opt × cons_main（須重現題 1 表 < 1e−9）。',
             extra='逐年差（j 中位數）：REAL_opt v3_main − REAL_opt cons_main、REAL_opt v3_main − 現行正式；日報酬相關（j 中位數）。'),
    feasibility_known_at_freeze=dict(
        paper_held_delist_events=HP, v7x_held_delist_events=HV,
        notes=['C0 名單沒有任何持有跨越下市事件 → C0 在所有引擎下數字相同（結構性；C0×v3_main 只當核對）。',
               '紙上 ALLPIT_REAL_opt 持有 10 起下市事件；v3_main 和 cons_main 規則不同的 5 起（2403、2432、2473、3009、3012）；lo≠hi 的 2 起（2403、3063）→ 紙上界限可觸發但幅度受限於 2 起。',
               'V7X REAL_opt 持有 15 起；v3≠cons 8 起；lo≠hi 6 起（1520、1716、2403、2446、2452、3063）→ V7X 下界可觸發。',
               '2311、2325、3474、6286（題 1 放回的大權重下市股）在 v3 與 cons 規則相同（題 2 沒改這 4 起；o17 已核實）→ 補回營收的主要影響不受引擎版本影響。',
               'D1 取代規則能否觸發：可以（v3≠cons 的事件確實被持有）。D2「界限兩端 > 0」能否觸發：可以。D3 G4「已排除」：結構上不能（覆蓋 19.8% < 80%），本整合不改。']),
    decision=dict(
        D1=('取代規則：ALLPIT_REAL_opt × v3_main 對 × cons_main（題 1 新正式），下列任一 |Δ年化| > 0.3pp 或 |Δt| > 0.05 → 建議正式數字改用 v3_main 版（兩題一起生效），題 1 版移附錄；'
            '否則題 1 新正式數字維持，旁加註「consideration_v3 對價差 Δ」。'
            '欄位：紙上一般成本 全期年化、t、SEG1、SEG2、SEG3；紙上壓力成本 全期年化；V7X 100 萬一般 exc j 中位數、p10、2019 起、t；100 萬壓力、300 萬、1,000 萬一般 exc j 中位數。'),
        D2='V7X 對價界限：REAL_opt × v3_lo 與 × v3_hi 的 100 萬一般 exc j 中位數都 > 0 → 加註「對價界限兩端 > 0」；否則寫出界限，並依 feasibility 的 lo≠hi 事件清單寫明由哪些事件決定。',
        D3='G4 第二段（2013～2018）：整合主口徑一般或壓力成本 ≤ 0 → 寫「可能翻號」；否則仍因覆蓋 < 80% 維持「存活者偏誤未排除」。G4「已排除」在結構上無法觸發。',
        D4='整合結果不推翻兩題自己的判定（兩題都「待觀察」，G6 獨立複製未做）。',
        D5='分級：主策略與所有整合變體依 G1～G8 判定；預期「不可投資」，不得寫「歷史最高標準通過」。',
        D6='敏感度：DSPR、KYX 對整合主口徑 |Δ年化| > 0.3pp → 加註；否則記「可忽略」。',
        pipeline='任一重現核對 > 1e−9 → 判定暫停，只報數字並標「管線不一致」。',
        G_table_rule='G1 j p10 > 0 且壓力 > 0；G2 全期 t ≥ 3.0 且 2019 起 t ≥ 2.0（可執行主欄）；G3 繼承 N 族群（SPA p 0.9997）；G4 三段 > 0、拿掉最好 3 年 t ≥ 2；G5 PIT；G6 複製；G7 三級資金 > 0；G8 前瞻 12 個月。'),
    predictions=dict(
        P1='紙上 ALLPIT_REAL_opt v3_main 對 cons_main：Δ年化 −0.10pp（區間 −1.00～+0.30，不利端放寬）；Δt −0.01（−0.15～+0.05）。依據：題 2 U_opt v3_main 對 o17 cons −0.07pp。',
        P2='紙上 v3_lo～v3_hi 寬度 0.3pp（≤ 0.8pp）；三段符號與 cons_main 相同。',
        P3='V7X REAL_opt v3_main 100 萬一般 exc j 中位數 +0.9pp（−0.5～+1.3）；p10 −2.2（−3.5～−1.5）；2019 起 −6.2（−7.5～−5.5）。',
        P4='V7X REAL_opt v3_lo +0.4pp（−1.5～+1.0）；v3_hi +1.2（+0.3～+1.6）；界限兩端 > 0 的機率約 50%。',
        P5='DSPR、KYX 對整合主口徑 |Δ年化| ≤ 0.5pp。',
        P6='D1 不觸發（紙上與 V7X 的 v3 對 cons 差都 ≤ 0.3pp）的機率約 60%。',
        P7='G1～G8 全部不過；主策略不可投資。'),
    readonly=('live_tracking/main_forward、o_forward_infra_live、o_g8p2a_protocol_addendum：前後 sha256 快照比對（ro_snapshot_start/end.json）。'
              '執行時段 2026-11-09 01:35Z～約 03:30Z（台北 09:35～11:30），不在每日 06:49 台北（22:49Z）排程，也不在 11-10／11-11 換股日；若有差異依時段判斷。不 import、不呼叫 o_g8p2a_protocol_addendum 判定器。'),
    discipline='python3 -I -B；只寫 integrate_o18/ 與 scratchpad/integrate_o18/（結束時刪除）；不 git commit；子代理不能 commit，PREREG 沒有外部錨定，以 written_utc 早於各 RUN 為準。',
    scratch=SCR, ref_results_sha_recorded=REF_SHA, file_sha256=FS)

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--dry':
        out = Path(sys.argv[2]); import copy
        P2 = copy.deepcopy(PR)
        for k in P2['paper']['variants']:
            P2['paper']['variants'][k] = dict(kind=PV['C0']['kind'])          # 名單全部代入 C0（不帶任何變體資訊）
        P2['file_sha256'] = {}
        P2['v7x']['plans_file'] = str(out / 'plans_dry.csv'); P2['v7x']['plans_sha256'] = sha(out / 'plans_dry.csv')
        P2['scratch'] = str(out)
        (out / 'results.json').write_text(json.dumps(dict(PREREG=P2, PREREG_sha256=hashlib.sha256(json.dumps(P2, ensure_ascii=False, sort_keys=True).encode()).hexdigest()), ensure_ascii=False, indent=1), encoding='utf-8')
        print('dry PREREG written', out)
    else:
        RJ = HERE / 'results.json'
        assert not RJ.exists(), 'results.json 已存在：PREREG 只寫一次'
        RJ.write_text(json.dumps(dict(PREREG=PR, PREREG_sha256=hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest()), ensure_ascii=False, indent=1), encoding='utf-8')
        print('PREREG written', PR['written_utc'], hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest())

#!/usr/bin/env python3
"""步驟 1：探索（只看形狀、不看平均超額）→ explore.json、d5_resid_sel.json。
 E1 挑選期（2006-04～2019-04）V7X GROW100 較差成本 JMED 殘差：ACF lag 1～12、24 個月和的變異比；與 2019 年起比較（重現審查 0.099／+0.163／1.22 與 −0.283／0.45）
 E2 J12／J0 重疊造成的機械 MA 成分：以實際 J 窗日期（交易日）重疊比例估理論 ρ1、θ
 E3 重現審查數字：v2（G8P-2 V1 原樣）在 AR ρ 0.057／0.08／0.1、MA θ 0.3、D5 的誤升級（σ 8.4／9.5、s 0.78）
 E4 治理漏洞核對：讀 g8_evaluate_v2.py 原始碼＋合成帳本示範攻擊（回溯更正、open 月份、adj 覆寫、committer date 偽造）
 E5 xlive.py CFG_V7X／R4 的 stress_odd_extra
用法：python3 -I -B s1_explore.py"""
import sys, json, math, re, shutil, subprocess, importlib.util, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import g8alib as L
import synth as SY

SCR = L.SCR / 'explore'; shutil.rmtree(SCR, ignore_errors=True); SCR.mkdir(parents=True)
out = dict(utc_start=L.now_utc())
P2 = L.v2_protocol()


def acf(x, k):
    x = x - x.mean(); return float((x[k:] * x[:-k]).sum() / (x * x).sum())


def shape(x, nm):
    x = np.asarray(x, float)
    ac = [acf(x, k) for k in range(1, 13)]
    vr11 = 1 + 2 * sum((1 - k / 24) * acf(x, k) for k in range(1, 12))
    vr23 = 1 + 2 * sum((1 - k / 24) * acf(x, k) for k in range(1, 24))
    # 經驗：非重疊 24 個月和的變異 ÷ (24·σ²)（環狀全部起點）
    n = len(x); y = x - x.mean(); yy = np.concatenate([y, y])
    s24 = np.array([yy[i:i + 24].sum() for i in range(n)])
    vr_emp = float(s24.var() / (24 * y.var()))
    return dict(name=nm, n=n, acf=[round(a, 4) for a in ac], acf1=round(ac[0], 4), acf6=round(ac[5], 4), se_acf=round(1 / math.sqrt(n), 3),
                VR24_trunc11=round(vr11, 3), VR24_trunc23=round(vr23, 3), VR24_circular_empirical=round(vr_emp, 3))


# ---------------------------------------------------------------- E1
Mn, Ms, F, win = L.hist_windows()
w = L.jmed_line(Mn, Ms)
sel = w[(w.index >= pd.Period('2006-04', 'M')) & (w.index <= pd.Period('2019-04', 'M'))].values.astype(float)
post = w[(w.index >= pd.Period('2019-05', 'M')) & (w.index <= pd.Period('2026-09', 'M'))].values.astype(float)
rp = np.array(P2['data']['resid_post2019'], float)
E1 = dict(sel_V7X_worse=shape(sel, 'sel 2006-04～2019-04 V7X GROW100 較差成本 JMED'),
          post_V7X_worse=shape(post, '2019-05～2026-09 V7X GROW100 較差成本 JMED（只描述，不新增使用）'),
          post_protocol_resid=shape(rp, 'G8P-2 協定內嵌 resid_post2019（D3 來源，89 個月）'),
          note='只取去均值後的形狀；平均超額沒有輸出、沒有用於任何設計步驟')
z = sel - sel.mean(); z = z / z.std(ddof=1)
d5 = dict(desc='D5：挑選期 2006-04～2019-04 V7X B10 K5T GROW 100 萬較差成本（逐窗 min(一般, 壓力)）JMED 線（有效 J 窗 ≥ 7），去均值後除以 sd（ddof＝1）；只用形狀',
          first='2006-04', last='2019-04', n=len(z), z=[round(float(v), 10) for v in z])
zz = np.array(d5['z']); zz = zz - zz.mean(); zz = zz / zz.std(ddof=1); d5['z'] = [float(v) for v in zz]
(HERE / 'd5_resid_sel.json').write_text(json.dumps(d5, ensure_ascii=False), encoding='utf-8')
E1['d5_file_sha256'] = L.sha(HERE / 'd5_resid_sel.json')
E1['d5_n'] = len(zz)
out['E1_acf'] = E1
print('E1', {k: (v['acf1'], v['acf6'], v['VR24_trunc11'], v['VR24_circular_empirical']) for k, v in E1.items() if isinstance(v, dict)})

# ---------------------------------------------------------------- E2 J 窗重疊的理論 MA
cal = F.cal()
pos = {d: i for i, d in enumerate(cal)}
anchors = sorted(set(win[0].index) & set(win[12].index))
r1s = []
for a, b in zip(anchors[:-1], anchors[1:]):
    if b != a + 1 or a > pd.Period('2019-04', 'M') or a < pd.Period('2006-04', 'M'): continue
    def iv(m):
        return [(pos.get(win[j].loc[m, 'start']), pos.get(win[j].loc[m, 'end'])) for j in range(13) if m in win[j].index]
    A_, B_ = iv(a), iv(b)
    if any(x is None for p in A_ + B_ for x in p): continue
    def ov(p, q): return max(0, min(p[1], q[1]) - max(p[0], q[0]))
    va = sum(ov(p, q) for p in A_ for q in A_) / len(A_) ** 2
    vb = sum(ov(p, q) for p in B_ for q in B_) / len(B_) ** 2
    cv = sum(ov(p, q) for p in A_ for q in B_) / (len(A_) * len(B_))
    r1s.append(cv / math.sqrt(va * vb))
r1 = float(np.mean(r1s))
theta = (1 - math.sqrt(1 - 4 * r1 * r1)) / (2 * r1) if r1 > 0 else 0.0
win_len = [int(pos[win[12].loc[m, 'end']] - pos[win[12].loc[m, 'start']]) for m in anchors if pd.Period('2006-04', 'M') <= m <= pd.Period('2019-04', 'M')]
j12_vs_j0 = [int(pos[win[12].loc[a, 'end']] - pos[win[0].loc[a + 1, 'start']]) for a in anchors
             if (a + 1) in win[0].index and pd.Period('2006-04', 'M') <= a <= pd.Period('2019-04', 'M')]
E2 = dict(method='x_m 近似為 13 個 J 窗報酬的平均；日超額 iid 等變異時，ρ1＝Cov(x_m, x_{m+1})／Var，以實際 J 窗起訖交易日的重疊天數計；θ 解 θ/(1+θ²)＝ρ1',
          n_pairs=len(r1s), rho1_theory=round(r1, 4), rho1_p10_p90=[round(float(np.percentile(r1s, 10)), 4), round(float(np.percentile(r1s, 90)), 4)],
          theta_theory=round(theta, 4), theta_from_max_rho1=round(float((1 - math.sqrt(1 - 4 * max(r1s) ** 2)) / (2 * max(r1s))), 4),
          j12_window_len_tdays_median=float(np.median(win_len)), j12_end_minus_next_j0_start_tdays_median=float(np.median(j12_vs_j0)),
          grid_rule='事前規則：估計 θ > 0.3 → 網格擴到該值；θ ≤ 0.3 → 網格 {0.1, 0.2, 0.3} 涵蓋',
          grid_extended=bool(theta > 0.3), empirical_acf1_sel=E1['sel_V7X_worse']['acf1'],
          note='中位數取代平均、J0 作廢（判定線實際 J1～J12）只會讓重疊比例小幅變動；經驗 acf1 0.099 含真實報酬相關與重疊兩者')
out['E2_overlap_theta'] = E2
print('E2', E2)

# ---------------------------------------------------------------- E3 重現審查（v2 原樣）
NP = 200_000
E3 = {}
for dgp in ('ARN057', 'ARN08', 'ARN10', 'AR057', 'AR08', 'AR10', 'MA3', 'MAT3', 'D5', 'D3', 'D1'):
    row = {}
    for sg_i, sg in ((5, 8.4), (7, 9.5)):
        rng = L.cell_rng(2026110700, dgp, sg_i, 4, 2, 0)
        X = L.gen(dgp, 0.78 - L.DW, sg, NP, 24, rng)
        Z = L.all_stats(X)
        o = L.outcomes_rule(Z['lit'], Z['v212'], Z['v218'], Z['v224'], L.B_V2, L.CD_V2)
        row[str(sg)] = round(float(o['up'].mean()), 5)
    E3[dgp] = row
out['E3_v2_false_up_s078'] = dict(P=NP, seed='SeedSequence([2026110700, DGP_ID, σ 序, 4, 2, 0])', table=E3,
                                  review_numbers=dict(AR057='5.64～5.83%（σ 8.4／9.5）', AR08='6.1%', AR10='6.5%', MA3='9.1～9.4%', D5='約 4.5%（σ 9.5）', D3='約 1.1%'),
                                  note='ARN＝常態創新 AR（審查口徑，只作敏感度）；AR＝t5 創新（本題 OC 口徑）；MA＝常態創新（本題 OC 口徑，較 t5 保守）；MAT＝t5 MA（敏感度）')
print('E3', E3)

# ---------------------------------------------------------------- E4 治理漏洞
src = (L.V2_DIR / 'g8_evaluate_v2.py').read_text(encoding='utf-8')
assert L.sha(L.V2_DIR / 'g8_evaluate_v2.py') == L.V2_EVAL_SHA
lines = src.splitlines()
def find(pat): return [i + 1 for i, l in enumerate(lines) if re.search(pat, l)]
code = dict(adj_overwrite_lines=find(r"adj = 100\.0 \* float\(c\['delta_per_side'\]\)"),
            open_skip_lines=find(r"state='open'\)\); continue") + find(r"if r\['state'\] != 'valid': continue"),
            void_count_only_void=find(r"if r\['state'\] == 'void':"),
            committer_date_lines=find(r"%H %cI"),
            backlog_text_search=find(r"sha\(man_p\) in BACKLOG"),
            status_recompute_no_chain=('status_chain' not in src), s1_requires_real_calib=('NO_REAL_CALIB' in src),
            registry_utc_check=('sched_date' in src and 'days=30' in src))
spec = importlib.util.spec_from_file_location('g8ev2_ro', L.V2_DIR / 'g8_evaluate_v2.py'); EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
REG = SCR / 'reg.jsonl'; shutil.copy(L.V2_DIR / 'calib_registry.jsonl', REG)
BASE = ['--test-protocol', str(L.V2_DIR / 'G8_PROTOCOL_v2.json'), '--test-manifest', str(L.V2_DIR / 'FREEZE_MANIFEST_v2.json'), '--test-skip-anchor', '--quiet']
def run(fwd, reg=REG, extra=()):
    return EV.main(['status', '--test-forward', str(fwd), '--test-registry', str(reg)] + BASE + list(extra))
# (a) 回溯更正：12 個月 Σx ≤ 0 → FORWARD_FAILED；事後把一個月的 forward_returns 改好 → 翻回
xs = [0.3, -1.0, 0.6, 0.2, -0.4, 0.8, 0.1, 0.5, 0.3, 0.2, 0.4, 0.2]       # 判定線＝x−0.5 → Σ＝−2.8
FW = SCR / 'fwd_a'; SY.make_ledger(FW, SY.months_seq(xs)); o1 = run(FW)
xs2 = list(xs); xs2[1] = 3.0                                                   # 更正第 2 個月 → Σ＝+1.2
SY.make_ledger(FW, SY.months_seq(xs2)); o2 = run(FW)
att_a = dict(before=o1['state_if_valid'], after_correction=o2['state_if_valid'], works_in_v2=o1['state_if_valid'] != o2['state_if_valid'])
# (b) open 月份：13 個月，其中第 2 個月很差；把它標成 open → 被跳過、不計作廢、第 13 個月遞補 → 判定翻轉
xs3 = [0.6] * 13; xs3[1] = -9.0
FW = SCR / 'fwd_b'; SY.make_ledger(FW, SY.months_seq(xs3)); o3 = run(FW)
SY.make_ledger(FW, SY.months_seq(xs3, opens=(1,))); o4 = run(FW)
att_b = dict(all_complete=o3['state_if_valid'], month2_open=o4['state_if_valid'],
             month2_row=[r for r in o4['by_mode']['retro']['anchors'] if r['anchor'] == '2026-12'][0],
             g8=o4['by_mode']['retro']['g8_literal'], works_in_v2=o3['state_if_valid'] != o4['state_if_valid'])
# (c) adj 覆寫：兩筆 REAL_CALIB（Δ 0.004 先、0.001 後）→ v2 用最後一筆，不是最大值
def ent(prev, seq, sd, delta, vf):
    e = dict(seq=seq, type='REAL_CALIB', sched_date=sd, utc=f'{sd}T08:00:00+00:00', valid_from=vf, counts_toward_slot=True, applied=True,
             delta_per_side=delta, prev_sha=prev)
    e['entry_sha'] = EV.entry_sha(e); return e
g = json.loads(REG.read_text().splitlines()[0])
e1 = ent(g['entry_sha'], 1, '2027-02-12', 0.004, '2026-11-01T00:00:00+00:00'); e2 = ent(e1['entry_sha'], 2, '2028-02-12', 0.001, '2026-11-01T00:00:00+00:00')
REGc = SCR / 'reg_c.jsonl'; REGc.write_text(REG.read_text() + EV.canon(e1) + '\n' + EV.canon(e2) + '\n', encoding='utf-8')
FW = SCR / 'fwd_c'; SY.make_ledger(FW, SY.months_seq([2.0] * 3), turnover=0.8)
FWs = SCR / 'fwd_c_stressfree'
o5 = run(FW, REGc, ['--test-now', '2026-12-01T00:00:00Z'])
xr = [r.get('x') for r in o5['by_mode']['retro']['anchors']]
att_c = dict(x_retro_v2=xr, expected_if_last=round(min(2.0 - 100 * 0.001 * 0.8, 1.5), 6), expected_if_max=round(min(2.0 - 100 * 0.004 * 0.8, 1.5), 6),
             note='excess_pp＝2.0、壓力＝1.5、換手 0.8：adj(last)＝0.08 → min(1.92,1.5)＝1.5（壓力仍較差，覆寫不顯形）；改用壓力不較差的情境再測',
             )
# 讓壓力不較差：excess_stress_pp 設很高 → 決策線由一般成本扣 adj 決定
fr = pd.read_csv(FW / 'forward_returns.csv'); fr['excess_stress_pp'] = 9.0; fr.to_csv(FW / 'forward_returns.csv', index=False)
o6 = run(FW, REGc, ['--test-now', '2026-12-01T00:00:00Z'])
xr2 = [r.get('x') for r in o6['by_mode']['retro']['anchors']]
att_c.update(x_retro_v2_stress_high=xr2, expected_last=2.0 - 100 * 0.001 * 0.8, expected_max=2.0 - 100 * 0.004 * 0.8,
             v2_uses_last_month1=bool(abs(xr2[0] - (2.0 - 100 * 0.001 * 0.8)) < 1e-9), v2_would_be_max_month1=2.0 - 100 * 0.004 * 0.8,
             note2='第 1 個月換手＝0.8（GROW 淨值起算 100 萬）；之後月份淨值成長、換手略降。v2 retro 取最後一筆 Δ＝0.001 而非最大 0.004')
# (d) committer date 偽造：暫存 repo，GIT_COMMITTER_DATE 早於期限
GR = SCR / 'gitforge'; GR.mkdir()
env = dict(os.environ, GIT_COMMITTER_DATE='2026-11-01T00:00:00+00:00', GIT_AUTHOR_DATE='2026-11-01T00:00:00+00:00',
           GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')
subprocess.run(['git', 'init', '-q', str(GR)], check=True)
(GR / 'm.json').write_text('{}'); subprocess.run(['git', '-C', str(GR), 'add', 'm.json'], check=True)
subprocess.run(['git', '-C', str(GR), 'commit', '-q', '-m', 'forged'], check=True, env=env)
ci = subprocess.run(['git', '-C', str(GR), 'log', '--format=%H %cI'], capture_output=True, text=True).stdout.strip()
att_d = dict(local_cI=ci.split(' ', 1)[1], real_utc_now=L.now_utc(), works_in_v2=ci.split(' ', 1)[1].startswith('2026-11-01'),
             note='v2 git_anchor() 以 %cI 判生效月：偽造 committer date 即可讓晚於期限的 commit 看起來早於期限')
out['E4_governance'] = dict(code_lines=code, attack_retro_correction=att_a, attack_open_month=att_b, adj_overwrite=att_c, committer_date_forgery=att_d,
                            n_attacks_change_v2=int(att_a['works_in_v2']) + int(att_b['works_in_v2']))
print('E4', json.dumps(out['E4_governance'], ensure_ascii=False, default=str)[:1500])

# ---------------------------------------------------------------- E5
xl = (L.AG / 'o_forward_infra_live/xlive.py').read_text(encoding='utf-8')
m = re.search(r'CFG_V7X = dict\((.*?)\)', xl, re.S)
r4 = json.loads((L.AG / 'o_exec_engine_rewrite_v7x/results.json').read_text(encoding='utf-8'))['PREREG']['orig_cfg']
xe = (L.AG / 'o_exec_engine_rewrite_v7x/x_exec.py').read_text(encoding='utf-8')
out['E5_stress_odd_extra'] = dict(xlive_CFG_V7X=re.sub(r'\s+', ' ', m.group(1)), xlive_true=('stress_odd_extra=True' in m.group(1)),
                                  R4_orig_cfg=r4, R4_true=bool(r4.get('stress_odd_extra')),
                                  x_exec_default=re.search(r".*stress_odd_extra.*EXTRA.*", xe).group(0).strip(),
                                  v2_protocol_text=[t for t in P2['cost_calibration']['uncalibratable_without_real_fills'] if 'stress_odd_extra' in t],
                                  correction='前瞻產生器 xlive.py CFG_V7X 與 V7X 正式 R4（PREREG orig_cfg）實際為 stress_odd_extra＝True（壓力線零股也加 0.4%）；'
                                             'v2 協定原文「預設關閉」只對 x_exec.py 未給設定時的程式預設成立，對前瞻產生器有誤')
print('E5', out['E5_stress_odd_extra'])
out['utc_end'] = L.now_utc()
(HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=L._js), encoding='utf-8')
L.save_key('EXPLORE', dict(file='explore.json', sha256=L.sha(HERE / 'explore.json'), utc=out['utc_end']))
shutil.rmtree(SCR, ignore_errors=True)

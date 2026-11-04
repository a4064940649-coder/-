#!/usr/bin/env python3
"""步驟 5（凍結後的最終 OC）：斷言 PREREG 存在；seed 集 B、每格 P＝40,000；主 DGP（D1～D3）＋敏感度 DGP；
選定版（與其他 3 版並列）、字面 G8、合併程序、預測 P1～P4 對照。輸出 results.json 的 OC_FINAL。
用法：python3 -I -B s5_oc_final.py"""
import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import g8lib as G
import s3_iterate as S3

PRE = G.check_reg('PREREG')
D = G.check_reg('PREREG_DESIGN')
H = 48
RES24 = np.load(G.SCR / 'resid_2024.npy')
SHIFT = G.load_res()['EXPLORE']['cost_shifts_pp']['stress_minus_normal_FIX100_sel']


def summarize(o, dgps, crit):
    det, ok = {}, True
    for g in dgps:
        fu = o[f'{g}|+0.0'][24]['up']; pw = o[f'{g}|-1.0'][24]['down']; fd = o[f'{g}|+0.0'][24]['down']
        det[g] = dict(false_up=fu, false_up_se=o[f'{g}|+0.0'][24]['se_up'], down_power=pw, false_down=fd,
                      up_p05=o[f'{g}|+0.5'][24]['up'], up_p1=o[f'{g}|+1.0'][24]['up'], up_p2=o[f'{g}|+2.0'][24]['up'],
                      down_m05=o[f'{g}|-0.5'][24]['down'], literal_g8_a0=o[f'{g}|+0.0']['literal_g8_pass'],
                      literal_g8_p05=o[f'{g}|+0.5']['literal_g8_pass'], literal_g8_m1=o[f'{g}|-1.0']['literal_g8_pass'],
                      combined_up_p1=o[f'{g}|+1.0']['combined'][24]['up'], combined_fail_a0=o[f'{g}|+0.0']['combined'][24]['fail'],
                      combined_fail_m1=o[f'{g}|-1.0']['combined'][24]['fail'], down_a0_48=o[f'{g}|+0.0'][48]['down'], down_m1_48=o[f'{g}|-1.0'][48]['down'])
        se = lambda p: 2 * G.mc_se(p, PS)
        ok &= (fu <= crit['false_up_24_max'] + se(fu)) and (pw >= crit['down_power_24_alpha_m1_min'] - se(pw)) and (fd <= crit['false_down_24_max'] + se(fd))
    return bool(ok), det


def main():
    global PS
    t0 = time.time()
    V = S3.params()
    S = D['sims']['final']; PS = S['P']
    prim = D['pass_criteria']['primary_dgps']
    grid = S3.run_grid(V, prim, D['alphas'], PS, S['seed_base'], H=H)
    sens = S3.run_grid(V, ['D3_2024', 'D3_L3', 'D3_L12', 'D4_o8'], D['alphas'], PS, S['seed_base'] + 1, H=H,
                       resid_map={'D3_2024': RES24}, block_map={'D3_L3': 3, 'D3_L12': 12})
    stress = S3.run_grid(V, prim, D['alphas'], PS, S['seed_base'] + 2, H=H, shift=SHIFT)
    out = dict(utc=G.now_utc(), P=PS, H=H, seed_base=S['seed_base'], stress_shift=SHIFT, versions={})
    for v in V:
        okp, detp = summarize(grid[v], prim, D['pass_criteria'])
        _, dets = summarize(sens[v], ['D3_2024', 'D3_L3', 'D3_L12', 'D4_o8'], D['pass_criteria'])
        _, detst = summarize(stress[v], prim, D['pass_criteria'])
        out['versions'][v] = dict(rule=V[v][0], pass_primary_within_2se=okp, primary=detp, sensitivity=dets, stress=detst,
                                  oc_primary={k: {h: x for h, x in val.items()} for k, val in grid[v].items()})
    sel = PRE['selected']
    fin = out['versions'][sel]
    out['selected'] = sel
    out['selected_pass_final'] = fin['pass_primary_within_2se']
    # 預測對照
    pr = {}
    p = fin['primary']
    pr['P1'] = dict(literal_a0_D1=p['D1']['literal_g8_a0'], literal_p05_D1=p['D1']['literal_g8_p05'],
                    ok=bool(abs(p['D1']['literal_g8_a0'] - 0.5) <= 0.02 and abs(p['D1']['literal_g8_p05'] - 0.611) <= 0.03))
    pr['P2'] = dict(false_up=[p[g]['false_up'] for g in prim], up_p1=[p[g]['up_p1'] for g in prim], up_p2=[p[g]['up_p2'] for g in prim],
                    ok_false_up=all(p[g]['false_up'] <= 0.05 for g in prim),
                    ok_p1_in_15_30=all(0.15 <= p[g]['up_p1'] <= 0.30 for g in prim), ok_p2_in_45_70=all(0.45 <= p[g]['up_p2'] <= 0.70 for g in prim))
    pr['P3'] = dict(down_m1=[p[g]['down_power'] for g in prim], false_down=[p[g]['false_down'] for g in prim],
                    ok=all(p[g]['down_power'] >= 0.5 and p[g]['false_down'] <= 0.3 for g in prim))
    s = fin['sensitivity']['D3_2024']
    pr['P4'] = dict(up_p1_D3=p['D3']['up_p1'], up_p1_D3_2024=s['up_p1'], diff_pp=100 * (p['D3']['up_p1'] - s['up_p1']),
                    ok_power_drop_5_10=bool(5 <= 100 * (p['D3']['up_p1'] - s['up_p1']) <= 10),
                    stress_shift_pp=SHIFT, ok_shift_in_m01_m02=bool(-0.2 <= SHIFT <= -0.1),
                    literal_a0_D1=p['D1']['literal_g8_a0'], literal_a0_D1_stress=fin['stress']['D1']['literal_g8_a0'],
                    literal_drop_pp=100 * (p['D1']['literal_g8_a0'] - fin['stress']['D1']['literal_g8_a0']),
                    ok_literal_drop_2_4=bool(2 <= 100 * (p['D1']['literal_g8_a0'] - fin['stress']['D1']['literal_g8_a0']) <= 4))
    out['predictions'] = pr
    out['runtime_s'] = time.time() - t0
    G.save_key('OC_FINAL', out, need=['PREREG'])
    for v in V:
        o = out['versions'][v]
        print(v, o['rule'], o['pass_primary_within_2se'], {g: {k: round(x, 4) for k, x in d.items() if k in ('false_up', 'down_power', 'false_down', 'up_p1', 'up_p2')} for g, d in o['primary'].items()})
    print(json.dumps(pr, ensure_ascii=False, default=G._js))
    print(json.dumps({g: {k: round(x, 4) for k, x in d.items()} for g, d in fin['sensitivity'].items()}, ensure_ascii=False))
    print(json.dumps({g: {k: round(x, 4) for k, x in d.items()} for g, d in fin['stress'].items()}, ensure_ascii=False))
    print('runtime', round(time.time() - t0))


if __name__ == '__main__':
    main()

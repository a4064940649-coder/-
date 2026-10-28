#!/usr/bin/env python3
"""步驟 2 的模型驗證（迭代只能針對這一步）：合成族群放入已知 α 分布（保留 SEL 期真實相關結構），檢查
(1) 80% 事後區間涵蓋率 75～85%；(2) Tweedie 修正後族群最大值不再系統性高估；(3) 換種子穩定。
用法：python3 -I -B iterate.py v1 v2 [v3]   → 結果附加到 iterations.json（每版一筆）。
只用 SEL 殘差與人為指定的真值，不碰主策略的事後結果。"""
import sys, json, time, math
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

TRUTHS = {'T1': [(1.0, -0.3, 0.6)], 'T2': [(0.75, -0.6, 0.7), (0.25, 0.4, 0.25)], 'T3': [(0.5, 0.0, 0.05), (0.5, -0.2, 0.8)]}
VERSIONS = {'v1': {'mode': 'naive', 'se_infl': 1.0}, 'v2': {'mode': 'decomp', 'se_infl': 1.0},
            'v3': {'mode': 'decomp', 'se_infl': 1.0, 'g2_c0': 0.1, 'g2_knots': 'grid',
                   'note': 'v1／v2 之後的事後定義：v2 的共同／個別拆解不變；G2 懲罰 c0 由 1 改為 0.1。'
                           '理由：v1、v2 的 G2 涵蓋率合格，但族群最大者仍高估 0.29～0.45pp（原始偏誤的 40～55%）。'
                           '診斷（合成資料 T2、40 次、種子 5，未碰主策略）：c0＝1 時 ‖β‖≈40，懲罰約 40 nats，把 g 拉向 [−6,6] 均勻分布，右尾過胖；'
                           'c0＝1/0.1/0.01 × 節點 grid/data 六組的最大者偏誤 0.21/0.19、0.03/0.09、−0.07/−0.05（原始 0.57），涵蓋率 0.81～0.83。'
                           '另有一次 30 次的冒煙（節點改 α̂ 分位、c0＝1）偏誤 0.36，捨棄。選 c0＝0.1、節點不改（deconvolveR 基底）。'}}

def draw_truth(name, n, rng):
    comps = TRUTHS[name]
    w = np.array([c[0] for c in comps]); k = rng.choice(len(comps), size=n, p=w / w.sum())
    mu = np.array([c[1] for c in comps])[k]; sd = np.array([c[2] for c in comps])[k]
    return mu + sd * rng.standard_normal(n)

def base():
    X, M = S.load_family()
    lab = pd.read_csv(S.DATA / 'clusters_k161.csv')
    med = lab[lab.is_medoid].member_id.tolist()
    F = S.load_focus()
    sel = S.wmask(X.index, *S.SEL)
    E = X.loc[sel, med].values
    ef = ((F['AB'] - F['r0050'])[sel].values * 100)
    E = E - E.mean(axis=0); ef = ef - ef.mean()
    return E, ef

def one_task(args):
    ver, model, truth, seed, reps = args
    E, ef = base()
    n, K = E.shape
    cfg = VERSIONS[ver]
    S.G2_KNOTS = cfg.get('g2_knots', 'grid'); S.G2_C0 = cfg.get('g2_c0', 1.0)
    rng = np.random.default_rng(seed)
    idxs = S.sb_index(n, n, 6, reps, seed + 1)
    cov, covf, bias_raw, bias_post, width = [], [], [], [], []
    for r in range(reps):
        a = draw_truth(truth, K, rng); af = draw_truth(truth, 1, rng)[0]
        Es = E[idxs[r]]; efs = ef[idxs[r]]
        D = a[None, :] + Es
        dfo = af + efs
        dec = S.decompose(D, {'f': dfo}, mode=cfg['mode'])
        s2 = dec['s2'] * cfg['se_infl'] ** 2
        pr = S.fit_prior(model, dec['xhat'], s2)
        m, lo, hi = S.post_all(pr, model, dec['xhat'], s2, dec['vP'])
        cov.append(float(((a >= lo) & (a <= hi)).mean()))
        width.append(float((hi - lo).mean()))
        j = int(np.argmax(dec['xhat']))
        bias_raw.append(float(dec['xhat'][j] - a[j])); bias_post.append(float(m[j] - a[j]))
        pf = S.posterior(pr, model, dec['focus_xhat']['f'], dec['focus_s2']['f'] * cfg['se_infl'] ** 2, dec['vP'])
        covf.append(float(pf['q10'] <= af <= pf['q90']))
    br, bp = np.array(bias_raw), np.array(bias_post)
    return {'version': ver, 'model': model, 'truth': truth, 'seed': seed, 'reps': reps,
            'coverage80': float(np.mean(cov)), 'coverage80_sd_rep': float(np.std(cov)), 'mean_width80_pp': float(np.mean(width)),
            'focus_coverage80': float(np.mean(covf)),
            'max_bias_raw_pp': float(br.mean()), 'max_bias_post_pp': float(bp.mean()),
            'max_bias_post_t': float(bp.mean() / (bp.std(ddof=1) / math.sqrt(len(bp)))),
            'max_bias_ratio': float(bp.mean() / br.mean())}

def judge(rows):
    g2 = [r for r in rows if r['model'] == 'G2']
    ok_cov = all(0.75 <= r['coverage80'] <= 0.85 for r in g2)
    ok_tw = all((r['max_bias_post_pp'] <= 0.25 * r['max_bias_raw_pp']) and (r['max_bias_post_t'] < 2) for r in g2)
    stab = {}
    for t in TRUTHS:
        rr = [r for r in g2 if r['truth'] == t]
        stab[t] = {'cov_spread': float(max(r['coverage80'] for r in rr) - min(r['coverage80'] for r in rr)),
                   'bias_spread': float(max(r['max_bias_post_pp'] for r in rr) - min(r['max_bias_post_pp'] for r in rr))}
    ok_st = all(v['cov_spread'] <= 0.03 and v['bias_spread'] <= 0.05 for v in stab.values())
    return {'G2_coverage_pass': ok_cov, 'G2_tweedie_pass': ok_tw, 'G2_seed_stability_pass': ok_st, 'stability': stab,
            'all_pass': ok_cov and ok_tw and ok_st}

def main():
    vers = sys.argv[1:] or ['v1', 'v2']
    tasks = []
    for ver in vers:
        for t in TRUTHS:
            for sd in (101, 202, 303):
                tasks.append((ver, 'G2', t, sd, 150))
            tasks.append((ver, 'G1', t, 101, 100))
            tasks.append((ver, 'G3', t, 101, 100))
    t0 = time.time()
    with Pool(4) as p:
        rows = p.map(one_task, tasks, chunksize=1)
    f = HERE / 'iterations.json'
    it = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {'versions': []}
    for ver in vers:
        rr = [r for r in rows if r['version'] == ver]
        it['versions'] = [v for v in it['versions'] if v['version'] != ver]
        it['versions'].append({'version': ver, 'config': VERSIONS[ver], 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                               'rows': rr, 'judge': judge(rr)})
    f.write_text(json.dumps(it, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in rows:
        print(r['version'], r['model'], r['truth'], r['seed'], 'cov %.3f fcov %.3f w %.2f braw %.3f bpost %.3f (t %.2f)' % (
            r['coverage80'], r['focus_coverage80'], r['mean_width80_pp'], r['max_bias_raw_pp'], r['max_bias_post_pp'], r['max_bias_post_t']))
    for ver in vers:
        print(ver, judge([r for r in rows if r['version'] == ver]))
    print('sec', time.time() - t0)

if __name__ == '__main__':
    main()

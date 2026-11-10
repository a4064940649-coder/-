# 凍結後描述（不改凍結程式，只呼叫）：(1) 30 萬整股／100 萬投入比率；(2) 抽樣區間（月份 block bootstrap、±1.96SE 年化）
import json, pickle, numpy as np, pandas as pd, sys
import xdata, xengine, xstats
HERE = '/home/user/-/research/tw_stock_selection/agents/o_g6_v7x_cleanroom_rewrite/'
KD = pickle.load(open(xdata.SCR.replace('g6A/', 'g6K_lists/') + 'k_all_lists.pkl', 'rb'))
W = xengine.World(); dates = W.dates
R = json.load(open(HERE + 'results.json'))['RESULTS']
out = {}
def conv(pl):
    return [(int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(s)))), int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(e)))), [W.ci[c] for c in pk]) for s, e, pk in pl]
class InvSim(xengine.Sim):
    def equity(self, t):
        v = super().equity(t)
        if not hasattr(self, 'inv'): self.inv = {}
        stk = sum(s * self.W.CF[t, j] for j, s in self.S.items() if s)
        self.inv[t] = stk / v if v > 0 else np.nan
        return v
cut = xstats.last_le(dates, '2020-10-25'); a21 = xstats.last_le(dates, '2020-12-31')
inv = {}
for cn, cfg in [('W30', dict(tier=300_000, whole_pre2020=True)), ('O30', dict(tier=300_000)), ('N100', dict(tier=1_000_000))]:
    vals_pre, vals_post = [], []
    for j in range(13):
        for sd in (0, 5, 10, 15):
            p = conv(KD['variants']['REAL_opt'][(j, sd)])
            s = InvSim(W, p, len(dates) - 1, dict(cfg, merger='cons_main')); s.run()
            iv = pd.Series(s.inv); vals_pre.append(iv[iv.index <= cut].mean()); vals_post.append(iv[iv.index > a21].mean())
    inv[cn] = {'invested_ratio_2006_2020_10': float(np.mean(vals_pre)), 'invested_ratio_2021_on': float(np.mean(vals_post)), 'n_sims': len(vals_pre)}
out['invested_ratio_REAL_opt_13j_x_4seeds'] = inv
# 抽樣區間：每格取 j 中位數那個 j 的訊號窗超額
rng = np.random.default_rng(2026)
def boot(ex, L=12, B=5000):
    ex = np.asarray(ex); n = len(ex); m = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + L) % n for s in rng.integers(0, n, int(np.ceil(n / L)))])[:n]
        m.append(ex[idx].mean())
    return np.percentile(m, [2.5, 97.5])
SI = {}
for c in R['cells']:
    if not c.startswith('E|REAL') and c != 'E|orig|orig|N100': continue
    pj = R['cells'][c]['per_j']; exs = {int(j): v['exc'] for j, v in pj.items()}
    med = np.median(list(exs.values())); jm = min(exs, key=lambda j: abs(exs[j] - med))
    ex = np.array(R['cells'][c]['win'][str(jm)]['ex']); se = ex.std(ddof=1) / np.sqrt(len(ex))
    lo, hi = boot(ex)
    SI[c] = {'j_used': jm, 'mean_win_ex_pp': 100 * ex.mean(), 'pm196se_annualized_pp': [100 * 12 * (ex.mean() - 1.96 * se), 100 * 12 * (ex.mean() + 1.96 * se)],
             'block12_boot95_annualized_pp': [100 * 12 * lo, 100 * 12 * hi]}
    st = np.array(R['cells'][c]['win'][str(jm)]['start'], dtype='datetime64[D]')
    e19 = ex[st >= np.datetime64('2019-01-01')]; se19 = e19.std(ddof=1) / np.sqrt(len(e19))
    SI[c]['y19_pm196se_annualized_pp'] = [100 * 12 * (e19.mean() - 1.96 * se19), 100 * 12 * (e19.mean() + 1.96 * se19)]
out['sampling_intervals'] = SI
json.dump(out, open(HERE + 'posthoc.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, default=float)[:3000])

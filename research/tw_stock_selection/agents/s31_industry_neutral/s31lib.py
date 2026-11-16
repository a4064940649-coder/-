#!/usr/bin/env python3
"""S31 共用程式庫：產業中性化——SURGE 改「產業內排名」或「限制單一產業檔數」，能不能更好、更分散？
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎）之上，本檔只新增：
- 開除權息修正：cslib.EXRIGHT_FIX = True（必須在 load() 之前；現行驗收 2019 起 48.99%、全期 34.54%、回撤 −32.11%、已平倉 381 筆）。
- PIT 產業：直接 import s21lib 的 industry_matrix()（讀 o_lc_sector_rotation_pit/raw/industry_pit.csv，只讀；訊號日當時有效段）。
- score_row()：每個訊號列的排名分數（產業內百分位、扣產業中位數、全市場與產業內百分位平均）。
- build_rule()：
  kind='score'：用新分數取代 SURGE 做「L_MA 排序」——進場＝新排序前 10 名中 RET20 > 10%（最多 10 檔），
                出場 X＝跌出新排序前 20 名（與現行 E0 同一條規則，只是排序換成新分數；進出場一致）。
  kind='cap'  ：現行 F1，但同一 PIT 產業最多 c 檔；多的依 SURGE 由 L_MA 下一名遞補（只到第 20 名，遞補也要 RET20 > 10% 且該產業未滿 c 檔）；
                出場 E0 完全不動（遞補只在前 20 名內，新買的股票照同一條規則出場）。
- holdings_div()：持股分散度（每日持有檔數依 PIT 產業分組：最大產業占比、產業數）。
中間檔放 scratchpad/s31/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S31 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s31')
sys.path.insert(0, str(AG / 's21_industry_resonance'))
import s21lib                                              # noqa: E402  只借 industry_matrix／load_industry_pit（不呼叫 s21lib.load）
from s21lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s21lib.S21 = S31; s20lib.S20 = S31; s17lib.S17 = S31; s10lib.S10 = S31; s9lib.S9 = S31; s8lib.S8 = S31; s7lib.S7 = S31; s6lib.S6 = S31
s5lib.S5 = S31; s4lib.S4 = S31; s3lib.S3 = S31; s2lib.S2 = S31; stlib.S1 = S31; cslib.TMP = S31
cslib.EXRIGHT_FIX = True

TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
SEGMENTS = s17lib.SEGMENTS
fresh_from = s20lib.fresh_from
industry_matrix = s21lib.industry_matrix
load_industry_pit = s21lib.load_industry_pit
MIN_PEERS = 5                                              # 同業（不含自己）≥ 5 家才用產業內統計；不足者用全市場
TIE = 1e-9                                                 # 同分時依 SURGE

RULES = [
    dict(id='IND_PCT', kind='score', score='ind_pct',
         desc='產業內百分位：訊號列股池（只上市、流動性、非創新板、SURGE 有值）依當時 PIT 產業分組，SURGE 在同業內的百分位（平均名次 ÷ 家數）；'
              '同業（不含自己）< 5 家或產業缺值者用全股池百分位。L_MA 依此分數排序（同分依 SURGE），前 10 名中 RET20 > 10% 進場；跌出前 20 名出場。'),
    dict(id='DEMED', kind='score', score='demed',
         desc='扣同業中位數：分數＝SURGE − 同業（含自己，同一股池）SURGE 中位數；同業（不含自己）< 5 家或缺值者扣全股池中位數。排序、進場、出場同 IND_PCT。'),
    dict(id='MIX', kind='score', score='mix',
         desc='部分中性化：分數＝0.5 × 全股池百分位＋0.5 × 產業內百分位（產業內百分位定義同 IND_PCT）。排序、進場、出場同 IND_PCT。'),
    dict(id='CAP2', kind='cap', cap=2,
         desc='限檔：現行 F1，但單一 PIT 產業最多 2 檔（依 SURGE 名次保留前 2 檔）；剔除幾檔就從 L_MA 第 11～20 名依 SURGE 遞補幾檔，'
              '遞補須 RET20 > 10% 且該產業未滿 2 檔；出場 E0 不動。產業缺值者不受限。'),
]


def load():
    S31.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True, '除權息修正沒有開'
    return P


def surge(P):
    F = s17lib.rev_family(P, defs=['STD'])
    return dict(S=F['S']['STD'], av=F['av'])


def _grp_pct(v, g):
    """v：數值；g：群組代碼（同長度）。回傳群組內百分位（平均名次 ÷ 家數）。"""
    return pd.Series(v).groupby(g).rank(method='average', pct=True).to_numpy()


def score_row(s, fin, ind, kind):
    """單一訊號列的分數（只在 fin＝股池且 SURGE 有值 的格子有值；其餘 NaN）。kind：'identity'／'ind_pct'／'demed'／'mix'。"""
    N = len(s); out = np.full(N, np.nan)
    jj = np.flatnonzero(fin)
    if not len(jj):
        return out
    v = s[jj]; g = ind[jj].copy()
    if kind == 'identity':
        out[jj] = v; return out
    ng = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    use_ind = (g >= 0) & (ng - 1 >= MIN_PEERS)
    gp = _grp_pct(v, np.zeros(len(v), int))                # 全股池百分位
    ip = np.where(use_ind, _grp_pct(v, np.where(use_ind, g, -1)), gp)
    if kind == 'ind_pct':
        out[jj] = ip
    elif kind == 'mix':
        out[jj] = 0.5 * gp + 0.5 * ip
    elif kind == 'demed':
        med_all = float(np.median(v))
        med_g = pd.Series(v).groupby(g).transform('median').to_numpy()
        out[jj] = v - np.where(use_ind, med_g, med_all)
    else:
        raise ValueError(kind)
    return out


def row_order(P, U, s, k, sc):
    """L_MA（股池＋SURGE 有值＋還原收盤 > 60 日均線）依 key＝sc＋TIE×s 由大到小排序（穩定）。回傳 fin、o_ma、key。"""
    fin = U[k] & np.isfinite(s)
    ok_ma = fin & P.F_MA60[k]
    key = sc + TIE * np.nan_to_num(s)
    idx = np.flatnonzero(ok_ma)
    o_ma = idx[np.argsort(-key[idx], kind='stable')]
    return fin, o_ma, key


def build_rule(P, U, RT, S0, IND, rule, check_f1=True):
    """S0＝make_signal_s10（同一 shift）。rule＝None（現行）或 dict(kind='score'|'cap', ...)。IND＝{k: 產業代碼陣列}。
    回傳 S1（sig、rank、X、XV['E0']、info 皆依規則）與 log（每列 month、k、orig、new）。"""
    if rule is None:
        return dict(S0), []
    S = RT['S']
    sig = np.zeros_like(S0['sig']); rank = np.full_like(S0['rank'], np.nan)
    X = S0['XV']['E0'].copy()
    log = []; info = []
    for x in S0['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        s = S.loc[m].to_numpy()
        fin, o_ma, key0 = row_order(P, U, s, k, s)
        r20 = P.RET20[k]
        tgt0 = o_ma[:10][r20[o_ma[:10]] > 0.10][:10]
        if check_f1:
            assert [str(P.COLS[j]) for j in tgt0] == x['targets'], ('重現 F1 失敗', x['month'], k)
        ind = IND[k]
        if rule['kind'] == 'score':
            sc = score_row(s, fin, ind, rule['score'])
            _, o2, key = row_order(P, U, s, k, sc)
            new = [int(j) for j in o2[:10] if r20[j] > 0.10][:10]
            X[k, :] = True; X[k, o2[:20]] = False
            for j in new:
                sig[k, j] = True; rank[k, j] = key[j]
        elif rule['kind'] == 'cap':
            c = rule['cap']; cnt = {}; new = []
            for j in tgt0:
                g = int(ind[j])
                if g >= 0 and cnt.get(g, 0) >= c:
                    continue
                new.append(int(j)); cnt[g] = cnt.get(g, 0) + 1
            nrem = len(tgt0) - len(new); nadd = 0
            for j in o_ma[10:20]:
                if nadd >= nrem:
                    break
                g = int(ind[j])
                if not (r20[j] > 0.10) or (g >= 0 and cnt.get(g, 0) >= c):
                    continue
                new.append(int(j)); cnt[g] = cnt.get(g, 0) + 1; nadd += 1
            for j in new:
                sig[k, j] = True; rank[k, j] = s[j]
        else:
            raise ValueError(rule['kind'])
        log.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt0], new=[str(P.COLS[j]) for j in new],
                        orig_ind=[int(ind[j]) for j in tgt0], new_ind=[int(ind[j]) for j in new]))
        y = dict(x); y['targets'] = [str(P.COLS[j]) for j in new]; info.append(y)
    S1 = dict(S0); S1['sig'] = sig; S1['rank'] = rank; S1['X'] = X; S1['XV'] = dict(S0['XV']); S1['XV']['E0'] = X; S1['info'] = info
    return S1, log


def permute_ind(IND, U, RT, S0, seed):
    """隨機對照：每個訊號列把股池內（SURGE 有值）股票的產業代碼隨機打散（各產業家數不變）。"""
    rng = np.random.default_rng(seed); S = RT['S']; out = {}
    for x in S0['info']:
        k = x['k']; s = S.loc[pd.Period(x['month'], 'M')].to_numpy()
        a = IND[k].copy(); jj = np.flatnonzero(U[k] & np.isfinite(s) & (a >= 0))
        a[jj] = a[jj][rng.permutation(len(jj))]
        out[k] = a
    return out


class IndLookup:
    """股票代碼 × 日期 → PIT 產業名稱（同 s21lib：start ≤ 日 < end）。"""
    def __init__(self):
        df = load_industry_pit()
        self.rows = {c: g[['start', 'end', 'industry']].to_numpy() for c, g in df.groupby('code')}

    def __call__(self, code, d):
        for s, e, ind in self.rows.get(str(code), ()):
            if s <= d < e:
                return ind if ind not in s21lib.MISSING else None
        return None


def holdings_div(P, lg, a, b, IL):
    """每日持股（已進場、尚未出場；出場日當天開盤已賣不算）依進場訊號日 PIT 產業分組（檔數，近似等權）。
    回傳：最大產業占比的日平均、最大產業檔數日平均、產業數日平均、「同產業 ≥ 3 檔」天數比例、最大產業占比日最大值。只算持有 ≥ 1 檔的交易日。"""
    i0 = int(np.searchsorted(P.IDX.values, pd.Timestamp(a).to_datetime64())); i1 = int(np.searchsorted(P.IDX.values, pd.Timestamp(b).to_datetime64(), side='right')) - 1
    n = i1 - i0 + 1
    if n <= 0 or not len(lg):
        return {}
    inds = [IL(c, d) or f'NA_{c}' for c, d in zip(lg['code'], pd.to_datetime(lg['sig_date']))]
    names = sorted(set(inds)); nid = {x: i for i, x in enumerate(names)}
    M = np.zeros((n, len(names)), np.int16)
    for g, e, x, w in zip(inds, lg['ent_t'].to_numpy(), lg['exit_t'].to_numpy(), lg['why'].to_numpy()):
        hi = x if w == 'open_end' else x - 1
        lo = max(e, i0); hi = min(hi, i1)
        if hi >= lo:
            M[lo - i0:hi - i0 + 1, nid[g]] += 1
    tot = M.sum(1); ok = tot > 0
    mx = M.max(1)
    return dict(days=int(ok.sum()), max_share_mean=float((mx[ok] / tot[ok]).mean()), max_count_mean=float(mx[ok].mean()),
                n_ind_mean=float((M[ok] > 0).sum(1).mean()), npos_mean=float(tot[ok].mean()), frac_days_ge3=float((mx[ok] >= 3).mean()),
                max_share_max=float((mx[ok] / tot[ok]).max()))


def list_div(log_or_info, IND_names=None):
    """名單層：每月買進名單的最大同產業檔數（平均）與「同產業 ≥ 3 檔」月份比例。log 每列需有 new_ind／orig_ind。"""
    out = {}
    for key in ('orig_ind', 'new_ind'):
        mx = []
        for x in log_or_info:
            v = [g for g in x[key] if g >= 0]
            mx.append(max(np.bincount(v)) if v else 0)
        mx = np.array(mx)
        out[key] = dict(max_same_mean=float(mx.mean()) if len(mx) else None, frac_months_ge3=float((mx >= 3).mean()) if len(mx) else None)
    return out

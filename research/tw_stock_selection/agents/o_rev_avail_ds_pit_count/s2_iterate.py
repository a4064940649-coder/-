#!/usr/bin/env python3
"""第 2 步：分類規則迭代（V1～V5），只用 D<S 訊號日的 C0 股池當安慰劑校準（期限已過，理論上幾乎全部可得；判成 UNAVAIL 的只應是逾期申報）。
不碰任何 D=S／D>S 槽位、不看名次變化、不看報酬。輸出 iterations.json。python3 -I -B s2_iterate.py"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import avail as A
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o17_revavail')
T0 = time.time()
DD = pd.read_csv(HERE / 'data/census_days.csv', dtype=str)
PL = pd.read_csv(SCR / 'c0_pool.csv.gz', dtype={'code': str, 'sig': str}).merge(DD[['sig', 'case', 'rev_month', 'seg4', 'D_engine']], on='sig')
X = PL[(PL.case == 'D<S') & (PL.sig >= '2013-01-01')].copy()
EV = {'sig': A.Evidence('sig'), '15': A.Evidence('15')}
VERS = {'V1': ('sig', '字面規則（計畫文字直譯）：UNAVAIL＝首見 > S 24:00（不管前一張快照時點）；截止日用期限 D；來源不含新聞'),
        'V2': ('sig', 'V1＋新聞（鉅亨／PTT／ELD／BOF，嚴格有效位數數值閘門）：只會把「查不到」改成「確定可得」或「未定」'),
        'V3': ('sig', '修正 UNAVAIL：必須有一張 ≥ S 13:30 的快照確定該檔不在（T_prev ≥ S 13:30），或公告日 > S；首見 > S 24:00 但前一張 < S 13:30（週末／缺快照）改判 UNDET；首次加入檔案無前一張 → UNDET（首見 > S+4 日 → NONE）'),
        'V4': ('sig', 'V3＋截止日一律用訊號日 S（D>S 日字面用期限 D 會把 S～D 之間公告的判成可得，前視）'),
        'V6': ('sig', 'V4，但快照的 UNAVAIL 改成「前一張快照日期 > S 仍不在」。理由：一致性閘門（Tasi 對 MOPS ann_date，2026-09 營收）顯示 Tasi 營收表落後公告日 1 天（1,337／1,457 筆公告日當晚快照還不在），所以 S 當晚不在不能證明 S 13:30 前沒公告。（此修訂在 V1～V5 跑完、看了閘門交叉表之後加入；沒有看任何 D=S 槽位）'),
        'V5': ('15', 'V4，但新聞數值閘門放寬到 ±15%（敏感度；不凍結，因為鉅亨多報合併營收，±15% 會混入錯月份或預估值）')}
res = {}
for v, (tol, why) in VERS.items():
    E = EV[tol]
    out = []
    for r in X.itertuples():
        s, det, sec = E.classify(pd.Timestamp(r.sig), r.code, r.rev_month, version=v, D=pd.Timestamp(r.D_engine))
        out.append((r.seg4, s))
    df = pd.DataFrame(out, columns=['seg4', 'state'])
    res[v] = {'why': why, 'news_tol': tol, 'placebo_DltS_pool_distribution': {sg: {k: round(float(x), 4) for k, x in g.state.value_counts(normalize=True).items()} | {'n': int(len(g))}
                                                                              for sg, g in df.groupby('seg4')}}
# 特別檢查：D<S 安慰劑中 V3/V4 的 UNAVAIL（應只剩逾期申報）——列前 20 筆證據
E = EV['sig']; bad = []
for r in X[X.seg4 == 'S2406'].itertuples():
    s, det, sec = E.classify(pd.Timestamp(r.sig), r.code, r.rev_month, version='V6')
    if s == 'UNAVAIL': bad.append(dict(sig=r.sig, code=r.code, ym=r.rev_month, det={k: v for k, v in det.items() if k != 'news'}, second=sec))
# 閘門交叉表：快照日期相對 MOPS 公告日（2026-09 營收）
FS = pd.read_csv(HERE / 'raw/ts_tasi_upstream_firstseen.csv.gz', dtype=str)
FS['Tt'] = A._tp(FS['T']); FS['Tp'] = A._tp(FS['T_prev'])
AN = pd.read_csv(HERE / 'raw/ts_mops_ann_date.csv', dtype=str); AN['a'] = pd.to_datetime(AN.ann_date)
GL = {}
for src in ('tasi', 'upstream'):
    m = FS[FS.source == src].merge(AN, on=['code', 'ym']); m = m[m.Tp.notna()]
    lt = (m.Tt.dt.normalize() - m.a).dt.days.clip(-1, 4); lp = (m.Tp.dt.normalize() - m.a).dt.days.clip(-3, 3)
    GL[src] = {'n': int(len(m)), 'crosstab_prevlag_x_firstlag': {f'{a}|{b}': int(v) for (a, b), v in pd.crosstab(lp, lt).stack().items() if v},
               'share_absent_on_ann_day_evening': round(float((lp == 0).mean()), 4), 'max_first_seen_lag_days': int(lt.max())}
it = {'utc': pd.Timestamp.now('UTC').isoformat(), 'saw_any_rank_or_DS_slot': False,
      'note': '迭代只修訂資料來源版本與分類規則；校準樣本＝D<S 訊號日（2013 起）C0 股池的時間戳分布（安慰劑）。沒有看任何 D=S／D>S 槽位、名次變化或報酬。截止時點（S 13:30）與門檻（> 0.05）從頭到尾沒有改。',
      'n_versions': len(VERS), 'versions': res,
      'placebo_V6_UNAVAIL_2024-06_examples': bad[:20], 'placebo_V6_UNAVAIL_2024-06_n': len(bad), 'gate_lag_tasi_vs_mops': GL,
      'chosen': 'V6', 'chosen_reason': 'V1～V3 的截止日用期限 D，在 D<S 安慰劑把期限日當晚才公告（合法）的股票判成「確定不可得」（V3 16.4%）；V4 改用訊號日 S；V6 再依 Tasi 落後 MOPS 1 天的閘門證據，把快照型 UNAVAIL 收緊到「S 之後的交易日快照仍不在」。V5 只作新聞閘門敏感度。',
      'news_valid': {'sig': [EV['sig'].n_news_valid, EV['sig'].n_news], '15': [EV['15'].n_news_valid, EV['15'].n_news]},
      'runtime_s': round(time.time() - T0, 1)}
json.dump(it, open(HERE / 'iterations.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: it[k] for k in ('versions', 'placebo_V6_UNAVAIL_2024-06_n', 'news_valid', 'gate_lag_tasi_vs_mops')}, ensure_ascii=False, indent=1, default=str))
for b in bad[:8]: print(b)

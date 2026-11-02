# 分母：o8 gate_population（2019-05～2026-09）＋自算 2026-10 股池，並以 2026-07～09 驗證自算口徑
import pandas as pd, numpy as np, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
SCR = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/'
adj = pd.read_parquet(P + 'adj.parquet'); cr = pd.read_parquet(P + 'close_raw.parquet')
op = pd.read_parquet(P + 'open.parquet'); val = pd.read_parquet(P + 'value.parquet')
cnt = cr.notna().sum(1); good = cnt[cnt >= 300].index
days = cr.index[cr.notna().sum(1) > 0]
sl = pd.read_csv(SCR + 'stock_list.csv', dtype=str)
fin = set(sl.loc[sl.industry == '金融保險業', 'code'])
kydr = set(sl.loc[sl.name.str.contains('KY|DR', na=False), 'code']) | {c for c in cr.columns if c.startswith('91')}
dsp = pd.concat([pd.read_csv(SCR + 'extras/disposal_5y.csv.gz', dtype=str), pd.read_csv(SCR + 'extras/warnings_hist.csv.gz', dtype=str)])
dsp = dsp[dsp.flag.str.contains('處置', na=False)]
def sig_day(m):
    p = pd.Period(m, 'M'); d0 = pd.Timestamp(p.year, p.month, 11)
    return days[days >= d0][0]
def pool(m):
    d = sig_day(m); nd = days[days > d][0]
    a = adj.loc[good]; a = a[a.index <= d]
    w52 = (a.iloc[-1] / a.rolling(252, min_periods=200).max().iloc[-1])
    v60 = val.loc[good][val.loc[good].index <= d].rolling(60, min_periods=40).mean().iloc[-1]
    v60 = v60.dropna().sort_values(ascending=False)
    top = list(v60.index[:300])
    ds = dsp[(dsp.start <= str(nd.date())) & (dsp.end >= str(d.date()))].code
    out = [c for c in top if cr.loc[d, c] >= 10 and pd.notna(w52.get(c)) and c not in set(ds)
           and pd.notna(op.loc[nd, c]) and c not in fin and c not in kydr]
    return d, out
o8 = pd.read_csv('/home/user/-/research/tw_stock_selection/agents/o_rev_freshness_attention/data/gate_population.csv.gz', dtype=str)
chk = {}
for m in ['2026-07', '2026-08', '2026-09']:
    d, me = pool(m); ref = set(o8.loc[o8.anchor == m, 'code'])
    chk[m] = dict(mine=len(me), o8=len(ref), both=len(ref & set(me)), only_mine=sorted(set(me) - ref), only_o8=sorted(ref - set(me)))
print(json.dumps(chk, ensure_ascii=False))
d, p10 = pool('2026-10')
print('2026-10', d.date(), len(p10))
den = o8[['anchor', 'rev_month', 'code']].copy()
den = pd.concat([den, pd.DataFrame(dict(anchor='2026-10', rev_month='2026-09', code=p10))], ignore_index=True)
den['sig_day'] = den.anchor.map(lambda m: str(sig_day(m).date()))
def deadline(m):
    p = pd.Period(m, 'M'); d0 = pd.Timestamp(p.year, p.month, 10)
    return str(days[days >= d0][0].date())
den['deadline'] = den.anchor.map(deadline)
den['kydr'] = den.code.isin(kydr)
den.to_csv(os.path.join(HERE, 'denominator.csv'), index=False)
print(len(den), den.anchor.nunique(), 'kydr', den.kydr.sum(), 'fin', den.code.isin(fin).sum())

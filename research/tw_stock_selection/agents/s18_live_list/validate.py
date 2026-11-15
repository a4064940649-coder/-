#!/usr/bin/env python3
"""S18 驗收第二步：make_list 對照 S10 E0 回測（validate_ref.py 先跑，產生 scratchpad/s18tmp/ref/ref_s10.json）。
V1：全部 129 個訊號月，訊號日、L_MA 前 20 名（順序）、SURGE、F1 名單（順序）逐檔相同。
V2：以回測當時的持股為「目前持股」，make_list 的 E0 賣出名單＝回測在該訊號日標記、次日起賣出的股票（逐檔相同）。
V3：make_list 的 F1 扣掉已持有、依名次補滿空位、跳過次日開盤漲停（回測規則）後，和回測實際買進逐檔比對。
V4：命令列端到端：2025-09、2026-08 用回測持股寫成 CSV 跑 make_list.py list，輸出 JSON 的 F1、賣出名單與 V1／V2 相同。
python3 -I -B validate.py → VALIDATION.json"""
import sys, json, time, resource, subprocess, hashlib
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s18tmp')
sys.path.insert(0, str(H / 'live_surge'))
import make_list as ML                                               # noqa: E402

t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
REF = json.loads((TMP / 'ref' / 'ref_s10.json').read_text())
tr = pd.DataFrame(REF['trades'])
sig_dates = sorted(m['sig_date'] for m in REF['months'])
D = ML.Data([ML.RAW_DEFAULT])
res_months = []
bad = dict(V1=[], V2=[], V3=[])
max_surge_diff = 0.0
n3 = dict(match=0, total=0, extra_explained=0)


def flag_sig(exit_date):
    """出場日之前最近的訊號日（回測在那天標記待賣）。"""
    prev = [s for s in sig_dates if s < exit_date]
    return prev[-1] if prev else None


for m in REF['months']:
    sm = m['signal_month']
    r = ML.compute(D, sm)
    rec = dict(signal_month=sm, sig_date_ref=m['sig_date'], sig_date=r.get('sig_date'), status=r['status'])
    ok1 = r['status'] == '有訊號' and r['sig_date'] == m['sig_date'] \
        and [x['code'] for x in r['top20']] == [x['code'] for x in m['top20']] and [x['code'] for x in r['f1']] == m['f1']
    if r['status'] == '有訊號':
        d = max((abs(a['surge'] - b['surge']) for a, b in zip(r['top20'], m['top20']) if a['code'] == b['code']), default=0.0)
        dr = max((abs(a['ret20'] - b['ret20']) for a, b in zip(r['top20'], m['top20']) if a['code'] == b['code']), default=0.0)
        max_surge_diff = max(max_surge_diff, d)
        rec.update(surge_maxdiff=d, ret20_maxdiff=dr, n_ma=r['n_ma'], n_ma_ref=m['n_ma'], f1=[x['code'] for x in r['f1']], f1_ref=m['f1'])
    rec['V1'] = bool(ok1)
    if not ok1:
        bad['V1'].append(sm)
    # V2：回測持股（訊號日收盤時）
    sd = m['sig_date']
    held = tr[(tr['ent_date'] <= sd) & (tr['exit_date'] > sd)]
    exp_sell = sorted(held.loc[(held['why'] == 'rule') & (held['exit_date'].map(flag_sig) == sd), 'code'])
    delist = sorted(held.loc[held['why'] == 'delist', 'code'])
    my_sell = []
    if r['status'] == '有訊號':
        for c in held['code']:
            st, _ = ML.held_status(r, c)
            if st['action'] == '賣出':
                my_sell.append(c)
    my_sell = sorted(my_sell)
    ok2 = my_sell == exp_sell
    rec.update(held=sorted(held['code']), sell=my_sell, sell_ref=exp_sell, V2=bool(ok2))
    if not ok2:
        bad['V2'].append(dict(month=sm, mine=my_sell, ref=exp_sell, delist_in_month=delist))
    # V3：買進
    ents = set(tr.loc[tr['sig_date'] == sd, 'code'])
    ent_ref = [c for c in m['f1'] if c in ents] + sorted(ents - set(m['f1']))      # 回測依 SURGE 名次買
    if r['status'] == '有訊號':
        keep = [c for c in held['code'] if c not in my_sell]
        free = ML.SLOTS - len(keep)
        ex = r['_ex']; ci = r['_ci']
        pred, skipped = [], []
        for c in [x['code'] for x in r['f1']]:
            if c in keep:
                continue
            if free <= 0:
                skipped.append((c, '已滿')); continue
            j = ci[c]
            o = ex['open'][j] if ex is not None else np.nan
            if not (np.isfinite(o) and o > 0):
                skipped.append((c, '無開盤價')); continue
            if o >= ex['limup'][j] - 1e-9:
                skipped.append((c, '開盤漲停')); continue
            pred.append(c); free -= 1
        ok3 = pred == ent_ref
        n3['total'] += 1; n3['match'] += int(ok3)
        rec.update(buy_pred=pred, buy_ref=ent_ref, buy_skipped=skipped, V3=bool(ok3))
        if not ok3:
            bad['V3'].append(dict(month=sm, pred=pred, ref=ent_ref, skipped=skipped))
    res_months.append(rec)
print('V1～V3 完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# V4：命令列端到端
v4 = {}
for sm in ('2025-09', '2026-08'):
    m = next(x for x in REF['months'] if x['signal_month'] == sm)
    sd = m['sig_date']
    held = tr[(tr['ent_date'] <= sd) & (tr['exit_date'] > sd)]
    hp = TMP / f'hold_{sm}.csv'
    pd.DataFrame({'代號': held['code'], '股數': held['shares']}).to_csv(hp, index=False)
    outp = TMP / f'v4_{sm}'
    cmd = [sys.executable, '-I', '-B', str(H / 'live_surge' / 'make_list.py'), 'list', '--month', sm, '--holdings', str(hp), '--cash', '0',
           '--out-prefix', str(outp), '--quiet']
    p = subprocess.run(cmd, capture_output=True, text=True)
    j = json.loads(Path(str(outp) + '.json').read_text())
    f1 = [x['code'] for x in j['f1']]
    sells = sorted(x['code'] for x in j['plan']['sells'])
    rec = next(x for x in res_months if x['signal_month'] == sm)
    v4[sm] = dict(returncode=p.returncode, sig_date=j['sig_date'], f1=f1, f1_ref=m['f1'], sells=sells, sells_ref=rec['sell_ref'],
                  keeps=sorted(x['code'] for x in j['plan']['keeps']), held=rec['held'],
                  buy_actions=[(b['code'], b['action'], b.get('shares')) for b in j['plan']['buys']],
                  top20=[x['code'] for x in j['top20']], top20_ref=[x['code'] for x in m['top20']],
                  ok=bool(p.returncode == 0 and f1 == m['f1'] and sells == rec['sell_ref'] and [x['code'] for x in j['top20']] == [x['code'] for x in m['top20']]))
print('V4', {k: v['ok'] for k, v in v4.items()}, flush=True)

# V3 差異分類：成交日在 2020-10-26（零股上路）之前時，一張的成本是否超過目標金額（目標≈ 回測淨值 × 70% ÷ 10，底倉在 60%～80% 間浮動）
navs = pd.read_csv(H.parent / 's10_exit_details' / 'navs.csv.gz', usecols=['date', 'E0|W10_70'], parse_dates=['date']).set_index('date')['E0|W10_70']
for f in bad['V3']:
    rec = next(x for x in res_months if x['signal_month'] == f['month'])
    r = ML.compute(D, f['month'])
    ex = r['_ex']; ci = r['_ci']
    nav = float(navs.loc[:pd.Timestamp(r['sig_date'])].iloc[-1])
    u = 0.7 * nav / 10
    f['exec_date'] = ex['date']; f['odd_lot_allowed'] = bool(pd.Timestamp(ex['date']) >= ML.ODD_DATE)
    f['missing'] = []
    for c in f['pred']:
        if c in f['ref']:
            continue
        o = float(ex['open'][ci[c]]); lot = 1000 * o * (1 + 0.5 * float(ML.tick(o)) / o + ML.XSLIP) * (1 + ML.FEE)
        f['missing'].append(dict(code=c, exec_open=o, one_lot_cost=round(lot), target_est=round(u), lot_over_target=round(lot / u, 3)))
    f['explained_by_whole_lot'] = bool(not f['odd_lot_allowed'] and all(x['lot_over_target'] > 1.0 for x in f['missing']) and all(c in f['pred'] for c in f['ref']))
n_post = sum(1 for x in res_months if x.get('V3') is not None and pd.Timestamp(x['sig_date_ref']) >= pd.Timestamp('2020-10-23'))
n_post_ok = sum(1 for x in res_months if x.get('V3') and pd.Timestamp(x['sig_date_ref']) >= pd.Timestamp('2020-10-23'))
summary = dict(
    ref=dict(source='s10lib.make_signal_s10 + s10lib.simulate_s10（E0｜W10_70，100 萬，嚴成本）', panel=REF['panel'], main=REF['main']),
    V1=dict(desc='訊號日、L_MA 前 20 名（含順序）、F1 名單（含順序）逐檔相同', n_months=len(res_months), n_ok=len(res_months) - len(bad['V1']), fail=bad['V1'],
            surge_max_abs_diff=max_surge_diff),
    V2=dict(desc='以回測持股為輸入，E0 賣出名單逐檔相同', n_months=len(res_months), n_ok=len(res_months) - len(bad['V2']), fail=bad['V2']),
    V3=dict(desc='F1 扣已持有、補滿空位、跳過次日開盤漲停後，和回測實際買進逐檔相同（金額規則造成的差異列在 fail）', n_months=n3['total'], n_ok=n3['match'],
            n_months_odd_lot_era=n_post, n_ok_odd_lot_era=n_post_ok, n_fail_explained_by_whole_lot=sum(1 for f in bad['V3'] if f['explained_by_whole_lot']), fail=bad['V3']),
    V4=v4,
    named_months={sm: next(x for x in res_months if x['signal_month'] == sm) for sm in ('2025-09', '2026-08')},
    make_list_sha256=hashlib.sha256((H / 'live_surge' / 'make_list.py').read_bytes()).hexdigest(),
    runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
summary['all_pass'] = bool(summary['V1']['n_ok'] == summary['V1']['n_months'] and summary['V2']['n_ok'] == summary['V2']['n_months'] and all(v['ok'] for v in v4.values()))
(H / 'VALIDATION.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
(TMP / 'validation_months.json').write_text(json.dumps(res_months, ensure_ascii=False, default=str))
print(json.dumps({k: (v if k in ('all_pass', 'runtime_s', 'maxrss_mb') else {a: b for a, b in v.items() if a in ('n_ok', 'n_months', 'surge_max_abs_diff')} if isinstance(v, dict) and 'n_ok' in v else '') for k, v in summary.items()}, ensure_ascii=False))
print('V3', {k: v for k, v in summary['V3'].items() if k != 'fail'}); print([ (f['month'], f['explained_by_whole_lot'], f['missing']) for f in bad['V3'] if not f['explained_by_whole_lot']])

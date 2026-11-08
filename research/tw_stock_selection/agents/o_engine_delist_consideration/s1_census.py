#!/usr/bin/env python3
"""步驟一（探索／普查，不看任何策略報酬）：掃 agents/*/ 下所有存檔名單，標出「持有期內最後成交日早於下一換股日」的持股。
python3 -I -B s1_census.py
輸出：census.json（本資料夾，小檔）、scratchpad/o17_delist/census_rows.parquet（逐筆）。
定義：
  名單＝(訊號日, [代碼…])；成交日＝檔內給的 exec（沒有就取訊號日後第一個面板交易日）；持有期＝[本期成交日, 下期成交日)。
  會被凍結＝下期成交日當天沒有開盤成交（與 ia_core.simulate 的 TRADED & isfinite(open) 相同）。
  永久（下市）＝同一證券（考慮代碼重用）之後再也沒有成交；否則為暫時停牌。
  兩種面板：U＝V7 面板＋下市價修補（最完整，判真實事件）；IA＝ia_core 用的 scratchpad/panel（紙上引擎看到的）。
  凍結資產占比＝等權進場、不計成本、凍結部位市值固定的簡化模擬（freeze 語意），報最大值與日期；只對有凍結事件的名單算。"""
import sys, json, gzip, pickle, re, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cz_lib as Z

T0 = time.time()
AG = Z.AG
SKIP_DIRS = {'raw', '.git', 'vendor_o7'}
MAXB = 20 * 2 ** 20
DRE = re.compile(r'^\d{4}-\d{2}-\d{2}')
CRE = re.compile(r'^[0-9][0-9A-Z]{3,5}$')


def is_codes(x):
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(c, str) and CRE.match(c) for c in x)


def walk(o, path, out):
    if isinstance(o, dict):
        ks = list(o.keys())
        dk = [k for k in ks if isinstance(k, str) and DRE.match(k)]
        if len(dk) >= 3 and sum(is_codes(o[k]) for k in dk) >= 0.8 * len(dk):
            out[path] = [(k[:10], None, list(o[k])) for k in dk if is_codes(o[k])]
            return
        if 'sig_days' in o and 'plans' in o and isinstance(o['plans'], list):
            sd = o['sig_days']; pl = o['plans']; rows = []
            for i, p in enumerate(pl):
                if is_codes(p) and i < len(sd): rows.append((str(sd[i])[:10], None, list(p)))
                elif isinstance(p, dict):
                    cc = p.get('picks') or p.get('codes')
                    s = p.get('sig') or p.get('signal') or (sd[i] if i < len(sd) else None)
                    if is_codes(cc) and s: rows.append((str(s)[:10], p.get('exec'), list(cc)))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    cc = [x for x in p if is_codes(x)]
                    if cc and i < len(sd): rows.append((str(sd[i])[:10], None, list(cc[0])))
            if len(rows) >= 3: out[path] = rows; return
        for k in ks:
            walk(o[k], f'{path}/{k}', out)
    elif isinstance(o, (list, tuple)):
        if len(o) >= 3 and all(isinstance(x, dict) for x in o[:5]):
            x0 = o[0]
            dkey = next((k for k in ('sig', 'signal', 'date', 'd', 'signal_date', 'sig_day') if k in x0), None)
            ckey = next((k for k in ('codes', 'picks', 'names', 'list', 'hold') if k in x0), None)
            if dkey and ckey:
                rows = [(str(x[dkey])[:10], x.get('exec'), list(x[ckey])) for x in o if isinstance(x, dict) and is_codes(x.get(ckey)) and DRE.match(str(x.get(dkey)))]
                if len(rows) >= 3: out[path] = rows; return
        if len(o) >= 3 and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in o[:5]):
            rows = []
            for x in o:
                d = next((str(v)[:10] for v in x if isinstance(v, (str, pd.Timestamp)) and DRE.match(str(v))), None)
                cc = next((v for v in x if is_codes(v)), None)
                if d and cc: rows.append((d, None, list(cc)))
            if len(rows) >= 3: out[path] = rows; return
        if len(o) < 2000:
            for i, x in enumerate(o[:2000]):
                if isinstance(x, (dict, list, tuple)): walk(x, f'{path}[{i}]', out)


def parse_table(df, name, out):
    cols = [c.lower() for c in df.columns]; df.columns = cols
    dcol = next((c for c in ('signal', 'signal_date', 'sig', 'date') if c in cols), None)
    if dcol is None: return
    keys = [c for c in ('strat', 'strategy', 'spec', 'j', 'seed', 'variant', 'n', 'tag', 'mode') if c in cols]
    if 'picks' in cols:
        g = df.groupby(keys, sort=False) if keys else [((), df)]
        for k, x in g:
            k = k if isinstance(k, tuple) else (k,)
            rows = [(str(s)[:10], (str(e)[:10] if 'exec' in cols else None), [c.strip() for c in str(p).split(',') if c.strip()])
                    for s, e, p in zip(x[dcol], x['exec'] if 'exec' in cols else [None] * len(x), x['picks'])]
            rows = [r for r in rows if DRE.match(r[0]) and r[2]]
            if len(rows) >= 3: out[f'{name}|' + '|'.join(map(str, k))] = rows
    elif 'code' in cols and 'weight' in cols:
        g = df.groupby(keys, sort=False) if keys else [((), df)]
        for k, x in g:
            k = k if isinstance(k, tuple) else (k,)
            rows = [(str(s)[:10], None, sorted(set(map(str, y['code'])))) for s, y in x.groupby(dcol)]
            if len(rows) >= 3: out[f'{name}|' + '|'.join(map(str, k))] = rows


def scan():
    files = []
    for p in sorted(AG.glob('*/**/*')):
        if not p.is_file(): continue
        rel = p.relative_to(AG)
        if any(s in SKIP_DIRS for s in rel.parts): continue
        if rel.parts[0] == 'o_engine_delist_consideration': continue
        if not (p.name.endswith(('.json', '.csv', '.csv.gz', '.tsv', '.pkl'))): continue
        files.append(p)
    res = {}; flog = []
    for p in files:
        rel = str(p.relative_to(AG)); sz = p.stat().st_size; out = {}
        if sz > (60 * 2 ** 20 if p.name.endswith(('.csv', '.csv.gz', '.tsv')) else MAXB):
            flog.append(dict(file=rel, size=sz, status='skipped_large')); continue
        try:
            if p.name.endswith('.json'):
                walk(json.loads(p.read_text(encoding='utf-8')), rel, out)
            elif p.name.endswith('.pkl'):
                with open(p, 'rb') as fh: walk(pickle.load(fh), rel, out)
            else:
                hdr = (gzip.open(p, 'rt') if p.name.endswith('.gz') else open(p, encoding='utf-8', errors='ignore')).readline().lower()
                if ('picks' in hdr or ('code' in hdr and 'weight' in hdr)) and any(k in hdr for k in ('signal', 'sig', 'date')):
                    df = pd.read_csv(p, sep='\t' if p.name.endswith('.tsv') else ',', dtype=str)
                    parse_table(df, rel, out)
            flog.append(dict(file=rel, size=sz, status='ok', n_results=len(out)))
        except Exception as e:
            flog.append(dict(file=rel, size=sz, status='error', err=repr(e)[:200]))
        res.update(out)
    return res, flog


def holdings(rows):
    """回傳 [(exec_i, exec_next, codes)]；同訊號日重複取最後一筆。"""
    d = {}
    for s, e, cc in rows:
        d[s] = (e, cc)
    ss = sorted(d)
    ex = []
    for s in ss:
        e = d[s][0]
        ei = Z.idx_of(e) if (e and DRE.match(str(e))) else Z.next_idx(s)
        ex.append(ei)
    out = []
    for i, s in enumerate(ss):
        e0 = ex[i]; e1 = ex[i + 1] if i + 1 < len(ss) else min(e0 + 25, len(Z.IDX) - 1)
        if e0 >= len(Z.IDX) or e1 <= e0: continue
        out.append((e0, e1, d[s][1], s))
    return out


def frozen_share(H, frzset):
    """簡化 freeze 模擬：等權、不計成本、日報酬 RU（缺值當 0）；凍結部位市值固定。"""
    val = {}; best = (0.0, None)
    reb = {h[0]: h for h in H}
    t0 = H[0][0]; t1 = H[-1][1]
    tot = 1.0
    for t in range(t0, t1 + 1):
        if t in reb:
            fz = {j: v for j, v in val.items() if not Z.TRU[t, j]}
            free = tot - sum(fz.values())
            cc = [Z.CI[c] for c in reb[t][2] if c in Z.CI]
            val = dict(fz)
            if cc:
                for j in cc: val[j] = val.get(j, 0.0) + free / len(cc)
            sh = sum(fz.values()) / tot if tot > 0 else 0
            if sh > best[0]: best = (sh, str(Z.IDX[t].date()))
        else:
            for j in list(val):
                if Z.TRU[t, j]:
                    r = Z.RU[t, j]; val[j] *= (1 + (r if np.isfinite(r) else 0.0))
        tot = sum(val.values()) if val else tot
    return best


def main():
    res, flog = scan()
    print('parsed results', len(res), 'files', len(flog), round(time.time() - T0), 's', flush=True)
    rows = []; per = {}
    for name, R in res.items():
        H = holdings(R)
        nslot = 0; nopanel = 0; fr = []
        for e0, e1, cc, s in H:
            for c in cc:
                nslot += 1
                j = Z.CI.get(c)
                if j is None: nopanel += 1; continue
                fu = not Z.TRU[e1, j]; fi = not Z.TRIA[e1, j]
                if not (fu or fi): continue
                ltu = Z.last_trade_before(j, e0, Z.TRU); lti = Z.last_trade_before(j, e0, Z.TRIA)
                fr.append(dict(result=name, signal=s, exec=str(Z.IDX[e0].date()), next_exec=str(Z.IDX[e1].date()), code=c,
                               frozen_U=bool(fu), frozen_IA=bool(fi),
                               perm_U=bool(fu and ltu < e1), perm_IA=bool(fi and lti < e1),
                               last_U=str(Z.IDX[ltu].date()) if ltu >= 0 else None, last_IA=str(Z.IDX[lti].date()) if lti >= 0 else None))
        rows += fr
        pu = [x for x in fr if x['perm_U']]; pi = [x for x in fr if x['perm_IA']]
        info = dict(n_lists=len(H), n_slots=nslot, n_slots_not_in_panel=nopanel,
                    first=H[0][3] if H else None, last=H[-1][3] if H else None,
                    frozen_slots_U=sum(x['frozen_U'] for x in fr), frozen_slots_IA=sum(x['frozen_IA'] for x in fr),
                    perm_slots_U=len(pu), perm_slots_IA=len(pi),
                    perm_events_U=sorted({(x['code'], x['last_U']) for x in pu}), perm_events_IA=sorted({(x['code'], x['last_IA']) for x in pi}),
                    price_gap_only_IA=sorted({(x['code'], x['last_IA']) for x in pi if not x['perm_U']}))
        if fr and H:
            sU = frozen_share(H, None)
            info['frozen_share_max_U'] = round(sU[0], 4); info['frozen_share_date_U'] = sU[1]
        per[name] = info
    df = pd.DataFrame(rows)
    df.to_parquet(Z.WORK / 'census_rows.parquet')
    # 彙總：依「結果（名單組）」與「檔案」
    affected = {k: v for k, v in per.items() if v['perm_slots_U'] or v['perm_slots_IA'] or v['frozen_slots_U']}
    ev_all = sorted({tuple(e) for v in per.values() for e in v['perm_events_U']})
    ev_ia = sorted({tuple(e) for v in per.values() for e in v['perm_events_IA']})
    gap = sorted({tuple(e) for v in per.values() for e in v['price_gap_only_IA']})
    out = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - T0, 1), no_returns=True,
               definition=__doc__, n_files_scanned=len(flog), n_files_ok=sum(f['status'] == 'ok' for f in flog),
               n_files_with_lists=sum(1 for f in flog if f.get('n_results')), n_results=len(res),
               files_skipped_large=[f['file'] for f in flog if f['status'] == 'skipped_large'],
               files_error=[(f['file'], f['err']) for f in flog if f['status'] == 'error'],
               n_results_affected=len(affected),
               distinct_perm_events_U=ev_all, n_distinct_perm_events_U=len(ev_all), n_issuers_U=len({e[0] for e in ev_all}),
               distinct_perm_events_IA=ev_ia, price_gap_only_IA_events=gap,
               per_result=per, n_patch_cells=Z.NPATCH)
    (Z.HERE / 'census.json').write_text(json.dumps(out, ensure_ascii=False, indent=0, default=str), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('n_files_scanned', 'n_files_with_lists', 'n_results', 'n_results_affected', 'n_distinct_perm_events_U', 'n_issuers_U')}, ensure_ascii=False))
    print('events U', ev_all)
    print('events IA', ev_ia)
    print('gap', gap)


if __name__ == '__main__':
    main()

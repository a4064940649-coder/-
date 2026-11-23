#!/usr/bin/env python3
"""試算：若「資料最後一天」就是訊號日，策略會買什麼、賣什麼（用最新一期已到齊的月營收）。
只是試算，不是名單；真正的名單只在每月訊號日收盤後產生（那天會改用新一期營收）。
用法：python3 -I -B xq_scripts/site/preview_today.py <營收月份對應的訊號月份 例 2026-11> <持股檔> <現金> <底倉股數> <輸出 json> --repo <倉庫> [--repo ...]
做法：沿用名單產生器 make_list.py 的 compute／plan，只把「訊號日」換成資料最後一天；不改 make_list.py。"""
import sys, json, importlib.util, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / 'research/tw_stock_selection/agents/s18_live_list/live_surge/make_list.py'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('month'); ap.add_argument('holdings'); ap.add_argument('cash', type=float); ap.add_argument('base_shares', type=float)
    ap.add_argument('out'); ap.add_argument('--repo', action='append', required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(ML.parent))
    spec = importlib.util.spec_from_file_location('make_list', ML); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    D = m.Data(a.repo)
    last = len(D.dates) - 1
    m.signal_day = lambda D_, month, deadline=False: (last, None, '')      # 訊號日＝資料最後一天
    hold = m.read_holdings(a.holdings)
    res = m.compute(D, a.month, exright=True, extra_codes=[c for c in hold if c != '0050'], deadline=True, reduce=True)
    asof = D.dates[-1]
    rows = []; meta = [{'label': '產生程式', 'value': 'xq_scripts/site/preview_today.py'}, {'label': '試算基準日', 'value': asof},
                       {'label': '營收月份', 'value': str(res.get('rev_month'))}, {'label': '狀態', 'value': str(res.get('status'))}]
    def pct(x):
        return None if x is None else round(float(x) * 100, 1)
    if res.get('status') == '有訊號':
        pl = m.plan(res, hold, total=None, cash=a.cash, base_shares=a.base_shares)
        for t in pl['sells']:
            rows.append({'基準日': asof, '動作': '試算賣出', '代號': t['code'], '名稱': t['name'], '名次': t.get('rank_ma'), 'SURGE': round(t['surge'], 2) if t.get('surge') is not None else None,
                         '前20日漲幅%': pct(t.get('ret20')), '距季線%': pct(t.get('dist_ma60')), '收盤': t.get('close'), '股數': t.get('shares'), '金額': round(t.get('value') or 0), '原因': t.get('reason', '')})
        for t in pl['keeps']:
            rows.append({'基準日': asof, '動作': '試算續抱', '代號': t['code'], '名稱': t['name'], '名次': t.get('rank_ma'), 'SURGE': round(t['surge'], 2) if t.get('surge') is not None else None,
                         '前20日漲幅%': pct(t.get('ret20')), '距季線%': pct(t.get('dist_ma60')), '收盤': t.get('close'), '股數': t.get('shares'), '金額': round(t.get('value') or 0), '原因': '仍在前 20 名且在季線上'})
        for t in pl['buys']:
            rows.append({'基準日': asof, '動作': '試算買進', '代號': t['code'], '名稱': t['name'], '名次': t.get('rank_ma'), 'SURGE': round(t['surge'], 2),
                         '前20日漲幅%': pct(t.get('ret20')), '距季線%': pct(t.get('dist_ma60')), '收盤': t.get('close'), '股數': t.get('shares'), '金額': round(t.get('est_value') or 0),
                         '原因': '前 10 名且前 20 日漲幅超過 10%' + (('；' + t['warn']) if t.get('warn') else '')})
        e = pl.get('e50') or {}
        rows.append({'基準日': asof, '動作': '0050', '代號': '0050', '名稱': '元大台灣50', '名次': None, 'SURGE': None, '前20日漲幅%': None, '距季線%': None, '收盤': e.get('price'),
                     '股數': e.get('shares_after'), '金額': round((e.get('shares_after') or 0) * (e.get('price') or 0)), '原因': str(e.get('action', ''))})
        meta += [{'label': '股池檔數', 'value': res.get('n_univ')}, {'label': '季線上且有分數的檔數', 'value': res.get('n_ma')},
                 {'label': '試算總資產', 'value': round(pl.get('total_in') or 0)}, {'label': '每檔目標金額', 'value': round(pl.get('unit') or 0)},
                 {'label': '前20名', 'value': '、'.join(f"{t['rank_ma']}.{t['code']}{t['name']}" for t in res['top20'])}]
    else:
        meta.append({'label': '原因', 'value': str(res.get('reason'))})
    meta.append({'label': '說明', 'value': '把資料最後一天當成訊號日、用最新一期已到齊的月營收試算。只是試算，不是名單：真正只在每月訊號日收盤後產生名單，那天會改用新一期營收重新排名，結果可能完全不同。'})
    Path(a.out).write_text(json.dumps({'rows': rows, 'meta': meta}, ensure_ascii=False, indent=1, default=str))
    for r in rows: print(r)

if __name__ == '__main__':
    main()

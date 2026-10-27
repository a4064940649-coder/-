#!/usr/bin/env python3
"""產生 main_picks_today.py 的 v2 建議修改：在 scratchpad 的複本上套用（不改原工具檔），輸出 unified diff 到 main_v2/main_picks_today_v2.patch。
用法：python3 -I -B make_tool_patch.py <複本路徑>"""
import sys, difflib
from pathlib import Path
HERE = Path(__file__).resolve().parent
ORIG = HERE.parents[1] / 'main_picks_today.py'
dst = Path(sys.argv[1])
s = ORIG.read_text(encoding='utf-8')
R = [
 # 1) 常數：拿掉 0.6% 容忍度，加跳動點漲停價與面板缺口日
 ("LIMIT, LTOL = 0.10, 0.006\n",
  "LIMIT = 0.10\n"
  "MIN_NOBS = 300                                                      # 有值檔數 < 300 的日子（2018-01-13）不參與滾動窗\n"
  "GAP_DAYS = pd.to_datetime(['2017-08-03', '2018-07-12'])             # 面板缺約 50 檔的兩天：窗內缺值豁免\n"
  "GAP_RANGE = (pd.Timestamp('2024-02-15'), pd.Timestamp('2024-03-15'))  # 來源 3 缺 98 檔的月份：窗內缺值豁免\n"
  "\n\n"
  "def tick(p):\n"
  "    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)\n"
  "\n\n"
  "def limit_up_price(ref, lim=LIMIT):\n"
  "    \"\"\"漲停價＝參考價×(1＋漲幅)，依該價位的跳動點無條件捨去（與 agents/independent_audit/ia_core.py 相同）。\"\"\"\n"
  "    raw = np.asarray(ref, float) * (1 + lim)\n"
  "    t = tick(raw)\n"
  "    return np.floor(raw / t + 1e-9) * t\n"
  "\n\n"
  "def _exempt(idx):\n"
  "    return idx.isin(GAP_DAYS) | ((idx >= GAP_RANGE[0]) & (idx <= GAP_RANGE[1]))\n"),
 # 2) 滾動窗 gapfix
 ("    cal = data['cal']; i = cal.get_loc(sd); cols = data['C'].columns\n"
  "    adj = data['ADJ'].iloc[i - W52 + 1:i + 1]\n"
  "    s52 = (adj.iloc[-1] / adj.max()).where(adj.notna().all()) if i >= W52 - 1 else pd.Series(np.nan, index=cols)\n"
  "    v = data['V'].iloc[i - W60 + 1:i + 1]\n"
  "    v60 = v.mean().where(v.notna().all()) if i >= W60 - 1 else pd.Series(np.nan, index=cols)\n",
  "    cal = data['cal']; i = cal.get_loc(sd); cols = data['C'].columns\n"
  "    # v2 gapfix：先剔除有值檔數 < 300 的日子再取 252／60 日窗；窗內缺值只豁免面板缺口日（GAP_DAYS、GAP_RANGE）\n"
  "    gcal = cal[(data['C'].notna().sum(axis=1) >= MIN_NOBS).to_numpy()]; gi = gcal.get_loc(sd)\n"
  "    adj = data['ADJ'].loc[gcal[max(gi - W52 + 1, 0):gi + 1]]\n"
  "    s52 = (adj.iloc[-1] / adj.max()).where(adj[~_exempt(adj.index)].notna().all()) if gi >= W52 - 1 else pd.Series(np.nan, index=cols)\n"
  "    v = data['V'].loc[gcal[max(gi - W60 + 1, 0):gi + 1]]\n"
  "    v60 = v.mean().where(v[~_exempt(v.index)].notna().all()) if gi >= W60 - 1 else pd.Series(np.nan, index=cols)\n"),
 # 3) 金融保險業定義與回測一致（另含 28xx、5876、5880）
 ("    if ex_fin: el = el & ~pd.Series(cols.map(lambda c: data['ind'].get(c, '') == FIN_IND), index=cols)\n",
  "    if ex_fin: el = el & ~pd.Series(cols.map(lambda c: data['ind'].get(c, '') == FIN_IND or c[:2] == '28' or c in ('5876', '5880')), index=cols)\n"),
 # 4) 同分次排序：營收 3 月年增率高者 → 代碼小者
 ("    order = df[df['el']].reset_index().sort_values(['score', 'code'], ascending=[False, True])\n",
  "    df['score_r'] = df['score'].round(10)\n"
  "    order = df[df['el']].reset_index().sort_values(['score_r', 'yoy3', 'code'], ascending=[False, False, True]).drop(columns='score_r')\n"),
 # 5) 開盤漲停：跳動點漲停價
 ("    with np.errstate(divide='ignore', invalid='ignore'):\n"
  "        gap = o * (a1 / c1) / a0 - 1\n",
  "    r1 = data['R'].loc[nd].reindex(order.index); prevc = data['C'].iloc[:i + 1].ffill().iloc[-1].reindex(order.index)\n"
  "    with np.errstate(divide='ignore', invalid='ignore'):\n"
  "        ref = (c1 / (1 + r1)).where(r1.notna(), prevc)          # 參考價（含除權息）；報酬缺值時用前收\n"
  "        gap = o / ref - 1\n"
  "    lu = pd.Series(limit_up_price(ref.to_numpy()), index=order.index)\n"),
 ("        if gap[code] >= LIMIT - LTOL and code not in held:\n"
  "            skipped.append((code, f'隔日開盤漲停（{gap[code]:+.1%}）')); continue\n",
  "        if np.isfinite(lu[code]) and o[code] >= lu[code] - 1e-6 and code not in held:\n"
  "            skipped.append((code, f'隔日開盤漲停（開 {o[code]:g} ≥ 漲停價 {lu[code]:g}）')); continue\n"),
 # 6) 說明文字
 ("綜合分數相同（{order.loc[picks[-1], \"score\"]:.4f}），依代號小者優先入選。'\n"
  "              '（回測程式 exlib 的同分順序取決於 numpy 排序，可能選另一檔；兩者擇一皆符合規則）')\n",
  "綜合分數相同（{order.loc[picks[-1], \"score\"]:.4f}），依營收 3 月年增率高者、再代號小者優先入選（v2 正式規則）。')\n"),
 ("  - 等權買前 N 名。隔日開盤漲停（還原開盤 ÷ 前收 − 1 ≥ 9.4%）而且還沒持有的不買，由下一名遞補；\n",
  "  - 同分時營收 3 月年增率高者優先、再代號小者（v2）。52 週／60 日窗只豁免面板缺口日的缺值（v2 gapfix）。\n"
  "  - 等權買前 N 名。隔日開盤 ≥ 依跳動點算出的漲停價（參考價×1.1 無條件捨去到跳動點）而且還沒持有的不買，由下一名遞補；\n"),
 ("隔日開盤若「開盤價 ÷ 今日收盤 − 1 ≥ 9.4%」（開盤漲停，除權息日請用參考價判斷）",
  "隔日開盤若「開盤價 ≥ 漲停價（參考價×1.1，依跳動點無條件捨去；除權息日用參考價）」"),
]
for a, b in R:
    assert s.count(a) == 1, ('找不到或不唯一', a[:60])
    s = s.replace(a, b)
assert 'LTOL' not in s
dst.write_text(s, encoding='utf-8')
diff = difflib.unified_diff(ORIG.read_text(encoding='utf-8').splitlines(True), s.splitlines(True), 'a/research/tw_stock_selection/main_picks_today.py', 'b/research/tw_stock_selection/main_picks_today.py')
(HERE / 'main_picks_today_v2.patch').write_text(''.join(diff), encoding='utf-8')
print('ok', dst)

#!/bin/bash
# 備援（基礎建設判不可用、daily_step0 回傳 2／3、或當天產生器失敗時）：手動跑 main_picks_today.py，輸出加時間戳與 sha256 存檔。
# 用法：bash fallback_manual.sh [資料目錄，預設本題 raw/screener/data]
# 2026-11-03 BF-4（bugfix_log.md）：main_picks_today 只讀 disposal_5y（2026-10-05 起停更），會漏排處置股（2026-10-12 的 2466、6213）。
#   另跑一份「處置聯集」版（disposal_5y ∪ warnings_hist 的處置記錄，和前瞻產生器同口徑）；下單與紀錄以聯集版為準。
# 2026-11-03 BF-5（bugfix_log.md）：第一步先抓官方月營收（rev_official_fetch.py → raw/official_rev/<時戳>/），
#   聯集版的 rev_hist 再補上「現在之前抓到」的官方月營收（只補上游缺的列）。daily_step0 回傳碼 4（月營收覆蓋率不足）時跑這支。
#   官方來源抓不到（網路政策擋下、代理 403 等）時照樣產生備援名單，但檔頭會寫「官方補值 0 列」，請到公開資訊觀測站人工核對。
set -u
F=/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live
TW=/home/user/-/research/tw_stock_selection
D=${1:-$F/raw/screener/data}
OUT=/home/user/-/research/tw_stock_selection/agents/live_tracking/main_forward/fallback
mkdir -p $OUT
TS=$(TZ=Asia/Taipei date +%Y%m%dT%H%M%S%z)
hdr() {
  echo "# wall_clock_taipei=$(TZ=Asia/Taipei date -Iseconds)"
  echo "# data_head=$(git -C $D/.. log -1 --format='%H %cI' 2>/dev/null)"
  echo "# main_picks_today.sha256=$(sha256sum $TW/main_picks_today.py | cut -d' ' -f1)"
}
# ---- 0) 官方月營收（BF-5）：每次一個新目錄，原始回應原樣存檔＋sha256；抓取完成時間＝可得時點 ----
OFF=$F/raw/official_rev
python3 -I -B $F/rev_official_fetch.py --out-root $OFF > $OUT/official_rev_fetch_${TS}.txt 2>&1
ORC=$?
sha256sum $OUT/official_rev_fetch_${TS}.txt >> $OUT/SHA256SUMS
if [ $ORC -ne 0 ]; then
  echo "【官方月營收抓取失敗】（rc=$ORC，詳見 $OUT/official_rev_fetch_${TS}.txt）：備援名單只用上游月營收；缺營收的股票請到公開資訊觀測站人工核對" >&2
fi
f=$OUT/main_picks_${TS}.txt
{
  hdr
  echo "# 資料：原始（處置只讀 disposal_5y，可能過期；以 main_picks_union_${TS}.txt 為準）"
  python3 -I $TW/main_picks_today.py --data $D --n 10 --ex-fin
} > $f 2>&1
sha256sum $f >> $OUT/SHA256SUMS
U=$F/work/mp_union
f2=$OUT/main_picks_union_${TS}.txt
{
  hdr
  echo "# 資料：同上，但 extras/disposal_5y 換成 disposal_5y ∪ warnings_hist（處置），rev_hist 補上官方月營收（check_official.py --make-union）"
  echo "# 官方月營收抓取 rc=$ORC（0＝成功；目錄 $OFF）"
  if python3 -I -B $F/check_official.py --make-union $D $U $OFF > /dev/null 2> $U.log; then
    sed 's/^/# /' $U.log; python3 -I $TW/main_picks_today.py --data $U --n 10 --ex-fin
  else
    sed 's/^/# /' $U.log; echo "【失敗】check_official.py --make-union 失敗，聯集版沒有產生名單"
  fi
} > $f2 2>&1
sha256sum $f2 >> $OUT/SHA256SUMS
rm -rf $U $U.log
python3 -I $TW/disposal_today.py --data $D --taiex $F/raw/finlab/taiex_twii.csv --tpex > $OUT/disposal_${TS}.txt 2>&1
sha256sum $OUT/disposal_${TS}.txt >> $OUT/SHA256SUMS
echo "寫入 $f"
echo "寫入 $f2（下單以這份為準；sha256 已記錄於 $OUT/SHA256SUMS）"
grep -A 13 -m1 -E '【(前 10 名|M2 實際買進名單)' $f2 || tail -20 $f2

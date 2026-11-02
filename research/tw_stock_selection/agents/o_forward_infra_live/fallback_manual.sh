#!/bin/bash
# 備援（基礎建設判不可用、daily_step0 回傳 2／3、或當天產生器失敗時）：手動跑 main_picks_today.py，輸出加時間戳與 sha256 存檔。
# 用法：bash fallback_manual.sh [資料目錄，預設本題 raw/screener/data]
# 2026-11-03 BF-4（bugfix_log.md）：main_picks_today 只讀 disposal_5y（2026-10-05 起停更），會漏排處置股（2026-10-12 的 2466、6213）。
#   另跑一份「處置聯集」版（disposal_5y ∪ warnings_hist 的處置記錄，和前瞻產生器同口徑）；下單與紀錄以聯集版為準。
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
  echo "# 資料：同上，但 extras/disposal_5y 換成 disposal_5y ∪ warnings_hist（處置）（check_official.py --make-union）"
  python3 -I -B $F/check_official.py --make-union $D $U >/dev/null && python3 -I $TW/main_picks_today.py --data $U --n 10 --ex-fin
} > $f2 2>&1
sha256sum $f2 >> $OUT/SHA256SUMS
rm -rf $U
python3 -I $TW/disposal_today.py --data $D --taiex $F/raw/finlab/taiex_twii.csv --tpex > $OUT/disposal_${TS}.txt 2>&1
sha256sum $OUT/disposal_${TS}.txt >> $OUT/SHA256SUMS
echo "寫入 $f"
echo "寫入 $f2（下單以這份為準；sha256 已記錄於 $OUT/SHA256SUMS）"
grep -A 13 -m1 -E '【(前 10 名|M2 實際買進名單)' $f2 || tail -20 $f2

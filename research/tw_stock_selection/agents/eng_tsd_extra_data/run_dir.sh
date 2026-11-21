#!/bin/bash
# 一次處理一個目錄：開臨時 clone → 抓該目錄 → 轉面板 → 刪臨時 clone。前後記錄磁碟。可用空間低於 800MB 就不做。
# 用法：run_dir.sh <臨時工作資料夾> <輸出資料夾> <來源目錄名> <輸出檔名> [起始日]
set -euo pipefail
W="$1"; OUT="$2"; D="$3"; NAME="$4"; START="${5:-0000}"
H="$(cd "$(dirname "$0")" && pwd)"; LOG="$H/disk_log.tsv"
avail(){ df -m --output=avail /home/user | tail -1 | tr -d ' '; }
A0=$(avail); if [ "$A0" -lt 800 ]; then echo "可用空間 ${A0}MB < 800MB，停止"; exit 3; fi
rm -rf "$W/c"; mkdir -p "$W"
git clone -q --filter=blob:none --depth=1 --no-checkout https://github.com/jb7k6295r7-bot/tw-stock-data "$W/c"
cd /tmp
python3 -I -B "$H/build_daily.py" "$W/c" "$OUT" "$D" "$NAME" "$START"
A1=$(avail); CS=$(du -sm "$W/c" | cut -f1)
rm -rf "$W/c"; A2=$(avail)
printf '%s\t%s\tbefore=%sMB\tpeak_clone=%sMB\tafter_build=%sMB\tafter_cleanup=%sMB\n' "$(date -Is)" "$D" "$A0" "$CS" "$A1" "$A2" | tee -a "$LOG"

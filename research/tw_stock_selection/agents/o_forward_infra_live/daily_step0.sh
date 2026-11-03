#!/bin/bash
# 每日流程第 0 步（o_forward_infra_live；PREREG_FORWARD 登記本檔 sha256；2026-11-03 bugfix BF-2、BF-3 見 bugfix_log.md）
# 排程：每天台北 06:49（既有每日排程開頭第一件事；週末也跑，週一前會補上週五的名單）。
#   訊號日（2026-11-10 J0、11-11 官方日）當晚 22:30 再加跑一次（上游約 21:55 提交）。
#   0) 磁碟檢查：剩餘 < 1GB 先清可重建快取（work/asof_*、work/check_official、work/mp_union），仍不足就中止（回傳碼 3）
#   1) 資料 git fetch（本題自建稀疏 clone，不動共用的 /home/user/stevenhsiehsheng-ctrl/tw-stock-screener）
#   2) make_forward.py（不帶 --asof）：依台北實際時鐘推算應有的最後交易日（18:00 後＝當天，否則＝前一交易日）；
#      來源資料不夠新 → 回傳碼 2、不寫名單（DATA_STALE）；資料到齊 → 訊號日名單在這一步凍結，時間早於成交日 09:00
#   3) 處置帳本每天更新（make_forward.py 內含）
#   BF-5（2026-11-03）：1b) 每月 9～20 日（台北）先抓官方月營收（rev_official_fetch.py → raw/official_rev/<時戳>/；
#      FWD_OFFICIAL_FETCH=1 強制抓、=0 不抓）；抓不到不中止。2) make_forward 帶 --official-rev：上游缺的月營收用官方值補，
#      並檢查 m* 月營收在股池的覆蓋率（J≥1 與官方日全檢；J0 只檢查官方已公告的部分）。
# 回傳碼：0＝正常；2＝資料過期（改跑 fallback_manual.sh）；3＝磁碟不足；
#   4＝月營收覆蓋率不足（官方補值後仍 < 99%；名單照常處理，但官方日晚上那次不凍結官方日線、留給成交日 06:49 那次；
#      請執行 fallback_manual.sh，並到公開資訊觀測站核對缺的公司）；其他＝程式錯誤（看 logs/）。
# 失敗時：git 被擋 → 依 read_documentation（environment.network）查網路設定；當天是訊號日就改跑 fallback_manual.sh。
set -u
F=/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live
mkdir -p $F/logs
LOG=$F/logs/daily_$(TZ=Asia/Taipei date +%Y%m%d_%H%M%S).log
MIN_KB=${FWD_MIN_KB:-$((1024 * 1024))}          # 1GB：低於此值中止（FWD_MIN_KB 只供測試）
WARN_KB=$((2 * 1024 * 1024))                    # 2GB：低於此值警告
avail_kb() { df -Pk "$F" | awk 'NR==2 {print $4}'; }

# ---- 0) 磁碟檢查（BF-3） ----
A=$(avail_kb)
if [ "$A" -lt "$MIN_KB" ]; then
  echo "磁碟剩餘 $((A / 1024))MB < 1GB：清除本題可重建的快取（work/asof_*、work/check_official、work/mp_union）" | tee -a "$LOG"
  rm -rf "$F"/work/asof_* "$F"/work/check_official "$F"/work/mp_union
  A=$(avail_kb)
fi
if [ "$A" -lt "$MIN_KB" ]; then
  MSG="【磁碟不足，中止】剩餘 $((A / 1024))MB < 1GB（每次執行約寫 25MB 以上，s6_forward.py 也需要空間）。請先清出至少 1GB（不要刪別人的資料）；今天若是訊號日，空間清出前先跑：bash $F/fallback_manual.sh"
  echo "$MSG" | tee -a "$LOG" >&2
  exit 3
elif [ "$A" -lt "$WARN_KB" ]; then
  echo "【警告】磁碟剩餘 $((A / 1024))MB < 2GB" | tee -a "$LOG"
fi

rc=1; ok=1
{
  echo "start $(TZ=Asia/Taipei date -Iseconds)；磁碟剩餘 $((A / 1024))MB"
  # 容器回收後 raw/ 不在 repo（.gitignore）：自動重建稀疏 clone
  if [ ! -d $F/raw/screener/.git ]; then
    ( mkdir -p $F/raw && cd $F/raw && timeout 900 git clone -q --filter=blob:none --no-checkout https://github.com/stevenhsiehsheng-ctrl/tw-stock-screener screener \
      && cd screener && git sparse-checkout init --no-cone \
      && printf '%s\n' /data/history.csv.gz /data/stock_list.csv /data/extras/rev_5y.csv.gz /data/extras/rev_hist.csv.gz /data/extras/disposal_5y.csv.gz \
         /data/extras/attention_5y.csv.gz /data/extras/exdiv_5y.csv.gz /data/extras/oddlot_hist.csv.gz /data/extras/oddlot.csv /data/extras/tw_holidays.json \
         /data/extras/updated.json /data/extras/warnings_hist.csv.gz /data/extras/warnings.csv /data/extras/src_log.csv > .git/info/sparse-checkout \
      && timeout 900 git checkout -q main ) || ok=0
  fi
  if [ ! -d $F/raw/finlab_repo/.git ]; then
    ( mkdir -p $F/raw/finlab && cd $F/raw && timeout 600 git clone -q --filter=blob:none --no-checkout https://github.com/jonatw/finlab-txd finlab_repo \
      && cd finlab_repo && git sparse-checkout init --no-cone && printf '%s\n' /data/raw/etf_0050.csv /data/raw/taiex_twii.csv > .git/info/sparse-checkout \
      && timeout 600 git checkout -q main ) || ok=0
  fi
  # 研究面板（只有 history 不足 252 日時才需要拼接；2026-11-10 起不需要，但影響股票代碼集合）
  [ -f /tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/adj.parquet ] || echo "注意：研究面板不存在（restore_env.sh）；history 已足 252 日時不影響名單"
  ( cd $F/raw/screener && timeout 900 git fetch -q origin main && git reset -q --hard origin/main ) || ok=0
  ( cd $F/raw/finlab_repo && timeout 300 git fetch -q origin main && git reset -q --hard origin/main \
      && cp data/raw/etf_0050.csv ../finlab/etf_0050.csv && cp data/raw/taiex_twii.csv ../finlab/taiex_twii.csv ) || ok=0
  echo "screener $(git -C $F/raw/screener log -1 --format='%H %cI')"
  echo "finlab   $(git -C $F/raw/finlab_repo log -1 --format='%H %cI')"
  if [ $ok -ne 1 ]; then echo "DATA_FETCH_FAILED：git 失敗，請查網路設定；今天若是訊號日改跑 fallback_manual.sh"; fi
  # 1b) 官方月營收（BF-5）
  DOM=$((10#$(TZ=Asia/Taipei date +%d)))
  if [ "${FWD_OFFICIAL_FETCH:-auto}" = "1" ] || { [ "${FWD_OFFICIAL_FETCH:-auto}" = "auto" ] && [ $DOM -ge 9 ] && [ $DOM -le 20 ]; }; then
    python3 -I -B $F/rev_official_fetch.py --out-root $F/raw/official_rev || echo "OFFICIAL_REV_FETCH_FAILED：官方月營收抓不到（不中止；只用上游月營收）"
  fi
  # FWD_EXTRA_ARGS 只供測試（例如 --out <scratch> --expect-now 2026-11-11T22:30）；正式排程不要設
  cd $F && python3 -I -B make_forward.py --official-rev $F/raw/official_rev ${FWD_EXTRA_ARGS:-}
  rc=$?
  echo "make_forward rc=$rc（0＝正常；2＝資料未更新到應有日期，未寫名單；4＝月營收覆蓋率不足）"
  echo "end $(TZ=Asia/Taipei date -Iseconds)"
} >> "$LOG" 2>&1
tail -3 "$LOG"
grep -h '^【' "$LOG" || true
if [ $rc -eq 2 ]; then
  MSG="【資料過期，未寫名單】daily_step0 回傳碼 2（詳見 $LOG）。今天若是訊號日（2026-11-10 J0、11-11 官方日），請立刻改跑：bash $F/fallback_manual.sh ，並在上游補上資料後重跑本檔。"
  echo "$MSG" >&2
elif [ $rc -eq 4 ]; then
  MSG="【月營收覆蓋率不足】daily_step0 回傳碼 4（詳見 $LOG）。官方補值後 m* 月營收在股池仍 < 99%。請執行：bash $F/fallback_manual.sh ，並到公開資訊觀測站核對缺的公司；今天是官方日（例如 2026-11-11）時，官方日線留給明天 06:49 那次凍結，請確認排程會跑（或 08:30 前手動再跑一次本檔）。"
  echo "$MSG" >&2
elif [ $rc -ne 0 ]; then
  echo "【執行失敗】make_forward rc=$rc（詳見 $LOG）；今天若是訊號日改跑 bash $F/fallback_manual.sh" >&2
elif [ $ok -ne 1 ]; then
  echo "【警告】git fetch 失敗，但現有資料已夠新（rc=0）；請查網路設定" >&2
fi
exit $rc

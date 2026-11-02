#!/bin/bash
# 每日流程第 0 步（o_forward_infra_live；PREREG_FORWARD 登記本檔 sha256）
# 排程：每天台北 06:49（既有每日排程開頭第一件事；週末也跑，週一前會補上週五的名單）。
#   1) 資料 git fetch（本題自建稀疏 clone，不動共用的 /home/user/stevenhsiehsheng-ctrl/tw-stock-screener）
#   2) make_forward.py（asof＝資料最後交易日）：訊號日的名單在這一步凍結，時間早於成交日 09:00
#   3) 處置帳本每天更新（make_forward.py 內含）
# 失敗時：git 被擋 → 依 read_documentation（environment.network）查網路設定；當月改用 fallback_manual.sh。
set -u
F=/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live
LOG=$F/logs/daily_$(TZ=Asia/Taipei date +%Y%m%d_%H%M).log
mkdir -p $F/logs
{
  echo "start $(TZ=Asia/Taipei date -Iseconds)"
  ok=1
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
  cd $F && python3 -I -B make_forward.py
  rc=$?
  echo "make_forward rc=$rc（2＝資料未更新到應有日期，未寫名單）"
  echo "end $(TZ=Asia/Taipei date -Iseconds)"
} >> "$LOG" 2>&1
tail -3 "$LOG"

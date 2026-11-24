#!/usr/bin/env bash
# 研究環境還原：容器被回收後，在新的雲端環境執行 `bash research/tw_stock_selection/restore_env.sh`。
# 1. 把 repo 內備份的價格面板複製回研究程式寫死的暫存路徑（BRIEF.md 的路徑）。
# 2. 依 data_sources.txt 重新 clone 外部資料來源，並盡量切到當時用的 commit。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
S=/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad
mkdir -p "$S/panel" "$S/scr_extra"
[ -f "$S/panel/adj.parquet" ] || cp "$HERE"/data_panel/panel/* "$S/panel/"
[ -f "$S/scr_extra/daytrade_5y.csv.gz" ] || cp "$HERE"/data_panel/scr_extra/* "$S/scr_extra/"
echo "panel: $(ls "$S/panel" | wc -l) 個檔案"

grep -v '^#' "$HERE/data_sources.txt" | while read -r path commit url; do
  dest="/home/user/$path"
  if git -C "$dest" rev-parse HEAD >/dev/null 2>&1; then echo "已存在 $path"; continue; fi
  if [ -e "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then echo "路徑被占用，略過 $path"; continue; fi
  mkdir -p "$(dirname "$dest")"
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 "$url" "$dest" || { echo "clone 失敗 $url"; continue; }
  if [ "$(git -C "$dest" rev-parse HEAD)" != "$commit" ]; then
    git -C "$dest" fetch -q --depth 1 origin "$commit" && git -C "$dest" checkout -q "$commit" \
      || echo "注意：$path 無法切到 $commit，使用最新版"
  fi
  echo "完成 $path"
done

# 3.（2026-11-25 加）容器重置後每日例行還需要的東西：Python 套件、官方日 K 倉庫的兩份本機複本。
pip install -q pyarrow scipy statsmodels openpyxl 2>/dev/null || echo "注意：pip 安裝失敗"
TSD="$S/tsd"
[ -d "$TSD/.git" ] || git clone -q --filter=blob:none --no-checkout --depth=1 https://github.com/jb7k6295r7-bot/tw-stock-data "$TSD" || echo "clone 失敗 tsd"
RAWK="$HERE/raw/jb7k6295r7-bot_tw-stock-data"
[ -d "$RAWK/.git" ] || git clone -q --filter=blob:none --no-checkout --depth=1 https://github.com/jb7k6295r7-bot/tw-stock-data "$RAWK" || echo "clone 失敗 raw"
# 正式引擎（cslib）用 GIT_NO_LAZY_FETCH 讀日 K，所以要先把 data/universe/daily 的內容一次取回本機
git -C "$RAWK" ls-tree HEAD data/universe/daily/ | awk '{print $3}' | git -C "$RAWK" -c fetch.negotiationAlgorithm=noop fetch origin --no-tags --no-write-fetch-head --recurse-submodules=no --filter=blob:none --stdin >/dev/null 2>&1 || echo "注意：日 K 內容取回失敗"
echo "還原完成；接著照 xq_scripts/site/state/README.md 跑每日例行"

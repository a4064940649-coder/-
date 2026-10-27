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

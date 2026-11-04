#!/bin/bash
# 第 1 步：只抓樹（blob:none、no-checkout、depth 1）列出路徑，看有沒有融資／外資持股封存。
# 用法：bash s1_probe.sh owner/repo [...]
SP=/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/opm2/clones
for r in "$@"; do
  d=$SP/$(echo $r | tr '/' '_')
  if [ ! -d "$d/.git" ]; then GIT_LFS_SKIP_SMUDGE=1 timeout 600 git clone -q --filter=blob:none --no-checkout --depth 1 https://github.com/$r "$d" 2>&1 | tail -2; fi
  echo "=== $r $(git -C $d rev-parse HEAD 2>/dev/null) $(git -C $d log -1 --format=%cI 2>/dev/null)"
  git -C $d ls-tree -r --name-only HEAD > $d.paths 2>/dev/null
  echo "files: $(wc -l < $d.paths)"
  awk -F/ '{p=$1; if(NF>2)p=$1"/"$2; if(NF>3)p=$1"/"$2"/"$3; c[p]++} END{for(k in c)print c[k], k}' $d.paths | sort -rn | head -25
  grep -i -E "margin|margn|qfii|foreign|融資|信用|外資|sharehold|credit" $d.paths | head -15
done

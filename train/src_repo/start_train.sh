#!/bin/bash
set -e
PYTHON=/opt/conda/bin/python
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
if [ ! -f "$PYTHON" ]; then
    echo "Python解释器未找到: $PYTHON"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$PYTHON -m pip install -i $PIP_INDEX -r "$SCRIPT_DIR/requirements.txt" --upgrade
cd "$SCRIPT_DIR"

CON_FILE="$SCRIPT_DIR/config.sh"
python - <<'PY' "$CON_FILE"
import json
import sys

env_file = sys.argv[1]

with open("/project/train/src_repo/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open(env_file, "w", encoding="utf-8") as f:
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, bool):
                value = "true" if value else "false"
            f.write(f"export {key.upper()}='{value}'\n")
PY
source "$CON_FILE"

$PYTHON dataset.py
$PYTHON train.py

if [ -d "$PROJECT/$NAME" ]; then
    FOLDER_PATH="$PROJECT/$NAME"
else
    FOLDER_PATH=$(find "$PROJECT" -maxdepth 1 -type d ! -path "$PROJECT" -print | head -n1)
fi
echo "模型路径"
echo "$FOLDER_PATH"

yolo export model="$FOLDER_PATH/weights/best.pt" format=engine imgsz=$IMGSZ device=0 half=True 
mv -f "$FOLDER_PATH/weights/best.engine" "$FOLDER_PATH/weights/best_fp16.engine"

import json
from pathlib import Path

def load_config(path: str | None = None) -> dict:
    if path:
        p = Path(path)
    else:
        p = Path(__file__).resolve().parents[0] / "config.json"
        if not p.exists():
            p = Path(__file__).resolve().parents[1] / "config.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
    

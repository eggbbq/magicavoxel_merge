"""UV JSON export helper."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import json


def write_uv_json(
    path: str | Path,
    *,
    width: int,
    height: int,
    model_rects: Dict[str, Tuple[float, float, float, float]],
) -> None:
    """Write UV extents as {"width":W,"height":H,"Model":[u0,v0,u1,v1],...}."""

    out_path = Path(path)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    items = [("width", width), ("height", height)] + list(model_rects.items())
    lines = ["{"]
    for idx, (key, value) in enumerate(items):
        ks = json.dumps(key, ensure_ascii=False)
        vs = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        comma = "," if idx != len(items) - 1 else ""
        lines.append(f"  {ks}:{vs}{comma}")
    lines.append("}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

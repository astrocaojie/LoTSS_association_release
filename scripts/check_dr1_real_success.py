#!/usr/bin/env python
from __future__ import annotations
import json, sys
from pathlib import Path
try:
    meta = json.loads(Path(sys.argv[1]).read_text())
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if meta.get('real_association_run') is True and meta.get('placeholder_adapter_output') is False else 1)

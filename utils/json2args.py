from __future__ import annotations

import argparse
import json


def json2args(json_path: str) -> argparse.Namespace:
    with open(json_path) as f:
        args = argparse.Namespace(**json.load(f))
    return args

# config/__init__.py
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


def load_settings(path=SETTINGS_PATH):

    with open(path) as f:
        return yaml.safe_load(f)

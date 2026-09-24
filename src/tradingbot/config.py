# config.py
from pathlib import Path

import yaml


# Repo root: src/tradingbot/config.py -> parents[2]. Data, models, logs and
# config/settings.yaml live there, outside the package.
ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


def load_settings(path=SETTINGS_PATH):

    with open(path) as f:
        return yaml.safe_load(f)

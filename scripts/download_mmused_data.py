"""Download / prepare MM-USED-Fallacy via MAMKit (run once)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mamkit.data.datasets import MMUSEDFallacy, InputMode

DATA_PATH = ROOT / "data"

MMUSEDFallacy(
    task_name="afc",
    input_mode=InputMode.TEXT_ONLY,
    base_data_path=DATA_PATH,
)
print("Done. Data at:", DATA_PATH / "MMUSED-fallacy")

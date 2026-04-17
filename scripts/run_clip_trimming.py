"""
Trim audio clips to match annotated snippet text using Whisper word timestamps.

From project root:
    python scripts/run_clip_trimming.py

Background:
    nohup python scripts/run_clip_trimming.py > logs/clip_trimming.log 2>&1 &
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clip_trimmer import run_clip_trimming

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

run_clip_trimming(
    model_name="medium",
    threshold=0.70,
    only_type1=True,
)

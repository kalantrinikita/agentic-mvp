"""Latency logger — captures TTFT and total generation time."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import LATENCY_LOG


class LatencyLogger:
    """Records TTFT (Time to First Token) and total generation time."""

    def __init__(self, log_path: Path = LATENCY_LOG):
        self.log_path = log_path

    def log(
        self,
        model: str,
        prompt: str,
        response: str,
        ttft_seconds: float,
        total_seconds: float,
        tokens_generated: Optional[int] = None,
        phase: str = "phase1",
    ) -> dict:
        """Append one latency record to the JSONL log."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase,
            "model": model,
            "prompt_preview": prompt[:100],
            "response_preview": response[:100],
            "ttft_seconds": round(ttft_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "tokens_generated": tokens_generated,
            "tokens_per_sec": (
                round(tokens_generated / total_seconds, 2)
                if tokens_generated and total_seconds > 0
                else None
            ),
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record
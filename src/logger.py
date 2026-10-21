"""Latency logger — captures retrieval + LLM metrics."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import LATENCY_LOG


class LatencyLogger:
    """Records retrieval + LLM latency metrics."""

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
        retrieval_seconds: Optional[float] = None,
        chunks_retrieved: Optional[int] = None,
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
            # Phase 2 additions:
            "retrieval_seconds": (
                round(retrieval_seconds, 3) if retrieval_seconds is not None else None
            ),
            "chunks_retrieved": chunks_retrieved,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record
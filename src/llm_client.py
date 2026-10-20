"""Ollama client wrapper with streaming + latency tracking."""
import time
from typing import Optional

import ollama

from src.config import DEFAULT_MODEL, OLLAMA_HOST
from src.logger import LatencyLogger


class LLMClient:
    """Thin wrapper around Ollama with built-in latency tracking."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        self.latency_logger = LatencyLogger()

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        phase: str = "phase1",
    ) -> dict:
        """Generate a response (streaming) and track latency metrics."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        ttft = None
        full_response = ""
        tokens = 0

        # Stream tokens so we can capture TTFT
        response_stream = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
        )

        for chunk in response_stream:
            if ttft is None:
                ttft = time.perf_counter() - start
            content = chunk.get("message", {}).get("content", "")
            full_response += content
            tokens += 1

        total = time.perf_counter() - start

        # Log metrics
        record = self.latency_logger.log(
            model=self.model,
            prompt=prompt,
            response=full_response,
            ttft_seconds=ttft or total,
            total_seconds=total,
            tokens_generated=tokens,
            phase=phase,
        )

        return {
            "response": full_response,
            "ttft": ttft,
            "total_time": total,
            "tokens": tokens,
            "model": self.model,
            "metrics": record,
        }
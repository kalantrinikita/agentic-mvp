"""Ollama client wrapper with streaming + latency tracking + RAG support."""
import time
from typing import Optional

import ollama

from src.config import DEFAULT_MODEL, OLLAMA_HOST
from src.logger import LatencyLogger


# Prompt template used for RAG — grounds the LLM in retrieved context
RAG_SYSTEM_PROMPT = """You are a helpful IT support assistant. Answer the user's question using ONLY the context provided below. 

RULES:
- If the answer is in the context, provide it clearly and cite the relevant section.
- If the answer is NOT in the context, say "I don't have information about that in the IT policy document."
- Do not make up information or use outside knowledge.
- Keep answers concise and factual.

CONTEXT:
{context}
"""


class LLMClient:
    """Thin wrapper around Ollama with latency tracking + RAG support."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        self.latency_logger = LatencyLogger()

    def _run_stream(
        self,
        messages: list,
        phase: str,
        prompt_for_log: str,
        retrieval_seconds: Optional[float] = None,
        chunks_retrieved: Optional[int] = None,
    ) -> dict:
        """Internal: run streaming chat and log metrics."""
        start = time.perf_counter()
        ttft = None
        full_response = ""
        tokens = 0

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

        record = self.latency_logger.log(
            model=self.model,
            prompt=prompt_for_log,
            response=full_response,
            ttft_seconds=ttft or total,
            total_seconds=total,
            tokens_generated=tokens,
            phase=phase,
            retrieval_seconds=retrieval_seconds,
            chunks_retrieved=chunks_retrieved,
        )

        return {
            "response": full_response,
            "ttft": ttft,
            "total_time": total,
            "tokens": tokens,
            "model": self.model,
            "metrics": record,
        }

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        phase: str = "phase1",
    ) -> dict:
        """Simple generation (Phase 1 style — no RAG)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self._run_stream(
            messages=messages,
            phase=phase,
            prompt_for_log=prompt,
        )

    def generate_with_context(
        self,
        question: str,
        context_chunks: list[str],
        retrieval_seconds: float,
        phase: str = "phase2",
    ) -> dict:
        """RAG generation — inject retrieved chunks as context."""
        # Join chunks with separators so the LLM sees them as distinct snippets
        context = "\n\n---\n\n".join(
            f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )

        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        return self._run_stream(
            messages=messages,
            phase=phase,
            prompt_for_log=question,
            retrieval_seconds=retrieval_seconds,
            chunks_retrieved=len(context_chunks),
        )
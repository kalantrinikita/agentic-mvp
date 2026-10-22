"""LLM-as-a-Judge evaluator.

For each question in the golden dataset:
  1. Ask the agent (Phase 4 MCP agent)
  2. Have a judge LLM score the answer vs expected answer
  3. Record scores on two axes: Groundedness + Helpfulness
"""
import json
import re
from pathlib import Path

from src.llm_client import LLMClient


# ──────────────────────────────────────────────────────────
# JUDGE PROMPT
# ──────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator judging the quality of an AI assistant's answer.

You will be given:
1. The user's QUESTION
2. The EXPECTED ANSWER (ground truth)
3. The AGENT'S ANSWER (what the AI actually said)

Score the agent's answer on TWO dimensions:

GROUNDEDNESS (1-5):
  5 = Answer fully matches expected answer; no fabrication; specific facts correct
  4 = Mostly correct; minor wording differences; key facts match
  3 = Partially correct; some facts missing or slightly wrong
  2 = Largely incorrect; wrong facts or significant omissions
  1 = Completely wrong or hallucinated

HELPFULNESS (1-5):
  5 = Clear, concise, directly answers the question
  4 = Good answer, maybe slightly verbose or formatted oddly
  3 = Answers but is unclear, overly long, or includes irrelevant content
  2 = Hard to understand or barely addresses the question
  1 = Does not answer the question meaningfully

OUTPUT FORMAT:
Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):

{"groundedness": <1-5>, "helpfulness": <1-5>, "reasoning": "<one-sentence explanation>"}
"""


class LLMJudge:
    """Uses a stronger LLM to score agent answers against expected answers."""

    def __init__(self, judge_model: str):
        # The judge uses its own LLMClient — ideally a stronger model
        self.judge_llm = LLMClient(model=judge_model)

    def score(self, question: str, expected: str, agent_answer: str) -> dict:
        """Judge one answer. Returns scores + reasoning."""
        user_prompt = (
            f"QUESTION: {question}\n\n"
            f"EXPECTED ANSWER: {expected}\n\n"
            f"AGENT'S ANSWER: {agent_answer}\n\n"
            f"Score the agent's answer. Respond with JSON only."
        )

        result = self.judge_llm.generate(
            prompt=user_prompt,
            system=JUDGE_SYSTEM_PROMPT,
            phase="phase5_judge",
        )

        parsed = self._extract_json(result["response"])

        return {
            "groundedness": parsed.get("groundedness", 0),
            "helpfulness": parsed.get("helpfulness", 0),
            "reasoning": parsed.get("reasoning", ""),
            "raw_judge_output": result["response"],
            "judge_latency": result["total_time"],
        }

    def _extract_json(self, text: str) -> dict:
        """Robustly extract JSON from LLM output."""
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {"groundedness": 0, "helpfulness": 0, "reasoning": "FAILED_TO_PARSE"}


# ──────────────────────────────────────────────────────────
# DATASET LOADER
# ──────────────────────────────────────────────────────────

def load_golden_dataset(path: Path) -> list[dict]:
    """Load the JSONL golden dataset."""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


# ──────────────────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────────────────

def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate metrics from eval results."""
    if not results:
        return {}

    groundedness_scores = [r["groundedness"] for r in results]
    helpfulness_scores = [r["helpfulness"] for r in results]
    latencies = [r["total_latency"] for r in results]

    def avg(vals: list) -> float:
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "n_questions": len(results),
        "avg_groundedness": avg(groundedness_scores),
        "avg_helpfulness": avg(helpfulness_scores),
        "avg_total_latency_sec": avg(latencies),
        "perfect_groundedness_count": sum(1 for s in groundedness_scores if s == 5),
        "poor_groundedness_count": sum(1 for s in groundedness_scores if s <= 2),
    }
"""Agent engine: route queries between RAG and tool calls."""
import json
import re
from typing import Any

from src.llm_client import LLMClient
from src.rag import RAGEngine
from src.tools import TOOLS, execute_tool, get_tools_description


# ──────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ──────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are an IT support agent router. Your job is to analyze the user's question and decide what action to take.

AVAILABLE ACTIONS:

1. "rag_query" — Search the IT policy documents.
   USE WHEN: User asks about policies, procedures, rules, or "how to" questions related to IT.
   Example questions: "What's the VPN policy?", "How many days can I work remotely?"

2. "tool_call" — Execute one of the available tools.
   USE WHEN: User wants to perform a specific action (check a ticket, submit a request).

AVAILABLE TOOLS:
{tools_description}

OUTPUT FORMAT:
You MUST respond with ONLY a valid JSON object (no markdown, no explanation, no extra text).

For RAG queries:
{{"action": "rag_query", "query": "<the user's question>"}}

For tool calls:
{{"action": "tool_call", "tool": "<tool_name>", "args": {{"param1": "value1"}}}}

If the question is unclear or unrelated to IT support, respond with:
{{"action": "rag_query", "query": "<user question>"}}

Respond with ONLY the JSON. No other text.
"""


SUMMARIZER_SYSTEM_PROMPT = """You are a helpful IT support assistant. The user asked a question, and a tool was executed to get the result. Your job is to summarize the result in clear, friendly language.

Keep your answer concise (1-3 sentences).
Mention specific values from the tool result (IDs, dates, statuses).
Do not add information not present in the tool result.
"""


# ──────────────────────────────────────────────────────────
# AGENT CLASS
# ──────────────────────────────────────────────────────────

class Agent:
    """Routes queries between RAG and tool calls, then executes."""

    def __init__(self, llm: LLMClient, rag: RAGEngine):
        self.llm = llm
        self.rag = rag

    # ──────────────────────────────────────────────────────
    # ROUTING
    # ──────────────────────────────────────────────────────

    def route(self, user_question: str) -> dict:
        """Ask the LLM to decide: RAG or tool call?"""
        system_prompt = ROUTER_SYSTEM_PROMPT.format(
            tools_description=get_tools_description()
        )

        # Call LLM, expecting pure JSON back
        result = self.llm.generate(
            prompt=user_question,
            system=system_prompt,
            phase="phase3_router",
        )

        raw_response = result["response"].strip()

        # Try to parse JSON (LLMs sometimes wrap it in markdown)
        parsed = self._extract_json(raw_response)

        return {
            "raw_response": raw_response,
            "parsed": parsed,
            "router_latency": result["total_time"],
        }

    def _extract_json(self, text: str) -> dict:
        """Robustly extract JSON from LLM output.

        LLMs sometimes wrap JSON in ```json ... ``` or add stray text.
        """
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: find the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Give up — return error marker
        return {"action": "error", "error": "Could not parse JSON", "raw": text}

    # ──────────────────────────────────────────────────────
    # EXECUTION
    # ──────────────────────────────────────────────────────

    def execute(self, decision: dict) -> dict:
        """Execute the routed action."""
        action = decision.get("action")

        if action == "rag_query":
            query = decision.get("query", "")
            retrieval = self.rag.retrieve(query, top_k=2)
            answer_result = self.llm.generate_with_context(
                question=query,
                context_chunks=retrieval["chunks"],
                retrieval_seconds=retrieval["retrieval_time"],
                phase="phase3_rag",
            )
            return {
                "type": "rag",
                "answer": answer_result["response"],
                "chunks": retrieval["chunks"],
                "retrieval_time": retrieval["retrieval_time"],
                "generation_time": answer_result["total_time"],
            }

        elif action == "tool_call":
            tool_name = decision.get("tool", "")
            args = decision.get("args", {})

            # Execute the tool
            tool_result = execute_tool(tool_name, args)

            # Summarize the result for the user
            summary_prompt = (
                f"User question: {decision.get('_user_question', '')}\n"
                f"Tool called: {tool_name}\n"
                f"Tool result: {json.dumps(tool_result, indent=2)}\n\n"
                f"Please summarize this result for the user."
            )
            summary = self.llm.generate(
                prompt=summary_prompt,
                system=SUMMARIZER_SYSTEM_PROMPT,
                phase="phase3_summary",
            )

            return {
                "type": "tool",
                "tool_name": tool_name,
                "tool_args": args,
                "tool_result": tool_result,
                "answer": summary["response"],
                "generation_time": summary["total_time"],
            }

        else:
            return {
                "type": "error",
                "answer": "Sorry, I couldn't understand how to handle that question.",
                "decision": decision,
            }

    # ──────────────────────────────────────────────────────
    # TOP-LEVEL ENTRY POINT
    # ──────────────────────────────────────────────────────

    def ask(self, user_question: str) -> dict:
        """End-to-end: route → execute → return everything."""
        # 1. Route
        routing = self.route(user_question)

        # Attach the user question so the summarizer can see it
        decision = routing["parsed"]
        decision["_user_question"] = user_question

        # 2. Execute
        execution = self.execute(decision)

        # 3. Return a bundle with everything (for the UI/logger)
        return {
            "user_question": user_question,
            "routing": routing,
            "execution": execution,
        }
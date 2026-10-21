"""Phase 4: Agent that uses tools via MCP (Model Context Protocol).

Unlike Phase 3 (hardcoded tools), this agent:
  1. Launches the MCP server as a subprocess
  2. Discovers available tools dynamically at startup
  3. Routes queries by calling tools via JSON-RPC over stdio
"""
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from src.llm_client import LLMClient
from src.rag import RAGEngine


# ──────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ──────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are an IT support agent router. Your job is to analyze the user's question and decide what action to take.

AVAILABLE ACTIONS:

1. "rag_query" — Search the IT policy documents.
   USE WHEN: User asks about policies, procedures, rules, or "how to" questions.

2. "tool_call" — Execute one of the available tools (discovered via MCP).
   USE WHEN: User wants to perform a specific action (check a ticket, submit a request).

AVAILABLE TOOLS (dynamically discovered from MCP server):
{tools_description}

OUTPUT FORMAT:
You MUST respond with ONLY a valid JSON object (no markdown, no explanation, no extra text).

For RAG queries:
{{"action": "rag_query", "query": "<the user's question>"}}

For tool calls:
{{"action": "tool_call", "tool": "<tool_name>", "args": {{"param1": "value1"}}}}

Respond with ONLY the JSON. No other text.
"""


SUMMARIZER_SYSTEM_PROMPT = """You are a helpful IT support assistant. The user asked a question, and a tool was executed via MCP to get the result. Summarize the result in clear, friendly language.

Keep your answer concise (1-3 sentences).
Mention specific values from the tool result (IDs, dates, statuses).
Do not add information not present in the tool result.
"""


# ──────────────────────────────────────────────────────────
# MCP CLIENT AGENT
# ──────────────────────────────────────────────────────────

class MCPAgent:
    """Agent that uses MCP for tool discovery and execution."""

    def __init__(self, llm: LLMClient, rag: RAGEngine, server_script: Path):
        self.llm = llm
        self.rag = rag
        self.server_script = server_script

        # These get set when we connect
        self.session: ClientSession | None = None
        self.tools: list[Any] = []
        self._exit_stack: AsyncExitStack | None = None

    # ──────────────────────────────────────────────────────
    # CONNECTION LIFECYCLE
    # ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Launch the MCP server as a subprocess and discover its tools."""
        self._exit_stack = AsyncExitStack()

        # Configure how to launch the server
        server_params = StdioServerParameters(
            command="python",
            args=[str(self.server_script)],
        )

        # Open the stdio pipes to the server subprocess
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        # Create the MCP client session
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # Perform the MCP initialization handshake
        await self.session.initialize()

        # Discover available tools (dynamic!)
        tools_response = await self.session.list_tools()
        self.tools = list(tools_response.tools)

    async def disconnect(self) -> None:
        """Clean up the subprocess + streams."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None

    # ──────────────────────────────────────────────────────
    # TOOLS DESCRIPTION (built from MCP discovery)
    # ──────────────────────────────────────────────────────

    def get_tools_description(self) -> str:
        """Format discovered MCP tools for the router prompt."""
        if not self.tools:
            return "(No tools available)"

        lines: list[str] = []

        for tool in self.tools:
            lines.append(f"\nTool: {tool.name}")

            # Keep only the first line of the description — the full docstring
            # (with Args/Returns) is noise for the LLM router prompt.
            raw_desc = (tool.description or "").strip()
            first_line = raw_desc.split("\n")[0] if raw_desc else ""
            lines.append(f"  Description: {first_line or '(no description)'}")

            # MCP tools come with a JSON schema for inputs
            # (snake_case `input_schema` in SDK v2.x)
            schema_obj: Any = getattr(tool, "input_schema", None) or {}
            schema: dict = schema_obj if isinstance(schema_obj, dict) else {}
            props: dict = schema.get("properties", {})
            required: list = schema.get("required", [])

            if props:
                lines.append("  Parameters:")
                for param_name, param_info in props.items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "")
                    req_marker = " (required)" if param_name in required else ""
                    lines.append(
                        f"    - {param_name} ({param_type}){req_marker}: {param_desc}"
                    )

            # Example JSON for the LLM to mimic
            example_args = {name: f"<{name}_value>" for name in props.keys()}
            example = {
                "action": "tool_call",
                "tool": tool.name,
                "args": example_args,
            }
            lines.append(f"  Example JSON: {json.dumps(example)}")

        result: str = "\n".join(lines)
        return result

    # ──────────────────────────────────────────────────────
    # ROUTING
    # ──────────────────────────────────────────────────────

    def route(self, user_question: str) -> dict:
        """Ask the LLM: RAG or tool call?"""
        system_prompt = ROUTER_SYSTEM_PROMPT.format(
            tools_description=self.get_tools_description()
        )

        result = self.llm.generate(
            prompt=user_question,
            system=system_prompt,
            phase="phase4_router",
        )

        raw = result["response"].strip()
        parsed = self._extract_json(raw)

        return {
            "raw_response": raw,
            "parsed": parsed,
            "router_latency": result["total_time"],
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

        return {"action": "error", "error": "Could not parse JSON", "raw": text}

    # ──────────────────────────────────────────────────────
    # EXECUTION (via MCP)
    # ──────────────────────────────────────────────────────

    async def execute(self, decision: dict) -> dict:
        """Execute the routed action."""
        action = decision.get("action")

        # ── RAG path ──────────────────────────────────────
        if action == "rag_query":
            query = decision.get("query", "")
            retrieval = self.rag.retrieve(query, top_k=2)
            answer = self.llm.generate_with_context(
                question=query,
                context_chunks=retrieval["chunks"],
                retrieval_seconds=retrieval["retrieval_time"],
                phase="phase4_rag",
            )
            return {
                "type": "rag",
                "answer": answer["response"],
                "chunks": retrieval["chunks"],
                "retrieval_time": retrieval["retrieval_time"],
                "generation_time": answer["total_time"],
            }

        # ── Tool call via MCP ─────────────────────────────
        elif action == "tool_call":
            if not self.session:
                return {"type": "error", "answer": "MCP session not connected."}

            tool_name = decision.get("tool", "")
            args = decision.get("args", {})

            # The actual MCP JSON-RPC call!
            tool_response = await self.session.call_tool(tool_name, args)

            # MCP returns a list of content items; we only consume text content.
            tool_result_text = ""
            for content in tool_response.content:
                if isinstance(content, TextContent):
                    tool_result_text += content.text

            # Try to parse the result as JSON for pretty display
            try:
                tool_result = json.loads(tool_result_text)
            except json.JSONDecodeError:
                tool_result = {"raw": tool_result_text}

            # Summarize via LLM
            summary_prompt = (
                f"User question: {decision.get('_user_question', '')}\n"
                f"Tool called (via MCP): {tool_name}\n"
                f"Tool result: {json.dumps(tool_result, indent=2)}\n\n"
                f"Please summarize this result for the user."
            )
            summary = self.llm.generate(
                prompt=summary_prompt,
                system=SUMMARIZER_SYSTEM_PROMPT,
                phase="phase4_summary",
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
                "answer": "Couldn't understand how to handle that question.",
                "decision": decision,
            }

    # ──────────────────────────────────────────────────────
    # TOP-LEVEL
    # ──────────────────────────────────────────────────────

    async def ask(self, user_question: str) -> dict:
        """End-to-end: route → execute via MCP → return bundle."""
        routing = self.route(user_question)

        decision = routing["parsed"]
        decision["_user_question"] = user_question

        execution = await self.execute(decision)

        return {
            "user_question": user_question,
            "routing": routing,
            "execution": execution,
        }
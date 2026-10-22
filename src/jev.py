"""JSON Execution Validation (JEV) for MCP tool calls.

Validates that the LLM's routing JSON output conforms EXACTLY to the
MCP server's expected format. Binary pass/fail — no partial credit.

Checks:
  1. Is the output valid JSON?
  2. Has the correct "action" field?
  3. Does the tool name exist on the MCP server?
  4. Are all REQUIRED arguments present?
  5. Are there NO extra (hallucinated) arguments?
  6. Do argument types match the schema?
  7. (Bonus) Do arg VALUES match the expected ones?
"""
import json
import re
from typing import Any


class JEVResult:
    """Result of validating one tool call."""

    def __init__(self):
        self.passed = False
        self.score = 0  # 0 or 1
        self.checks: list[dict] = []
        self.first_failure: str | None = None

    def add_check(self, name: str, passed: bool, detail: str = "") -> None:
        """Record one validation check."""
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed and self.first_failure is None:
            self.first_failure = f"{name}: {detail}"

    def finalize(self) -> None:
        """Compute final pass/fail. ALL checks must pass."""
        self.passed = all(c["passed"] for c in self.checks)
        self.score = 1 if self.passed else 0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "first_failure": self.first_failure,
            "checks": self.checks,
        }


class JEVValidator:
    """Validates LLM tool-call JSON against MCP tool schemas."""

    def __init__(self, mcp_tools: list[Any]):
        """
        Args:
            mcp_tools: List of tool objects discovered from the MCP server.
                       (Each has .name, .input_schema, .description)
        """
        # Build a lookup: tool_name -> schema
        self.tool_schemas = {}
        for tool in mcp_tools:
            schema_obj = getattr(tool, "input_schema", None) or {}
            schema: dict = schema_obj if isinstance(schema_obj, dict) else {}
            self.tool_schemas[tool.name] = {
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }

    def validate(
        self,
        raw_llm_output: str,
        expected_tool: str,
        expected_args: dict,
    ) -> JEVResult:
        """Validate the LLM's raw output against MCP spec + expectations."""
        result = JEVResult()

        # ── Check 1: Valid JSON parse ────────────────────
        parsed = self._extract_json(raw_llm_output)
        if parsed is None:
            result.add_check("valid_json", False, "Could not parse any JSON from output")
            result.finalize()
            return result
        result.add_check("valid_json", True, "JSON parsed OK")

        # ── Check 2: Has "action" field ──────────────────
        action = parsed.get("action")
        if action != "tool_call":
            result.add_check(
                "action_is_tool_call",
                False,
                f"Expected action='tool_call', got '{action}'",
            )
            result.finalize()
            return result
        result.add_check("action_is_tool_call", True, "action=tool_call")

        # ── Check 3: Has "tool" field ────────────────────
        tool_name = parsed.get("tool")
        if not tool_name:
            result.add_check("tool_field_present", False, "Missing 'tool' field")
            result.finalize()
            return result
        result.add_check("tool_field_present", True, f"tool='{tool_name}'")

        # ── Check 4: Tool exists on MCP server ───────────
        if tool_name not in self.tool_schemas:
            result.add_check(
                "tool_exists",
                False,
                f"Tool '{tool_name}' not registered on MCP server",
            )
            result.finalize()
            return result
        result.add_check("tool_exists", True, "Tool is registered")

        # ── Check 5: Correct tool chosen ─────────────────
        if tool_name != expected_tool:
            result.add_check(
                "correct_tool_chosen",
                False,
                f"Expected '{expected_tool}', got '{tool_name}'",
            )
            result.finalize()
            return result
        result.add_check("correct_tool_chosen", True, "Correct tool")

        # ── Check 6: Has "args" field ────────────────────
        args = parsed.get("args")
        if not isinstance(args, dict):
            result.add_check(
                "args_is_dict",
                False,
                f"'args' must be dict, got {type(args).__name__}",
            )
            result.finalize()
            return result
        result.add_check("args_is_dict", True, "args is a dict")

        # ── Check 7: All required args present ───────────
        schema = self.tool_schemas[tool_name]
        required_args = schema["required"]
        missing = [p for p in required_args if p not in args]
        if missing:
            result.add_check(
                "required_args_present",
                False,
                f"Missing required args: {missing}",
            )
            result.finalize()
            return result
        result.add_check("required_args_present", True, "All required args present")

        # ── Check 8: No extra/hallucinated args ──────────
        allowed_args = set(schema["properties"].keys())
        extra = [a for a in args if a not in allowed_args]
        if extra:
            result.add_check(
                "no_extra_args",
                False,
                f"Hallucinated args not in schema: {extra}",
            )
            result.finalize()
            return result
        result.add_check("no_extra_args", True, "No extra args")

        # ── Check 9: Arg types match schema ──────────────
        type_errors = []
        for param_name, param_info in schema["properties"].items():
            if param_name not in args:
                continue
            expected_type = param_info.get("type", "string")
            value = args[param_name]
            if not self._type_matches(value, expected_type):
                type_errors.append(
                    f"{param_name}: expected {expected_type}, got {type(value).__name__}"
                )
        if type_errors:
            result.add_check("arg_types_valid", False, "; ".join(type_errors))
            result.finalize()
            return result
        result.add_check("arg_types_valid", True, "All arg types match")

        # ── Check 10: Arg VALUES match expected ──────────
        value_mismatches = []
        for key, exp_val in expected_args.items():
            actual = args.get(key)
            # Compare as strings for robustness (LLMs may vary spacing/casing)
            if str(actual).strip().lower() != str(exp_val).strip().lower():
                value_mismatches.append(f"{key}: expected '{exp_val}', got '{actual}'")
        if value_mismatches:
            result.add_check(
                "arg_values_match",
                False,
                "; ".join(value_mismatches),
            )
            result.finalize()
            return result
        result.add_check("arg_values_match", True, "All arg values match")

        # All checks passed!
        result.finalize()
        return result

    # ──────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────

    def _extract_json(self, text: str) -> dict | None:
        """Extract a JSON object from LLM output."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass

        return None

    def _type_matches(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches an MCP JSON schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_py = type_map.get(expected_type, str)
        return isinstance(value, expected_py)
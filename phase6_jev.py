"""Phase 6: JSON Execution Validation (JEV) for tool-call routing."""
import asyncio
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import DEFAULT_MODEL, LOGS_DIR, PROJECT_ROOT
from src.evaluator import load_golden_dataset
from src.jev import JEVValidator
from src.llm_client import LLMClient
from src.mcp_agent import MCPAgent
from src.rag import RAGEngine

console = Console()

GOLDEN_DATASET_PATH = PROJECT_ROOT / "golden_dataset" / "eval_set.jsonl"
JEV_RESULTS_PATH = LOGS_DIR / "phase6_jev_results.jsonl"


async def evaluate_tool_call(
    agent: MCPAgent,
    validator: JEVValidator,
    case: dict,
) -> dict:
    """Validate one tool-call case."""
    console.print(Panel(
        f"[bold cyan]{case['id']}:[/] {case['question']}\n"
        f"[dim]Expected tool: {case['expected_tool']}[/]\n"
        f"[dim]Expected args: {case['expected_args']}[/]",
        border_style="cyan",
    ))

    # 1. Have the agent route the query (same as Phase 4)
    routing = agent.route(case["question"])
    raw_output = routing["raw_response"]
    router_latency = routing["router_latency"]

    console.print(f"[yellow]Raw LLM output:[/] {raw_output[:200]}")

    # 2. Validate with JEV
    jev_result = validator.validate(
        raw_llm_output=raw_output,
        expected_tool=case["expected_tool"],
        expected_args=case["expected_args"],
    )

    # 3. Display verdict
    verdict = "[bold green]✓ PASS[/]" if jev_result.passed else "[bold red]✗ FAIL[/]"
    console.print(f"\n[bold]JEV Result:[/] {verdict}  [dim](score = {jev_result.score})[/]")

    if not jev_result.passed:
        console.print(f"[red]First failure:[/] {jev_result.first_failure}")

    # Show all checks
    table = Table(show_header=True, title="\nValidation Checks")
    table.add_column("#", style="dim")
    table.add_column("Check", style="cyan")
    table.add_column("Result")
    table.add_column("Detail", style="dim")
    for i, check in enumerate(jev_result.checks, 1):
        status = "[green]✓[/]" if check["passed"] else "[red]✗[/]"
        table.add_row(str(i), check["name"], status, check["detail"][:60])
    console.print(table)
    console.print("─" * 70 + "\n")

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_tool": case["expected_tool"],
        "expected_args": case["expected_args"],
        "raw_llm_output": raw_output,
        "jev_score": jev_result.score,
        "jev_passed": jev_result.passed,
        "first_failure": jev_result.first_failure,
        "all_checks": jev_result.checks,
        "router_latency": round(router_latency, 2),
        "agent_model": DEFAULT_MODEL,
    }


async def main() -> None:
    console.print("[bold magenta]═══ Phase 6: JEV (JSON Execution Validation) ═══[/]\n")
    console.print(f"[cyan]Agent model:[/] {DEFAULT_MODEL}\n")

    # Set up agent
    llm = LLMClient(model=DEFAULT_MODEL)
    rag = RAGEngine()
    server_script = PROJECT_ROOT / "mcp_server" / "server.py"
    agent = MCPAgent(llm=llm, rag=rag, server_script=server_script)

    # Load golden dataset; filter to tool cases only
    all_cases = load_golden_dataset(GOLDEN_DATASET_PATH)
    tool_cases = [c for c in all_cases if c.get("type") == "tool"]
    console.print(f"[green]✓ Loaded {len(tool_cases)} tool-call test cases[/]\n")

    try:
        await agent.connect()
        console.print(
            f"[green]✓ Connected to MCP server ({len(agent.tools)} tools)[/]\n"
        )

        # Create validator with live schemas from MCP
        validator = JEVValidator(mcp_tools=agent.tools)

        # Run eval
        results = []
        for case in tool_cases:
            result = await evaluate_tool_call(agent, validator, case)
            results.append(result)

        # Save
        with open(JEV_RESULTS_PATH, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r["jev_passed"])
        pass_rate = (passed / total * 100) if total > 0 else 0

        console.print("[bold magenta]═══ JEV Summary ═══[/]\n")
        table = Table(show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Agent Model", DEFAULT_MODEL)
        table.add_row("Total Tool Cases", str(total))
        table.add_row("PASSED (score=1)", f"[green]{passed}[/]")
        table.add_row("FAILED (score=0)", f"[red]{total - passed}[/]")
        table.add_row("Pass Rate", f"{pass_rate:.1f}%")
        console.print(table)

        # Breakdown of first-failure reasons
        failure_counts: dict[str, int] = {}
        for r in results:
            if not r["jev_passed"] and r["first_failure"]:
                key = r["first_failure"].split(":")[0]
                failure_counts[key] = failure_counts.get(key, 0) + 1

        if failure_counts:
            console.print("\n[bold yellow]Failure breakdown:[/]")
            for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
                console.print(f"  • {reason}: {count}")

        console.print(f"\n[green]✓ Results saved to:[/] {JEV_RESULTS_PATH}")
        console.print("[bold green]✓ Phase 6 complete![/]")

    finally:
        await agent.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
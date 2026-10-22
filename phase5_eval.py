"""Phase 5: Run golden dataset through the agent and judge the answers."""
import asyncio
import json
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import DEFAULT_MODEL, LARGE_MODEL, LOGS_DIR, PROJECT_ROOT
from src.evaluator import LLMJudge, compute_summary, load_golden_dataset
from src.llm_client import LLMClient
from src.mcp_agent import MCPAgent
from src.rag import RAGEngine

console = Console()

# Judge uses the larger model (llama3.1:8b) for better evaluation quality
JUDGE_MODEL = LARGE_MODEL
AGENT_MODEL = DEFAULT_MODEL

GOLDEN_DATASET_PATH = PROJECT_ROOT / "golden_dataset" / "eval_set.jsonl"
EVAL_RESULTS_PATH = LOGS_DIR / "phase5_eval_results.jsonl"


async def evaluate_one(
    agent: MCPAgent,
    judge: LLMJudge,
    case: dict,
    console: Console,
) -> dict:
    """Run one question through the agent and judge the answer."""
    console.print(Panel(
        f"[bold cyan]{case['id']}:[/] {case['question']}",
        border_style="cyan"
    ))

    # 1. Ask the agent
    bundle = await agent.ask(case["question"])
    agent_answer = bundle["execution"].get("answer", "(no answer)")
    agent_latency = (
        bundle["routing"]["router_latency"]
        + bundle["execution"].get("generation_time", 0)
    )

    console.print(f"[bold yellow]Agent answer:[/] {agent_answer}")
    console.print(f"[dim]Expected: {case['expected_answer']}[/]")

    # 2. Judge the answer
    judgment = judge.score(
        question=case["question"],
        expected=case["expected_answer"],
        agent_answer=agent_answer,
    )

    # 3. Display verdict
    g = judgment["groundedness"]
    h = judgment["helpfulness"]
    g_color = "green" if g >= 4 else "yellow" if g >= 3 else "red"
    h_color = "green" if h >= 4 else "yellow" if h >= 3 else "red"

    table = Table(show_header=True)
    table.add_column("Dimension", style="cyan")
    table.add_column("Score (1-5)")
    table.add_row("Groundedness", f"[{g_color}]{g}/5[/]")
    table.add_row("Helpfulness", f"[{h_color}]{h}/5[/]")
    console.print(table)
    console.print(f"[dim]Judge reasoning: {judgment['reasoning']}[/]\n")
    console.print("─" * 70 + "\n")

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_answer": case["expected_answer"],
        "agent_answer": agent_answer,
        "groundedness": g,
        "helpfulness": h,
        "reasoning": judgment["reasoning"],
        "agent_latency": round(agent_latency, 2),
        "judge_latency": round(judgment["judge_latency"], 2),
        "total_latency": round(agent_latency + judgment["judge_latency"], 2),
        "agent_model": AGENT_MODEL,
        "judge_model": JUDGE_MODEL,
    }


async def main() -> None:
    console.print("[bold magenta]═══ Phase 5: LLM-as-a-Judge Evaluation ═══[/]\n")
    console.print(f"[cyan]Agent model:[/] {AGENT_MODEL}")
    console.print(f"[cyan]Judge model:[/] {JUDGE_MODEL}\n")

    # Set up agent (same as Phase 4)
    llm = LLMClient(model=AGENT_MODEL)
    rag = RAGEngine()

    if rag.count() == 0:
        console.print("[bold red]✗ No documents in ChromaDB! Run: python ingest.py[/]")
        return

    server_script = PROJECT_ROOT / "mcp_server" / "server.py"
    agent = MCPAgent(llm=llm, rag=rag, server_script=server_script)

    # Set up judge
    judge = LLMJudge(judge_model=JUDGE_MODEL)

    # Load dataset
    cases = load_golden_dataset(GOLDEN_DATASET_PATH)
    console.print(f"[green]✓ Loaded {len(cases)} test cases[/]\n")

    try:
        await agent.connect()
        console.print(f"[green]✓ Connected to MCP server ({len(agent.tools)} tools)[/]\n")

        # Run eval
        results = []
        for case in cases:
            result = await evaluate_one(agent, judge, case, console)
            results.append(result)

        # Save results to JSONL
        with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        # Summary
        summary = compute_summary(results)
        console.print("[bold magenta]═══ Evaluation Summary ═══[/]\n")

        table = Table(show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Agent Model", AGENT_MODEL)
        table.add_row("Judge Model", JUDGE_MODEL)
        table.add_row("Test Cases", str(summary["n_questions"]))
        table.add_row("Avg Groundedness", f"{summary['avg_groundedness']}/5")
        table.add_row("Avg Helpfulness", f"{summary['avg_helpfulness']}/5")
        table.add_row("Perfect Groundedness (5/5)", str(summary["perfect_groundedness_count"]))
        table.add_row("Poor Groundedness (≤2)", str(summary["poor_groundedness_count"]))
        table.add_row("Avg Total Latency", f"{summary['avg_total_latency_sec']} s")
        console.print(table)

        console.print(f"\n[green]✓ Results saved to:[/] {EVAL_RESULTS_PATH}")
        console.print("[bold green]✓ Phase 5 complete![/]")

    finally:
        await agent.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
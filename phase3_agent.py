"""Phase 3: Agent that routes between RAG and tool calls."""
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent import Agent
from src.config import DEFAULT_MODEL
from src.llm_client import LLMClient
from src.rag import RAGEngine

console = Console()


def run(agent: Agent, question: str) -> None:
    """Run one agent turn and display everything."""
    console.print(Panel(f"[bold cyan]Q:[/] {question}", border_style="cyan"))

    bundle = agent.ask(question)

    routing = bundle["routing"]
    execution = bundle["execution"]
    decision = routing["parsed"]

    # Show the routing decision
    console.print("\n[bold magenta]🧭 Routing decision:[/]")
    console.print(f"  [dim]Raw LLM output:[/] {routing['raw_response'][:200]}")
    console.print(f"  [dim]Parsed JSON:[/]")
    console.print_json(json.dumps(decision))

    # Show what was executed
    exec_type = execution.get("type", "unknown")
    console.print(f"\n[bold magenta]⚡ Executed:[/] [yellow]{exec_type}[/]")

    if exec_type == "tool":
        console.print(
            f"  [dim]Tool:[/] {execution['tool_name']}  "
            f"[dim]Args:[/] {execution['tool_args']}"
        )
        console.print(f"  [dim]Tool result:[/]")
        console.print_json(json.dumps(execution["tool_result"]))

    elif exec_type == "rag":
        console.print(f"  [dim]Retrieved {len(execution['chunks'])} chunks[/]")

    # Final answer
    console.print(f"\n[bold green]A:[/] {execution.get('answer', '(no answer)')}")

    # Latency summary
    table = Table(title="\nLatency Breakdown", show_header=True)
    table.add_column("Stage", style="cyan")
    table.add_column("Time", style="yellow")
    table.add_row("Router (LLM decision)", f"{routing['router_latency']:.2f} s")
    if "retrieval_time" in execution:
        table.add_row("Retrieval", f"{execution['retrieval_time']*1000:.1f} ms")
    if "generation_time" in execution:
        table.add_row("Final generation", f"{execution['generation_time']:.2f} s")
    console.print(table)
    console.print("─" * 70 + "\n")


def main() -> None:
    console.print("[bold magenta]═══ Phase 3: Agent (Routing + Tool Calling) ═══[/]\n")

    llm = LLMClient(model=DEFAULT_MODEL)
    rag = RAGEngine()
    agent = Agent(llm=llm, rag=rag)

    if rag.count() == 0:
        console.print("[bold red]✗ No documents in ChromaDB![/]")
        console.print("[yellow]  Run: python ingest.py first[/]")
        return

    # Test questions covering all paths
    questions = [
        # RAG queries
        "How long until a VPN session disconnects?",
        "What's the maximum budget for hardware requests without approval?",

        # Tool calls
        "What's the status of ticket IT-9921?",
        "Please request a new mouse for employee E-7788.",

        # Mixed/edge case
        "Check ticket IT-5555 status please",
    ]

    for q in questions:
        run(agent, q)

    console.print("[bold green]✓ Phase 3 complete![/]")


if __name__ == "__main__":
    main()
"""Phase 4: MCP-powered agent demo."""
import asyncio
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import DEFAULT_MODEL, PROJECT_ROOT
from src.llm_client import LLMClient
from src.mcp_agent import MCPAgent
from src.rag import RAGEngine

console = Console()


async def run(agent: MCPAgent, question: str) -> None:
    """Run one agent turn and display everything."""
    console.print(Panel(f"[bold cyan]Q:[/] {question}", border_style="cyan"))

    bundle = await agent.ask(question)
    routing = bundle["routing"]
    execution = bundle["execution"]
    decision = routing["parsed"]

    # Show routing decision
    console.print("\n[bold magenta]🧭 Routing decision:[/]")
    console.print(f"  [dim]Raw LLM output:[/] {routing['raw_response'][:200]}")
    console.print(f"  [dim]Parsed JSON:[/]")
    console.print_json(json.dumps(decision))

    # Show execution path
    exec_type = execution.get("type", "unknown")
    console.print(f"\n[bold magenta]⚡ Executed:[/] [yellow]{exec_type}[/]")

    if exec_type == "tool":
        console.print(
            f"  [dim]Tool (via MCP):[/] {execution['tool_name']}  "
            f"[dim]Args:[/] {execution['tool_args']}"
        )
        console.print(f"  [dim]Tool result from MCP server:[/]")
        console.print_json(json.dumps(execution["tool_result"]))

    elif exec_type == "rag":
        console.print(f"  [dim]Retrieved {len(execution['chunks'])} chunks[/]")

    # Final answer
    console.print(f"\n[bold green]A:[/] {execution.get('answer', '(no answer)')}")

    # Latency breakdown
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


async def main() -> None:
    console.print("[bold magenta]═══ Phase 4: Agent + MCP (dynamic tool discovery) ═══[/]\n")

    llm = LLMClient(model=DEFAULT_MODEL)
    rag = RAGEngine()

    if rag.count() == 0:
        console.print("[bold red]✗ No documents in ChromaDB![/]")
        console.print("[yellow]  Run: python ingest.py first[/]")
        return

    # Point to the MCP server script
    server_script = PROJECT_ROOT / "mcp_server" / "server.py"
    agent = MCPAgent(llm=llm, rag=rag, server_script=server_script)

    try:
        # Connect to MCP server (launches it as subprocess)
        console.print("[dim]Connecting to MCP server...[/]")
        await agent.connect()
        console.print(
            f"[bold green]✓ Connected![/] "
            f"Discovered [cyan]{len(agent.tools)}[/] tools via MCP:"
        )
        for tool in agent.tools:
            console.print(f"  • [yellow]{tool.name}[/] — {tool.description}")
        console.print()

        # Run the same test suite as Phase 3
        questions = [
            "How long until a VPN session disconnects?",
            "What's the maximum budget for hardware requests without approval?",
            "What's the status of ticket IT-9921?",
            "Please request a new mouse for employee E-7788.",
            "Check ticket IT-5555 status please",
        ]

        for q in questions:
            await run(agent, q)

        console.print("[bold green]✓ Phase 4 complete![/]")

    finally:
        # Always clean up the MCP subprocess
        console.print("[dim]Disconnecting from MCP server...[/]")
        await agent.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
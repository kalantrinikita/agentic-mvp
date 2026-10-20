"""Phase 1: Basic LLM query with latency tracking."""
from rich.console import Console
from rich.table import Table

from src.config import DEFAULT_MODEL
from src.llm_client import LLMClient

console = Console()


def run_query(client: LLMClient, prompt: str) -> dict:
    """Run a single query and display results."""
    console.print(f"\n[bold cyan]Model:[/] {client.model}")
    console.print(f"[bold cyan]Prompt:[/] {prompt}\n")

    result = client.generate(prompt)

    console.print(f"[bold green]Response:[/]\n{result['response']}\n")

    # Metrics table
    table = Table(title="Latency Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Time to First Token (TTFT)", f"{result['ttft']:.3f} s")
    table.add_row("Total Generation Time", f"{result['total_time']:.3f} s")
    table.add_row("Tokens Generated", str(result["tokens"]))
    if result["tokens"] > 0 and result["total_time"] > 0:
        tps = result["tokens"] / result["total_time"]
        table.add_row("Tokens/Second", f"{tps:.2f}")
    console.print(table)

    return result


def main() -> None:
    """Phase 1 entry point."""
    console.print("[bold magenta]═══ Phase 1: Baseline LLM Setup ═══[/]\n")

    client = LLMClient(model=DEFAULT_MODEL)

    test_prompts = [
        "What is the capital of France? Answer in one sentence.",
        "Explain quantum computing in exactly 2 sentences.",
        "List 3 benefits of local LLMs.",
    ]

    for prompt in test_prompts:
        run_query(client, prompt)
        console.print("─" * 60)

    console.print(
        "\n[bold green]✓ Done![/] Latency logs saved to [cyan]logs/latency.jsonl[/]"
    )


if __name__ == "__main__":
    main()
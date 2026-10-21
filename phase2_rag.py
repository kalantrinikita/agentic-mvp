"""Phase 2: RAG — answer IT questions using retrieved context."""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import DEFAULT_MODEL
from src.llm_client import LLMClient
from src.rag import RAGEngine

console = Console()


def ask(client: LLMClient, rag: RAGEngine, question: str, top_k: int = 2) -> None:
    """Run one RAG question: retrieve → generate → display."""
    console.print(Panel(f"[bold cyan]Q:[/] {question}", border_style="cyan"))

    # 1. Retrieve relevant chunks
    retrieval = rag.retrieve(question, top_k=top_k)

    # 2. Show what was retrieved (for transparency)
    console.print(f"\n[dim]Retrieved {len(retrieval['chunks'])} chunks in "
                  f"{retrieval['retrieval_time']*1000:.1f} ms[/]")
    for i, (chunk, dist) in enumerate(
        zip(retrieval["chunks"], retrieval["distances"]), 1
    ):
        preview = chunk[:120].replace("\n", " ")
        console.print(f"  [dim]#{i} (dist={dist:.3f}): {preview}...[/]")

    # 3. Generate answer using the chunks as context
    console.print("\n[bold green]A:[/]")
    result = client.generate_with_context(
        question=question,
        context_chunks=retrieval["chunks"],
        retrieval_seconds=retrieval["retrieval_time"],
    )
    console.print(result["response"])

    # 4. Show full latency breakdown
    table = Table(title="\nLatency Breakdown", show_header=True)
    table.add_column("Stage", style="cyan")
    table.add_column("Time", style="yellow")
    table.add_row("Retrieval", f"{retrieval['retrieval_time']*1000:.1f} ms")
    table.add_row("TTFT (LLM)", f"{result['ttft']*1000:.1f} ms")
    table.add_row("Total Generation", f"{result['total_time']:.2f} s")
    table.add_row(
        "End-to-End",
        f"{(retrieval['retrieval_time'] + result['total_time']):.2f} s",
    )
    console.print(table)
    console.print("─" * 70 + "\n")


def main() -> None:
    console.print("[bold magenta]═══ Phase 2: RAG-powered IT Helpdesk ═══[/]\n")

    client = LLMClient(model=DEFAULT_MODEL)
    rag = RAGEngine()

    if rag.count() == 0:
        console.print("[bold red]✗ No documents in ChromaDB![/]")
        console.print("[yellow]  Run: python ingest.py first[/]")
        return

    console.print(f"[dim]Loaded {rag.count()} chunks from ChromaDB[/]\n")

    # Test questions designed to exercise RAG
    questions = [
        "How long until a VPN session disconnects?",
        "What's the maximum budget for hardware requests without approval?",
        "Can I use my personal laptop for work?",
        "What's the capital of France?",  # Should refuse — not in docs!
    ]

    for q in questions:
        ask(client, rag, q)

    console.print("[bold green]✓ Phase 2 complete![/]")


if __name__ == "__main__":
    main()
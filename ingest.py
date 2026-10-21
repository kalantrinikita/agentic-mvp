"""Ingest the IT policy document into ChromaDB (one-time setup)."""
from rich.console import Console
from rich.table import Table

from src.config import DATA_DIR
from src.rag import RAGEngine

console = Console()


def main() -> None:
    console.print("[bold magenta]═══ Ingesting IT Policies into ChromaDB ═══[/]\n")

    # 1. Initialize the RAG engine
    rag = RAGEngine()

    # 2. Clean slate — delete any previous data to avoid duplicates
    if rag.count() > 0:
        console.print(
            f"[yellow]⚠ Found {rag.count()} existing chunks. Resetting collection...[/]"
        )
        rag.reset_collection()

    # 3. Ingest the IT policies document
    policy_file = DATA_DIR / "it_policies.txt"
    if not policy_file.exists():
        console.print(f"[bold red]✗ File not found: {policy_file}[/]")
        return

    console.print(f"[cyan]Loading:[/] {policy_file.name}")
    chunk_count = rag.ingest_document(policy_file, doc_id_prefix="it_policy")
    console.print(f"[green]✓ Stored {chunk_count} chunks in ChromaDB[/]\n")

    # 4. Smoke test — run a sample query to verify it works
    console.print("[bold magenta]═══ Smoke Test: Sample Retrieval ═══[/]\n")

    test_queries = [
        "How do I set up the VPN?",
        "What is the password policy?",
        "Can I work from a coffee shop?",
    ]

    for query in test_queries:
        console.print(f"[bold cyan]Query:[/] {query}")

        result = rag.retrieve(query, top_k=2)

        # Show retrieval metrics
        table = Table(show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Retrieval Time", f"{result['retrieval_time']*1000:.1f} ms")
        table.add_row("Chunks Retrieved", str(len(result["chunks"])))
        table.add_row("Top Distance", f"{result['distances'][0]:.4f}")
        console.print(table)

        # Show the retrieved chunks (first 150 chars each)
        for i, (chunk, dist) in enumerate(zip(result["chunks"], result["distances"]), 1):
            preview = chunk[:150].replace("\n", " ")
            console.print(f"  [bold]#{i}[/] (distance: {dist:.4f})")
            console.print(f"      {preview}...\n")

        console.print("─" * 60 + "\n")

    console.print("[bold green]✓ Ingestion complete! ChromaDB is ready to use.[/]")


if __name__ == "__main__":
    main()
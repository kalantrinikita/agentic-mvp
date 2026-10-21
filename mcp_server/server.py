"""IT Helpdesk MCP Server.

A standalone MCP server exposing IT support tools over the Model Context Protocol.
Can be used by any MCP-compatible client (Claude Desktop, Cursor, our custom agent).

Transport: stdio (standard input/output)

Uses MCP SDK v2.x API.
"""
import random
from datetime import datetime, timedelta
from typing import Any

from mcp.server.mcpserver import MCPServer


# Initialize the MCP server with a descriptive name
mcp = MCPServer("IT Helpdesk Server")


# ──────────────────────────────────────────────────────────
# TOOL 1: check_ticket_status
# ──────────────────────────────────────────────────────────

@mcp.tool()
def check_ticket_status(ticket_id: str) -> dict[str, Any]:
    """Check the current status of an IT support ticket by its ID.

    Args:
        ticket_id: The ticket ID (e.g., 'IT-9921').

    Returns:
        A dictionary containing ticket details: status, assignee,
        creation date, last update, and priority.
    """
    # Mock data — in production, this would query a real ticketing API
    mock_statuses = ["Open", "In Progress", "Resolved", "Pending Customer"]
    mock_assignees = ["Alice Chen", "Bob Kumar", "Priya Shah", "John Doe"]

    # Seed with ticket_id for reproducible mock responses
    random.seed(ticket_id)

    created_days_ago = random.randint(1, 30)
    status = random.choice(mock_statuses)
    assignee = random.choice(mock_assignees)

    return {
        "ticket_id": ticket_id,
        "status": status,
        "assigned_to": assignee,
        "created_at": (datetime.now() - timedelta(days=created_days_ago)).isoformat(),
        "last_updated": datetime.now().isoformat(),
        "priority": random.choice(["Low", "Medium", "High"]),
    }


# ──────────────────────────────────────────────────────────
# TOOL 2: request_hardware
# ──────────────────────────────────────────────────────────

@mcp.tool()
def request_hardware(item_name: str, employee_id: str) -> dict[str, Any]:
    """Submit a hardware request on behalf of an employee.

    Args:
        item_name: The hardware item requested (e.g., 'laptop', 'mouse', 'monitor').
        employee_id: The requesting employee's ID (e.g., 'E-7788').

    Returns:
        A dictionary containing the request confirmation with request ID,
        status, and estimated fulfillment timeline.
    """
    request_id = f"HW-{random.randint(1000, 9999)}"

    return {
        "request_id": request_id,
        "item_name": item_name,
        "employee_id": employee_id,
        "status": "Submitted",
        "estimated_fulfillment_days": random.randint(3, 7),
        "submitted_at": datetime.now().isoformat(),
        "message": f"Request for '{item_name}' submitted for employee {employee_id}.",
    }


# ──────────────────────────────────────────────────────────
# SERVER STARTUP
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run the server over stdio (standard MCP transport for local servers)
    mcp.run(transport="stdio")
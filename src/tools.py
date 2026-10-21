"""Mock tools the agent can call."""
import random
from datetime import datetime, timedelta


# ──────────────────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ──────────────────────────────────────────────────────────

def check_ticket_status(ticket_id: str) -> dict:
    """Check the status of an IT support ticket.

    Args:
        ticket_id: The ticket ID (e.g., 'IT-9921').

    Returns:
        dict with ticket details.
    """
    # Mock data — in real life, this would hit a ticketing API
    mock_statuses = ["Open", "In Progress", "Resolved", "Pending Customer"]
    mock_assignees = ["Alice Chen", "Bob Kumar", "Priya Shah", "John Doe"]

    # Use ticket_id as seed for reproducibility
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


def request_hardware(item_name: str, employee_id: str) -> dict:
    """Submit a hardware request for an employee.

    Args:
        item_name: The hardware item (e.g., 'laptop', 'mouse').
        employee_id: The employee's ID (e.g., 'E-7788').

    Returns:
        dict with request confirmation.
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
# TOOL REGISTRY — tells the agent what tools exist
# ──────────────────────────────────────────────────────────

TOOLS = {
    "check_ticket_status": {
        "function": check_ticket_status,
        "description": "Check the current status of an IT support ticket by its ID.",
        "parameters": {
            "ticket_id": "string — the ticket ID, e.g., 'IT-9921'",
        },
        "example": {
            "action": "tool_call",
            "tool": "check_ticket_status",
            "args": {"ticket_id": "IT-9921"},
        },
    },
    "request_hardware": {
        "function": request_hardware,
        "description": "Submit a hardware request on behalf of an employee.",
        "parameters": {
            "item_name": "string — the hardware item (e.g., 'laptop', 'monitor', 'mouse')",
            "employee_id": "string — employee ID (e.g., 'E-7788')",
        },
        "example": {
            "action": "tool_call",
            "tool": "request_hardware",
            "args": {"item_name": "mouse", "employee_id": "E-7788"},
        },
    },
}


def execute_tool(tool_name: str, args: dict) -> dict:
    """Dispatch a tool call to the right function."""
    if tool_name not in TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}

    tool_fn = TOOLS[tool_name]["function"]
    try:
        return tool_fn(**args)
    except TypeError as e:
        return {"error": f"Invalid arguments for {tool_name}: {e}"}
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}


def get_tools_description() -> str:
    """Generate a text description of available tools for the LLM prompt."""
    lines = []
    for name, info in TOOLS.items():
        lines.append(f"\nTool: {name}")
        lines.append(f"  Description: {info['description']}")
        lines.append(f"  Parameters:")
        for param, desc in info["parameters"].items():
            lines.append(f"    - {param}: {desc}")
        lines.append(f"  Example JSON: {info['example']}")
    return "\n".join(lines)
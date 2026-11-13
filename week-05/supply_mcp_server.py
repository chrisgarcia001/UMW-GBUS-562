"""MCP server offering Brookfield Bicycle Company supply tools to AI apps over stdio.

Tools (all days are counted from today; day 0 is today):
    get_part             A part's stock on hand, scheduled shipments, and the models that use it.
    get_orders_for_part  The open orders that need a part, earliest ship day first.
    check_coverage       Which orders a new arrival schedule for a part would leave short.

The data comes from the CSV files in week-05/brookfield-bikes/, read again on every call.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py
"""

import logging
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

import supply_tools


class Arrival(BaseModel):
    qty: int = Field(ge=1, description="Number of parts arriving.")
    day: int = Field(ge=0, description="Days from today when they arrive (0 is today).")


def answer(func, *args):
    try:
        return func(*args)
    except supply_tools.DataError as e:
        raise ToolError(str(e)) from e


def build_server():
    server = MCPServer("brookfield-supply")

    @server.tool()
    def get_part(part_id: str) -> dict:
        """Get a part's stock on hand, its scheduled shipments, and the bike models that use it.

        Takes a part ID such as MC-500. Each shipment gives its PO number, supplier,
        quantity, and arrives_in_days (days from today).
        """
        return answer(supply_tools.get_part, part_id)

    @server.tool()
    def get_orders_for_part(part_id: str) -> dict:
        """List the open customer orders that need a part, earliest ship day first.

        Returns the part_id and its orders. Each order gives its customer, model, bikes, parts_per_bike, units_needed
        (bikes times parts per bike), ship_day (days from today), and late_penalty.
        """
        return answer(supply_tools.get_orders_for_part, part_id)

    @server.tool()
    def check_coverage(part_id: str, arrivals: list[Arrival]) -> dict:
        """Check which orders for a part would be short under a new arrival schedule.

        arrivals replaces ALL of the part's scheduled shipments; stock on hand still
        counts. Give every shipment still expected, each as a quantity and a day from
        today. An empty list means nothing more arrives.

        Orders are filled complete, earliest ship day first, so a short order holds up
        the orders after it. Parts arriving on an order's ship day count for it. Each
        order comes back "covered", or "short" with units_short, ready_day (the first day
        it can ship complete, or null if these arrivals never cover it), and days_late.
        """
        return answer(supply_tools.check_coverage, part_id, [a.model_dump() for a in arrivals])

    return server


def main():
    # stdout carries the MCP protocol, so logs must go to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    build_server().run("stdio")


if __name__ == "__main__":
    main()

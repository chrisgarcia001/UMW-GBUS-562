"""MCP server offering Hartwell Foods HR tools to AI apps over stdio.

Tools:
    search_policy      Returns the 4 HR document chunks closest to a search query.
    get_leave_balance  Looks up a leave balance from hartwell-hr/employees.csv.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py
        HR staff mode: get_leave_balance takes any employee's work email.
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --as-employee dana.whitfield@hartwellfoods.example
        Employee mode: get_leave_balance takes no inputs and returns only that employee's balance.
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --rebuild
        Rebuilds the Chroma database (week-04/chroma_db/), then exits.

The database is also built on the first search if it's missing, and rebuilt on a search
if the documents have changed.
"""

import argparse
import logging
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import hr_tools


def lookup(email):
    try:
        return hr_tools.leave_balance(email)
    except hr_tools.EmployeeRecordError as e:
        raise ToolError(str(e)) from e


def build_server(as_employee=None):
    server = MCPServer("hartwell-hr")

    @server.tool()
    def search_policy(query: str) -> list[dict]:
        """Search Hartwell Foods' HR policy documents.

        Returns the 4 closest sections, each with its source file, title, section,
        version note, similarity score, and text. The closest sections may not answer
        the question, so check that the text actually does.
        """
        if not query.strip():
            raise ToolError("The search query is empty. Provide words to search for.")
        return hr_tools.search_docs(query)

    if as_employee is None:
        @server.tool()
        def get_leave_balance(email: str) -> dict:
            """Get an employee's vacation, sick, and personal days left, for HR staff.

            Takes the employee's exact work email address.
            """
            return lookup(email)
    else:
        @server.tool()
        def get_leave_balance() -> dict:
            """Get the vacation, sick, and personal days left for the employee using this session.

            Takes no inputs. No other employee's balance is available.
            """
            return lookup(as_employee)

    return server


def main():
    parser = argparse.ArgumentParser(description="MCP server with Hartwell Foods HR tools.")
    parser.add_argument("--as-employee", metavar="EMAIL",
                        help="limit get_leave_balance to this employee; it then takes no inputs")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the Chroma database, then exit")
    args = parser.parse_args()

    # stdout carries the MCP protocol, so logs must go to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")

    if args.rebuild:
        hr_tools.rebuild_index()
        return
    build_server(args.as_employee).run("stdio")


if __name__ == "__main__":
    main()

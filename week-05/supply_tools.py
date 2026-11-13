"""Brookfield Bicycle Company supply data, read from the CSV files in brookfield-bikes/.

Used by supply_mcp_server.py. The files are read again on every call, so edits to them
show up straight away. All days are counted from today: day 0 is today.

Every order in orders.csv is treated as open.
"""

import csv
import logging
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "brookfield-bikes"

log = logging.getLogger("supply_tools")


class DataError(Exception):
    """A request can't be answered. The message is shown to the model."""


# ---------------------------------------------------------------------------
# Reading the CSV files
# ---------------------------------------------------------------------------


def read_csv(name):
    path = DATA_DIR / name
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except OSError as e:
        log.error("Could not read %s: %r", path, e)
        raise DataError(f"The data file {name} could not be read.") from e


def as_int(row, field, name):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as e:
        log.error("Bad %s in %s: %r", field, name, row)
        raise DataError(f"The data file {name} has a missing or invalid {field}.") from e


def find_part(part_id):
    """Return the parts.csv row for part_id (ignoring case and spaces), or raise DataError."""
    wanted = part_id.strip().upper()
    parts = read_csv("parts.csv")
    for row in parts:
        if (row.get("part_id") or "").strip().upper() == wanted:
            return row
    known = ", ".join(sorted(r.get("part_id", "") for r in parts))
    raise DataError(f"No part matches '{part_id}'. Known part IDs: {known}.")


def models_using(part_id):
    """Return [{model, qty_per_bike}] for each bike model that uses the part."""
    return [
        {"model": row["model"], "qty_per_bike": as_int(row, "qty_per_bike", "bill-of-materials.csv")}
        for row in read_csv("bill-of-materials.csv")
        if row.get("part_id", "").strip().upper() == part_id
    ]


# ---------------------------------------------------------------------------
# The three tools
# ---------------------------------------------------------------------------


def get_part(part_id):
    """Return a part's stock on hand, scheduled shipments, and the models that use it."""
    row = find_part(part_id)
    pid = row["part_id"].strip().upper()
    shipments = sorted(
        (
            {
                "po_number": s["po_number"],
                "supplier": s["supplier"],
                "qty": as_int(s, "qty", "inbound.csv"),
                "arrives_in_days": as_int(s, "arrives_in_days", "inbound.csv"),
            }
            for s in read_csv("inbound.csv")
            if s.get("part_id", "").strip().upper() == pid
        ),
        key=lambda s: s["arrives_in_days"],
    )
    return {
        "part_id": pid,
        "description": row["description"],
        "supplier": row["supplier"],
        "on_hand": as_int(row, "on_hand", "parts.csv"),
        "scheduled_shipments": shipments,
        "used_in_models": models_using(pid),
    }


def get_orders_for_part(part_id):
    """Return the open orders that need the part, earliest ship day first."""
    pid = find_part(part_id)["part_id"].strip().upper()
    per_bike = {m["model"]: m["qty_per_bike"] for m in models_using(pid)}
    orders = []
    for row in read_csv("orders.csv"):
        if row.get("model") not in per_bike:
            continue
        bikes = as_int(row, "qty", "orders.csv")
        penalty = (row.get("late_penalty") or "").strip().lower()
        if penalty not in ("yes", "no"):
            log.error("Bad late_penalty in orders.csv: %r", row)
            raise DataError("The data file orders.csv has a missing or invalid late_penalty.")
        orders.append({
            "order_id": row["order_id"],
            "customer": row["customer"],
            "model": row["model"],
            "bikes": bikes,
            "parts_per_bike": per_bike[row["model"]],
            "units_needed": bikes * per_bike[row["model"]],
            "ship_day": as_int(row, "ship_in_days", "orders.csv"),
            "late_penalty": penalty == "yes",
        })
    orders.sort(key=lambda o: (o["ship_day"], o["order_id"]))
    return {"part_id": pid, "orders": orders}


def check_arrivals(arrivals):
    """Return the arrivals as [{qty, day}] sorted by day, or raise DataError."""
    checked = []
    for i, a in enumerate(arrivals, start=1):
        qty, day = a.get("qty"), a.get("day")
        if isinstance(qty, bool) or not isinstance(qty, int) or qty < 1:
            raise DataError(f"Arrival {i}: qty must be a whole number of at least 1.")
        if isinstance(day, bool) or not isinstance(day, int) or day < 0:
            raise DataError(f"Arrival {i}: day must be a whole number of days from today, 0 or more.")
        checked.append({"qty": qty, "day": day})
    return sorted(checked, key=lambda a: a["day"])


def check_coverage(part_id, arrivals):
    """Check which orders the stock on hand plus a new arrival schedule can cover.

    The arrivals replace all of the part's scheduled shipments. Parts that arrive on an
    order's ship day can be used for it. Orders are filled complete, earliest ship day
    first, so an order that is short holds up every order after it.
    """
    part = get_part(part_id)
    arrivals = check_arrivals(arrivals)
    orders = get_orders_for_part(part["part_id"])["orders"]

    # Total parts available by the end of each day on which stock changes.
    steps = [(0, part["on_hand"])]
    for a in arrivals:
        steps.append((a["day"], steps[-1][1] + a["qty"]))

    def available_by(day):
        return max(total for d, total in steps if d <= day)

    def first_day_with(needed):
        return next((d for d, total in steps if total >= needed), None)

    results = []
    needed_so_far = 0
    for o in orders:
        needed_so_far += o["units_needed"]
        short = min(o["units_needed"], max(0, needed_so_far - available_by(o["ship_day"])))
        result = {k: o[k] for k in ("order_id", "customer", "model", "units_needed", "ship_day", "late_penalty")}
        if short == 0:
            result["status"] = "covered"
        else:
            ready = first_day_with(needed_so_far)
            result["status"] = "short"
            result["units_short"] = short
            # The first day the order can ship complete; None if these arrivals never cover it.
            result["ready_day"] = ready
            result["days_late"] = None if ready is None else ready - o["ship_day"]
        results.append(result)

    return {
        "part_id": part["part_id"],
        "on_hand": part["on_hand"],
        "arrivals": arrivals,
        "total_supply": steps[-1][1],
        "total_needed": needed_so_far,
        "orders": results,
        "orders_short": [r["order_id"] for r in results if r["status"] == "short"],
    }

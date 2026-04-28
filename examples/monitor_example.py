"""ParallelMonitor example: create / retrieve / delete a monitor (alpha)."""

from __future__ import annotations

from langchain_parallel import ParallelMonitor

# Set your API key: export PARALLEL_API_KEY="your-api-key"


def monitor_crud_cycle() -> None:
    print("=== ParallelMonitor: create / retrieve / delete ===")
    client = ParallelMonitor()

    created = client.create(
        query="Latest peer-reviewed papers on net-energy-gain fusion",
        frequency="1d",
        metadata={"label": "example-monitor-fusion"},
    )
    monitor_id = created["monitor_id"]
    print("created:", monitor_id, "/ status:", created["status"])

    got = client.retrieve(monitor_id)
    print("retrieved query:", got["query"][:80])

    events = client.list_events(monitor_id, lookback_period="7d")
    print("events:", len(events.get("events", [])))

    deleted = client.delete(monitor_id)
    print("deleted status:", deleted["status"])


def main() -> None:
    monitor_crud_cycle()


if __name__ == "__main__":
    main()

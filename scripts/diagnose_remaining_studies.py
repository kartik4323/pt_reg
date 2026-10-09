"""Read saved suite failures without importing or modifying experiment code."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def log_tail(path, lines=50, limit=4500):
    if not path.is_file():
        return "(log missing)"
    # Bound memory even when a generation log contains extensive model output.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        value = stream.read().decode("utf-8", errors="replace")
    return "\n".join(value.splitlines()[-lines:])[-limit:]


def diagnose(root, samples=1):
    root = Path(root).expanduser().resolve()
    status = read_json(root / "suite_status.json")
    if not status:
        raise ValueError("No readable suite_status.json at " + str(root))
    print("ROOT:", root)
    print("TASK STATES:", dict(Counter(status.get("tasks", {}).values())))
    print("ACTIVE:", status.get("active", []))
    print("SCHEDULER LOCK PRESENT:", (root / "suite.lock").exists())
    print("Saved files only; process liveness and termination causes are not inferred.")
    for task, state in sorted(status.get("tasks", {}).items()):
        if state not in ("failed", "complete_with_failures"):
            continue
        run = root / "runs" / task
        print("\nTASK:", task, "STATE:", state)
        result = read_json(run / "task_result.json")
        print("TASK RESULT:", "present" if result else "absent or unreadable")
        counts, examples = Counter(), {}
        for path in sorted((run / "jobs").glob("*/result.json")):
            row = read_json(path)
            stage = row.get("stage", "unknown")
            counts[(stage, row.get("status", "unreadable"))] += 1
            if row.get("status") == "failed":
                examples.setdefault(stage, []).append((path.parent, row))
        print("JOB COUNTS:", {stage + ":" + state: n for (stage, state), n in sorted(counts.items())})
        if state == "failed":
            path = root / "logs" / (task + ".log")
            print("TASK LOG:", path)
            print(log_tail(path, lines=30))
        if result.get("errors"):
            print("FIRST WORKER ERROR:", str(result["errors"][0])[-1500:])
        for stage, jobs in sorted(examples.items()):
            for directory, row in jobs[:samples]:
                print("\nFAILED JOB:", stage, row.get("arm"), directory.name)
                print("OUTER ERROR:", str(row.get("error", ""))[-1500:])
                request = read_json(directory / "request.json")
                config = request.get("config") or {}
                if isinstance(config, dict):
                    print("MODEL:", request.get("model"), "DEVICE:", config.get("device"))
                path = directory / "worker.log"
                if path.is_file():
                    print("CHILD WORKER LOG:", path)
                    print(log_tail(path))
                else:
                    print("JOB ERROR LOG:", directory / "error.txt")
                    print(log_tail(directory / "error.txt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--samples-per-stage", type=int, default=1)
    args = parser.parse_args()
    if args.samples_per_stage < 1:
        parser.error("--samples-per-stage must be positive")
    diagnose(args.root, args.samples_per_stage)


if __name__ == "__main__":
    main()

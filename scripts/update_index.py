#!/usr/bin/env python3
"""Scan web/data runs and update web/data/index.json."""
import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to web/data")
    args = parser.parse_args()

    runs = []
    for name in sorted(os.listdir(args.data)):
        run_dir = os.path.join(args.data, name)
        if not os.path.isdir(run_dir):
            continue
        run_json = os.path.join(run_dir, "run.json")
        if not os.path.exists(run_json):
            continue
        with open(run_json, "r", encoding="utf-8") as f:
            run = json.load(f)
        label = run.get("prompt", name)
        runs.append({"id": name, "label": label})

    index = {"runs": runs}
    with open(os.path.join(args.data, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Updated index.json with {len(runs)} runs")


if __name__ == "__main__":
    main()

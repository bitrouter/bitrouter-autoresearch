"""Aggregate benchmark results and produce statistical comparisons."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def aggregate(results_dir: Path) -> list[dict]:
    """Read all verdict + metadata files and return a flat list of records."""
    rows: list[dict] = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        verdict_path = run_dir / "verdict.json"
        meta_path = run_dir / "metadata.json"
        if not verdict_path.exists() or not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        with open(verdict_path) as f:
            verdict = json.load(f)

        rows.append({
            "trial_id": meta["trial_id"],
            "task_id": meta["task_id"],
            "condition": meta["condition"],
            "difficulty": meta["task_id"].split("/")[0],
            "composite_score": verdict["composite_score"],
            "completion_score": verdict["completion_score"],
            "interaction_quality": verdict["interaction_quality"],
            "resource_efficiency": verdict["resource_efficiency"],
            "cost_usd": meta.get("total_cost_usd", 0),
            "turn_count": meta.get("turn_count", 0),
            "stop_reason": meta.get("stop_reason", ""),
        })

    return rows


def compare_conditions(rows: list[dict]) -> dict:
    """Produce a statistical comparison between conditions."""
    conditions: dict[str, list[float]] = {}
    for row in rows:
        cond = row["condition"]
        conditions.setdefault(cond, []).append(row["composite_score"])

    summary = {}
    for cond, scores in conditions.items():
        summary[cond] = {
            "n": len(scores),
            "mean": statistics.mean(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }

    # Win/loss/tie per task
    if len(conditions) == 2:
        cond_names = sorted(conditions.keys())
        task_scores: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            task_scores.setdefault(row["task_id"], {}).setdefault(
                row["condition"], [],
            ).append(row["composite_score"])

        wins = {c: 0 for c in cond_names}
        ties = 0
        for task_id, cond_data in task_scores.items():
            avgs = {
                c: statistics.mean(cond_data.get(c, [0]))
                for c in cond_names
            }
            if abs(avgs[cond_names[0]] - avgs[cond_names[1]]) < 0.01:
                ties += 1
            elif avgs[cond_names[0]] > avgs[cond_names[1]]:
                wins[cond_names[0]] += 1
            else:
                wins[cond_names[1]] += 1

        summary["comparison"] = {
            "wins": wins,
            "ties": ties,
            "total_tasks": len(task_scores),
        }

    return summary


if __name__ == "__main__":
    import sys

    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/runs")
    rows = aggregate(results_path)

    if not rows:
        print("No results found.")
        sys.exit(1)

    comparison = compare_conditions(rows)
    print(json.dumps(comparison, indent=2))

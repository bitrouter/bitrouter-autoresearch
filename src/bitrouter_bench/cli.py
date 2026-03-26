"""CLI entry point for BitRouter Bench."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from bitrouter_bench.config import BenchConfig, budget_for_difficulty, load_config

console = Console()


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=False),
    default="configs/bench.yaml",
    help="Path to bench config YAML.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, config_path: str, verbose: bool) -> None:
    """BitRouter Bench -- benchmark harness for evaluating LLM routing."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(Path(config_path))


@main.command()
@click.option("--difficulty", type=click.Choice(["easy", "medium", "hard"]))
@click.option("--condition", type=str, help="Run only this condition.")
@click.option("--task", "task_filter", type=str, help="Filter tasks by ID substring.")
@click.option("--repeats", type=int, help="Override repeat count.")
@click.option(
    "--live",
    is_flag=True,
    help="Start API server for live observation (default: port 8788).",
)
@click.option("--port", type=int, default=8788, help="API server port (with --live).")
@click.option("--no-shuffle", is_flag=True, help="Disable trial order randomization.")
@click.option("--skip-preflight", is_flag=True, help="Skip pre-flight checks (not recommended).")
@click.pass_context
def run(
    ctx: click.Context,
    difficulty: str | None,
    condition: str | None,
    task_filter: str | None,
    repeats: int | None,
    live: bool,
    port: int,
    no_shuffle: bool,
    skip_preflight: bool,
) -> None:
    """Run benchmark trials."""
    from bitrouter_bench.judge import Judge, save_verdict
    from bitrouter_bench.openclaw import OpenClawRunner
    from bitrouter_bench.runner import BenchRunner
    from bitrouter_bench.trajectory import trial_dir

    config: BenchConfig = ctx.obj["config"]

    console.print("[bold]BitRouter Bench[/bold] -- running trials")
    console.print(f"  BitRouter: {config.bitrouter_url}")
    console.print(f"  Tasks dir: {config.tasks_dir}")
    if live:
        console.print(f"  Live API:  http://localhost:{port}")
    console.print()

    # Pre-flight checks (hard gate)
    if not skip_preflight:
        from bitrouter_bench.preflight import Preflight

        console.print("[bold]Running pre-flight checks...[/bold]")
        try:
            asyncio.run(Preflight(config).run())
            console.print("[green]All pre-flight checks passed.[/green]\n")
        except SystemExit:
            raise
    else:
        console.print("[yellow]Skipping pre-flight checks (--skip-preflight)[/yellow]\n")

    runner = BenchRunner(config, live=live)
    openclaw = OpenClawRunner(openclaw_bin=config.openclaw_bin)
    judge = Judge(
        openclaw=openclaw,
        agent_id=config.agent_id_judge,
        weights=(config.weight_completion, config.weight_interaction, config.weight_efficiency),
    )

    async def _run_with_server() -> list:
        """Run trials, optionally with an API server in the background."""
        if live:
            from bitrouter_bench.api import create_app

            import uvicorn

            app = create_app(config)
            server_config = uvicorn.Config(
                app, host="0.0.0.0", port=port, log_level="warning",
            )
            server = uvicorn.Server(server_config)
            server_task = asyncio.create_task(server.serve())

            await asyncio.sleep(0.5)
            console.print(f"[green]Live API server started on port {port}[/green]\n")

        trajectories = await runner.run(
            task_filter=task_filter,
            difficulty_filter=difficulty,
            condition_filter=condition,
            repeats=repeats,
            shuffle=not no_shuffle,
        )

        if live:
            await asyncio.sleep(2)
            server.should_exit = True
            await server_task

        return trajectories

    trajectories = asyncio.run(_run_with_server())

    if not trajectories:
        console.print("[yellow]No trials were run.[/yellow]")
        return

    # Judge all trajectories
    console.print(f"\n[bold]Judging {len(trajectories)} trials...[/bold]\n")

    from bitrouter_bench.task_loader import discover_tasks

    tasks = discover_tasks(config.tasks_dir)
    task_map = {t.task_id: t for t in tasks}

    results_table = Table(title="Trial Results")
    results_table.add_column("Task", style="cyan")
    results_table.add_column("Condition", style="magenta")
    results_table.add_column("Completion", justify="right")
    results_table.add_column("Interaction", justify="right")
    results_table.add_column("Efficiency", justify="right")
    results_table.add_column("Composite", justify="right", style="bold")
    results_table.add_column("Cost", justify="right")
    results_table.add_column("Turns", justify="right")
    results_table.add_column("Stop", style="dim")

    for traj in trajectories:
        task = task_map.get(traj.task_id)
        if not task:
            console.print(f"[yellow]Task not found for {traj.task_id}, skipping judge[/yellow]")
            continue

        budget = (
            task.meta.budget_usd
            if task.meta.budget_usd is not None
            else budget_for_difficulty(config, task.meta.difficulty)
        )

        verdict = asyncio.run(
            judge.evaluate(
                task, traj,
                workspace=traj.workspace or None,
                budget_usd=budget,
            ),
        )

        output = trial_dir(config.results_dir, traj.trial_id)
        save_verdict(output, verdict)

        results_table.add_row(
            traj.task_id,
            traj.condition,
            f"{verdict.completion_score:.2f}",
            f"{verdict.interaction_quality:.2f}",
            f"{verdict.resource_efficiency:.2f}",
            f"{verdict.composite_score:.2f}",
            f"${traj.total_cost_usd:.4f}",
            str(len(traj.turns)),
            traj.stop_reason,
        )

    console.print(results_table)


@main.command("generate-tasks")
@click.option("--count", type=int, default=10, help="Number of tasks to generate.")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option(
    "--output",
    "output_dir",
    type=click.Path(),
    default=None,
    help="Output directory (default: tasks/generated/).",
)
@click.pass_context
def generate_tasks(
    ctx: click.Context,
    count: int,
    seed: int | None,
    output_dir: str | None,
) -> None:
    """Generate benchmark tasks using LLM."""
    from bitrouter_bench.openclaw import OpenClawRunner
    from bitrouter_bench.task_generator import GenerationConfig, TaskGenerator

    config: BenchConfig = ctx.obj["config"]
    out = Path(output_dir) if output_dir else config.generated_tasks_dir

    gen_config = GenerationConfig(
        max_tasks_per_run=count,
        seed=seed or config.generate_seed,
    )

    openclaw = OpenClawRunner(openclaw_bin=config.openclaw_bin)
    generator = TaskGenerator(openclaw=openclaw, config=gen_config)

    console.print(f"[bold]Generating {count} benchmark tasks...[/bold]")
    console.print(f"  Output: {out}")
    console.print(f"  Seed: {seed or 'random'}")
    console.print()

    tasks = asyncio.run(generator.generate_and_save(out, count=count))

    if not tasks:
        console.print("[yellow]No tasks were generated.[/yellow]")
        return

    table = Table(title=f"Generated {len(tasks)} Tasks")
    table.add_column("Task ID", style="cyan")
    table.add_column("Difficulty", style="magenta")
    table.add_column("Category")
    table.add_column("Assertions", justify="right")
    table.add_column("Description")

    for task in tasks:
        table.add_row(
            task.task_id,
            task.meta.difficulty,
            task.meta.category,
            str(len(task.eval_criteria.programmatic_assertions)),
            task.meta.description[:60],
        )

    console.print(table)


@main.command()
@click.option("--port", type=int, default=8788, help="Port to listen on.")
@click.option("--host", type=str, default="0.0.0.0", help="Host to bind to.")
@click.pass_context
def serve(ctx: click.Context, port: int, host: str) -> None:
    """Start the read-only API server (for browsing results without running trials)."""
    import uvicorn

    from bitrouter_bench.api import create_app

    config: BenchConfig = ctx.obj["config"]
    app = create_app(config)

    console.print("[bold]BitRouter Bench API[/bold]")
    console.print(f"  http://{host}:{port}")
    console.print(f"  Docs: http://localhost:{port}/docs")
    console.print()

    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command("list-tasks")
@click.option("--difficulty", type=click.Choice(["easy", "medium", "hard"]))
@click.pass_context
def list_tasks(ctx: click.Context, difficulty: str | None) -> None:
    """List all discovered tasks."""
    from bitrouter_bench.task_loader import discover_tasks

    config: BenchConfig = ctx.obj["config"]
    tasks = discover_tasks(config.tasks_dir, difficulty)

    if not tasks:
        console.print("[yellow]No tasks found.[/yellow]")
        return

    table = Table(title="Benchmark Tasks")
    table.add_column("Task ID", style="cyan")
    table.add_column("Difficulty", style="magenta")
    table.add_column("Category")
    table.add_column("Budget", justify="right")
    table.add_column("Max Turns", justify="right")
    table.add_column("Description")

    for task in tasks:
        budget = task.meta.budget_usd or budget_for_difficulty(
            config, task.meta.difficulty,
        )
        table.add_row(
            task.task_id,
            task.meta.difficulty,
            task.meta.category,
            f"${budget:.2f}",
            str(task.meta.max_turns),
            task.meta.description[:60],
        )

    console.print(table)


@main.command("validate-tasks")
@click.pass_context
def validate_tasks(ctx: click.Context) -> None:
    """Validate all TASK.md files parse correctly."""
    from bitrouter_bench.task_loader import discover_tasks

    config: BenchConfig = ctx.obj["config"]

    try:
        tasks = discover_tasks(config.tasks_dir)
    except Exception as exc:
        console.print(f"[red]Parse error: {exc}[/red]")
        raise SystemExit(1) from exc

    console.print(f"[green]All {len(tasks)} tasks parsed successfully.[/green]")
    for task in tasks:
        scenario = task.scenario
        criteria = task.eval_criteria
        console.print(
            f"  {task.task_id}: "
            f"persona={'yes' if scenario.persona else 'no'}, "
            f"known_info={'yes' if scenario.known_info else 'no'}, "
            f"assertions={len(criteria.programmatic_assertions)}, "
            f"llm_judge={'yes' if criteria.llm_judge_prompt else 'no'}",
        )


@main.command()
@click.option("--format", "fmt", type=click.Choice(["table", "csv", "json"]), default="table")
@click.pass_context
def results(ctx: click.Context, fmt: str) -> None:
    """Aggregate and display benchmark results."""
    config: BenchConfig = ctx.obj["config"]
    runs_dir = config.results_dir

    if not runs_dir.exists():
        console.print("[yellow]No results found.[/yellow]")
        return

    rows: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()):
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
            "task_id": meta["task_id"],
            "condition": meta["condition"],
            "composite": verdict["composite_score"],
            "completion": verdict["completion_score"],
            "interaction": verdict["interaction_quality"],
            "efficiency": verdict["resource_efficiency"],
            "cost": meta.get("total_cost_usd", 0),
            "turns": meta.get("turn_count", 0),
            "stop_reason": meta.get("stop_reason", ""),
        })

    if not rows:
        console.print("[yellow]No judged results found.[/yellow]")
        return

    if fmt == "json":
        console.print(json.dumps(rows, indent=2))
        return

    if fmt == "csv":
        import csv
        import sys

        writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return

    # Table format with condition comparison
    table = Table(title="Benchmark Results")
    table.add_column("Task", style="cyan")
    table.add_column("Condition", style="magenta")
    table.add_column("Composite", justify="right", style="bold")
    table.add_column("Completion", justify="right")
    table.add_column("Interaction", justify="right")
    table.add_column("Efficiency", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Turns", justify="right")

    for row in sorted(rows, key=lambda r: (r["task_id"], r["condition"])):
        table.add_row(
            row["task_id"],
            row["condition"],
            f"{row['composite']:.2f}",
            f"{row['completion']:.2f}",
            f"{row['interaction']:.2f}",
            f"{row['efficiency']:.2f}",
            f"${row['cost']:.4f}",
            str(row["turns"]),
        )

    console.print(table)

    # Summary by condition
    conditions = set(r["condition"] for r in rows)
    if len(conditions) > 1:
        console.print("\n[bold]Summary by Condition[/bold]")
        summary_table = Table()
        summary_table.add_column("Condition", style="magenta")
        summary_table.add_column("Avg Composite", justify="right", style="bold")
        summary_table.add_column("Avg Cost", justify="right")
        summary_table.add_column("Trials", justify="right")

        for cond in sorted(conditions):
            cond_rows = [r for r in rows if r["condition"] == cond]
            avg_score = sum(r["composite"] for r in cond_rows) / len(cond_rows)
            avg_cost = sum(r["cost"] for r in cond_rows) / len(cond_rows)
            summary_table.add_row(
                cond,
                f"{avg_score:.3f}",
                f"${avg_cost:.4f}",
                str(len(cond_rows)),
            )

        console.print(summary_table)

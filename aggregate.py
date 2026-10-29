import csv, json, statistics, sys

from run import RUNS_DIR

# Combines the summaries of several runs, e.g. the seeds of one Slurm job array:
#   uv run python aggregate.py "L0_open_water_sac_seed*_job12345"
#   uv run python aggregate.py "L0_open_water_*"      (any glob pattern on the folder names in results/runs/)
# Prints one row per run and the mean and spread across the runs, and saves the rows as
# results/aggregate_<pattern>.csv. Runs without a summary.json are skipped.

COLUMNS = ("run", "solver", "solver_seed", "status", "episodes", "runtime_h", "collision_episodes", "first_collision_episode",
           "best_reward", "best_closest_distance", "most_likely_collision_log_likelihood",
           "last_10_percent_mean_reward", "eval_collisions", "eval_starts", "eval_median_closest_distance")
NUMERIC = ("runtime_h", "collision_episodes", "first_collision_episode", "best_reward", "best_closest_distance",
           "most_likely_collision_log_likelihood", "last_10_percent_mean_reward", "eval_collisions", "eval_median_closest_distance")

def aggregate(pattern) -> list:
    rows = []
    for runDir in sorted(RUNS_DIR.glob(pattern)):
        if not (runDir / "summary.json").is_file():
            print(f'Skipped {runDir.name}: no summary.json')
            continue
        with open(runDir / "summary.json") as f:
            summary = json.load(f)
        with open(runDir / "settings.json") as f:
            settings = json.load(f)["run"]
        first, likely, evaluation = summary["first_collision"], summary.get("most_likely_collision"), summary.get("evaluation")
        rows.append({
            "run": runDir.name, "solver": settings["solver"], "solver_seed": settings["solver_seed"], "status": summary["status"],
            "episodes": summary["episodes"], "runtime_h": round(summary["runtime_seconds"]/3600, 2),
            "collision_episodes": summary["collision_episodes"],
            "first_collision_episode": None if first is None else first["episode"],
            "best_reward": summary["best"]["reward"], "best_closest_distance": summary["best"]["closest_distance"],
            "most_likely_collision_log_likelihood": None if likely is None else likely["log_likelihood"],
            "last_10_percent_mean_reward": summary["progress"]["last_10_percent"]["mean_reward"],
            "eval_collisions": None if evaluation is None else evaluation["collisions"],
            "eval_starts": None if evaluation is None else evaluation["starts"],
            "eval_median_closest_distance": None if evaluation is None else evaluation["median_closest_distance"]})
    return rows

def print_table(rows) -> None:
    widths = {column: max(len(column), *(len(str(row[column])) for row in rows)) for column in COLUMNS}
    print("  ".join(column.ljust(widths[column]) for column in COLUMNS))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in COLUMNS))
    print()
    print(f'{len(rows)} runs. Mean (standard deviation) over the runs that have a value:')
    for column in NUMERIC:
        values = [row[column] for row in rows if row[column] is not None]
        if values:
            spread = f' ({statistics.stdev(values):.3g})' if len(values) > 1 else ''
            print(f'  {column:<40} {statistics.mean(values):.4g}{spread}   [{len(values)} of {len(rows)} runs]')
    withCollision = sum(row["collision_episodes"] > 0 for row in rows)
    print(f'  {"runs that found a collision":<40} {withCollision} of {len(rows)}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Usage: uv run python aggregate.py "<glob pattern for folders in results/runs/>"')
    pattern = sys.argv[1]
    rows = aggregate(pattern)
    if not rows:
        sys.exit(f'No finished runs match {pattern!r} in {RUNS_DIR}')
    print_table(rows)
    outFile = RUNS_DIR.parent / f'aggregate_{"".join(c if c.isalnum() or c in "-_" else "_" for c in pattern)}.csv'
    with open(outFile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved {outFile}')

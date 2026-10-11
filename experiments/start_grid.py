"""Map where MCTS finds crashes as a function of the steerable boat's start position.

For every steerable start (x, y) on a 10-unit grid over the 100x100 world, runs
MCTSController.loop() for several random seeds and records how often a crash is
found. The straight boat keeps its start from parameters.json.

Outputs results/start_grid.csv and figs/start_grid.png.

Run from the repo root:  uv run python experiments/start_grid.py
"""
import contextlib, csv, io, json, os, random, shutil, statistics, sys, tempfile
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")  # Never open plot windows
import matplotlib.pyplot as plt
import numpy as np

from mcts.mctsController import MCTSController

STARTS = range(0, 101, 2)
SEEDS = range(5)
ITERATIONS = 2000
CSV_PATH = REPO / "results" / "start_grid.csv"
FIG_PATH = REPO / "figs" / "start_grid.png"


class QuietMCTS(MCTSController):  # No stats printout or plots

    def stats(self, Gs, runTime):
        pass


def init_worker(tmpRoot):
    # loop() writes crashStates.json to the working directory, and the sim reads
    # parameters.json from it. Give each worker its own directory with a copy of
    # parameters.json so the repo's crashStates.json is never overwritten.
    workDir = tempfile.mkdtemp(dir=tmpRoot)
    shutil.copy(REPO / "parameters.json", workDir)
    os.chdir(workDir)


def run_one(job):
    x, y, seed = job
    random.seed(seed)
    mcts = QuietMCTS(steerable_start=(x, y))

    # Record the loop iteration of the first crash. loop() calls simulate() once per
    # iteration on a freshly reset sim (empty trace); the recursive calls never are.
    iteration = -1
    firstCrash = None
    simulate = mcts.simulate
    def tracked_simulate():
        nonlocal iteration, firstCrash
        topLevel = len(mcts.simIntefrace.get_state()) == 0
        if topLevel:
            iteration += 1
        G = simulate()
        if topLevel and firstCrash is None and mcts.crashStates:
            firstCrash = iteration
        return G
    mcts.simulate = tracked_simulate

    with contextlib.redirect_stdout(io.StringIO()):
        _, bestReward = mcts.loop(ITERATIONS)
    return x, y, seed, len(mcts.crashStates) > 0, bestReward, firstCrash


def summarize(runs):
    rows = []
    for x in STARTS:
        for y in STARTS:
            cell = [r for r in runs if r[0] == x and r[1] == y]
            firstCrashes = [r[5] for r in cell if r[5] is not None]
            rows.append({
                "x": x,
                "y": y,
                "runs": len(cell),
                "crash_runs": sum(r[3] for r in cell),
                "crash_fraction": sum(r[3] for r in cell)/len(cell),
                "median_best_reward": statistics.median(r[4] for r in cell),
                "median_first_crash_iteration": statistics.median(firstCrashes) if firstCrashes else None,
            })
    return rows


def save_csv(rows):
    CSV_PATH.parent.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(rows, params):
    xs, ys = list(STARTS), list(STARTS)
    grid = np.full((len(ys), len(xs)), np.nan)          # NaN = start not run (too close)
    for row in rows:
        grid[ys.index(row["y"]), xs.index(row["x"])] = row["crash_fraction"]

    fig, ax = plt.subplots(figsize=(10, 8))
    half = (xs[1] - xs[0]) / 2
    extent = [xs[0] - half, xs[-1] + half, ys[0] - half, ys[-1] + half]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")                            # colour for skipped cells
    im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=1, extent=extent)
    fig.colorbar(im, ax=ax, label=f"Fraction of {len(SEEDS)} runs that found a crash")

    if half * 2 >= 5:                                     # labels get unreadable on finer grids
        for row in rows:
            ax.text(row["x"], row["y"], f'{row["crash_runs"]}/{row["runs"]}', ha="center", va="center",
                    fontsize=7, color="white" if row["crash_fraction"] < 0.5 else "black")

    # The straight boat moves in +x until it passes x = 100; clip the line to the grid
    sx, sy = params["straight_pos"]
    ax.plot([sx, extent[1]], [sy, sy], color="red", linewidth=2, label="Straight boat path")
    ax.scatter(sx, sy, color="red", marker="*", s=300, edgecolors="white", zorder=5, label="Straight boat start")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, frameon=False)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Steerable boat start x")
    ax.set_ylabel("Steerable boat start y")
    ax.set_title(f"MCTS crash rate by steerable start ({ITERATIONS} iterations per run)")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)


def main():
    with open(REPO / "parameters.json") as f:
        params = json.load(f)
    jobs = [(x, y, seed) for x in STARTS for y in STARTS for seed in SEEDS]
    print(f"{len(jobs)} runs: {len(STARTS)**2} starts x {len(SEEDS)} seeds, {ITERATIONS} iterations each")
    print(f'Straight boat start: {params["straight_pos"]}')

    runs = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpRoot:
        with Pool(initializer=init_worker, initargs=(tmpRoot,)) as pool:
            for i, run in enumerate(pool.imap_unordered(run_one, jobs), 1):
                runs.append(run)
                if i % 50 == 0 or i == len(jobs):
                    print(f"{i}/{len(jobs)} runs done", flush=True)

    rows = summarize(runs)
    save_csv(rows)
    save_heatmap(rows, params)

    n = len(SEEDS)
    counts = [row["crash_runs"] for row in rows]
    print("-"*35)
    print(f'{"Starts with crashes in 0/" + str(n):<28} | {counts.count(0)}')
    print(f'{"Starts with crashes in some":<28} | {sum(1 for c in counts if 0 < c < n)}')
    print(f'{"Starts with crashes in " + str(n) + "/" + str(n):<28} | {counts.count(n)}')
    print(f"Saved {CSV_PATH.relative_to(REPO)} and {FIG_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()

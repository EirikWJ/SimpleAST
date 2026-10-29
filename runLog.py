import csv, json, statistics, subprocess

from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EPISODE_COLUMNS = ("episode", "start_seed", "reward", "log_likelihood", "collision", "closest_distance",
                   "terminal_reason", "actions", "seconds", "trace")

class RunLog:
    ''' Records every training episode of a run in episodes.csv (one row per episode, written
        as it ends), and builds summary.json from those rows. The solvers call record() at the
        end of each training episode. The summary only depends on the rows, so it can be
        written at any point: after training, after an interruption, and after the evaluation.
    '''

    def __init__(self, resultsDir, plannedEpisodes) -> None:
        self.resultsDir = Path(resultsDir)
        self.plannedEpisodes = plannedEpisodes
        self.rows = []
        self.started = datetime.now()
        self.lastEpisodeEnd = self.started
        self.extra = {}  # Sections added after training, e.g. the trained policy and the evaluation
        self.git = get_git_state()  # At the start, since the code can change while the run goes on
        with open(self.resultsDir / "episodes.csv", "w", newline="") as f:
            csv.writer(f).writerow(EPISODE_COLUMNS)

    def record(self, sim, G) -> None:  # Called by the solver when a training episode has ended. sim still holds that episode
        now = datetime.now()
        outcome = get_outcome(sim)
        terminalReward = sim.reward(1, outcome["collision"], outcome["closest_distance"], True)
        row = {"episode": len(self.rows), "start_seed": getattr(sim, "startSeed", None), "reward": round(float(G), 4),
               "log_likelihood": round(float(G - terminalReward), 4),  # Sum of log p. The reward leaves out the last action's p
               **outcome, "actions": len(sim.get_state()), "seconds": round((now - self.lastEpisodeEnd).total_seconds(), 2),
               "trace": [float(seed) for seed in sim.get_state()]}
        self.lastEpisodeEnd = now
        self.rows.append(row)
        with open(self.resultsDir / "episodes.csv", "a", newline="") as f:
            csv.writer(f).writerow([json.dumps(row[column]) if column == "trace" else row[column] for column in EPISODE_COLUMNS])
        if len(self.rows) == self.plannedEpisodes:
            self.write_summary("training done")
        elif len(self.rows) % 50 == 0:  # So a run that is killed still has a recent summary
            self.write_summary("training")

    def write_summary(self, status) -> dict:
        summary = self.get_summary(status)
        with open(self.resultsDir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        return summary

    def get_summary(self, status) -> dict:
        rows = self.rows
        collisions = [row for row in rows if row["collision"]]
        summary = {
            "status": status,  # "training", "training done", "evaluating", "finished" or "interrupted"
            "started": self.started.isoformat(timespec="seconds"),
            "git": self.git,
            "runtime_seconds": round((datetime.now() - self.started).total_seconds(), 1),
            "episodes": len(rows),
            "episodes_planned": self.plannedEpisodes,
            "collision_episodes": len(collisions),
            "unique_collision_traces": len({(row["start_seed"], tuple(row["trace"])) for row in collisions}),
            "first_collision": None if not collisions else {
                "episode": collisions[0]["episode"],
                "seconds_into_run": round(sum(row["seconds"] for row in rows[:collisions[0]["episode"] + 1]), 1)},
            "terminal_reasons": dict(Counter(row["terminal_reason"] for row in rows).most_common()),
        }
        if rows:
            share = max(1, len(rows)//10)
            summary["progress"] = {"first_10_percent": get_progress(rows[:share]), "last_10_percent": get_progress(rows[-share:])}
            summary["best"] = get_episode_summary(max(rows, key=lambda row: row["reward"]))
            # AST looks for the most likely failure: the collision with the highest likelihood, not the highest reward
            summary["most_likely_collision"] = None if not collisions else get_episode_summary(max(collisions, key=lambda row: row["log_likelihood"]))
        return {**summary, **self.extra}


def get_outcome(sim) -> dict:  # The episode the sim holds now
    closest = sim.closest_distance if hasattr(sim, "closest_distance") else sim.simWorld.closest_boat_distance
    collision = bool(sim.is_failure_episode())
    reason = getattr(sim, "terminal_reason", "collision" if collision else "straight boat left the world")  # SimInterface has no reason
    return {"collision": collision, "closest_distance": round(float(closest), 3), "terminal_reason": reason}


def get_episode_summary(row) -> dict:
    keys = ("episode", "start_seed", "reward", "log_likelihood", "collision", "closest_distance", "terminal_reason", "actions", "trace")
    return {key: row[key] for key in keys}


def get_progress(rows) -> dict:
    return {"episodes": f'{rows[0]["episode"]} to {rows[-1]["episode"]}',
            "mean_reward": round(statistics.mean(row["reward"] for row in rows), 3),
            "collision_rate": round(sum(row["collision"] for row in rows)/len(rows), 3),
            "median_closest_distance": round(statistics.median(row["closest_distance"] for row in rows), 1)}


def get_git_state() -> dict:  # Which code made the result. uncommitted_changes includes new files
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "uncommitted_changes": bool(changes)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}

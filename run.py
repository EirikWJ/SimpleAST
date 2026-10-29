import argparse, json, random

from datetime import datetime
from pathlib import Path

import plotOutput

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "results" / "runs"
LEVELS = ("L0_open_water", "L1_stangvik_local", "L2_stangvik_full")  # Parameter files in levels/
SOLVERS = ("mcts", "sac", "random")
RANDOM_STARTS = (False, "once", "each")

# One entry point for every run:
#   from run import run, replay
#   run(simulator="marast", solver="sac", n=500, level="L1", random_start="each", swimmers=2)
#   replay()  # The best result of the latest run
# or from the terminal (see main.py):
#   uv run python main.py --solver sac --n 500 --level L1 --random-start each --swimmers 2
#   uv run python main.py --replay
# Each run gets its own folder results/runs/<name>/ with its settings, best result, crash
# states, plots and (for SAC) the trained model.

def run(simulator = "marast",       # "marast" or "brage" (the simple boat simulator)
        solver = "mcts",            # "mcts", "sac" or "random" (baseline: random actions)
        n = 500,                    # Iterations (MCTS) or episodes (SAC, random)
        level = None,               # MAR-AST: L0_open_water (default), L1_stangvik_local, L2_stangvik_full, or just "L0", "L1", "L2"
        random_start = False,       # False: the level's fixed adversary start. "once": one random start from start_seed, the
                                    # same every episode. "each": a new random start every episode (start_seed, start_seed+1, ...)
        start_seed = 0,
        swimmers = 0,               # Number of swimmers. Swimmer i gets seed swimmer_seed + i
        swimmer_seed = 42,
        own_colav = True,           # False: the own ship never evades (like Brage's straight boat)
        collision_distance = None,  # [m]. None: the level's value (50 m)
        solver_seed = 0,            # Seeds MCTS and random search (Python's random) and SAC (network and exploration)
        evaluate = None,            # SAC with random_start "each": unseen starts to test the trained policy on. None: 20
        show_plots = True,          # False: plots are only saved in the run's folder
        name = None):               # Folder name in results/runs/. None: <level>_<solver>_<date>_<time>
    level = _check_options(simulator, solver, n, level, random_start, swimmers, own_colav, collision_distance, evaluate)
    if evaluate is None:
        evaluate = 20 if solver == "sac" and random_start == "each" else 0
    settings = {"simulator": simulator, "solver": solver, "n": n, "level": level, "random_start": random_start,
                "start_seed": start_seed, "swimmers": swimmers, "swimmer_seed": swimmer_seed, "own_colav": own_colav,
                "collision_distance": collision_distance, "solver_seed": solver_seed, "evaluate": evaluate}
    if simulator == "marast":
        params = get_marast_params(level, random_start, start_seed, swimmers, swimmer_seed, own_colav, collision_distance)
    else:
        with open(REPO_ROOT / "parameters.json") as f:
            params = json.load(f)
    if name is None:
        name = f'{level if simulator == "marast" else "brage"}_{solver}_{datetime.now():%Y-%m-%d_%H-%M-%S}'
    resultsDir = RUNS_DIR / name
    resultsDir.mkdir(parents=True, exist_ok=True)
    with open(resultsDir / "settings.json", "w") as f:
        json.dump({"run": settings, "params": params}, f, indent=4)
    plotOutput.configure(resultsDir, show_plots)
    simFactory = _get_sim_factory(simulator, params)

    print(f'Run {name}: {solver} on {level if simulator == "marast" else "Brage simulator"}, n = {n}')
    timeStart = datetime.now()
    if solver == "mcts":
        from mcts.mctsController import MCTSController
        random.seed(solver_seed)
        controller = MCTSController(simFactory=simFactory, resultsDir=resultsDir)
    elif solver == "sac":
        from sac.sacController import SACController
        controller = SACController(simFactory=simFactory, seed=solver_seed, resultsDir=resultsDir)
    else:
        from randomsearch.randomSearchController import RandomSearchController
        controller = RandomSearchController(simFactory=simFactory, seed=solver_seed, resultsDir=resultsDir)
    controller.loop(n)
    runTime = datetime.now() - timeStart
    evaluation = controller.evaluate(evaluate) if evaluate > 0 else None

    summary = _summarise(controller, simFactory, simulator, n, runTime, evaluation)
    with open(resultsDir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    _print_summary(name, summary, simulator)
    plotOutput.configure()
    return summary


def replay(trace = None, start_seed = None, name = None, show = True):
    ''' Replays an action trace with the settings of a run, prints the outcome and plots it.
        name: the run's folder in results/runs/. None: the latest run.
        trace: None replays the run's best trace (and its start seed).
        start_seed: needed with random starts that changed every episode, unless replaying the best trace.
    '''
    runDir = RUNS_DIR / name if name is not None else _get_latest_run()
    with open(runDir / "settings.json") as f:
        saved = json.load(f)
    if trace is None:
        with open(runDir / "summary.json") as f:
            best = json.load(f)["best"]
        trace = best["trace"]
        start_seed = best["start_seed"] if start_seed is None else start_seed
    sim = _get_sim_factory(saved["run"]["simulator"], saved["params"])()
    _set_state(sim, trace, start_seed)
    outcome = _get_outcome(sim)
    print(f'Replay of {runDir.name}{"" if start_seed is None else f", start seed {start_seed}"}: {outcome}')
    plotOutput.configure(None, show)
    sim.plot()
    return outcome


def get_marast_params(level, random_start = False, start_seed = 0, swimmers = 0, swimmer_seed = 42,
                      own_colav = True, collision_distance = None) -> dict:
    ''' The level's parameter file, with the run options applied '''
    with open(REPO_ROOT / "levels" / f'{level}.json') as f:
        params = json.load(f)
    free = params["free_adversary"]
    if random_start:
        free["start"] = "random"
        free["randomise_start_each_episode"] = random_start == "each"
    free["start_seed"] = start_seed
    params["swimmers"]["count"] = swimmers
    params["swimmers"]["first_seed"] = swimmer_seed
    params["own_colav"] = "sbmpc" if own_colav else "none"
    if collision_distance is not None:
        params["collision_distance"] = collision_distance
    return params


def main_cli(argv = None) -> None:  # Terminal interface for run() and replay(), used by main.py
    parser = argparse.ArgumentParser(description="Run a solver on Brage's simulator or MAR-AST, or replay a run")
    parser.add_argument("--sim", default="marast", choices=("marast", "brage"))
    parser.add_argument("--solver", default="mcts", choices=SOLVERS)
    parser.add_argument("--n", type=int, default=500, help="iterations (MCTS) or episodes (SAC, random)")
    parser.add_argument("--level", help="MAR-AST level: L0, L1, L2 or the full name (default L0)")
    parser.add_argument("--random-start", default="off", choices=("off", "once", "each"))
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--swimmers", type=int, default=0)
    parser.add_argument("--swimmer-seed", type=int, default=42)
    parser.add_argument("--no-colav", action="store_true", help="the own ship never evades")
    parser.add_argument("--collision-distance", type=float)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--evaluate", type=int, help="SAC with --random-start each: unseen starts to test on (default 20)")
    parser.add_argument("--no-plots", action="store_true", help="only save the plots in the run's folder")
    parser.add_argument("--name", help="the run's folder name in results/runs/ (for --replay: which run, default the latest)")
    parser.add_argument("--replay", action="store_true", help="replay the best trace of a run (or --trace) instead of running")
    parser.add_argument("--trace", help="with --replay: an action trace, e.g. \"[0.5, 0.2, 0.9]\"")
    args = parser.parse_args(argv)
    if args.replay:
        replay(json.loads(args.trace) if args.trace else None, args.start_seed if args.trace else None, args.name)
        return
    run(simulator=args.sim, solver=args.solver, n=args.n, level=args.level,
        random_start=False if args.random_start == "off" else args.random_start, start_seed=args.start_seed,
        swimmers=args.swimmers, swimmer_seed=args.swimmer_seed, own_colav=not args.no_colav,
        collision_distance=args.collision_distance, solver_seed=args.solver_seed, evaluate=args.evaluate,
        show_plots=not args.no_plots, name=args.name)


def _check_options(simulator, solver, n, level, random_start, swimmers, own_colav, collision_distance, evaluate) -> str:
    # Returns the full level name. Raises ValueError for options that cannot work together
    if simulator not in ("marast", "brage"):
        raise ValueError(f'simulator must be "marast" or "brage", not {simulator!r}')
    if solver not in SOLVERS:
        raise ValueError(f'solver must be one of {SOLVERS}, not {solver!r}')
    if random_start not in RANDOM_STARTS:
        raise ValueError(f'random_start must be False, "once" or "each", not {random_start!r}')
    if n < 1:
        raise ValueError("n must be at least 1")
    if simulator == "brage":
        notInBrage = {"level": level is not None, "random_start": random_start is not False, "swimmers": swimmers > 0,
                      "own_colav": not own_colav, "collision_distance": collision_distance is not None}
        used = [option for option, isUsed in notInBrage.items() if isUsed]
        if used:
            raise ValueError(f'Brage\'s simulator has no {", ".join(used)}. These options are only for MAR-AST')
    else:
        level = _get_level_name(level)
    if solver == "mcts" and random_start == "each":
        raise ValueError('MCTS needs the same start every iteration, since its tree assumes it. Use random_start "once" or False')
    if evaluate and not (solver == "sac" and random_start == "each"):
        raise ValueError('evaluate tests a trained SAC policy on unseen starts, so it needs solver "sac" and random_start "each"')
    if swimmers < 0:
        raise ValueError("swimmers cannot be negative")
    return level


def _get_level_name(level) -> str:  # "L1" -> "L1_stangvik_local". None -> L0
    if level is None:
        return LEVELS[0]
    for name in LEVELS:
        if level in (name, name.split("_")[0]):
            return name
    raise ValueError(f'Unknown level {level!r}. Choose one of {LEVELS} (or just L0, L1, L2)')


def _get_sim_factory(simulator, params):
    if simulator == "brage":
        from sim.simInterface import SimInterface
        return SimInterface  # Reads parameters.json itself
    from sim.marastInterface import MarAstSimInterface
    return lambda: MarAstSimInterface(params=params)


def _set_state(sim, trace, startSeed) -> None:  # SimInterface has no start seed
    if startSeed is None:
        sim.set_state(trace)
    else:
        sim.set_state(trace, startSeed)


def _get_outcome(sim) -> dict:  # The episode the sim holds now
    closest = sim.closest_distance if hasattr(sim, "closest_distance") else sim.simWorld.closest_boat_distance
    return {"collision": bool(sim.is_failure_episode()), "closest_distance": round(float(closest), 3),
            "terminal_reason": getattr(sim, "terminal_reason", "collision" if sim.is_failure_episode() else "straight boat left the world")}


def _summarise(controller, simFactory, simulator, n, runTime, evaluation) -> dict:
    bestStartSeed = getattr(controller, "bestStartSeed", None)
    bestTrace = [float(seed) for seed in controller.bestState]
    sim = simFactory()
    _set_state(sim, bestTrace, bestStartSeed)  # Replays the best trace, to get its closest distance
    summary = {
        "runtime_seconds": round(runTime.total_seconds(), 1),
        "episodes": n,
        "collision_episodes": len(controller.crashStates),
        "unique_collision_traces": len(set(controller.crashStates)),
        "best": {"reward": round(float(controller.bestReward), 3), "start_seed": bestStartSeed, **_get_outcome(sim), "trace": bestTrace},
    }
    if evaluation is not None:
        summary["evaluation"] = {
            "collisions": sum(collided for _, collided, _, _ in evaluation),
            "starts": len(evaluation),
            "results": [{"start_seed": startSeed, "collision": bool(collided), "reward": round(float(G), 3), "closest_distance": round(float(d), 1)}
                        for startSeed, collided, G, d in evaluation]}
    return summary


def _print_summary(name, summary, simulator) -> None:
    unit = "m" if simulator == "marast" else "units"
    best = summary["best"]
    print("="*35)
    print(f'{"Run":<25} | {name}')
    print(f'{"Runtime":<25} | {summary["runtime_seconds"]} s ({round(summary["runtime_seconds"]/summary["episodes"], 2)} s per episode)')
    print(f'{"Episodes with collision":<25} | {summary["collision_episodes"]} of {summary["episodes"]} ({summary["unique_collision_traces"]} unique)')
    print(f'{"Best reward":<25} | {best["reward"]}')
    print(f'{"Best closest distance":<25} | {best["closest_distance"]} {unit} ({best["terminal_reason"]})')
    if best["start_seed"] is not None:
        print(f'{"Best start seed":<25} | {best["start_seed"]}')
    if "evaluation" in summary:
        evaluation = summary["evaluation"]
        print(f'{"Evaluation":<25} | {evaluation["collisions"]} collisions on {evaluation["starts"]} unseen starts')
    print(f'{"Results":<25} | {RUNS_DIR / name}')
    print("="*35)


def _get_latest_run() -> Path:
    runs = [folder for folder in RUNS_DIR.glob("*") if (folder / "summary.json").is_file()]
    if not runs:
        raise FileNotFoundError(f'No finished runs in {RUNS_DIR}')
    return max(runs, key=lambda folder: (folder / "summary.json").stat().st_mtime)

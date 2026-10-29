# Simple test for using MCTS for finding failure stats in simulator

Based on the work done by Lee, Ritchie et al. in "Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning". Used as a proof of concept before doing adaptive stress testing of more advanced agents and simulators.

Simulator is dumb boat going in straight line, while disturbance is steerable boat trying to crash in to the streerable boat. The boats operate in a 100x100 world with a step length of 5.

The MCTS has no knowledge of the workings of the sim-world, and can only pass in a random action-seed and is returned a set of variables to estimate the reward of the current state of the sim-world.

## MCTS

The monte carlo tree search is quite simple, where one loop searches down through the tree, finds a leaf node and does a single rollout from that node. This project is based on pseudocode on page 13 in Lee, Ritchie et al., "Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning"

## Results and effectivity of the model

Using MCTS to search for failure states seems to highly increase the chances of finding them. Through testing, over 500,000 random paths were generated of which none resulted in a crash, while about 70-80% of searches using MCTS with 4000 simulations/rollouts resulted in multiple failure states.

### Figures

----

![Image of a generated, most likely crash path](figs/crashPath.png?raw=true "Crash Path")

An example of a generated, most likely crash path. Steerable boat starts at x=17, y=0. Connected dots are positions in the same time step, colors indicate distances between boats.

----

![Image of a route with random seed actions](figs/randomPath.png?raw=true "Random path")

An example of a random path without using MCTS.

----

![Reward over time](figs/scoreOverTime.png?raw=true "Reward over time")

An example of the score of each simulation as it develops over time. First crash is found after 717 simulations.

## Running on IDUN

[IDUN](https://www.hpc.ntnu.no/idun/) is NTNU's computing cluster ([documentation](https://www.hpc.ntnu.no/idun/documentation/)). A single run is no faster there than on a laptop: the time goes into the simulator, which runs on one core, and a GPU does not help. What IDUN gives is many runs at once, e.g. ten solver seeds of the same experiment as one job array. Use the CPU partition (`CPUQ`, the default, at most 30 days per job).

**1. Log in** with your NTNU user name (from outside the NTNU network, connect to the NTNU VPN first):
```
ssh <username>@idun-login1.hpc.ntnu.no     # or idun-login2
```
The login nodes are for files, installing and submitting jobs. Runs go to the compute nodes through Slurm (steps 5 and 6).

**2. Get the code**, with the MAR-AST submodule. Put it in your work directory, which IDUN recommends for job output (1 TB; neither home nor work is backed up, so copy results you want to keep):
```
cd /cluster/work/$USER
git clone --recurse-submodules https://github.com/EirikWJ/SimpleAST.git
cd SimpleAST
```
After a `git pull` later on, also run `git submodule update --init`.

**3. Install uv and the dependencies** once, on the login node:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync --frozen
```
The installer puts the single `uv` program in `~/.local/bin` (IDUN advises against `pip install --user`, so don't install uv that way). uv downloads Python 3.11 itself if needed, and keeps the packages in `.venv/` inside the repo. The first sync takes a while: on Linux, PyTorch from PyPI includes CUDA libraries (several GB), even though only the CPU is used.

**4. Set your Slurm account.** List your accounts, and the default one, with:
```
sacctmgr show assoc where user=$USER format=account%25
sacctmgr show user $USER format=defaultaccount
idun-slurm-quota      # CPU hours used and left this month, per account
```
Accounts starting with `share-` have high priority; use one if you have it. Then add the account to `~/.bashrc`, so every `sbatch` uses it:
```
echo 'export SBATCH_ACCOUNT=<account>' >> ~/.bashrc && source ~/.bashrc
```

**5. Test in an interactive job** before submitting anything long. The `short` partition is meant for this (at most 20 minutes). Don't add `--ntasks` or `--nodes` to `srun ... --pty bash`: IDUN then ends the job after 60 seconds.
```
srun --account=$SBATCH_ACCOUNT --partition=short --cpus-per-task=2 --time=00:20:00 --pty bash
MPLBACKEND=Agg uv run --frozen python main.py --solver sac --n 20 --level L0 --random-start each --no-plots
exit
```

**6. Submit a job array.** `idun/run.slurm` passes its arguments on to `main.py`, and gives each array task `--solver-seed <task index>`. For ten SAC runs with seeds 0 to 9:
```
mkdir -p logs
sbatch --array=0-9 idun/run.slurm --solver sac --n 1000 --level L0 --random-start each
```
Any other run works the same way, e.g. the random-search baseline with `--solver random`. The script asks for 2 cores, 8 GB and 12 hours per task; override on the command line, e.g. `sbatch --time=0-04:00:00 --array=0-9 idun/run.slurm ...`. Each task's hours count against the account's monthly CPU quota (`idun-slurm-quota`). To limit how many tasks run at once, add `%`: `--array=0-49%10` runs at most 10 of the 50 at a time.

**7. Follow the jobs:** `squeue -u $USER` lists them, `scancel <job id>` stops them. Each task writes its terminal output to `logs/ast_<array job id>_<task>.out` and its results to `results/runs/<level>_<solver>_seed<task>_job<array job id>/`. A task that hits the time limit keeps the summary written so far (`summary.json` has status `training`).

**8. Combine the seeds** into one table, with mean and spread over the runs:
```
uv run python aggregate.py "L0_open_water_sac_seed*_job<array job id>"
```
It also saves the table as `results/aggregate_<pattern>.csv`.

**9. Copy the results home.** From your own machine:
```
scp -r <username>@idun-login1.hpc.ntnu.no:/cluster/work/<username>/SimpleAST/results/runs/<run folder> results/runs/
```
Then `uv run python main.py --replay --name <run folder>` shows a run's best result locally.

IDUN asks that any thesis, report or article that used its compute time acknowledges it, and cites: M. Själander, M. Jahre, G. Tufte and N. Reissmann, "EPIC: An Energy-Efficient, High-Performance GPGPU Computing Research Infrastructure", 2019, arXiv:1912.05848 (BibTeX on [hpc.ntnu.no](https://www.hpc.ntnu.no/)).

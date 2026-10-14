import json, sys

import matplotlib.pyplot as plt
import numpy as np

from sim.marast.marastPath import REPO_ROOT, MARAST_ROOT
from sim.marast.scenario import build_moving_object

from utils.get_path import get_ship_route_path, get_map_path
from utils.prepare_map import get_gdf_from_gpkg

# Draws the map and routes from a MAR-AST parameter file, without simulating anything.
# Run from the repo root:  uv run python -m sim.marast.plotScenario [parameters_marast.json]

def plot_scenario(parameterFile = "parameters_marast.json", zoomToRoutes = True) -> None:
    with open(REPO_ROOT / parameterFile) as f:
        params = json.load(f)
    frame_gdf, _, land_gdf, _, _ = get_gdf_from_gpkg(get_map_path(MARAST_ROOT, params["map"]),
                                                     "frame_3857", "ocean_3857", "land_3857", "coast_3857", "water_3857")
    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    land_gdf.plot(ax=ax, color="#D8D2C4", zorder=0)
    frame_gdf.boundary.plot(ax=ax, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.plot([], [], color="gray", linestyle="--", label="Map frame")
    routes = []
    shipStyles = (  # Wide solid line under a thin dashed one, so routes on top of each other are both visible
        ("Own ship", "own_route", "tab:blue", {"linewidth": 5, "alpha": 0.5}, (8, 4)),
        ("Adversary ship", "adversary_route", "tab:red", {"linewidth": 1.5, "linestyle": "--"}, (-14, -12)))
    for name, key, color, lineStyle, labelOffset in shipStyles:
        route = np.atleast_2d(np.loadtxt(get_ship_route_path(MARAST_ROOT, params[key])))  # Columns: north, east
        routes.append(route)
        north, east = route[:, 0], route[:, 1]
        ax.plot(east, north, color=color, label=f'{name} route ({params[key]})', **lineStyle)
        ax.scatter(east, north, color=color, marker=".", zorder=4)
        ax.scatter(east[0], north[0], color=color, marker="o", s=80, zorder=5)
        ax.scatter(east[-1], north[-1], color=color, marker="*", s=200, zorder=5)
        for i, (n, e) in enumerate(route):  # Waypoint numbers, 0 is the start
            ax.annotate(str(i), (e, n), textcoords="offset points", xytext=labelOffset, color=color, fontsize=9)
    movingObject = build_moving_object(params)
    if movingObject is not None:  # Its path only depends on its seed, so it can be drawn without simulating
        objectSteps = int(params["max_sim_time"]/params["time_step"]) + 1
        north, east = movingObject.pathNorth[:objectSteps], movingObject.pathEast[:objectSteps]
        routes.append(np.column_stack([north, east]))
        ax.plot(east, north, color="tab:green", label=f'{movingObject.name_tag} path (seed {movingObject.seed}, whole max_sim_time)')
        ax.scatter(east[0], north[0], color="tab:green", marker="o", s=80, zorder=5)
    ax.scatter([], [], color="black", marker="o", label="Start (first waypoint)")
    ax.scatter([], [], color="black", marker="*", s=100, label="Goal (last waypoint)")
    if zoomToRoutes:
        allPoints = np.vstack(routes)
        margin = 2000
        ax.set_xlim(allPoints[:, 1].min() - margin, allPoints[:, 1].max() + margin)
        ax.set_ylim(allPoints[:, 0].min() - margin, allPoints[:, 0].max() + margin)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_title(f'{parameterFile}: map {params["map"]}')
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    fig.legend(loc="outside right upper")  # Outside the map, so it never hides a route
    plt.show()

if __name__ == "__main__":
    plot_scenario(*sys.argv[1:2])

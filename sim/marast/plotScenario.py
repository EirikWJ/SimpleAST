import json, sys

import matplotlib.pyplot as plt
import numpy as np

from sim.marast.marastPath import REPO_ROOT
from sim.marast.scenario import build_swimmers, get_map_file, get_start_area_center, has_fixed_start, load_route, sample_free_start
from sim.marastInterface import SWIMMER_COLORS

from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from utils.prepare_map import get_gdf_from_gpkg, get_polygon_from_gdf

# Draws the map and routes from a MAR-AST parameter file, without simulating anything.
# Run from the repo root:  uv run python -m sim.marast.plotScenario [parameters_marast.json]

def plot_free_starts(ax, params, land_gdf, ownRoute, numberOfStarts):
    # The start area, and the starts of the first numberOfStarts start seeds with an arrow for the heading.
    # With a fixed start, only that start
    freeParams = params["free_adversary"]
    map_obj = PolygonObstacle(get_polygon_from_gdf(land_gdf))
    arrowLength = 600
    def draw_start(north, east, yaw):
        ax.scatter(east, north, color="tab:red", marker="o", s=30, zorder=5)
        ax.annotate("", xy=(east + arrowLength*np.sin(yaw), north + arrowLength*np.cos(yaw)), xytext=(east, north),
                    arrowprops={"arrowstyle": "->", "color": "tab:red"}, zorder=5)
    if has_fixed_start(freeParams):
        north, east, yaw = sample_free_start(freeParams, None, map_obj, ownRoute)
        draw_start(north, east, yaw)
        ax.scatter([], [], color="tab:red", marker="o", s=30, label=f'Adversary start, fixed (heading {freeParams["start"]["heading_deg"]} deg)')
        return np.array([[north - arrowLength, east - arrowLength], [north + arrowLength, east + arrowLength]])  # For the zoom
    center_north, center_east = get_start_area_center(freeParams, ownRoute)
    ax.add_patch(plt.Circle((center_east, center_north), freeParams["start_radius"], color="tab:red", fill=False, linestyle=":"))
    ax.plot([], [], color="tab:red", linestyle=":", label=f'Adversary start area ({freeParams["start_radius"]} m)')
    minDistance = freeParams.get("min_start_distance", 0)
    if minDistance > 0:  # No random start this close to the own ship's start
        ax.add_patch(plt.Circle((ownRoute[0][1], ownRoute[0][0]), minDistance, color="tab:blue", alpha=0.12, linewidth=0))
        ax.scatter([], [], color="tab:blue", alpha=0.25, marker="s", label=f'No adversary start within {minDistance} m of the own ship')
    firstSeed = freeParams["start_seed"]
    for startSeed in range(firstSeed, firstSeed + numberOfStarts):
        north, east, yaw = sample_free_start(freeParams, startSeed, map_obj, ownRoute)
        draw_start(north, east, yaw)
        ax.annotate(str(startSeed), (east, north), textcoords="offset points", xytext=(-12, -12), color="tab:red", fontsize=8)
    if freeParams["randomise_start_each_episode"]:
        label = f'Adversary starts, start seeds {firstSeed} to {firstSeed + numberOfStarts - 1} (one per episode)'
    else:
        label = f'Adversary start, start seed {firstSeed} (every episode)'
    ax.scatter([], [], color="tab:red", marker="o", s=30, label=label)
    radius = freeParams["start_radius"]
    return np.array([[center_north - radius, center_east - radius], [center_north + radius, center_east + radius]])  # For the zoom

def plot_scenario(parameterFile = "parameters_marast.json", zoomToRoutes = True, numberOfStarts = None) -> None:
    # numberOfStarts: how many free adversary starts to draw. Default 20 if randomised each episode, else 1
    with open(REPO_ROOT / parameterFile) as f:
        params = json.load(f)
    if numberOfStarts is None:
        freeParams = params.get("free_adversary", {})
        randomised = freeParams.get("randomise_start_each_episode", False) and not has_fixed_start(freeParams)
        numberOfStarts = 20 if randomised else 1
    frame_gdf, _, land_gdf, _, _ = get_gdf_from_gpkg(get_map_file(params["map"]),
                                                     "frame_3857", "ocean_3857", "land_3857", "coast_3857", "water_3857")
    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    land_gdf.plot(ax=ax, color="#D8D2C4", zorder=0)
    frame_gdf.boundary.plot(ax=ax, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.plot([], [], color="gray", linestyle="--", label="Map frame")
    routes = []
    freeAdversary = params.get("adversary_mode", "route") == "free"
    shipStyles = (  # Wide solid line under a thin dashed one, so routes on top of each other are both visible
        ("Own ship", "own_route", "tab:blue", {"linewidth": 5, "alpha": 0.5}, (8, 4)),
        ("Adversary ship", "adversary_route", "tab:red", {"linewidth": 1.5, "linestyle": "--"}, (-14, -12)))
    if freeAdversary:  # The free adversary has no route
        shipStyles = shipStyles[:1]
    for name, key, color, lineStyle, labelOffset in shipStyles:
        route = load_route(params[key])  # Columns: north, east
        routes.append(route)
        north, east = route[:, 0], route[:, 1]
        routeName = params[key] if isinstance(params[key], str) else "waypoints in the parameter file"
        ax.plot(east, north, color=color, label=f'{name} route ({routeName})', **lineStyle)
        ax.scatter(east, north, color=color, marker=".", zorder=4)
        ax.scatter(east[0], north[0], color=color, marker="o", s=80, zorder=5)
        ax.scatter(east[-1], north[-1], color=color, marker="*", s=200, zorder=5)
        for i, (n, e) in enumerate(route):  # Waypoint numbers, 0 is the start
            ax.annotate(str(i), (e, n), textcoords="offset points", xytext=labelOffset, color=color, fontsize=9)
    if freeAdversary:
        routes.append(plot_free_starts(ax, params, land_gdf, routes[0], numberOfStarts))
    swimmerSteps = int(params["max_sim_time"]/params["time_step"]) + 1
    for i, swimmer in enumerate(build_swimmers(params)):  # A path only depends on its seed, so it can be drawn without simulating
        color = SWIMMER_COLORS[i % len(SWIMMER_COLORS)]
        north, east = swimmer.pathNorth[:swimmerSteps], swimmer.pathEast[:swimmerSteps]
        routes.append(np.column_stack([north, east]))
        ax.plot(east, north, color=color, label=f'{swimmer.name_tag} path (seed {swimmer.seed}, whole max_sim_time)')
        ax.scatter(east[0], north[0], color=color, marker="o", s=80, zorder=5)
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

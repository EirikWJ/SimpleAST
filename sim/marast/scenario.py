from types import SimpleNamespace

import numpy as np

from sim.marast.marastPath import MARAST_ROOT
from sim.marast.adversaryShip import AdversaryShipModel
from sim.marast.movingObject import MovingObjectModel

from env_wrappers.multiship_env.env import AssetInfo, ShipAsset, MultiShipEnv
from simulator.ship_in_transit.sub_systems.ship_model import ShipConfiguration, SimulationConfiguration, ShipModel
from simulator.ship_in_transit.sub_systems.ship_engine import MachinerySystemConfiguration, MachineryMode, MachineryModeParams, MachineryModes, SpecificFuelConsumptionBaudouin6M26Dot3, SpecificFuelConsumptionWartila6L26
from simulator.ship_in_transit.sub_systems.LOS_guidance import LosParameters
from simulator.ship_in_transit.sub_systems.obstacle import PolygonObstacle
from simulator.ship_in_transit.sub_systems.controllers import ThrottleControllerGains, HeadingControllerGains
from simulator.ship_in_transit.sub_systems.wave_model import WaveModelConfiguration
from simulator.ship_in_transit.sub_systems.current_model import CurrentModelConfiguration
from simulator.ship_in_transit.sub_systems.wind_model import WindModelConfiguration
from simulator.ship_in_transit.utils import check_condition
from utils.get_path import get_ship_route_path, get_map_path
from utils.prepare_map import get_gdf_from_gpkg, get_polygon_from_gdf

# Two-ship scenario for MAR-AST's Ship in Transit simulator.
# Ship, machinery and environment values are the ones used in
# external/MAR-AST/test_beds/ship_simu_test/test_double_ship_map.py


def get_point_along_route(route, fraction):  # (north, east) at a fraction of the route's length
    legLengths = np.hypot(*np.diff(route, axis=0).T)
    distance = fraction*legLengths.sum()
    for (start, end), legLength in zip(zip(route[:-1], route[1:]), legLengths):
        if distance <= legLength:
            return start + (end - start)*distance/legLength
        distance -= legLength
    return route[-1]


def sample_free_start(freeParams, startSeed, map_obj, ownRoute):
    ''' Random (north, east, yaw) for the free adversary, the same for the same startSeed.
        Uniform within start_radius m of the point start_route_fraction along the own ship's
        route, inside the map and at least land_clearance m from land. Uniform heading.
    '''
    rng = np.random.default_rng(startSeed)
    center_north, center_east = get_point_along_route(ownRoute, freeParams["start_route_fraction"])
    for _ in range(1000):
        distance = freeParams["start_radius"]*np.sqrt(rng.uniform())
        direction = rng.uniform(0, 2*np.pi)
        north = center_north + distance*np.cos(direction)
        east = center_east + distance*np.sin(direction)
        clearance = 2*freeParams["land_clearance"]  # MAR-AST's checks use ship_length/2 as margin
        if not (check_condition.is_grounding(map_obj, [north, east], clearance)
                or check_condition.is_pos_outside_horizon(map_obj, [north, east], clearance)):
            return float(north), float(east), float(rng.uniform(-np.pi, np.pi))
    raise ValueError(f'No start on open water found for start seed {startSeed}. Lower land_clearance or move the start area')


def get_dummy_route(north, east, yaw):  # Two waypoints, the second so far away that it is never reached
    farAway = 1e6
    return np.array([[north, east], [north + farAway*np.cos(yaw), east + farAway*np.sin(yaw)]])


def build_swimmers(params):
    ''' Returns the swimmers described by params["swimmers"], as a list of MovingObjectModel
        (empty if count is 0 or the block is missing). Swimmer i gets seed first_seed + i, so
        every swimmer has its own path, and the same seeds always give the same paths. Each
        starts within start_radius m of the point route_fraction along the own ship's route.
    '''
    swimmerParams = params.get("swimmers")
    if swimmerParams is None or swimmerParams["count"] == 0:
        return []
    ownRoute = np.loadtxt(get_ship_route_path(MARAST_ROOT, params["own_route"]))
    center_north, center_east = get_point_along_route(ownRoute, swimmerParams["route_fraction"])
    return [MovingObjectModel(
                center_north=center_north,
                center_east=center_east,
                start_radius=swimmerParams["start_radius"],
                speed=swimmerParams["speed"],
                heading_change_sigma=np.deg2rad(swimmerParams["heading_change_sigma_deg"]),
                time_step=params["time_step"],
                max_sim_time=params["max_sim_time"],
                seed=swimmerParams["first_seed"] + i,
                name_tag=f'Swimmer {i + 1}',  # MAR-AST tells assets apart by name, so the names must differ
                length=swimmerParams["length"],
                width=swimmerParams["width"])
            for i in range(swimmerParams["count"])]


def build_two_ship_env(params):
    ''' Returns (env, assets, map_gdfs). assets[0] is the own ship (system under test,
        SBMPC collision avoidance), assets[1] is the adversary (AdversaryShipModel).
        assets[2:] are the swimmers from params["swimmers"] (MovingObjectModel). The own
        ship's SBMPC avoids them, and MAR-AST's collision check stops a ship that hits one.
    '''
    time_step = params["time_step"]
    args = SimpleNamespace(time_step=time_step, ship_draw=False, time_since_last_ship_drawing=30)

    # np.random is used for the wave phases when a ship is created
    np.random.seed(params["env_seed"])

    map_gdfs = get_gdf_from_gpkg(get_map_path(MARAST_ROOT, params["map"]),
                                 "frame_3857", "ocean_3857", "land_3857", "coast_3857", "water_3857")
    frame_gdf, land_gdf = map_gdfs[0], map_gdfs[2]
    map = [PolygonObstacle(get_polygon_from_gdf(land_gdf)), frame_gdf]

    ship_config = ShipConfiguration(
        coefficient_of_deadweight_to_displacement=0.7,
        bunkers=200000,
        ballast=200000,
        length_of_ship=80,
        width_of_ship=16,
        added_mass_coefficient_in_surge=0.4,
        added_mass_coefficient_in_sway=0.4,
        added_mass_coefficient_in_yaw=0.4,
        dead_weight_tonnage=3850000,
        mass_over_linear_friction_coefficient_in_surge=130,
        mass_over_linear_friction_coefficient_in_sway=18,
        mass_over_linear_friction_coefficient_in_yaw=90,
        nonlinear_friction_coefficient__in_surge=2400,
        nonlinear_friction_coefficient__in_sway=4000,
        nonlinear_friction_coefficient__in_yaw=400
    )
    wave_model_config = WaveModelConfiguration(
        minimum_wave_frequency=0.4,
        maximum_wave_frequency=2.5,
        wave_frequency_discrete_unit_count=50,
        minimum_spreading_angle=-np.pi,
        maximum_spreading_angle=np.pi,
        spreading_angle_discrete_unit_count=10,
        spreading_coefficient=1,
        rho=1025.0,
        timestep_size=time_step
    )
    current_model_config = CurrentModelConfiguration(
        initial_current_velocity=0.01,
        current_velocity_standard_deviation=0.0075,
        current_velocity_decay_rate=0.025,
        initial_current_direction=np.deg2rad(0.0),
        current_direction_standard_deviation=0.025,
        current_direction_decay_rate=0.025,
        timestep_size=time_step
    )
    wind_model_config = WindModelConfiguration(
        initial_mean_wind_velocity=None,
        mean_wind_velocity_decay_rate=0.025,
        mean_wind_velocity_standard_deviation=0.005,
        initial_wind_direction=np.deg2rad(0.0),
        wind_direction_decay_rate=0.025,
        wind_direction_standard_deviation=0.025,
        minimum_mean_wind_velocity=0.0,
        maximum_mean_wind_velocity=42.0,
        minimum_wind_gust_frequency=0.06,
        maximum_wind_gust_frequency=0.4,
        wind_gust_frequency_discrete_unit_count=100,
        clip_speed_nonnegative=True,
        kappa_parameter=0.0026,
        U10=10.0,
        wind_evaluation_height=5.0,
        timestep_size=time_step
    )

    main_engine_capacity = 2160e3
    diesel_gen_capacity = 510e3
    pto_mode = MachineryMode(params=MachineryModeParams(
        main_engine_capacity=main_engine_capacity, electrical_capacity=0,
        shaft_generator_state='GEN', name_tag='PTO'))
    mec_mode = MachineryMode(params=MachineryModeParams(
        main_engine_capacity=main_engine_capacity, electrical_capacity=diesel_gen_capacity,
        shaft_generator_state='OFF', name_tag='MEC'))
    pti_mode = MachineryMode(params=MachineryModeParams(
        main_engine_capacity=0, electrical_capacity=2*diesel_gen_capacity,
        shaft_generator_state='MOTOR', name_tag='PTI'))
    machinery_config = MachinerySystemConfiguration(
        machinery_modes=MachineryModes([pto_mode, mec_mode, pti_mode]),
        machinery_operating_mode=1,
        linear_friction_main_engine=68,
        linear_friction_hybrid_shaft_generator=57,
        gear_ratio_between_main_engine_and_propeller=0.6,
        gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
        propeller_inertia=6000,
        propeller_diameter=3.1,
        propeller_speed_to_torque_coefficient=7.5,
        propeller_speed_to_thrust_force_coefficient=1.7,
        hotel_load=200000,
        rated_speed_main_engine_rpm=1000,
        rudder_angle_to_sway_force_coefficient=50e3,
        rudder_angle_to_yaw_force_coefficient=500e3,
        max_rudder_angle_degrees=35,
        max_rudder_rate_degree_per_s=2.3,
        specific_fuel_consumption_coefficients_me=SpecificFuelConsumptionWartila6L26().fuel_consumption_coefficients(),
        specific_fuel_consumption_coefficients_dg=SpecificFuelConsumptionBaudouin6M26Dot3().fuel_consumption_coefficients()
    )

    def build_asset(shipClass, name_tag, route_name, heading_gains):  # route_name: route file path, or an array of waypoints
        route = np.loadtxt(route_name) if isinstance(route_name, str) else route_name  # Columns: north, east
        (north_0, east_0), (north_1, east_1) = route[0], route[1]
        simulation_config = SimulationConfiguration(
            initial_north_position_m=north_0,
            initial_east_position_m=east_0,
            initial_yaw_angle_rad=np.arctan2(east_1 - east_0, north_1 - north_0),  # Along the first route leg
            initial_forward_speed_m_per_s=params["initial_forward_speed"],
            initial_sideways_speed_m_per_s=0.0,
            initial_yaw_rate_rad_per_s=0.0,
            integration_step=time_step,
            simulation_time=params["max_sim_time"],
        )
        ship = shipClass(
            ship_config=ship_config,
            simulation_config=simulation_config,
            wave_model_config=wave_model_config,
            current_model_config=current_model_config,
            wind_model_config=wind_model_config,
            machinery_config=machinery_config,
            throttle_controller_gain=ThrottleControllerGains(
                kp_ship_speed=2.50, ki_ship_speed=0.025, kp_shaft_speed=0.05, ki_shaft_speed=0.0001),
            heading_controller_gain=HeadingControllerGains(**heading_gains),
            los_parameters=LosParameters(
                radius_of_acceptance=300, lookahead_distance=1000, integral_gain=0.002, integrator_windup_limit=4000),
            name_tag=name_tag,
            route_name=route_name,
            engine_steps_per_time_step=params["engine_step_count"],
            initial_propeller_shaft_speed_rad_per_s=0,
            initial_propeller_shaft_acc_rad_per_sec2=0,
            desired_speed=params["desired_speed"],
            cross_track_error_tolerance=750,
            nav_fail_time=300,
            map_obj=map[0],
            colav_mode='sbmpc',  # Own ship: SBMPC. Adversary: hook for the injected heading offset
            print_status=False
        )
        return wrap_asset(ship)

    def wrap_asset(model):
        info = AssetInfo(
            current_north=model.north,
            current_east=model.east,
            current_yaw_angle=model.yaw_angle,
            forward_speed=model.forward_speed,
            sideways_speed=model.sideways_speed,
            name_tag=model.name_tag,
            ship_length=model.l_ship,
            ship_width=model.w_ship
        )
        return ShipAsset(ship_model=model, info=info)

    ownRouteName = get_ship_route_path(MARAST_ROOT, params["own_route"])
    if params.get("adversary_mode", "route") == "free":  # No route: random start, steered directly by the solver
        freeParams = params["free_adversary"]
        start = sample_free_start(freeParams, freeParams["start_seed"], map[0], np.loadtxt(ownRouteName))
        adversaryRoute = get_dummy_route(*start)
    else:
        adversaryRoute = get_ship_route_path(MARAST_ROOT, params["adversary_route"])
    assets = [
        build_asset(ShipModel, 'Own ship', ownRouteName, params["own_heading_gains"]),
        build_asset(AdversaryShipModel, 'Adversary ship', adversaryRoute, params["adversary_heading_gains"]),
    ]
    if params.get("adversary_mode", "route") == "free":
        assets[1].ship_model.steer_freely()
    assets += [wrap_asset(swimmer) for swimmer in build_swimmers(params)]
    env = MultiShipEnv(
        assets=assets,
        map=map,
        wave_model_config=wave_model_config,
        current_model_config=current_model_config,
        wind_model_config=wind_model_config,
        args=args,
        include_wave=params["include_wave"],
        include_wind=params["include_wind"],
        include_current=params["include_current"],
        seed=params["env_seed"])
    return env, assets, map_gdfs

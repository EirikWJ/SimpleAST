from sim.marast import marastPath  # Puts external/MAR-AST on sys.path

from simulator.ship_in_transit.sub_systems.ship_model import ShipModel
from simulator.ship_in_transit.sub_systems.controllers import HeadingBySampledRouteController


class AdversaryShipModel(ShipModel):
    ''' Ship that follows its route like any other ShipModel, but with a heading offset
        chosen from the outside (the AST solver) added to its LOS heading.

        ShipModel.step() already adds the offset returned by sbmpc_colav_override() to the
        heading reference when colav_mode == 'sbmpc'. Overriding that method lets us inject
        our own offset without touching the MAR-AST code. Must be created with
        colav_mode='sbmpc' for the hook to be called.

        After steer_freely() the ship has no route at all: it steers to
        auto_pilot.commanded_heading, which the solver sets directly.
    '''

    heading_offset = 0.0  # [rad], added to the LOS heading reference
    speed_factor = 1.0    # Scales the desired speed

    def sbmpc_colav_override(self, asset_infos):
        return self.speed_factor, self.heading_offset

    def steer_freely(self) -> None:
        # Same autopilot object, but its steering method no longer looks at the route. The ship
        # still needs a route for MAR-AST's navigation checks, so give it a dummy one that is
        # never reached (see scenario.py), and no navigation failure can happen
        self.auto_pilot.__class__ = FreeHeadingAutopilot


class FreeHeadingAutopilot(HeadingBySampledRouteController):
    ''' Autopilot that holds commanded_heading [rad] with the ship's own heading controller
        (PID, rudder angle and rate limits), instead of following a route with LOS guidance.
        The cross-track error stays 0, so MAR-AST never flags a navigation failure.
    '''

    commanded_heading = 0.0

    def rudder_angle_from_sampled_route(self, north_position, east_position, heading, desired_heading_offset=0.0):
        self.heading_ref = self.commanded_heading
        self.heading_mea = heading
        return self.heading_controller.rudder_angle_from_heading_setpoint(heading_ref=self.commanded_heading, measured_heading=heading)

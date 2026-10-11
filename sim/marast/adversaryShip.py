from sim.marast import marastPath  # Puts external/MAR-AST on sys.path

from simulator.ship_in_transit.sub_systems.ship_model import ShipModel


class AdversaryShipModel(ShipModel):
    ''' Ship that follows its route like any other ShipModel, but with a heading offset
        chosen from the outside (the AST solver) added to its LOS heading.

        ShipModel.step() already adds the offset returned by sbmpc_colav_override() to the
        heading reference when colav_mode == 'sbmpc'. Overriding that method lets us inject
        our own offset without touching the MAR-AST code. Must be created with
        colav_mode='sbmpc' for the hook to be called.
    '''

    heading_offset = 0.0  # [rad], added to the LOS heading reference
    speed_factor = 1.0    # Scales the desired speed

    def sbmpc_colav_override(self, asset_infos):
        return self.speed_factor, self.heading_offset

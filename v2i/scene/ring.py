"""Contains the ring road scenario class."""


# Import base class of network
from v2i.scene.base import Network

from flow.networks.ring import ADDITIONAL_NET_PARAMS


class RingNetwork(Network):

    def __init__(self,
                scenario_params,
                human_vehicle_params):

        super(RingNetwork, self).__init__("RingNetwork", scenario_params, human_vehicle_params)
        for param in ADDITIONAL_NET_PARAMS:
            if param not in self.scenario_params.keys():
                self.scenario_params[param] = ADDITIONAL_NET_PARAMS[param]
""" Base class for v2i network """

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams


class v2iNetwork(Network):

    """
    This is the base class to initializes a new v2i network. 
    Networks are used to specify the geometry of the road network. 
    A network consists of nodes, edges and routes.

    This class extends the flow.network.base classes by implementing
    extra functionalities required for v2i simulator.

    To learn, how to create a custom network, please have a look
    at tutorials provided by flow. 
    https://github.com/flow-project/flow/blob/master/tutorials/tutorial05_networks.ipynb

    """

    def __init__(self,
                 name, 
                 vehicles, 
                 net_params, 
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

                 
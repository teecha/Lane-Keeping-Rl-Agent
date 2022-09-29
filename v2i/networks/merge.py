""" Contains the custom merge network class. """

from v2i.networks.base import v2iNetwork
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos

INFLOW_EDGE_LEN = 200 # length of the inflow edges (needed for resets)
VEHICLE_LENGTH = 5

ADDITIONAL_NET_PARAMS = {
    # length of the merge edge
    "merge_length": 100,
    # number of lanes in the merge
    "merge_lanes": 3,
    # number of lanes in the highway
    "highway_lanes": 4,
    # max speed limit of the network
    "speed_limit": 30,
}

class twoInMergeNetwork(v2iNetwork):
    """Network class for highways with two in-merges.

    This network consists of single or multi-lane highway network with 
    two on-ramps with a variable number of lanes that can be used to generate
    periodic pertubation.

    Requires from net_params:

    * **merge_length** : length of the merge edge
    * **pre_merge_length** : length of the highway leading to the merge
    * **post_merge_length** : length of the highway past the merge
    * **merge_lanes** : number of lanes in the merge
    * **highway_lanes** : number of lanes in the highway
    * **speed_limit** : max speed limit of the network

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from v2i.networks import twoInMergeNetwork
    >>>
    >>> network = twoInMergeNetwork(
    >>>     name='merge',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'merge_length': 100,
    >>>             'pre_merge_length': 200,
    >>>             'post_merge_length': 100,
    >>>             'merge_lanes': 1,
    >>>             'highway_lanes': 1,
    >>>             'speed_limit': 30
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self, name, vehicles, net_params, initial_config=InitialConfig(), traffic_lights=TrafficLightParams()):
        """Initialize a twoInMergeNetwork."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config, traffic_lights)
    
    def specify_nodes(self, net_params):
        merge_one_angle = pi / 4
        merge_two_angle = (3*pi)/4
        merge = net_params.additional_params["merge_length"]
        
        nodes = [
            {
                'id': 'node_1',
                'x': -INFLOW_EDGE_LEN,
                'y': 0
            },
            {
                'id': 'node_2',
                'x': 0,
                'y': 0
            },
            {
                'id': 'node_3',
                'x': INFLOW_EDGE_LEN,
                'y': 0,
            },
            {
                'id': 'node_4',
                'x': (-merge * cos(merge_one_angle)),
                'y': -merge * sin(merge_one_angle)
            },
            {
                'id': 'node_5',
                'x': 2*INFLOW_EDGE_LEN,
                'y': 0
            },
            {
                'id': 'node_6',
                'x': merge * cos(merge_two_angle) + (INFLOW_EDGE_LEN),
                'y': merge * sin(merge_two_angle)
            }, {
                'id': 'node_rl',
                'x': -(INFLOW_EDGE_LEN + (2 * VEHICLE_LENGTH)),
                'y': 0
            }]
        return nodes
    
    def specify_edges(self, net_params):
        merge = net_params.additional_params["merge_length"]
        edges = [{
            "id": "highway_0",
            "type": "highwayType",
            "from": "node_1",
            "to": "node_2",
            "length": INFLOW_EDGE_LEN
        }, {
            "id": "highway_1",
            "type": "highwayType",
            "from": "node_2",
            "to": "node_3",
            "length": INFLOW_EDGE_LEN
        }, {
            "id": "merge_one",
            "type": "mergeType",
            "from": "node_4",
            "to": "node_2",
            "length": merge
        }, {
            "id": "highway_2",
            "type" : "highwayType",
            "from": "node_3",
            "to": "node_5",
            "length": INFLOW_EDGE_LEN
        }, {
            "id": "merge_two",
            "type": "mergeType",
            "from": "node_6",
            "to": "node_3",
            "length": merge
        }, {
            "id": "rl_highway",
            "type": "highwayType",
            "from": "node_rl",
            "to": "node_1",
            "length": 2*VEHICLE_LENGTH
        }]

        return edges
    
    def specify_types(self, net_params):
        
        h_lanes = net_params.additional_params["highway_lanes"]
        m_lanes = net_params.additional_params["merge_lanes"]
        speed = net_params.additional_params["speed_limit"]

        types = [{
            "id": "highwayType",
            "numLanes": h_lanes,
            "speed": speed
        }, {
            "id": "mergeType",
            "numLanes": m_lanes,
            "speed": speed
        }]

        return types
    
    def specify_routes(self, net_params):
        """
        'edge_0' : [ 'node_1', 'node_2', ..., ]
        """

        rts = {
            "highway_0": ["highway_0", "highway_1", "highway_2"],
            "merge_one": ["merge_one", "highway_1", "highway_2"],
            "merge_two": ["merge_two", "highway_2"],
            "rl_highway": ["rl_highway", "highway_0", "highway_1", "highway_2"],
            "rl_0": ["rl_highway", "highway_0", "highway_1", "highway_2"]
        }
        return rts
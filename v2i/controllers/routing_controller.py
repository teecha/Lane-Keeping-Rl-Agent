"""Contains a list of custom routing controllers"""

from flow.controllers.routing_controllers import BaseRouter

class twoNetworkMergeRouter(BaseRouter):
    """A router used to continuously re-route of the vehicle in a in-merge scenario.
    This class is useful if vehicles are expected to continuously follow the
    The same route is repeated, once it reaches its end.
    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class.
        
        Adopt one of the current edge's routes if about to leave the network.
        """
        edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)

        if len(current_route) == 0:
            return None
        elif edge == 'highway_2':
            return ['highway_2', 'highway_3', 'highway_1']
        return None

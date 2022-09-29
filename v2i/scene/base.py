
__all_scenes__ = ['RingNetwork']
__all_human_long_controllers = ['IDM']
__all_human_lc_controllers = ['MOBIL']

from flow.core.params import VehicleParams

class Network:

    def __init__(self,
                scenario_name,
                scenario_params,
                human_vehicle_params):
        
        self.scenario_name = scenario_name
        self.scenario_params = scenario_params

        self.human_vehicle_params 

''' Defines Merge Network '''

from v2i.networks.merge import twoInMergeNetwork
from v2i.networks.merge import ADDITIONAL_NET_PARAMS

from flow.envs.base import Env
from flow.core.params import InFlows
from flow.core.params import NetParams
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import InitialConfig

from flow.controllers.car_following_models import IDMController
from flow.controllers.rlcontroller import RLController
from v2i.controllers.routing_controller import twoNetworkMergeRouter
from flow.controllers.lane_change_controllers import SimLaneChangeController

from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS

from v2i.utils.fileOps import parseYaml
from v2i.core.validate import validateAndMergeConfig
from v2i.core.vehicle import Vehicle

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class mergeNetwork(Env):

    def __init__(self, envConfig, numRLVehicles=3):

        # Parse simulation parameters.
        self.envConfig = parseYaml(envConfig)
        
        # Validate and merge simulation parameters
        validator = validateAndMergeConfig(self.envConfig)
        validator.validate()
        self.additional_net_params, self.idmParams, self.genCarFollowingParams =  validator.mergeAll()

        env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
        env_params.horizon = 4000

        sim_params = SumoParams(sim_step=0.1, render=True, restart_instance=True)

        vehicles = VehicleParams()
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     car_following_params=SumoCarFollowingParams(speed_mode="obey_safe_speed", 
                     max_speed=self.envConfig.get_param), 
                     lane_change_controller=(SimLaneChangeController, {}),
                     color='red',
                     num_vehicles=0)
        
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     car_following_params=SumoCarFollowingParams(speed_mode="right_of_way"), 
                     lane_change_controller=(SimLaneChangeController, {}), 
                     color='yellow',
                     num_vehicles=1)
 
        self.densities = np.linspace(0.1, 1.0, num=10)
        self.densities = np.round(self.densities, 1)

        inflow = InFlows()        
        inflow.add(veh_type="human", edge="highway_0", probability=0.1, depart_lane="random", depart_speed="random")
        inflow.add(veh_type="human", edge="merge_one", probability=0.0, depart_lane="random", depart_speed="random")
        inflow.add(veh_type="human", edge="merge_two", probability=0.0, depart_lane="random", depart_speed="random")

        initial_config = InitialConfig()
        
        additional_net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS, inflows=inflow)
        
        network = twoInMergeNetwork(name="merge", vehicles=vehicles, net_params=additional_net_params, initial_config=initial_config)
        simulator = "traci"
        super().__init__(env_params, sim_params, network, simulator)
    
    def get_state(self):
        return np.zeros(10)
    
    def action_space(self):
        return Discrete(2)
    
    def apply_rl_actions(self, rl_actions):
        pass

    def additional_command(self):
        """Reintroduce any RL vehicle that may have exited in the last step.
        This is used to maintain a constant number of RL vehicle in the system
        at all times, in order to comply with a fixed size observation and
        action space.

        Adapted from flow.envs.bottleneck.py : additional_command
        """
        
        super().additional_command()
        num_rl = self.k.vehicle.num_rl_vehicles
        
        if num_rl != 1:
            # find the vehicles that have exited
            diff_list = (
                set(self.rl_id_list).difference(self.k.vehicle.get_rl_ids()))
            for rl_id in diff_list:
                lane_num = 0
                try:
                    self.k.vehicle.add(
                        veh_id=rl_id,
                        edge="highway_0",
                        type_id=str('rl'),
                        lane=str(lane_num),
                        pos="0",
                        speed="max")
                except Exception:
                    pass
        

    def reset(self, density=None):
        """See parent class.

        The sumo instance is reset with a new inflow_rate.
        """
        # reset the step counter
        self.step_counter = 0

        # update the network
        initial_config = InitialConfig()
        if density is None:
            vehicle_density = np.random.choice(self.densities)
        else:
            vehicle_density = density

        inflow = InFlows()        
        inflow.add(veh_type="human", edge="highway_0", probability=vehicle_density, depart_lane="random", depart_speed="random")
        inflow.add(veh_type="human", edge="merge_one", probability=vehicle_density/2, depart_lane="random", depart_speed="random")
        inflow.add(veh_type="human", edge="merge_two", probability=vehicle_density, depart_lane="random", depart_speed="random")

        additional_net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS,
                                inflows=inflow)

        self.network = self.network.__class__(
            self.network.orig_name, 
            self.network.vehicles, 
            additional_net_params, 
            self.network.initial_config, 
            self.network.traffic_lights)
        self.k.vehicle = deepcopy(self.initial_vehicles)
        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

        # restart the sumo instance
        self.restart_simulation(
            sim_params=self.sim_params,
            render=self.sim_params.render)
        
        # perform the generic reset function
        super().reset()
        
        self.rl_id_list = deepcopy(self.k.vehicle.get_rl_ids())

        return None


if __name__ == "__main__":
    '''
    numRuns = 10
    veh_pers_hour = np.linspace(0.1, 1, num=10)
    veh_pers_hour = np.round(veh_pers_hour, 1)
    
    avg_vehicles = np.zeros((veh_pers_hour.shape[0], numRuns))


    for idx, rate in enumerate(veh_pers_hour):
        print("Working for %.1f"%(rate))
        env = mergeNetwork("experiments/ring/sim-config.yml")
        for run in range(0, numRuns):
            print("Runs : %d/%d"%(run, numRuns))
            env.reset(density=rate)
            veh_count = 0
            step_count = 0
            while True:
                ns, reward, done, _ = env.step(rl_actions=0)
                veh_count += len(env.k.vehicle.get_ids())
                step_count += 1
                if done:
                    assert step_count == env.env_params.horizon
                    avg_vehicles[idx][run] = veh_count/env.env_params.horizon
                    break
        env.terminate()
         
    plt.bar(np.arange(0, len(veh_pers_hour)), avg_vehicles.mean(axis=1))
    plt.xticks(np.arange(0, len(veh_pers_hour)), veh_pers_hour)
    plt.xlabel("Vehicle Density")
    plt.ylabel("Avg number of vehicles")
    plt.show()
    '''
    env = mergeNetwork("experiments/ring/sim-config.yml")
    veh_pers_hour = np.linspace(0.1, 1, num=10)
    veh_pers_hour = np.round(veh_pers_hour, 1)
    veh_pers_hour = [1.0]
    for den in veh_pers_hour:
        print(den)
        env.reset(den)
        for i in range(0, 4000):
            env.step(rl_actions=0)
        break
    env.terminate()
    
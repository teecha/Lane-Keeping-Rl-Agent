''' Defines RingEnv '''

from gym.spaces import Discrete
from flow.envs.base import Env
from flow.networks.ring import RingNetwork
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.core.params import SumoParams
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import InFlows

import matplotlib.pyplot as plt
import numpy as np

from v2i.utils.fileOps import parseYaml
from v2i.core.validate import validateAndMergeConfig
from v2i.core.vehicle import Vehicle

class RingEnv(Env):

    def __init__(self, envConfig):

        # Parse simulation parameters.
        self.envConfig = parseYaml(envConfig)
        
        # Validate and merge simulation parameters
        validator = validateAndMergeConfig(self.envConfig)
        validator.validate()
        self.additional_net_params, self.idmParams, self.genCarFollowingParams =  validator.mergeAll()

        # Set environments configuration
        #print(ADDITIONAL_ENV_PARAMS)
        env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
        env_params.horizon = 100

        # Set sumo parameters
        sim_params = SumoParams(sim_step=0.1, render=False)

        # Inflow parameters
        inflow = InFlows()

        # Add vehicles to the scene
        vehicles = Vehicle(numLanes=self.additional_net_params['ring']['lanes'], \
                ringLength=self.additional_net_params['ring']['length'])
        
        # Reset configuration
        init_config = InitialConfig(spacing="uniform", shuffle=False, min_gap=0, perturbation=0)

        # Network configuration
        additional_net_params = NetParams(additional_params=self.additional_net_params['ring'])
        
        # Create Ring Network with defined parameters.
        network = RingNetwork(name="ringNetwork", vehicles=vehicles.add(0.9, 0), net_params=additional_net_params, initial_config=init_config)
        simulator = 'traci'

        super().__init__(env_params, sim_params, network, simulator)
    
    
    def get_state(self):
        return np.zeros(10)
    
    def action_space(self):
        return Discrete(2)
    
    def apply_rl_actions(self, rl_actions):
        pass

    def reset(self):
        obs = super().reset()
        #print(obs)
    

if __name__ == "__main__":
    
    env = RingEnv("experiments/ring/sim-config.yml")

    numRuns = 5
    horizon = 1000
    position_logs = {}

    for episode in range(0, numRuns):
        env.reset()
        for step in range(0, horizon):
            for vehID in env.k.vehicle.get_ids():
                if vehID not in position_logs.keys():
                    position_logs[vehID] = np.zeros((numRuns, horizon))
                position_logs[vehID][episode][step] = env.k.vehicle.get_speed(vehID)
            env.step(rl_actions=0)
    
    for veh in position_logs.keys():
        mean = position_logs[veh].mean(axis=0)
        std = position_logs[veh].std(axis=0)
        plt.errorbar(np.arange(horizon), mean, std, label=veh, fmt="-o")
        #break
    plt.legend()
    plt.show()
import warnings
import logging
import os
import sys
import numpy as np
from datetime import datetime
import unittest
from unittest import TestCase
import matplotlib.pyplot as plt
from v2i.utils.fileOps import parseYaml, dumpYaml
from v2i.envs.merge import twoInMergesEnv

class twoInMergeEnvTest(TestCase):

    def setUp(self):
        self.configPath = os.path.dirname(os.path.realpath(__file__))
        self.envConfig = self.configPath + "/test-sim-config.yml"
    
    def tearDown(self):
        pass
    
    def test_rl_env_speed_density(self):
        log = logging.getLogger("Test code : rl_env_density")
        # create config
        test_config = {}
        test_config['sim_params'] = {}
        test_config['sim_params']['render'] = False
        test_config['sim_params']['sim_step'] = 0.1
        test_config['sim_params']['max-speed'] = 35
        
        test_config['rl_params'] = {}
        test_config['rl_params']['num_agents'] = 2
        
        # dump config
        dumpYaml(self.envConfig, test_config)

        env = twoInMergesEnv(self.envConfig)

        numRuns = 5
        densities = np.array([1e-8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        vehicles = np.zeros((densities.shape[0], numRuns))
        speeds = np.zeros((densities.shape[0], numRuns))
        log.debug("Running test for %d densities with runs %d."%(densities.shape[0], numRuns))
        for idx in range(0, densities.shape[0]):
            for episode in range(0, numRuns):
                env.reset(densities[idx])
                assert len(env.k.vehicle.get_rl_ids()) == test_config['rl_params']['num_agents']
                vehicle_count = 0
                steps_count = 0
                speed_sum = 0
                log.debug("Running for density : %.2f, numRuns: %d/%d"%(densities[idx], episode+1, numRuns))    
                
                while True:
                    vehicle_count += len(env.k.vehicle.get_ids())
                    steps_count += 1
                    _, __, done, ____ = env.step(rl_actions=None)
                    assert len(env.k.vehicle.get_rl_ids()) == test_config['rl_params']['num_agents']
                    
                    tmp_speed_sum = 0
                    veh_ids = env.k.vehicle.get_ids()
    
                    for veh_id in veh_ids:
                        sp = env.k.vehicle.get_speed(veh_id)
                        # handler error for teleported vehicles.
                        if sp < 0:
                            sp = 0.0
                        tmp_speed_sum += sp

                    tmp_speed_sum /= (len(veh_ids) + 1e-12)
                    
                    speed_sum += tmp_speed_sum

                    if done:
                        vehicles[idx][episode] = vehicle_count/env.env_params.horizon
                        speeds[idx][episode] = speed_sum/env.env_params.horizon
                        break
        
        avg_vehicles = vehicles.mean(axis=1)
        avg_vehicles = list(avg_vehicles)

        avg_speeds = speeds.mean(axis=1)
        avg_speeds = list(avg_speeds)
        value = avg_speeds[0]
        del avg_speeds[0]

        vehicle_flag = 0
        test_list1 = avg_vehicles[:] 
        test_list1.sort() 
        if (test_list1 == avg_vehicles): 
            vehicle_flag = 1
        
        speed_flag = 0
        test_list1 = avg_speeds[:]
        test_list1.sort()
        if (test_list1 == avg_speeds):
            speed_flag = 1
        
        avg_speeds.insert(0, value)

        plt.clf()
        plt.bar(np.arange(densities.shape[0]), avg_vehicles)
        plt.xticks(np.arange(densities.shape[0]), densities)
        plt.xlabel("vehicle density")
        plt.ylabel('avg-vehicles in scene.')
        plot_title = str(datetime.now()) + " max-speed : " + str(test_config['sim_params']['max-speed'])  
        plt.title(plot_title)
        plt.savefig(self.configPath + "/test_rl_env_density_twoInMerge.png")

        plt.clf()
        plt.bar(np.arange(densities.shape[0]), avg_speeds)
        plt.xticks(np.arange(densities.shape[0]), densities)
        plt.xlabel("vehicle density")
        plt.ylabel('avg-speed in scene.')
        plot_title = str(datetime.now()) + " max-speed : " + str(test_config['sim_params']['max-speed'])  
        plt.title(plot_title)
        plt.savefig(self.configPath + "/test_rl_env_speed_twoInMerge.png")
        
        env.terminate()

    def test_norl_env_speed_density(self):
        log = logging.getLogger("Test code : no_rl_env_density")
        # create config
        test_config = {}
        test_config['sim_params'] = {}
        test_config['sim_params']['render'] = False
        test_config['sim_params']['sim_step'] = 0.1
        test_config['sim_params']['max-speed'] = 35
        
        test_config['rl_params'] = {}
        test_config['rl_params']['num_agents'] = 0
        
        # dump config
        dumpYaml(self.envConfig, test_config)

        env = twoInMergesEnv(self.envConfig)

        numRuns = 5
        densities = np.array([1e-8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        vehicles = np.zeros((densities.shape[0], numRuns))
        speeds = np.zeros((densities.shape[0], numRuns))
        log.debug("Running test for %d densities with runs %d."%(densities.shape[0], numRuns))
        for idx in range(0, densities.shape[0]):
            for episode in range(0, numRuns):
                env.reset(densities[idx])
                vehicle_count = 0
                steps_count = 0
                speed_sum = 0
                log.debug("Running for density : %.2f, numRuns: %d/%d"%(densities[idx], episode+1, numRuns))    
                
                while True:
                    vehicle_count += len(env.k.vehicle.get_ids())
                    steps_count += 1
                    _, __, done, ____ = env.step(rl_actions=None)
                    
                    tmp_speed_sum = 0
                    veh_ids = env.k.vehicle.get_ids()
    
                    for veh_id in veh_ids:
                        sp = env.k.vehicle.get_speed(veh_id)
                        # handler error for teleported vehicles.
                        if sp < 0:
                            sp = 0.0
                        tmp_speed_sum += sp

                    tmp_speed_sum /= (len(veh_ids) + 1e-12)
                    
                    speed_sum += tmp_speed_sum

                    if done:
                        assert steps_count == env.env_params.horizon
                        vehicles[idx][episode] = vehicle_count/env.env_params.horizon
                        speeds[idx][episode] = speed_sum/env.env_params.horizon
                        break
        
        avg_vehicles = vehicles.mean(axis=1)
        avg_vehicles = list(avg_vehicles)

        avg_speeds = speeds.mean(axis=1)
        avg_speeds = list(avg_speeds)
        value = avg_speeds[0]
        del avg_speeds[0]

        vehicle_flag = 0
        test_list1 = avg_vehicles[:] 
        test_list1.sort() 
        if (test_list1 == avg_vehicles): 
            vehicle_flag = 1
        
        speed_flag = 0
        test_list1 = avg_speeds[:]
        test_list1.sort()
        if (test_list1 == avg_speeds):
            speed_flag = 1
        
        avg_speeds.insert(0, value)

        plt.clf()
        plt.bar(np.arange(densities.shape[0]), avg_vehicles)
        plt.xticks(np.arange(densities.shape[0]), densities)
        plt.xlabel("vehicle density")
        plt.ylabel('avg-vehicles in scene.')
        plot_title = str(datetime.now()) + " max-speed : " + str(test_config['sim_params']['max-speed'])  
        plt.title(plot_title)
        plt.savefig(self.configPath + "/test_norl_env_density_twoInMerge.png")

        plt.clf()
        plt.bar(np.arange(densities.shape[0]), avg_speeds)
        plt.xticks(np.arange(densities.shape[0]), densities)
        plt.xlabel("vehicle density")
        plt.ylabel('avg-speed in scene.')
        plot_title = str(datetime.now()) + " max-speed : " + str(test_config['sim_params']['max-speed'])  
        plt.title(plot_title)
        plt.savefig(self.configPath + "/test_norl_env_speed_twoInMerge.png")
        
        env.terminate()
    
logging.basicConfig(stream=sys.stderr,
level=logging.ERROR)
warnings.filterwarnings("ignore", category=ResourceWarning)
logging.getLogger("Test code : no_rl_env_density").setLevel(logging.DEBUG)
logging.getLogger("Test code : rl_env_density").setLevel(logging.DEBUG)

if __name__ == "__main__":
    unittest.main()
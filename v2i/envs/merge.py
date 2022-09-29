''' Define custom merge Environments here '''

# import all v2i modules here
from v2i.utils.fileOps import parseYaml
from v2i.networks.merge import twoInMergeNetwork
from v2i.observation.base import observationBuffer
from v2i.core.params import EXPERIMENT_PARAMS_KEYS
from v2i.networks.merge import ADDITIONAL_NET_PARAMS
from v2i.observation.merge import twoinMergeOccupancy
from v2i.core.params import twoInMergesExperimentParams


# import all flow modules here
from flow.envs.base import Env
from flow.core.params import InFlows
from flow.core.params import NetParams
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.core.params import VehicleParams
from flow.core.params import InitialConfig
from flow.core.params import SumoCarFollowingParams

from flow.controllers.car_following_models import IDMController
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController

# all general import goes here
import ray
import time
import warnings
import numpy as np
from copy import deepcopy
from gym.spaces import Box
import traci.constants as tc


ADDITIONAL_ENV_PARAMS = dict()
NUM_OTHER_VEHICLE_CRASHES_WARN = 5

@ray.remote
class twoInMergesEnv(Env):

    """
    Partially observable lane change and acceleration environment.

    This environment is used to train autonomous vehicles to learn to
    drive with local view.    
    
    Required from envConfig:

    * sim_step: the duration between environment steps.
    * max_accel: maximum acceleration for all vehicles, in m/s^2.
    * max_decel: maximum deceleration for all vehicles, in m/s^2.
    * num_agents: number of rl agents.
    * max_speed: the maximum speed for all vehicles, in m/s^2.
    * lane_change_duration: minimum duration between lane changes for rl vehicles, in s.

    States:

    
    Actions:
        Action consist of:

        * a continous acceleration from -abs(max_decel) to max_accel.
        * a continous lane-change from -1 to 1, used to determine the lateral
        direction the vehicle will take.
    
    Rewards

    Termination
        A rollout is terminated if the time horion is reached or any autonomous
        vehicle crashes into one another.

    """

    def __init__(self, envConfig):
        
        # Read and validate environment configuration file.
        self.envConfig = twoInMergesExperimentParams(parseYaml(envConfig))

        # Collision penalty.
        self.collision_penalty = -30

        # sumo simulation parameters
        sim_params = SumoParams(sim_step=self.envConfig.get_parameter_value('sim_params', 'sim_step'),
                                 render=self.envConfig.get_parameter_value('sim_params', 'render'),
                                 restart_instance=True)

        # sumo environment parameters
        sim_seconds = 150
        horizon = int(sim_seconds/self.envConfig.get_parameter_value('sim_params', 'sim_step'))
        env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS,
                               horizon=horizon)
        
        # setup traffic densities
        self.densities = np.linspace(1e-08, 1.0, num=15)
        self.densities = np.round(self.densities, 8)

        # inital config
        initial_config = InitialConfig()

        # net params
        ADDITIONAL_NET_PARAMS['speed_limit'] = self.envConfig.get_parameter_value('sim_params', 'max-speed')
        net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS, 
                                inflows=None)
        
        # create vehicle types.
        vehicles = VehicleParams()
        
        # add human vehicle type
        vehicles.add("human", 
                     acceleration_controller=(IDMController, {}),
                     car_following_params=SumoCarFollowingParams(speed_mode="obey_safe_speed",
                                                                 tau=2.0,
                                                                 accel=self.envConfig.get_parameter_value('sim_params', 'accel'),
                                                                 decel=self.envConfig.get_parameter_value('sim_params', 'decel'),
                                                                 max_speed=self.envConfig.get_parameter_value('sim_params', 'max-speed')),
                     color='red',
                     num_vehicles=0)

        # add rl vehicles
        vehicles.add("rl",
                     acceleration_controller=(RLController, {}),
                     car_following_params=SumoCarFollowingParams(speed_mode=0,
                                                                 tau=2.0,
                                                                 max_speed=self.envConfig.get_parameter_value('sim_params', 'max-speed'),
                                                                 accel=self.envConfig.get_parameter_value('sim_params', 'accel'),
                                                                 decel=self.envConfig.get_parameter_value('sim_params', 'decel')),
                     color='yellow',
                     num_vehicles=1,
                     initial_speed=0.0)

        # create in-merge network for Env initialization
        network = twoInMergeNetwork(name="twoInMergesNetwork",
                                    vehicles=vehicles,
                                    net_params=net_params,
                                    initial_config=initial_config)
        simulator = "traci"

        super().__init__(env_params, sim_params, network, simulator)

        # Occupancy grid handler
        self.gridHandler = twoinMergeOccupancy(view_length=self.envConfig.get_parameter_value('rl_params', 'view-size-length'),
                                               view_width=self.envConfig.get_parameter_value('rl_params', 'view-size-width'),
                                               scale=5)
        
        # Create an occupancy buffer to hold past occupancies
        self.occBuffer = observationBuffer(memorySize=self.envConfig.get_parameter_value('rl_params', 'k-frames'),
                                           expectedShape=self.gridHandler.surface.size,
                                           initial_value=0.0)
        
        # Create an ego-vehicle speed buffer to hold past speeds.
        self.egoSpeedBuffer = observationBuffer(memorySize=self.envConfig.get_parameter_value('rl_params', 'k-frames'),
                                                expectedShape=(1,),
                                                initial_value=-1)
        
    
    def get_state(self):
        return np.zeros(10)
    
    def update_inflow_rate(self, vehicle_density):

        inflow = InFlows()
        
        # add inflow to highway_0 edge as per vehicle_density
        inflow.add(veh_type="human",
                   edge="highway_0",
                   probability=vehicle_density,
                   depart_lane="random",
                   depart_speed="random")
        
        # add inflow to merge_one edge as per vehicle_density
        
        inflow.add(veh_type="human",
                   edge="merge_one",
                   probability=vehicle_density/2,
                   depart_lane="random",
                   depart_speed="random")
        
        # add inflow to merge_two edge as per vehicle_density
        inflow.add(veh_type="human",
                   edge="merge_two",
                   probability=vehicle_density,
                   depart_lane="random",
                   depart_speed="random")
        
        net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS,
                               inflows=inflow)

        self.network = self.network.__class__(
            self.network.orig_name,
            self.network.vehicles,
            net_params,
            self.network.initial_config,
            self.network.traffic_lights)
        self.k.vehicle = deepcopy(self.initial_vehicles)
        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

    def reset(self, density=None):
        """See parent class.

        The sumo instance is reset with a new inflow_rate.
        """
        
        # reset the step counter
        self.step_counter = 0
        self.custom_step_counter = 0

        # reset other vehicle crash counters
        self.other_vehicle_crash_count = 0

        # update the network
        if density is None:
            vehicle_density = np.random.choice(self.densities)
        else:
            vehicle_density = density

        # update inflow rate as per vehicular density
        self.update_inflow_rate(vehicle_density)
        
        # restart the sumo instance
        self.restart_simulation(
            sim_params=self.sim_params,
            render=self.sim_params.render)

        # perform the generic reset function
        super().reset()

        # Make sure the episode always start with required number of rl agents
        if self.k.vehicle.num_rl_vehicles != 1:
            raise AssertionError("Episode didn't started with required number of rl agents.")
        
        # Get rl vehicles ids
        self.rl_id_list = sorted(deepcopy(self.k.vehicle.get_rl_ids()))
        
        '''
        warm-up iterations to make sure steady state has arrived 
        before introducing any rl vehicle.
        '''
        print("Running Warm up iters.")
        warm_up_time = 60
        warm_up_time_steps = int(warm_up_time/self.envConfig.get_parameter_value('sim_params', 'sim_step'))
        for _ in range(0, warm_up_time_steps):
            # DONOT use rl_actions = None, as it puts the control
            # of ego-vehicle in the hand of SUMO.
            self.step(rl_actions=np.array([0.0, 0.0]), idle=True)
        print("Running Warm up iters. Done.")
        
        # Update occupancy observations.
        self.occBuffer.reset()
        self.gridHandler.update_surface(self.k)
        self.occBuffer.addObs(self.gridHandler.get_occupancy(normalize=True).transpose())

        # Update ego-vehicle speed observations.
        self.egoSpeedBuffer.reset()
        self.egoSpeedBuffer.addObs(np.array([self.k.vehicle.get_speed('rl_0')]))

        # Start tracking previous vehicle speed.
        self.ego_prev_speed = self.k.vehicle.get_speed('rl_0')
        assert self.ego_prev_speed != -2**30

        if self.envConfig.get_parameter_value('sim_params', 'render'):
            self.gridHandler.render()
        
        return (self.verify_occupancy_observation(self.occBuffer.getObs()), 
                self.verify_speed_observation(self.egoSpeedBuffer.getObs().flatten()))
    
    def additional_command(self):
        """Reintroduce any RL vehicle that may have exited in the last step.
        This is used to maintain a constant number of RL vehicle in the system
        at all times, in order to comply with a fixed size observation and
        action space.

        Adapted from flow.envs.bottleneck.py : additional_command
        """
        super().additional_command()
        num_rl_vehicles = self.k.vehicle.num_rl_vehicles

        if num_rl_vehicles != 1:
            # find the vehicles that have exited.
            diff_list = (
                set(self.rl_id_list).difference(self.k.vehicle.get_rl_ids()))
            for rl_id in diff_list:
                try:
                    self.k.vehicle.add(
                        veh_id=rl_id,
                        edge="rl_highway",
                        type_id=str('rl'),
                        lane="random",
                        pos="0",
                        speed=0.0,)
                    
                except Exception as e:
                    pass
    
    def frame_step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicle and TrafficLight kernels, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : array_like
            agent's observation of the current environment
        reward : float
            amount of reward associated with the previous state/action pair
        done : bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # Get ids of collided vehicles
            collided_vehicle_ids = set(self.k.kernel_api.simulation.getStartingTeleportIDList())
            
            if len(collided_vehicle_ids) > 0:
                print("List of collided vehicles : ", collided_vehicle_ids)

            # Get rl vehicle ids
            rl_vehicle_ids = set(['rl_0'])

            # Get list of other vehiles minus rl vehicles
            other_vehicle_ids = set(self.k.vehicle.get_ids()) - rl_vehicle_ids

            if (len(rl_vehicle_ids.intersection(collided_vehicle_ids)) > 0):
                print("Setting crash to true.")
                crash = True
            
            elif (len(other_vehicle_ids.intersection(collided_vehicle_ids)) > 0):
                self.other_vehicle_crash_count += len(other_vehicle_ids.intersection(
                    collided_vehicle_ids))
                crash = False
            
            else:
                crash = False

            # make sure crash is set to true if it is in collision list.
            if 'rl_0' in collided_vehicle_ids and crash != True:
                raise ValueError("RL vehicle was collided but crash was set to False.")
            
            # stop collecting new simulation steps if there is a collision
            if crash:
                break

            # render a frame
            self.render()

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        # test if the environment should terminate due to a collision
        done = crash
        
        if done and self.other_vehicle_crash_count > NUM_OTHER_VEHICLE_CRASHES_WARN:
            warnings.warn("Episode ended with %d other vehicle crashes."
                          %(self.other_vehicle_crash_count))  
        
        return done


    def compute_reward(self, rl_actions, **kwargs):
        '''Reward function is a simple ego-vehicle speed based 
        reward.
        '''
        current_speed = self.k.vehicle.get_speed('rl_0')
        error_on_speed = False
        if current_speed == -2**30:
            current_speed = self.ego_prev_speed
            error_on_speed = True
        else:
            self.ego_prev_speed = current_speed
        
        """
        DONOT use k.vehicle.get_max_speed().
        The function may-cause issues, if terminal state is reached.
        At-terminal state ego-vehicle is removed somtimes. 
        Hence, querying the speed for 'rl_0' may return vehicle 
        is not known. 
        """
        max_speed = self.envConfig.get_parameter_value('sim_params', 'max-speed') + 0.1
        reward = current_speed/max_speed
        
        if reward > 1.1 or reward < 0.0:
            raise ValueError("Reward should be between [0, 1]. ")
        
        return reward, error_on_speed, current_speed
    
    def get_action_space(self, ):
        return self.action_space
    
    def get_observation_space(self, ):
        return self.observation_space
    
    @property
    def action_space(self):
        max_accel = self.envConfig.get_parameter_value('sim_params', 'accel')
        max_decel = self.envConfig.get_parameter_value('sim_params', 'decel')

        lb = [-abs(max_decel), -1] * 1
        ub = [max_accel, 1] * 1

        return Box(np.array(lb), np.array(ub), dtype=np.float32)
    
    @property
    def observation_space(self):
        self.obs_var_labels = ['occupancy-grid', 'ego-vehicle speed']
        
        # Occupancy grid bounds
        occ_shape = []
        occ_shape.append(self.envConfig.get_parameter_value('rl_params', 'k-frames'))
        occ_shape = occ_shape + list(self.gridHandler.surface.size)

        image_low_bound = np.zeros(occ_shape)
        image_high_bound = np.ones(occ_shape)
        
        # Occupancy observation space
        occ_obs_space = Box(
            low=image_low_bound,
            high=image_high_bound,
            dtype=np.float)
        
        # Ego-vehicle speed bounds
        speed_lower_bound = -1.0
        speed_higher_bound = self.envConfig.get_parameter_value('sim_params', 'max-speed') + 0.1

        # Ego-vehicle speed observation space

        speed_obs_space = Box(
            low=speed_lower_bound,
            high=speed_higher_bound,
            shape=(1 * self.envConfig.get_parameter_value('rl_params', 'k-frames'), ),
            dtype=np.float)

        return occ_obs_space, speed_obs_space
    
    def _apply_rl_actions(self, rl_actions):
        acceleration = rl_actions[::2][:len(self.rl_id_list)]
        direction = np.round(rl_actions[1::2])[:len(self.rl_id_list)]
        
        self.k.vehicle.apply_acceleration(self.rl_id_list, 
                                          acc=acceleration,
                                          max_speed=self.envConfig.get_parameter_value('sim_params', 'max-speed'))
        self.k.vehicle.apply_lane_change(self.rl_id_list, 
                                         direction=direction)
    
    def verify_occupancy_observation(self, grid):
        if (grid >= self.observation_space[0].low.min()).all() and (grid <= self.observation_space[0].high.max()).all():
            return grid
        else:
            raise RuntimeError("Occupancy grid should lie between [0, 1]")
    
    def verify_speed_observation(self, speeds):
        if (speeds >= self.observation_space[1].low.min()).all() and (speeds <= self.observation_space[1].high.max()).all():
            return speeds
        else:
            warnings.warn("Ego-speed observation should lie between [-1, %.2f], Got, Min : %.6f, Max:%.6f"%(self.k.vehicle.get_max_speed('rl_0'),
                                                                                                                 speeds.min(),
                                                                                                                 speeds.max()))
            return speeds
    
    def check_horizon_reached(self, ):
        return (self.custom_step_counter >= self.env_params.horizon)
    
    def step(self, rl_actions, idle=False):
        
        # Increment the step counter
        if not idle:
            self.custom_step_counter += 1

        # Check if horizon is reached
        if self.custom_step_counter > self.env_params.horizon:
            raise RuntimeError("Env horizon is reached. Please start a new episode by calling env.reset()")

        done = self.frame_step(rl_actions)
        count = 0
        while len(self.k.vehicle.get_rl_ids()) != 1:
            if done:
                break
            count += 1
            done = self.frame_step(rl_actions=None)
            num_steps = (1/self.envConfig.get_parameter_value('sim_params', 'sim_step')) * 10
            if count >= num_steps:
                raise RuntimeError("step function got stuck in while loop for %d steps."%(count))
        
        # compute the info for each agent
        infos = {}
        infos['other_vehicle_crash_count'] = self.other_vehicle_crash_count

        # compute the reward
        if self.env_params.clip_actions:
            rl_clipped = self.clip_actions(rl_actions)
            reward, error_on_speed, ego_speed = self.compute_reward(rl_clipped, fail=done)
        else:
            reward, error_on_speed, ego_speed = self.compute_reward(rl_actions, fail=done)

        if error_on_speed:
            warnings.warn("Error was detected on fetching ego-vehicle speed but fixed automatically.")
            assert done == True, "Error on getting agent-speed. However, state wasn't terminal."
            # Update ego speed observation
            self.egoSpeedBuffer.addObs(np.array([ego_speed]))
            infos['ego-speed'] = ego_speed

        else:
            # Update ego-vehicle speed observations.
            infos['ego-speed'] = ego_speed
            self.egoSpeedBuffer.addObs(np.array([self.k.vehicle.get_speed('rl_0')]))

        # Update occupancies
        '''
        After some episodes SUMO returns invalid postion for rl vehicle
        after collision. However, the position is required to create the
        occupancy map. To solve this issue, we augument the observation of
        zeros as the terminal state for such scenarios.
        '''
        try:
            self.gridHandler.update_surface(self.k)
            self.occBuffer.addObs(self.gridHandler.get_occupancy(normalize=True).transpose())

        except Exception as e:
            if error_on_speed == False:
                print("ERROR: Didn't caused due to termination of episode.")
            self.occBuffer.addObs(np.zeros(self.occBuffer.expectedShape))
        
        if self.envConfig.get_parameter_value('sim_params', 'render'):
            self.gridHandler.render()
        
        # Compute observation
        obs = (self.verify_occupancy_observation(self.occBuffer.getObs()), 
               self.verify_speed_observation(self.egoSpeedBuffer.getObs().flatten()))
        
        # Check whether the horizon is reached.
        infos['horizon_reached'] = self.check_horizon_reached()

        # Penalize if collision takes place.
        if done:
            reward = self.collision_penalty

        return obs, reward, done, infos

if __name__ == "__main__":

    env = twoInMergesEnv("experiments/inMerge/sim-config.yml")
    numRuns = 1
    #print(env.env_params.horizon)
    
    for run in range(0, numRuns):
        env.reset(1.0)
        count = 0
        speed_sum = 0.0
        reward_sum = 0.0

        while True:
            #time.sleep(0.2)
            count += 1
            actions = np.array([2.7, 0.0])
            #actions = env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            reward_sum += reward

            if done or info['horizon_reached']:
                print(reward_sum)
                print("Run %d completed"%(run+1))
                break
        
        #input('waiting to exit')
        
    env.terminate()
    
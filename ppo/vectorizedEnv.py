""" Defines parallel gym environments """
import ray
import time
import torch
import psutil
import warnings
import importlib
import cloudpickle
import numpy as np
from torch.multiprocessing import Process, Pipe
    
class vecEnv:

    """
    Base class for vectorizing v2i Envs using ray.

    This base class provides method similar to gym 
    APIs to access the internal state of the environment.
    However, the functions are vectorized and executes 
    for all workers concurrently.

    Attributes
    ----------

    * num_workers: Number of parallel environments to spawn.
    """

    def __init__(self,
                 num_workers):        
        if num_workers > psutil.cpu_count():
            warnings.warn("Using more than logical workers. May reduce performance. Logical cores : {}, Requested : {}.".format(psutil.cpu_count(), num_workers))

        self.num_workers = num_workers
        ray.init(num_cpus=self.num_workers)
    
    def init_space(self, env):
        """Initializes the observation given an instance of
        the environment.
        """
        # Get observation space
        oid = env.get_observation_space.remote()
        self.observation_space = self.fetch_data(oid)
        # Get action space
        oid = env.get_action_space.remote()
        self.action_space = self.fetch_data(oid)
    
    def fetch_data(self, object_ids):
        '''This function is used to fetch the
            data returned by the ray.remote 
            function.

        Attributes
        ----------

        * object_ids: ray.remote function object ids.
        '''
        return ray.get(object_ids)
    
    def sample_actions(self, envs):
        '''This function is used sample action
        from the specified enviornment action space. 

        Attributes
        ----------

        * envs: The list of (remote) instances of workers.
        '''
        object_ids = [env.action_space.remote() for env in self.envs]
        actions = [space.sample() for space in self.fetch_data(object_ids)]
        return actions
    
    def reset(self, ):
        '''Used to reset the state of all of the 
            workers.
        '''
        raise NotImplementedError
    
    def step(self, rl_actions):
        '''Used to step throught the state of the
            environment.
        '''
        raise NotImplementedError

class vecV2IEnv(vecEnv):

    def __init__(self,
                 env_id,
                 env_config,
                 num_workers):

        super().__init__(num_workers)
        module = importlib.import_module('v2i.envs')
        self.envs = []
        for _ in range(0, self.num_workers):
            self.envs.append(getattr(module,
                                     env_id).remote(env_config))
        
        self.init_space()
    
    def init_space(self, ):
        super().init_space(self.envs[0])
    
    def reset(self, density=None):
        if density is not None:
            assert type(density) == list and len(density) == self.num_workers, "Density should be a list of size equal to number of vectorized envs."
        else:
            density = [None] * self.num_workers
        
        # Execute reset
        dataIds = [env.reset.remote(density=density[idx]) for idx, env in enumerate(self.envs) ]
        return self.fetch_data(dataIds)
    
    def step(self, rl_actions):
        assert type(rl_actions) == list and len(rl_actions) == self.num_workers, "rl_Actions should be a list of size equal to number of vectorizd envs."

        # Execute step
        object_ids = [env.step.remote(rl_actions[idx]) for idx, env in enumerate(self.envs)]
        
        next_states = []
        rewards = []
        dones = []
        infos = []
        
        for idx, (next_state, reward, done, info) in enumerate(self.fetch_data(object_ids)):
            
            if done or info['horizon_reached']:
                oid = self.envs[idx].reset.remote()
                next_state = self.fetch_data(oid)

            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return next_states, rewards, dones, infos
    
    def sample_actions(self):
        return super().sample_actions(self.envs)
    
    def _del_(self, ):
        for env in self.envs:
            env.terminate.remote()
        
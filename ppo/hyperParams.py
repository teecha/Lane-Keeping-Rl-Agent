""" Defines the hyper-parameters for PPO algorithm. """

from v2i.core.base_params import ExperimentParams

EXPERIMENT_PARAMS_KEYS = {
    'ppo_params': [('env', str), ('env_config', str), ('workers', int), ('num_training_steps', int), 
                   ('num_steps', int), ('tau', float), ('gamma', float), ('ppo_iters', int), ('mini_batch_size', int),
                   ('clip_ratio', float), ('critic_coef', float), ('actor_coef', float), ('entropy_coef', float),
                   ('lr', float)],
}

class ppoExperimentParams(ExperimentParams):

    def __init__(self, config):
        super().__init__(config,
                         EXPERIMENT_PARAMS_KEYS)
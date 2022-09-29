""" Defines the hyper-parameters for an environment. """

from flow.core.params import EnvParams
from flow.core.params import NetParams

from v2i.core.base_params import ExperimentParams

EXPERIMENT_PARAMS_KEYS = {
    'sim_params': [('sim_step', float), ('render', bool), ('max-speed', int), ('accel', float), ('decel', float)],
    'rl_params': [('view-size-length', int), ('view-size-width', int), ('k-frames', int)],
}

class twoInMergesExperimentParams(ExperimentParams):

    def __init__(self, expConfig):
        super().__init__(expConfig,
                         EXPERIMENT_PARAMS_KEYS)

if __name__ == "__main__":

    config = {
        'sim_params': {
            'sim_step': 0.1,
            'render': 1.0,
        }
    }

    env = twoInMergesExperimentParams(config)
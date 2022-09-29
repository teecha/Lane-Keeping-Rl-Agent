import wandb
import torch
import argparse
import torch.nn as nn
from datetime import datetime

parser = argparse.ArgumentParser(description="training script for v2i environments.")
parser.add_argument("-c", "--config", required=True, type=str, help="experiment configuration file.")
parser.add_argument("-l", "--log", default=1, type=int, help="Enable/disable logging using wandb. (1 - Enable) else disable.")
parser.add_argument("-p", "--project", default='default', type=str, help="project name for wandb logging.")

from v2i.envs import __all__
from v2i.utils.fileOps import parseYaml
from ppo.vectorizedEnv import vecV2IEnv
from ppo.hyperParams import ppoExperimentParams
from ppo.policy import gaussianPolicy
from ppo.algos import ppoTrainer
import torch.multiprocessing as mp

def get_hyperParameters(expConfig, key):
    config = {}
    params = expConfig.get_paramters(key)
    for param in params:
        config[param] = expConfig.get_parameter_value(key, param)
    return config

if __name__ == "__main__":

    mp.set_start_method('spawn')
    
    # Check if CUDA is available.
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # Parse arguments
    args = parser.parse_args()

    # Read experiment parameters.
    expConfig = ppoExperimentParams(parseYaml(args.config))
    config = get_hyperParameters(expConfig, 'ppo_params')

    # Initialize wandb
    if args.log:
        curr_time = datetime.now()
        exp_name = curr_time.strftime("%d/%m/%Y %H:%M:%S")
        wandb.init(name=exp_name,
                project=args.project,
                save_code=True,
                config=config)
    
    # Make sure specified environment is an v2i env.
    if expConfig.get_parameter_value('ppo_params', 'env') not in __all__:
        raise ValueError('%s is not a valid v2i environment.'%(expConfig.get_parameter_value('ppo_params', 'env')))

    # Create environment objects.
    envs = vecV2IEnv(expConfig.get_parameter_value('ppo_params', 'env'),
                     expConfig.get_parameter_value('ppo_params', 'env_config'),
                     expConfig.get_parameter_value('ppo_params', 'workers'))
    
    # Create a centralized actor and critic.
    policy = gaussianPolicy(obs_space=envs.observation_space,
                            action_space=envs.action_space)
    if args.log:
        wandb.watch(policy)
    
    # Create a ppo trainer.
    trainer = ppoTrainer(policy, envs, device, args.log)
    
    # Start training
    trainer.train(num_train_steps=expConfig.get_parameter_value('ppo_params', 'num_training_steps'),
                  num_steps=expConfig.get_parameter_value('ppo_params', 'num_steps'),
                  gamma=expConfig.get_parameter_value('ppo_params', 'gamma'),
                  tau=expConfig.get_parameter_value('ppo_params', 'tau'),
                  ppo_iters=expConfig.get_parameter_value('ppo_params', 'ppo_iters'),
                  mini_batch_size=expConfig.get_parameter_value('ppo_params', 'mini_batch_size'),
                  clip_ratio=expConfig.get_parameter_value('ppo_params', 'clip_ratio'),
                  critic_coef=expConfig.get_parameter_value('ppo_params', 'critic_coef'),
                  actor_coef=expConfig.get_parameter_value('ppo_params', 'actor_coef'),
                  entropy_coef=expConfig.get_parameter_value('ppo_params', 'entropy_coef'),
                  lr=expConfig.get_parameter_value('ppo_params', 'lr'))
    
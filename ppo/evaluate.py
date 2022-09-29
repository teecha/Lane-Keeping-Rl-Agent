import torch
import argparse
import torch.nn as nn

parser = argparse.ArgumentParser(description="Evaluation script for v2i environments.")
parser.add_argument("-c", "--config", required=True, type=str, help="experiment configuration file.")
parser.add_argument("-m", "--policy-checkpoint", default='default', type=str, help="policy checkpoint path.")

from v2i.envs import __all__
from ppo.policy import gaussianPolicy
from v2i.utils.fileOps import parseYaml
from ppo.vectorizedEnv import vecV2IEnv
from ppo.hyperParams import ppoExperimentParams

if __name__ == "__main__":

    # Check if CUDA is available
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # Parse arguments
    args = parser.parse_args()

    # Read experiment parameters.
    expConfig = ppoExperimentParams(parseYaml(args.config))
    
    # Make sure specified environment is an v2i env.
    if expConfig.get_parameter_value('ppo_params', 'env') not in __all__:
        raise ValueError('%s is not a valid v2i environment.'%(expConfig.get_parameter_value('ppo_params', 'env')))
    
    # Create environment objects.
    envs = vecV2IEnv(expConfig.get_parameter_value('ppo_params', 'env'),
                     expConfig.get_parameter_value('ppo_params', 'env_config'),
                     1)
    
    # Create a centralized actor and critic.
    policy = gaussianPolicy(obs_space=envs.observation_space,
                            action_space=envs.action_space)
    
    # Load policy from checkpoint
    policy.set_weights(torch.load(args.policy_checkpoint, map_location=torch.device(device)))

    # Put policy in eval model.
    policy.eval()


    policy.to(device)

    for episode in range(0, 10):
        prev_occ, prev_speeds = zip(*envs.reset([0.8]))
        episode_reward = 0.0

        while True:
            
            prev_occ = torch.FloatTensor(prev_occ).to(device)
            prev_speeds = torch.FloatTensor(prev_speeds).to(device)

            with torch.no_grad():
                dist, _ = policy.forward((prev_occ, prev_speeds))
            
            action = dist.sample()[0].cpu().numpy()
            next_state, reward, done, info = envs.step([action])
            episode_reward += 1

            next_occ, next_speeds = zip(*next_state)
            
            prev_occ = next_occ
            prev_speeds = next_speeds

            if done[0] or info[0]['horizon_reached']:
                print("Episode Ended with reward {}.".format(episode_reward))
                break


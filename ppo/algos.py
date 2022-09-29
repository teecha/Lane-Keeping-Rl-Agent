import os
import time
import wandb
import torch
import numpy as np
import torch.nn as nn

class ppoTrainer:
    """
    Class to create a ppo trainer object.

    The trainer object provide method to traing and
    evluate a ppo based Reinforcement learning agent.

    Attributes
    ----------

    * policy: A pyTorch policy that provide methods to
                sample actions.
    * envs: v2i Envrionments to interact with the simulation.
    * device: The device to use for training or evaluating.
    * log: Enable/disable logging.

    """
    def __init__(self, policy, envs, device, log):
        self.policy = policy
        self.envs = envs
        self.device = device
        self.log = log

        # Move the policy to specified device.
        self.policy = self.policy.to(device)

    def compute_gae(self, 
                    next_value,
                    rewards,
                    masks,
                    values,
                    gamma,
                    tau):
        
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])

        return returns
    
    def ppo_iter(self, 
                 mini_batch_size,
                 states,
                 actions,
                 log_probs,
                 returns    ,
                 advantage):
        
        batch_size = len(states[0])
        occupancies = states[0]
        speeds = states[1]

        ids = np.random.permutation(batch_size)
        ids = np.split(ids[:batch_size//mini_batch_size * mini_batch_size], batch_size//mini_batch_size)
        
        for i in range(len(ids)):
            yield occupancies[ids[i], :], speeds[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]
        
    def ppo_update(self,
                   ppo_epochs,
                   mini_batch_size,
                   states,
                   actions,
                   log_probs,
                   returns,
                   advantage,
                   clip_ratio,
                   entropy_coef,
                   actor_coef,
                   critic_coef):

        logging_metrics = {}
        logging_metrics['combined_loss'] = 0.0
        logging_metrics['actor_loss'] = 0.0
        logging_metrics['critic_loss'] = 0.0
        logging_metrics['entropy'] = 0.0

        for _ in range(0, ppo_epochs):
            for occ, speeds, action, old_log_prob, return_, advantages in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
                dist, value = self.policy.forward((occ, speeds))
                
                # policy entropy
                entropy = dist.entropy().mean()
                logging_metrics['entropy'] += entropy.item()

                new_log_probs = dist.log_prob(action)
                ratio = (new_log_probs - old_log_prob).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages

                # policy loss
                actor_loss = -torch.min(surr1, surr2).mean()
                logging_metrics['actor_loss'] += actor_loss.item()

                # value function loss
                value_loss = (return_ - value).pow(2).mean()
                logging_metrics['critic_loss'] += value_loss.item()

                # combined loss
                loss = (actor_coef * actor_loss) + (critic_coef * value_loss) - (entropy_coef * entropy)
                logging_metrics['combined_loss'] += loss.item()

                # clear previous grads.
                self.optimizer.zero_grad()

                # backprop
                loss.backward()

                # update parameters
                self.optimizer.step()
        
        logging_metrics['combined_loss'] /= ppo_epochs
        logging_metrics['critic_loss'] /= ppo_epochs
        logging_metrics['actor_loss'] /= ppo_epochs
        logging_metrics['entropy'] /= ppo_epochs
        
        return logging_metrics

    def train(self, 
              num_train_steps, 
              num_steps,
              gamma,
              tau,
              ppo_iters,
              mini_batch_size,
              clip_ratio,
              actor_coef,
              entropy_coef,
              critic_coef,
              lr):

        # Create an optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), 
                                          lr=lr)

        # log episodic rewards
        episodic_rewards = [[] for _ in range(self.envs.num_workers)]
        reward_counter = [0] * self.envs.num_workers

        # log epiosdes length
        episode_length = [[] for _ in range(self.envs.num_workers)]
        episode_length_counter = [0] * self.envs.num_workers

        # Counters for statistics
        episode_counter = 0
        collision_counter = 0
    
        occ_states, speed_states = zip(*self.envs.reset())

        for _iter_ in range(0, num_train_steps):

            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            occupancies_states = []
            ego_speed_states = []
            entropy = 0

            collect_start_time = time.time()

            for step in range(0, num_steps):
                
                occ_states = torch.FloatTensor(occ_states).to(self.device)
                speed_states = torch.FloatTensor(speed_states).to(self.device)
                
                dist, value = self.policy.forward((occ_states, speed_states))    
                action = dist.sample()
                action_list = list(action.cpu().numpy())

                next_state, reward, done, info = self.envs.step(action_list)
                
                # log rewards and other stats.
                for agent_id, (agent_reward, terminate, step_info) in enumerate(zip(reward, done, info)):
                    
                    reward_counter[agent_id] += agent_reward
                    episode_length_counter[agent_id] += 1

                    if terminate or step_info['horizon_reached']:
                        episode_counter += 1

                        # reward logging
                        episodic_rewards[agent_id].append(reward_counter[agent_id])
                        reward_counter[agent_id] = 0.0

                        # episode length logging
                        episode_length[agent_id].append(episode_length_counter[agent_id])
                        episode_length_counter[agent_id] = 0.0
                    
                    if terminate:
                        collision_counter += 1

                occ_next, speed_next = zip(*next_state)
                done = np.array(done)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1-done).unsqueeze(1).to(self.device))
                
                occupancies_states.append(occ_states)
                ego_speed_states.append(speed_states)
                actions.append(action)

                occ_states = occ_next
                speed_states = speed_next

            collect_end_time = time.time()

            occ_next = torch.FloatTensor(occ_next).to(self.device)
            speed_next = torch.FloatTensor(speed_next).to(self.device)
            _, next_value = self.policy.forward((occ_next, speed_next))

            # Compute generalized advantage estimate.
            returns = self.compute_gae(next_value,
                                       rewards,
                                       masks,
                                       values,
                                       gamma,
                                       tau)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            occupancies_states = torch.cat(occupancies_states)
            ego_speed_states = torch.cat(ego_speed_states)
            actions = torch.cat(actions)
            advantage = returns - values
            
            metrics = self.ppo_update(ppo_epochs=ppo_iters, 
                            mini_batch_size=mini_batch_size,
                            states=(occupancies_states, ego_speed_states),
                            actions=actions,
                            log_probs=log_probs,
                            returns=returns,
                            advantage=advantage,
                            clip_ratio=clip_ratio,
                            entropy_coef=entropy_coef,
                            critic_coef=critic_coef,
                            actor_coef=actor_coef)
            
            # Save the updated policy
            if ((_iter_ + 1) % 5) == 0:
                model_name = "checkpoint_{}.pkt".format(_iter_+1)
                torch.save(self.policy.get_weights(), os.path.join(wandb.run.dir, model_name))
                print("Saved ", model_name, " in ", wandb.run.dir)

            # Calculate episodic reward statistics
            episodic_rewards_mean = []
            for reward_list in episodic_rewards:
                reward_list = np.array(reward_list)
                env_mean_reward = reward_list.sum()/(reward_list.shape[0] + 1e-8)
                episodic_rewards_mean.append(env_mean_reward)
            
            # Calculate episode length stats.
            episode_length_mean = []
            for episode_length_list in episode_length:
                episode_length_list = np.array(episode_length_list)
                mean_episode_length = episode_length_list.sum()/(episode_length_list.shape[0] + 1e-8)
                episode_length_mean.append(mean_episode_length)
            
            metrics['train_step'] = _iter_
            metrics['episodic_rewards_mean'] = np.array(episodic_rewards_mean).mean()
            metrics['episodic_reward_max'] = np.array(episodic_rewards_mean).max()
            metrics['episodic_reward_min'] = np.array(episodic_rewards_mean).min()

            metrics['episode_length_mean'] = np.array(episode_length_mean).mean()
            metrics['episode_length_max'] = np.array(episode_length_mean).max()
            metrics['episode_length_min'] = np.array(episode_length_mean).min()
            
            # Counter based statistics
            metrics['episodes_completed'] = episode_counter
            metrics['collision_mean (0-1)'] = collision_counter/(episode_counter + 1e-8)
            metrics['throughput (ray-envs) samples/sec'] = (num_steps * self.envs.num_workers)/((collect_end_time-collect_start_time) + 1e-8)

            if self.log:
                wandb.log(metrics)
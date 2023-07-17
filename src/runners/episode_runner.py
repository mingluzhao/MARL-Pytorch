from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np

import matplotlib.pyplot as plt
import os
import logging
import shutil
import copy
import sys
import torch as th
import time


class EpisodeRunner:
    ### need to re-write for different envrionments

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        MAX_STEPS = 25
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = MAX_STEPS
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.verbose = args.verbose

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    ### re-write get_env_info
    def get_env_info(self):
        n_agents = self.env.n
        actor_dims = []
        for i in range(n_agents):
            actor_dims.append(self.env.observation_space[i].shape[0])
        obs_dims = self.env.observation_space[0].shape[0]
        n_actions = self.env.action_space[0].n

        env_info = {"state_shape": 62,
                    "obs_shape": obs_dims,   ### changed to 64
                    "n_actions": n_actions,
                    "n_agents": n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
    

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        obs = self.env.reset()  ### changed this
        self.t = 0
        return obs

    def run(self, test_mode=False, t_episode=0):
       
        obs = self.reset()
        #print("observation", obs)
        state = np.concatenate(obs, axis=0).astype(
                np.float32
            )
        #print(state)

        # zero pad the last obs
        obs[-1] = np.pad(obs[-1], (0, 2), 'constant', constant_values=(0))

        
        n_agents = self.env.n
        episode_step = 1
        terminated = False
        episode_return = 0
        score_history = []
        PRINT_INTERVAL = 100
        score = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        replay_data = []
        # if self.verbose:
        #     if t_episode < 2:
        #         save_path = os.path.join(self.args.local_results_path,
        #                                  "pic_replays",
        #                                  self.args.unique_token,
        #                                  str(t_episode))
        #         if os.path.exists(save_path):
        #             shutil.rmtree(save_path)
        #         os.makedirs(save_path)
        #         role_color = np.array(['r', 'y', 'b', 'c', 'm', 'g'])
        #         print(self.mac.role_action_spaces.detach().cpu().numpy())
        #         logging.getLogger('matplotlib.font_manager').disabled = True
        #     all_roles = []

        while not terminated:
            # print("episode number", episode_step, "---------")

           
            self.env.render()
            time.sleep(0.1)

            pre_transition_data = {
                "state": [state], # get global state
                "avail_actions": [[1, 1, 1, 1, 1] for i in range(n_agents)],
                "obs": [obs]
            }

            # if self.verbose:
            #     # These outputs are designed for SMAC
            #     ally_info, enemy_info = self.env.get_structured_state()
            #     replay_data.append([ally_info, enemy_info])
            
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, roles, role_avail_actions = self.mac.select_actions(self.batch, t_ep=self.t,
                                                                         t_env=self.t_env, test_mode=test_mode)
            self.batch.update({"role_avail_actions": role_avail_actions.tolist()}, ts=self.t)

            # if self.verbose:
            #     roles_detach = roles.detach().cpu().squeeze().numpy()
            #     ally_info = replay_data[-1][0]
            #     p_roles = np.where(ally_info['health'] > 0, roles_detach,
            #                        np.array([-5 for _ in range(self.args.n_agents)]))

            #     all_roles.append(copy.deepcopy(p_roles))

            #     if t_episode < 2:
            #         figure = plt.figure()

            #         print(self.t, p_roles)
            #         ally_health = ally_info['health']
            #         ally_health_max = ally_info['health_max']
            #         if 'shield' in ally_info.keys():
            #             ally_health += ally_info['shield']
            #             ally_health_max += ally_info['shield_max']
            #         ally_health_status = ally_health / ally_health_max
            #         plt.scatter(ally_info['x'], ally_info['y'], s=20*ally_health_status, c=role_color[roles_detach])
            #         for agent_i in range(self.args.n_agents):
            #             plt.text(ally_info['x'][agent_i], ally_info['y'][agent_i], '{:d}'.format(agent_i+1), c='y')

            #         enemy_info = replay_data[-1][1]
            #         enemy_health = enemy_info['health']
            #         enemy_health_max = enemy_info['health_max']
            #         if 'shield' in enemy_info.keys():
            #             enemy_health += enemy_info['shield']
            #             enemy_health_max += enemy_info['shield_max']
            #         enemy_health_status = enemy_health / enemy_health_max
            #         plt.scatter(enemy_info['x'], enemy_info['y'], s=20*enemy_health_status, c='k')
            #         for enemy_i in range(len(enemy_info['x'])):
            #             plt.text(enemy_info['x'][enemy_i], enemy_info['y'][enemy_i], '{:d}'.format(enemy_i+1))

            #         plt.xlim(0, 32)
            #         plt.ylim(0, 32)
            #         plt.title('t={:d}'.format(self.t))
            #         pic_name = os.path.join(save_path, str(self.t) + '.png')
            #         plt.savefig(pic_name)
            #         plt.close()

            #reward, terminated, env_info = self.env.step(actions[0])
            ### NOTE: changed in envrionment.py self.discrete_action_input = True #False
            # if true, even the action is continuous, action will be performed discretely
            #print("actions", actions[0])
            # convert [4,2,0,1] into a list of one-hot vector
            new_actions = th.nn.functional.one_hot(actions[0], num_classes=5)
            # actions_int = [int(a) for a in actions[0]]
            # ac = np.eye(5)[np.array(actions_int)]
            # print(ac)


            obs_, reward, done, env_info = self.env.step(new_actions[0])
            terminated = any(done)
            
            #print(episode_return) ### change the problem here
            episode_return += sum(reward)
            #print(sum(reward))
            if episode_step >= self.episode_limit:
                terminated = True

            post_transition_data = {
                "actions": actions,
                "roles": roles,
                "role_avail_actions": role_avail_actions,
                "reward": [(sum(reward),)],
                "terminated": [(episode_step >= self.episode_limit,)],
            }
            

            self.batch.update(post_transition_data, ts=self.t)

            episode_step += 1
            self.t += 1
            state = np.concatenate(obs_, axis=0).astype(
                np.float32
            )
            obs = obs_
            # zero pad the last obs
            obs[-1] = np.pad(obs[-1], (0, 2), 'constant', constant_values=(0))
            score += sum(reward)
         
        last_data = {
            "state": [state],
            "avail_actions": [[1, 1, 1, 1, 1] for i in range(n_agents)],
            "obs": [obs]
        }
        self.batch.update(last_data, ts=self.t)
        

        # if self.verbose:
        #     # These outputs are designed for SMAC
        #     ally_info, enemy_info = self.env.get_structured_state()
        #     replay_data.append([ally_info, enemy_info])

        # Select actions in the last stored state
        
        actions, roles, role_avail_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions, "roles": roles, "role_avail_actions": role_avail_actions}, ts=self.t)

       

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
       
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        # cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        #print(cur_stats, cur_returns)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        i = cur_stats["n_episodes"]
        # print('episode', i)
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))


        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if self.verbose:
            return self.batch, np.array(all_roles)
        
      
        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

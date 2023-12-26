#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import copy
import torch
import random 
import time 

from itertools import chain
import imageio
import seaborn as sns
import matplotlib.pyplot as plt

from onpolicy.config import get_config

from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from itertools import product
from scipy.stats import sem  # sem is used to calculate standard error
import pandas as pd

def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


class ChaseAgent:
    def __init__(self, all_args, agent_id, agent_trainer):
        self.agent_id = agent_id
        self.agent_trainer = agent_trainer

        self.rnn_state = np.zeros((1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        self.mask = np.ones((1, 1), dtype=np.float32)
        self.action_dim = all_args.action_dim

        self.num_agents = all_args.num_agents
        self.num_prey = all_args.num_good_agents
        self.num_pred = all_args.num_adversaries
        self.visible_predator_id = list(range(self.num_pred))
        self.visible_prey_id = list(range(self.num_pred, self.num_pred + self.num_prey))

    def observe(self, world):
        agent = world.agents[self.agent_id]
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        other_pos = []
        other_vel = []
        visible_agents_id = self.visible_predator_id + self.visible_prey_id
        for other_agent_id in visible_agents_id:
            if other_agent_id == self.agent_id: continue
            other = world.agents[other_agent_id]
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        ag_health = [world.agents[preyid].health for preyid in self.visible_prey_id]
        agent_obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + [ag_health])
        agent_obs = agent_obs.reshape(1, -1)
        return agent_obs

    def act(self, agent_obs, save_rnn = False):
        self.agent_trainer.prep_rollout()
        action, rnn_state = self.agent_trainer.policy.act(np.array(list(agent_obs)), self.rnn_state, self.mask, deterministic=True)
        action = action.detach().cpu().numpy()
        agent_action = np.squeeze(np.eye(self.action_dim)[action], 1)[0]
        if save_rnn:
            self.rnn_state = rnn_state.detach().cpu().numpy()

        return agent_action


class AllAgents:
    def __init__(self, all_args, trainer):
        self.agents = []
        self.num_agents = all_args.num_agents
        for agent_id in range(self.num_agents):
            agent_trainer = trainer[agent_id]
            agent = ChaseAgent(all_args, agent_id, agent_trainer)
            self.agents.append(agent)

    def act(self, world):
        all_actions = []
        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            agent_obs = agent.observe(world)
            agent_action = agent.act(agent_obs, save_rnn = True)
            all_actions.append(agent_action)
        return all_actions


def main():
    num_pred_prey_list = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 4)]
    speed_list = [1, 1.15, 1.3]
    combinations = list(product(num_pred_prey_list, speed_list))

    index = pd.MultiIndex.from_tuples([(*num_pred_prey, speed) for num_pred_prey, speed in combinations],
                                      names=["num_pred", "num_prey", "speed"])
    df = pd.DataFrame(index=index)

    for (num_pred, num_prey), speed in combinations:
        parser = get_config()
        all_args = parser.parse_args()

        all_args.eval = True

        all_args.env_name="MPE"
        all_args.scenario_name="iw_env"
        all_args.num_good_agents= num_prey
        all_args.num_adversaries= num_pred
        all_args.num_landmarks=2
        all_args.algorithm_name="rmappo" #"mappo" "ippo"
        all_args.prey_speed = speed
        all_args.experiment_name=f"preyspeed{all_args.prey_speed}"
        all_args.seed_max=1
        all_args.seed = 0

        all_args.cuda = False
        all_args.share_policy = False
        all_args.n_training_threads = 1
        all_args.n_rollout_threads = 1
        all_args.num_mini_batch = 1
        all_args.episode_length = 75
        all_args.num_env_steps = 20000000
        all_args.ppo_epoch = 10
        all_args.use_ReLU = False
        all_args.gain = 0.01
        all_args.lr = 7e-4
        all_args.critic_lr = 7e-4
        all_args.save_gifs = True
        all_args.use_render = True
        all_args.render_episodes = 50
        all_args.user_name = "minglu-zhao"
        all_args.num_agents= all_args.num_good_agents + all_args.num_adversaries
        all_args.discrete_action = True

        exp_name = "IW"
        foldername = f'{all_args.num_adversaries}pred_{all_args.num_good_agents}prey_preyspeed{all_args.prey_speed}'

        all_args.model_dir = f"../results/MPE/{all_args.scenario_name}/rmappo/{foldername}/wandb/latest-run/files"
        all_args.render_verbose = False

        if all_args.algorithm_name == "rmappo":
            print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
            all_args.use_recurrent_policy = True
            all_args.use_naive_recurrent_policy = False
        elif all_args.algorithm_name == "mappo":
            print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
            all_args.use_recurrent_policy = False
            all_args.use_naive_recurrent_policy = False
        elif all_args.algorithm_name == "ippo":
            print("u are choosing to use ippo, we set use_centralized_V to be False.")
            all_args.use_centralized_V = False
        else:
            raise NotImplementedError

        assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
            "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

        assert all_args.use_render, ("u need to set use_render be True")
        assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
        assert all_args.n_rollout_threads==1, ("only support to use 1 env to render.")


        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

        # run dir
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / exp_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
            str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

        # seed
        torch.manual_seed(all_args.seed)
        torch.cuda.manual_seed_all(all_args.seed)
        np.random.seed(all_args.seed)

        # env init
        envs = make_render_env(all_args)
        eval_envs = None
        num_agents = all_args.num_agents
        all_args.action_dim = envs.action_space[0].n

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": run_dir
        }

        # run experiments
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

        runner = Runner(config)
        multiagent = AllAgents(all_args, runner.trainer)


        num_bites_list = []
        for episode in range(all_args.render_episodes):
            episode_rewards = []
            obs = envs.reset()
            num_bites = 0

            for step in range(all_args.episode_length):
                world = envs.envs[0].world
                all_actions = multiagent.act(world)
                obs, rewards, dones, infos = envs.step([all_actions])
                episode_rewards.append(rewards)
                num_bites += int(rewards[:, 0]) # in eval, wolf reward = num bites

            episode_rewards = np.array(episode_rewards)
            num_bites_list.append(num_bites)
            for agent_id in range(num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
            print(f"total number of bites: {num_bites}")


        df.loc[(num_pred, num_prey, speed), 'mean_bites'] = np.mean(num_bites_list)
        df.loc[(num_pred, num_prey, speed), 'se_bites'] = sem(num_bites_list)  # Standard error
        print(df)
        print('mean_bites ', df['mean_bites'].tolist())
        print('se_bites ', df['se_bites'].tolist())

        # post process
        envs.close()


    print(df)
    print('mean_bites ', df['mean_bites'].tolist())
    print('se_bites ', df['se_bites'].tolist())

    df['mean_bites'] = [1.32, 1.54, 0.22, 8.48, 6.32, 2.52, 16.48, 9.36, 5.58, 20.82, 11.76, 4.24, 16.48, 8.98, 4.86]
    df['se_bites'] = [0.26869298161162847, 0.33589356914939345, 0.07165677830848258, 0.6441193957011133, 0.4707267503227194, 0.3106806221501299,
                      0.9515787097325825, 0.6932679194678384, 0.41707435836810114, 1.5334874220121233, 0.7111703490065671, 0.5311212629130633, 0.9072227186443744, 0.651835712368053, 0.4659946570596741]

    df_reset = df.reset_index()
    # Create a new column that combines num_pred and num_prey for the x-axis
    df_reset['num_pred_prey'] = df_reset[['num_pred', 'num_prey']].apply(lambda x: f"({x[0]}, {x[1]})", axis=1)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_reset, x='num_pred_prey', y='mean_bites', hue='speed') #, legend='brief', label = 1)
    plt.show()

    # Adding shaded areas for standard error
    for speed in speed_list:
        speed_df = df_reset[df_reset['speed'] == speed]
        plt.fill_between(speed_df['num_pred_prey'], speed_df['mean_bites'] - speed_df['se_bites'], speed_df['mean_bites'] + speed_df['se_bites'], alpha=0.2)

    plt.title('Number of Bites by Number of Predators and Preys with SE')
    plt.xlabel('(Number of Predators, Number of Preys)')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    filename = 'Eval_rMAPPO_NumBites.png'

    # Save the figure
    plt.savefig(filename)

    plt.show()




if __name__ == "__main__":
    main()

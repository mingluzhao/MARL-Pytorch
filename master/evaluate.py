import argparse
import torch
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..', '..', '..'))

import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG


def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    epsRewardTot = []
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        # if config.save_gifs:
        #     frames = []
        #     frames.append(env.render('rgb_array')[0])
        # env.render('human')

        epsReward = np.zeros(4)
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            epsReward += rewards

            # if config.save_gifs:
            #     frames.append(env.render('rgb_array')[0])
            # calc_end = time.time()
            # elapsed = calc_end - calc_start
            # if elapsed < ifi:
            #     time.sleep(ifi - elapsed)
            # env.render('human')
        # if config.save_gifs:
        #     gif_num = 0
        #     while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
        #         gif_num += 1
        #     imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
        #                     frames, duration=ifi)
        epsRewardTot.append(epsReward)

    print("Mean Eps Reward: ", np.mean(epsRewardTot, axis=0))
    print("SE Eps Reward: ", np.std(epsRewardTot, axis=0)/np.sqrt(len(epsRewardTot)))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default= "simple_tag", type = str, help="Name of environment")
    parser.add_argument("--model_name", default= "hunt_gym", type = str, help="Name of directory to store " + "model/training contents")
    parser.add_argument("--run_num", default=2, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " + "rather than final policy")
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--episode_length", default=75, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    run(config)
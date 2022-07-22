import argparse
import torch
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from maddpg.src.utils.make_env import make_env
from maddpg.src.utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from maddpg.src.maddpg import MADDPG
from maddpg.src.utils.buffer import ReplayBuffer

USE_CUDA = False  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed, config.discrete_action)

    obsShape = [obsp.shape[0] for obsp in env.observation_space]
    actionDimList = [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space]
    hiddenDimList = [config.hidden_dim]* config.hidden_layer_num

    numAgents = len(env.agent_types)

    algorithmTypes = ['MADDPG']* numAgents
    isDiscreteAction = True
    policyInputDimList = obsShape
    policyOutputDimList = actionDimList
    criticInputDimList = [np.sum(obsShape) + np.sum(policyOutputDimList) for _ in range(numAgents)] # all agents actions and observations

    maddpg = MADDPG.init_from_env(algorithmTypes, isDiscreteAction, policyInputDimList, policyOutputDimList, criticInputDimList,
                                  tau=config.tau, lr=config.lr, hiddenDimList= hiddenDimList)

    buffer = ReplayBuffer(config.bufferSize, numAgents, obsShape, actionDimList)

    t = 0
    for ep_i in range(0, config.maxEpisode, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1, ep_i + 1 + config.n_rollout_threads, config.maxEpisode))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, numAgents)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scaleNoise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.resetNoise()

        for et_i in range(config.maxTimeStep):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(maddpg.numAgents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.act(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(buffer) >= config.minibatchSize and
                (t % config.learnInterval) < config.n_rollout_threads):
                maddpg.prep_training(device='gpu') if USE_CUDA else maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.numAgents):
                        sample = buffer.sample(config.minibatchSize, to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = buffer.get_average_rewards(config.maxTimeStep * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_predators", default=3, type=int, help="num_predators")
    parser.add_argument("--speed", default=1, type=float, help="speed")
    parser.add_argument("--cost", default=0, type=float, help="cost")
    parser.add_argument("--selfish", default=1, type=float, help="selfish")

    parser.add_argument("--env_id", default= "simple_tag", type = str, help="Name of environment")
    parser.add_argument("--model_name", default= "hunt_gym", type = str, help="Name of directory to store " + "model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--bufferSize", default=int(1e6), type=int)
    parser.add_argument("--maxTimeStep", default=75, type=int)
    parser.add_argument("--maxEpisode", default=60000, type=int)
    parser.add_argument("--learnInterval", default=100, type=int)
    parser.add_argument("--minibatchSize", default=1024, type=int, help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=10000, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action", action='store_true')
    
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--hidden_layer_num", default=64, type=int)
    config = parser.parse_args()

    run(config)
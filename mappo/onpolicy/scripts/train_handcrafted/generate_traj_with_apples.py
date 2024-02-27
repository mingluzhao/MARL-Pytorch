import sys
import os
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
import json
import pickle

from onpolicy.envs.hunting_environment.chasingEnv.multiAgentEnv import TransitMultiAgentChasing, ApplyActionForce, ApplyEnvironForce, \
    ResetMultiAgentChasingWithCaughtHistoryWithApples, ResetStateWithCaughtHistory, ReshapeAction, \
    CalSheepCaughtHistory, RewardSheepWithBiteAndKill, RewardWolfWithBiteKillAndApples, ObserveWithCaughtHistoryWithApples, \
    GetCollisionForce, IntegrateStateWithCaughtHistory, IsCollision, PunishForOutOfBound, \
    getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState
from onpolicy.envs.hunting_environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividualWithBiteKillAndApples
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict


def saveToPickle(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()

def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object


class AllAgents:
    def __init__(self, all_args, trainer):
        self.all_agent_trainer = trainer
        self.num_agents = all_args.num_agents
        self.rnn_state = [np.zeros((1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32) for _ in
                          range(self.num_agents)]
        self.mask = [np.ones((1, 1), dtype=np.float32) for _ in range(self.num_agents)]
        self.discrete_action = all_args.discrete_action
        self.action_dim = all_args.action_dim

    def act(self, all_obs, save_rnn=True):
        all_actions = []
        for agent_id in range(self.num_agents):
            agent_obs = all_obs[agent_id]
            agent_obs = agent_obs.reshape(1, -1)
            agent_trainer = self.all_agent_trainer[agent_id]
            agent_trainer.prep_rollout()

            action, rnn_state = agent_trainer.policy.act(agent_obs,
                                                         self.rnn_state[agent_id], self.mask[agent_id],
                                                         deterministic=True)
            action = action.detach().cpu().numpy()
            if self.discrete_action:
                agent_action = np.squeeze(np.eye(self.action_dim)[action], 1)[0]
            else:
                agent_action = action[0]
            if save_rnn:
                self.rnn_state[agent_id] = rnn_state.detach().cpu().numpy()

            all_actions.append(agent_action)

        return all_actions


def generateTrajWolfSheepIterativeTrain(df, save_dir):
    numSheep = df.index.get_level_values('numSheep')[0]
    trainSheepSpeed = df.index.get_level_values('trainSheepSpeed')[0]
    evalSheepSpeed = df.index.get_level_values('evalSheepSpeed')[0]
    individualRewardWolf = df.index.get_level_values('individualRewardWolf')[0]
    seed = df.index.get_level_values('seed')[0]

    numWolves = 3
    numBlocks = 0   
    numApples = 3
    maxTimeStep = 40#400
    numEpisodes = 10
    killZoneRatio = 1.2
    discrete_action = False

    # --------------- environment information ---------------
    numAgents = numWolves + numSheep
    numEntities = numAgents + numBlocks + numApples
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numAgents + numBlocks))
    applesID = list(range(numAgents + numBlocks, numAgents + numBlocks + numApples))

    wolfSize = 0.065
    sheepSize = 0.065
    blockSize = 0.2
    appleSize = 0.065
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheep + [blockSize] * numBlocks + [appleSize] * numApples

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None
    appleMaxSpeed = None
    sheepMaxSpeedOriginal = 1.0
    sheepMaxSpeed = sheepMaxSpeedOriginal * evalSheepSpeed

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheep + [blockMaxSpeed] * numBlocks + [appleMaxSpeed] * numApples
    entitiesMovableList = [True] * numAgents + [False] * numBlocks + [False] * numApples
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState, killZoneRatio)
    punishForOutOfBound = PunishForOutOfBound()
    sheepLife = 6
    biteReward = 0.1
    killReward = 1
    appleReward = 0.02
    rewardSheep = RewardSheepWithBiteAndKill(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                                             punishForOutOfBound, getCaughtHistoryFromAgentState, sheepLife, biteReward,
                                             killReward)

    if individualRewardWolf:
        rewardWolf = RewardWolfIndividualWithBiteKillAndApples(wolvesID, sheepsID, applesID, entitiesSizeList, isCollision, 
                                                               getCaughtHistoryFromAgentState, sheepLife, biteReward, killReward, appleReward)
        
    else:
        rewardWolf = RewardWolfWithBiteKillAndApples(wolvesID, sheepsID, applesID, entitiesSizeList, isCollision, 
                                                     getCaughtHistoryFromAgentState, sheepLife, biteReward, killReward, appleReward)

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    observeOneAgent = lambda agentID: ObserveWithCaughtHistoryWithApples(agentID, wolvesID, sheepsID, blocksID, applesID,
                                                               getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    calSheepCaughtHistory = CalSheepCaughtHistory(wolvesID, sheepsID, entitiesSizeList, isCollision)
    integrateState = IntegrateStateWithCaughtHistory(numEntities, entitiesMovableList, massList, entityMaxSpeedList,
                                                     getVelFromAgentState, getPosFromAgentState, calSheepCaughtHistory)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    resetState = ResetMultiAgentChasingWithCaughtHistoryWithApples(numAgents, numBlocks, numApples)
    reset = ResetStateWithCaughtHistory(resetState, calSheepCaughtHistory)
    # reset = ResetMultiAgentChasing(numAgents, numBlocks)

    isTerminal = lambda state: [False] * numAgents
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    actionDim = worldDim * 2 + 1


    # ------------ models ------------------------

    parser = get_config()
    all_args = parser.parse_args()
    all_args.env_name="MPE"
    all_args.scenario_name="iw22_add_apples"
    all_args.discrete_action = discrete_action
    all_args.algorithm_name="rmappo" #"mappo" "ippo"
    # all_args.seed_max=1
    # all_args.seed = 0

    # ------------ training parameters -----------
    all_args.cuda = False 
    all_args.share_policy = False
    all_args.n_training_threads = 1
    all_args.n_rollout_threads = 1
    all_args.num_mini_batch = 1
    all_args.num_env_steps = 20000000
    all_args.ppo_epoch = 10
    all_args.use_ReLU = False
    all_args.gain = 0.01
    all_args.lr = 7e-4
    all_args.critic_lr = 7e-4
    all_args.user_name = "minglu-zhao"
    all_args.num_agents= numAgents
    all_args.eval = True
    all_args.use_wandb = False


    import glob
    foldername = f'{numWolves}pred_{numSheep}prey-{numBlocks}block-sheepspeed{trainSheepSpeed}-indivd{individualRewardWolf}-discrete{all_args.discrete_action}-killzone{killZoneRatio}-seed{seed}'
    exp_name = f'../results/MPE/iw22_add_apples/rmappo/{foldername}'

    # pattern = f'../results/MPE/{all_args.scenario_name}/rmappo/{numWolves}pred_{numSheep}prey-{numBlocks}block-sheepspeed{trainSheepSpeed}-indivd{individualRewardWolf}-discrete{all_args.discrete_action}-seed*'
    # exp_name = glob.glob(pattern)[0]
    #
    # start_index = exp_name.find(f"{numWolves}pred")
    # foldername = exp_name[start_index:]

    all_args.model_dir = f"{exp_name}/wandb/latest-run/files"
    all_args.render_verbose = False

    # print(all_args)
    if all_args.algorithm_name == "rmappo":
        # print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / foldername
    # setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
    #     str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    all_args.action_dim = actionDim
    all_share_observation_space = [sum(obsShape)]* numAgents
    all_observation_space = obsShape
    all_action_space = [actionDim]* numAgents
    config = {
        "all_args": all_args,
        "device": device,
        "run_dir": run_dir,
        "all_share_observation_space": all_share_observation_space,
        "all_observation_space": all_observation_space,
        "all_action_space": all_action_space,
        "discrete_action": all_args.discrete_action
    }

    from onpolicy.runner.separated.mpe_runner_hunting import MPERunner as Runner
    from onpolicy.envs.hunting_environment.chasingEnv.parallel_env import EvalVecEnv



    env = EvalVecEnv(reset, observe, transit, rewardFunc, isTerminal)

    runner = Runner(config, env)
    allagents = AllAgents(all_args, runner.trainer)

    rewards_list = []
    traj_list = []
    for episode in range(numEpisodes):
        episode_reward_list = []
        state = env.reset()
        traj = []

        for step in range(maxTimeStep):
            obs = env.observe(state)[0]
            actions = allagents.act(obs)
            nextState, nextObs, rewards, dones, infos = env.step(state, [actions])

            if np.any(np.array(rewards) == killReward):
                print(f"killed --- step {step} reward {rewards}")

            if individualRewardWolf:
                group_reward = np.sum(rewards[0][:numWolves])
            else:
                group_reward = rewards[0][0]

            traj.append((state, actions, group_reward, nextState))

            episode_reward_list.append(group_reward)
            state = nextState

        rewards_list.append(np.sum(episode_reward_list))
        traj_list.append(traj)

    meanTrajReward = np.mean(rewards_list)
    seTrajReward = np.std(rewards_list) / np.sqrt(len(rewards_list) - 1)

    print(exp_name)
    print(f'mean: {meanTrajReward}, se: {seTrajReward}')

    traj_loc = os.path.join(save_dir, f'{foldername}-evalspeed{evalSheepSpeed}.pkl')
    saveToPickle(traj_list, traj_loc)

    return pd.Series({'mean': meanTrajReward, 'se': seTrajReward})




def main():

    independentVariables = OrderedDict()
    independentVariables['evalSheepSpeed'] = [1.1]
    independentVariables['trainSheepSpeed'] = [1.1] # [0.7, 1.1, 1.5]
    independentVariables['individualRewardWolf'] = [0, 1]
    independentVariables['numSheep'] = [2]
    independentVariables['seed'] = [101, 1001, 10001, 100001, 1000001]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)

    save_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results/trajectories/apples/continuous")
    if not save_dir.exists():
        os.makedirs(str(save_dir))
    generate = lambda df: generateTrajWolfSheepIterativeTrain(df, save_dir)
    resultDF = toSplitFrame.groupby(levelNames).apply(generate)
    print(resultDF)


    dirName = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results/eval_result")
    if not dirName.exists():
        os.makedirs(str(dirName))


    resultLoc = os.path.join(dirName, 'continuous__apples_generate_traj_result.pkl')
    saveToPickle(resultDF, resultLoc)


if __name__ == '__main__':
    main()

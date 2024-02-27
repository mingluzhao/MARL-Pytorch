#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from datetime import datetime
from onpolicy.config import get_config
import numpy as np
import json

from onpolicy.envs.hunting_environment.chasingEnv.multiAgentEnv import TransitMultiAgentChasing, ApplyActionForce, ApplyEnvironForce, \
    ResetMultiAgentChasing, ResetMultiAgentChasingWithCaughtHistory, ResetStateWithCaughtHistory, ReshapeAction, \
    CalSheepCaughtHistory, RewardSheep, RewardSheepWithBiteAndKill, RewardWolf, RewardWolfWithBiteAndKill, ObserveWithCaughtHistory, \
    GetCollisionForce, IntegrateState, IntegrateStateWithCaughtHistory, IsCollision, PunishForOutOfBound, \
    getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState
from onpolicy.envs.hunting_environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividualWithBiteAndKill


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
                agent_action = action

            if save_rnn:
                self.rnn_state[agent_id] = rnn_state.detach().cpu().numpy()

            all_actions.append(agent_action)

        return all_actions



def main():
    numWolves = 3
    numSheeps = 1
    numBlocks = 1
    maxTimeStep = 75
    sheepSpeedMultiplier = 1
    individualRewardWolf = int(False)

    debug = False # False

    # --------------- environment information ---------------
    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    wolfSize = 0.065
    sheepSize = 0.065
    blockSize = 0.2
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None
    sheepMaxSpeedOriginal = 1.0
    sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    killZoneRatio = 1.2
    isCollision = IsCollision(getPosFromAgentState, killZoneRatio)
    punishForOutOfBound = PunishForOutOfBound()
    sheepLife = 6
    biteReward = 1
    killReward = 10
    rewardSheep = RewardSheepWithBiteAndKill(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                                             punishForOutOfBound, getCaughtHistoryFromAgentState, sheepLife, biteReward,
                                             killReward)

    if individualRewardWolf:
        rewardWolf = RewardWolfIndividualWithBiteAndKill(wolvesID, sheepsID, entitiesSizeList, isCollision,
                                                         getCaughtHistoryFromAgentState, sheepLife, biteReward,
                                                         killReward)
    else:
        rewardWolf = RewardWolfWithBiteAndKill(wolvesID, sheepsID, entitiesSizeList, isCollision,
                                               getCaughtHistoryFromAgentState, sheepLife, biteReward, killReward)

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    observeOneAgent = lambda agentID: ObserveWithCaughtHistory(agentID, wolvesID, sheepsID, blocksID,
                                                               getPosFromAgentState,
                                                               getVelFromAgentState, getCaughtHistoryFromAgentState)
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

    resetState = ResetMultiAgentChasingWithCaughtHistory(numAgents, numBlocks)
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
    all_args.scenario_name="iw22_env"
    all_args.discrete_action = True
    all_args.algorithm_name="rmappo" #"mappo" "ippo"
    all_args.seed_max=1
    all_args.seed = 0

    # ------------ training parameters -----------
    all_args.cuda = False 
    all_args.share_policy = False
    all_args.n_training_threads = 1 
    all_args.n_rollout_threads = 128
    all_args.num_mini_batch = 1 
    all_args.episode_length = maxTimeStep
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

    foldername = f'{numWolves}pred_{numSheeps}prey-{numBlocks}block'

    all_args.model_dir = f"../results/MPE/{all_args.scenario_name}/rmappo/{foldername}/wandb/latest-run/files"
    all_args.render_verbose = False

    print(all_args)
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
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

    # run dir
    exp_name = f'{numWolves}pred_{numSheeps}prey'
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / exp_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

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

    num_bites_list = []
    for episode in range(all_args.render_episodes):
        episode_rewards = []
        state = env.reset()
        num_bites = 0

        for step in range(all_args.episode_length):
            obs = env.observe(state)[0]
            actions = allagents.act(obs)
            nextState, nextObs, rewards, dones, infos = env.step(state, [actions])

            episode_rewards.append(rewards)
            num_bites += int(rewards[:, 0])  # in eval, wolf reward = num bites
            state = nextState

        episode_rewards = np.array(episode_rewards)
        num_bites_list.append(num_bites)
        for agent_id in range(numAgents):
            average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
            print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        print(f"total number of bites: {num_bites}")



if __name__ == "__main__":
    main()

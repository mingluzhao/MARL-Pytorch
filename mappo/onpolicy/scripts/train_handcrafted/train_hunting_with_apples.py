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
    ResetMultiAgentChasingWithCaughtHistoryWithApples, ResetStateWithCaughtHistory, ReshapeAction, \
    CalSheepCaughtHistory, RewardSheepWithBiteAndKill, RewardWolfWithBiteKillAndApples, ObserveWithCaughtHistoryWithApples, \
    GetCollisionForce, IntegrateStateWithCaughtHistory, IsCollision, PunishForOutOfBound, \
    getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState
from onpolicy.envs.hunting_environment.chasingEnv.multiAgentEnvWithIndividReward import RewardWolfIndividualWithBiteKillAndApples

# 0.7 1.1 1.5
# share individual 
# 75 
# 3 vs 124. 0 block 
import base64

def main():
    debug = False 
    if debug:
        numSheeps = 4
        sheepSpeedMultiplier = 1.1 # 0.7, 1.1, 1.5
        individualRewardWolf = 0
        discrete_action = False
        seed = 42
        killZoneRatio = 1.2
    else:
        # print(sys.argv)
        encoded_json = sys.argv[1]
        decoded_json = base64.b64decode(encoded_json).decode('utf-8')
        condition = json.loads(decoded_json)
        print("condition")
        print(condition)
        sys.argv = [sys.argv[0]] + sys.argv[2:]

        numSheeps = int(condition['numSheeps'])
        sheepSpeedMultiplier = float(condition['sheepSpeedMultiplier'])
        individualRewardWolf = int(condition['individualRewardWolf'])
        discrete_action = bool(condition['discrete_action'])
        seed = int(condition['seed'])
        killZoneRatio = float(condition['killZoneRatio'])

    numWolves = 3
    numBlocks = 0   
    numApples = 3
    maxTimeStep = 75

    # --------------- environment information ---------------
    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks + numApples
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numAgents + numBlocks))
    applesID = list(range(numAgents + numBlocks, numAgents + numBlocks + numApples))

    wolfSize = 0.065
    sheepSize = 0.065
    blockSize = 0.2
    appleSize = 0.065
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks + [appleSize] * numApples

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None
    appleMaxSpeed = None
    sheepMaxSpeedOriginal = 1.0
    sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks + [appleMaxSpeed] * numApples
    entitiesMovableList = [True] * numAgents + [False] * numBlocks + [False] * numApples
    massList = [1.0] * numEntities

    isCollision = IsCollision(getPosFromAgentState, killZoneRatio)
    punishForOutOfBound = PunishForOutOfBound()
    sheepLife = 6
    biteReward = 1
    killReward = 10
    appleReward = 0.2
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

# conda activate marl
# python train_handcrafted_env.py
    
    # ------------ models ------------------------
    parser = get_config()
    all_args = parser.parse_args()
    all_args.env_name="MPE"
    all_args.scenario_name="iw22_add_apples"
    all_args.discrete_action = discrete_action
    all_args.algorithm_name="rmappo" #"mappo" "ippo"
    all_args.seed_max= 25
    all_args.seed = seed

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
    all_args.eval = False
    if debug:
        all_args.use_wandb = False

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
    exp_name = f'{numWolves}pred_{numSheeps}prey-{numBlocks}block-sheepspeed{sheepSpeedMultiplier}-indivd{individualRewardWolf}-discrete{all_args.discrete_action}-killzone{killZoneRatio}-seed{all_args.seed}'


    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / exp_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    contin_str = "discrete" if all_args.discrete_action else "continuous"
    if not debug:
        current_date = datetime.today().strftime('%m-%d')
        run = wandb.init(config=all_args,
                            project=all_args.env_name,
                            entity=all_args.user_name,
                            notes=socket.gethostname(),
                            name= f"{current_date}-{exp_name}-{all_args.algorithm_name}-{all_args.episode_length}steps",
                            group=f"{all_args.scenario_name}_rmappo",
                            dir=str(run_dir),
                            job_type= f"sheepspeed{sheepSpeedMultiplier}-indivd{individualRewardWolf}-killzone{killZoneRatio}-{contin_str}" ,
                            reinit=True)

    setproctitle.setproctitle(exp_name)
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
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
    from onpolicy.envs.hunting_environment.chasingEnv.parallel_env import ParallelEnv

    num_envs = all_args.n_rollout_threads
    parallel_env = ParallelEnv(num_envs, reset, observe, transit, rewardFunc, isTerminal)

    runner = Runner(config, parallel_env)
    runner.run()
    
    # post process
    parallel_env.close()

    if all_args.use_wandb and not debug:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main()

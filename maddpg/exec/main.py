import argparse
import torch
import os
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from maddpg.src.utils.buffer import ReplayBuffer
from maddpg.src.maddpg import MADDPG
from environment.multiAgentEnv import TransitMultiAgentChasing, ApplyActionForce, ApplyEnvironForce, \
    ResetMultiAgentChasing, ReshapeAction, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState, GetActionCost
from environment.reward import *

USE_CUDA = False  # torch.cuda.is_available()

def run(config):
    model_dir = Path('./models')#TODO / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if str(folder.name).startswith('run')]
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

    numPredators = config.num_predators
    preySpeedMultiplier = config.speed
    costActionRatio = config.cost
    selfishIndex = config.selfish

    numPrey = 1
    numBlocks = 2
    saveAllmodels = 0 # save all models during training
    maxTimeStep = 75
    killReward = 10
    killProportion = 0.2
    biteReward = 0.0
    maxEpisode = 60000

    print("train: {} predators, {} prey, {} blocks, {} episodes with {} steps each eps, preySpeed: {}x, cost: {}, selfish: {}".
          format(numPredators, numPrey, numBlocks, maxEpisode, maxTimeStep, preySpeedMultiplier, costActionRatio, selfishIndex))

    numAgents = numPredators + numPrey
    numEntities = numAgents + numBlocks
    predatorsID = list(range(numPredators))
    preyGroupID = list(range(numPredators, numAgents))
    blocksID = list(range(numAgents, numEntities))

    predatorSize = 0.075
    preySize = 0.05
    blockSize = 0.2
    entitiesSizeList = [predatorSize] * numPredators + [preySize] * numPrey + [blockSize] * numBlocks

    predatorMaxSpeed = 1.0
    blockMaxSpeed = None
    preyMaxSpeedOriginal = 1.3
    preyMaxSpeed = preyMaxSpeedOriginal * preySpeedMultiplier

    entityMaxSpeedList = [predatorMaxSpeed] * numPredators + [preyMaxSpeed] * numPrey + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    collisionReward = 10 # originalPaper = 10*3
    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardPrey = RewardPrey(predatorsID, preyGroupID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound, collisionPunishment = collisionReward)

    collisionDist = predatorSize + preySize
    getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(selfishIndex, collisionDist)
    terminalCheck = TerminalCheck()
    getCollisionPredatorReward = GetCollisionPredatorReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck)
    getPredatorPreyDistance = GetPredatorPreyDistance(computeVectorNorm, getPosFromAgentState)
    rewardPredator = RewardPredatorsWithKillProb(predatorsID, preyGroupID, entitiesSizeList, isCollision, terminalCheck, getPredatorPreyDistance,
                 getAgentsPercentageOfRewards, getCollisionPredatorReward)

    reshapeAction = ReshapeAction()
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    getPredatorsAction = lambda action: [action[predatorID] for predatorID in predatorsID]
    rewardPredatorWithActionCost = lambda state, action, nextState: np.array(rewardPredator(state, action, nextState)) - np.array(getActionCost(getPredatorsAction(action)))

    rewardFunc = lambda state, action, nextState: \
        list(rewardPredatorWithActionCost(state, action, nextState)) + list(rewardPrey(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, predatorsID, preyGroupID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(predatorsID, preyGroupID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: terminalCheck.terminal
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    actionDimList = [worldDim * 2 + 1 for _ in range(numAgents)]
    hiddenDimList = [128, 128]

    algorithmTypes = ['MADDPG']* numAgents
    isDiscreteAction = True
    policyInputDimList = obsShape
    policyOutputDimList = actionDimList
    criticInputDimList = [np.sum(obsShape) + np.sum(policyOutputDimList) for _ in range(numAgents)] # all agents actions and observations

    maddpg = MADDPG.init_from_env(algorithmTypes, isDiscreteAction, policyInputDimList, policyOutputDimList, criticInputDimList,
                                  tau=config.tau, lr=config.lr, hiddenDimList= hiddenDimList)

    buffer = ReplayBuffer(config.bufferSize, numAgents, obsShape, actionDimList)
    totalRunTime = 0
    for epsID in range(maxEpisode):
        state = reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - epsID) / config.n_exploration_eps
        maddpg.scaleNoise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.resetNoise()

        for timeStep in range(config.maxTimeStep):
            obs = observe(state)
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack([obs[i]])), requires_grad=False) for i in range(numAgents)]

            # get actions as torch Variables
            torchAllActions = maddpg.act(torch_obs, explore=True)
            # convert actions to numpy arrays
            allActions = [ac.data.numpy() for ac in torchAllActions]
            # rearrange actions to be per environment
            actions = [ac[0] for ac in allActions]

            nextState = transit(state, actions)
            nextObs = observe(nextState)
            rewards = rewardFunc(state, actions, nextState)
            dones = isTerminal(state)
            buffer.push(obs, allActions, rewards, nextObs, dones)
            state = nextState

            totalRunTime += 1
            if (len(buffer) >= config.minibatchSize and (totalRunTime % config.learnInterval) == 0): # start learn
                maddpg.prep_training(device='gpu') if USE_CUDA else maddpg.prep_training(device='cpu')
                for agentID in range(numAgents):
                    sample = buffer.sample(config.minibatchSize, to_gpu=USE_CUDA)
                    maddpg.update(sample, agentID, logger=logger)
                maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        epsRewards = buffer.get_tot_rewards(config.maxTimeStep)
        
        for agentID, agentEpsReward in enumerate(epsRewards):
            logger.add_scalar('agent%i/tot_episode_rewards' % agentID, agentEpsReward, epsID)

        if epsID % config.save_interval == 0:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (epsID + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("model_name", help="Name of directory to store " + "model/training contents")

    parser.add_argument("--num_predators", default=3, type=int, help="num_predators")
    parser.add_argument("--speed", default=1, type=float, help="speed")
    parser.add_argument("--cost", default=0, type=float, help="cost")
    parser.add_argument("--selfish", default=1, type=float, help="selfish")

    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--bufferSize", default=int(1e6), type=int)
    parser.add_argument("--maxTimeStep", default=75, type=int)
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

    config = parser.parse_args()

    run(config)

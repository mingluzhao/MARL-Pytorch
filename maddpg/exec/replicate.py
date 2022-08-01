import argparse
import torch
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

from pathlib import Path
from torch.autograd import Variable
from maddpg.src.maddpg import MADDPG
from environment.multiAgentEnv import TransitMultiAgentChasing, ApplyActionForce, ApplyEnvironForce, \
    ResetMultiAgentChasing, ReshapeAction, Observe, GetCollisionForce, IntegrateState, \
    IsCollision, PunishForOutOfBound, getPosFromAgentState, getVelFromAgentState, GetActionCost
from environment.reward import *
import pandas as pd
from maddpg.src.utils.loadSaveModel import saveToPickle


class EvaluateDf:
    def __init__(self, evaluate):
        self.evaluate = evaluate

    def __call__(self, df):
        num_predators = df.index.get_level_values('num_predators')[0]
        speed = df.index.get_level_values('speed')[0]
        cost = df.index.get_level_values('cost')[0]
        selfish = df.index.get_level_values('selfish')[0]

        parser = argparse.ArgumentParser()
        parser.add_argument("--num_predators", default=num_predators, type=int, help="num_predators")
        parser.add_argument("--speed", default=speed, type=float, help="speed")
        parser.add_argument("--cost", default=cost, type=float, help="cost")
        parser.add_argument("--selfish", default=selfish, type=float, help="selfish")

        parser.add_argument("--run_num", default=1, type=int)
        parser.add_argument("--incremental", default=None, type=int,
                            help="Load incremental policy from given episode " + "rather than final policy")
        parser.add_argument("--maxTimeStep", default=75, type=int)
        parser.add_argument("--maxEpisodeToSample", default=100, type=int)
        parser.add_argument("--model_name", default="CollectiveHunting", type=str,
                            help="Name of directory to store " + "model/training contents")

        config = parser.parse_args()

        return self.evaluate(config)


def evaluate(config):
    numPredators = config.num_predators
    preySpeedMultiplier = float(config.speed)
    costActionRatio = float(config.cost)
    selfishIndex = float(config.selfish)

    model_name = "model{}predators{}cost{}speed{}selfish".format(numPredators, costActionRatio, preySpeedMultiplier, selfishIndex) + 'run%i' % config.run_num
    model_path = Path('../models') / config.model_name / model_name
    model_path = model_path / 'incremental' / ('model_ep%i.pt' % config.incremental) if config.incremental is not None else model_path / 'model.pt'
    maddpg = MADDPG.init_from_save(model_path)

    ### Hunting environment
    numPrey = 1
    numBlocks = 2
    killReward = 10
    killProportion = 0.2
    biteReward = 0.0

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

    collisionReward = 10  # originalPaper = 10*3
    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardPrey = RewardPrey(predatorsID, preyGroupID, entitiesSizeList, getPosFromAgentState, isCollision,
                            punishForOutOfBound, collisionPunishment=collisionReward)

    collisionDist = predatorSize + preySize
    getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(selfishIndex, collisionDist)
    terminalCheck = TerminalCheck()
    getCollisionPredatorReward = GetCollisionPredatorReward(biteReward, killReward, killProportion,
                                                            sampleFromDistribution, terminalCheck)
    getPredatorPreyDistance = GetPredatorPreyDistance(computeVectorNorm, getPosFromAgentState)
    rewardPredator = RewardPredatorsWithKillProb(predatorsID, preyGroupID, entitiesSizeList, isCollision, terminalCheck,
                                                 getPredatorPreyDistance,
                                                 getAgentsPercentageOfRewards, getCollisionPredatorReward)

    reshapeAction = ReshapeAction()
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    getPredatorsAction = lambda action: [action[predatorID] for predatorID in predatorsID]
    rewardPredatorWithActionCost = lambda state, action, nextState: np.array(rewardPredator(state, action, nextState)) - \
                                                                    np.array(getActionCost(getPredatorsAction(action)))

    rewardFunc = lambda state, action, nextState: \
        list(rewardPredatorWithActionCost(state, action, nextState)) + list(rewardPrey(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, predatorsID, preyGroupID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(predatorsID, preyGroupID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)
    isTerminal = lambda state: terminalCheck.terminal

    maddpg.prep_rollouts(device='cpu')
    epsRewardAgentsTotalList = []

    for epsID in range(config.maxEpisodeToSample):
        state = reset()
        epsRewardAgentsTotal = 0
        for timeStep in range(config.maxTimeStep):
            obs = observe(state)
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack([obs[i]])), requires_grad=False) for i in range(numAgents)]

            torchAllActions = maddpg.act(torch_obs, explore=True)
            allActions = [ac.data.numpy() for ac in torchAllActions]
            actions = [ac[0] for ac in allActions]
            nextState = transit(state, actions)
            rewards = rewardFunc(state, actions, nextState)
            state = nextState
            epsRewardAgentsTotal += np.sum(rewards[:numPredators])

            if isTerminal(nextState):
                break
        epsRewardAgentsTotalList.append(epsRewardAgentsTotal)

    meanTrajReward = np.mean(epsRewardAgentsTotalList, axis=0)
    seTrajReward = np.std(epsRewardAgentsTotalList, axis=0)/np.sqrt(len(epsRewardAgentsTotalList)-1)

    print("Mean Eps Reward: ", meanTrajReward, "SE Eps Reward: ", seTrajReward)
    return pd.Series({'mean': meanTrajReward, 'se': seTrajReward})


if __name__ == '__main__':
    independentVariables = dict()
    independentVariables['num_predators'] = [3, 4, 5, 6]
    independentVariables['speed'] = [1]
    independentVariables['cost'] = [0, 0.01, 0.02, 0.03]
    independentVariables['selfish'] = [0, 1, 10000]

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    eval = EvaluateDf(evaluate)
    resultDF = toSplitFrame.groupby(levelNames).apply(eval)
    print(resultDF)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    fileName = 'evalAll.pkl'
    resultLoc = os.path.join(resultPath, fileName)
    saveToPickle(resultDF, resultLoc)

    print('saved to ', fileName)
    print(resultDF)
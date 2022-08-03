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
from visualize.drawDemo import *


def run(config):
    numPredators = config.num_predators
    preySpeedMultiplier = float(config.speed)
    costActionRatio = float(config.cost)
    selfishIndex = float(config.selfish)

    model_name = "model{}predators{}cost{}speed{}selfish".format(numPredators, costActionRatio, preySpeedMultiplier, selfishIndex) + 'run%i' % config.run_num
    model_path = Path('../models') / config.model_name / model_name
    model_path = model_path / 'incremental' / ('model_ep%i.pt' % config.incremental) if config.incremental is not None else model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

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
    observeOneAgent = lambda agentID: Observe(agentID, predatorsID, preyGroupID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(predatorsID, preyGroupID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce,
                                          getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: terminalCheck.terminal
    maddpg.prep_rollouts(device='cpu')

    epsRewardTot = []
    trajList = []
    for epsID in range(config.numTrajToSample):
        state = reset()
        trajectory = []
        epsReward = np.zeros(numAgents)
        
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
            rewards = rewardFunc(state, actions, nextState)
            trajectory.append((state, actions, rewards, nextState))

            epsReward += rewards
            
            state = nextState
            if isTerminal(state):
                state = reset()

        epsRewardTot.append(epsReward)
        trajList.append(trajectory)

    meanTrajReward = np.mean(epsRewardTot, axis=0)
    seTrajReward = np.std(epsRewardTot, axis=0) / np.sqrt(len(epsRewardTot) - 1)

    print("Mean Eps Reward: ", meanTrajReward)
    print("SE Eps Reward: ", seTrajReward)


    # visualize ------------

    if config.visualize:
        BLACK = (0, 0, 0)
        GRAY = (127, 127, 127)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)

        screenWidth = 700
        screenHeight = 700
        screen = pg.display.set_mode((screenWidth, screenHeight))
        screenColor = BLACK
        xBoundary = [0, 700]
        yBoundary = [0, 700]
        lineColor = WHITE
        lineWidth = 4
        drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)

        FPS = 10
        numBlocks = 2
        predatorColor = WHITE
        preyColor = GREEN
        blockColor = GRAY
        circleColorSpace = [predatorColor] * numPredators + [preyColor] * numPrey + [blockColor] * numBlocks
        viewRatio = 1.5
        preySize = int(0.05 * screenWidth / (2 * viewRatio))
        predatorSize = int(0.075 * screenWidth / (3 * viewRatio)) # without boarder
        blockSize = int(0.2 * screenWidth / (2 * viewRatio))
        circleSizeSpace = [predatorSize] * numPredators + [preySize] * numPrey + [blockSize] * numBlocks
        positionIndex = [0, 1]
        agentIdsToDraw = list(range(numPredators + numPrey + numBlocks))

        imageSavePath = os.path.join(dirName, '..', 'trajectories', model_name)
        if not os.path.exists(imageSavePath):
            os.makedirs(imageSavePath)
        imageFolderName = str('forDemo')
        saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)

        outsideCircleColor = [RED] * numPredators
        outsideCircleSize = int(predatorSize * 1.5)
        drawCircleOutside = DrawCircleOutside(screen, predatorsID, positionIndex,
                                              outsideCircleColor, outsideCircleSize, viewRatio= viewRatio)

        drawState = DrawState(FPS, screen, circleColorSpace, circleSizeSpace, agentIdsToDraw,
                              positionIndex, config.saveImage, saveImageDir, preyGroupID, predatorsID,
                              drawBackground, drawCircleOutside=drawCircleOutside, viewRatio= viewRatio)

        # MDP Env
        stateID = 0
        nextStateID = 3
        predatorSizeForCheck = 0.075
        preySizeForCheck = 0.05
        checkStatus = CheckStatus(predatorsID, preyGroupID, isCollision, predatorSizeForCheck, preySizeForCheck, stateID, nextStateID)
        chaseTrial = ChaseTrialWithKillNotation(stateID, drawState, checkStatus)
        [chaseTrial(trajectory) for trajectory in np.array(trajList[:20])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_predators", default=6, type=int, help="num_predators")
    parser.add_argument("--speed", default=1, type=float, help="speed")
    parser.add_argument("--cost", default=0, type=float, help="cost")
    parser.add_argument("--selfish", default=10000, type=float, help="selfish")

    parser.add_argument("--visualize", default=1, type=int)
    parser.add_argument("--saveImage", default=0, type=int)
    parser.add_argument("--run_num", default=1, type=int)
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " + "rather than final policy")
    parser.add_argument("--maxTimeStep", default=75, type=int)
    parser.add_argument("--numTrajToSample", default=100, type=int)

    parser.add_argument("--model_name", default= "CollectiveHunting", type = str, help="Name of directory to store " + "model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--save_gifs", action="store_true", help="Saves gif of each episode into model directory")
    parser.add_argument("--fps", default=20, type=int)

    config = parser.parse_args()

    run(config)
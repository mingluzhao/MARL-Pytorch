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
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]


    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    epsRewardTot = []
    for epsID in range(config.maxEpisodeToSample):
        print("Episode %i of %i" % (epsID + 1, config.maxEpisodeToSample))
        state = reset()
        # if config.save_gifs:
        #     frames = []
        #     frames.append(env.render('rgb_array')[0])
        # env.render('human')

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
            nextObs = observe(nextState)
            rewards = rewardFunc(state, actions, nextState)
            dones = isTerminal(state)
            state = nextState

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

    meanTrajReward = np.mean(epsRewardTot, axis=0)
    seTrajReward = np.std(epsRewardTot, axis=0) / np.sqrt(len(epsRewardTot) - 1)

    print("Mean Eps Reward: ", meanTrajReward)
    print("SE Eps Reward: ", seTrajReward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("num_predators", default=6, type=int, help="num_predators")
    parser.add_argument("speed", default=1, type=float, help="speed")
    parser.add_argument("cost", default=0, type=float, help="cost")
    parser.add_argument("selfish", default=1, type=float, help="selfish")

    parser.add_argument("--run_num", default=1, type=int)
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " + "rather than final policy")
    parser.add_argument("--maxTimeStep", default=75, type=int)
    parser.add_argument("--maxEpisodeToSample", default=100, type=int)

    parser.add_argument("--model_name", default= "CollectiveHunting", type = str, help="Name of directory to store " + "model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--save_gifs", action="store_true", help="Saves gif of each episode into model directory")
    parser.add_argument("--fps", default=20, type=int)

    config = parser.parse_args()

    run(config)
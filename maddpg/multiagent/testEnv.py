import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

import unittest
from ddt import ddt, data, unpack
from environment.multiAgentEnv import *
from environment.reward import *
import random
from functionTools.loadSaveModel import loadFromPickle

@ddt
class TestMultiAgentEnv(unittest.TestCase):
    def setUp(self):
        # set up original env

        numPredators = 3
        numPrey = 1
        numBlocks = 2

        numAgents = numPredators + numPrey
        numEntities = numAgents + numBlocks
        self.predatorID = list(range(numPredators))
        self.preyGroupID = list(range(numPredators, numAgents))
        self.blocksID = list(range(numAgents, numEntities))
        self.isCollision = IsCollision(getPosFromAgentState)

        predatorSize = 0.075
        preySize = 0.05
        blockSize = 0.2
        self.entitiesSizeList = [predatorSize] * numPredators + [preySize] * numPrey + [blockSize] * numBlocks

        self.punishForOutOfBound = PunishForOutOfBound()
        self.collisionDist = predatorSize + preySize

        preySpeedMultiplier = 1
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
        self.rewardPrey  = RewardPrey(self.predatorsID, self.preyGroupID, self.entitiesSizeList, getPosFromAgentState, isCollision,
                                punishForOutOfBound, collisionPunishment=collisionReward)
        
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
        
    
    # Sanity check for "larger distance -> smaller reward (percent)"
    def testPercetageSanityCheck(self):
        sensitivity = 5
        getPercent = lambda dist: (dist + 1 - self.collisionDist) ** (-sensitivity)
        self.assertTrue(getPercent(10) < getPercent(5))
        self.assertTrue(getPercent(.2) < getPercent(.1))
    #
    # @data((5),(0), (1000))
    # @unpack
    # def testPercentageNorm(self, sensitivity):
    #     getPercentageRewards = GetAgentsPercentageOfRewards(sensitivity, self.collisionDist)
    #     distanceList = [0.01, 0.2, 0.5]
    #     percentageList = getPercentageRewards(distanceList, 0)
    #     self.assertTrue(np.sum(percentageList), 1)

    def testPercentageIndivid(self):
        sensitivity = 1e4
        getPercentageRewards = GetAgentsPercentageOfRewards(sensitivity, self.collisionDist)
        distanceList = [0.01, 0.2, 0.5]
        percentageList = getPercentageRewards(distanceList, 0)
        self.assertTrue(tuple(percentageList), (1, 0, 0))

    def testPercentageShare(self):
        sensitivity = 0
        getPercentageRewards = GetAgentsPercentageOfRewards(sensitivity, self.collisionDist)
        distanceList = [0.01, 0.2, 0.5]
        percentageList = getPercentageRewards(distanceList, 0)
        self.assertTrue(tuple(percentageList), (1/3, 1/3, 1/3))

    def testCollisionRewardWithKillOnly(self):
        biteReward = 0.1
        killReward = 10
        killProportion = 1
        terminalCheck = TerminalCheck()
        getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck)

        numPredators = 3
        killRewardPercent = [0.2, 0.5, 0.3]
        collisionID = 1
        reward = getCollisionWolfReward(numPredators, killRewardPercent, collisionID)
        trueReward = [2, 5, 3]
        self.assertEqual(tuple(reward), tuple(trueReward))


    @data(([0.2, 0.5, 0.3], 1, {(2, 5, 3): 0.2, (0, .1, 0): 0.8}),
          ([0.1, 0, 0.9], 2, {(1, 0, 9): 0.2, (0, 0, .1): 0.8})
          )
    @unpack
    def testCollisionRewardDist(self, killRewardPercent, collisionID, trueRewardDict):
        biteReward = 0.1
        killReward = 10
        killProportion = .2
        class TerminalCheckWithNoTerminal(object):
            def __init__(self):
                self.reset()

            def reset(self):
                self.terminal = False

            def isTerminal(self):
                self.terminal = False

        terminalCheck = TerminalCheckWithNoTerminal()
        getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck)

        numPredators = 3
        iterationTime = 100000
        trueDict = {rew: trueRewardDict[rew] * iterationTime for rew in trueRewardDict.keys()}
        rewardList = [tuple(getCollisionWolfReward(numPredators, killRewardPercent, collisionID)) for _ in range(iterationTime)]
        for reward in trueDict.keys():
            self.assertAlmostEqual(rewardList.count(tuple(reward)), trueDict[reward], delta=500)

    '''
    In the new reward formulation: 
        shared reward ~ sensitivity = 0 + bite reward = 10 
        individual reward ~ sensitivity = inf + bite reward = 10
        individual reward ~ sensitivity = inf + kill reward = 10 + no terminal
    '''
    def testRewardWithIndividComparedWithTrajWithKill(self):
        rewardSensitivityToDistance = 10000
        biteReward = -100
        killReward = 10
        killProportion = 1

        class TerminalCheckWithNoTerminal(object):
            def __init__(self):
                self.reset()

            def reset(self):
                self.terminal = False

            def isTerminal(self):
                self.terminal = False

        terminalCheckWithNoTerminal = TerminalCheckWithNoTerminal()
        getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheckWithNoTerminal)
        getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(rewardSensitivityToDistance, self.collisionDist)
        getWolfSheepDistance = GetWolfSheepDistance(computeVectorNorm, getPosFromAgentState)
        rewardWolf = RewardpredatorWithKillProb(self.predatorID, self.preyGroupID, self.entitiesSizeList, self.isCollision, terminalCheckWithNoTerminal,
                                              getWolfSheepDistance, getAgentsPercentageOfRewards, getCollisionWolfReward)

        trajPath = os.path.join(dirName, 'maddpg3predator1prey2blocks60000episodes75steppreypeed1.0WolfActCost0.0individ1.0_mixTraj')

        trajList = loadFromPickle(trajPath)
        for traj in trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueWolfReward = [np.array(trueReward)[predatorID] for predatorID in self.predatorID]
                agentsReward = rewardWolf(state, action, nextState)
                print(agentsReward, trueWolfReward)
                self.assertEqual(tuple(trueWolfReward), tuple(agentsReward))

    def testRewardWithIndividComparedWithTrajWithBite(self):
        rewardSensitivityToDistance = 10000
        biteReward = 10
        killReward = 0
        killProportion = 0

        class TerminalCheckWithNoTerminal(object):
            def __init__(self):
                self.reset()

            def reset(self):
                self.terminal = False

            def isTerminal(self):
                self.terminal = False

        terminalCheckWithNoTerminal = TerminalCheckWithNoTerminal()
        getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheckWithNoTerminal)
        getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(rewardSensitivityToDistance, self.collisionDist)
        getWolfSheepDistance = GetWolfSheepDistance(computeVectorNorm, getPosFromAgentState)
        rewardWolf = RewardpredatorWithKillProb(self.predatorID, self.preyGroupID, self.entitiesSizeList, self.isCollision, terminalCheckWithNoTerminal,
                                              getWolfSheepDistance, getAgentsPercentageOfRewards, getCollisionWolfReward)

        trajPath = os.path.join(dirName, 'maddpg3predator1prey2blocks60000episodes75steppreypeed1.0WolfActCost0.0individ1.0_mixTraj')

        trajList = loadFromPickle(trajPath)
        for traj in trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueWolfReward = [np.array(trueReward)[predatorID] for predatorID in self.predatorID]
                agentsReward = rewardWolf(state, action, nextState)
                self.assertEqual(tuple(trueWolfReward), tuple(agentsReward))

    def testRewardWithSharedComparedWithTraj(self):
        rewardSensitivityToDistance = 0
        biteReward = -100
        killReward = 10
        killProportion = 1

        class TerminalCheckWithNoTerminal(object):
            def __init__(self):
                self.reset()

            def reset(self):
                self.terminal = False

            def isTerminal(self):
                self.terminal = False

        terminalCheckWithNoTerminal = TerminalCheckWithNoTerminal()
        getCollisionWolfReward = GetCollisionWolfReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheckWithNoTerminal)
        getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(rewardSensitivityToDistance, self.collisionDist)
        getWolfSheepDistance = GetWolfSheepDistance(computeVectorNorm, getPosFromAgentState)
        rewardWolf = RewardpredatorWithKillProb(self.predatorID, self.preyGroupID, self.entitiesSizeList, self.isCollision, terminalCheckWithNoTerminal,
                                              getWolfSheepDistance, getAgentsPercentageOfRewards, getCollisionWolfReward)

        trajPath = os.path.join(dirName, 'maddpg3predator1prey2blocks60000episodes75steppreypeed1.0WolfActCost0.0individ0.0_mixTraj')

        trajList = loadFromPickle(trajPath)
        for traj in trajList:
            for timeStep in range(len(traj)):
                state = traj[timeStep][0]
                action = traj[timeStep][1]
                trueReward = traj[timeStep][2]
                nextState = traj[timeStep][3]

                trueWolfReward = [np.array(trueReward)[predatorID] for predatorID in self.predatorID]
                agentsReward = rewardWolf(state, action, nextState)

                self.assertEqual(tuple(np.round(trueWolfReward, 7)), tuple(np.round(agentsReward, 7)))

if __name__ == '__main__':
    unittest.main()

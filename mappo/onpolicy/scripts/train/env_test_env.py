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


def main():
    numWolves = 3
    numSheeps = 1
    numBlocks = 1
    maxTimeStep = 75
    sheepSpeedMultiplier = 1
    individualRewardWolf = int(True)

    debug = True # False

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

    from onpolicy.envs.hunting_environment.chasingEnv.parallel_env import ParallelEnv
    num_envs=4
    parallel_env = ParallelEnv(num_envs, reset, observe, transit, rewardFunc, isTerminal)


    for eps in range(2):
        states = parallel_env.reset()
        for timestep in range(2):
            print(f"------------------------ time {timestep} -----------------------")
            actions = np.array([[[0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0.],
            [1., 0., 0., 0., 0.]],

           [[1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 1., 0.]],

           [[0., 0., 0., 1., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.]],

           [[0., 0., 0., 1., 0.],
            [0., 0., 0., 1., 0.],
            [0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0.]]])

            nextState, nextObs, rewards, dones, infos = parallel_env.step(states, actions)
            print(parallel_env.observe(states))
            states = nextState
            # Process observations, rewards, etc.
    parallel_env.close()

if __name__ == "__main__":
    main()
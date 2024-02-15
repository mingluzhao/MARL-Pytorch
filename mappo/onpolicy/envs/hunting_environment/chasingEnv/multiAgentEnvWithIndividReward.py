
class RewardWolfIndividual:
    def __init__(self, wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward=10):
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.entitiesSizeList = entitiesSizeList
        self.isCollision = isCollision
        self.collisionReward = collisionReward

    def __call__(self, state, action, nextState):
        reward = []

        for wolfID in self.wolvesID:
            currentWolfReward = 0
            wolfSize = self.entitiesSizeList[wolfID]
            wolfNextState = nextState[wolfID]
            for sheepID in self.sheepsID:
                sheepSize = self.entitiesSizeList[sheepID]
                sheepNextState = nextState[sheepID]

                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    currentWolfReward += self.collisionReward

            reward.append(currentWolfReward)
        return reward


class RewardWolfIndividualWithBiteAndKill:
    def __init__(self, wolvesID, sheepsID, entitiesSizeList, isCollision, getCaughtHistoryFromAgentState, sheepLife=3,
                 biteReward=1, killReward=10):
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.entitiesSizeList = entitiesSizeList
        self.isCollision = isCollision
        self.getEntityCaughtHistory = lambda state, entityID: getCaughtHistoryFromAgentState(state[entityID])
        self.sheepLife = sheepLife
        self.biteReward = biteReward
        self.killReward = killReward

    def __call__(self, state, action, nextState):
        reward = []
        for wolfID in self.wolvesID:
            currentWolfReward = 0
            wolfSize = self.entitiesSizeList[wolfID]
            wolfNextState = nextState[wolfID]
            for sheepID in self.sheepsID:
                sheepSize = self.entitiesSizeList[sheepID]
                sheepNextState = nextState[sheepID]
                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    currentWolfReward += self.biteReward
                    sheepCaughtHistory = self.getEntityCaughtHistory(state, sheepID)
                    if sheepCaughtHistory == self.sheepLife:
                        currentWolfReward += self.killReward
            reward.append(currentWolfReward)
        return reward


class RewardWolfIndividualWithBiteKillAndApples:
    def __init__(self, wolvesID, sheepsID, applesID, entitiesSizeList, isCollision, getCaughtHistoryFromAgentState, sheepLife=3,
                 biteReward=1, killReward=10, appleReward=5):
        self.wolvesID = wolvesID
        self.sheepsID = sheepsID
        self.applesID = applesID  # Added applesID parameter
        self.entitiesSizeList = entitiesSizeList
        self.isCollision = isCollision
        self.getEntityCaughtHistory = lambda state, entityID: getCaughtHistoryFromAgentState(state[entityID])
        self.sheepLife = sheepLife
        self.biteReward = biteReward
        self.killReward = killReward
        self.appleReward = appleReward  # Added appleReward parameter

    def __call__(self, state, action, nextState):
        reward = []
        for wolfID in self.wolvesID:
            currentWolfReward = 0
            wolfSize = self.entitiesSizeList[wolfID]
            wolfNextState = nextState[wolfID]
            # Check collision with sheep and apply bite/kill rewards
            for sheepID in self.sheepsID:
                sheepSize = self.entitiesSizeList[sheepID]
                sheepNextState = nextState[sheepID]
                if self.isCollision(wolfNextState, sheepNextState, wolfSize, sheepSize):
                    currentWolfReward += self.biteReward
                    sheepCaughtHistory = self.getEntityCaughtHistory(state, sheepID)
                    if sheepCaughtHistory == self.sheepLife:
                        currentWolfReward += self.killReward
            # Check collision with apples and apply apple rewards
            for appleID in self.applesID:
                appleSize = self.entitiesSizeList[appleID]
                appleNextState = nextState[appleID]
                if self.isCollision(wolfNextState, appleNextState, wolfSize, appleSize):
                    currentWolfReward += self.appleReward
            reward.append(currentWolfReward)
        return reward

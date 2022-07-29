import numpy as np
from maddpg.multiagent.core import World, Agent, Landmark
from maddpg.multiagent.scenario import BaseScenario
import random

# predator-prey environment

'''
make_world function: allowed to adjust
- num_good_agents (prey)
- num_adversaries (predator)
- num_landmarks
- max_speed_good
- max_speed_adv
- cost_action_ratio
- selfish_ind

'''
class Scenario(BaseScenario):
    def make_world(self, arglist):
        world = World()
        world.dim_c = 2

        # add agents
        num_agents = arglist.num_adversaries + arglist.num_good_agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < arglist.num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = arglist.max_speed_adv if agent.adversary else arglist.max_speed_good
            agent.selfish = arglist.selfish_ind
            agent.cost_action_ratio = arglist.cost_action_ratio

        # add landmarks
        world.landmarks = [Landmark() for i in range(arglist.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions

        world.kill_prop = 0.2
        self.reset_world(world)

        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.alive = True

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    # def reward(self, agent, world):
    #     main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
    #     return main_reward

    def reward(self, world):
        adv_reward = self.all_adversary_reward(world)
        agents_reward = [self.agent_reward(agent, world) for agent in world.agents]
        return list(adv_reward) + list(agents_reward)

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def all_adversary_reward(self, world):
        from maddpg.multiagent.utils import sampleFromDistribution

        # Adversaries are rewarded for collisions with agents
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        killReward = 10
        biteReward = 0
        selfish_level = adversaries[0].selfish
        collisionMinDist = agents[0].size + adversaries[0].size
        getPercentReward = lambda dist: (dist + 1 - collisionMinDist) ** (-selfish_level)

        all_adversary_reward = np.zeros(len(adversaries))
        for prey in agents:
            # randomly order predators so that when more than one predator catches the prey, random one samples first
            predatorsID = np.arange(len(adversaries))
            random.shuffle(predatorsID)

            for predatorID in predatorsID:
                predator = adversaries[predatorID]

                if self.is_collision(predator, prey) and prey.alive: # if already killed, add zero for all agents
                    # sample to see whether killed
                    killed = sampleFromDistribution({1: world.kill_prop, 0: 1-world.kill_prop})
                    if killed:
                        if selfish_level > 100:
                            all_adversary_reward[predatorID] = all_adversary_reward[predatorID] + killReward
                        else:
                            allPredatorDist = [np.sqrt(np.sum(np.square(pred.state.p_pos - prey.state.p_pos))) for
                                               pred in adversaries]
                            percentageRaw = [getPercentReward(dist) for dist in allPredatorDist]
                            reward = killReward * np.array(percentageRaw) / np.sum(percentageRaw)
                            all_adversary_reward = np.array(all_adversary_reward) + np.array(reward)
                    else:
                        all_adversary_reward[predatorID] = all_adversary_reward[predatorID] + biteReward
        all_adversary_reward_with_cost = [reward - self.get_predator_action_cost(agent) for reward, agent in
                                          zip(all_adversary_reward, adversaries)]

        return all_adversary_reward_with_cost

    def get_predator_action_cost(self, agent):
        actionMagnitude = np.linalg.norm(np.array(agent.action.u), ord=2)
        cost = agent.cost_action_ratio * actionMagnitude

        return cost

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue # skip itself
            comm.append(other.state.c) # append other people's comm chanel : not appended
            other_pos.append(other.state.p_pos - agent.state.p_pos)# append other people's physical distance from self : 3* 2
            if not other.adversary: # also observe the velocity of prey: 2
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

import numpy as np
from torch import Tensor
from torch.autograd import Variable

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, bufferSize, numAgents, obs_dims, ac_dims):
        """
        Inputs:
            bufferSize (int): Maximum number of timepoints to store in buffer
            numAgents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.bufferSize = bufferSize
        self.numAgents = numAgents
        self.obsBuffer = []
        self.actionsBuffer = []
        self.rewardsBuffer = []
        self.nexObsBuffer = []
        self.terminalBuffer = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obsBuffer.append(np.zeros((bufferSize, odim)))
            self.actionsBuffer.append(np.zeros((bufferSize, adim)))
            self.rewardsBuffer.append(np.zeros(bufferSize))
            self.nexObsBuffer.append(np.zeros((bufferSize, odim)))
            self.terminalBuffer.append(np.zeros(bufferSize))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        nentries = 1#observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.bufferSize:
            rollover = self.bufferSize - self.curr_i # num of indices to roll over
            for agent_i in range(self.numAgents):
                self.obsBuffer[agent_i] = np.roll(self.obsBuffer[agent_i], rollover, axis=0)
                self.actionsBuffer[agent_i] = np.roll(self.actionsBuffer[agent_i], rollover, axis=0)
                self.rewardsBuffer[agent_i] = np.roll(self.rewardsBuffer[agent_i], rollover)
                self.nexObsBuffer[agent_i] = np.roll( self.nexObsBuffer[agent_i], rollover, axis=0)
                self.terminalBuffer[agent_i] = np.roll(self.terminalBuffer[agent_i], rollover)
            self.curr_i = 0
            self.filled_i = self.bufferSize

        for agent_i in range(self.numAgents):
            # actions are already batched by agent, so they are indexed differently
            # self.obsBuffer[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack([observations[agent_i]]) #TODO for handcrafted env
            # self.actionsBuffer[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            # self.rewardsBuffer[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]
            # self.nexObsBuffer[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack([next_observations[agent_i]])
            # self.terminalBuffer[agent_i][self.curr_i:self.curr_i + nentries] = dones

            self.obsBuffer[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(observations[:, agent_i]) #TODO for gym env
            self.actionsBuffer[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rewardsBuffer[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.nexObsBuffer[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(next_observations[:, agent_i])
            self.terminalBuffer[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]

        self.curr_i += nentries
        if self.filled_i < self.bufferSize:
            self.filled_i += nentries
        if self.curr_i == self.bufferSize:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rewardsBuffer[i][inds] - self.rewardsBuffer[i][:self.filled_i].mean()) /
                             self.rewardsBuffer[i][:self.filled_i].std()) for i in range(self.numAgents)]
        else:
            ret_rews = [cast(self.rewardsBuffer[i][inds]) for i in range(self.numAgents)]
        return ([cast(self.obsBuffer[i][inds]) for i in range(self.numAgents)],
                [cast(self.actionsBuffer[i][inds]) for i in range(self.numAgents)], ret_rews,
                [cast(self.nexObsBuffer[i][inds]) for i in range(self.numAgents)],
                [cast(self.terminalBuffer[i][inds]) for i in range(self.numAgents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.bufferSize:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rewardsBuffer[i][inds].mean() for i in range(self.numAgents)]

    def get_tot_rewards(self, N):
        if self.filled_i == self.bufferSize:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rewardsBuffer[i][inds].sum() for i in range(self.numAgents)]

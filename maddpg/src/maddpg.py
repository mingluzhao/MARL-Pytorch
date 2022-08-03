import torch
# from gym.spaces import Box, Discrete
from maddpg.src.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from maddpg.src.utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()

                 
class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agentsInitParams, algorithmTypes,hiddenDimList, isDiscreteAction,
                 gamma=0.95, tau=0.01, lr=0.01 ):
        """
        Inputs:
            agentsInitParams (list of dict): List of dicts with parameters to initialize each agent
                dimPolicyInput (int): Input dimensions to policy
                dimPolicyOutput (int): Output dimensions to policy
                dimCriticInput (int): Input dimensions to critic
            algorithmTypes (list of str): Learning algorithm for each agent (DDPG or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hiddenDim (int): Number of hidden dimensions for networks
            isDiscreteAction (bool): Whether or not to use discrete action space
        """
        self.numAgents = len(algorithmTypes)
        self.algorithmTypes = algorithmTypes
        self.agents = [DDPGAgent(lr=lr, isDiscreteAction=isDiscreteAction, layerNum = len(hiddenDimList),
                                 hiddenDim=hiddenDimList[0], **params)
                       for params in agentsInitParams]
        self.agentsInitParams = agentsInitParams
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.isDiscreteAction = isDiscreteAction
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def trainPolicies(self):
        return [a.policyTrain for a in self.agents]

    @property
    def targetPolicies(self):
        return [a.policyTarget for a in self.agents]

    def scaleNoise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for agent in self.agents:
            agent.scaleNoise(scale)

    def resetNoise(self):
        for agent in self.agents:
            agent.resetNoise()

    def act(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [agent.act(obs, explore=explore) for agent, obs in zip(self.agents, observations)]

    def update(self, sample, agentID, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agentID (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        allObs, allActions, allRewards, allNextObs, allTerminal = sample
        
        currentAgent = self.agents[agentID]
        currentAgent.critic_optimizer.zero_grad()
        
        if self.algorithmTypes[agentID] == 'MADDPG':
            # if self.isDiscreteAction: # one-hot encode action
            #     allTargetNextActions = [onehot_from_logits(targetPolicy(nextObs)) for targetPolicy, nextObs in zip(self.targetPolicies, allNextObs)]
            # else:
            #     allTargetNextActions = [targetPolicy(nextObs) for targetPolicy, nextObs in zip(self.targetPolicies, allNextObs)]
            # TODO: use one hot?
            allTargetNextActions = [targetPolicy(nextObs) for targetPolicy, nextObs in zip(self.targetPolicies, allNextObs)]
            criticTargetInput = torch.cat((*allNextObs, *allTargetNextActions), dim=1)
        else:  # DDPG
            criticTargetInput = torch.cat((allNextObs[agentID], currentAgent.policyTarget(allNextObs[agentID])), dim=1)
            # if self.isDiscreteAction:
            #     criticTargetInput = torch.cat((allNextObs[agentID], onehot_from_logits(currentAgent.policyTarget(allNextObs[agentID]))), dim=1)
            # else:
            #     criticTargetInput = torch.cat((allNextObs[agentID], currentAgent.policyTarget(allNextObs[agentID])), dim=1)
        criticUpdateTarget = (allRewards[agentID].view(-1, 1) + self.gamma * currentAgent.criticTarget(criticTargetInput) * 
                              (1 - allTerminal[agentID].view(-1, 1)))

        if self.algorithmTypes[agentID] == 'MADDPG':
            criticTrainInput = torch.cat((*allObs, *allActions), dim=1)
        else:  # DDPG
            criticTrainInput = torch.cat((allObs[agentID], allActions[agentID]), dim=1)
        criticTrainOutput = currentAgent.criticTrain(criticTrainInput)
        criticLoss = MSELoss(criticTrainOutput, criticUpdateTarget.detach())
        criticLoss.backward()
        if parallel:
            average_gradients(currentAgent.criticTrain)
        torch.nn.utils.clip_grad_norm_(currentAgent.criticTrain.parameters(), 0.5)
        currentAgent.critic_optimizer.step()

        # train Actor
        currentAgent.policy_optimizer.zero_grad()
        if self.isDiscreteAction:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            actorTrainOutput = currentAgent.policyTrain(allObs[agentID])
            noisyTrainAction = gumbel_softmax(actorTrainOutput, hard=False)
            # noisyTrainAction = gumbel_softmax(actorTrainOutput, hard=True) # TODO original code = hard
        else:
            actorTrainOutput = currentAgent.policyTrain(allObs[agentID])
            noisyTrainAction = actorTrainOutput

        if self.algorithmTypes[agentID] == 'MADDPG':
            # following MADDPG paper: only the current agent uses train policy, others use buffer actions
            all_pol_acs = allActions.copy()
            all_pol_acs[agentID] = noisyTrainAction

            # TODO: specific for this implementation: sampling directly from current policy
            # all_pol_acs = []
            # for currentID, agentPolicyTrain, ob in zip(range(self.numAgents), self.trainPolicies, allObs):
            #     if currentID == agentID:
            #         all_pol_acs.append(noisyTrainAction)
            #     elif self.isDiscreteAction:
            #         all_pol_acs.append(onehot_from_logits(agentPolicyTrain(ob)))
            #     else:
            #         all_pol_acs.append(agentPolicyTrain(ob))
            criticTrainInput = torch.cat((*allObs, *all_pol_acs), dim=1)
        else:  # DDPG
            criticTrainInput = torch.cat((allObs[agentID], noisyTrainAction), dim=1)
        actorLoss = -currentAgent.criticTrain(criticTrainInput).mean()
        actorLoss += (actorTrainOutput**2).mean() * 1e-3
        actorLoss.backward()
        if parallel:
            average_gradients(currentAgent.policyTrain)
        torch.nn.utils.clip_grad_norm_(currentAgent.policyTrain.parameters(), 0.5)
        currentAgent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agentID, {'criticLoss': criticLoss, 'actorLoss': actorLoss}, self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.criticTarget, a.criticTrain, self.tau)
            soft_update(a.policyTarget, a.policyTrain, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policyTrain.train()
            a.criticTrain.train()
            a.policyTarget.train()
            a.criticTarget.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policyTrain = fn(a.policyTrain)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.criticTrain = fn(a.criticTrain)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.policyTarget = fn(a.policyTarget)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policyTrain.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policyTrain for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policyTrain = fn(a.policyTrain)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, algorithmTypes, isDiscreteAction, policyInputDimList, policyOutputDimList, criticInputDimList,
                      hiddenDimList, gamma=0.95, tau=0.01, lr=0.01):
        """
        Instantiate instance of this class from multi-agent environment
        """
        # policyInputDimList
        # policyOutputDimList 
        # criticInputDimList # all agents actions and observations for MADDPG, agentObservation and Action size for DDPG
        
        agentsInitParams = [{'dimPolicyInput': dimPolicyInput, 'dimPolicyOutput': dimPolicyOutput, 'dimCriticInput': dimCriticInput}
                            for dimPolicyInput, dimPolicyOutput, dimCriticInput 
                            in zip(policyInputDimList, policyOutputDimList, criticInputDimList)]

        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hiddenDimList': hiddenDimList, 'algorithmTypes': algorithmTypes,
                     'agentsInitParams': agentsInitParams, 'isDiscreteAction': isDiscreteAction}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance


    # @classmethod
    # def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
    #                   gamma=0.95, tau=0.01, lr=0.01, hiddenDim=64):
    #     """
    #     Instantiate instance of this class from multi-agent environment
    #     """
    #     agentsInitParams = []
    #     algorithmTypes = [adversary_alg if atype == 'adversary' else agent_alg for atype in env.agent_types]
    #     for acsp, obsp, algtype in zip(env.action_space, env.observation_space, algorithmTypes):
    #         dimPolicyInput = obsp.shape[0]
    #         if isinstance(acsp, Box):
    #             isDiscreteAction = False
    #             get_shape = lambda x: x.shape[0]
    #         else:  # Discrete
    #             isDiscreteAction = True
    #             get_shape = lambda x: x.n
    #         dimPolicyOutput = get_shape(acsp)
    #         if algtype == "MADDPG":
    #             dimCriticInput = 0
    #             for oobsp in env.observation_space:
    #                 dimCriticInput += oobsp.shape[0]
    #             for oacsp in env.action_space:
    #                 dimCriticInput += get_shape(oacsp)
    #         else:
    #             dimCriticInput = obsp.shape[0] + get_shape(acsp)
    #         agentsInitParams.append({'dimPolicyInput': dimPolicyInput,
    #                                   'dimPolicyOutput': dimPolicyOutput,
    #                                   'dimCriticInput': dimCriticInput})
    #     init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
    #                  'hiddenDim': hiddenDim,
    #                  'algorithmTypes': algorithmTypes,
    #                  'agentsInitParams': agentsInitParams,
    #                  'isDiscreteAction': isDiscreteAction}
    #     instance = cls(**init_dict)
    #     instance.init_dict = init_dict
    #     return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance